#ifndef SOFTFP_FLOAT_H
#define SOFTFP_FLOAT_H

#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <limits>

#include "util/bit_op.h"
#include "util/int128.h"
#include "util/select_int.h"

namespace softfp {

template<int W, int T>
class Float;

namespace internal {

template<typename>
struct Is_float : std::false_type {};

template<int W, int T>
struct Is_float<Float<W, T>> : std::true_type {};

}

enum class Rounding_mode: uint8_t {
    ties_to_even = 0b000,
    toward_zero = 0b001,
    toward_negative = 0b010,
    toward_positive = 0b011,
    ties_to_away = 0b100
};

enum class Exception_flag: uint8_t {
    none = 0,
    inexact = 1,
    underflow = 2,
    overflow = 4,
    divide_by_zero = 8,
    invalid_operation = 16
};

static Exception_flag operator | (Exception_flag a, Exception_flag b) {
    return static_cast<Exception_flag>(static_cast<int>(a) | static_cast<int>(b));
}

static void operator |= (Exception_flag& a, Exception_flag b) {
    a = a | b;
}

enum class Class: uint8_t {
    negative_infinity = 0,
    negative_normal = 1,
    negative_subnormal = 2,
    negative_zero = 3,
    positive_zero = 4,
    positive_subnormal = 5,
    positive_normal = 6,
    positive_infinity = 7,
    signaling_nan = 8,
    quiet_nan = 9,
};

enum class Ordering: int8_t {
    less_than = -1,
    equal = 0,
    greater_than = 1,
};

extern Rounding_mode rounding_mode;
extern Exception_flag exception_flags;

template<int W, int T>
class Float {
public:
    static constexpr int exponent_width = W;
    static constexpr int significand_width = T;

    // Need to be able to store exponent of a normalized subnormal number or biased exponent
    // normalized subnormal number has range [minimum_exponent - significand_width, maximum_exponent]
    // biased exponent has range [0, infinity_biased_exponent]
    // normalized subnormal number calculation requires double the precision
    using Exponent = typename util::Select_int<exponent_width + 2>::fast;

    // Following the definition of Exponent, we need to make sure all possible values fit
    // minimum_bias - significand_width >= Exponent.min implies
    // - (2 ** (exponent_width - 1) - 2) - significand_width >= -(2 ** exponent_width) implies
    // significand_width >= 2 + 2 ** (exponent_width - 1)
    static_assert(significand_width <= 2 + (Exponent(1) << (exponent_width - 1)), "Exponent is not large enough");

    // This need to be able intermediatary calculation results.
    // for addition: significand + hidden bit + carry bit + two more bits precision for rounding.
    // for subtract: significand + hidden bit + borrow bit + two more bits precision for rounding.
    //
    // Significand is majorly used to represent fixed point numbers.
    // The decimal point will be placed between (significand_width + 1)th and (significand_width + 2)th bit,
    // so "normalized" numbers will have 1 in its (significand_width + 2)th bit.
    using Significand = typename util::Select_uint<significand_width + 4>::fast;

    // Need to contain the product of significand with hidden bits, plus two rounding bits.
    using Double_significand = typename util::Select_uint<2 * significand_width + 4>::fast;

    // The type to hold the entire value. We use a number instead of multiple fields to be able to be
    // bitwise compatible with IEEE-defined binary exchange formats.
    using Holder = typename util::Select_uint<significand_width + exponent_width + 1>::least;

    // Special exponent values
    static constexpr Exponent infinity_biased_exponent = (1 << exponent_width) - 1;
    static constexpr Exponent maximum_biased_exponent = infinity_biased_exponent - 1;

    static constexpr Exponent exponent_bias = (1 << (exponent_width - 1)) - 1;
    static constexpr Exponent minimum_exponent = 1 - exponent_bias;
    static constexpr Exponent maximum_exponent = maximum_biased_exponent - exponent_bias;

private:
    Holder value_;

public:
    /* Constant special values */
    static constexpr Float quiet_nan() noexcept {
        Float value {};
        value.sign(false);
        value.biased_exponent(infinity_biased_exponent);
        value.trailing_significand(1ULL << (significand_width - 1));
        return value;
    }

    static constexpr Float infinity(bool sign = false) noexcept {
        Float value {};
        value.sign(sign);
        value.biased_exponent(infinity_biased_exponent);
        return value;
    }

    static constexpr Float zero(bool sign = false) noexcept {
        Float value {};
        value.sign(sign);
        return value;
    }

private:

    // Shift while preserving rounding property.
    // The last bit is the OR of all bits shifted away.
    // For details check the comment of round()
    template<typename IntType>
    static IntType right_shift(IntType significand, Exponent shift_amount) noexcept {
        if (shift_amount >= static_cast<Exponent>(sizeof(IntType) * 8)) {
            return significand != 0;
        } else {
            return (significand >> shift_amount) |
                ((significand & ((IntType(1) << shift_amount) - 1)) != 0);
        }
    }

    // Normalized the significand while preserving rounding property.
    template<typename IntType>
    static std::tuple<Exponent, Significand> normalize(Exponent exponent, IntType significand) noexcept {
        int width = util::log2_floor(significand) - 2;
        if (width < significand_width) {
            int width_diff = significand_width - width;
            return std::make_tuple(exponent - width_diff, static_cast<Significand>(significand << width_diff));
        } else if (width > significand_width) {
            int width_diff = width - significand_width;
            return std::make_tuple(exponent + width_diff, static_cast<Significand>(right_shift(significand, width_diff)));
        }
        return std::make_tuple(exponent, significand);
    }

    // Round the significand based on current rounding mode and last two bits.
    static std::tuple<bool, Significand> round_significand(bool sign, Significand significand) {
        bool inexact = false;

        if ((significand & 3) != 0) {
            inexact = true;

            switch (rounding_mode) {
                case Rounding_mode::ties_to_even:
                    significand += ((significand >> 2) & 1) + 1;
                case Rounding_mode::toward_zero:
                    break;
                case Rounding_mode::toward_negative:
                    if (sign) significand += 3;
                    break;
                case Rounding_mode::toward_positive:
                    if (!sign) significand += 3;
                    break;
                case Rounding_mode::ties_to_away:
                    // If last two bits are 10 or 11, then round up.
                    significand += 2;
                    break;
            }
        }

        return std::make_tuple(inexact, significand >> 2);
    }

    // Get the finite number overflowing result in current rounding mode.
    static Float round_overflow(bool sign) {
        exception_flags |= Exception_flag::overflow | Exception_flag::inexact;

        // When we are rounding away from the Infinity, we set the result
        // to be the largest finite number.
        Float value = infinity(sign);

        if ((sign && rounding_mode == Rounding_mode::toward_positive) ||
            (!sign && rounding_mode == Rounding_mode::toward_negative) ||
            rounding_mode == Rounding_mode::toward_zero) {

            // Decrement by one will shift value from infinity to max finite number
            value.value_--;
        }

        return value;
    }

    // Principle about rounding: to round correctly, we need two piece of information:
    // a) the first bit beyond target precision
    // b) whether we discard any bits after that bit.
    // If a=0, b=0, then the remainder is 0.
    // If a=0, b=1, then the remainder is in range (0, 0.5)
    // If a=1, b=0, then the remainder is 0.5.
    // If a=1, b=1, then the remainder is in range (0.5, 1).
    // Therefore we require signicand to contain two more bits beyond precision.
    // Input must be normal.
    static Float round(bool sign, Exponent exponent, Significand significand) {
        if (exponent > maximum_exponent) return round_overflow(sign);

        Float value {};
        value.sign(sign);

        // To yield correct result, we need to first subnormalize the number before rounding.
        Significand rounded = significand;
        if (exponent < minimum_exponent) {
            rounded = right_shift(rounded, minimum_exponent - exponent);
        }

        bool inexact;
        std::tie(inexact, rounded) = round_significand(sign, rounded);

        if (inexact) {
            exception_flags |= Exception_flag::inexact;

            // When the significand is all 1 and rounding causes it to round up.
            // Since when this happens, resulting significand should be all zero.
            // In this case casting significand to "Significand" will yield correct
            // result, but we need to increment exponent.
            if (rounded == (static_cast<Significand>(1) << (significand_width + 1))) {
                exponent++;
                rounded >>= 1;
            }
        }

        // Underflow or subnormal
        if (exponent < minimum_exponent) {

            // The border between subnormal and normal.
            if (rounded == (static_cast<Significand>(1) << significand_width)) {

                // In this special case, we need to deal with underflow flag very carefully.
                // IEEE specifies that the underflow flag should only be set if rounded result
                // in *unbounded* exponent will yield to an overflow.
                if (std::get<1>(round_significand(sign, significand)) !=
                    (static_cast<Significand>(1) << (significand_width + 1))) {

                    exception_flags |= Exception_flag::underflow;
                }

                value.biased_exponent(1);
                value.trailing_significand(0);
                return value;
            }

            if (inexact) {
                exception_flags |= Exception_flag::underflow;
            }

            value.biased_exponent(0);
            value.trailing_significand(rounded);
            return value;
        }

        if (exponent > maximum_exponent) return round_overflow(sign);

        value.biased_exponent(exponent + exponent_bias);
        value.trailing_significand(rounded);
        return value;
    }

    template<typename IntType>
    static Float normalize_and_round(bool sign, Exponent exponent, IntType significand) {
        Significand final_significand;
        std::tie(exponent, final_significand) = normalize(exponent, significand);
        return round(sign, exponent, final_significand);
    }

public:
    constexpr Float() noexcept: value_{0} {}

private:
    /* Component accessors */
    constexpr Exponent biased_exponent() const noexcept {
        constexpr auto mask = (static_cast<Exponent>(1) << exponent_width) - 1;
        return (value_ >> significand_width) & mask;
    }

    // Set the biased exponent of the floating pointer number.
    // Only up to exponent_width bits are respected and all other bits are ignored.
    constexpr void biased_exponent(Exponent value) noexcept {
        constexpr auto mask = ((static_cast<Holder>(1) << exponent_width) - 1) << significand_width;
        value_ = (value_ &~ mask) | ((static_cast<Holder>(value) << significand_width) & mask);
    }

    constexpr Significand trailing_significand() const noexcept {
        constexpr auto mask = (static_cast<Significand>(1) << significand_width) - 1;
        return static_cast<Significand>(value_ & mask);
    }

    // Set the trailing significand of the floating pointer number.
    // Only up to significand_width bits are respected and all other bits are ignored.
    constexpr void trailing_significand(Significand value) noexcept {
        constexpr auto mask = (static_cast<Holder>(1) << significand_width) - 1;
        value_ = (value_ &~ mask) | (value & mask);
    }

    std::tuple<Exponent, Significand> get_normalized_significand() const noexcept {
        Exponent biased_exponent = this->biased_exponent();
        Significand trailing_significand = this->trailing_significand();

        // We couldn't handle this
        if (biased_exponent == infinity_biased_exponent ||
            (biased_exponent == 0 && trailing_significand == 0))
            std::terminate();

        if (biased_exponent == 0) {
            int width_diff = significand_width - util::log2_floor(trailing_significand);
            return std::make_tuple(minimum_exponent - width_diff, trailing_significand << (width_diff + 2));
        }

        Exponent exponent = biased_exponent - exponent_bias;
        Significand significand = trailing_significand | (static_cast<Significand>(1) << significand_width);
        return std::make_tuple(exponent, significand << 2);
    }

    std::tuple<Exponent, Significand> get_significand() const noexcept {
        Exponent biased_exponent = this->biased_exponent();
        Significand trailing_significand = this->trailing_significand();

        // We couldn't handle this
        if (biased_exponent == infinity_biased_exponent)
            std::terminate();

        if (biased_exponent == 0) {

            // Perform lvalue-rvalue conversion to get rid of ODR-use of minimum_exponent.
            return std::make_tuple(Exponent(minimum_exponent), trailing_significand << 2);
        }

        Exponent exponent = biased_exponent - exponent_bias;
        Significand significand = trailing_significand | (static_cast<Significand>(1) << significand_width);
        return std::make_tuple(exponent, significand << 2);
    }

    constexpr bool sign() const noexcept {
        return (value_ >> (exponent_width + significand_width)) != 0;
    }

    constexpr void sign(bool value) noexcept {
        constexpr auto mask = static_cast<Holder>(1) << (exponent_width + significand_width);
        value_ = (value_ &~ mask) | (static_cast<Holder>(value) << (exponent_width + significand_width));
    }

    static Float propagate_nan(Float a, Float b) noexcept {
        if (a.is_signaling() || b.is_signaling()) {
            exception_flags |= Exception_flag::invalid_operation;
        }

        return quiet_nan();
    }

    static Float cancellation_zero() noexcept {
        return zero(rounding_mode == Rounding_mode::toward_negative);
    }

public:
    /* IEEE 754-2008 5.3.1 Homogeneous general-computational operations > General operations */
    static std::tuple<Float, Float> min_max(Float a, Float b) noexcept {
        if (a.is_nan() || b.is_nan()) {
            if (a.is_signaling() || b.is_signaling()) {
                exception_flags |= Exception_flag::invalid_operation;
                return {quiet_nan(), quiet_nan()};
            }

            if (a.is_nan()) {
                if (b.is_nan()) {
                    return {quiet_nan(), quiet_nan()};
                }
                return {b, b};
            }

            return {a, a};
        }

        if (total_order(a, b) == Ordering::less_than) {
            return {a, b};
        } else {
            return {b, a};
        }
    }

    static Float min(Float a, Float b) noexcept {
        return std::get<0>(min_max(a, b));
    }

    static Float max(Float a, Float b) noexcept {
        return std::get<1>(min_max(a, b));
    }

private:
    /* IEEE 754-2008 5.4.1 formatOf general-computational operations > Arithmetic operations */
    // Magnitude add. a and b must have the same sign and not NaN.
    // a must have greater magnitude.
    static Float add_magnitude(Float a, Float b) noexcept {

        // Handling for Infinity
        if (a.is_infinite()) return a;

        // If both are subnormal, then neither signifcand we retrieved below will be normal.
        // So we handle them specially here.
        if (a.biased_exponent() == 0) {
            Significand significand_sum = a.trailing_significand() + b.trailing_significand();

            Float ret;
            ret.value_ = significand_sum;
            ret.sign(a.sign());
            return ret;
        }

        Exponent exponent_a;
        Exponent exponent_b;
        Significand significand_a;
        Significand significand_b;

        std::tie(exponent_a, significand_a) = a.get_significand();
        std::tie(exponent_b, significand_b) = b.get_significand();

        // Align a and b so they share the same exponent.
        if (exponent_a != exponent_b) {
            significand_b = right_shift(significand_b, exponent_a - exponent_b);
        }

        // Add significands and take care of the carry bit
        Significand significand_sum = significand_a + significand_b;
        if ((significand_sum & (1ULL << (significand_width + 3))) != 0) {
            exponent_a += 1;
            significand_sum = right_shift(significand_sum, 1);
        }

        return round(a.sign(), exponent_a, significand_sum);
    }

    // Magnitude subtract. a and b must have the same sign and not NaN.
    // a must have greater magnitude.
    static Float subtract_magnitude(Float a, Float b) noexcept {

        // Special handling for infinity
        if (a.is_infinite()) {

            // Subtracting two infinities
            if (b.is_infinite()) {
                exception_flags |= Exception_flag::invalid_operation;
                return quiet_nan();
            }

            return a;
        }

        Exponent exponent_a;
        Exponent exponent_b;
        Significand significand_a;
        Significand significand_b;

        std::tie(exponent_a, significand_a) = a.get_significand();
        std::tie(exponent_b, significand_b) = b.get_significand();

        if (exponent_a == exponent_b) {

            Significand significand_difference = significand_a - significand_b;

            // Special treatment on zero
            if (significand_difference == 0) {
                return cancellation_zero();
            }

            return normalize_and_round(a.sign(), exponent_a, significand_difference);
        }

        // When we subtract two numbers, we might lose significance.
        // In order to still yield correct rounded result, we need one more bit to account for this.
        significand_a <<= 1;
        significand_b <<= 1;

        // Align a and b for substraction
        significand_b = right_shift(significand_b, exponent_a - exponent_b);

        Significand significand_difference = significand_a - significand_b;

        // Need to reduce exponent_a by 1 to account for the shift.
        return normalize_and_round(a.sign(), exponent_a - 1, significand_difference);
    }

public:

    static Float add(Float a, Float b) noexcept {

        // Enforce |a| > |b| for easier handling.
        if (total_order_magnitude(a, b) == Ordering::less_than) {
            std::swap(a, b);
        }

        if (a.is_nan()) return propagate_nan(a, b);

        if (a.sign() == b.sign()) {
            return add_magnitude(a, b);
        } else {
            return subtract_magnitude(a, -b);
        }
    }

    static Float subtract(Float a, Float b) noexcept {
        return add(a, -b);
    }

    static Float multiply(Float a, Float b) noexcept {

        // Enforce |a| > |b| for easier handling.
        if (total_order_magnitude(a, b) == Ordering::less_than) {
            std::swap(a, b);
        }

        if (a.is_nan()) return propagate_nan(a, b);

        bool sign = a.sign() ^ b.sign();

        // Handling infinities
        if (a.is_infinite()) {
            if (b.is_zero()) {
                exception_flags |= Exception_flag::invalid_operation;
                return quiet_nan();
            }
            a.sign(sign);
            return a;
        }

        // If either is zero
        if (b.is_zero()) {
            b.sign(sign);
            return b;
        }

        Exponent exponent_a;
        Exponent exponent_b;
        Double_significand significand_a;
        Double_significand significand_b;

        std::tie(exponent_a, significand_a) = a.get_normalized_significand();
        std::tie(exponent_b, significand_b) = b.get_normalized_significand();

        Exponent product_exponent = exponent_a + exponent_b - significand_width;

        // Normalized significand reserve 2 bits for rounding for both significand_a and significand_b
        // and we only need 2 bits, so shift one of them back by 2.
        Double_significand product = (significand_a >> 2) * significand_b;

        return normalize_and_round(sign, product_exponent, product);
    }

    static Float divide(Float a, Float b) noexcept {

        // Handling NaN
        if (a.is_nan() || b.is_nan()) return propagate_nan(a, b);

        bool sign = a.sign() ^ b.sign();

        // Handling Infinities
        if (a.is_infinite()) {

            // inf / inf = NaN
            if (b.is_infinite()) {
                exception_flags |= Exception_flag::invalid_operation;
                return quiet_nan();
            }

            a.sign(sign);
            return a;
        }

        if (b.is_infinite())  {
            return zero(sign);
        }

        // Handling zeroes
        if (a.is_zero()) {

            // 0 / 0 = NaN
            if (b.is_zero()) {
                exception_flags |= Exception_flag::invalid_operation;
                return quiet_nan();
            }

            a.sign(sign);
            return a;
        }

        // finite / 0, signaling divide_by_zero exception
        if (b.is_zero()) {
            exception_flags |= Exception_flag::divide_by_zero;
            return infinity(sign);
        }

        Exponent exponent_a;
        Exponent exponent_b;
        Significand significand_a;
        Significand significand_b;

        std::tie(exponent_a, significand_a) = a.get_normalized_significand();
        std::tie(exponent_b, significand_b) = b.get_normalized_significand();

        Exponent quotient_exponent = exponent_a - exponent_b;

        // Adjust exponent in some cases so the quotient will always be in range [1, 2)
        if (significand_a < significand_b) {
            significand_a <<= 1;
            quotient_exponent--;
        }

        // Digit-by-digit algorithm is (psuedo-code):
        //
        // quotient = 1 (since we know the quotient will be normal)
        // for (bit = 0.5; bit != 1ulp; bit /= 2) {
        //     if (significand_a >= (quotient + bit) * significand_b) {
        //         quotient += bit;
        //     }
        // }
        // if (significand_a >= quotient * significand_b) quotient += 1ulp
        //
        // As an optimization, we also keep variable
        //     remainder_over_bit = (significand_a - quotient * significand_b) / bit
        //         This value will always be smaller than (significand_b << 1), so it can always fit in Significand.
        //         The initial value of this variable will be (significand_a - significand_b) << 1.
        Significand quotient = static_cast<Significand>(1) << (significand_width + 2);
        Significand bit = static_cast<Significand>(1) << (significand_width + 1);
        Significand remainder_over_bit = (significand_a - significand_b) << 1;

        while (bit != 1) {
            // significand_a >= (quotient + bit) * significand_b <=>
            // significand_a - quotient * significand_b >= bit * significand_b <=>
            // (significand_a - quotient * significand_b) / bit >= significand_b <=>
            if (remainder_over_bit >= significand_b) {

                // we need to update the new value to (significand_a - (quotient + bit) * significand_b) / bit
                // so decrement by significand_b
                remainder_over_bit -= significand_b;
                quotient += bit;
            }

            // update bit' = bit >> 1;
            bit >>= 1;
            remainder_over_bit <<= 1;
        }

        // We still need to take action to make sure rounding is correct.
        if (remainder_over_bit != 0) {
            quotient |= 1;
        }

        // Since the result is in range [1, 2), it will always stay normalized.
        return round(sign, quotient_exponent, quotient);
    }

    Float square_root() const noexcept {

        // Handling NaN
        if (is_nan()) {
            if (is_signaling()) {
                exception_flags |= Exception_flag::invalid_operation;
                return quiet_nan();
            }

            return quiet_nan();
        }

        // Zero always return as is
        if (is_zero()) return *this;

        // For non-zero negative number, sqrt is not defined
        if (sign()) {
            exception_flags |= Exception_flag::invalid_operation;
            return quiet_nan();
        }

        if (is_infinite()) {
            return *this;
        }

        Exponent exponent;
        Significand significand;

        std::tie(exponent, significand) = get_normalized_significand();

        // Consider the number of form (1 + x) * (2 ** exponent)
        // Then if exponent is odd, the sqrt is sqrt(1 + x) * (2 ** (exponent / 2))
        // Otherwise the result is sqrt(2 * (1 + x)) * (2 ** ((exponet - 1) / 2))
        // So make sure the exponent is even.
        // After this we have to deal with calculating the square root of 1 + x or 2 * (1 + x)
        // which is in range [1, 4), so the result will be in range [1, 2). This is fine
        // since the result will remain normal.
        if (exponent % 2 != 0) {
            exponent--;
            significand <<= 1;
        }

        // Digit-by-digit algorithm is (psuedo-code):
        //
        // result = 1 (since we know the result will be normal)
        // for (bit = 0.5; bit != 1ulp; bit /= 2) {
        //     if (significand >= (result + bit) ** 2) {
        //         result += bit;
        //     }
        // }
        // if (significand >= result ** 2) result += 1ulp
        //
        // As an optimization, we also keep variable
        //     half_bit = bit / 2 (since the last bit will always be zero)
        //     half_significand_minus_result_squared_over_bit = (significand - result ** 2) / bit / 2
        //         horrible name, but since it multiplies by (1 / bit), the last bit will always be zero.
        //         By not including the bit, the result will be bounded by [0, 4) and thus can fit in Significand.
        //         and even better, given initial value of bit is 0.5, and initial value of result is 1,
        //         initial value of this variable will be exactly significand - result!
        Significand result = static_cast<Significand>(1) << (significand_width + 2);
        Significand half_bit = static_cast<Significand>(1) << significand_width;
        Significand half_significand_minus_result_squared_over_bit = significand - result;

        while (half_bit != 0) {
            // significand >= (result + bit) ** 2 <=>
            // significand >= result ** 2 + 2 * result * bit + bit ** 2 <=>
            // signicand - result ** 2 >= 2 * result * bit + bit ** 2 <=>
            // (signicand - result ** 2) / bit / 2 >= result + bit / 2 <=>
            if (half_significand_minus_result_squared_over_bit >= result + half_bit) {

                // we need to update the new value to (significand - (result + bit) ** 2) / bit / 2
                // so decrement by (significand - result**2) / bit / 2 - (significand - (result + bit)**2) / 2 which is
                // ((result + bit) ** 2 - result ** 2) / bit / 2 = result + bit / 2
                // so update half_significand_minus_result_squared_over_bit accordingly.
                half_significand_minus_result_squared_over_bit -= result + half_bit;
                result += half_bit << 1;
            }

            // update bit' = bit >> 1;
            half_bit >>= 1;
            half_significand_minus_result_squared_over_bit <<= 1;
        }

        // We still need to take action to make sure rounding is correct.
        if (half_significand_minus_result_squared_over_bit != 0) {
            result |= 1;
        }

        // Since the result is in range [1, 2), it will always stay normalized.
        return round(false, exponent / 2, result);
    }

    static Float fused_multiply_add(Float a, Float b, Float c) {

        // Enforce |a| > |b| for easier handling.
        if (total_order_magnitude(a, b) == Ordering::less_than) {
            std::swap(a, b);
        }

        // Handle NaNs.
        if (a.is_nan()) return propagate_nan(propagate_nan(a, b), c);

        bool sign_product = a.sign() ^ b.sign();

        // Handle Infinity cases
        if (a.is_infinite()) {

            // If Infinity * 0, then invalid operation
            if (b.is_zero()) {
                exception_flags |= Exception_flag::invalid_operation;
                return quiet_nan();
            }

            // Infinity + NaN
            if (c.is_nan()) return propagate_nan(c, c);

            // Infinity - Infinity
            if (c.is_infinite() && c.sign() != sign_product) {
                exception_flags |= Exception_flag::invalid_operation;
                return quiet_nan();
            }

            a.sign(sign_product);
            return a;
        }

        // NaN and Infinity handling for the addition
        if (c.is_nan()) return propagate_nan(c, c);
        if (c.is_infinite()) return c;

        // The product gives a zero
        if (b.is_zero()) {

            // 0 - 0, special treatment
            if (c.is_zero()) return cancellation_zero();

            return c;
        }

        // The following is similar to the multiplication code
        Exponent exponent_a;
        Exponent exponent_b;
        Double_significand significand_a;
        Double_significand significand_b;

        std::tie(exponent_a, significand_a) = a.get_normalized_significand();
        std::tie(exponent_b, significand_b) = b.get_normalized_significand();

        Exponent product_exponent = exponent_a + exponent_b - significand_width;
        Double_significand product = (significand_a >> 2) * significand_b;

        // a * b + 0, we can return early.
        if (c.is_zero()) {
            return normalize_and_round(sign_product, product_exponent, product);
        }

        Exponent exponent_c;
        Double_significand significand_c;

        std::tie(exponent_c, significand_c) = c.get_normalized_significand();

        // Adjust significand of c, so when the values are cancelling each other we have enough significand.
        // Note that since the last two bit of product is always zero, the shift amount can be modified to be
        // significand_width +- 1.
        exponent_c -= significand_width;
        significand_c <<= significand_width;

        // Align product and c.
        if (exponent_c < product_exponent) {
            significand_c = right_shift(significand_c, product_exponent - exponent_c);
        } else if (exponent_c > product_exponent) {
            product = right_shift(product, exponent_c - product_exponent);
            product_exponent = exponent_c;
        }

        // a * b + c
        if (c.sign() == sign_product) {
            product += significand_c;
        } else {

            // Cancellation
            if (product == significand_c) {
                return cancellation_zero();
            }

            // Make sure it is BIG - small
            if (product < significand_c) {
                std::swap(product, significand_c);
                sign_product = !sign_product;
            }

            product -= significand_c;
        }

        return normalize_and_round(sign_product, product_exponent, product);
    }

private:
    template<typename IntType>
    static Float convert_from_int_with_sign(bool sign, IntType value) {
        if (value == 0) {
            return zero();
        }

        using Result_type = std::conditional_t<(sizeof(IntType) < sizeof(Significand)), Significand, IntType>;
        Result_type significand = value;

        return normalize_and_round(sign, significand_width + 2, significand);
    }

public:
    template<typename IntType>
    static Float convert_from_int(IntType value) {
        using Unsigned_type = std::make_unsigned_t<IntType>;
        Unsigned_type unsigned_value = static_cast<Unsigned_type>(value);

        // For signed types, in addition deal with sign bits.
        if (std::is_signed<IntType>::value && value < 0) {
            return convert_from_int_with_sign<Unsigned_type>(true, -unsigned_value);
        } else {
            return convert_from_int_with_sign<Unsigned_type>(false, unsigned_value);
        }
    }

    template<typename IntType>
    IntType convert_to_int() {
        using Unsigned_type = std::make_unsigned_t<IntType>;
        constexpr IntType min = std::numeric_limits<IntType>::min();
        constexpr IntType max = std::numeric_limits<IntType>::max();

        // In unsigned case we expect compilers to optimize check for positive_max away.
        constexpr Unsigned_type positive_max = static_cast<Unsigned_type>(max);

        // By adding one here, we allow compiler to see >= 0 in unsigned case, thus optimize it away.
        constexpr Unsigned_type negative_max_inc = std::is_signed<IntType>::value ? -static_cast<Unsigned_type>(min) + 1 : 0;

        // Round NaN to the maximum value
        if (is_nan()) {
            exception_flags |= Exception_flag::invalid_operation;
            return max;
        }

        bool sign = this->sign();

        // Round positive/negative infinities to max/min, respectively.
        if (is_infinite()) {
            exception_flags |= Exception_flag::invalid_operation;
            return sign ? min : max;
        }

        if (is_zero()) {
            return 0;
        }

        Exponent exponent;
        Significand significand;
        std::tie(exponent, significand) = get_normalized_significand();

        // In case we need to round, since the last two bits are for round only,
        // we want 0b100 in signicand to represent integer 1.
        // So the effective exponent here will be significand_width.
        if (exponent < significand_width) {
            significand = right_shift(significand, significand_width - exponent);

            bool inexact;
            std::tie(inexact, significand) = round_significand(sign, significand);

            if (significand == 0) {
                if (inexact) exception_flags |= Exception_flag::inexact;
                return 0;
            }

            // Do bound check and return.
            if (!sign && significand > positive_max) {
                exception_flags |= Exception_flag::invalid_operation;
                return max;
            }

            if (sign && significand >= negative_max_inc) {
                exception_flags |= Exception_flag::invalid_operation;
                return min;
            }

            if (inexact) exception_flags |= Exception_flag::inexact;
            return sign ? -significand : significand;
        }

        // When we land here, then the number we are dealing with is actually an integer.
        // So get rid of the rounding bits.
        significand >>= 2;

        // Note that the effective exponent here is significand_width,
        // so we are going to left shift by (exponent - significand_width).
        // However the target type may not have enough bits for that. Check we have enough remaining bits.
        // Also, if the remaining_bits is negative following if will always be true.
        // However we added explicit check to hint compiler optimizations (especially in lower optimization modes).
        constexpr int remaining_bits = static_cast<int>(sizeof(IntType)) * 8 - (significand_width + 1);
        if (remaining_bits < 0 || exponent - significand_width > remaining_bits) {

            // Overflow case.
            exception_flags |= Exception_flag::invalid_operation;
            return sign ? min : max;
        }

        // Make sure the value is casted to the target type, as the target type maybe larger.
        // If target type is smaller, than remaining_bits should be negative value and this will be never reached.
        Unsigned_type result = static_cast<Unsigned_type>(significand) << (exponent - significand_width);

        // Do bound check and return.
        if (!sign && result > positive_max) {
            exception_flags |= Exception_flag::invalid_operation;
            return max;
        }

        if (sign && result >= negative_max_inc) {
            exception_flags |= Exception_flag::invalid_operation;
            return min;
        }

        return sign ? -result : result;
    }

    Float operator +(Float another) const noexcept {
        return add(*this, another);
    }

    Float operator -(Float another) const noexcept {
        return subtract(*this, another);
    }

    Float operator *(Float another) const noexcept {
        return multiply(*this, another);
    }

    Float operator /(Float another) const noexcept {
        return divide(*this, another);
    }

    /* IEEE 754-2008 5.4.2 formatOf general-computational operations > Conversion operations */
    template<typename U>
    U convert_format() const noexcept {
        static_assert(internal::Is_float<U>::value, "convert_format can only be instantiated on Float");

        // Handle NaN, infinity and zero
        if (is_nan()) {
            if (is_signaling()) exception_flags |= Exception_flag::invalid_operation;
            return U::quiet_nan();
        }

        if (is_infinite()) return U::infinity(sign());
        if (is_zero()) return U::zero(sign());

        Exponent exponent;
        Significand significand;
        std::tie(exponent, significand) = get_normalized_significand();

        return U::normalize_and_round(sign(), exponent - significand_width + U::significand_width, significand);
    }

    /* IEEE 754-2008 5.5.1 Quiet-computational operations > Sign bit operations */
    constexpr Float abs() const noexcept {
        Float ret = *this;
        ret.sign(0);
        return ret;
    }

    constexpr Float negate() const noexcept {
        Float ret = *this;
        ret.sign(!ret.sign());
        return ret;
    }

    constexpr Float copy_sign(Float another) const noexcept {
        Float ret = *this;
        ret.sign(another.sign());
        return ret;
    }

    constexpr Float copy_sign_negated(Float another) const noexcept {
        Float ret = *this;
        ret.sign(!another.sign());
        return ret;
    }

    constexpr Float copy_sign_xored(Float another) const noexcept {
        Float ret = *this;
        ret.sign(sign() ^ another.sign());
        return ret;
    }

    constexpr Float operator +() const noexcept {
        return *this;
    }

    constexpr Float operator -() const noexcept {
        return negate();
    }

    /* IEEE 754-2008 5.6.1 Signaling-computational operations > Comparisions */
    static std::optional<Ordering> compare_quiet(Float a, Float b) noexcept {
        if (a.is_nan() || b.is_nan()) {
            if (a.is_signaling() || b.is_signaling()) exception_flags |= Exception_flag::invalid_operation;
            return std::nullopt;
        }
        if (a.is_zero() && b.is_zero()) return Ordering::equal;
        return total_order(a, b);
    }

    static std::optional<Ordering> compare_signaling(Float a, Float b) noexcept {
        if (a.is_nan() || b.is_nan()) {
            exception_flags |= Exception_flag::invalid_operation;
            return std::nullopt;
        }
        if (a.is_zero() && b.is_zero()) return Ordering::equal;
        return total_order(a, b);
    }

    bool operator ==(Float another) const noexcept {
        auto result = compare_quiet(*this, another);
        if (result) {
            return *result == Ordering::equal;
        } else {
            return false;
        }
    }

    bool operator !=(Float another) const noexcept {
        return !(*this == another);
    }

    bool operator <(Float another) const noexcept {
        auto result = compare_signaling(*this, another);
        if (result) {
            return *result == Ordering::less_than;
        } else {
            return false;
        }
    }

    bool operator >(Float another) const noexcept {
        return another < *this;
    }

    bool operator <=(Float another) const noexcept {
        auto result = compare_signaling(*this, another);
        if (result) {
            return *result != Ordering::greater_than;
        } else {
            return false;
        }
    }

    bool operator >=(Float another) const noexcept {
        return another <= *this;
    }

    /* IEEE 754-2008 5.7.2 Non-computation operations > General operations */
    constexpr Class classify() const noexcept {
        bool sign = this->sign();
        Exponent exponent = biased_exponent();
        Significand significand = trailing_significand();
        if (exponent == 0) {
            if (significand == 0) return sign ? Class::negative_zero : Class::positive_zero;
            return sign ? Class::negative_subnormal : Class::positive_subnormal;
        } else if (exponent == infinity_biased_exponent) {
            if (significand == 0) return sign ? Class::negative_infinity : Class::positive_infinity;
            if ((significand & (static_cast<Significand>(1) << (significand_width - 1))) == 0) return Class::signaling_nan;
            return Class::quiet_nan;
        } else {
            return sign ? Class::negative_normal : Class::positive_normal;
        }
    }

    constexpr bool is_normal() const noexcept {
        Exponent exponent = biased_exponent();
        if (exponent != 0 && exponent != infinity_biased_exponent) {
            return true;
        }
        return false;
    }

    constexpr bool is_finite() const noexcept {
        return biased_exponent() != infinity_biased_exponent;
    }

    constexpr bool is_zero() const noexcept {
        return biased_exponent() == 0 && trailing_significand() == 0;
    }

    constexpr bool is_subnormal() const noexcept {
        return biased_exponent() == 0 && trailing_significand() != 0;
    }

    constexpr bool is_infinite() const noexcept {
        return biased_exponent() == infinity_biased_exponent && trailing_significand() == 0;
    }

    constexpr bool is_nan() const noexcept {
        return biased_exponent() == infinity_biased_exponent && trailing_significand() != 0;
    }

    constexpr bool is_signaling() const noexcept {

        // Special exponent for Infinites and NaNs
        if (biased_exponent() != infinity_biased_exponent) return false;

        auto t = trailing_significand();

        // t == 0 is Infinity
        if (t == 0) return false;

        // Signaling NaN has MSB = 0
        if ((t & (1ULL << (significand_width - 1))) == 0) return true;

        // Otherwise quite NaN
        return false;
    }

    static constexpr Ordering total_order(Float a, Float b) noexcept {
        if (a.sign() == b.sign()) {
            Ordering ret = total_order_magnitude(a, b);
            if (a.sign()) {
                return static_cast<Ordering>(-static_cast<int>(ret));
            } else {
                return ret;
            }
        } else if (a.sign()) {
            return Ordering::less_than;
        } else {
            return Ordering::greater_than;
        }
    }

    // Compare the total order between abs(a) and abs(b)
    static constexpr Ordering total_order_magnitude(Float a, Float b) noexcept {
        a.sign(false);
        b.sign(false);

        if (a.value_ == b.value_) {
            return Ordering::equal;
        } else if (a.value_ < b.value_) {
            return Ordering::less_than;
        } else {
            return Ordering::greater_than;
        }
    }

    template<int W1, int T1>
    friend class Float;
};

using Single = Float<8, 23>;
using Double = Float<11, 52>;

extern template class Float<8, 23>;
extern template class Float<11, 52>;

}

#endif
