#![allow(dead_code)]

use std::ops;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU32, Ordering as MemOrder};
use super::int::{CastFrom, CastTo, Int, UInt};

// #region Rounding mode constant and manipulation
//

const RM_TIES_TO_EVEN    : u32 = 0b000;
const RM_TOWARD_ZERO     : u32 = 0b001;
const RM_TOWARD_NEGATIVE : u32 = 0b010;
const RM_TOWARD_POSITIVE : u32 = 0b011;
const RM_TIES_TO_AWAY    : u32 = 0b100;

thread_local!(
    /// We use atomic here merely to avoid the cost of RefCell. So we should use relaxed ordering.
    static ROUNDING_MODE: AtomicU32 = AtomicU32::new(0)
);

#[inline]
fn get_rounding_mode() -> u32 {
    ROUNDING_MODE.with(|flags| {
        flags.load(MemOrder::Relaxed)
    })
}

#[inline]
pub fn set_rounding_mode(flag: u32) {
    ROUNDING_MODE.with(|flags| {
        flags.store(flag, MemOrder::Relaxed);
    })
}

//
// #endregion

// #region Exception flags constant and manipulation

const EX_INEXACT           : u32 = 1;
const EX_UNDERFLOW         : u32 = 2;
const EX_OVERFLOW          : u32 = 4;
const EX_DIVIDE_BY_ZERO    : u32 = 8;
const EX_INVALID_OPERATION : u32 = 16;

thread_local!(
    /// We use atomic here merely to avoid the cost of RefCell. So we should use relaxed ordering.
    static EXCEPTION_FLAGS: AtomicU32 = AtomicU32::new(0)
);

#[inline]
pub fn clear_exception_flag() {
    EXCEPTION_FLAGS.with(|flags| {
        flags.store(0, MemOrder::Relaxed);
    })
}

#[inline]
pub fn get_exception_flag() -> u32 {
    EXCEPTION_FLAGS.with(|flags| {
        flags.load(MemOrder::Relaxed)
    })
}

#[inline]
fn set_exception_flag(flag: u32) {
    EXCEPTION_FLAGS.with(|flags| {
        flags.fetch_or(flag, MemOrder::Relaxed);
    })
}

//
// #endregion

const CLASS_NEGATIVE_INFINITY  : u32 = 0;
const CLASS_NEGATIVE_NORMAL    : u32 = 1;
const CLASS_NEGATIVE_SUBNORMAL : u32 = 2;
const CLASS_NEGATIVE_ZERO      : u32 = 3;
const CLASS_POSITIVE_ZERO      : u32 = 4;
const CLASS_POSITIVE_SUBNORMAL : u32 = 5;
const CLASS_POSITIVE_NORMAL    : u32 = 6;
const CLASS_POSITIVE_INFINITY  : u32 = 7;
const CLASS_SIGNALING_NAN      : u32 = 8;
const CLASS_QUIET_NAN          : u32 = 9;

pub trait FpDesc: Copy {
    const EXPONENT_WIDTH: u32;
    const SIGNIFICAND_WIDTH: u32;
    type Holder: UInt + CastFrom<u32> + CastTo<u32> + CastTo<Self::DoubleHolder>;
    type DoubleHolder: UInt + CastTo<Self::Holder>;
}

#[derive(Clone, Copy)]
pub struct Fp<Desc: FpDesc>(pub Desc::Holder);

impl<Desc: FpDesc> Fp<Desc> {

    // Special exponent values
    // All biased exponents have type u32, and unbiased exponents have type i32.
    const INFINITY_BIASED_EXPONENT: u32 = (1 << Desc::EXPONENT_WIDTH) - 1;
    const MAXIMUM_BIASED_EXPONENT : u32 = Self::INFINITY_BIASED_EXPONENT - 1;
    const EXPONENT_BIAS           : i32 = (1 << (Desc::EXPONENT_WIDTH - 1)) - 1;
    const MINIMUM_EXPONENT        : i32 = 1 - Self::EXPONENT_BIAS;
    const MAXIMUM_EXPONENT        : i32 = Self::MAXIMUM_BIASED_EXPONENT as i32 - Self::EXPONENT_BIAS;

    #[inline]
    pub fn new(value: Desc::Holder) -> Self {
        Fp(value)
    }

    // #region Special constants values
    //

    pub fn quiet_nan() -> Self {
        let mut value = Self(Desc::Holder::zero());
        value.set_biased_exponent(Self::INFINITY_BIASED_EXPONENT);
        value.set_trailing_significand(Desc::Holder::one() << (Desc::SIGNIFICAND_WIDTH - 1));
        value
    }

    pub fn infinity(sign: bool) -> Self {
        let mut value = Self(Desc::Holder::zero());
        value.set_sign(sign);
        value.set_biased_exponent(Self::INFINITY_BIASED_EXPONENT);
        value
    }

    pub fn zero(sign: bool) -> Self {
        let mut value = Self(Desc::Holder::zero());
        value.set_sign(sign);
        value
    }

    //
    // #endregion

    // #region shifting and rounding helper functions
    //

    /// Shift while preserving rounding property.
    /// The last bit is the OR of all bits shifted away.
    /// For details check the comment of round()
    fn right_shift<T: UInt>(significand: T, shift_amount: u32) -> T {
        let (mut value, residue) = if shift_amount >= T::bit_width() {
            // If shift amount is large enough, then the entire significand is residue
            (T::zero(), significand)
        } else {
            let residue =  significand & ((T::one() << shift_amount) - T::one());
            (significand >> shift_amount, residue)
        };
        if residue != T::zero() {
            value |= T::one();
        }
        value
    }

    /// Normalized the significand while preserving rounding property.
    fn normalize<T: UInt + CastTo<Desc::Holder>>(exponent: i32, significand: T) -> (i32, Desc::Holder) {
        let width = significand.log2_floor() - 2;
        let width_diff = width as i32 - Desc::SIGNIFICAND_WIDTH as i32;
        let exponent = exponent + width_diff;
        let significand = if width_diff <= 0 {
            // For left-shift, convert before shift, in case it is smaller than Desc::Holder
            significand.cast_to() << (-width_diff as u32)
        } else {
            // For right-shift, keep the original type first before shift, in case where it is
            // larger than Desc::Holder.
            Self::right_shift(significand, width_diff as u32).cast_to()
        };
        (exponent, significand)
    }

    /// Round the significand based on current rounding mode and last two bits.
    fn round_significand(sign: bool, mut significand: Desc::Holder) -> (bool, Desc::Holder) {
        let mut inexact = false;

        if (significand & 3u32.cast_to()) != Desc::Holder::zero() {
            inexact = true;

            match get_rounding_mode() {
                RM_TIES_TO_EVEN =>
                    significand += ((significand >> 2) & Desc::Holder::one()) + Desc::Holder::one(),
                RM_TOWARD_ZERO => (),
                RM_TOWARD_NEGATIVE => if sign { significand += 3u32.cast_to() }
                RM_TOWARD_POSITIVE => if !sign { significand += 3u32.cast_to() }
                RM_TIES_TO_AWAY =>
                    // If last two bits are 10 or 11, then round up.
                    significand += 2u32.cast_to(),
                _ => unreachable!(),
            }
        }

        (inexact, significand >> 2)
    }

    /// Get the finite number overflowing result in current rounding mode.
    fn round_overflow(sign: bool) -> Self {
        set_exception_flag(EX_OVERFLOW | EX_INEXACT);

        // When we are rounding away from the Infinity, we set the result
        // to be the largest finite number.
        let mut value = Self::infinity(sign);

        let rm = get_rounding_mode();
        if (sign && rm == RM_TOWARD_POSITIVE) || (!sign && rm == RM_TOWARD_NEGATIVE) || rm == RM_TOWARD_ZERO {

            // Decrement by one will shift value from infinity to max finite number
            value.0 -= Desc::Holder::one();
        }

        value
    }

    /// Principle about rounding: to round correctly, we need two piece of information:
    /// 1. the first bit beyond target precision
    /// 2. whether we discard any bits after that bit.
    /// * If a=0, b=0, then the remainder is 0.
    /// * If a=0, b=1, then the remainder is in range (0, 0.5)
    /// * If a=1, b=0, then the remainder is 0.5.
    /// * If a=1, b=1, then the remainder is in range (0.5, 1).
    /// Therefore we require signicand to contain two more bits beyond precision.
    /// Input must be normal.
    fn round(sign: bool, mut exponent: i32, significand: Desc::Holder) -> Self {
        if exponent > Self::MAXIMUM_EXPONENT { return Self::round_overflow(sign) }

        let mut value = Self(Desc::Holder::zero());
        value.set_sign(sign);

        // To yield correct result, we need to first subnormalize the number before rounding.
        let mut rounded = significand;
        if exponent < Self::MINIMUM_EXPONENT {
            rounded = Self::right_shift(rounded, (Self::MINIMUM_EXPONENT - exponent) as u32);
        }

        let (inexact, mut rounded) = Self::round_significand(sign, rounded);
        if inexact {
            set_exception_flag(EX_INEXACT);

            // When the significand is all 1 and rounding causes it to round up.
            // Since when this happens, resulting significand should be all zero.
            // In this case casting significand to "Significand" will yield correct
            // result, but we need to increment exponent.
            if rounded == Desc::Holder::one() << (Desc::SIGNIFICAND_WIDTH + 1) {
                exponent += 1;
                rounded >>= 1;
            }
        }

        // Underflow or subnormal
        if exponent < Self::MINIMUM_EXPONENT {

            // The border between subnormal and normal.
            if rounded == Desc::Holder::one() << Desc::SIGNIFICAND_WIDTH {

                // In this special case, we need to deal with underflow flag very carefully.
                // IEEE specifies that the underflow flag should only be set if rounded result
                // in *unbounded* exponent will yield to an overflow.
                if Self::round_significand(sign, significand).1 !=
                    Desc::Holder::one() << (Desc::SIGNIFICAND_WIDTH + 1) {

                    set_exception_flag(EX_UNDERFLOW);
                }

                value.set_biased_exponent(1);
                value.set_trailing_significand(Desc::Holder::zero());
                return value;
            }

            if inexact {
                set_exception_flag(EX_UNDERFLOW);
            }

            value.set_biased_exponent(0);
            value.set_trailing_significand(rounded);
            return value;
        }

        if exponent > Self::MAXIMUM_EXPONENT { return Self::round_overflow(sign) }

        value.set_biased_exponent((exponent + Self::EXPONENT_BIAS) as u32);
        value.set_trailing_significand(rounded);
        value
    }

    fn normalize_and_round<T: UInt + CastTo<Desc::Holder>>(sign: bool, exponent: i32, significand: T) -> Self {
        let (exponent, final_significand) = Self::normalize(exponent, significand);
        Self::round(sign, exponent, final_significand)
    }

    fn propagate_nan(a: Self, b: Self) -> Self {
        if a.is_signaling() || b.is_signaling() {
            set_exception_flag(EX_INVALID_OPERATION);
        }
        Self::quiet_nan()
    }

    fn cancellation_zero() -> Self {
        Self::zero(get_rounding_mode() == RM_TOWARD_NEGATIVE)
    }

    // #endregion

    // #region Component accessors
    //

    fn sign(&self) -> bool {
        self.0 >> (Desc::EXPONENT_WIDTH + Desc::SIGNIFICAND_WIDTH) != Desc::Holder::zero()
    }

    fn set_sign(&mut self, sign: bool) {
        let mask = Desc::Holder::one() << (Desc::EXPONENT_WIDTH + Desc::SIGNIFICAND_WIDTH);
        if sign {
            self.0 |= mask;
        } else {
            self.0 &=! mask;
        }
    }

    fn biased_exponent(&self) -> u32 {
        let mask = (Desc::Holder::one() << Desc::EXPONENT_WIDTH) - Desc::Holder::one();
        CastTo::<u32>::cast_to((self.0 >> Desc::SIGNIFICAND_WIDTH) & mask)
    }

    /// Set the biased exponent of the floating pointer number.
    /// Only up to exponent_width bits are respected and all other bits are ignored.
    fn set_biased_exponent(&mut self, exp: u32) {
        let mask = ((Desc::Holder::one() << Desc::EXPONENT_WIDTH) - Desc::Holder::one()) << Desc::SIGNIFICAND_WIDTH;
        self.0 = (self.0 &! mask) | ((CastTo::<Desc::Holder>::cast_to(exp) << Desc::SIGNIFICAND_WIDTH) & mask);
    }

    fn trailing_significand(&self) -> Desc::Holder {
        let mask = (Desc::Holder::one() << Desc::SIGNIFICAND_WIDTH) - Desc::Holder::one();
        self.0 & mask
    }

    // Set the trailing significand of the floating pointer number.
    // Only up to significand_width bits are respected and all other bits are ignored.
    fn set_trailing_significand(&mut self, value: Desc::Holder) {
        let mask = (Desc::Holder::one() << Desc::SIGNIFICAND_WIDTH) - Desc::Holder::one();
        self.0 = (self.0 &! mask) | (value & mask);
    }

    fn get_normalized_significand(&self) -> (i32, Desc::Holder) {
        let biased_exponent = self.biased_exponent();
        let trailing_significand = self.trailing_significand();

        // We couldn't handle this
        if biased_exponent == Self::INFINITY_BIASED_EXPONENT ||
            (biased_exponent == 0 && trailing_significand == Desc::Holder::zero()) { panic!() }

        if biased_exponent == 0 {
            let width_diff = Desc::SIGNIFICAND_WIDTH - trailing_significand.log2_floor();
            return (Self::MINIMUM_EXPONENT - width_diff as i32, trailing_significand << (width_diff + 2));
        }

        let exponent = biased_exponent as i32 - Self::EXPONENT_BIAS;
        let significand = trailing_significand | (Desc::Holder::one() << Desc::SIGNIFICAND_WIDTH);
        (exponent, significand << 2)
    }

    fn get_significand(&self) -> (i32, Desc::Holder)  {
        let biased_exponent = self.biased_exponent();
        let trailing_significand = self.trailing_significand();

        // We couldn't handle this
        if biased_exponent == Self::INFINITY_BIASED_EXPONENT  { panic!() }

        if biased_exponent == 0 {
            // Perform lvalue-rvalue conversion to get rid of ODR-use of minimum_exponent.
            return (Self::MINIMUM_EXPONENT, trailing_significand << 2);
        }

        let exponent = biased_exponent as i32 - Self::EXPONENT_BIAS;
        let significand = trailing_significand | (Desc::Holder::one() << Desc::SIGNIFICAND_WIDTH);
        (exponent, significand << 2)
    }

    //
    // #endregion

    // #region arithmetic operations
    // IEEE 754-2008 5.4.1 formatOf general-computational operations > Arithmetic operations
    //

    /// Magnitude add. a and b must have the same sign and not NaN.
    /// a must have greater magnitude.
    fn add_magnitude(a: Self, b: Self) -> Self {

        // Handling for Infinity
        if a.is_infinite() { return a }

        // If both are subnormal, then neither signifcand we retrieved below will be normal.
        // So we handle them specially here.
        if a.biased_exponent() == 0 {
            let significand_sum = a.trailing_significand() + b.trailing_significand();
            let mut ret = Self(significand_sum);
            ret.set_sign(a.sign());
            return ret;
        }

        let (mut exponent_a, significand_a) = a.get_significand();
        let (exponent_b, mut significand_b) = b.get_significand();

        // Align a and b so they share the same exponent.
        if exponent_a != exponent_b {
            significand_b = Self::right_shift(significand_b, (exponent_a - exponent_b) as u32);
        }

        // Add significands and take care of the carry bit
        let mut significand_sum = significand_a + significand_b;
        if (significand_sum & (Desc::Holder::one() << (Desc::SIGNIFICAND_WIDTH + 3))) != Desc::Holder::zero() {
            exponent_a += 1;
            significand_sum = Self::right_shift(significand_sum, 1);
        }

        Self::round(a.sign(), exponent_a, significand_sum)
    }

    /// Magnitude subtract. a and b must have the same sign and not NaN.
    /// a must have greater magnitude.
    fn subtract_magnitude(a: Self, b: Self) -> Self {

        // Special handling for infinity
        if a.is_infinite() {

            // Subtracting two infinities
            if b.is_infinite() {
                set_exception_flag(EX_INVALID_OPERATION);
                return Self::quiet_nan();
            }

            return a;
        }

        let (exponent_a, mut significand_a) = a.get_significand();
        let (exponent_b, mut significand_b) = b.get_significand();

        if exponent_a == exponent_b {
            let significand_difference = significand_a - significand_b;

            // Special treatment on zero
            if significand_difference == Desc::Holder::zero() {
                return Self::cancellation_zero();
            }

            return Self::normalize_and_round(a.sign(), exponent_a, significand_difference);
        }

        // When we subtract two numbers, we might lose significance.
        // In order to still yield correct rounded result, we need one more bit to account for this.
        significand_a <<= 1;
        significand_b <<= 1;

        // Align a and b for substraction
        significand_b = Self::right_shift(significand_b, (exponent_a - exponent_b) as u32);

        let significand_difference = significand_a - significand_b;

        // Need to reduce exponent_a by 1 to account for the shift.
        return Self::normalize_and_round(a.sign(), exponent_a - 1, significand_difference);
    }

    fn multiply(a: Self, b: Self) -> Self {

        // Enforce |a| > |b| for easier handling.
        let (mut a, mut b) = if Self::total_order_magnitude(a, b) == Ordering::Less { (b, a) } else { (a, b) };

        if a.is_nan() { return Self::propagate_nan(a, b) }

        let sign = a.sign() ^ b.sign();

        // Handling infinities
        if a.is_infinite() {
            if b.is_zero() {
                set_exception_flag(EX_INVALID_OPERATION);
                return Self::quiet_nan();
            }
            a.set_sign(sign);
            return a;
        }

        // If either is zero
        if b.is_zero() {
            b.set_sign(sign);
            return b;
        }

        let (exponent_a, significand_a) = a.get_normalized_significand();
        let (exponent_b, significand_b) = b.get_normalized_significand();

        let product_exponent = exponent_a + exponent_b - Desc::SIGNIFICAND_WIDTH as i32;

        // Normalized significand reserve 2 bits for rounding for both significand_a and significand_b
        // and we only need 2 bits, so shift one of them back by 2.
        let product = CastTo::<Desc::DoubleHolder>::cast_to(significand_a >> 2) *
                      CastTo::<Desc::DoubleHolder>::cast_to(significand_b);

        Self::normalize_and_round(sign, product_exponent, product)
    }

    fn divide(a: Self, b: Self) -> Self {

        // Handling NaN
        if a.is_nan() || b.is_nan() { return Self::propagate_nan(a, b) }

        let sign = a.sign() ^ b.sign();

        // Handling Infinities
        if a.is_infinite() {

            // inf / inf = NaN
            if b.is_infinite() {
                set_exception_flag(EX_INVALID_OPERATION);
                return Self::quiet_nan();
            }

            return Self::infinity(sign);
        }

        if b.is_infinite()  {
            return Self::zero(sign);
        }

        // Handling zeroes
        if a.is_zero() {

            // 0 / 0 = NaN
            if b.is_zero() {
                set_exception_flag(EX_INVALID_OPERATION);
                return Self::quiet_nan();
            }

            return Self::zero(sign);
        }

        // finite / 0, signaling divide_by_zero exception
        if b.is_zero() {
            set_exception_flag(EX_DIVIDE_BY_ZERO);
            return Self::infinity(sign);
        }

        let (exponent_a, mut significand_a) = a.get_normalized_significand();
        let (exponent_b, significand_b) = b.get_normalized_significand();

        let mut quotient_exponent = exponent_a - exponent_b;

        // Adjust exponent in some cases so the quotient will always be in range [1, 2)
        if significand_a < significand_b {
            significand_a <<= 1;
            quotient_exponent -= 1;
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
        let mut quotient = Desc::Holder::one() << (Desc::SIGNIFICAND_WIDTH + 2);
        let mut bit = Desc::Holder::one() << (Desc::SIGNIFICAND_WIDTH + 1);
        let mut remainder_over_bit = (significand_a - significand_b) << 1;

        while bit != Desc::Holder::one() {
            // significand_a >= (quotient + bit) * significand_b <=>
            // significand_a - quotient * significand_b >= bit * significand_b <=>
            // (significand_a - quotient * significand_b) / bit >= significand_b <=>
            if remainder_over_bit >= significand_b {

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
        if remainder_over_bit != Desc::Holder::zero() {
            quotient |= Desc::Holder::one();
        }

        // Since the result is in range [1, 2), it will always stay normalized.
        Self::round(sign, quotient_exponent, quotient)
    }

    pub fn square_root(self) -> Self {

        // Handling NaN
        if self.is_nan() {
            if self.is_signaling() {
                set_exception_flag(EX_DIVIDE_BY_ZERO);
                return Self::quiet_nan();
            }

            return Self::quiet_nan();
        }

        // Zero always return as is
        if self.is_zero() { return self }

        // For non-zero negative number, sqrt is not defined
        if self.sign() {
            set_exception_flag(EX_INVALID_OPERATION);
            return Self::quiet_nan();
        }

        if self.is_infinite() { return self }

        let (mut exponent, mut significand) = self.get_normalized_significand();

        // Consider the number of form (1 + x) * (2 ** exponent)
        // Then if exponent is odd, the sqrt is sqrt(1 + x) * (2 ** (exponent / 2))
        // Otherwise the result is sqrt(2 * (1 + x)) * (2 ** ((exponet - 1) / 2))
        // So make sure the exponent is even.
        // After this we have to deal with calculating the square root of 1 + x or 2 * (1 + x)
        // which is in range [1, 4), so the result will be in range [1, 2). This is fine
        // since the result will remain normal.
        if exponent % 2 != 0 {
            exponent -= 1;
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
        let mut result = Desc::Holder::one() << (Desc::SIGNIFICAND_WIDTH + 2);
        let mut half_bit = Desc::Holder::one() << Desc::SIGNIFICAND_WIDTH;
        let mut half_significand_minus_result_squared_over_bit = significand - result;

        while half_bit != Desc::Holder::zero() {
            // significand >= (result + bit) ** 2 <=>
            // significand >= result ** 2 + 2 * result * bit + bit ** 2 <=>
            // signicand - result ** 2 >= 2 * result * bit + bit ** 2 <=>
            // (signicand - result ** 2) / bit / 2 >= result + bit / 2 <=>
            if half_significand_minus_result_squared_over_bit >= result + half_bit {

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
        if half_significand_minus_result_squared_over_bit != Desc::Holder::zero() {
            result |= Desc::Holder::one();
        }

        // Since the result is in range [1, 2), it will always stay normalized.
        return Self::round(false, exponent / 2, result);
    }

    pub fn fused_multiply_add(a: Self, b: Self, c: Self) -> Self {

        // Enforce |a| > |b| for easier handling.
        let (a, b) = if Self::total_order_magnitude(a, b) == Ordering::Less { (b, a) } else { (a, b) };

        // Handle NaNs.
        if a.is_nan() { return Self::propagate_nan(Self::propagate_nan(a, b), c) }

        let mut sign_product = a.sign() ^ b.sign();

        // Handle Infinity cases
        if a.is_infinite() {

            // If Infinity * 0, then invalid operation
            if b.is_zero() {
                set_exception_flag(EX_INVALID_OPERATION);
                return Self::quiet_nan();
            }

            // Infinity + NaN
            if c.is_nan() { return Self::propagate_nan(c, c) }

            // Infinity - Infinity
            if c.is_infinite() && c.sign() != sign_product {
                set_exception_flag(EX_INVALID_OPERATION);
                return Self::quiet_nan();
            }

            return Self::infinity(sign_product);
        }

        // NaN and Infinity handling for the addition
        if c.is_nan() { return Self::propagate_nan(c, c) }
        if c.is_infinite() { return c }

        // The product gives a zero
        if b.is_zero() {
            // 0 - 0, special treatment
            if c.is_zero() { return Self::cancellation_zero() }
            return c;
        }

        // The following is similar to the multiplication code
        let (exponent_a, significand_a) = a.get_normalized_significand();
        let (exponent_b, significand_b) = b.get_normalized_significand();

        let mut product_exponent = exponent_a + exponent_b - Desc::SIGNIFICAND_WIDTH as i32;
        let mut product = CastTo::<Desc::DoubleHolder>::cast_to(significand_a >> 2) *
                          CastTo::<Desc::DoubleHolder>::cast_to(significand_b);

        // a * b + 0, we can return early.
        if c.is_zero() {
            return Self::normalize_and_round(sign_product, product_exponent, product);
        }

        let (mut exponent_c, significand_c) = c.get_normalized_significand();
        let mut significand_c = CastTo::<Desc::DoubleHolder>::cast_to(significand_c);

        // Adjust significand of c, so when the values are cancelling each other we have enough significand.
        // Note that since the last two bit of product is always zero, the shift amount can be modified to be
        // significand_width +- 1.
        exponent_c -= Desc::SIGNIFICAND_WIDTH as i32;
        significand_c <<= Desc::SIGNIFICAND_WIDTH;

        // Align product and c.
        if exponent_c < product_exponent {
            significand_c = Self::right_shift(significand_c, (product_exponent - exponent_c) as u32);
        } else if exponent_c > product_exponent {
            product = Self::right_shift(product, (exponent_c - product_exponent) as u32);
            product_exponent = exponent_c;
        }

        // a * b + c
        if c.sign() == sign_product {
            product += significand_c;
        } else {

            // Cancellation
            if product == significand_c {
                return Self::cancellation_zero();
            }

            // Make sure it is BIG - small
            if product < significand_c {
                sign_product = !sign_product;
                product = significand_c - product
            } else {
                product -= significand_c;
            }
        }

        return Self::normalize_and_round(sign_product, product_exponent, product);
    }

    // #region sign bit operations
    // IEEE 754-2008 5.5.1 Quiet-computational operations > Sign bit operations
    //

    pub fn abs(mut self) -> Self {
        self.set_sign(false);
        self
    }

    pub fn negate(mut self) -> Self {
        self.set_sign(!self.sign());
        self
    }

    pub fn copy_sign(mut self, another: Self) -> Self {
        self.set_sign(another.sign());
        self
    }

    pub fn copy_sign_negated(mut self, another: Self) -> Self {
        self.set_sign(!another.sign());
        self
    }

    pub fn copy_sign_xored(mut self, another: Self) -> Self {
        self.set_sign(self.sign() ^ another.sign());
        self
    }

    //
    // #endregion

    // #region Classification
    //

    #[allow(dead_code)]
    pub fn is_normal(&self) -> bool {
        let exponent = self.biased_exponent();
        exponent != 0 && exponent != Self::INFINITY_BIASED_EXPONENT
    }

    #[allow(dead_code)]
    pub fn is_finite(&self) -> bool {
        self.biased_exponent() != Self::INFINITY_BIASED_EXPONENT
    }

    pub fn is_zero(&self) -> bool {
        self.biased_exponent() == 0 &&
        self.trailing_significand() == Desc::Holder::zero()
    }

    #[allow(dead_code)]
    pub fn is_subnormal(&self) -> bool {
        self.biased_exponent() == 0 &&
        self.trailing_significand() != Desc::Holder::zero()
    }

    pub fn is_infinite(&self) -> bool {
        self.biased_exponent() == Self::INFINITY_BIASED_EXPONENT &&
        self.trailing_significand() == Desc::Holder::zero()
    }

    pub fn is_nan(&self) -> bool {
        self.biased_exponent() == Self::INFINITY_BIASED_EXPONENT &&
        self.trailing_significand() != Desc::Holder::zero()
    }

    pub fn is_signaling(&self) -> bool {
        // Special exponent for Infinites and NaNs
        if self.biased_exponent() != Self::INFINITY_BIASED_EXPONENT { return false }

        let t = self.trailing_significand();

        // t == 0 is Infinity
        if t == Desc::Holder::zero() { return false }

        // Signaling NaN has MSB = 0, otherwise quiet NaN
        t & (Desc::Holder::one() << (Desc::SIGNIFICAND_WIDTH - 1)) == Desc::Holder::zero()
    }

    /* IEEE 754-2008 5.7.2 Non-computation operations > General operations */
    pub fn classify(&self) -> u32 {
        let sign = self.sign();
        let exponent = self.biased_exponent();
        let significand = self.trailing_significand();
        let positive_class = if exponent == 0 {
            if significand == Desc::Holder::zero() {
                CLASS_POSITIVE_ZERO
            } else {
                CLASS_POSITIVE_SUBNORMAL
            }
        } else if exponent == Self::INFINITY_BIASED_EXPONENT {
            if significand == Desc::Holder::zero() {
                CLASS_POSITIVE_INFINITY
            } else if (significand & (Desc::Holder::one() << (Desc::SIGNIFICAND_WIDTH - 1))) == Desc::Holder::zero() {
                return CLASS_SIGNALING_NAN;
            } else {
                return CLASS_QUIET_NAN;
            }
        } else {
            CLASS_POSITIVE_NORMAL
        };
        // We use the property that negative and positive classes add up to 7.
        if sign { 7 - positive_class } else { positive_class }
    }

    //
    // #endregion

    // #region Comparison
    //

    /* IEEE 754-2008 5.3.1 Homogeneous general-computational operations > General operations */
    pub fn min_max(a: Self, b: Self) -> (Self, Self) {
        if a.is_nan() || b.is_nan() {
            if a.is_signaling() || b.is_signaling() {
                set_exception_flag(EX_INVALID_OPERATION);
                return (Self::quiet_nan(), Self::quiet_nan());
            }

            if a.is_nan() {
                if b.is_nan() {
                    return (Self::quiet_nan(), Self::quiet_nan());
                }
                return (b, b);
            }

            return (a, a);
        }

        if Self::total_order(a, b) == Ordering::Less {
            return (a, b);
        } else {
            return (b, a);
        }
    }

    pub fn min(a: Self, b: Self) -> Self {
        Self::min_max(a, b).0
    }

    pub fn max(a: Self, b: Self) -> Self {
        Self::min_max(a, b).1
    }

    /* IEEE 754-2008 5.6.1 Signaling-computational operations > Comparisions */
    pub fn compare_quiet(a: Self, b: Self) -> Option<Ordering> {
        if a.is_nan() || b.is_nan() {
            return None;
        }
        if a.is_zero() && b.is_zero() { return Some(Ordering::Equal) }
        Some(Self::total_order(a, b))
    }

    pub fn compare_signaling(a: Self, b: Self) -> Option<Ordering> {
        if a.is_nan() || b.is_nan() {
            set_exception_flag(EX_INVALID_OPERATION);
            return None;
        }
        if a.is_zero() && b.is_zero() { return Some(Ordering::Equal) }
        Some(Self::total_order(a, b))
    }

    pub fn total_order(a: Self, b: Self) -> Ordering {
        if a.sign() == b.sign() {
            let ret = Self::total_order_magnitude(a, b);
            if a.sign() {
                ret.reverse()
            } else {
                ret
            }
        } else if a.sign() {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }

    /// Compare the total order between abs(a) and abs(b)
    pub fn total_order_magnitude(mut a: Self, mut b: Self) -> Ordering {
        a.set_sign(false);
        b.set_sign(false);
        a.0.cmp(&b.0)
    }

    //
    // #endregion
}

impl<Desc: FpDesc> std::cmp::PartialEq for Fp<Desc> {
    fn eq(&self, other: &Self) -> bool {
        Fp::compare_quiet(*self, *other) == Some(Ordering::Equal)
    }
}

impl<Desc: FpDesc> std::cmp::PartialOrd for Fp<Desc> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Fp::compare_signaling(*self, *other)
    }
}

impl<Desc: FpDesc> ops::Add<Self> for Fp<Desc> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {

        // Enforce |a| > |b| for easier handling.
        let (a, b) = if Self::total_order_magnitude(self, rhs) == Ordering::Less { (rhs, self) } else { (self, rhs) };

        if a.is_nan() { return Self::propagate_nan(a, b) }

        if a.sign() == b.sign() {
            Self::add_magnitude(a, b)
        } else {
            Self::subtract_magnitude(a, -b)
        }
    }
}

impl<Desc: FpDesc> ops::Sub<Self> for Fp<Desc> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}

impl<Desc: FpDesc> ops::Mul<Self> for Fp<Desc> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::multiply(self, rhs)
    }
}

impl<Desc: FpDesc> ops::Div<Self> for Fp<Desc> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::divide(self, rhs)
    }
}

impl<Desc: FpDesc> ops::Neg for Fp<Desc> {
    type Output = Self;
    fn neg(self) -> Self {
        self.negate()
    }
}

#[derive(Clone, Copy)]
pub struct F32Desc;

impl FpDesc for F32Desc {
    const EXPONENT_WIDTH: u32 = 8;
    const SIGNIFICAND_WIDTH: u32 = 23;
    type Holder = u32;
    type DoubleHolder = u64;
}

#[derive(Clone, Copy)]
pub struct F64Desc;

impl FpDesc for F64Desc {
    const EXPONENT_WIDTH: u32 = 11;
    const SIGNIFICAND_WIDTH: u32 = 52;
    type Holder = u64;
    type DoubleHolder = u128;
}

pub type F32 = Fp<F32Desc>;
pub type F64 = Fp<F64Desc>;
