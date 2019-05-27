#ifndef UTIL_BIT_OP_H
#define UTIL_BIT_OP_H

#include <type_traits>

namespace util {

namespace internal {

template<typename IntType, typename = void>
struct Bit_op_builtin {
    static constexpr int count_leading_zero(IntType value) noexcept {
        int bits = 0;
        while (value != 0) {
            value >>= 1;
            bits++;
        }

        // The rest of bits are all leading zeroes.
        return sizeof(IntType) * 8 - bits;
    }
};

// If the compiler we use is compatible to GCC, then use builtin clz instead.
#ifdef __GNUC__

template<>
struct Bit_op_builtin<unsigned int> {
    static constexpr int count_leading_zero(unsigned int value) noexcept {
        return __builtin_clz(value);
    }
};

template<>
struct Bit_op_builtin<unsigned long> {
    static constexpr int count_leading_zero(unsigned long value) noexcept {
        return __builtin_clzl(value);
    }
};

template<>
struct Bit_op_builtin<unsigned long long> {
    static constexpr int count_leading_zero(unsigned long long value) noexcept {
        return __builtin_clzll(value);
    }
};

#endif

// For any integers smaller than int, just use int.
template<typename IntType>
struct Bit_op_builtin<IntType, std::enable_if_t<(sizeof(IntType) < sizeof(unsigned int))>>: 
    Bit_op_builtin<unsigned int> {};

}

template<typename IntType>
static constexpr int count_leading_zero(IntType value) noexcept {
    return internal::Bit_op_builtin<std::make_unsigned_t<IntType>>::count_leading_zero(value);
}

template<typename IntType>
static constexpr int log2_floor(IntType value) noexcept {
    return sizeof(IntType) * 8 - count_leading_zero<IntType>(value) - 1;
}

}

#endif
