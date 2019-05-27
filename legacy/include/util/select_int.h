#ifndef UTIL_SELECT_INT_H
#define UTIL_SELECT_INT_H

#include <type_traits>

#include "util/int128.h"

namespace util {

template<int B, typename = void>
struct Select_int {};

template<int B>
struct Select_int<B, std::enable_if_t<(B <= 8)>> {
    using fast = int_fast8_t;
    using least = int8_t;
};

template<int B>
struct Select_int<B, std::enable_if_t<(B > 8 && B <= 16)>> {
    using fast = int_fast16_t;
    using least = int16_t;
};

template<int B>
struct Select_int<B, std::enable_if_t<(B > 16 && B <= 32)>> {
    using fast = int_fast32_t;
    using least = int32_t;
};

template<int B>
struct Select_int<B, std::enable_if_t<(B > 32 && B <= 64)>> {
    using fast = int_fast64_t;
    using least = int64_t;
};

template<int B>
struct Select_int<B, std::enable_if_t<(B > 64 && B <= 128)>> {
    // Using the compiler builtin types here since it is not
    // a gurantee that int128.h will be included.
    using fast = int128_t;
    using least = int128_t;
};

template<int B, typename = void>
struct Select_uint {};

template<int B>
struct Select_uint<B, std::enable_if_t<(B <= 8)>> {
    using fast = uint_fast8_t;
    using least = uint8_t;
};

template<int B>
struct Select_uint<B, std::enable_if_t<(B > 8 && B <= 16)>> {
    using fast = uint_fast16_t;
    using least = uint16_t;
};

template<int B>
struct Select_uint<B, std::enable_if_t<(B > 16 && B <= 32)>> {
    using fast = uint_fast32_t;
    using least = uint32_t;
};

template<int B>
struct Select_uint<B, std::enable_if_t<(B > 32 && B <= 64)>> {
    using fast = uint_fast64_t;
    using least = uint64_t;
};

template<int B>
struct Select_uint<B, std::enable_if_t<(B > 64 && B <= 128)>> {
    // Using the compiler builtin types here since it is not
    // a gurantee that int128.h will be included.
    using fast = uint128_t;
    using least = uint128_t;
};

}

#endif
