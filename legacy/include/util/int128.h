#ifndef UTIL_INT128_H
#define UTIL_INT128_H

namespace util {

#ifdef __SIZEOF_INT128__
using int128_t = __int128_t;
using uint128_t = __uint128_t;
#else
// Define these types even if there is no native 128-bit integer support. This will help auto completion.
using int128_t = long long;
using uint128_t = unsigned long long;
#error "int128 shim not implemented"
#endif

}

// Extend is_signed, is_unsigned, make_signed and make_unsigned to make templates depending on these type traits work.
namespace std {

template<class T>
struct is_signed;

template<class T>
struct is_unsigned;

template<class T>
struct make_signed;

template<class T>
struct make_unsigned;

template<>
struct is_signed<util::int128_t>: std::true_type {};

template<>
struct is_unsigned<util::uint128_t>: std::true_type {};

template<>
struct make_signed<util::int128_t> {
    using type = util::int128_t;
};

template<>
struct make_signed<util::uint128_t> {
    using type = util::int128_t;
};

template<>
struct make_unsigned<util::int128_t> {
    using type = util::uint128_t;
};

template<>
struct make_unsigned<util::uint128_t> {
    using type = util::uint128_t;
};

}

#endif
