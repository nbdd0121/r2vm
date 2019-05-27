// This project utilizes latest C++ features. Some of these features are not available in some old compilers. Therefore
// it is essential that features are tested.

// Included to test existence of std::byte
#include <cstddef>

/* C++11 and C++14 features */

// We require the compiler to fully support C++14.
// MSVC does not define __cplusplus correctly, and check _MSC_VER instead.
#if __cplusplus < 201402L && _MSC_VER < 1910
#   error "C++14 must be fully supported"
#endif

/* C++17 features */

// Test existence of [[maybe_unused]], std::byte, and relaxed enum class initialization rule.
[[maybe_unused]]
static void test_byte() {
    [[maybe_unused]]
    std::byte byte { 42 };
}

// Test nested namespace support.
namespace a::b {}

/* Platform compatibility test */

// Big-endian CPUs is rarely used nowadays and supporting them is just a pain. Assert that the machine is little
// endian. For MSVC we don't have to perform the check since Windows supports only little endian architectures.
#if !defined(_MSC_VER) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#   error "Byte order must be little endian"
#endif
