#ifndef UTIL_ASSERT_H
#define UTIL_ASSERT_H

#include <stdexcept>

#include "config.h"
#include "util/macro.h"

// Available strategies in case of assertion failure:
// 0. Throw an util::Assertion_error when assertion failed.
// 1. Calls std::terminate when assertion failed.
// 2. Do nothing. Assuming that an assertion will never fail can yield best performance.
#define ASSERT_STRATEGY_THROW 0
#define ASSERT_STRATEGY_TERMINATE 1
#define ASSERT_STRATEGY_ASSUME 2

// Default strategy for ASSERT is throw.
#ifndef ASSERT_STRATEGY
#   define ASSERT_STRATEGY ASSERT_STRATEGY_THROW
#endif

// Hint to the compiler the likely outcome of a condition for optimisation.
#ifdef __GNUC__
#   define LIKELY(cond) __builtin_expect(!!(cond), 1)
#   define UNLIKELY(cond) __builtin_expect(!!(cond), 0)
#else
#   define LIKELY(cond) (!!(cond))
#   define UNLIKELY(cond) (!!(cond))
#endif

// Hint to the compiler that the condition will always be true for optimisation.
#ifdef _MSC_VER
#   define ASSUME(cond) __assume(cond)
#   define UNREACHABLE() ASSUME(0)
#elif defined(__clang__)
#   define UNREACHABLE() __builtin_unreachable()
#   define ASSUME(cond) __builtin_assume(cond)
#elif defined(__GNUC__)
#   define UNREACHABLE() __builtin_unreachable()
#   define ASSUME(cond) ((cond) ? static_cast<void>(0) : UNREACHABLE())
#else
#   define ASSUME(cond) static_cast<void>(0)
#   define UNREACHABLE() ASSUME(0)
#endif

namespace util {

struct Assertion_error: std::logic_error {
    explicit Assertion_error(const char *message) : std::logic_error(message) {}
};

namespace internal {

// Put this into a function so GCC will not warn about throw in noexcept.
[[noreturn]] void assertion_fail(const char *message);

}

} // util

#if ASSERT_STRATEGY == ASSERT_STRATEGY_THROW
#   define ASSERT(cond) \
    (LIKELY(cond) ? static_cast<void>(0) \
                  : util::internal::assertion_fail("assertion `" #cond "` failed at " __FILE__ ":" STRINGIFY(__LINE__)))
#elif ASSERT_STRATEGY == ASSERT_STRATEGY_TERMINATE
#   define ASSERT(cond) (LIKELY(cond) ? static_cast<void>(0) : std::terminate())
#else
#   define ASSERT(cond) ASSUME(cond)
#endif

#endif
