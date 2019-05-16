#ifndef UTIL_SCOPE_EXIT_H
#define UTIL_SCOPE_EXIT_H

#include <utility>

#include "util/macro.h"

#define CONCAT_IMPL(x,y) x##y
#define CONCAT(x,y) CONCAT_IMPL(x,y)

// Defines a handy macro for dealing with code that has to be executed when leaving the scope. It can be used like this:
// SCOPE_EXIT { /* code to execute */ };  // <-- note that the semicolon is needed.
#define SCOPE_EXIT auto CONCAT(scope_exit_guard_, __LINE__) = util::internal::Scope_exit_guard_builder{} << [&]

namespace util {

namespace internal {

// The actual scope-exit guard that executes the callback when leaving the scope.
template<typename T>
struct Scope_exit_guard {
    T callback;
    Scope_exit_guard(T&& t): callback(std::forward<T>(t)) {};
    ~Scope_exit_guard() {
        callback();
    }
};

// The builder which is used in the macro to allow better coding style. It's only usage is to provide a binary
// operator so that the macro can take lambda expression after it.
struct Scope_exit_guard_builder {
    template<typename T>
    Scope_exit_guard<T> operator <<(T&& t) {
        return { std::forward<T>(t) };
    }
};

}

} // util

#endif
