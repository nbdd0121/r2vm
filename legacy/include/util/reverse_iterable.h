#ifndef UTIL_REVERSE_ITERABLE_H
#define UTIL_REVERSE_ITERABLE_H

#include <iterator>

namespace util {

template<typename T>
struct reversion_wrapper { T& iterable; };

template<typename T>
auto begin(reversion_wrapper<T> w) {
    using std::rbegin;
    return rbegin(w.iterable);
}

template<typename T>
auto end(reversion_wrapper<T> w) {
    using std::rend;
    return rend(w.iterable);
}

template<typename T>
reversion_wrapper<T> reverse_iterable(T&& iterable) { return { iterable }; }

} // util

#endif
