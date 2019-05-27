#ifndef UTIL_MEMORY_H
#define UTIL_MEMORY_H

#include <cstring>

namespace util {

// Accessing a byte array as another type will violate the strict aliasing rule. Considering that operation is very
// common, read_as and write_as are implemented to maximise standard conformance.

template<typename T>
T read_as(const void *pointer) {
    T ret;
    memcpy(&ret, pointer, sizeof(T));
    return ret;
}

template<typename T>
void write_as(void *pointer, T value) {
    memcpy(pointer, &value, sizeof(T));
}

} // util

#endif
