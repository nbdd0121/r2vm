#ifndef UTIL_SAFE_MEMORY_H
#define UTIL_SAFE_MEMORY_H

#include <cstddef>

namespace util {

template<typename T>
T safe_read(void *pointer);

template<typename T>
void safe_write(void *pointer, T value);

void safe_memcpy(void *dst, const void *src, size_t size);

void safe_memset(void *memory, int byte, size_t size);

} // util

#endif
