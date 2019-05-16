#include <cstddef>
#include <cstdint>
#include <cstring>

#include "util/safe_memory.h"

namespace util {

// This file must be compiled with -fnon-call-exceptions for the exceptions to work properly.

template<typename T>
T safe_read(void* pointer) {
    T ret;
    memcpy(&ret, pointer, sizeof(T));
    return ret;
}

template<typename T>
void safe_write(void *pointer, T value) {
    memcpy(pointer, &value, sizeof(T));
}

void safe_memcpy(void *dst, const void *src, size_t n) {
    std::byte *c_dst = reinterpret_cast<std::byte*>(dst);
    const std::byte *c_src = reinterpret_cast<const std::byte*>(src);
    for (size_t i = 0; i < n; i++, c_dst++, c_src++) {
        *c_dst = *c_src;
    }
}

void safe_memset(void *memory, int byte, size_t size) {
    unsigned char data = static_cast<unsigned char>(byte);
    unsigned char* pointer = reinterpret_cast<unsigned char*>(memory);
    for (unsigned char* end = pointer + size; pointer < end; pointer++) {
        *pointer = data;
    }
}

template uint8_t safe_read<uint8_t>(void*);
template uint16_t safe_read<uint16_t>(void*);
template uint32_t safe_read<uint32_t>(void*);
template uint64_t safe_read<uint64_t>(void*);
template void safe_write<uint8_t>(void*, uint8_t);
template void safe_write<uint16_t>(void*, uint16_t);
template void safe_write<uint32_t>(void*, uint32_t);
template void safe_write<uint64_t>(void*, uint64_t);

}
