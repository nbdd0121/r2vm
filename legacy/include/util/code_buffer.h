#ifndef UTIL_CODE_BUFFER_H
#define UTIL_CODE_BUFFER_H

#include <cstddef>
#include <vector>

namespace util {

namespace internal {

void* code_allocate(size_t element_size, size_t element_alignment, size_t count);
void code_deallocate(void* ptr, size_t element_size, size_t element_alignment, size_t count) noexcept;

}

template<typename T>
struct Code_allocator {
    typedef T value_type;

    Code_allocator() = default;

    // Allow rebind
    template <class U> constexpr Code_allocator(const Code_allocator<U>&) noexcept {}

    T* allocate(std::size_t count) {
        return reinterpret_cast<T*>(internal::code_allocate(sizeof(T), alignof(T), count));
    }

    void deallocate(T* ptr, std::size_t count) noexcept {
        internal::code_deallocate(ptr, sizeof(T), alignof(T), count);
    }
};

template <class T, class U>
bool operator==(const Code_allocator<T>&, const Code_allocator<U>&) { return true; }

template <class T, class U>
bool operator!=(const Code_allocator<T>&, const Code_allocator<U>&) { return false; }

class Code_buffer: public std::vector<std::byte, Code_allocator<std::byte>> {
public:
    using vector::vector;
};

}

#endif
