#include <sys/mman.h>

#include "util/assert.h"
#include "util/code_buffer.h"
#include "util/format.h"

namespace util::internal {

void* code_allocate(size_t element_size, size_t element_alignment, size_t count) {

    // In a generic allocator this should throw bad_alloc. But in our case we don't really need bad_alloc.
    // If these ever happen, it will be a programming error.
    ASSERT(count <= static_cast<size_t>(-1) / element_size);
    ASSERT(4096 % element_alignment == 0);

    size_t size = (element_size * count + 4095) &~ 4095;

    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (!addr) {
        throw std::bad_alloc{};
    }

    return addr;
}

void code_deallocate(void *ptr, size_t element_size, size_t element_alignment, size_t count) noexcept {

    ASSERT(count <= static_cast<size_t>(-1) / element_size);
    ASSERT(4096 % element_alignment == 0);

    if (!ptr) return;
    size_t size = (element_size * count + 4095) &~ 4095;
    munmap(ptr, size);
}

}
