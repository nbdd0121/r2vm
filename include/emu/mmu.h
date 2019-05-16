#ifndef EMU_MMU_H
#define EMU_MMU_H

#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "config.h"
#include "emu/typedef.h"
#include "util/assert.h"
#include "util/safe_memory.h"

namespace emu {

static constexpr reg_t page_size = 0x1000;
static constexpr reg_t page_mask = page_size - 1;
static constexpr reg_t log_page_size = 12;

inline std::byte* translate_address(reg_t address) {
    return reinterpret_cast<std::byte*>(address);
}

reg_t guest_mmap(reg_t address, reg_t size, int prot, int flags, int fd, reg_t offset);
reg_t guest_mmap_nofail(reg_t address, reg_t size, int prot, int flags, int fd, reg_t offset);
int guest_mprotect(reg_t address, reg_t size, int prot);
int guest_munmap(reg_t address, reg_t size);

template<typename T>
inline T load_memory(reg_t address) {
    return util::safe_read<T>(translate_address(address));
}

template<typename T>
inline void store_memory(reg_t address, T value) {
    util::safe_write<T>(translate_address(address), value);
}

inline void copy_from_host(reg_t address, const void* target, size_t size) {
    util::safe_memcpy(translate_address(address), target, size);
}

inline void copy_to_host(reg_t address, void* target, size_t size) {
    util::safe_memcpy(target, translate_address(address), size);
}

inline void zero_memory(reg_t address, size_t size) {
    util::safe_memset(translate_address(address), 0, size);
}

}

#endif
