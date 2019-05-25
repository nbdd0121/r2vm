#ifndef EMU_MMU_H
#define EMU_MMU_H

#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "config.h"
#include "emu/typedef.h"

extern "C" uint32_t phys_read(uint64_t, uint32_t);
extern "C" void phys_write(uint64_t, uint64_t, uint32_t);

namespace emu {

static constexpr reg_t page_size = 0x1000;
static constexpr reg_t page_mask = page_size - 1;
static constexpr reg_t log_page_size = 12;

reg_t guest_mmap(reg_t address, reg_t size, int prot, int flags, int fd, reg_t offset);
reg_t guest_mmap_nofail(reg_t address, reg_t size, int prot, int flags, int fd, reg_t offset);
int guest_mprotect(reg_t address, reg_t size, int prot);
int guest_munmap(reg_t address, reg_t size);

template<typename T>
inline T load_memory(reg_t address) {
    return (T)phys_read(address, sizeof(T));
}

template<typename T>
inline void store_memory(reg_t address, T value) {
    phys_write(address, value, sizeof(T));
}

}

#endif
