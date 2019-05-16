#include <cstring>
#include <sys/mman.h>

#include "emu/mmu.h"
#include "util/memory.h"

namespace emu {

// Establish a mapping for guest.
reg_t guest_mmap(reg_t address, reg_t size, int prot, int flags, int fd, reg_t offset) {

    // For PROT_EXEC request we translate it into PROT_READ, as we need to interpret it.
    if (prot & PROT_EXEC) {
        prot &=~ PROT_EXEC;
        prot |= PROT_READ;
    }

    return reinterpret_cast<reg_t>(mmap(translate_address(address), size, prot, flags, fd, offset));
}

reg_t guest_mmap_nofail(reg_t address, reg_t size, int prot, int flags, int fd, reg_t offset) {
    reg_t ret = guest_mmap(address, size, prot, flags, fd, offset);
    if (ret == static_cast<reg_t>(-1)) throw std::bad_alloc{};
    return ret;
}

int guest_mprotect(reg_t address, reg_t size, int prot) {

    // Translate protection flags.
    if (prot & PROT_EXEC) {
        prot &=~ PROT_EXEC;
        prot |= PROT_READ;
    }

    return mprotect(translate_address(address), size, prot);
}

int guest_munmap(reg_t address, reg_t size) {
    return munmap(translate_address(address), size);
}

}
