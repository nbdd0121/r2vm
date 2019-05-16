#include <elf.h>
#include <fcntl.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

#include "emu/mmu.h"
#include "emu/state.h"
#include "util/scope_exit.h"

#define EM_RISCV 243

namespace emu {

struct Elf_file {
    int fd = -1;
    long file_size;
    std::byte *memory = nullptr;

    ~Elf_file();

    void load(const char* filename);
    void validate();
    std::string find_interpreter();
};

Elf_file::~Elf_file() {
    if (fd != -1) {
        close(fd);
    }

    if (memory == nullptr) {
        munmap(memory, file_size);
    }
}

void Elf_file::load(const char *filename) {

    // Similar to sysroot lookup in syscall.cc, prioritize sysroot directory.
    std::string sysroot_path = state::sysroot + filename;
    if (filename[0] == '/' && access(sysroot_path.c_str(), F_OK) == 0) {
        fd = open(sysroot_path.c_str(), O_RDONLY);

    } else {
        fd = open(filename, O_RDONLY);
    }

    if (fd == -1) {
        throw std::runtime_error { "cannot open file" };
    }

    // Get the size of the file.
    {
        struct stat s;
        int status = fstat(fd, &s);
        if (status == -1) {
            throw std::runtime_error { "cannot fstat file" };
        }

        file_size = s.st_size;
    }

    // Map the file to memory.
    memory = reinterpret_cast<std::byte*>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (memory == nullptr) {
        throw std::runtime_error { "cannot mmap file" };
    }
}

void Elf_file::validate() {
    Elf64_Ehdr *header = reinterpret_cast<Elf64_Ehdr*>(memory);

    // Check the ELF magic numbers
    if (memcmp(header->e_ident, "\177ELF", 4) != 0) {
        throw std::runtime_error { "the program to be loaded is not elf." };
    }

    // We can only proceed with executable or dynamic binary.
    if (header->e_type != ET_EXEC && header->e_type != ET_DYN) {
        throw std::runtime_error { "the binary is not executable." };
    }

    // Check that the ELF is for RISC-V
    if (header->e_machine != EM_RISCV) {
        throw std::runtime_error { "the binary is not for RISC-V." };
    }

    // TODO: Also check for flags about ISA extensions
}

std::string Elf_file::find_interpreter() {
    Elf64_Ehdr *header = reinterpret_cast<Elf64_Ehdr*>(memory);
    for (int i = 0; i < header->e_phnum; i++) {
        Elf64_Phdr *h = reinterpret_cast<Elf64_Phdr*>(memory + header->e_phoff + header->e_phentsize * i);
        if (h->p_type == PT_INTERP) {
            if (*reinterpret_cast<char*>(memory + h->p_offset + h->p_filesz - 1) != 0) {
                throw std::runtime_error { "interpreter name should be null-terminated." };
            }

            return reinterpret_cast<char*>(memory + h->p_offset);
        }
    }

    return {};
}

reg_t load_elf_image(Elf_file& file, reg_t& load_addr, reg_t& brk) {

    // Parse the ELF header and load the binary into memory.
    auto memory = file.memory;
    Elf64_Ehdr *header = reinterpret_cast<Elf64_Ehdr*>(memory);

    // Scan the bounds of the image.
    reg_t loaddr = -1;
    reg_t hiaddr = 0;
    for (int i = 0; i < header->e_phnum; i++) {
        Elf64_Phdr *h = reinterpret_cast<Elf64_Phdr*>(memory + header->e_phoff + header->e_phentsize * i);
        if (h->p_type == PT_LOAD) {
            if (h->p_vaddr < loaddr) loaddr = h->p_vaddr;
            if (h->p_vaddr + h->p_memsz > hiaddr) hiaddr = h->p_vaddr + h->p_memsz;
        }
    }

    loaddr &=~ page_mask;
    hiaddr = (hiaddr + page_mask) &~ page_mask;

    reg_t bias = 0;
    // For dynamic binaries, we need to allocate a location for it.
    if (header->e_type == ET_DYN) {
        bias = guest_mmap_nofail(
            0,
            hiaddr - loaddr,
            PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0
        ) - loaddr;

    } else {
        auto address = guest_mmap_nofail(
            loaddr,
            hiaddr - loaddr,
            PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0
        );

        if (address != loaddr) {
            guest_munmap(address, hiaddr - loaddr);
            throw std::bad_alloc{};
        }
    }

    brk = 0;
    for (int i = 0; i < header->e_phnum; i++) {
        Elf64_Phdr *h = reinterpret_cast<Elf64_Phdr*>(memory + header->e_phoff + header->e_phentsize * i);
        if (h->p_type == PT_LOAD) {

            // size in memory cannot be smaller than size in file
            if (h->p_filesz > h->p_memsz) {
                throw std::runtime_error { "invalid elf file: constraint p_filesz <= p_memsz is not satisified" };
            }

            reg_t vaddr_map_end = h->p_vaddr + h->p_filesz;
            reg_t vaddr_end = h->p_vaddr + h->p_memsz;
            reg_t file_offset = h->p_offset;
            reg_t page_start = h->p_vaddr &~ page_mask;
            reg_t map_end = vaddr_map_end &~ page_mask;
            reg_t page_end = (vaddr_end + page_mask) &~ page_mask;

            int prot = 0;
            if (h->p_flags & PF_R) prot |= PROT_READ;
            if (h->p_flags & PF_W) prot |= PROT_WRITE;
            if (h->p_flags & PF_X) prot |= PROT_EXEC;

            // First page is not aligned, we need t adjust it so that it is aligned.
            if (h->p_vaddr != page_start) {
                file_offset -= h->p_vaddr - page_start;
            }

            // Map until map_end.
            if (map_end != page_start) {
                guest_mmap_nofail(
                    bias + page_start,
                    map_end - page_start,
                    prot, MAP_PRIVATE | MAP_FIXED, file.fd, file_offset
                );
            }

            // For those beyond map_end, we need those beyond filesz to be zero. As we know the memory location will
            // initially be zero, we just make them writable and copy the remaining part across.
            if (map_end != page_end) {

                // Make it writable.
                guest_mprotect(bias + map_end, page_end - map_end, PROT_READ | PROT_WRITE);

                // Copy across.
                copy_from_host(bias + map_end, memory + file_offset + (map_end - page_start), vaddr_map_end - map_end);

                if (prot != (PROT_READ | PROT_WRITE)) {
                    guest_mprotect(bias + map_end, page_end - map_end, prot);
                }
            }

            // Set brk to the address past the last program segment.
            if (vaddr_end > brk) {
                brk = vaddr_end;
            }
        }
    }

    // Return information needed by the caller.
    load_addr = bias + loaddr;
    brk = bias + ((brk + page_mask) &~ page_mask);
    return bias + header->e_entry;
}

reg_t load_elf(const char *filename, reg_t& sp) {

    Elf_file file;
    file.load(filename);
    file.validate();
    auto header = reinterpret_cast<Elf64_Ehdr*>(file.memory);

    reg_t load_addr;
    reg_t interp_addr = 0;
    reg_t brk;
    reg_t entry = load_elf_image(file, load_addr, brk);
    reg_t actual_entry = entry;

    // If an interpreter exists, load it as well.
    std::string interpreter = file.find_interpreter();
    if (!interpreter.empty()) {
        Elf_file interp_file;
        interp_file.load(interpreter.c_str());
        interp_file.validate();

        if (!interp_file.find_interpreter().empty()) {
            throw std::runtime_error { "interpreter cannot require other interpreters" };
        }

        reg_t interp_brk;
        actual_entry = load_elf_image(interp_file, interp_addr, interp_brk);
    }

    // Setup brk.
    brk = (brk + page_mask) &~ page_mask;
    state::original_brk = brk;
    state::brk = brk;
    state::heap_start = brk;
    state::heap_end = state::heap_start;

    auto push = [&sp](reg_t value) {
        sp -= sizeof(reg_t);
        store_memory<reg_t>(sp, value);
    };

    // Setup auxillary vectors.
    push(load_addr + header->e_phoff);
    push(AT_PHDR);
    push(header->e_phentsize);
    push(AT_PHENT);
    push(header->e_phnum);
    push(AT_PHNUM);
    push(page_size);
    push(AT_PAGESZ);
    push(interp_addr);
    push(AT_BASE);
    push(0);
    push(AT_FLAGS);
    push(entry);
    push(AT_ENTRY);

    return actual_entry;
}

}
