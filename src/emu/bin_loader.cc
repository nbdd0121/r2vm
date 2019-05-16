#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

#include "emu/mmu.h"
#include "emu/state.h"

namespace emu {

struct Bin_file {
    int fd = -1;
    long file_size;
    std::byte *memory = nullptr;

    ~Bin_file();

    void load(const char* filename);
};

Bin_file::~Bin_file() {
    if (fd != -1) {
        close(fd);
    }

    if (memory == nullptr) {
        munmap(memory, file_size);
    }
}

void Bin_file::load(const char *filename) {

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

void load_bin(const char *filename, reg_t location) {

    Bin_file file;
    file.load(filename);
    copy_from_host(location, file.memory, file.file_size);
}

}
