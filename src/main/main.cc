#include <sys/auxv.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "emu/mmu.h"
#include "emu/state.h"
#include "riscv/context.h"

extern "C" {
    extern char **environ;
}

extern "C" void rs_main(int argc, const char **argv);

extern "C" void setup_mem(riscv::Context& context, void* loader, int argc, const char **argv) {
    if (emu::state::get_flags().user_only) {
        // Set sp to be the highest possible address.
        emu::reg_t sp = 0x7fff0000;
        emu::guest_mmap(sp - 0x800000, 0x800000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);

        // This contains (guest) pointers to all argument strings annd environment variables.
        std::vector<emu::reg_t> env_pointers;
        std::vector<emu::reg_t> arg_pointers(argc);

        // Copy all environment variables into guest user space.
        for (char** env = environ; *env; env++) {
            size_t env_length = strlen(*env) + 1;

            // Allocate memory from stack and copy to that region.
            sp -= env_length;
            memcpy((void*)sp, *env, env_length);
            env_pointers.push_back(sp);
        }

        // Copy all arguments into guest user space.
        for (int i = argc - 1; i >= 0; i--) {
            size_t arg_length = strlen(argv[i]) + 1;

            // Allocate memory from stack and copy to that region.
            sp -= arg_length;
            memcpy((void*)sp, argv[i], arg_length);
            arg_pointers[i] = sp;
        }

        // Align the stack to 8-byte boundary.
        sp &= ~7;

        auto push = [&sp](emu::reg_t value) {
            sp -= sizeof(emu::reg_t);
            emu::store_memory<emu::reg_t>(sp, value);
        };

        // Random data
        {
            std::default_random_engine rd;
            push(rd());
            push(rd());
            push(rd());
            push(rd());
        }

        emu::reg_t random_data = sp;

        // Setup auxillary vectors.
        push(0);
        push(AT_NULL);

        // Initialize context, and set up ELF-specific auxillary vectors.
        context.pc = emu::load_elf(loader, sp);

        push(getuid());
        push(AT_UID);
        push(geteuid());
        push(AT_EUID);
        push(getgid());
        push(AT_GID);
        push(getegid());
        push(AT_EGID);
        push(0);
        push(AT_HWCAP);
        push(100);
        push(AT_CLKTCK);
        push(random_data);
        push(AT_RANDOM);

        // fill in environ, last is nullptr
        push(0);
        sp -= env_pointers.size() * sizeof(emu::reg_t);
        memcpy((void*)sp, env_pointers.data(), env_pointers.size() * sizeof(emu::reg_t));

        // fill in argv, last is nullptr
        push(0);
        sp -= arg_pointers.size() * sizeof(emu::reg_t);
        memcpy((void*)sp, arg_pointers.data(), arg_pointers.size() * sizeof(emu::reg_t));

        // set argc
        push(arg_pointers.size());

        // sp
        context.registers[2] = sp;
        // libc adds this value into exit hook, so we need to make sure it is zero.
        context.registers[10] = 0;
        context.prv = 0;
    } else {
        // Allocate a 1G memory for physical address, starting at 0x200000.
        emu::guest_mmap_nofail(
            0x200000, 0x40000000 - 0x200000,
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
        emu::reg_t size = emu::load_bin(loader, 0x200000);
        // emu::load_bin("dt", 0x200000 + size);

        // a0 is the current hartid
        context.registers[10] = 0;
        // a1 should be the device tree
        context.registers[11] = 0x200000 + size;
        context.pc = 0x200000;
        context.prv = 1;
    }
}

int main(int argc, const char **argv) {
    rs_main(argc, argv);
    return 0;
}
