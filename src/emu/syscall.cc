#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <vector>

#include "emu/mmu.h"
#include "emu/state.h"
#include "riscv/abi.h"
#include "util/format.h"

namespace emu::state {

reg_t original_brk;
reg_t brk;
reg_t heap_start;
reg_t heap_end;

}

namespace {

// Format for nullable pointers.
struct Pointer_formatter {
    emu::reg_t value;
};

Pointer_formatter pointer(emu::reg_t value) {
    return { value };
}

std::ostream& operator <<(std::ostream& stream, Pointer_formatter formatter) {
    if (formatter.value) {
        stream << std::hex << std::showbase << formatter.value;
    } else {
        stream << "NULL";
    }
    return stream;
}

/* Converters between guest and host data enums structure */

extern "C" int convert_mmap_prot_to_host(typename riscv::abi::int_t prot);
extern "C" int convert_mmap_flags_to_host(typename riscv::abi::int_t flags);
extern "C" emu::sreg_t return_errno(emu::sreg_t val);

}

namespace emu {

extern "C"
reg_t legacy_syscall(
    riscv::abi::Syscall_number nr,
    reg_t arg0, reg_t arg1, reg_t arg2, reg_t arg3, reg_t arg4, reg_t arg5
) {
    switch (nr) {
        case riscv::abi::Syscall_number::brk: {
            if (arg0 < state::original_brk) {
                // Cannot reduce beyond original_brk
            } else if (arg0 <= state::heap_end) {
                if (arg0 > state::brk) {
                    memset((void*)state::brk, 0, arg0 - state::brk);
                }
                state::brk = arg0;
            } else {
                reg_t new_heap_end = std::max(state::heap_start, (arg0 + page_mask) &~ page_mask);

                // The heap needs to be expanded
                reg_t addr = guest_mmap(
                    state::heap_end, new_heap_end - state::heap_end,
                    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0
                );

                if (addr != state::heap_end) {
                    // We failed to expand the brk.
                    guest_munmap(addr, new_heap_end - state::heap_end);

                } else {

                    // Memory should be zeroed here as this is expected by glibc.
                    memset((void*)state::brk, 0, state::heap_end - state::brk);
                    state::heap_end = new_heap_end;
                    state::brk = arg0;
                }
            }

            reg_t ret = state::brk;
            if (state::get_flags().strace) {
                util::log("brk({}) = {}\n", pointer(arg0), pointer(ret));
            }
            return ret;
        }
        case riscv::abi::Syscall_number::munmap: {
            reg_t ret = return_errno(guest_munmap(arg0, arg1));
            if (state::get_flags().strace) {
                util::error("munmap({:#x}, {}) = {}\n", arg0, arg1, ret);
            }

            return ret;
        }
        // This is linux specific call, we will just return ENOSYS.
        case riscv::abi::Syscall_number::mremap: {
            if (state::get_flags().strace) {
                util::error("mremap({:#x}, {}, {}, {}, {:#x}) = -ENOSYS\n", arg0, arg1, arg2, arg3, arg4);
            }
            return -static_cast<sreg_t>(riscv::abi::Errno::enosys);;
        }
        case riscv::abi::Syscall_number::mmap: {
            int prot = convert_mmap_prot_to_host(arg2);
            int flags = convert_mmap_flags_to_host(arg3);
            reg_t ret = reinterpret_cast<reg_t>(guest_mmap(arg0, arg1, prot, flags, arg4, arg5));
            if (state::get_flags().strace) {
                util::error("mmap({:#x}, {}, {}, {}, {}, {}) = {:#x}\n", arg0, arg1, arg2, arg3, arg4, arg5, ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::mprotect: {
            int prot = convert_mmap_prot_to_host(arg2);
            sreg_t ret = return_errno(guest_mprotect(arg0, arg1, prot));
            if (state::get_flags().strace) {
                util::error("mprotect({:#x}, {}, {}) = {:#x}\n", arg0, arg1, arg2, ret);
            }

            return ret;
        }
        default: {
            util::error("illegal syscall {}({}, {})\n", static_cast<int>(nr), arg0, arg1);
            return -static_cast<sreg_t>(riscv::abi::Errno::enosys);
        }
    }
}

}

