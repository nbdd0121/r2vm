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

// Formatter for escaped strings.
struct Escape_formatter {
    const char* pointer;
    size_t length;
};

Escape_formatter escape(const char *pointer) {
    return { pointer, strlen(pointer) };
}

std::ostream& operator <<(std::ostream& stream, Escape_formatter helper) {
    // State_saver saver {stream};

    const char *start = helper.pointer;
    const char *end = start + (helper.length > 64 ? 64 : helper.length);
    stream.put('"');

    // These are for escaped characters
    for (const char *pointer = helper.pointer; pointer != end; pointer++) {

        // Skip normal characters.
        if (*pointer != '"' && *pointer != '\\' && isprint(*pointer)) continue;

        // Print out all unprinted normal characters.
        if (pointer != start) stream.write(start, pointer - start);

        switch (*pointer) {
            case '"': stream << "\\\""; break;
            case '\\': stream << "\\\\"; break;
            case '\n': stream << "\\n"; break;
            case '\t': stream << "\\t"; break;
            default:
                stream << "\\x" << std::setfill('0') << std::setw(2) << std::hex
                       << static_cast<int>(static_cast<unsigned char>(*pointer));
                break;
        }

        start = pointer + 1;
    }

    if (end != start) stream.write(start, end - start);

    stream.put('"');
    if (helper.length > 64) stream << "...";

    return stream;
}

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

extern "C" int convert_open_flags_to_host(int flags);
extern "C" void convert_stat_from_host(riscv::abi::stat *guest_stat, struct stat *host_stat);
extern "C" int convert_mmap_prot_to_host(typename riscv::abi::int_t prot);
extern "C" int convert_mmap_flags_to_host(typename riscv::abi::int_t flags);
extern "C" emu::sreg_t return_errno(emu::sreg_t val);

// Detect whether the path is referencing /proc/self/.
// returns null if the path does not match /proc/self/, and return the remaining part if it matches.
const char* is_proc_self(const char* pathname) {
    if (strncmp(pathname, "/proc/", strlen("/proc/")) != 0) return nullptr;
    pathname += strlen("/proc/");
    if (strncmp(pathname, "self/", strlen("self/")) == 0) return pathname + strlen("self/");

    // We still need to check /proc/pid/
    char* end;
    long pid = strtol(pathname, &end, 10);

    // Not in form /proc/pid/
    if (end == pathname || end[0] != '/') return nullptr;

    // Not this process
    if (pid != getpid()) return nullptr;

    return end + 1;
}

std::string path_buffer;
const char* translate_path(const char* pathname) {
    if (pathname[0] != '/' || !*emu::state::get_flags().sysroot) return pathname;

    // The file exists in sysroot, then use it.
    path_buffer = std::string(emu::state::get_flags().sysroot) + pathname;
    if (access(path_buffer.c_str(), F_OK) == 0) {
        if (emu::state::get_flags().strace) {
            util::log("Translate {} to {}\n", pathname, path_buffer);
        }
        return path_buffer.c_str();
    }

    return pathname;
}

}

namespace emu {

extern "C"
reg_t legacy_syscall(
    riscv::abi::Syscall_number nr,
    reg_t arg0, reg_t arg1, reg_t arg2, reg_t arg3, reg_t arg4, reg_t arg5
) {
    switch (nr) {
        case riscv::abi::Syscall_number::unlinkat: {
            int dirfd = static_cast<sreg_t>(arg0) == riscv::abi::guest_AT_FDCWD ? AT_FDCWD : arg0;
            auto pathname = reinterpret_cast<char*>(arg1);
            sreg_t ret = return_errno(unlinkat(dirfd, translate_path(pathname), arg2));

            if (state::get_flags().strace) {
                util::log(
                    "unlinkat({}, {}, {}) = {}\n", static_cast<sreg_t>(arg0), escape(pathname), arg2, ret
                );
            }

            return ret;
        }
        case riscv::abi::Syscall_number::faccessat: {
            int dirfd = static_cast<sreg_t>(arg0) == riscv::abi::guest_AT_FDCWD ? AT_FDCWD : arg0;
            auto pathname = reinterpret_cast<char*>(arg1);
            sreg_t ret = return_errno(faccessat(dirfd, translate_path(pathname), arg2, arg3));

            if (state::get_flags().strace) {
                util::log(
                    "faccessat({}, {}, {}, {}) = {}\n", static_cast<sreg_t>(arg0), escape(pathname), arg2, arg3, ret
                );
            }

            return ret;
        }
        case riscv::abi::Syscall_number::openat: {
            int dirfd = static_cast<sreg_t>(arg0) == riscv::abi::guest_AT_FDCWD ? AT_FDCWD : arg0;
            auto pathname = reinterpret_cast<char*>(arg1);
            auto flags = convert_open_flags_to_host(arg2);
            auto proc_self = is_proc_self(pathname);
            sreg_t ret;
            if (proc_self != nullptr) {
                if (strcmp(proc_self, "exe") == 0) {
                    ret = return_errno(openat(dirfd, state::get_flags().exec_path, flags, arg3));
                } else {
                    // Also handle cmdline, stat, auxv, cmdline here!"
                    ret = return_errno(openat(dirfd, translate_path(pathname), flags, arg3));
                }
            } else {
                ret = return_errno(openat(dirfd, translate_path(pathname), flags, arg3));
            }

            if (state::get_flags().strace) {
                util::log(
                    "openat({}, {}, {}, {}) = {}\n", static_cast<sreg_t>(arg0), escape(pathname), arg2, arg3, ret
                );
            }

            return ret;
        }
        case riscv::abi::Syscall_number::readlinkat: {
            int dirfd = static_cast<sreg_t>(arg0) == riscv::abi::guest_AT_FDCWD ? AT_FDCWD : arg0;
            auto pathname = reinterpret_cast<char*>(arg1);
            auto buffer = reinterpret_cast<char*>(arg2);
            auto proc_self = is_proc_self(pathname);
            sreg_t ret;
            if (proc_self != nullptr && strcmp(proc_self, "exe") == 0) {
                char* path = realpath(state::get_flags().exec_path, NULL);
                if (path != nullptr) {
                    strncpy(buffer, path, arg3);
                    ret = strlen(path);
                    free(path);
                } else {
                    ret = return_errno(-1);
                }
            } else {
                ret = return_errno(readlinkat(dirfd, translate_path(pathname), buffer, arg3));
            }

            if (state::get_flags().strace) {
                if (ret > 0) {
                    util::log(
                        "readlinkat({}, {}, {}, {}) = {}\n",
                        static_cast<sreg_t>(arg0), escape(pathname), escape(buffer), arg3, ret
                    );
                } else {
                    util::log("readlinkat({}, {}, {:#x}, {}) = {}\n", arg0, escape(pathname), arg2, arg3, ret);
                }
            }

            return ret;
        }
        case riscv::abi::Syscall_number::fstatat: {
            int dirfd = static_cast<sreg_t>(arg0) == riscv::abi::guest_AT_FDCWD ? AT_FDCWD : arg0;
            auto pathname = reinterpret_cast<char*>(arg1);

            struct stat host_stat;
            sreg_t ret = return_errno(fstatat(dirfd, translate_path(pathname), &host_stat, arg3));

            // When success, convert stat format to guest format.
            if (ret == 0) {
                struct riscv::abi::stat *guest_stat = reinterpret_cast<riscv::abi::stat*>(arg2);
                convert_stat_from_host(guest_stat, &host_stat);
            }

            if (state::get_flags().strace) {
                if (ret == 0) {
                    util::log(
                        "fstatat({}, {}, {{st_mode={:#o}, st_size={}, ...}}) = 0\n",
                        static_cast<sreg_t>(arg0), escape(pathname), host_stat.st_mode, host_stat.st_size, arg3
                    );
                } else {
                    util::log("fstatat({}, {}, {:#x}, {}) = {}\n", arg0, escape(pathname), arg2, arg3, ret);
                }
            }

            return ret;
        }
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
        case riscv::abi::Syscall_number::open: {
            auto pathname = reinterpret_cast<char*>(arg0);
            auto flags = convert_open_flags_to_host(arg1);

            sreg_t ret = return_errno(open(translate_path(pathname), flags, arg2));
            if (state::get_flags().strace) {
                util::log("open({}, {}, {}) = {}\n", escape(pathname), arg1, arg2, ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::unlink: {
            auto pathname = reinterpret_cast<char*>(arg0);
            sreg_t ret = return_errno(unlink(translate_path(pathname)));
            if (state::get_flags().strace) {
                util::log("unlink({}) = {}\n", escape(pathname), ret);
            }
            return ret;
        }
        case riscv::abi::Syscall_number::stat: {
            auto pathname = reinterpret_cast<char*>(arg0);

            struct stat host_stat;
            sreg_t ret = return_errno(stat(translate_path(pathname), &host_stat));

            // When success, convert stat format to guest format.
            if (ret == 0) {
                struct riscv::abi::stat *guest_stat = reinterpret_cast<riscv::abi::stat*>(arg1);
                convert_stat_from_host(guest_stat, &host_stat);
            }

            if (state::get_flags().strace) {
                if (ret == 0) {
                    util::log("stat({}, {{st_mode={:#o}, st_size={}, ...}}) = 0\n", pathname, host_stat.st_mode, host_stat.st_size);
                } else {
                    util::log("stat({}, {:#x}) = {}\n", pathname, arg1, ret);
                }
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

