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

namespace {

// Formatter for escaped strings.
struct Escape_formatter {
    const char* pointer;
    size_t length;
};

Escape_formatter escape(const char *pointer, size_t length) {
    return { pointer, length };
}

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

// Only translate POSIX specified subset of error numbers should be sufficient.
riscv::abi::Errno convert_errno_from_host(int number) {
    switch (number) {
        case E2BIG              : return riscv::abi::Errno::e2big;
        case EACCES             : return riscv::abi::Errno::eacces;
        case EADDRINUSE         : return riscv::abi::Errno::eaddrinuse;
        case EADDRNOTAVAIL      : return riscv::abi::Errno::eaddrnotavail;
        case EAFNOSUPPORT       : return riscv::abi::Errno::eafnosupport;
        case EAGAIN             : return riscv::abi::Errno::eagain;
        case EALREADY           : return riscv::abi::Errno::ealready;
        case EBADF              : return riscv::abi::Errno::ebadf;
        case EBADMSG            : return riscv::abi::Errno::ebadmsg;
        case EBUSY              : return riscv::abi::Errno::ebusy;
        case ECANCELED          : return riscv::abi::Errno::ecanceled;
        case ECHILD             : return riscv::abi::Errno::echild;
        case ECONNABORTED       : return riscv::abi::Errno::econnaborted;
        case ECONNREFUSED       : return riscv::abi::Errno::econnrefused;
        case ECONNRESET         : return riscv::abi::Errno::econnreset;
        case EDEADLK            : return riscv::abi::Errno::edeadlk;
        case EDESTADDRREQ       : return riscv::abi::Errno::edestaddrreq;
        case EDOM               : return riscv::abi::Errno::edom;
        case EDQUOT             : return riscv::abi::Errno::edquot;
        case EEXIST             : return riscv::abi::Errno::eexist;
        case EFAULT             : return riscv::abi::Errno::efault;
        case EFBIG              : return riscv::abi::Errno::efbig;
        case EHOSTUNREACH       : return riscv::abi::Errno::ehostunreach;
        case EIDRM              : return riscv::abi::Errno::eidrm;
        case EILSEQ             : return riscv::abi::Errno::eilseq;
        case EINPROGRESS        : return riscv::abi::Errno::einprogress;
        case EINTR              : return riscv::abi::Errno::eintr;
        case EINVAL             : return riscv::abi::Errno::einval;
        case EIO                : return riscv::abi::Errno::eio;
        case EISCONN            : return riscv::abi::Errno::eisconn;
        case EISDIR             : return riscv::abi::Errno::eisdir;
        case ELOOP              : return riscv::abi::Errno::eloop;
        case EMFILE             : return riscv::abi::Errno::emfile;
        case EMLINK             : return riscv::abi::Errno::emlink;
        case EMSGSIZE           : return riscv::abi::Errno::emsgsize;
        case EMULTIHOP          : return riscv::abi::Errno::emultihop;
        case ENAMETOOLONG       : return riscv::abi::Errno::enametoolong;
        case ENETDOWN           : return riscv::abi::Errno::enetdown;
        case ENETRESET          : return riscv::abi::Errno::enetreset;
        case ENETUNREACH        : return riscv::abi::Errno::enetunreach;
        case ENFILE             : return riscv::abi::Errno::enfile;
        case ENOBUFS            : return riscv::abi::Errno::enobufs;
        case ENODEV             : return riscv::abi::Errno::enodev;
        case ENOENT             : return riscv::abi::Errno::enoent;
        case ENOEXEC            : return riscv::abi::Errno::enoexec;
        case ENOLCK             : return riscv::abi::Errno::enolck;
        case ENOLINK            : return riscv::abi::Errno::enolink;
        case ENOMEM             : return riscv::abi::Errno::enomem;
        case ENOMSG             : return riscv::abi::Errno::enomsg;
        case ENOPROTOOPT        : return riscv::abi::Errno::enoprotoopt;
        case ENOSPC             : return riscv::abi::Errno::enospc;
        case ENOSYS             : return riscv::abi::Errno::enosys;
        case ENOTCONN           : return riscv::abi::Errno::enotconn;
        case ENOTDIR            : return riscv::abi::Errno::enotdir;
        case ENOTEMPTY          : return riscv::abi::Errno::enotempty;
        case ENOTRECOVERABLE    : return riscv::abi::Errno::enotrecoverable;
        case ENOTSOCK           : return riscv::abi::Errno::enotsock;
#if ENOTSUP != EOPNOTSUPP
        case ENOTSUP            : return riscv::abi::Errno::enotsup;
#endif
        case ENOTTY             : return riscv::abi::Errno::enotty;
        case ENXIO              : return riscv::abi::Errno::enxio;
        case EOPNOTSUPP         : return riscv::abi::Errno::eopnotsupp;
        case EOVERFLOW          : return riscv::abi::Errno::eoverflow;
        case EOWNERDEAD         : return riscv::abi::Errno::eownerdead;
        case EPERM              : return riscv::abi::Errno::eperm;
        case EPIPE              : return riscv::abi::Errno::epipe;
        case EPROTO             : return riscv::abi::Errno::eproto;
        case EPROTONOSUPPORT    : return riscv::abi::Errno::eprotonosupport;
        case EPROTOTYPE         : return riscv::abi::Errno::eprototype;
        case ERANGE             : return riscv::abi::Errno::erange;
        case EROFS              : return riscv::abi::Errno::erofs;
        case ESPIPE             : return riscv::abi::Errno::espipe;
        case ESRCH              : return riscv::abi::Errno::esrch;
        case ESTALE             : return riscv::abi::Errno::estale;
        case ETIMEDOUT          : return riscv::abi::Errno::etimedout;
        case ETXTBSY            : return riscv::abi::Errno::etxtbsy;
#if EWOULDBLOCK != EAGAIN
        case EWOULDBLOCK        : return riscv::abi::Errno::ewouldblock;
#endif
        case EXDEV              : return riscv::abi::Errno::exdev;
        default:
            util::log("Fail to translate host errno = {} to guest errno\n", number);
            return riscv::abi::Errno::enosys;
    }
}

int convert_open_flags_to_host(int flags) {
    int ret = 0;
    if (flags & 01) ret |= O_WRONLY;
    if (flags & 02) ret |= O_RDWR;
    if (flags & 0100) ret |= O_CREAT;
    if (flags & 0200) ret |= O_EXCL;
    if (flags & 01000) ret |= O_TRUNC;
    if (flags & 02000) ret |= O_APPEND;
    if (flags & 04000) ret |= O_NONBLOCK;
    if (flags & 04010000) ret |= O_SYNC;
    return ret;
}

void convert_stat_from_host(riscv::abi::stat *guest_stat, struct stat *host_stat) {
    guest_stat->st_dev          = host_stat->st_dev;
    guest_stat->st_ino          = host_stat->st_ino;
    guest_stat->st_mode         = host_stat->st_mode;
    guest_stat->st_nlink        = host_stat->st_nlink;
    guest_stat->st_uid          = host_stat->st_uid;
    guest_stat->st_gid          = host_stat->st_gid;
    guest_stat->st_rdev         = host_stat->st_rdev;
    guest_stat->st_size         = host_stat->st_size;
    guest_stat->st_blocks       = host_stat->st_blocks;
    guest_stat->st_blksize      = host_stat->st_blksize;
    guest_stat->guest_st_atime  = host_stat->st_atime;
    guest_stat->st_atime_nsec   = host_stat->st_atim.tv_nsec;
    guest_stat->guest_st_mtime  = host_stat->st_mtim.tv_sec;
    guest_stat->st_mtime_nsec   = host_stat->st_mtim.tv_nsec;
    guest_stat->guest_st_ctime  = host_stat->st_ctim.tv_sec;
    guest_stat->st_ctime_nsec   = host_stat->st_ctim.tv_nsec;
}

void convert_timeval_from_host(riscv::abi::timeval *guest_tv, struct timeval *host_tv) {
    guest_tv->tv_sec   = host_tv->tv_sec;
    guest_tv->tv_usec  = host_tv->tv_usec;
}

template<typename Abi>
constexpr bool need_iovec_conversion() {
    return sizeof(struct iovec) != sizeof(typename Abi::iovec) ||
           alignof(struct iovec) != alignof(typename Abi::iovec) ||
           offsetof(struct iovec, iov_base) != offsetof(typename Abi::iovec, iov_base) ||
           offsetof(struct iovec, iov_len) != offsetof(typename Abi::iovec, iov_len);
}

template<typename Abi>
void convert_iovec_to_host(struct iovec *host_iov, const typename Abi::iovec* guest_iov) {
    host_iov->iov_base = emu::translate_address(guest_iov->iov_base);
    host_iov->iov_len = guest_iov->iov_len;
}

template<typename Abi>
constexpr bool need_utsname_conversion() {

    // Assume all fields have the same length, so if the total size is equal, no conversion would be needed.
    return sizeof(struct utsname) != sizeof(typename Abi::utsname) ||
           offsetof(struct iovec, iov_base) != offsetof(typename Abi::iovec, iov_base) ||
           offsetof(struct iovec, iov_len) != offsetof(typename Abi::iovec, iov_len);
}

template<typename Abi>
void convert_utsname_from_host(typename Abi::utsname *guest_utsname, const struct utsname *host_utsname) {
    strncpy(guest_utsname->sysname, host_utsname->sysname, Abi::guest_UTSNAME_LENGTH - 1);
    strncpy(guest_utsname->nodename, host_utsname->nodename, Abi::guest_UTSNAME_LENGTH - 1);
    strncpy(guest_utsname->release, host_utsname->release, Abi::guest_UTSNAME_LENGTH - 1);
    strncpy(guest_utsname->version, host_utsname->version, Abi::guest_UTSNAME_LENGTH - 1);
    strncpy(guest_utsname->machine, host_utsname->machine, Abi::guest_UTSNAME_LENGTH - 1);
#ifdef _GNU_SOURCE
    strncpy(guest_utsname->domainname, host_utsname->domainname, Abi::guest_UTSNAME_LENGTH - 1);
#endif
}

// When an error occurs during a system call, Linux will return the negated value of the error number. Library
// functions, on the other hand, usually return -1 and set errno instead.
// Helper for converting library functions which use state variable `errno` to carry error information to a linux
// syscall style which returns a negative value representing the errno.
emu::sreg_t return_errno(emu::sreg_t val) {
    if (val != -1) return val;
    return -static_cast<emu::sreg_t>(convert_errno_from_host(errno));
}

template<typename Abi>
int convert_mmap_prot_from_host(typename Abi::int_t prot) {
    int ret = 0;
    if (prot & Abi::guest_PROT_READ) ret |= PROT_READ;
    if (prot & Abi::guest_PROT_WRITE) ret |= PROT_WRITE;
    if (prot & Abi::guest_PROT_EXEC) ret |= PROT_EXEC;
    return ret;
}

template<typename Abi>
int convert_mmap_flags_from_host(typename Abi::int_t flags) {
    int ret = 0;
    if (flags & Abi::guest_MAP_SHARED) ret |= MAP_SHARED;
    if (flags & Abi::guest_MAP_PRIVATE) ret |= MAP_PRIVATE;
    if (flags & Abi::guest_MAP_FIXED) ret |= MAP_FIXED;
    if (flags & Abi::guest_MAP_ANON) ret |= MAP_ANON;
    return ret;
}

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
    if (pathname[0] != '/' || emu::state::sysroot.empty()) return pathname;

    // The file exists in sysroot, then use it.
    path_buffer = emu::state::sysroot + pathname;
    if (access(path_buffer.c_str(), F_OK) == 0) {
        if (emu::state::strace) {
            util::log("Translate {} to {}\n", pathname, path_buffer);
        }
        return path_buffer.c_str();
    }

    return pathname;
}

}

namespace emu {

reg_t syscall(
    riscv::abi::Syscall_number nr,
    reg_t arg0, reg_t arg1, reg_t arg2, [[maybe_unused]] reg_t arg3, [[maybe_unused]] reg_t arg4, [[maybe_unused]] reg_t arg5
) {
    using Abi = riscv::abi::Abi;

    switch (nr) {
        case riscv::abi::Syscall_number::getcwd: {
            char *buffer = reinterpret_cast<char*>(translate_address(arg0));
            size_t size = arg1;
            sreg_t ret = getcwd(buffer, size) ? 0 : -static_cast<sreg_t>(riscv::abi::Errno::einval);
            if (state::strace) {
                if (ret == 0) {
                    util::log("getcwd({}, {}) = 0\n", escape(buffer), size);
                } else {
                    util::log("getcwd({}, {}) = {}\n", buffer, size, ret);
                }
            }

            return ret;
        }
        case riscv::abi::Syscall_number::unlinkat: {
            int dirfd = static_cast<sreg_t>(arg0) == Abi::guest_AT_FDCWD ? AT_FDCWD : arg0;
            auto pathname = reinterpret_cast<char*>(translate_address(arg1));
            sreg_t ret = return_errno(unlinkat(dirfd, translate_path(pathname), arg2));

            if (state::strace) {
                util::log(
                    "unlinkat({}, {}, {}) = {}\n", static_cast<sreg_t>(arg0), escape(pathname), arg2, ret
                );
            }

            return ret;
        }
        case riscv::abi::Syscall_number::faccessat: {
            int dirfd = static_cast<sreg_t>(arg0) == Abi::guest_AT_FDCWD ? AT_FDCWD : arg0;
            auto pathname = reinterpret_cast<char*>(translate_address(arg1));
            sreg_t ret = return_errno(faccessat(dirfd, translate_path(pathname), arg2, arg3));

            if (state::strace) {
                util::log(
                    "faccessat({}, {}, {}, {}) = {}\n", static_cast<sreg_t>(arg0), escape(pathname), arg2, arg3, ret
                );
            }

            return ret;
        }
        case riscv::abi::Syscall_number::openat: {
            int dirfd = static_cast<sreg_t>(arg0) == Abi::guest_AT_FDCWD ? AT_FDCWD : arg0;
            auto pathname = reinterpret_cast<char*>(translate_address(arg1));
            auto flags = convert_open_flags_to_host(arg2);
            auto proc_self = is_proc_self(pathname);
            sreg_t ret;
            if (proc_self != nullptr) {
                if (strcmp(proc_self, "exe") == 0) {
                    ret = return_errno(openat(dirfd, state::exec_path.c_str(), flags, arg3));
                } else {
                    // Auto handle cmdline, stat, auxv, cmdline here!"
                    ret = return_errno(openat(dirfd, translate_path(pathname), flags, arg3));
                }
            } else {
                ret = return_errno(openat(dirfd, translate_path(pathname), flags, arg3));
            }

            if (state::strace) {
                util::log(
                    "openat({}, {}, {}, {}) = {}\n", static_cast<sreg_t>(arg0), escape(pathname), arg2, arg3, ret
                );
            }

            return ret;
        }
        case riscv::abi::Syscall_number::close: {
            // Handle standard IO specially, pretending close is sucessful.
            sreg_t ret;
            if (arg0 == 1 || arg0 == 2) {
                ret = 0;
            } else {
                ret = return_errno(close(arg0));
            }

            if (state::strace) {
                util::log("close({}) = {}\n", arg0, ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::lseek: {
            sreg_t ret = return_errno(lseek(arg0, arg1, arg2));
            if (state::strace) {
                util::log("lseek({}, {}, {}) = {}\n", arg0, arg1, arg2, ret);
            }
            return ret;
        }
        case riscv::abi::Syscall_number::read: {
            auto buffer = reinterpret_cast<char*>(translate_address(arg1));

            // Handle standard IO specially, since it is shared between emulator and guest program.
            sreg_t ret = return_errno(read(arg0, buffer, arg2));

            if (state::strace) {
                util::log("read({}, {}, {}) = {}\n",
                    arg0,
                    escape(buffer, arg2),
                    arg2,
                    ret
                );
            }

            return ret;
        }
        case riscv::abi::Syscall_number::write: {
            auto buffer = reinterpret_cast<const char*>(translate_address(arg1));

            sreg_t ret = return_errno(write(arg0, buffer, arg2));

            if (state::strace) {
                util::log("write({}, {}, {}) = {}\n",
                    arg0,
                    escape(buffer, arg2),
                    arg2,
                    ret
                );
            }

            return ret;
        }
        case riscv::abi::Syscall_number::writev: {
            sreg_t ret;
            if constexpr (need_iovec_conversion<Abi>()) {
                std::vector<struct iovec> host_iov(arg2);
                Abi::iovec *guest_iov = reinterpret_cast<Abi::iovec*>(translate_address(arg1));
                for (unsigned i = 0; i < arg2; i++) convert_iovec_to_host<Abi>(&host_iov[i], &guest_iov[i]);
                ret = return_errno(writev(arg0, host_iov.data(), arg2));

            } else {
                ret = return_errno(writev(arg0, reinterpret_cast<struct iovec*>(translate_address(arg1)), arg2));
            }

            if (state::strace) {
                util::log("writev({}, {}, {}) = {}\n",
                    arg0,
                    arg1,
                    arg2,
                    ret
                );
            }

            return ret;
        }
        case riscv::abi::Syscall_number::readlinkat: {
            int dirfd = static_cast<sreg_t>(arg0) == Abi::guest_AT_FDCWD ? AT_FDCWD : arg0;
            auto pathname = reinterpret_cast<char*>(translate_address(arg1));
            auto buffer = reinterpret_cast<char*>(translate_address(arg2));
            auto proc_self = is_proc_self(pathname);
            sreg_t ret;
            if (proc_self != nullptr && strcmp(proc_self, "exe") == 0) {
                char* path = realpath(state::exec_path.c_str(), NULL);
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

            if (state::strace) {
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
            int dirfd = static_cast<sreg_t>(arg0) == Abi::guest_AT_FDCWD ? AT_FDCWD : arg0;
            auto pathname = reinterpret_cast<char*>(translate_address(arg1));

            struct stat host_stat;
            sreg_t ret = return_errno(fstatat(dirfd, translate_path(pathname), &host_stat, arg3));

            // When success, convert stat format to guest format.
            if (ret == 0) {
                struct riscv::abi::stat *guest_stat = reinterpret_cast<riscv::abi::stat*>(translate_address(arg2));
                convert_stat_from_host(guest_stat, &host_stat);
            }

            if (state::strace) {
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
        case riscv::abi::Syscall_number::fstat: {
            struct stat host_stat;
            sreg_t ret = return_errno(fstat(arg0, &host_stat));

            // When success, convert stat format to guest format.
            if (ret == 0) {
                struct riscv::abi::stat *guest_stat = reinterpret_cast<riscv::abi::stat*>(translate_address(arg1));
                convert_stat_from_host(guest_stat, &host_stat);
            }

            if (state::strace) {
                if (ret == 0) {
                    util::log("fstat({}, {{st_mode={:#o}, st_size={}, ...}}) = 0\n", arg0, host_stat.st_mode, host_stat.st_size);
                } else {
                    util::log("fstat({}, {:#x}) = {}\n", arg0, arg1, ret);
                }
            }

            return ret;
        }
        case riscv::abi::Syscall_number::exit: {
            if (state::strace) {
                util::log("exit({}) = ?\n", arg0);
            }

            // Record the exit_code so that the emulator can correctly return it.
            throw emu::Exit_control { static_cast<uint8_t>(arg0) };
        }
        case riscv::abi::Syscall_number::exit_group: {
            if (state::strace) {
                util::log("exit_group({}) = ?\n", arg0);
            }

            // Record the exit_code so that the emulator can correctly return it.
            throw emu::Exit_control { static_cast<uint8_t>(arg0) };
        }
        case riscv::abi::Syscall_number::uname: {
            sreg_t ret;
            if constexpr (need_utsname_conversion<Abi>()) {
                struct utsname host_utsname;
                ret = return_errno(uname(&host_utsname));
                convert_utsname_from_host<Abi>(
                    reinterpret_cast<Abi::utsname*>(translate_address(arg0)), &host_utsname
                );

            } else {
                ret = return_errno(uname(reinterpret_cast<struct utsname*>(translate_address(arg0))));
            }

            if (state::strace) {
                util::log("uname({:#x}) = {}\n", arg0, ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::gettimeofday: {
            struct timeval host_tv;

            // TODO: gettimeofday is obsolescent. Even if some applications require this syscall, we should try to work
            // around it instead of using the obsolescent function.
            sreg_t ret = return_errno(gettimeofday(&host_tv, nullptr));

            if (ret == 0) {
                struct riscv::abi::timeval *guest_tv = reinterpret_cast<riscv::abi::timeval*>(translate_address(arg0));
                convert_timeval_from_host(guest_tv, &host_tv);
            }

            if (state::strace) {
                if (ret == 0) {
                    util::log("gettimeofday({{{}, {}}}, NULL) = 0\n", host_tv.tv_sec, host_tv.tv_usec);
                } else {
                    util::log("gettimeofday({:#x}) = {}\n", arg0, ret);
                }
            }

            return ret;
        }
        case riscv::abi::Syscall_number::getpid: {
            reg_t ret = getpid();
            if (state::strace) {
                util::log("getpid() = {}\n", ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::getppid: {
            reg_t ret = getppid();
            if (state::strace) {
                util::log("getppid() = {}\n", ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::getuid: {
            reg_t ret = getuid();
            if (state::strace) {
                util::log("getuid() = {}\n", ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::geteuid: {
            reg_t ret = geteuid();
            if (state::strace) {
                util::log("geteuid() = {}\n", ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::getgid: {
            reg_t ret = getgid();
            if (state::strace) {
                util::log("getgid() = {}\n", ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::getegid: {
            reg_t ret = getegid();
            if (state::strace) {
                util::log("getegid() = {}\n", ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::brk: {
            if (arg0 < state::original_brk) {
                // Cannot reduce beyond original_brk
            } else if (arg0 <= state::heap_end) {
                if (arg0 > state::brk) {
                    zero_memory(state::brk, arg0 - state::brk);
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
                    zero_memory(state::brk, state::heap_end - state::brk);
                    state::heap_end = new_heap_end;
                    state::brk = arg0;
                }
            }

            reg_t ret = state::brk;
            if (state::strace) {
                util::log("brk({}) = {}\n", pointer(arg0), pointer(ret));
            }
            return ret;
        }
        case riscv::abi::Syscall_number::munmap: {
            reg_t ret = return_errno(guest_munmap(arg0, arg1));
            if (state::strace) {
                util::error("munmap({:#x}, {}) = {}\n", arg0, arg1, ret);
            }

            return ret;
        }
        // This is linux specific call, we will just return ENOSYS.
        case riscv::abi::Syscall_number::mremap: {
            if (state::strace) {
                util::error("mremap({:#x}, {}, {}, {}, {:#x}) = -ENOSYS\n", arg0, arg1, arg2, arg3, arg4);
            }
            return -static_cast<sreg_t>(riscv::abi::Errno::enosys);;
        }
        case riscv::abi::Syscall_number::mmap: {
            int prot = convert_mmap_prot_from_host<Abi>(arg2);
            int flags = convert_mmap_flags_from_host<Abi>(arg3);
            reg_t ret = reinterpret_cast<reg_t>(guest_mmap(arg0, arg1, prot, flags, arg4, arg5));
            if (state::strace) {
                util::error("mmap({:#x}, {}, {}, {}, {}, {}) = {:#x}\n", arg0, arg1, arg2, arg3, arg4, arg5, ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::mprotect: {
            int prot = convert_mmap_prot_from_host<Abi>(arg2);
            sreg_t ret = return_errno(guest_mprotect(arg0, arg1, prot));
            if (state::strace) {
                util::error("mprotect({:#x}, {}, {}) = {:#x}\n", arg0, arg1, arg2, ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::open: {
            auto pathname = reinterpret_cast<char*>(translate_address(arg0));
            auto flags = convert_open_flags_to_host(arg1);

            sreg_t ret = return_errno(open(translate_path(pathname), flags, arg2));
            if (state::strace) {
                util::log("open({}, {}, {}) = {}\n", escape(pathname), arg1, arg2, ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::unlink: {
            auto pathname = reinterpret_cast<char*>(translate_address(arg0));
            sreg_t ret = return_errno(unlink(translate_path(pathname)));
            if (state::strace) {
                util::log("unlink({}) = {}\n", escape(pathname), ret);
            }
            return ret;
        }
        case riscv::abi::Syscall_number::stat: {
            auto pathname = reinterpret_cast<char*>(translate_address(arg0));

            struct stat host_stat;
            sreg_t ret = return_errno(stat(translate_path(pathname), &host_stat));

            // When success, convert stat format to guest format.
            if (ret == 0) {
                struct riscv::abi::stat *guest_stat = reinterpret_cast<riscv::abi::stat*>(translate_address(arg1));
                convert_stat_from_host(guest_stat, &host_stat);
            }

            if (state::strace) {
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

