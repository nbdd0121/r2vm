#ifndef RISCV_ABI_H
#define RISCV_ABI_H

#include <cstdint>

// This file should be ported from RISC-V Linux headers to make it ABI-compatible.

namespace riscv::abi {

enum class Syscall_number {
    io_setup = 0,
    io_destroy = 1,
    io_submit = 2,
    io_cancel = 3,
    io_getevents = 4,
    setxattr = 5,
    lsetxattr = 6,
    fsetxattr = 7,
    getxattr = 8,
    lgetxattr = 9,
    fgetxattr = 10,
    listxattr = 11,
    llistxattr = 12,
    flistxattr = 13,
    removexattr = 14,
    lremovexattr = 15,
    fremovexattr = 16,
    getcwd = 17,
    lookup_dcookie = 18,
    eventfd2 = 19,
    epoll_create1 = 20,
    epoll_ctl = 21,
    epoll_pwait = 22,
    dup = 23,
    dup3 = 24,
    fcntl = 25,
    inotify_init1 = 26,
    inotify_add_watch = 27,
    inotify_rm_watch = 28,
    ioctl = 29,
    ioprio_set = 30,
    ioprio_get = 31,
    flock = 32,
    mknodat = 33,
    mkdirat = 34,
    unlinkat = 35,
    symlinkat = 36,
    linkat = 37,
    umount = 39,
    mount = 40,
    pivot_root = 41,
    ni_syscall = 42,
    statfs = 43,
    fstatfs = 44,
    truncate = 45,
    ftruncate = 46,
    fallocate = 47,
    faccessat = 48,
    chdir = 49,
    fchdir = 50,
    chroot = 51,
    fchmod = 52,
    fchmodat = 53,
    fchownat = 54,
    fchown = 55,
    openat = 56,
    close = 57,
    vhangup = 58,
    pipe2 = 59,
    quotactl = 60,
    getdents64 = 61,
    lseek = 62,
    read = 63,
    write = 64,
    readv = 65,
    writev = 66,
    pread64 = 67,
    pwrite64 = 68,
    preadv = 69,
    pwritev = 70,
    sendfile64 = 71,
    pselect6 = 72,
    ppoll = 73,
    signalfd4 = 74,
    vmsplice = 75,
    splice = 76,
    tee = 77,
    readlinkat = 78,
    fstatat = 79,
    fstat = 80,
    sync = 81,
    fsync = 82,
    fdatasync = 83,
    sync_file_range = 84,
    timerfd_create = 85,
    timerfd_settime = 86,
    timerfd_gettime = 87,
    utimensat = 88,
    acct = 89,
    capget = 90,
    capset = 91,
    personality = 92,
    exit = 93,
    exit_group = 94,
    waitid = 95,
    set_tid_address = 96,
    unshare = 97,
    futex = 98,
    set_robust_list = 99,
    get_robust_list = 100,
    nanosleep = 101,
    getitimer = 102,
    setitimer = 103,
    kexec_load = 104,
    init_module = 105,
    delete_module = 106,
    timer_create = 107,
    timer_gettime = 108,
    timer_getoverrun = 109,
    timer_settime = 110,
    timer_delete = 111,
    clock_settime = 112,
    clock_gettime = 113,
    clock_getres = 114,
    clock_nanosleep = 115,
    syslog = 116,
    ptrace = 117,
    sched_setparam = 118,
    sched_setscheduler = 119,
    sched_getscheduler = 120,
    sched_getparam = 121,
    sched_setaffinity = 122,
    sched_getaffinity = 123,
    sched_yield = 124,
    sched_get_priority_max = 125,
    sched_get_priority_min = 126,
    sched_rr_get_interval = 127,
    restart_syscall = 128,
    kill = 129,
    tkill = 130,
    tgkill = 131,
    sigaltstack = 132,
    rt_sigsuspend = 133,
    rt_sigaction = 134,
    rt_sigprocmask = 135,
    rt_sigpending = 136,
    rt_sigtimedwait = 137,
    rt_sigqueueinfo = 138,
    rt_sigreturn = 139,
    setpriority = 140,
    getpriority = 141,
    reboot = 142,
    setregid = 143,
    setgid = 144,
    setreuid = 145,
    setuid = 146,
    setresuid = 147,
    getresuid = 148,
    setresgid = 149,
    getresgid = 150,
    setfsuid = 151,
    setfsgid = 152,
    times = 153,
    setpgid = 154,
    getpgid = 155,
    getsid = 156,
    setsid = 157,
    getgroups = 158,
    setgroups = 159,
    uname = 160,
    sethostname = 161,
    setdomainname = 162,
    getrlimit = 163,
    setrlimit = 164,
    getrusage = 165,
    umask = 166,
    prctl = 167,
    getcpu = 168,
    gettimeofday = 169,
    settimeofday = 170,
    adjtimex = 171,
    getpid = 172,
    getppid = 173,
    getuid = 174,
    geteuid = 175,
    getgid = 176,
    getegid = 177,
    gettid = 178,
    sysinfo = 179,
    mq_open = 180,
    mq_unlink = 181,
    mq_timedsend = 182,
    mq_timedreceive = 183,
    mq_notify = 184,
    mq_getsetattr = 185,
    msgget = 186,
    msgctl = 187,
    msgrcv = 188,
    msgsnd = 189,
    semget = 190,
    semctl = 191,
    semtimedop = 192,
    semop = 193,
    shmget = 194,
    shmctl = 195,
    shmat = 196,
    shmdt = 197,
    socket = 198,
    socketpair = 199,
    bind = 200,
    listen = 201,
    accept = 202,
    connect = 203,
    getsockname = 204,
    getpeername = 205,
    sendto = 206,
    recvfrom = 207,
    setsockopt = 208,
    getsockopt = 209,
    shutdown = 210,
    sendmsg = 211,
    recvmsg = 212,
    readahead = 213,
    brk = 214,
    munmap = 215,
    mremap = 216,
    add_key = 217,
    request_key = 218,
    keyctl = 219,
    clone = 220,
    execve = 221,
    mmap = 222,
    fadvise64_64 = 223,
    swapon = 224,
    swapoff = 225,
    mprotect = 226,
    msync = 227,
    mlock = 228,
    munlock = 229,
    mlockall = 230,
    munlockall = 231,
    mincore = 232,
    madvise = 233,
    remap_file_pages = 234,
    mbind = 235,
    get_mempolicy = 236,
    set_mempolicy = 237,
    migrate_pages = 238,
    move_pages = 239,
    rt_tgsigqueueinfo = 240,
    perf_event_open = 241,
    accept4 = 242,
    recvmmsg = 243,
    wait4 = 260,
    prlimit64 = 261,
    fanotify_init = 262,
    fanotify_mark = 263,
    name_to_handle_at = 264,
    open_by_handle_at = 265,
    clock_adjtime = 266,
    syncfs = 267,
    setns = 268,
    sendmmsg = 269,
    process_vm_readv = 270,
    process_vm_writev = 271,
    kcmp = 272,
    finit_module = 273,
    sched_setattr = 274,
    sched_getattr = 275,
    renameat2 = 276,
    seccomp = 277,
    getrandom = 278,
    memfd_create = 279,
    bpf = 280,
    execveat = 281,
    userfaultfd = 282,
    membarrier = 283,
    mlock2 = 284,
    copy_file_range = 285,
    preadv2 = 286,
    pwritev2 = 287,
    pkey_mprotect = 288,
    pkey_alloc = 289,
    pkey_free = 290,
    statx = 291,

    open = 1024,
    unlink = 1026,
    stat = 1038,
};

using ulong_t = uint64_t;
using uint_t = uint32_t;
using long_t = int64_t;
using int_t = int32_t;

enum class Errno {
    eperm           = 1,    /* Operation not permitted */
    enoent          = 2,    /* No such file or directory */
    esrch           = 3,    /* No such process */
    eintr           = 4,    /* Interrupted system call */
    eio             = 5,    /* I/O error */
    enxio           = 6,    /* No such device or address */
    e2big           = 7,    /* Argument list too long */
    enoexec         = 8,    /* Exec format error */
    ebadf           = 9,    /* Bad file number */
    echild          = 10,   /* No child processes */
    eagain          = 11,   /* Try again */
    enomem          = 12,   /* Out of memory */
    eacces          = 13,   /* Permission denied */
    efault          = 14,   /* Bad address */
    enotblk         = 15,   /* Block device required */
    ebusy           = 16,   /* Device or resource busy */
    eexist          = 17,   /* File exists */
    exdev           = 18,   /* Cross-device link */
    enodev          = 19,   /* No such device */
    enotdir         = 20,   /* Not a directory */
    eisdir          = 21,   /* Is a directory */
    einval          = 22,   /* Invalid argument */
    enfile          = 23,   /* File table overflow */
    emfile          = 24,   /* Too many open files */
    enotty          = 25,   /* Not a typewriter */
    etxtbsy         = 26,   /* Text file busy */
    efbig           = 27,   /* File too large */
    enospc          = 28,   /* No space left on device */
    espipe          = 29,   /* Illegal seek */
    erofs           = 30,   /* Read-only file system */
    emlink          = 31,   /* Too many links */
    epipe           = 32,   /* Broken pipe */
    edom            = 33,   /* Math argument out of domain of func */
    erange          = 34,   /* Math result not representable */
    edeadlk         = 35,   /* Resource deadlock would occur */
    enametoolong    = 36,   /* File name too long */
    enolck          = 37,   /* No record locks available */
    enosys          = 38,   /* Invalid system call number */
    enotempty       = 39,   /* Directory not empty */
    eloop           = 40,   /* Too many symbolic links encountered */
    ewouldblock     = eagain,   /* Operation would block */
    enomsg          = 42,   /* No message of desired type */
    eidrm           = 43,   /* Identifier removed */
    echrng          = 44,   /* Channel number out of range */
    el2nsync        = 45,   /* Level 2 not synchronized */
    el3hlt          = 46,   /* Level 3 halted */
    el3rst          = 47,   /* Level 3 reset */
    elnrng          = 48,   /* Link number out of range */
    eunatch         = 49,   /* Protocol driver not attached */
    enocsi          = 50,   /* No CSI structure available */
    el2hlt          = 51,   /* Level 2 halted */
    ebade           = 52,   /* Invalid exchange */
    ebadr           = 53,   /* Invalid request descriptor */
    exfull          = 54,   /* Exchange full */
    enoano          = 55,   /* No anode */
    ebadrqc         = 56,   /* Invalid request code */
    ebadslt         = 57,   /* Invalid slot */
    edeadlock       = edeadlk,
    ebfont          = 59,   /* Bad font file format */
    enostr          = 60,   /* Device not a stream */
    enodata         = 61,   /* No data available */
    etime           = 62,   /* Timer expired */
    enosr           = 63,   /* Out of streams resources */
    enonet          = 64,   /* Machine is not on the network */
    enopkg          = 65,   /* Package not installed */
    eremote         = 66,   /* Object is remote */
    enolink         = 67,   /* Link has been severed */
    eadv            = 68,   /* Advertise error */
    esrmnt          = 69,   /* Srmount error */
    ecomm           = 70,   /* Communication error on send */
    eproto          = 71,   /* Protocol error */
    emultihop       = 72,   /* Multihop attempted */
    edotdot         = 73,   /* RFS specific error */
    ebadmsg         = 74,   /* Not a data message */
    eoverflow       = 75,   /* Value too large for data type */
    enotuniq        = 76,   /* Name not unique on network */
    ebadfd          = 77,   /* File descriptor in bad state */
    eremchg         = 78,   /* Remote address changed */
    elibacc         = 79,   /* Can not access a needed shared library */
    elibbad         = 80,   /* Accessing a corrupted shared library */
    elibscn         = 81,   /* .lib section in a.out corrupted */
    elibmax         = 82,   /* Attempting to link in too many shared libraries */
    elibexec        = 83,   /* Cannot exec a shared library directly */
    eilseq          = 84,   /* Illegal byte sequence */
    erestart        = 85,   /* Interrupted system call should be restarted */
    estrpipe        = 86,   /* Streams pipe error */
    eusers          = 87,   /* Too many users */
    enotsock        = 88,   /* Socket operation on non-socket */
    edestaddrreq    = 89,   /* Destination address required */
    emsgsize        = 90,   /* Message too long */
    eprototype      = 91,   /* Protocol wrong type for socket */
    enoprotoopt     = 92,   /* Protocol not available */
    eprotonosupport = 93,   /* Protocol not supported */
    esocktnosupport = 94,   /* Socket type not supported */
    eopnotsupp      = 95,   /* Operation not supported on transport endpoint */
    epfnosupport    = 96,   /* Protocol family not supported */
    eafnosupport    = 97,   /* Address family not supported by protocol */
    eaddrinuse      = 98,   /* Address already in use */
    eaddrnotavail   = 99,   /* Cannot assign requested address */
    enetdown        = 100,  /* Network is down */
    enetunreach     = 101,  /* Network is unreachable */
    enetreset       = 102,  /* Network dropped connection because of reset */
    econnaborted    = 103,  /* Software caused connection abort */
    econnreset      = 104,  /* Connection reset by peer */
    enobufs         = 105,  /* No buffer space available */
    eisconn         = 106,  /* Transport endpoint is already connected */
    enotconn        = 107,  /* Transport endpoint is not connected */
    eshutdown       = 108,  /* Cannot send after transport endpoint shutdown */
    etoomanyrefs    = 109,  /* Too many references: cannot splice */
    etimedout       = 110,  /* Connection timed out */
    econnrefused    = 111,  /* Connection refused */
    ehostdown       = 112,  /* Host is down */
    ehostunreach    = 113,  /* No route to host */
    ealready        = 114,  /* Operation already in progress */
    einprogress     = 115,  /* Operation now in progress */
    estale          = 116,  /* Stale file handle */
    euclean         = 117,  /* Structure needs cleaning */
    enotnam         = 118,  /* Not a XENIX named type file */
    enavail         = 119,  /* No XENIX semaphores available */
    eisnam          = 120,  /* Is a named type file */
    eremoteio       = 121,  /* Remote I/O error */
    edquot          = 122,  /* Quota exceeded */
    enomedium       = 123,  /* No medium found */
    emediumtype     = 124,  /* Wrong medium type */
    ecanceled       = 125,  /* Operation Canceled */
    enokey          = 126,  /* Required key not available */
    ekeyexpired     = 127,  /* Key has expired */
    ekeyrevoked     = 128,  /* Key has been revoked */
    ekeyrejected    = 129,  /* Key was rejected by service */
    eownerdead      = 130,  /* Owner died */
    enotrecoverable = 131,  /* State not recoverable */
    erfkill         = 131,  /* Operation not possible due to RF-kill */
    ehwpoison       = 133,  /* Memory page has hardware error */

    // Somehow not present in Linux headers, but per POSIX spec should be present.
    enotsup         = eopnotsupp,
};

struct stat {
    ulong_t st_dev;         /* Device */
    ulong_t st_ino;         /* File serial number */
    uint_t  st_mode;        /* File mode */
    uint_t  st_nlink;       /* Link count */
    uint_t  st_uid;         /* User ID of the file's owner */
    uint_t  st_gid;         /* Group ID of the file's group */
    ulong_t st_rdev;        /* Device number, if device */
    ulong_t pad1;
    long_t  st_size;        /* Size of file, in bytes */
    int_t   st_blksize;     /* Optimal block size for I/O */
    int_t   pad2;
    long_t  st_blocks;      /* Number 512-byte blocks allocated */
    long_t  guest_st_atime; /* Time of last access */
    ulong_t st_atime_nsec;
    long_t  guest_st_mtime; /* Time of last modification */
    ulong_t st_mtime_nsec;
    long_t  guest_st_ctime; /* Time of last status change */
    ulong_t st_ctime_nsec;
    uint_t  unused4;
    uint_t  unused5;
};

struct timeval {
    long_t tv_sec;
    long_t tv_usec;
};

struct Abi {
    using int_t = int32_t;

    enum {
        guest_AT_FDCWD = -100,
        guest_UTSNAME_LENGTH = 65,
        guest_PROT_READ = 1,
        guest_PROT_WRITE = 2,
        guest_PROT_EXEC = 4,
        guest_MAP_SHARED = 1,
        guest_MAP_PRIVATE = 2,
        guest_MAP_FIXED = 0x10,
        guest_MAP_ANON = 0x20,
    };

    struct iovec {
        ulong_t iov_base;
        ulong_t iov_len;
    };

    struct utsname {
        char sysname[guest_UTSNAME_LENGTH];
        char nodename[guest_UTSNAME_LENGTH];
        char release[guest_UTSNAME_LENGTH];
        char version[guest_UTSNAME_LENGTH];
        char machine[guest_UTSNAME_LENGTH];
        char domainname[guest_UTSNAME_LENGTH];
    };
};

}

#endif
