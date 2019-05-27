#![allow(bad_style,unused)]

pub type c_ulong = u64;
pub type c_uint = u32;
pub type c_long = i64;
pub type c_int = i32;
pub type ssize_t = i64;

pub const SYS_io_setup              : c_long = 0;
pub const SYS_io_destroy            : c_long = 1;
pub const SYS_io_submit             : c_long = 2;
pub const SYS_io_cancel             : c_long = 3;
pub const SYS_io_getevents          : c_long = 4;
pub const SYS_setxattr              : c_long = 5;
pub const SYS_lsetxattr             : c_long = 6;
pub const SYS_fsetxattr             : c_long = 7;
pub const SYS_getxattr              : c_long = 8;
pub const SYS_lgetxattr             : c_long = 9;
pub const SYS_fgetxattr             : c_long = 10;
pub const SYS_listxattr             : c_long = 11;
pub const SYS_llistxattr            : c_long = 12;
pub const SYS_flistxattr            : c_long = 13;
pub const SYS_removexattr           : c_long = 14;
pub const SYS_lremovexattr          : c_long = 15;
pub const SYS_fremovexattr          : c_long = 16;
pub const SYS_getcwd                : c_long = 17;
pub const SYS_lookup_dcookie        : c_long = 18;
pub const SYS_eventfd2              : c_long = 19;
pub const SYS_epoll_create1         : c_long = 20;
pub const SYS_epoll_ctl             : c_long = 21;
pub const SYS_epoll_pwait           : c_long = 22;
pub const SYS_dup                   : c_long = 23;
pub const SYS_dup3                  : c_long = 24;
pub const SYS_fcntl                 : c_long = 25;
pub const SYS_inotify_init1         : c_long = 26;
pub const SYS_inotify_add_watch     : c_long = 27;
pub const SYS_inotify_rm_watch      : c_long = 28;
pub const SYS_ioctl                 : c_long = 29;
pub const SYS_ioprio_set            : c_long = 30;
pub const SYS_ioprio_get            : c_long = 31;
pub const SYS_flock                 : c_long = 32;
pub const SYS_mknodat               : c_long = 33;
pub const SYS_mkdirat               : c_long = 34;
pub const SYS_unlinkat              : c_long = 35;
pub const SYS_symlinkat             : c_long = 36;
pub const SYS_linkat                : c_long = 37;
pub const SYS_umount                : c_long = 39;
pub const SYS_mount                 : c_long = 40;
pub const SYS_pivot_root            : c_long = 41;
pub const SYS_ni_syscall            : c_long = 42;
pub const SYS_statfs                : c_long = 43;
pub const SYS_fstatfs               : c_long = 44;
pub const SYS_truncate              : c_long = 45;
pub const SYS_ftruncate             : c_long = 46;
pub const SYS_fallocate             : c_long = 47;
pub const SYS_faccessat             : c_long = 48;
pub const SYS_chdir                 : c_long = 49;
pub const SYS_fchdir                : c_long = 50;
pub const SYS_chroot                : c_long = 51;
pub const SYS_fchmod                : c_long = 52;
pub const SYS_fchmodat              : c_long = 53;
pub const SYS_fchownat              : c_long = 54;
pub const SYS_fchown                : c_long = 55;
pub const SYS_openat                : c_long = 56;
pub const SYS_close                 : c_long = 57;
pub const SYS_vhangup               : c_long = 58;
pub const SYS_pipe2                 : c_long = 59;
pub const SYS_quotactl              : c_long = 60;
pub const SYS_getdents64            : c_long = 61;
pub const SYS_lseek                 : c_long = 62;
pub const SYS_read                  : c_long = 63;
pub const SYS_write                 : c_long = 64;
pub const SYS_readv                 : c_long = 65;
pub const SYS_writev                : c_long = 66;
pub const SYS_pread64               : c_long = 67;
pub const SYS_pwrite64              : c_long = 68;
pub const SYS_preadv                : c_long = 69;
pub const SYS_pwritev               : c_long = 70;
pub const SYS_sendfile64            : c_long = 71;
pub const SYS_pselect6              : c_long = 72;
pub const SYS_ppoll                 : c_long = 73;
pub const SYS_signalfd4             : c_long = 74;
pub const SYS_vmsplice              : c_long = 75;
pub const SYS_splice                : c_long = 76;
pub const SYS_tee                   : c_long = 77;
pub const SYS_readlinkat            : c_long = 78;
pub const SYS_fstatat               : c_long = 79;
pub const SYS_fstat                 : c_long = 80;
pub const SYS_sync                  : c_long = 81;
pub const SYS_fsync                 : c_long = 82;
pub const SYS_fdatasync             : c_long = 83;
pub const SYS_sync_file_range       : c_long = 84;
pub const SYS_timerfd_create        : c_long = 85;
pub const SYS_timerfd_settime       : c_long = 86;
pub const SYS_timerfd_gettime       : c_long = 87;
pub const SYS_utimensat             : c_long = 88;
pub const SYS_acct                  : c_long = 89;
pub const SYS_capget                : c_long = 90;
pub const SYS_capset                : c_long = 91;
pub const SYS_personality           : c_long = 92;
pub const SYS_exit                  : c_long = 93;
pub const SYS_exit_group            : c_long = 94;
pub const SYS_waitid                : c_long = 95;
pub const SYS_set_tid_address       : c_long = 96;
pub const SYS_unshare               : c_long = 97;
pub const SYS_futex                 : c_long = 98;
pub const SYS_set_robust_list       : c_long = 99;
pub const SYS_get_robust_list       : c_long = 100;
pub const SYS_nanosleep             : c_long = 101;
pub const SYS_getitimer             : c_long = 102;
pub const SYS_setitimer             : c_long = 103;
pub const SYS_kexec_load            : c_long = 104;
pub const SYS_init_module           : c_long = 105;
pub const SYS_delete_module         : c_long = 106;
pub const SYS_timer_create          : c_long = 107;
pub const SYS_timer_gettime         : c_long = 108;
pub const SYS_timer_getoverrun      : c_long = 109;
pub const SYS_timer_settime         : c_long = 110;
pub const SYS_timer_delete          : c_long = 111;
pub const SYS_clock_settime         : c_long = 112;
pub const SYS_clock_gettime         : c_long = 113;
pub const SYS_clock_getres          : c_long = 114;
pub const SYS_clock_nanosleep       : c_long = 115;
pub const SYS_syslog                : c_long = 116;
pub const SYS_ptrace                : c_long = 117;
pub const SYS_sched_setparam        : c_long = 118;
pub const SYS_sched_setscheduler    : c_long = 119;
pub const SYS_sched_getscheduler    : c_long = 120;
pub const SYS_sched_getparam        : c_long = 121;
pub const SYS_sched_setaffinity     : c_long = 122;
pub const SYS_sched_getaffinity     : c_long = 123;
pub const SYS_sched_yield           : c_long = 124;
pub const SYS_sched_get_priority_max: c_long = 125;
pub const SYS_sched_get_priority_min: c_long = 126;
pub const SYS_sched_rr_get_interval : c_long = 127;
pub const SYS_restart_syscall       : c_long = 128;
pub const SYS_kill                  : c_long = 129;
pub const SYS_tkill                 : c_long = 130;
pub const SYS_tgkill                : c_long = 131;
pub const SYS_sigaltstack           : c_long = 132;
pub const SYS_rt_sigsuspend         : c_long = 133;
pub const SYS_rt_sigaction          : c_long = 134;
pub const SYS_rt_sigprocmask        : c_long = 135;
pub const SYS_rt_sigpending         : c_long = 136;
pub const SYS_rt_sigtimedwait       : c_long = 137;
pub const SYS_rt_sigqueueinfo       : c_long = 138;
pub const SYS_rt_sigreturn          : c_long = 139;
pub const SYS_setpriority           : c_long = 140;
pub const SYS_getpriority           : c_long = 141;
pub const SYS_reboot                : c_long = 142;
pub const SYS_setregid              : c_long = 143;
pub const SYS_setgid                : c_long = 144;
pub const SYS_setreuid              : c_long = 145;
pub const SYS_setuid                : c_long = 146;
pub const SYS_setresuid             : c_long = 147;
pub const SYS_getresuid             : c_long = 148;
pub const SYS_setresgid             : c_long = 149;
pub const SYS_getresgid             : c_long = 150;
pub const SYS_setfsuid              : c_long = 151;
pub const SYS_setfsgid              : c_long = 152;
pub const SYS_times                 : c_long = 153;
pub const SYS_setpgid               : c_long = 154;
pub const SYS_getpgid               : c_long = 155;
pub const SYS_getsid                : c_long = 156;
pub const SYS_setsid                : c_long = 157;
pub const SYS_getgroups             : c_long = 158;
pub const SYS_setgroups             : c_long = 159;
pub const SYS_uname                 : c_long = 160;
pub const SYS_sethostname           : c_long = 161;
pub const SYS_setdomainname         : c_long = 162;
pub const SYS_getrlimit             : c_long = 163;
pub const SYS_setrlimit             : c_long = 164;
pub const SYS_getrusage             : c_long = 165;
pub const SYS_umask                 : c_long = 166;
pub const SYS_prctl                 : c_long = 167;
pub const SYS_getcpu                : c_long = 168;
pub const SYS_gettimeofday          : c_long = 169;
pub const SYS_settimeofday          : c_long = 170;
pub const SYS_adjtimex              : c_long = 171;
pub const SYS_getpid                : c_long = 172;
pub const SYS_getppid               : c_long = 173;
pub const SYS_getuid                : c_long = 174;
pub const SYS_geteuid               : c_long = 175;
pub const SYS_getgid                : c_long = 176;
pub const SYS_getegid               : c_long = 177;
pub const SYS_gettid                : c_long = 178;
pub const SYS_sysinfo               : c_long = 179;
pub const SYS_mq_open               : c_long = 180;
pub const SYS_mq_unlink             : c_long = 181;
pub const SYS_mq_timedsend          : c_long = 182;
pub const SYS_mq_timedreceive       : c_long = 183;
pub const SYS_mq_notify             : c_long = 184;
pub const SYS_mq_getsetattr         : c_long = 185;
pub const SYS_msgget                : c_long = 186;
pub const SYS_msgctl                : c_long = 187;
pub const SYS_msgrcv                : c_long = 188;
pub const SYS_msgsnd                : c_long = 189;
pub const SYS_semget                : c_long = 190;
pub const SYS_semctl                : c_long = 191;
pub const SYS_semtimedop            : c_long = 192;
pub const SYS_semop                 : c_long = 193;
pub const SYS_shmget                : c_long = 194;
pub const SYS_shmctl                : c_long = 195;
pub const SYS_shmat                 : c_long = 196;
pub const SYS_shmdt                 : c_long = 197;
pub const SYS_socket                : c_long = 198;
pub const SYS_socketpair            : c_long = 199;
pub const SYS_bind                  : c_long = 200;
pub const SYS_listen                : c_long = 201;
pub const SYS_accept                : c_long = 202;
pub const SYS_connect               : c_long = 203;
pub const SYS_getsockname           : c_long = 204;
pub const SYS_getpeername           : c_long = 205;
pub const SYS_sendto                : c_long = 206;
pub const SYS_recvfrom              : c_long = 207;
pub const SYS_setsockopt            : c_long = 208;
pub const SYS_getsockopt            : c_long = 209;
pub const SYS_shutdown              : c_long = 210;
pub const SYS_sendmsg               : c_long = 211;
pub const SYS_recvmsg               : c_long = 212;
pub const SYS_readahead             : c_long = 213;
pub const SYS_brk                   : c_long = 214;
pub const SYS_munmap                : c_long = 215;
pub const SYS_mremap                : c_long = 216;
pub const SYS_add_key               : c_long = 217;
pub const SYS_request_key           : c_long = 218;
pub const SYS_keyctl                : c_long = 219;
pub const SYS_clone                 : c_long = 220;
pub const SYS_execve                : c_long = 221;
pub const SYS_mmap                  : c_long = 222;
pub const SYS_fadvise64_64          : c_long = 223;
pub const SYS_swapon                : c_long = 224;
pub const SYS_swapoff               : c_long = 225;
pub const SYS_mprotect              : c_long = 226;
pub const SYS_msync                 : c_long = 227;
pub const SYS_mlock                 : c_long = 228;
pub const SYS_munlock               : c_long = 229;
pub const SYS_mlockall              : c_long = 230;
pub const SYS_munlockall            : c_long = 231;
pub const SYS_mincore               : c_long = 232;
pub const SYS_madvise               : c_long = 233;
pub const SYS_remap_file_pages      : c_long = 234;
pub const SYS_mbind                 : c_long = 235;
pub const SYS_get_mempolicy         : c_long = 236;
pub const SYS_set_mempolicy         : c_long = 237;
pub const SYS_migrate_pages         : c_long = 238;
pub const SYS_move_pages            : c_long = 239;
pub const SYS_rt_tgsigqueueinfo     : c_long = 240;
pub const SYS_perf_event_open       : c_long = 241;
pub const SYS_accept4               : c_long = 242;
pub const SYS_recvmmsg              : c_long = 243;
pub const SYS_wait4                 : c_long = 260;
pub const SYS_prlimit64             : c_long = 261;
pub const SYS_fanotify_init         : c_long = 262;
pub const SYS_fanotify_mark         : c_long = 263;
pub const SYS_name_to_handle_at     : c_long = 264;
pub const SYS_open_by_handle_at     : c_long = 265;
pub const SYS_clock_adjtime         : c_long = 266;
pub const SYS_syncfs                : c_long = 267;
pub const SYS_setns                 : c_long = 268;
pub const SYS_sendmmsg              : c_long = 269;
pub const SYS_process_vm_readv      : c_long = 270;
pub const SYS_process_vm_writev     : c_long = 271;
pub const SYS_kcmp                  : c_long = 272;
pub const SYS_finit_module          : c_long = 273;
pub const SYS_sched_setattr         : c_long = 274;
pub const SYS_sched_getattr         : c_long = 275;
pub const SYS_renameat2             : c_long = 276;
pub const SYS_seccomp               : c_long = 277;
pub const SYS_getrandom             : c_long = 278;
pub const SYS_memfd_create          : c_long = 279;
pub const SYS_bpf                   : c_long = 280;
pub const SYS_execveat              : c_long = 281;
pub const SYS_userfaultfd           : c_long = 282;
pub const SYS_membarrier            : c_long = 283;
pub const SYS_mlock2                : c_long = 284;
pub const SYS_copy_file_range       : c_long = 285;
pub const SYS_preadv2               : c_long = 286;
pub const SYS_pwritev2              : c_long = 287;
pub const SYS_pkey_mprotect         : c_long = 288;
pub const SYS_pkey_alloc            : c_long = 289;
pub const SYS_pkey_free             : c_long = 290;
pub const SYS_statx                 : c_long = 291;
pub const SYS_open                  : c_long = 1024;
pub const SYS_unlink                : c_long = 1026;
pub const SYS_stat                  : c_long = 1038;

/// Operation not permitted
pub const EPERM          : c_int = 1;
/// No such file or directory
pub const ENOENT         : c_int = 2;
/// No such process
pub const ESRCH          : c_int = 3;
/// Interrupted system call
pub const EINTR          : c_int = 4;
/// I/O error
pub const EIO            : c_int = 5;
/// No such device or address
pub const ENXIO          : c_int = 6;
/// Argument list too long
pub const E2BIG          : c_int = 7;
/// Exec format error
pub const ENOEXEC        : c_int = 8;
/// Bad file number
pub const EBADF          : c_int = 9;
/// No child processes
pub const ECHILD         : c_int = 10;
/// Try again
pub const EAGAIN         : c_int = 11;
/// Out of memory
pub const ENOMEM         : c_int = 12;
/// Permission denied
pub const EACCES         : c_int = 13;
/// Bad address
pub const EFAULT         : c_int = 14;
/// Block device required
pub const ENOTBLK        : c_int = 15;
/// Device or resource busy
pub const EBUSY          : c_int = 16;
/// File exists
pub const EEXIST         : c_int = 17;
/// Cross-device link
pub const EXDEV          : c_int = 18;
/// No such device
pub const ENODEV         : c_int = 19;
/// Not a directory
pub const ENOTDIR        : c_int = 20;
/// Is a directory
pub const EISDIR         : c_int = 21;
/// Invalid argument
pub const EINVAL         : c_int = 22;
/// File table overflow
pub const ENFILE         : c_int = 23;
/// Too many open files
pub const EMFILE         : c_int = 24;
/// Not a typewriter
pub const ENOTTY         : c_int = 25;
/// Text file busy
pub const ETXTBSY        : c_int = 26;
/// File too large
pub const EFBIG          : c_int = 27;
/// No space left on device
pub const ENOSPC         : c_int = 28;
/// Illegal seek
pub const ESPIPE         : c_int = 29;
/// Read-only file system
pub const EROFS          : c_int = 30;
/// Too many links
pub const EMLINK         : c_int = 31;
/// Broken pipe
pub const EPIPE          : c_int = 32;
/// Math argument out of domain of func
pub const EDOM           : c_int = 33;
/// Math result not representable
pub const ERANGE         : c_int = 34;
/// Resource deadlock would occur
pub const EDEADLK        : c_int = 35;
/// File name too long
pub const ENAMETOOLONG   : c_int = 36;
/// No record locks available
pub const ENOLCK         : c_int = 37;
/// Invalid system call number
pub const ENOSYS         : c_int = 38;
/// Directory not empty
pub const ENOTEMPTY      : c_int = 39;
/// Too many symbolic links encountered
pub const ELOOP          : c_int = 40;
///  Operation would block
pub const EWOULDBLOCK    : c_int = EAGAIN;
/// No message of desired type
pub const ENOMSG         : c_int = 42;
/// Identifier removed
pub const EIDRM          : c_int = 43;
/// Channel number out of range
pub const ECHRNG         : c_int = 44;
/// Level 2 not synchronized
pub const EL2NSYNC       : c_int = 45;
/// Level 3 halted
pub const EL3HLT         : c_int = 46;
/// Level 3 reset
pub const EL3RST         : c_int = 47;
/// Link number out of range
pub const ELNRNG         : c_int = 48;
/// Protocol driver not attached
pub const EUNATCH        : c_int = 49;
/// No CSI structure available
pub const ENOCSI         : c_int = 50;
/// Level 2 halted
pub const EL2HLT         : c_int = 51;
/// Invalid exchange
pub const EBADE          : c_int = 52;
/// Invalid request descriptor
pub const EBADR          : c_int = 53;
/// Exchange full
pub const EXFULL         : c_int = 54;
/// No anode
pub const ENOANO         : c_int = 55;
/// Invalid request code
pub const EBADRQC        : c_int = 56;
/// Invalid slot
pub const EBADSLT        : c_int = 57;
pub const EDEADLOCK      : c_int = EDEADLK;
/// Bad font file format
pub const EBFONT         : c_int = 59;
/// Device not a stream
pub const ENOSTR         : c_int = 60;
/// No data available
pub const ENODATA        : c_int = 61;
/// Timer expired
pub const ETIME          : c_int = 62;
/// Out of streams resources
pub const ENOSR          : c_int = 63;
/// Machine is not on the network
pub const ENONET         : c_int = 64;
/// Package not installed
pub const ENOPKG         : c_int = 65;
/// Object is remote
pub const EREMOTE        : c_int = 66;
/// Link has been severed
pub const ENOLINK        : c_int = 67;
/// Advertise error
pub const EADV           : c_int = 68;
/// Srmount error
pub const ESRMNT         : c_int = 69;
/// Communication error on send
pub const ECOMM          : c_int = 70;
/// Protocol error
pub const EPROTO         : c_int = 71;
/// Multihop attempted
pub const EMULTIHOP      : c_int = 72;
/// RFS specific error
pub const EDOTDOT        : c_int = 73;
/// Not a data message
pub const EBADMSG        : c_int = 74;
/// Value too large for data type
pub const EOVERFLOW      : c_int = 75;
/// Name not unique on network
pub const ENOTUNIQ       : c_int = 76;
/// File descriptor in bad state
pub const EBADFD         : c_int = 77;
/// Remote address changed
pub const EREMCHG        : c_int = 78;
/// Can not access a needed shared library
pub const ELIBACC        : c_int = 79;
/// Accessing a corrupted shared library
pub const ELIBBAD        : c_int = 80;
/// .lib section in a.out corrupted
pub const ELIBSCN        : c_int = 81;
/// Attempting to link in too many shared libraries
pub const ELIBMAX        : c_int = 82;
/// Cannot exec a shared library directly
pub const ELIBEXEC       : c_int = 83;
/// Illegal byte sequence
pub const EILSEQ         : c_int = 84;
/// Interrupted system call should be restarted
pub const ERESTART       : c_int = 85;
/// Streams pipe error
pub const ESTRPIPE       : c_int = 86;
/// Too many users
pub const EUSERS         : c_int = 87;
/// Socket operation on non-socket
pub const ENOTSOCK       : c_int = 88;
/// Destination address required
pub const EDESTADDRREQ   : c_int = 89;
/// Message too long
pub const EMSGSIZE       : c_int = 90;
/// Protocol wrong type for socket
pub const EPROTOTYPE     : c_int = 91;
/// Protocol not available
pub const ENOPROTOOPT    : c_int = 92;
/// Protocol not supported
pub const EPROTONOSUPPORT: c_int = 93;
/// Socket type not supported
pub const ESOCKTNOSUPPORT: c_int = 94;
/// Operation not supported on transport endpoint
pub const EOPNOTSUPP     : c_int = 95;
/// Protocol family not supported
pub const EPFNOSUPPORT   : c_int = 96;
/// Address family not supported by protocol
pub const EAFNOSUPPORT   : c_int = 97;
/// Address already in use
pub const EADDRINUSE     : c_int = 98;
/// Cannot assign requested address
pub const EADDRNOTAVAIL  : c_int = 99;
/// Network is down
pub const ENETDOWN       : c_int = 100;
/// Network is unreachable
pub const ENETUNREACH    : c_int = 101;
/// Network dropped connection because of reset
pub const ENETRESET      : c_int = 102;
/// Software caused connection abort
pub const ECONNABORTED   : c_int = 103;
/// Connection reset by peer
pub const ECONNRESET     : c_int = 104;
/// No buffer space available
pub const ENOBUFS        : c_int = 105;
/// Transport endpoint is already connected
pub const EISCONN        : c_int = 106;
/// Transport endpoint is not connected
pub const ENOTCONN       : c_int = 107;
/// Cannot send after transport endpoint shutdown
pub const ESHUTDOWN      : c_int = 108;
/// Too many references: cannot splice
pub const ETOOMANYREFS   : c_int = 109;
/// Connection timed out
pub const ETIMEDOUT      : c_int = 110;
/// Connection refused
pub const ECONNREFUSED   : c_int = 111;
/// Host is down
pub const EHOSTDOWN      : c_int = 112;
/// No route to host
pub const EHOSTUNREACH   : c_int = 113;
/// Operation already in progress
pub const EALREADY       : c_int = 114;
/// Operation now in progress
pub const EINPROGRESS    : c_int = 115;
/// Stale file handle
pub const ESTALE         : c_int = 116;
/// Structure needs cleaning
pub const EUCLEAN        : c_int = 117;
/// Not a XENIX named type file
pub const ENOTNAM        : c_int = 118;
/// No XENIX semaphores available
pub const ENAVAIL        : c_int = 119;
/// Is a named type file
pub const EISNAM         : c_int = 120;
/// Remote I/O error
pub const EREMOTEIO      : c_int = 121;
/// Quota exceeded
pub const EDQUOT         : c_int = 122;
/// No medium found
pub const ENOMEDIUM      : c_int = 123;
/// Wrong medium type
pub const EMEDIUMTYPE    : c_int = 124;
/// Operation Canceled
pub const ECANCELED      : c_int = 125;
/// Required key not available
pub const ENOKEY         : c_int = 126;
/// Key has expired
pub const EKEYEXPIRED    : c_int = 127;
/// Key has been revoked
pub const EKEYREVOKED    : c_int = 128;
/// Key was rejected by service
pub const EKEYREJECTED   : c_int = 129;
/// Owner died
pub const EOWNERDEAD     : c_int = 130;
/// State not recoverable
pub const ENOTRECOVERABLE: c_int = 131;
/// Operation not possible due to RF-kill
pub const ERFKILL        : c_int = 131;
/// Memory page has hardware error
pub const EHWPOISON      : c_int = 133;

// Somehow not present in Linux headers; but per POSIX spec should be present.
pub const ENOTSUP        : c_int = EOPNOTSUPP;

#[repr(C)]
pub struct stat {
    /// Device
    pub st_dev       : c_ulong,
    /// File serial number
    pub st_ino       : c_ulong,
    /// File mode
    pub st_mode      : c_uint,
    /// Link count
    pub st_nlink     : c_uint,
    /// User ID of the file's owner
    pub st_uid       : c_uint,
    /// Group ID of the file's group
    pub st_gid       : c_uint,
    /// Device number, if device
    pub st_rdev      : c_ulong,
    pub pad1         : c_ulong,
    /// Size of file, in bytes
    pub st_size      : c_long,
    /// Optimal block size for I/O
    pub st_blksize   : c_int,
    pub pad2         : c_int,
    /// Number 512-byte blocks allocated
    pub st_blocks    : c_long,
    /// Time of last access
    pub st_atime     : c_long,
    pub st_atime_nsec: c_ulong,
    /// Time of last modification
    pub st_mtime     : c_long,
    pub st_mtime_nsec: c_ulong,
    /// Time of last status change
    pub st_ctime     : c_long,
    pub st_ctime_nsec: c_ulong,
    pub unused4      : c_uint,
    pub unused5      : c_uint,
}

#[repr(C)]
pub struct timeval {
    pub tv_sec : c_long,
    pub tv_usec: c_long,
}

pub const AT_FDCWD       : c_int = -100;
pub const PROT_READ      : c_int = 1;
pub const PROT_WRITE     : c_int = 2;
pub const PROT_EXEC      : c_int = 4;
pub const MAP_SHARED     : c_int = 1;
pub const MAP_PRIVATE    : c_int = 2;
pub const MAP_FIXED      : c_int = 0x10;
pub const MAP_ANON       : c_int = 0x20;

#[repr(C)]
pub struct iovec {
    pub iov_base: c_ulong,
    pub iov_len : c_ulong,
}

pub const AT_NULL          : c_ulong = 0;
pub const AT_IGNORE        : c_ulong = 1;
pub const AT_EXECFD        : c_ulong = 2;
pub const AT_PHDR          : c_ulong = 3;
pub const AT_PHENT         : c_ulong = 4;
pub const AT_PHNUM         : c_ulong = 5;
pub const AT_PAGESZ        : c_ulong = 6;
pub const AT_BASE          : c_ulong = 7;
pub const AT_FLAGS         : c_ulong = 8;
pub const AT_ENTRY         : c_ulong = 9;
pub const AT_NOTELF        : c_ulong = 10;
pub const AT_UID           : c_ulong = 11;
pub const AT_EUID          : c_ulong = 12;
pub const AT_GID           : c_ulong = 13;
pub const AT_EGID          : c_ulong = 14;
pub const AT_PLATFORM      : c_ulong = 15;
pub const AT_HWCAP         : c_ulong = 16;
pub const AT_CLKTCK        : c_ulong = 17;
pub const AT_SECURE        : c_ulong = 23;
pub const AT_BASE_PLATFORM : c_ulong = 24;
pub const AT_RANDOM        : c_ulong = 25;
pub const AT_HWCAP2        : c_ulong = 26;
pub const AT_EXECFN        : c_ulong = 31;
