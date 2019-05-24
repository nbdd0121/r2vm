#![allow(non_camel_case_types)]

pub type c_ulong = u64;
pub type c_uint = u32;
pub type c_long = i64;
pub type c_int = i32;
pub type ssize_t = i64;

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
