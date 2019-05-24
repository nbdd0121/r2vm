use std::fmt::{self, Write};
use super::abi;

struct Escape<'a>(&'a [u8]);

impl<'a> fmt::Display for Escape<'a> {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut start = 0;
        let end = std::cmp::min(self.0.len(), 64);

        for i in 0..end {
            let code = self.0[i];
            let ch: char = code.into();

            // Skip printable ASCII characters.
            if code <= 0x7F && (ch != '"' && ch != '\\' && !ch.is_ascii_control()) { continue }

            // Print out all unprinted normal characters.
            if i != start {
                // It's okay to use unchecked here becuase we know this is all ASCII
                f.write_str(unsafe { std::str::from_utf8_unchecked(&self.0[start..i]) })?;
            }
        
            f.write_char('\\')?;
            match ch {
                '"' | '\\' => f.write_char(ch)?,
                '\n' => f.write_char('n')?,
                '\t' => f.write_char('t')?,
                _ => write!(f, "{:02x}", code)?,
            }

            start = i + 1;
        }

        if end != start {
            f.write_str(unsafe { std::str::from_utf8_unchecked(&self.0[start..end]) })?;
        }

        f.write_char('"')?;
        if self.0.len() > 64 { f.write_str("...")? }

        Ok(())
    }
}

struct Pointer(u64);

impl fmt::Display for Pointer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 != 0 {
            write!(f, "{:#x}", self.0)
        } else {
            f.write_str("NULL")
        }
    }
}

#[no_mangle]
#[allow(unreachable_patterns)]
extern "C" fn convert_errno_from_host(number: libc::c_int) -> abi::c_int {
    match number {
        libc::EPERM           => abi::EPERM          ,
        libc::ENOENT          => abi::ENOENT         ,
        libc::ESRCH           => abi::ESRCH          ,
        libc::EINTR           => abi::EINTR          ,
        libc::EIO             => abi::EIO            ,
        libc::ENXIO           => abi::ENXIO          ,
        libc::E2BIG           => abi::E2BIG          ,
        libc::ENOEXEC         => abi::ENOEXEC        ,
        libc::EBADF           => abi::EBADF          ,
        libc::ECHILD          => abi::ECHILD         ,
        libc::EAGAIN          => abi::EAGAIN         ,
        libc::ENOMEM          => abi::ENOMEM         ,
        libc::EACCES          => abi::EACCES         ,
        libc::EFAULT          => abi::EFAULT         ,
        libc::ENOTBLK         => abi::ENOTBLK        ,
        libc::EBUSY           => abi::EBUSY          ,
        libc::EEXIST          => abi::EEXIST         ,
        libc::EXDEV           => abi::EXDEV          ,
        libc::ENODEV          => abi::ENODEV         ,
        libc::ENOTDIR         => abi::ENOTDIR        ,
        libc::EISDIR          => abi::EISDIR         ,
        libc::EINVAL          => abi::EINVAL         ,
        libc::ENFILE          => abi::ENFILE         ,
        libc::EMFILE          => abi::EMFILE         ,
        libc::ENOTTY          => abi::ENOTTY         ,
        libc::ETXTBSY         => abi::ETXTBSY        ,
        libc::EFBIG           => abi::EFBIG          ,
        libc::ENOSPC          => abi::ENOSPC         ,
        libc::ESPIPE          => abi::ESPIPE         ,
        libc::EROFS           => abi::EROFS          ,
        libc::EMLINK          => abi::EMLINK         ,
        libc::EPIPE           => abi::EPIPE          ,
        libc::EDOM            => abi::EDOM           ,
        libc::ERANGE          => abi::ERANGE         ,
        libc::EDEADLK         => abi::EDEADLK        ,
        libc::ENAMETOOLONG    => abi::ENAMETOOLONG   ,
        libc::ENOLCK          => abi::ENOLCK         ,
        libc::ENOSYS          => abi::ENOSYS         ,
        libc::ENOTEMPTY       => abi::ENOTEMPTY      ,
        libc::ELOOP           => abi::ELOOP          ,
        libc::EWOULDBLOCK     => abi::EWOULDBLOCK    ,
        libc::ENOMSG          => abi::ENOMSG         ,
        libc::EIDRM           => abi::EIDRM          ,
        libc::ECHRNG          => abi::ECHRNG         ,
        libc::EL2NSYNC        => abi::EL2NSYNC       ,
        libc::EL3HLT          => abi::EL3HLT         ,
        libc::EL3RST          => abi::EL3RST         ,
        libc::ELNRNG          => abi::ELNRNG         ,
        libc::EUNATCH         => abi::EUNATCH        ,
        libc::ENOCSI          => abi::ENOCSI         ,
        libc::EL2HLT          => abi::EL2HLT         ,
        libc::EBADE           => abi::EBADE          ,
        libc::EBADR           => abi::EBADR          ,
        libc::EXFULL          => abi::EXFULL         ,
        libc::ENOANO          => abi::ENOANO         ,
        libc::EBADRQC         => abi::EBADRQC        ,
        libc::EBADSLT         => abi::EBADSLT        ,
        libc::EDEADLOCK       => abi::EDEADLOCK      ,
        libc::EBFONT          => abi::EBFONT         ,
        libc::ENOSTR          => abi::ENOSTR         ,
        libc::ENODATA         => abi::ENODATA        ,
        libc::ETIME           => abi::ETIME          ,
        libc::ENOSR           => abi::ENOSR          ,
        libc::ENONET          => abi::ENONET         ,
        libc::ENOPKG          => abi::ENOPKG         ,
        libc::EREMOTE         => abi::EREMOTE        ,
        libc::ENOLINK         => abi::ENOLINK        ,
        libc::EADV            => abi::EADV           ,
        libc::ESRMNT          => abi::ESRMNT         ,
        libc::ECOMM           => abi::ECOMM          ,
        libc::EPROTO          => abi::EPROTO         ,
        libc::EMULTIHOP       => abi::EMULTIHOP      ,
        libc::EDOTDOT         => abi::EDOTDOT        ,
        libc::EBADMSG         => abi::EBADMSG        ,
        libc::EOVERFLOW       => abi::EOVERFLOW      ,
        libc::ENOTUNIQ        => abi::ENOTUNIQ       ,
        libc::EBADFD          => abi::EBADFD         ,
        libc::EREMCHG         => abi::EREMCHG        ,
        libc::ELIBACC         => abi::ELIBACC        ,
        libc::ELIBBAD         => abi::ELIBBAD        ,
        libc::ELIBSCN         => abi::ELIBSCN        ,
        libc::ELIBMAX         => abi::ELIBMAX        ,
        libc::ELIBEXEC        => abi::ELIBEXEC       ,
        libc::EILSEQ          => abi::EILSEQ         ,
        libc::ERESTART        => abi::ERESTART       ,
        libc::ESTRPIPE        => abi::ESTRPIPE       ,
        libc::EUSERS          => abi::EUSERS         ,
        libc::ENOTSOCK        => abi::ENOTSOCK       ,
        libc::EDESTADDRREQ    => abi::EDESTADDRREQ   ,
        libc::EMSGSIZE        => abi::EMSGSIZE       ,
        libc::EPROTOTYPE      => abi::EPROTOTYPE     ,
        libc::ENOPROTOOPT     => abi::ENOPROTOOPT    ,
        libc::EPROTONOSUPPORT => abi::EPROTONOSUPPORT,
        libc::ESOCKTNOSUPPORT => abi::ESOCKTNOSUPPORT,
        libc::EOPNOTSUPP      => abi::EOPNOTSUPP     ,
        libc::EPFNOSUPPORT    => abi::EPFNOSUPPORT   ,
        libc::EAFNOSUPPORT    => abi::EAFNOSUPPORT   ,
        libc::EADDRINUSE      => abi::EADDRINUSE     ,
        libc::EADDRNOTAVAIL   => abi::EADDRNOTAVAIL  ,
        libc::ENETDOWN        => abi::ENETDOWN       ,
        libc::ENETUNREACH     => abi::ENETUNREACH    ,
        libc::ENETRESET       => abi::ENETRESET      ,
        libc::ECONNABORTED    => abi::ECONNABORTED   ,
        libc::ECONNRESET      => abi::ECONNRESET     ,
        libc::ENOBUFS         => abi::ENOBUFS        ,
        libc::EISCONN         => abi::EISCONN        ,
        libc::ENOTCONN        => abi::ENOTCONN       ,
        libc::ESHUTDOWN       => abi::ESHUTDOWN      ,
        libc::ETOOMANYREFS    => abi::ETOOMANYREFS   ,
        libc::ETIMEDOUT       => abi::ETIMEDOUT      ,
        libc::ECONNREFUSED    => abi::ECONNREFUSED   ,
        libc::EHOSTDOWN       => abi::EHOSTDOWN      ,
        libc::EHOSTUNREACH    => abi::EHOSTUNREACH   ,
        libc::EALREADY        => abi::EALREADY       ,
        libc::EINPROGRESS     => abi::EINPROGRESS    ,
        libc::ESTALE          => abi::ESTALE         ,
        libc::EUCLEAN         => abi::EUCLEAN        ,
        libc::ENOTNAM         => abi::ENOTNAM        ,
        libc::ENAVAIL         => abi::ENAVAIL        ,
        libc::EISNAM          => abi::EISNAM         ,
        libc::EREMOTEIO       => abi::EREMOTEIO      ,
        libc::EDQUOT          => abi::EDQUOT         ,
        libc::ENOMEDIUM       => abi::ENOMEDIUM      ,
        libc::EMEDIUMTYPE     => abi::EMEDIUMTYPE    ,
        libc::ECANCELED       => abi::ECANCELED      ,
        libc::ENOKEY          => abi::ENOKEY         ,
        libc::EKEYEXPIRED     => abi::EKEYEXPIRED    ,
        libc::EKEYREVOKED     => abi::EKEYREVOKED    ,
        libc::EKEYREJECTED    => abi::EKEYREJECTED   ,
        libc::EOWNERDEAD      => abi::EOWNERDEAD     ,
        libc::ENOTRECOVERABLE => abi::ENOTRECOVERABLE,
        libc::ERFKILL         => abi::ERFKILL        ,
        libc::EHWPOISON       => abi::EHWPOISON      ,
        libc::ENOTSUP         => abi::ENOTSUP        ,
        _ => {
            warn!(target: "syscall", "fail to translate host errno = {} to guest errno", number);
            abi::ENOSYS
        }
    }
}

#[no_mangle]
extern "C" fn convert_open_flags_to_host(flags: abi::c_int) -> libc::c_int {
    let mut ret = 0;
    if flags & 01 != 0 { ret |= libc::O_WRONLY }
    if flags & 02 != 0 { ret |= libc::O_RDWR }
    if flags & 0100 != 0 { ret |= libc::O_CREAT }
    if flags & 0200 != 0 { ret |= libc::O_EXCL }
    if flags & 01000 != 0 { ret |= libc::O_TRUNC }
    if flags & 02000 != 0 { ret |= libc::O_APPEND }
    if flags & 04000 != 0 { ret |= libc::O_NONBLOCK }
    if flags & 04010000 != 0 { ret |= libc::O_SYNC }
    ret
}

#[no_mangle]
extern "C" fn convert_stat_from_host(guest_stat: &mut abi::stat, host_stat: &libc::stat) {
    guest_stat.st_dev        = host_stat.st_dev;
    guest_stat.st_ino        = host_stat.st_ino;
    guest_stat.st_mode       = host_stat.st_mode;
    guest_stat.st_nlink      = host_stat.st_nlink as _;
    guest_stat.st_uid        = host_stat.st_uid;
    guest_stat.st_gid        = host_stat.st_gid;
    guest_stat.st_rdev       = host_stat.st_rdev;
    guest_stat.st_size       = host_stat.st_size;
    guest_stat.st_blocks     = host_stat.st_blocks;
    guest_stat.st_blksize    = host_stat.st_blksize as _;
    guest_stat.st_atime      = host_stat.st_atime;
    guest_stat.st_atime_nsec = host_stat.st_atime_nsec as _;
    guest_stat.st_mtime      = host_stat.st_mtime;
    guest_stat.st_mtime_nsec = host_stat.st_mtime_nsec as _;
    guest_stat.st_ctime      = host_stat.st_ctime;
    guest_stat.st_ctime_nsec = host_stat.st_ctime_nsec as _;
}

#[no_mangle]
extern "C" fn convert_timeval_from_host(guest_tv: &mut abi::timeval, host_tv: &libc::timeval) {
    guest_tv.tv_sec  = host_tv.tv_sec  as _;
    guest_tv.tv_usec = host_tv.tv_usec as _;
}

#[no_mangle]
extern "C" fn convert_iovec_to_host(host_iov: &mut libc::iovec, guest_iov: &abi::iovec) {
    host_iov.iov_base = guest_iov.iov_base as _;
    host_iov.iov_len  = guest_iov.iov_len  as _;
}

#[no_mangle]
extern "C" fn convert_mmap_prot_to_host(prot: abi::c_int) -> libc::c_int {
    let mut ret = 0;
    if (prot & abi::PROT_READ) != 0 { ret |= libc::PROT_READ }
    if (prot & abi::PROT_WRITE) != 0 { ret |= libc::PROT_WRITE }
    if (prot & abi::PROT_EXEC) != 0 { ret |= libc::PROT_EXEC }
    ret
}

#[no_mangle]
extern "C" fn convert_mmap_flags_to_host(flags: abi::c_int) -> libc::c_int {
    let mut ret = 0;
    if (flags & abi::MAP_SHARED) != 0 { ret |= libc::MAP_SHARED }
    if (flags & abi::MAP_PRIVATE) != 0 { ret |= libc::MAP_PRIVATE }
    if (flags & abi::MAP_FIXED) != 0 { ret |= libc::MAP_FIXED }
    if (flags & abi::MAP_ANON) != 0 { ret |= libc::MAP_ANON }
    ret
}

/// When an error occurs during a system call, Linux will return the negated value of the error
/// number. Library functions, on the other hand, usually return -1 and set errno instead.
/// Helper for converting library functions which use state variable `errno` to carry error
/// information to a linux syscall style which returns a negative value representing the errno.
#[no_mangle]
extern "C" fn return_errno(val: abi::ssize_t) -> abi::ssize_t {
    if val != -1 { return val }
    return -convert_errno_from_host(unsafe { *libc::__errno_location() }) as _;
}


extern {
    pub fn syscall(nr: u64, arg0: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> u64;
}
