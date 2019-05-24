use std::fmt::{self, Write};
use super::abi;

struct Escape<'a>(&'a [u8]);

impl<'a> fmt::Display for Escape<'a> {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut start = 0;
        let end = std::cmp::min(self.0.len(), 64);

        f.write_char('"')?;

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

#[allow(unreachable_patterns)]
fn convert_errno_from_host(number: libc::c_int) -> abi::c_int {
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

fn convert_iovec_to_host(guest_iov: &abi::iovec) -> libc::iovec {
    libc::iovec {
        iov_base: guest_iov.iov_base as _,
        iov_len : guest_iov.iov_len as _,
}
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
extern "C" fn return_errno(val: i64) -> u64 {
    if val != -1 { return val as u64 }
    return (-convert_errno_from_host(unsafe { *libc::__errno_location() })) as _;
}


extern {
    pub fn legacy_syscall(nr: u64, arg0: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> u64;
}

/// In this project, we consider everything that guest can reasonably do as "safe".
pub unsafe fn syscall(nr: u64, arg0: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> u64 {
    match nr as i64 {
        abi::SYS_getcwd => {
            let buffer = arg0 as *mut i8;
            let size = arg1 as usize;
            let ret = if libc::getcwd(buffer, size) != std::ptr::null_mut() { 0 } else { -abi::EINVAL };
            if crate::get_flags().strace {
                if ret == 0 {
                    eprintln!(
                        "getcwd({}, {}) = 0",
                        Escape(std::slice::from_raw_parts(buffer as _, libc::strlen(buffer))),
                        size
                    );
                } else {
                    eprintln!("getcwd({}, {}) = {}", Pointer(arg0), size, ret);
                }
            }
            ret as u64
        }
        abi::SYS_close => {
            // Handle standard IO specially, pretending close is sucessful.
            let ret = if arg0 <= 2 {
                0
            } else {
                return_errno(libc::close(arg0 as _) as _)
            };
            if crate::get_flags().strace {
                eprintln!("close({}) = {}", arg0, ret);
            }
            ret
        }
        abi::SYS_lseek => {
            let ret = return_errno(libc::lseek(arg0 as _, arg1 as _, arg2 as _));
            if crate::get_flags().strace {
                eprintln!("lseek({}, {}, {}) = {}", arg0, arg1, arg2, ret);
            }
            ret
        }
        abi::SYS_read => {
            let buffer = arg1 as usize as _;
            let ret = return_errno(libc::read(arg0 as _, buffer, arg2 as _) as _);
            if crate::get_flags().strace {
                eprintln!("read({}, {}, {}) = {}",
                    arg0,
                    Escape(std::slice::from_raw_parts(buffer as _, arg2 as usize)),
                    arg2,
                    ret
                );
            }
            ret
        }
        abi::SYS_write => {
            let buffer = arg1 as usize as _;
            let ret = return_errno(libc::write(arg0 as _, buffer, arg2 as _) as _);
            if crate::get_flags().strace {
                eprintln!("write({}, {}, {}) = {}",
                    arg0,
                    Escape(std::slice::from_raw_parts(buffer as _, arg2 as usize)),
                    arg2,
                    ret
                );
            }
            ret
        }
        abi::SYS_writev => {
            let guest_iov = std::slice::from_raw_parts(arg1 as usize as *const abi::iovec, arg2 as _);
            let host_iov: Vec<_> = guest_iov.iter().map(convert_iovec_to_host).collect();
            let ret = return_errno(libc::writev(arg0 as _, host_iov.as_ptr(), arg2 as _) as _);
            if crate::get_flags().strace {
                eprintln!("writev({}, {}, {}) = {}",
                    arg0,
                    arg1,
                    arg2,
                    ret
                );
            }
            ret
        }
        abi::SYS_fstat => {
            let mut host_stat = std::mem::uninitialized(); 
            let ret = return_errno(libc::fstat(arg0 as _, &mut host_stat) as _);

            // When success, convert stat format to guest format.
            if ret == 0 {
                let guest_stat = &mut *(arg1 as usize as *mut abi::stat);
                convert_stat_from_host(guest_stat, &host_stat);
            }

            if crate::get_flags().strace {
                if ret == 0 {
                    eprintln!("fstat({}, {{st_mode={:#o}, st_size={}, ...}}) = 0", arg0, host_stat.st_mode, host_stat.st_size);
                } else {
                    eprintln!("fstat({}, {:#x}) = {}", arg0, arg1, ret);
                }
            }

            return ret;
        }
        abi::SYS_exit => {
            if crate::get_flags().strace {
                eprintln!("exit({}) = ?", arg0);
            }
            std::process::exit(arg0 as i32)
        }
        abi::SYS_exit_group => {
            if crate::get_flags().strace {
                eprintln!("exit_group({}) = ?", arg0);
            }
            std::process::exit(arg0 as i32)
        }
        abi::SYS_uname => {
            let ret = return_errno(libc::uname(arg0 as _) as _);
            if crate::get_flags().strace {
                eprintln!("uname({:#x}) = {}", arg0, ret);
            }
            ret
        }
        abi::SYS_gettimeofday => {
            use std::time::SystemTime;
            let time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
            let guest_tv = &mut *(arg0 as usize as *mut abi::timeval);
            guest_tv.tv_sec = time.as_secs() as _;
            guest_tv.tv_usec = time.subsec_micros() as _;
            if crate::get_flags().strace {
                eprintln!("gettimeofday({{{}, {}}}, NULL) = 0", time.as_secs(), time.subsec_micros());
            }
            0
        }
        abi::SYS_getpid => {
            let ret = libc::getpid() as u64;
            if crate::get_flags().strace {
                eprintln!("getpid() = {}", ret);
            }
            ret
        }
        abi::SYS_getppid => {
            let ret = libc::getppid() as u64;
            if crate::get_flags().strace {
                eprintln!("getppid() = {}", ret);
            }
            ret
        }
        abi::SYS_getuid => {
            let ret = libc::getuid() as u64;
            if crate::get_flags().strace {
                eprintln!("getuid() = {}", ret);
            }
            ret
        }
        abi::SYS_geteuid => {
            let ret = libc::geteuid() as u64;
            if crate::get_flags().strace {
                eprintln!("geteuid() = {}", ret);
            }
            ret
        }
        abi::SYS_getgid => {
            let ret = libc::getgid() as u64;
            if crate::get_flags().strace {
                eprintln!("getgid() = {}", ret);
            }
            ret
        }
        abi::SYS_getegid => {
            let ret = libc::getegid() as u64;
            if crate::get_flags().strace {
                eprintln!("getegid() = {}", ret);
            }
            ret
        }
        _ => legacy_syscall(nr, arg0, arg1, arg2, arg3, arg4, arg5),
    }
}
