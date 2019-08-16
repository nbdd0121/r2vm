use std::fmt::{self, Write};
use std::ffi::{CStr, CString};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use crate::util::RoCell;

use super::abi;

static mut ORIGINAL_BRK: u64 = 0;
static mut BRK         : u64 = 0;
static mut HEAP_START  : u64 = 0;
static mut HEAP_END    : u64 = 0;

/// A flag to determine whether to trace all system calls. If true then all guest system calls will be logged.
pub static STRACE: RoCell<bool> = RoCell::new(false);

/// The actual path of the executable. Needed to redirect /proc/self/*
pub static EXEC_PATH: RoCell<CString> = unsafe { RoCell::new_uninit() };

/// Path of sysroot. When the guest application tries to open a file, and the corresponding file exists in sysroot,
/// it will be redirected.
pub static SYSROOT: RoCell<PathBuf> = unsafe { RoCell::new_uninit() };

/// Initialise brk when program is being loaded. Should not be called after execution has started.
pub unsafe fn init_brk(brk: u64) {
    ORIGINAL_BRK = brk;
    BRK          = brk;
    HEAP_START   = brk;
    HEAP_END     = brk;
}

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

fn convert_open_flags_to_host(flags: abi::c_int) -> libc::c_int {
    let mut ret = 0;
    if flags & 0o1 != 0 { ret |= libc::O_WRONLY }
    if flags & 0o2 != 0 { ret |= libc::O_RDWR }
    if flags & 0o100 != 0 { ret |= libc::O_CREAT }
    if flags & 0o200 != 0 { ret |= libc::O_EXCL }
    if flags & 0o1000 != 0 { ret |= libc::O_TRUNC }
    if flags & 0o2000 != 0 { ret |= libc::O_APPEND }
    if flags & 0o4000 != 0 { ret |= libc::O_NONBLOCK }
    if flags & 0o4010000 != 0 { ret |= libc::O_SYNC }
    ret
}

fn convert_stat_from_host(guest_stat: &mut abi::stat, host_stat: &libc::stat) {
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

fn convert_mmap_prot_to_host(prot: abi::c_int) -> libc::c_int {
    let mut ret = 0;
    if (prot & abi::PROT_READ) != 0 { ret |= libc::PROT_READ }
    if (prot & abi::PROT_WRITE) != 0 { ret |= libc::PROT_WRITE }
    // Guest code isn't directly executable, map them to PROT_READ instead.
    if (prot & abi::PROT_EXEC) != 0 { ret |= libc::PROT_READ }
    ret
}

fn convert_mmap_flags_to_host(flags: abi::c_int) -> libc::c_int {
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
fn return_errno(val: i64) -> i64 {
    if val != -1 { return val }
    -convert_errno_from_host(unsafe { *libc::__errno_location() }) as _
}

/// Detect whether the path is referencing /proc/self/ or friends.
/// returns None if the path does not match /proc/self/, and return the remaining part if it matches.
fn is_proc_self(path: &CStr) -> Option<&Path> {
    let path = match path.to_str() {
        Err(_) => return None,
        Ok(v) => Path::new(v),
    };
    let path = match path.strip_prefix("/proc") {
        Err(_) => return None,
        Ok(v) => v,
    };
    if let Ok(v) = path.strip_prefix("self") {
        return Some(v)
    }
    // We still need to check /proc/pid
    let pid = format!("{}", std::process::id());
    match path.strip_prefix(pid) {
        Ok(v) => Some(v),
        Err(_) => None,
    }
}

pub fn translate_path(path: &Path) -> Cow<Path> {
    // We assume relative paths cannot point to sysroot
    if path.is_relative() { return Cow::Borrowed(path) }
    // TODO: Replace this once we get proper type for sysroot
    let newpath = SYSROOT.join(path.strip_prefix("/").unwrap());
    if !newpath.exists() { return Cow::Borrowed(path) }
    if *STRACE {
        eprintln!("Translate {} to {}", path.display(), newpath.display());
    }
    Cow::Owned(newpath)
}

/// Convert a guest path to actual path. When guest is accessing some files in sysroot, this
/// step is necessary.
fn translate_path_cstr(pathname: &CStr) -> Cow<CStr> {
    let path = match pathname.to_str() {
        // Just return as is if the path cannot be converted to str
        Err(_) => return Cow::Borrowed(pathname),
        Ok(v) => Path::new(v),
    };
    match translate_path(path) {
        Cow::Borrowed(_) => Cow::Borrowed(pathname),
        Cow::Owned(v) => Cow::Owned(CString::new(v.into_os_string().into_string().unwrap()).unwrap()),
    }
}

pub unsafe fn syscall(nr: u64, arg0: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> u64 {
    let ret: i64 = match nr as i64 {
        abi::SYS_getcwd => {
            let buffer = arg0 as *mut i8;
            let size = arg1 as usize;
            let ret = if libc::getcwd(buffer, size).is_null() { -abi::EINVAL } else { 0 };
            if *STRACE {
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
            ret as _
        }
        abi::SYS_unlinkat => {
            let arg0 = arg0 as abi::c_int;
            let dirfd: i32 = if arg0 == abi::AT_FDCWD { libc::AT_FDCWD } else { arg0 as _ };
            let pathname = CStr::from_ptr(arg1 as usize as _);
            let ret = return_errno(libc::unlinkat(dirfd, translate_path_cstr(pathname).as_ptr(), arg2 as _) as _);
            if *STRACE {
                eprintln!(
                    "unlinkat({}, {}, {}) = {}", arg0, Escape(pathname.to_bytes()), arg2, ret
                );
            }
            ret
        }
        abi::SYS_faccessat => {
            let arg0 = arg0 as abi::c_int;
            let dirfd: i32 = if arg0 == abi::AT_FDCWD { libc::AT_FDCWD } else { arg0 as _ };
            let pathname = CStr::from_ptr(arg1 as usize as _);
            let ret = return_errno(libc::faccessat(
                dirfd,
                translate_path_cstr(pathname).as_ptr(),
                arg2 as _,
                arg3 as _
            ) as _);
            if *STRACE {
                eprintln!(
                    "faccessat({}, {}, {}, {}) = {}", arg0, Escape(pathname.to_bytes()), arg2, arg3, ret
                );
            }
            ret
        }
        abi::SYS_openat => {
            let arg0 = arg0 as abi::c_int;
            let arg3 = arg0 as libc::mode_t;
            let dirfd: i32 = if arg0 == abi::AT_FDCWD { libc::AT_FDCWD } else { arg0 as _ };
            let pathname = CStr::from_ptr(arg1 as usize as _);
            let flags = convert_open_flags_to_host(arg2 as _);
            let proc_self = is_proc_self(pathname);
            let ret = match proc_self {
                Some(v) if v == std::ffi::OsStr::new("exe") => {
                    libc::openat(dirfd, EXEC_PATH.as_ptr(), flags, arg3) as _
                }
                _ => return_errno(libc::openat(dirfd, translate_path_cstr(pathname).as_ptr(), flags, arg3) as _),
            };
            if *STRACE {
                eprintln!(
                    "openat({}, {}, {}, {}) = {}", arg0, Escape(pathname.to_bytes()), arg2, arg3, ret
                );
            }
            ret
        }
        abi::SYS_close => {
            // Handle standard IO specially, pretending close is sucessful.
            let ret = if arg0 <= 2 {
                0
            } else {
                return_errno(libc::close(arg0 as _) as _)
            };
            if *STRACE {
                eprintln!("close({}) = {}", arg0, ret);
            }
            ret
        }
        abi::SYS_lseek => {
            let ret = return_errno(libc::lseek(arg0 as _, arg1 as _, arg2 as _));
            if *STRACE {
                eprintln!("lseek({}, {}, {}) = {}", arg0, arg1, arg2, ret);
            }
            ret
        }
        abi::SYS_read => {
            let buffer = arg1 as usize as _;
            let ret = return_errno(libc::read(arg0 as _, buffer, arg2 as _) as _);
            if *STRACE {
                eprintln!("read({}, {}, {}) = {}",
                    arg0,
                    Escape(std::slice::from_raw_parts(buffer as _, if ret >= 0 { ret as usize } else { arg2 as usize })),
                    arg2,
                    ret
                );
            }
            ret
        }
        abi::SYS_write => {
            let buffer = arg1 as usize as _;
            let ret = return_errno(libc::write(arg0 as _, buffer, arg2 as _) as _);
            if *STRACE {
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
            if *STRACE {
                eprintln!("writev({}, {}, {}) = {}",
                    arg0,
                    arg1,
                    arg2,
                    ret
                );
            }
            ret
        }
        abi::SYS_readlinkat => {
            let arg0 = arg0 as abi::c_int;
            let dirfd: i32 = if arg0 == abi::AT_FDCWD { libc::AT_FDCWD } else { arg0 as _ };
            let pathname = CStr::from_ptr(arg1 as usize as _);
            let buffer = arg2 as usize as *mut i8;
            let proc_self = is_proc_self(pathname);
            let ret = match proc_self {
                Some(v) if v == std::ffi::OsStr::new("exe") => {
                    let path = libc::realpath(EXEC_PATH.as_ptr(), std::ptr::null_mut());
                    if path.is_null() {
                        return_errno(-1)
                    } else {
                        libc::strncpy(buffer, path, arg3 as _);
                        let ret = libc::strlen(path);
                        libc::free(path as *mut _);
                        ret as i64
                    }
                }
                _ => return_errno(libc::readlinkat(dirfd, translate_path_cstr(pathname).as_ptr(), buffer, arg3 as _) as _)
            };
            if *STRACE {
                if ret > 0 {
                    eprintln!(
                        "readlinkat({}, {}, {}, {}) = {}",
                        arg0,
                        Escape(pathname.to_bytes()),
                        Escape(CStr::from_ptr(buffer).to_bytes()), arg3, ret
                    );
                } else {
                    eprintln!("readlinkat({}, {}, {:#x}, {}) = {}", arg0, Escape(pathname.to_bytes()), arg2, arg3, ret);
                }
            }
            ret
        }
        abi::SYS_fstatat => {
            let arg0 = arg0 as abi::c_int;
            let dirfd: i32 = if arg0 == abi::AT_FDCWD { libc::AT_FDCWD } else { arg0 as _ };
            let pathname = CStr::from_ptr(arg1 as usize as _);

            let mut host_stat = std::mem::uninitialized();
            let ret = return_errno(libc::fstatat(
                dirfd,
                translate_path_cstr(pathname).as_ptr(),
                &mut host_stat,
                arg3 as _
            ) as _);

            // When success, convert stat format to guest format.
            if ret == 0 {
                let guest_stat = &mut *(arg2 as usize as *mut abi::stat);
                convert_stat_from_host(guest_stat, &host_stat);
            }

            if *STRACE {
                if ret == 0 {
                    eprintln!(
                        "fstatat({}, {}, {{st_mode={:#o}, st_size={}, ...}}, {}) = 0",
                        arg0, Escape(pathname.to_bytes()), host_stat.st_mode, host_stat.st_size, arg3
                    );
                } else {
                    eprintln!("fstatat({}, {}, {:#x}, {}) = {}", arg0, Escape(pathname.to_bytes()), arg2, arg3, ret);
                }
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

            if *STRACE {
                if ret == 0 {
                    eprintln!("fstat({}, {{st_mode={:#o}, st_size={}, ...}}) = 0", arg0, host_stat.st_mode, host_stat.st_size);
                } else {
                    eprintln!("fstat({}, {:#x}) = {}", arg0, arg1, ret);
                }
            }
            ret
        }
        abi::SYS_exit => {
            if *STRACE {
                eprintln!("exit({}) = ?", arg0);
            }
            crate::shutdown(crate::ExitReason::Exit(arg0 as i32));
            0
        }
        abi::SYS_exit_group => {
            if *STRACE {
                eprintln!("exit_group({}) = ?", arg0);
            }
            crate::shutdown(crate::ExitReason::Exit(arg0 as i32));
            0
        }
        abi::SYS_uname => {
            let ret = return_errno(libc::uname(arg0 as _) as _);
            if *STRACE {
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
            if *STRACE {
                eprintln!("gettimeofday({{{}, {}}}, NULL) = 0", time.as_secs(), time.subsec_micros());
            }
            0
        }
        abi::SYS_getpid => {
            let ret = libc::getpid();
            if *STRACE {
                eprintln!("getpid() = {}", ret);
            }
            ret as i64
        }
        abi::SYS_getppid => {
            let ret = libc::getppid();
            if *STRACE {
                eprintln!("getppid() = {}", ret);
            }
            ret as i64
        }
        abi::SYS_getuid => {
            let ret = libc::getuid();
            if *STRACE {
                eprintln!("getuid() = {}", ret);
            }
            ret as i64
        }
        abi::SYS_geteuid => {
            let ret = libc::geteuid();
            if *STRACE {
                eprintln!("geteuid() = {}", ret);
            }
            ret as i64
        }
        abi::SYS_getgid => {
            let ret = libc::getgid();
            if *STRACE {
                eprintln!("getgid() = {}", ret);
            }
            ret as i64
        }
        abi::SYS_getegid => {
            let ret = libc::getegid();
            if *STRACE {
                eprintln!("getegid() = {}", ret);
            }
            ret as i64
        }
        abi::SYS_brk => {
            if arg0 < ORIGINAL_BRK {
                // Cannot reduce beyond original_brk
            } else if arg0 <= HEAP_END {
                if arg0 > BRK {
                    libc::memset(BRK as usize as _, 0, (arg0 - BRK) as _);
                }
                BRK = arg0;
            } else {
                let new_heap_end = std::cmp::max(HEAP_START, (arg0 + 4095) &! 4095);

                // The heap needs to be expanded
                let addr = libc::mmap(
                    HEAP_END as _, (new_heap_end - HEAP_END) as _,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
                    -1, 0
                ) as isize as i64;

                if addr == -1 {
                    // We failed to expand the brk
                } else {
                    // Memory should be zeroed here as this is expected by glibc.
                    libc::memset(BRK as usize as _, 0, (HEAP_END - BRK) as _);
                    HEAP_END = new_heap_end;
                    BRK = arg0;
                }
            }
            if *STRACE {
                eprintln!("brk({}) = {}", Pointer(arg0), Pointer(BRK));
            }
            BRK as i64
        }
        abi::SYS_munmap => {
            let ret = return_errno(libc::munmap(arg0 as _, arg1 as _) as _);
            if *STRACE {
                eprintln!("munmap({:#x}, {}) = {}", arg0, arg1, ret);
            }
            ret
        }
        // This is linux specific call, we will just return ENOSYS.
        abi::SYS_mremap => {
            if *STRACE {
                eprintln!("mremap({}, {}, {}, {}, {:#x}) = -ENOSYS", arg0, arg1, arg2, arg3, arg4);
            }
            -abi::ENOSYS as i64
        }
        abi::SYS_mmap => {
            let prot = convert_mmap_prot_to_host(arg2 as _);
            let flags = convert_mmap_flags_to_host(arg3 as _);
            let arg4 = arg4 as _;
            let ret = return_errno(libc::mmap(arg0 as _, arg1 as _, prot, flags, arg4, arg5 as _) as _);
            if *STRACE {
                eprintln!("mmap({}, {}, {}, {}, {}, {}) = {:#x}", Pointer(arg0), arg1, arg2, arg3, arg4, arg5, ret);
            }
            ret
        }
        abi::SYS_mprotect => {
            let prot = convert_mmap_prot_to_host(arg2 as _);
            let ret = return_errno(libc::mprotect(arg0 as _, arg1 as _, prot) as _);
            if *STRACE {
                eprintln!("mprotect({:#x}, {}, {}) = {:#x}", arg0, arg1, arg2, ret);
            }
            ret
        }
        abi::SYS_open => {
            let pathname = CStr::from_ptr(arg0 as usize as _);
            let flags = convert_open_flags_to_host(arg1 as _);
            let ret = return_errno(libc::open(translate_path_cstr(pathname).as_ptr(), flags, arg2 as libc::mode_t) as _);
            if *STRACE {
                eprintln!("open({}, {}, {}) = {}", Escape(pathname.to_bytes()), arg1, arg2, ret);
            }
            ret
        }
        abi::SYS_unlink => {
            let pathname = CStr::from_ptr(arg0 as usize as _);
            let ret = return_errno(libc::unlink(translate_path_cstr(pathname).as_ptr()) as _);
            if *STRACE {
                eprintln!("unlink({}) = {}", Escape(pathname.to_bytes()), ret);
            }
            ret
        }
        abi::SYS_stat => {
            let pathname = CStr::from_ptr(arg0 as usize as _);
            let mut host_stat = std::mem::uninitialized();
            let ret = return_errno(libc::stat(translate_path_cstr(pathname).as_ptr(), &mut host_stat) as _);

            // When success, convert stat format to guest format.
            if ret == 0 {
                let guest_stat = &mut *(arg1 as usize as *mut abi::stat);
                convert_stat_from_host(guest_stat, &host_stat);
            }

            if *STRACE {
                if ret == 0 {
                    eprintln!(
                        "stat({}, {{st_mode={:#o}, st_size={}, ...}}) = 0",
                        Escape(pathname.to_bytes()), host_stat.st_mode, host_stat.st_size,
                    );
                } else {
                    eprintln!("stat({}, {:#x}) = {}", Escape(pathname.to_bytes()), arg1, ret);
                }
            }
            ret
        }
        _ => {
            eprintln!("illegal syscall {}({}, {})", nr, arg0, arg1);
            -abi::ENOSYS as i64
        }
    };
    ret as u64
}
