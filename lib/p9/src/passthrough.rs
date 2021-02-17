use super::serialize::{Qid, SetAttr, Stat, StatFs};
use super::{FileSystem, Inode, LockType};
use std::ffi::CString;
use std::fs::ReadDir;
use std::io::Result;
use std::io::{Read, Seek, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub struct File {
    path: PathBuf,
    meta: std::fs::Metadata,
    fd: Option<std::fs::File>,
    dir: Option<(ReadDir, u64)>,
}

impl Clone for File {
    fn clone(&self) -> Self {
        File { path: self.path.clone(), meta: self.meta.clone(), fd: None, dir: None }
    }
}

pub struct Passthrough {
    root: PathBuf,
}

impl Passthrough {
    pub fn new(root: &Path) -> std::io::Result<Passthrough> {
        Ok(Passthrough { root: root.canonicalize()? })
    }
}

impl Inode for File {
    fn qid(&mut self) -> Qid {
        Qid {
            r#type: match (self.meta.is_dir(), self.meta.is_file()) {
                (false, false) => super::serialize::P9_QTSYMLINK,
                (true, false) => super::serialize::P9_QTDIR,
                (false, true) => super::serialize::P9_QTFILE,
                (true, true) => unreachable!(),
            },
            version: 0,
            path: self.meta.ino(),
        }
    }

    fn mode(&mut self) -> u32 {
        self.meta.mode()
    }
}

fn set_mode(opt: &mut std::fs::OpenOptions, flags: u32) {
    opt.read(true);
    if flags & 0o1 != 0 {
        opt.read(false).write(true);
    }
    if flags & 0o2 != 0 {
        opt.write(true);
    }
    if flags & 0o100 != 0 {
        opt.create(true);
    }
    if flags & 0o1000 != 0 {
        opt.truncate(true);
    }
    if flags & 0o2000 != 0 {
        opt.append(true);
    }
}

impl FileSystem for Passthrough {
    type File = File;

    fn statfs(&mut self, file: &mut Self::File) -> std::io::Result<StatFs> {
        let path_str_c = CString::new(file.path.as_os_str().as_bytes()).unwrap();
        unsafe {
            let mut buf = std::mem::MaybeUninit::uninit();
            // We use statvfs over statfs because statfs's f_fsid has type fsid_t but we need to
            // return u64.
            if libc::statvfs(path_str_c.as_ptr(), buf.as_mut_ptr()) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            let buf = buf.assume_init();
            Ok(StatFs {
                r#type: 0,
                bsize: buf.f_bsize as _,
                blocks: buf.f_blocks,
                bfree: buf.f_bfree,
                bavail: buf.f_bavail,
                files: buf.f_files,
                ffree: buf.f_ffree,
                fsid: buf.f_fsid as _,
                namelen: buf.f_namemax as _,
            })
        }
    }

    fn attach(&mut self) -> std::io::Result<Self::File> {
        let pathbuf = self.root.clone();
        let metadata = std::fs::symlink_metadata(&pathbuf)?;
        Ok(File { path: pathbuf, meta: metadata, fd: None, dir: None })
    }

    fn readlink(&mut self, file: &mut Self::File) -> Result<String> {
        let path = std::fs::read_link(&file.path)?;
        Ok(path.into_os_string().into_string().unwrap())
    }

    fn getattr(&mut self, file: &mut Self::File) -> std::io::Result<Stat> {
        Ok(Stat {
            mode: file.meta.mode(),
            uid: file.meta.uid(),
            gid: file.meta.gid(),
            nlink: file.meta.nlink(),
            rdev: file.meta.rdev(),
            size: file.meta.len(),
            blksize: file.meta.blksize(),
            blocks: file.meta.blocks(),
            atime: file.meta.accessed().unwrap_or(SystemTime::UNIX_EPOCH),
            mtime: file.meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            ctime: file.meta.created().unwrap_or(SystemTime::UNIX_EPOCH),
        })
    }

    fn walk(&mut self, file: &mut Self::File, path: &str) -> std::io::Result<Self::File> {
        let pathbuf = file.path.join(path);
        let metadata = std::fs::symlink_metadata(&pathbuf)?;
        Ok(File { path: pathbuf, meta: metadata, fd: None, dir: None })
    }

    fn open(&mut self, file: &mut Self::File, flags: u32) -> std::io::Result<()> {
        if file.meta.is_dir() {
            let dir = std::fs::read_dir(&file.path)?;
            file.dir = Some((dir, 0));
        } else {
            let mut fd = std::fs::OpenOptions::new();
            set_mode(&mut fd, flags);
            let fd = fd.open(&file.path)?;
            file.fd = Some(fd);
        }
        Ok(())
    }

    fn create(
        &mut self,
        file: &mut Self::File,
        name: &str,
        flags: u32,
        mode: u32,
        _gid: u32,
    ) -> std::io::Result<Self::File> {
        let pathbuf = file.path.join(name);
        let mut fd = std::fs::OpenOptions::new();
        set_mode(&mut fd, flags);
        fd.mode(mode);
        let fd = fd.open(&pathbuf)?;
        let meta = fd.metadata()?;
        Ok(File { path: pathbuf, meta, fd: Some(fd), dir: None })
    }

    fn symlink(
        &mut self,
        file: &mut Self::File,
        name: &str,
        symtgt: &str,
        _gid: u32,
    ) -> std::io::Result<Self::File> {
        let pathbuf = file.path.join(name);
        std::os::unix::fs::symlink(symtgt, &pathbuf)?;
        let meta = std::fs::symlink_metadata(&pathbuf)?;
        Ok(File { path: pathbuf, meta, fd: None, dir: None })
    }

    fn readdir(
        &mut self,
        file: &mut Self::File,
        offset: u64,
    ) -> std::io::Result<Option<(String, Self::File)>> {
        if offset == 0 {
            return Ok(Some((".".to_owned(), file.clone())));
        } else if offset == 1 {
            let path = file.path.parent().unwrap_or(&file.path).to_owned();
            let meta = std::fs::symlink_metadata(&path)?;
            return Ok(Some(("..".to_owned(), File { path, meta, fd: None, dir: None })));
        }
        // Exclude . and ..
        let offset = offset - 2;
        let dir = file.dir.as_mut().unwrap();
        // Need to rewind
        if offset < dir.1 {
            dir.0 = std::fs::read_dir(&file.path)?;
            dir.1 = 0;
        }
        // Need to skip
        while offset > dir.1 {
            dir.0.next();
            dir.1 += 1;
        }
        dir.1 += 1;
        let entry = match dir.0.next() {
            None => return Ok(None),
            Some(v) => v?,
        };
        let name = entry.file_name().into_string().unwrap();
        let path = entry.path();
        let meta = entry.metadata()?;
        Ok(Some((name, File { path, meta, fd: None, dir: None })))
    }

    fn mkdir(
        &mut self,
        dir: &mut Self::File,
        name: &str,
        _mode: u32,
        _gid: u32,
    ) -> Result<Self::File> {
        let path = dir.path.join(name);
        std::fs::create_dir(&path)?;
        let meta = std::fs::symlink_metadata(&path)?;
        Ok(File { path, meta, fd: None, dir: None })
    }

    fn renameat(
        &mut self,
        olddir: &mut Self::File,
        oldname: &str,
        newdir: &mut Self::File,
        newname: &str,
    ) -> Result<()> {
        std::fs::rename(olddir.path.join(oldname), newdir.path.join(newname))
    }

    fn unlinkat(&mut self, dir: &mut Self::File, name: &str) -> std::io::Result<()> {
        let path = dir.path.join(name);
        if std::fs::symlink_metadata(&path)?.is_dir() {
            std::fs::remove_dir(&path)
        } else {
            std::fs::remove_file(&path)
        }
    }

    fn read(
        &mut self,
        file: &mut Self::File,
        offset: u64,
        buf: &mut [u8],
    ) -> std::io::Result<usize> {
        let fd = file.fd.as_mut().unwrap();
        fd.seek(std::io::SeekFrom::Start(offset))?;
        fd.read(buf)
    }

    fn write(&mut self, file: &mut Self::File, offset: u64, buf: &[u8]) -> std::io::Result<usize> {
        let fd = file.fd.as_mut().unwrap();
        fd.seek(std::io::SeekFrom::Start(offset))?;
        fd.write(buf)
    }

    fn fsync(&mut self, file: &mut Self::File) -> std::io::Result<()> {
        let fd = file.fd.as_mut().unwrap();
        fd.sync_data()
    }

    fn lock(&mut self, file: &mut Self::File, ty: LockType) -> std::io::Result<()> {
        let fd = file.fd.as_mut().unwrap();
        let flag = match ty {
            LockType::Shared => libc::LOCK_SH | libc::LOCK_NB,
            LockType::Exclusive => libc::LOCK_EX | libc::LOCK_NB,
            LockType::Unlock => libc::LOCK_UN,
        };
        if unsafe { libc::flock(fd.as_raw_fd(), flag) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    fn setattr(&mut self, file: &mut Self::File, valid: u32, stat: SetAttr) -> std::io::Result<()> {
        if valid & 0x0000_0001 != 0 {
            std::fs::set_permissions(&file.path, PermissionsExt::from_mode(stat.mode))?;
        }

        if valid & (0x0000_0002 | 0x0000_0004) != 0 {
            // TODO
        }

        if valid & 0x0000_0008 != 0 {
            std::fs::OpenOptions::new().write(true).open(&file.path)?.set_len(stat.size)?;
        }

        if valid & (0x0000_0010 | 0x0000_0020) != 0 {
            let atime = (if valid & 0x0000_0080 != 0 {
                stat.atime
            } else if valid & 0x0000_0010 != 0 {
                SystemTime::now()
            } else {
                file.meta.accessed()?
            })
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap();

            let mtime = (if valid & 0x0000_0100 != 0 {
                stat.mtime
            } else if valid & 0x0000_0020 != 0 {
                SystemTime::now()
            } else {
                file.meta.modified()?
            })
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap();

            let path_str_c = CString::new(file.path.as_os_str().as_bytes()).unwrap();
            let time = [
                libc::timeval { tv_sec: atime.as_secs() as _, tv_usec: atime.subsec_micros() as _ },
                libc::timeval { tv_sec: mtime.as_secs() as _, tv_usec: mtime.subsec_micros() as _ },
            ];
            if unsafe { libc::utimes(path_str_c.as_ptr(), time.as_ptr()) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }

        // Update metadata
        file.meta = std::fs::symlink_metadata(&file.path)?;

        Ok(())
    }
}
