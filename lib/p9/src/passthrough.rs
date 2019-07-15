

use super::serialize::{Qid, Stat};
use super::{FileSystem, Inode};
use std::path::{Path, PathBuf};
use std::os::unix::fs::MetadataExt;
use std::fs::ReadDir;
use std::io::Read;
use std::time::SystemTime;

pub struct File {
    path: PathBuf,
    meta: std::fs::Metadata,
    fd: Option<std::fs::File>,
    dir: Option<(ReadDir, u64)>,
}

impl Clone for File {
    fn clone(&self) -> Self {
        File {
            path: self.path.clone(),
            meta: self.meta.clone(),
            fd: None,
            dir: None,
        }
    }
}

pub struct Passthrough(PathBuf);

impl Passthrough {
    pub fn new(root: &Path) -> Passthrough {
        Passthrough(root.to_owned())
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

impl FileSystem for Passthrough {
    type File = File;

    fn attach(&mut self) -> std::io::Result<Self::File> {
        let pathbuf = self.0.clone();
        let metadata = std::fs::symlink_metadata(&pathbuf)?;
        Ok(File{ 
            path: pathbuf,
            meta: metadata,
            fd: None,
            dir: None,
        })
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
        let mut pathbuf = file.path.clone();
        pathbuf.push(path);
        let metadata = std::fs::symlink_metadata(&pathbuf)?;
        Ok(File {
            path: pathbuf,
            meta: metadata,
            fd: None,
            dir: None,
        })
    }

    fn open(&mut self, file: &mut Self::File, flags: u32) -> std::io::Result<()> {
        if file.meta.is_dir() {
            let dir = std::fs::read_dir(&file.path)?;
            file.dir = Some((dir, 0));
        } else {
            let mut fd = std::fs::OpenOptions::new();
            fd.read(true);
            if flags & 0o1 != 0 { fd.read(false).write(true); }
            if flags & 0o2 != 0 { fd.write(true); }
            if flags & 0o100 != 0 { fd.create(true); }
            if flags & 0o1000 != 0 { fd.truncate(true); }
            if flags & 0o2000 != 0 { fd.append(true); }
            let fd = fd.open(&file.path)?;
            file.fd = Some(fd);
        }
        Ok(())
    }

    fn readdir(&mut self, file: &mut Self::File, offset: u64) -> std::io::Result<Option<(String, Self::File)>> {
        if offset == 0 {
            return Ok(Some((".".to_owned(), file.clone())))
        } else if offset == 1 {
            let path = file.path.parent().unwrap_or(&file.path).to_owned();
            let meta = std::fs::symlink_metadata(&path)?;
            return Ok(Some(("..".to_owned(), File {
                path,
                meta,
                fd: None,
                dir: None
            })))
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
        Ok(Some((name, File {
            path,
            meta,
            fd: None,
            dir: None
        })))
    }
    
    fn read(&mut self, file: &mut Self::File, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
        use std::io::Seek;
        let fd = file.fd.as_mut().unwrap();
        fd.seek(std::io::SeekFrom::Start(offset))?;
        fd.read(buf)
    }
}