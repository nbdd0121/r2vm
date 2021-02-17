extern crate byteorder;
#[macro_use]
extern crate log;
extern crate fnv;
extern crate libc;

mod passthrough;
pub mod serialize;

pub use passthrough::Passthrough;

use fnv::FnvHashMap;
use serialize::{DirEntry, Fcall};
use std::io::{ErrorKind, Result};

pub trait Inode: Clone {
    fn qid(&mut self) -> serialize::Qid;
    fn mode(&mut self) -> u32;
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum LockType {
    Shared = 0,
    Exclusive = 1,
    Unlock = 2,
}

pub trait FileSystem {
    type File: Inode;

    fn statfs(&mut self, file: &mut Self::File) -> Result<serialize::StatFs>;
    fn attach(&mut self) -> Result<Self::File>;

    fn readlink(&mut self, _file: &mut Self::File) -> Result<String> {
        Err(std::io::Error::from_raw_os_error(libc::ENOTSUP))
    }

    fn getattr(&mut self, file: &mut Self::File) -> Result<serialize::Stat>;
    fn walk(&mut self, file: &mut Self::File, path: &str) -> Result<Self::File>;
    fn open(&mut self, file: &mut Self::File, flags: u32) -> Result<()>;
    fn create(
        &mut self,
        file: &mut Self::File,
        name: &str,
        flags: u32,
        mode: u32,
        gid: u32,
    ) -> Result<Self::File>;
    fn symlink(
        &mut self,
        file: &mut Self::File,
        name: &str,
        symtgt: &str,
        gid: u32,
    ) -> Result<Self::File>;
    fn readdir(
        &mut self,
        file: &mut Self::File,
        offset: u64,
    ) -> Result<Option<(String, Self::File)>>;
    fn mkdir(
        &mut self,
        dir: &mut Self::File,
        name: &str,
        mode: u32,
        gid: u32,
    ) -> Result<Self::File>;
    fn renameat(
        &mut self,
        olddir: &mut Self::File,
        oldname: &str,
        newdir: &mut Self::File,
        newname: &str,
    ) -> Result<()>;
    fn unlinkat(&mut self, file: &mut Self::File, name: &str) -> Result<()>;
    fn read(&mut self, file: &mut Self::File, offset: u64, buf: &mut [u8]) -> Result<usize>;
    fn write(&mut self, file: &mut Self::File, offset: u64, buf: &[u8]) -> Result<usize>;
    fn fsync(&mut self, file: &mut Self::File) -> Result<()>;
    fn lock(&mut self, file: &mut Self::File, ty: LockType) -> Result<()>;
    fn setattr(
        &mut self,
        file: &mut Self::File,
        valid: u32,
        stat: serialize::SetAttr,
    ) -> Result<()>;
}

pub struct P9Handler<T: FileSystem> {
    pub fs: T,
    iounit: u32,
    fids: FnvHashMap<u32, <T as FileSystem>::File>,
}

const O_LARGEFILE: u32 = 0o100000;

impl<T: FileSystem> P9Handler<T> {
    pub fn new(fs: T) -> P9Handler<T> {
        P9Handler { fs, iounit: 0, fids: FnvHashMap::default() }
    }

    /// Actual fcall processing, returning Result<Fcall> to allow easier error handling.
    fn handle_fcall_internal(&mut self, fcall: Fcall) -> Result<Fcall> {
        Ok(match fcall {
            Fcall::Tlopen { fid, flags } => {
                let file = self.fids.get_mut(&fid).unwrap();
                // Set O_LARGEFILE to zero
                self.fs.open(file, flags & !O_LARGEFILE)?;
                let qid = file.qid();
                Fcall::Rlopen { qid, iounit: self.iounit }
            }
            Fcall::Tlcreate { fid, name, flags, mode, gid } => {
                let file = self.fids.get_mut(&fid).unwrap();
                // Set O_LARGEFILE to zero
                let newfile = self.fs.create(file, &name, flags & !O_LARGEFILE, mode, gid)?;
                *file = newfile;
                let qid = file.qid();
                Fcall::Rlcreate { qid, iounit: self.iounit }
            }
            Fcall::Tsymlink { fid, name, symtgt, gid } => {
                let file = self.fids.get_mut(&fid).unwrap();
                let mut newfile = self.fs.symlink(file, &name, &symtgt, gid)?;
                let qid = newfile.qid();
                Fcall::Rsymlink { qid }
            }
            Fcall::Tstatfs { fid } => {
                let file = self.fids.get_mut(&fid).unwrap();
                Fcall::Rstatfs { stat: self.fs.statfs(file)? }
            }
            Fcall::Treadlink { fid } => {
                let file = self.fids.get_mut(&fid).unwrap();
                Fcall::Rreadlink { target: self.fs.readlink(file)? }
            }
            Fcall::Tgetattr { fid, request_mask } => {
                let file = self.fids.get_mut(&fid).unwrap();
                let stat = self.fs.getattr(file)?;
                Fcall::Rgetattr { valid: request_mask, qid: file.qid(), stat }
            }
            Fcall::Tsetattr { fid, valid, attr } => {
                let file = self.fids.get_mut(&fid).unwrap();
                self.fs.setattr(file, valid, attr)?;
                Fcall::Rsetattr {}
            }
            // We don't support xattr yet.
            Fcall::Txattrwalk { .. } | Fcall::Txattrcreate { .. } => {
                return Err(std::io::Error::from_raw_os_error(libc::ENOTSUP));
            }
            Fcall::Treaddir { fid, offset, count } => {
                let mut offset = if offset == 0 { 0 } else { offset + 1 };
                let file = self.fids.get_mut(&fid).unwrap();
                let mut cur_len = 0;
                let mut vec = Vec::new();
                loop {
                    let (name, mut child) = match self.fs.readdir(file, offset)? {
                        None => break,
                        Some(it) => it,
                    };
                    if cur_len + name.len() + 24 > count as usize {
                        break;
                    }
                    cur_len += name.len() + 24;
                    vec.push(DirEntry {
                        qid: child.qid(),
                        offset: offset as u64,
                        r#type: (child.mode() >> 12) as u8,
                        name,
                    });
                    offset += 1;
                }
                Fcall::Rreaddir { count: cur_len as u32, data: vec }
            }
            Fcall::Tmkdir { dfid, name, mode, gid } => {
                let dir = self.fids.get_mut(&dfid).unwrap();
                let mut file = self.fs.mkdir(dir, &name, mode, gid)?;
                Fcall::Rmkdir { qid: file.qid() }
            }
            Fcall::Trenameat { olddirfid, oldname, newdirfid, newname } => {
                let mut olddir = self.fids.get(&olddirfid).unwrap().clone();
                let newdir = self.fids.get_mut(&newdirfid).unwrap();
                self.fs.renameat(&mut olddir, &oldname, newdir, &newname)?;
                Fcall::Rrenameat {}
            }
            Fcall::Tunlinkat { dirfd, name, .. } => {
                let dir = self.fids.get_mut(&dirfd).unwrap();
                self.fs.unlinkat(dir, &name)?;
                Fcall::Runlinkat {}
            }
            Fcall::Tversion { msize, .. } => {
                self.iounit = msize - 24;
                Fcall::Rversion { msize, version: "9P2000.L".to_owned() }
            }
            Fcall::Tattach { fid, .. } => {
                let mut file = self.fs.attach()?;
                let qid = file.qid();
                self.fids.insert(fid, file);
                Fcall::Rattach { qid }
            }
            Fcall::Tflush { .. } => Fcall::Rflush {},
            Fcall::Twalk { fid, newfid, wnames } => {
                let mut file = self.fids.get(&fid).unwrap().clone();
                let mut qids = Vec::with_capacity(wnames.len());
                for name in wnames {
                    file = self.fs.walk(&mut file, &name)?;
                    qids.push(file.qid());
                }
                self.fids.insert(newfid, file);
                Fcall::Rwalk { wqids: qids }
            }
            Fcall::Tread { fid, offset, count } => {
                let file = self.fids.get_mut(&fid).unwrap();
                let mut buf = Vec::with_capacity(count as usize);
                unsafe { buf.set_len(count as usize) }
                let len = self.fs.read(file, offset, &mut buf)?;
                buf.truncate(len);
                Fcall::Rread { data: buf }
            }
            Fcall::Twrite { fid, offset, data } => {
                let file = self.fids.get_mut(&fid).unwrap();
                let len = self.fs.write(file, offset, &data)?;
                Fcall::Rwrite { count: len as _ }
            }
            Fcall::Tfsync { fid } => {
                let file = self.fids.get_mut(&fid).unwrap();
                self.fs.fsync(file)?;
                Fcall::Rfsync {}
            }
            Fcall::Tlock { fid, r#type, flags, .. } => {
                // Check for unknown flags
                const P9_LOCK_FLAGS_BLOCK: u32 = 1;
                if flags & !P9_LOCK_FLAGS_BLOCK != 0 {
                    return Err(std::io::Error::from_raw_os_error(libc::ENOTSUP));
                }

                let ty = match r#type {
                    0 => LockType::Shared,
                    1 => LockType::Exclusive,
                    2 => LockType::Unlock,
                    _ => return Err(std::io::Error::from_raw_os_error(libc::EINVAL)),
                };

                let file = self.fids.get_mut(&fid).unwrap();

                const P9_LOCK_SUCCESS: u8 = 0;
                const P9_LOCK_BLOCKED: u8 = 1;
                const P9_LOCK_ERROR: u8 = 2;
                match self.fs.lock(file, ty) {
                    Ok(_) => Fcall::Rlock { status: P9_LOCK_SUCCESS },
                    Err(err) if err.kind() == ErrorKind::WouldBlock => {
                        Fcall::Rlock { status: P9_LOCK_BLOCKED }
                    }
                    Err(_) => Fcall::Rlock { status: P9_LOCK_ERROR },
                }
            }
            Fcall::Tclunk { fid } => {
                self.fids.remove(&fid);
                Fcall::Rclunk {}
            }
            _ => {
                error!(target: "9p", "unhandled message {:?}", fcall);
                Fcall::Rlerror { ecode: libc::ENOTSUP as u32 }
            }
        })
    }

    pub fn handle_fcall(&mut self, fcall: Fcall) -> Fcall {
        match self.handle_fcall_internal(fcall) {
            Ok(fcall) => fcall,
            Err(err) => {
                Fcall::Rlerror { ecode: err.raw_os_error().unwrap_or(libc::ENOTSUP) as u32 }
            }
        }
    }
}
