extern crate byteorder;
#[macro_use]
extern crate log;
extern crate fnv;
extern crate libc;

pub mod serialize;
mod passthrough;

pub use passthrough::Passthrough;

use std::io::Result;
use fnv::FnvHashMap;
use serialize::{DirEntry, Fcall};

pub trait Inode: Clone {
    fn qid(&mut self) -> serialize::Qid;
    fn mode(&mut self) -> u32;
}

pub trait FileSystem {
    type File: Inode;

    fn attach(&mut self) -> Result<Self::File>;
    fn getattr(&mut self, file: &mut Self::File) -> Result<serialize::Stat>;
    fn walk(&mut self, file: &mut Self::File, path: &str) -> Result<Self::File>;
    fn open(&mut self, file: &mut Self::File, flags: u32) -> Result<()>;
    fn readdir(&mut self, file: &mut Self::File, offset: u64) -> Result<Option<(String, Self::File)>>;
    fn read(&mut self, file: &mut Self::File, offset: u64, buf: &mut [u8]) -> Result<usize>;
}

pub struct P9Handler<T: FileSystem> {
    pub fs: T,
    iounit: u32,
    fids: FnvHashMap<u32, <T as FileSystem>::File>,
}


impl<T: FileSystem> P9Handler<T> {
    pub fn new(fs: T) -> P9Handler<T> {
        P9Handler {
            fs,
            iounit: 0,
            fids: FnvHashMap::default(),
        }
    }

    /// Actual fcall processing, returning Result<Fcall> to allow easier error handling.
    fn handle_fcall_internal(&mut self, fcall: Fcall) -> Result<Fcall> {
        Ok(match fcall {
            Fcall::Tlopen { fid, flags } => {
                let file = self.fids.get_mut(&fid).unwrap();
                // Set O_LARGEFILE to zero
                self.fs.open(file, flags &! 0o100000)?;
                let qid = file.qid();
                Fcall::Rlopen { qid, iounit: self.iounit }
            }
            Fcall::Tgetattr { fid, request_mask } => {
                let file = self.fids.get_mut(&fid).unwrap();
                let stat = self.fs.getattr(file)?;
                Fcall::Rgetattr { valid: request_mask, qid: file.qid(), stat }
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
                        break
                    }
                    cur_len += name.len() + 24;
                    vec.push(DirEntry {
                        qid: child.qid(),
                        offset: offset as u64,
                        r#type: (child.mode() >> 12) as u8,
                        name: name,
                    });
                    offset += 1;
                }
                Fcall::Rreaddir { count: cur_len as u32, data: vec }
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
            Fcall::Tflush {..} => Fcall::Rflush {},
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
            Err(err) => Fcall::Rlerror { ecode: err.raw_os_error().unwrap_or(libc::ENOTSUP) as u32 },
        }
    }
}


