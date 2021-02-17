//! This module defines the wire protocol of 9P, including routines for serialize to or deserialize
//! from the wire binary format.
use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use std::io::{Read, Result, Write};
use std::time::{Duration, SystemTime};

pub const P9_QTDIR: u8 = 0x80;
pub const P9_QTSYMLINK: u8 = 0x02;
pub const P9_QTFILE: u8 = 0x00;

#[derive(Debug)]
pub struct Qid {
    pub r#type: u8,
    pub version: u32,
    pub path: u64,
}

#[derive(Debug)]
pub struct DirEntry {
    pub qid: Qid,
    pub offset: u64,
    pub r#type: u8,
    pub name: String,
}

#[derive(Debug)]
pub struct Stat {
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub nlink: u64,
    pub rdev: u64,
    pub size: u64,
    pub blksize: u64,
    pub blocks: u64,
    pub atime: SystemTime,
    pub mtime: SystemTime,
    pub ctime: SystemTime,
}

#[derive(Debug)]
pub struct SetAttr {
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    pub atime: SystemTime,
    pub mtime: SystemTime,
}

#[derive(Debug)]
pub struct StatFs {
    pub r#type: u32,
    pub bsize: u32,
    pub blocks: u64,
    pub bfree: u64,
    pub bavail: u64,
    pub files: u64,
    pub ffree: u64,
    pub fsid: u64,
    pub namelen: u32,
}

#[derive(Debug)]
pub enum Fcall {
    Rlerror {
        ecode: u32,
    },
    Tstatfs {
        fid: u32,
    },
    Rstatfs {
        stat: StatFs,
    },
    Tlopen {
        fid: u32,
        flags: u32,
    },
    Rlopen {
        qid: Qid,
        iounit: u32,
    },
    Tlcreate {
        fid: u32,
        name: String,
        flags: u32,
        mode: u32,
        gid: u32,
    },
    Rlcreate {
        qid: Qid,
        iounit: u32,
    },
    Tsymlink {
        fid: u32,
        name: String,
        symtgt: String,
        gid: u32,
    },
    Rsymlink {
        qid: Qid,
    },
    Tmknod {
        dfid: u32,
        name: String,
        mode: u32,
        major: u32,
        minor: u32,
        gid: u32,
    },
    Rmknod {
        qid: Qid,
    },
    // Rename is not supported as we support Renameat
    Treadlink {
        fid: u32,
    },
    Rreadlink {
        target: String,
    },
    Tgetattr {
        fid: u32,
        request_mask: u64,
    },
    Rgetattr {
        valid: u64,
        qid: Qid,
        stat: Stat,
    },
    Tsetattr {
        fid: u32,
        valid: u32,
        attr: SetAttr,
    },
    Rsetattr {},
    Txattrwalk {
        fid: u32,
        newfid: u32,
        name: String,
    },
    Rxattrwalk {
        size: u64,
    },
    Txattrcreate {
        fid: u32,
        name: String,
        attr_size: u64,
        flags: u32,
    },
    Rxattrcreate {},
    Treaddir {
        fid: u32,
        offset: u64,
        count: u32,
    },
    Rreaddir {
        count: u32,
        data: Vec<DirEntry>,
    },
    Tfsync {
        fid: u32,
    },
    Rfsync {},
    Tlock {
        fid: u32,
        r#type: u8,
        flags: u32,
        start: u64,
        length: u64,
        proc_id: u32,
        client_id: String,
    },
    Rlock {
        status: u8,
    },
    Tgetlock {
        fid: u32,
        r#type: u8,
        start: u64,
        length: u64,
        proc_id: u64,
        client_id: String,
    },
    Rgetlock {
        r#type: u8,
        start: u64,
        length: u64,
        proc_id: u32,
        client_id: String,
    },
    Tlink {
        dfid: u32,
        fid: u32,
        name: String,
    },
    Rlink {},
    Tmkdir {
        dfid: u32,
        name: String,
        mode: u32,
        gid: u32,
    },
    Rmkdir {
        qid: Qid,
    },
    Trenameat {
        olddirfid: u32,
        oldname: String,
        newdirfid: u32,
        newname: String,
    },
    Rrenameat {},
    Tunlinkat {
        dirfd: u32,
        name: String,
        flags: u32,
    },
    Runlinkat {},

    Tauth {
        afid: u32,
        uname: String,
        aname: String,
        n_uname: u32,
    },
    Rauth {
        aqid: Qid,
    },
    Tattach {
        fid: u32,
        afid: u32,
        uname: String,
        aname: String,
        n_uname: u32,
    },
    Rattach {
        qid: Qid,
    },

    Tversion {
        msize: u32,
        version: String,
    },
    Rversion {
        msize: u32,
        version: String,
    },
    Tflush {
        oldtag: u16,
    },
    Rflush {},
    Twalk {
        fid: u32,
        newfid: u32,
        wnames: Vec<String>,
    },
    Rwalk {
        wqids: Vec<Qid>,
    },
    Tread {
        fid: u32,
        offset: u64,
        count: u32,
    },
    Rread {
        data: Vec<u8>,
    },
    Twrite {
        fid: u32,
        offset: u64,
        data: Vec<u8>,
    },
    Rwrite {
        count: u32,
    },
    Tclunk {
        fid: u32,
    },
    Rclunk {},
    // Remove is not supported as we support Unlinkat
    Unknown {},
}

fn read_exact(reader: &mut dyn Read, size: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(size);
    unsafe { bytes.set_len(size) }
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

pub trait Serializable
where
    Self: Sized,
{
    fn decode(reader: &mut dyn Read) -> Result<Self>;
    fn encode(&self, writer: &mut dyn Write) -> Result<()>;
}

impl Serializable for u8 {
    fn decode(reader: &mut dyn Read) -> Result<Self> {
        reader.read_u8()
    }
    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_u8(*self)
    }
}

impl Serializable for u16 {
    fn decode(reader: &mut dyn Read) -> Result<Self> {
        reader.read_u16::<LE>()
    }
    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_u16::<LE>(*self)
    }
}

impl Serializable for u32 {
    fn decode(reader: &mut dyn Read) -> Result<Self> {
        reader.read_u32::<LE>()
    }
    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_u32::<LE>(*self)
    }
}

impl Serializable for u64 {
    fn decode(reader: &mut dyn Read) -> Result<Self> {
        reader.read_u64::<LE>()
    }
    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_u64::<LE>(*self)
    }
}

impl Serializable for String {
    fn decode(reader: &mut dyn Read) -> Result<Self> {
        let len = reader.read_u16::<LE>()? as usize;
        let bytes = read_exact(reader, len)?;
        match String::from_utf8(bytes) {
            Ok(v) => Ok(v),
            Err(v) => Err(std::io::Error::new(std::io::ErrorKind::InvalidData, v)),
        }
    }

    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_u16::<LE>(self.len() as u16)?;
        writer.write_all(self.as_bytes())?;
        Ok(())
    }
}

impl Serializable for Qid {
    fn decode(reader: &mut dyn Read) -> Result<Self> {
        Ok(Qid {
            r#type: reader.read_u8()?,
            version: reader.read_u32::<LE>()?,
            path: reader.read_u64::<LE>()?,
        })
    }

    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_u8(self.r#type)?;
        writer.write_u32::<LE>(self.version)?;
        writer.write_u64::<LE>(self.path)?;
        Ok(())
    }
}

impl Serializable for SystemTime {
    fn decode(reader: &mut dyn Read) -> Result<Self> {
        Ok(SystemTime::UNIX_EPOCH
            + Duration::new(reader.read_u64::<LE>()?, reader.read_u64::<LE>()? as u32))
    }

    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        let duration = self.duration_since(SystemTime::UNIX_EPOCH).unwrap();
        duration.as_secs().encode(writer)?;
        (duration.subsec_nanos() as u64).encode(writer)
    }
}

impl Serializable for Stat {
    fn decode(reader: &mut dyn Read) -> Result<Self> {
        macro_rules! decode {
            () => {
                Serializable::decode(reader)?
            };
        }
        let stat = Stat {
            mode: decode!(),
            uid: decode!(),
            gid: decode!(),
            nlink: decode!(),
            rdev: decode!(),
            size: decode!(),
            blksize: decode!(),
            blocks: decode!(),
            atime: decode!(),
            mtime: decode!(),
            ctime: decode!(),
        };
        let _: SystemTime = decode!();
        let _: u64 = decode!();
        let _: u64 = decode!();
        Ok(stat)
    }

    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        self.mode.encode(writer)?;
        self.uid.encode(writer)?;
        self.gid.encode(writer)?;
        self.nlink.encode(writer)?;
        self.rdev.encode(writer)?;
        self.size.encode(writer)?;
        self.blksize.encode(writer)?;
        self.blocks.encode(writer)?;
        self.atime.encode(writer)?;
        self.mtime.encode(writer)?;
        self.ctime.encode(writer)?;
        SystemTime::UNIX_EPOCH.encode(writer)?;
        0u64.encode(writer)?;
        0u64.encode(writer)
    }
}

impl Serializable for SetAttr {
    fn decode(reader: &mut dyn Read) -> Result<Self> {
        macro_rules! decode {
            () => {
                Serializable::decode(reader)?
            };
        }
        let stat = SetAttr {
            mode: decode!(),
            uid: decode!(),
            gid: decode!(),
            size: decode!(),
            atime: decode!(),
            mtime: decode!(),
        };
        Ok(stat)
    }

    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        self.mode.encode(writer)?;
        self.uid.encode(writer)?;
        self.gid.encode(writer)?;
        self.size.encode(writer)?;
        self.atime.encode(writer)?;
        self.mtime.encode(writer)
    }
}

impl Serializable for StatFs {
    fn decode(_reader: &mut dyn Read) -> Result<Self> {
        unimplemented!()
    }

    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        self.r#type.encode(writer)?;
        self.bsize.encode(writer)?;
        self.blocks.encode(writer)?;
        self.bfree.encode(writer)?;
        self.bavail.encode(writer)?;
        self.files.encode(writer)?;
        self.ffree.encode(writer)?;
        self.fsid.encode(writer)?;
        self.namelen.encode(writer)
    }
}

impl Serializable for (u16, Fcall) {
    fn decode(reader: &mut dyn Read) -> Result<(u16, Fcall)> {
        macro_rules! decode {
            () => {
                Serializable::decode(reader)?
            };
        }
        let msg = reader.read_u8()?;
        let tag = reader.read_u16::<LE>()?;
        Ok((
            tag,
            match msg {
                8 => Fcall::Tstatfs { fid: decode!() },
                12 => Fcall::Tlopen { fid: decode!(), flags: decode!() },
                14 => Fcall::Tlcreate {
                    fid: decode!(),
                    name: decode!(),
                    flags: decode!(),
                    mode: decode!(),
                    gid: decode!(),
                },
                16 => Fcall::Tsymlink {
                    fid: decode!(),
                    name: decode!(),
                    symtgt: decode!(),
                    gid: decode!(),
                },
                18 => Fcall::Tmknod {
                    dfid: decode!(),
                    name: decode!(),
                    mode: decode!(),
                    major: decode!(),
                    minor: decode!(),
                    gid: decode!(),
                },
                22 => Fcall::Treadlink { fid: decode!() },
                24 => Fcall::Tgetattr { fid: decode!(), request_mask: decode!() },
                26 => Fcall::Tsetattr { fid: decode!(), valid: decode!(), attr: decode!() },
                30 => Fcall::Txattrwalk { fid: decode!(), newfid: decode!(), name: decode!() },
                32 => Fcall::Txattrcreate {
                    fid: decode!(),
                    name: decode!(),
                    attr_size: decode!(),
                    flags: decode!(),
                },
                40 => Fcall::Treaddir { fid: decode!(), offset: decode!(), count: decode!() },
                50 => Fcall::Tfsync { fid: decode!() },
                52 => Fcall::Tlock {
                    fid: decode!(),
                    r#type: decode!(),
                    flags: decode!(),
                    start: decode!(),
                    length: decode!(),
                    proc_id: decode!(),
                    client_id: decode!(),
                },
                70 => Fcall::Tlink { dfid: decode!(), fid: decode!(), name: decode!() },
                72 => Fcall::Tmkdir {
                    dfid: decode!(),
                    name: decode!(),
                    mode: decode!(),
                    gid: decode!(),
                },
                74 => Fcall::Trenameat {
                    olddirfid: decode!(),
                    oldname: decode!(),
                    newdirfid: decode!(),
                    newname: decode!(),
                },
                76 => Fcall::Tunlinkat { dirfd: decode!(), name: decode!(), flags: decode!() },
                100 => Fcall::Tversion { msize: decode!(), version: decode!() },
                101 => Fcall::Rversion { msize: decode!(), version: decode!() },
                102 => Fcall::Tauth {
                    afid: decode!(),
                    uname: decode!(),
                    aname: decode!(),
                    n_uname: decode!(),
                },
                104 => Fcall::Tattach {
                    fid: decode!(),
                    afid: decode!(),
                    uname: decode!(),
                    aname: decode!(),
                    n_uname: decode!(),
                },
                108 => Fcall::Tflush { oldtag: decode!() },
                110 => Fcall::Twalk {
                    fid: decode!(),
                    newfid: decode!(),
                    wnames: {
                        let len: u16 = decode!();
                        let mut vec: Vec<String> = Vec::with_capacity(len as usize);
                        for _ in 0..len {
                            vec.push(decode!());
                        }
                        vec
                    },
                },
                116 => Fcall::Tread { fid: decode!(), offset: decode!(), count: decode!() },
                118 => Fcall::Twrite {
                    fid: reader.read_u32::<LE>()?,
                    offset: reader.read_u64::<LE>()?,
                    data: {
                        let len: u32 = decode!();
                        let mut buf = Vec::with_capacity(len as usize);
                        unsafe { buf.set_len(len as usize) }
                        reader.read_exact(&mut buf)?;
                        buf
                    },
                },
                120 => Fcall::Tclunk { fid: decode!() },
                _ => {
                    error!(target: "9p", "unimplemented mesasge type {}", msg);
                    Fcall::Unknown {}
                }
            },
        ))
    }

    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        let msg_type = match self.1 {
            Fcall::Rlerror { .. } => 7,
            Fcall::Rstatfs { .. } => 9,
            Fcall::Rlopen { .. } => 13,
            Fcall::Rlcreate { .. } => 15,
            Fcall::Rsymlink { .. } => 17,
            Fcall::Rmknod { .. } => 19,
            Fcall::Rreadlink { .. } => 23,
            Fcall::Rgetattr { .. } => 25,
            Fcall::Rsetattr { .. } => 27,
            Fcall::Rxattrwalk { .. } => 31,
            Fcall::Rxattrcreate { .. } => 33,
            Fcall::Rreaddir { .. } => 41,
            Fcall::Rfsync { .. } => 51,
            Fcall::Rlock { .. } => 53,
            Fcall::Rlink { .. } => 71,
            Fcall::Rmkdir { .. } => 73,
            Fcall::Rrenameat { .. } => 75,
            Fcall::Runlinkat { .. } => 77,
            Fcall::Rversion { .. } => 101,
            Fcall::Rauth { .. } => 103,
            Fcall::Rattach { .. } => 105,
            Fcall::Rflush { .. } => 109,
            Fcall::Rwalk { .. } => 111,
            Fcall::Rread { .. } => 117,
            Fcall::Rwrite { .. } => 119,
            Fcall::Rclunk { .. } => 121,
            _ => unimplemented!(),
        };
        writer.write_u8(msg_type)?;
        writer.write_u16::<LE>(self.0)?;
        match &self.1 {
            Fcall::Rlerror { ecode } => {
                writer.write_u32::<LE>(*ecode)?;
            }
            Fcall::Rstatfs { stat } => {
                stat.encode(writer)?;
            }
            Fcall::Rlopen { qid, iounit } | Fcall::Rlcreate { qid, iounit } => {
                qid.encode(writer)?;
                iounit.encode(writer)?;
            }
            Fcall::Rsymlink { qid } | Fcall::Rmknod { qid } | Fcall::Rmkdir { qid } => {
                qid.encode(writer)?;
            }
            Fcall::Rreadlink { target } => {
                target.encode(writer)?;
            }
            Fcall::Rgetattr { valid, qid, stat } => {
                valid.encode(writer)?;
                qid.encode(writer)?;
                stat.encode(writer)?;
            }
            Fcall::Rsetattr {} => (),
            Fcall::Rxattrwalk { size } => {
                size.encode(writer)?;
            }
            Fcall::Rxattrcreate {} => (),
            Fcall::Rreaddir { count, data } => {
                count.encode(writer)?;
                for datum in data {
                    datum.qid.encode(writer)?;
                    datum.offset.encode(writer)?;
                    datum.r#type.encode(writer)?;
                    datum.name.encode(writer)?;
                }
            }
            Fcall::Rfsync {} => (),
            Fcall::Rlock { status } => {
                status.encode(writer)?;
            }
            Fcall::Rlink {} => (),
            Fcall::Rrenameat {} => (),
            Fcall::Runlinkat {} => (),
            Fcall::Rversion { msize, version } => {
                writer.write_u32::<LE>(*msize)?;
                version.encode(writer)?;
            }
            Fcall::Rauth { aqid } => {
                aqid.encode(writer)?;
            }
            Fcall::Rattach { qid } => {
                qid.encode(writer)?;
            }
            Fcall::Rflush {} => (),
            Fcall::Rwalk { wqids } => {
                (wqids.len() as u16).encode(writer)?;
                for qid in wqids {
                    qid.encode(writer)?;
                }
            }
            Fcall::Rread { data } => {
                (data.len() as u32).encode(writer)?;
                writer.write_all(&data)?;
            }
            Fcall::Rwrite { count } => {
                count.encode(writer)?;
            }
            Fcall::Rclunk {} => (),
            _ => unimplemented!(),
        }
        Ok(())
    }
}
