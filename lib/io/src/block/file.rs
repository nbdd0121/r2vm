use super::Block;
use std::io::{Error, ErrorKind, Result, Seek, SeekFrom};

#[cfg(unix)]
use std::os::unix::fs::FileExt;

#[cfg(windows)]
use std::os::windows::fs::FileExt;

/// Block device backed by a file.
pub struct File {
    file: std::fs::File,
    len: u64,
}

fn query_len(f: &mut std::fs::File) -> Result<u64> {
    if let Ok(metadata) = f.metadata() {
        if metadata.is_file() {
            return Ok(metadata.len());
        }
    }

    if let Ok(x) = f.seek(SeekFrom::End(0)) {
        if x != 0 {
            return Ok(x);
        }
    }

    Err(Error::new(ErrorKind::Other, "cannot get file size"))
}

impl File {
    /// Create a new `File` with [`std::fs::File`].
    ///
    /// [`Err`] is returned if essential metadata cannot be retrieved.
    pub fn new(mut file: std::fs::File) -> Result<Self> {
        let len = query_len(&mut file)?;
        Ok(File { file, len })
    }
}

impl Block for File {
    #[cfg(unix)]
    fn read_exact_at(&mut self, buf: &mut [u8], offset: u64) -> Result<()> {
        self.file.read_exact_at(buf, offset)
    }

    #[cfg(windows)]
    fn read_exact_at(&mut self, mut buf: &mut [u8], mut offset: u64) -> Result<()> {
        while !buf.is_empty() {
            match self.file.seek_read(buf, offset) {
                Ok(0) => {
                    return Err(Error::new(
                        ErrorKind::UnexpectedEof,
                        "failed to fill whole buffer",
                    ));
                }
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    #[cfg(unix)]
    fn write_all_at(&mut self, buf: &[u8], offset: u64) -> Result<()> {
        self.file.write_all_at(buf, offset)
    }

    #[cfg(windows)]
    fn write_all_at(&mut self, mut buf: &[u8], mut offset: u64) -> Result<()> {
        while !buf.is_empty() {
            match self.file.seek_write(buf, offset) {
                Ok(0) => {
                    return Err(Error::new(ErrorKind::WriteZero, "failed to write whole buffer"));
                }
                Ok(n) => {
                    buf = &buf[n..];
                    offset += n as u64;
                }
                Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.file.sync_data()
    }

    fn len(&self) -> u64 {
        self.len
    }
}
