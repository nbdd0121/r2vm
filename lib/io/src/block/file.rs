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

#[cfg(windows)]
fn get_disk_len(f: &mut std::fs::File) -> Result<u64> {
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::Ioctl::{GET_LENGTH_INFORMATION, IOCTL_DISK_GET_LENGTH_INFO};

    let handle = HANDLE(f.as_raw_handle() as _);
    let mut out_buffer: GET_LENGTH_INFORMATION = Default::default();
    let mut bytes_returned = 0;

    let ret = unsafe {
        windows::Win32::System::IO::DeviceIoControl(
            handle,
            IOCTL_DISK_GET_LENGTH_INFO,
            None,
            0,
            Some(&mut out_buffer as *mut GET_LENGTH_INFORMATION as _),
            std::mem::size_of::<GET_LENGTH_INFORMATION>() as _,
            Some(&mut bytes_returned),
            None,
        )
        .as_bool()
    };

    if !ret {
        return Err(std::io::Error::last_os_error());
    }

    Ok(out_buffer.Length as _)
}

fn query_len(f: &mut std::fs::File) -> Result<u64> {
    if let Ok(metadata) = f.metadata() {
        if metadata.is_file() {
            return Ok(metadata.len());
        }
    }

    if let Ok(len) = f.seek(SeekFrom::End(0)) {
        if len != 0 {
            return Ok(len);
        }
    }

    #[cfg(windows)]
    if let Ok(len) = get_disk_len(f) {
        return Ok(len);
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
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        self.file.read_exact_at(buf, offset)
    }

    #[cfg(windows)]
    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> Result<()> {
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
    fn write_all_at(&self, buf: &[u8], offset: u64) -> Result<()> {
        self.file.write_all_at(buf, offset)
    }

    #[cfg(windows)]
    fn write_all_at(&self, mut buf: &[u8], mut offset: u64) -> Result<()> {
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

    fn flush(&self) -> Result<()> {
        self.file.sync_data()
    }

    fn len(&self) -> u64 {
        self.len
    }
}
