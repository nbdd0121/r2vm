use super::Block;
use std::io::Result;
use std::os::unix::fs::FileExt;

/// Block device backed by a file.
pub struct File {
    file: std::fs::File,
    len: u64,
}

impl File {
    /// Create a new `File` with [`std::fs::File`].
    ///
    /// [`Err`] is returned if essential metadata cannot be retrieved.
    pub fn new(file: std::fs::File) -> Result<Self> {
        let len = file.metadata()?.len();
        Ok(File { file, len })
    }
}

impl Block for File {
    fn read_exact_at(&mut self, buf: &mut [u8], offset: u64) -> Result<()> {
        self.file.read_exact_at(buf, offset)
    }

    fn write_all_at(&mut self, buf: &[u8], offset: u64) -> Result<()> {
        self.file.write_all_at(buf, offset)
    }

    fn flush(&mut self) -> Result<()> {
        self.file.sync_data()
    }

    fn len(&self) -> u64 {
        self.len
    }
}
