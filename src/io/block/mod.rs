mod shadow;
pub use shadow::Shadow;

use std::io::Result;
use std::fs::File;
use std::os::unix::fs::FileExt;

pub trait Block {
    fn read_exact_at(&mut self, buf: &mut [u8], offset: u64) -> Result<()>;
    fn write_all_at(&mut self, buf: &[u8], offset: u64) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
    fn len(&mut self) -> Result<u64>;
}

impl Block for File {
    fn read_exact_at(&mut self, buf: &mut [u8], offset: u64) -> Result<()> {
        FileExt::read_exact_at(self, buf, offset)
    }

    fn write_all_at(&mut self, buf: &[u8], offset: u64) -> Result<()> {
        FileExt::write_all_at(self, buf, offset)
    }

    fn flush(&mut self) -> Result<()> {
        self.sync_data()
    }

    fn len(&mut self) -> Result<u64> {
        Ok(self.metadata()?.len())
    }
}
