//! Block devices.
//!
//! This module provides a [`Block`] trait which bridges underlying block device implementation and
//! I/O devices that behaves as HBAs.

#[cfg(feature = "block-file")]
mod file;
#[cfg(feature = "block-file")]
pub use file::File;
#[cfg(feature = "block-shadow")]
mod shadow;
#[cfg(feature = "block-shadow")]
pub use shadow::Shadow;

use std::io::Result;

/// Capability description of a block device.
#[non_exhaustive]
pub struct Capability {
    /// Size of a block for this block device.
    pub blksize: usize,

    /// Whether discard operation is supported by the block device.
    pub discard: bool,
}

impl Default for Capability {
    fn default() -> Self {
        Capability { blksize: 512, discard: false }
    }
}

/// Abstraction of a block device.
pub trait Block {
    /// Reads the exact number of byte required to fill buf from the given offset.
    ///
    /// Caller must ensure `offset` and buffer size is aligned to `blksize` queried by `capability`.
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<()>;

    /// Attempts to write an entire buffer starting from a given offset.
    ///
    /// Caller must ensure `offset` and buffer size is aligned to `blksize` queried by `capability`.
    fn write_all_at(&self, buf: &[u8], offset: u64) -> Result<()>;

    /// Attempts to write zero to a given offset.
    ///
    /// Caller must ensure `offset` and `len` is aligned to `blksize` queried by `capability`.
    fn write_zero_at(&self, offset: u64, len: usize) -> Result<()> {
        let buf = vec![0; len];
        self.write_all_at(&buf, offset)
    }

    /// Discard contents at the given offset.
    ///
    /// Caller must ensure `offset` and `len` is aligned to `blksize` queried by `capability`.
    fn discard(&self, offset: u64, len: usize) -> Result<()> {
        let _ = (offset, len);
        Ok(())
    }

    /// Flush this block device.
    fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// Return the total size of this block device.
    fn len(&self) -> u64;

    /// Return the capability
    fn capability(&self) -> Capability {
        Default::default()
    }
}
