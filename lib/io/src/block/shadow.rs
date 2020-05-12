use super::{Block, Capability};
use fnv::FnvHashMap;
use std::io::Result;

/// A shadow block device that captures all write requests to the underlying block device.
///
/// All modified data will be kept in memory and not forwarded to the underlying block device.
pub struct Shadow {
    underlying: Box<dyn Block + Send>,
    overlay: FnvHashMap<u64, Box<[u8]>>,
}

impl Shadow {
    /// Construct a new `Shadow`.
    pub fn new(underlying: impl Block + Send + 'static) -> Self {
        Shadow { underlying: Box::new(underlying), overlay: FnvHashMap::default() }
    }
}

impl Block for Shadow {
    fn read_exact_at(&mut self, buf: &mut [u8], mut offset: u64) -> Result<()> {
        for chunk in buf.chunks_mut(512) {
            match self.overlay.get(&offset) {
                None => self.underlying.read_exact_at(chunk, offset)?,
                Some(v) => chunk.copy_from_slice(v),
            }
            offset += 512
        }
        Ok(())
    }

    fn write_all_at(&mut self, buf: &[u8], mut offset: u64) -> Result<()> {
        for chunk in buf.chunks(512) {
            self.overlay.insert(offset, chunk.to_owned().into_boxed_slice());
            offset += 512
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn len(&self) -> u64 {
        self.underlying.len()
    }

    fn capability(&self) -> Capability {
        let underlying_cap = self.underlying.capability();
        let mut cap = Capability::default();
        cap.blksize = underlying_cap.blksize;
        cap
    }
}
