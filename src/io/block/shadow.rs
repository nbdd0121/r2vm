use super::Block;
use fnv::FnvHashMap;
use std::fs::File;
use std::io::Result;

pub struct Shadow {
    file: File,
    overlay: FnvHashMap<u64, Box<[u8]>>,
}

impl Shadow {
    pub fn new(file: File) -> Shadow {
        Shadow {
            file,
            overlay: FnvHashMap::default(),
        }
    }
}

impl Block for Shadow {
    fn read_exact_at(&mut self, buf: &mut [u8], mut offset: u64) -> Result<()> {
        for chunk in buf.chunks_mut(512) {
            match self.overlay.get(&offset) {
                None => self.file.read_exact_at(chunk, offset)?,
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

    fn flush(&mut self) -> Result<()> { Ok(()) }

    fn len(&mut self) -> Result<u64> { self.file.len() }
}

