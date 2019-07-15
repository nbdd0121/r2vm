const VIRTQ_DESC_F_NEXT    : u16 = 1;
const VIRTQ_DESC_F_WRITE   : u16 = 2;
// We don't support indirect yet
#[allow(dead_code)]
const VIRTQ_DESC_F_INDIRECT: u16 = 4;

#[repr(C)]
#[derive(Clone, Copy)]
struct VirtqDesc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

pub struct Queue {
    pub ready: bool,
    pub num: u16,
    pub desc_addr: u64,
    pub avail_addr: u64,
    pub used_addr: u64,
    last_avail_idx: u16,
}

impl Queue {
    pub const fn new() -> Queue {
        Queue {
            ready: false,
            num: 32768,
            desc_addr: 0,
            avail_addr: 0,
            used_addr: 0,
            last_avail_idx: 0,
        }
    }

    pub fn reset(&mut self) {
        self.ready = false;
        self.num = 32768;
        self.desc_addr = 0;
        self.avail_addr = 0;
        self.used_addr = 0;
        self.last_avail_idx = 0;
    }

    /// Try to get a buffer from the available ring. If there are no new buffers, `None` will be
    /// returned.
    pub fn take(&mut self) -> Option<Buffer> {
        // Read the current index
        let avail_idx = unsafe { *((self.avail_addr + 2) as usize as *const u16) };

        // No extra elements in this queue
        if self.last_avail_idx == avail_idx { return None }

        // Obtain the corresponding descriptor index for a given index of available ring.
        // Each index is 2 bytes, and there are flags and idx (2 bytes each) before the ring, so
        // we have + 4 here.
        let idx_ptr = self.avail_addr + 4 + (self.last_avail_idx & (self.num - 1)) as u64 * 2;
        let mut idx = unsafe { *(idx_ptr as usize as *const u16) };

        // Now we have obtained this descriptor, increment the index to skip over this.
        self.last_avail_idx += 1;

        let mut avail = Buffer {
            idx: idx,
            bytes_written: 0,
            read: Vec::new(),
            write: Vec::new(),
        };

        loop {
            let desc = unsafe {
                std::ptr::read((self.desc_addr + (idx & (self.num - 1)) as u64 * 16) as usize as *const VirtqDesc)
            };

            // Add to the corresponding buffer (read/write)
            if (desc.flags & VIRTQ_DESC_F_WRITE) == 0 {
                avail.read.push(desc);
            } else {
                avail.write.push(desc);
            };

            // Follow the linked list until we've see a descritpro without NEXT flag.
            if (desc.flags & VIRTQ_DESC_F_NEXT) == 0 {
                break;
            }
            idx = desc.next;
        }

        Some(avail)
    }

    /// Put back a buffer to the ring. This function is unsafe because there is no guarantee that
    /// the buffer to put back comes from this queue.
    pub unsafe fn put(&self, avail: Buffer) {
        let used_idx_ptr = (self.used_addr + 2) as usize as *mut u16;
        let used_idx = *used_idx_ptr;
        let elem_ptr = self.used_addr + 4 + (used_idx & (self.num - 1)) as u64 * 8;
        *(elem_ptr as usize as *mut u32) = avail.idx as u32;
        *((elem_ptr + 4) as usize as *mut u32) = avail.bytes_written as u32;
        *used_idx_ptr = used_idx + 1;
    }
}

pub struct Buffer {
    idx: u16,
    bytes_written: usize,
    read: Vec<VirtqDesc>,
    write: Vec<VirtqDesc>,
}

impl Buffer {
    fn len_of(vec: &[VirtqDesc]) -> usize {
        let mut total_len = 0;
        for desc in vec {
            total_len += desc.len as usize;
        }
        total_len
    }

    pub fn read_len(&self) -> usize { Self::len_of(&self.read) }
    pub fn write_len(&self) -> usize { Self::len_of(&self.write) }

    pub fn read(&self, mut offset: usize, mut buf: &mut [u8]) -> usize {
        let mut total_len = 0;

        for desc in &self.read {
            // No further read is necessary
            if buf.len() == 0 { break }

            // offset >= len means that this block should be completely skipped
            let len = desc.len as usize;
            if offset >= len {
                offset -= len;
                continue
            }

            // The maximum number we can read within this descriptor
            let len = std::cmp::min(buf.len(), len - offset);
            unsafe { std::ptr::copy_nonoverlapping((desc.addr as usize + offset) as *const u8, buf.as_mut_ptr(), len); }

            // Set offset to 0, as we completed seeking.
            offset = 0;
            total_len += len;
            buf = &mut buf[len..];
        }

        total_len
    }

    pub fn write(&mut self, offset: usize, mut buf: &[u8]) -> usize {
        let mut total_len = 0;
        let mut rem_offset = offset;

        for desc in &self.write {
            // No further read is necessary
            if buf.len() == 0 { break }

            // rem_offset >= len means that this block should be completely skipped
            let len = desc.len as usize;
            if rem_offset >= len {
                rem_offset -= len;
                continue
            }

            // The maximum number we can write within this descriptor
            let len = std::cmp::min(buf.len(), len - rem_offset);
            unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), (desc.addr as usize + rem_offset) as *mut u8, len); }

            // Set offset to 0, as we completed seeking.
            rem_offset = 0;
            total_len += len;
            buf = &buf[len..];
        }

        self.bytes_written = std::cmp::max(self.bytes_written, offset + total_len);
        total_len
    }
}

pub struct BufferReader<'a> {
    buffer: &'a Buffer,
    pub pos: usize,
}

impl<'a> BufferReader<'a> {
    pub fn new(buffer: &'a mut Buffer) -> Self {
        Self { buffer, pos: 0 }
    }
}

impl<'a> std::io::Seek for BufferReader<'a> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let (base, offset) = match pos {
            std::io::SeekFrom::Start(n) => { (0, n as i64) }
            std::io::SeekFrom::End(n) => (self.buffer.read_len(), n),
            std::io::SeekFrom::Current(n) => (self.pos, n),
        };
        let new_pos = (base as i64 + offset) as usize;
        self.pos = new_pos;
        Ok(new_pos as u64)
    }
}

impl<'a> std::io::Read for BufferReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let size = self.buffer.read(self.pos, buf);
        self.pos += size;
        Ok(size)
    }
}

pub struct BufferWriter<'a> {
    buffer: &'a mut Buffer,
    pub pos: usize,
}

impl<'a> BufferWriter<'a> {
    pub fn new(buffer: &'a mut Buffer) -> Self {
        Self { buffer, pos: 0 }
    }
}

impl<'a> std::io::Seek for BufferWriter<'a> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let (base, offset) = match pos {
            std::io::SeekFrom::Start(n) => { (0, n as i64) }
            std::io::SeekFrom::End(n) => (self.buffer.write_len(), n),
            std::io::SeekFrom::Current(n) => (self.pos, n),
        };
        let new_pos = (base as i64 + offset) as usize;
        self.pos = new_pos;
        Ok(new_pos as u64)
    }
}

impl<'a> std::io::Write for BufferWriter<'a> {
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }

    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let size = self.buffer.write(self.pos, buf);
        self.pos += size;
        Ok(size)
    }
}
