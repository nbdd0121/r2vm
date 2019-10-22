use parking_lot::Mutex;
use std::io::{IoSlice, IoSliceMut, Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;
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

/// Queue structures shared by both virtio and the device
pub(super) struct QueueInner {
    pub ready: bool,
    pub num: u16,
    pub num_max: u16,
    pub desc_addr: u64,
    pub avail_addr: u64,
    pub used_addr: u64,
    pub waker: Option<Waker>,
}

pub struct Queue {
    pub(super) inner: Arc<Mutex<QueueInner>>,
    last_avail_idx: u16,
    last_used_idx: u16,
}

impl Queue {
    pub fn new() -> Queue {
        Self::new_with_max(32768)
    }

    pub fn new_with_max(max: u16) -> Queue {
        let inner = Arc::new(Mutex::new(QueueInner {
            ready: false,
            num: max,
            num_max: max,
            desc_addr: 0,
            avail_addr: 0,
            used_addr: 0,
            waker: None,
        }));
        Queue { inner, last_avail_idx: 0, last_used_idx: 0 }
    }

    pub fn reset(&mut self) {
        let mut inner = self.inner.lock();
        inner.ready = false;
        inner.num = inner.num_max;
        inner.desc_addr = 0;
        inner.avail_addr = 0;
        inner.used_addr = 0;
        inner.waker = None;
        self.last_avail_idx = 0;
        self.last_used_idx = 0;
    }

    /// Try to get a buffer from the available ring. If there are no new buffers, `None` will be
    /// returned.
    pub fn try_take(&mut self) -> Option<Buffer> {
        let inner = self.inner.lock();

        // If the queue is not ready, trying to take item from it can cause segfault.
        if !inner.ready {
            return None;
        }

        let avail_idx_ptr = unsafe { &*((inner.avail_addr + 2) as usize as *const AtomicU16) };

        // Read the current index
        let avail_idx = avail_idx_ptr.load(Ordering::Acquire);

        // No extra elements in this queue
        if self.last_avail_idx == avail_idx {
            return None;
        }

        // Obtain the corresponding descriptor index for a given index of available ring.
        // Each index is 2 bytes, and there are flags and idx (2 bytes each) before the ring, so
        // we have + 4 here.
        let idx_ptr = inner.avail_addr + 4 + (self.last_avail_idx & (inner.num - 1)) as u64 * 2;
        let mut idx = unsafe { *(idx_ptr as usize as *const u16) };

        // Now we have obtained this descriptor, increment the index to skip over this.
        self.last_avail_idx = self.last_avail_idx.wrapping_add(1);

        let mut avail = Buffer { idx, bytes_written: 0, read: Vec::new(), write: Vec::new() };

        loop {
            let desc = unsafe {
                std::ptr::read(
                    (inner.desc_addr + (idx & (inner.num - 1)) as u64 * 16) as usize
                        as *const VirtqDesc,
                )
            };

            // Add to the corresponding buffer (read/write)
            if (desc.flags & VIRTQ_DESC_F_WRITE) == 0 {
                avail.read.push(IoSlice::new(unsafe {
                    std::slice::from_raw_parts((desc.addr as usize) as *const u8, desc.len as usize)
                }));
            } else {
                avail.write.push(IoSliceMut::new(unsafe {
                    std::slice::from_raw_parts_mut(
                        (desc.addr as usize) as *mut u8,
                        desc.len as usize,
                    )
                }));
            };

            // Follow the linked list until we've see a descritpro without NEXT flag.
            if (desc.flags & VIRTQ_DESC_F_NEXT) == 0 {
                break;
            }
            idx = desc.next;
        }

        Some(avail)
    }

    // This is to be changed to async fn once 1.39 arrives.
    pub fn take(&mut self) -> impl Future<Output = Buffer> + '_ {
        /// The future returned for calling async `wake` function of `Queue`.
        struct Take<'a> {
            queue: &'a mut Queue,
        }

        impl Future for Take<'_> {
            type Output = Buffer;

            fn poll(mut self: Pin<&mut Self>, ctx: &mut Context) -> Poll<Buffer> {
                if let Some(v) = self.queue.try_take() {
                    return Poll::Ready(v);
                }
                self.queue.inner.lock().waker = Some(ctx.waker().clone());
                Poll::Pending
            }
        }

        Take { queue: self }
    }

    /// Put back a buffer to the ring. This function is unsafe because there is no guarantee that
    /// the buffer to put back comes from this queue.
    pub unsafe fn put(&mut self, avail: Buffer) {
        let inner = self.inner.lock();
        let used_idx_ptr = &*((inner.used_addr + 2) as usize as *const AtomicU16);

        // Write requires invalidating icache
        for slice in avail.write {
            let start = slice.as_ptr() as usize;
            let end = start + slice.len();
            crate::emu::interp::icache_invalidate(start, end);
        }

        let elem_ptr = inner.used_addr + 4 + (self.last_used_idx & (inner.num - 1)) as u64 * 8;
        *(elem_ptr as usize as *mut u32) = avail.idx as u32;
        *((elem_ptr + 4) as usize as *mut u32) = avail.bytes_written as u32;
        self.last_used_idx = self.last_used_idx.wrapping_add(1);

        used_idx_ptr.store(self.last_used_idx, Ordering::Release);
    }
}

pub struct Buffer {
    idx: u16,
    bytes_written: usize,
    read: Vec<IoSlice<'static>>,
    write: Vec<IoSliceMut<'static>>,
}

impl Buffer {
    pub fn reader(&self) -> BufferReader {
        BufferReader::new(&self.read)
    }

    pub fn writer(&mut self) -> BufferWriter {
        BufferWriter::new(&mut self.write, &mut self.bytes_written)
    }

    pub fn reader_writer(&mut self) -> (BufferReader, BufferWriter) {
        (BufferReader::new(&self.read), BufferWriter::new(&mut self.write, &mut self.bytes_written))
    }
}

pub struct BufferReader<'a> {
    buffer: &'a [IoSlice<'static>],
    len: usize,
    pos: usize,
    slice_idx: usize,
    slice_offset: usize,
}

impl<'a> BufferReader<'a> {
    fn new(buffer: &'a [IoSlice<'static>]) -> Self {
        let len = buffer.iter().map(|x| x.len()).sum();
        Self { buffer, len, pos: 0, slice_idx: 0, slice_offset: 0 }
    }

    fn seek_slice(&mut self, mut offset: usize) {
        for (i, desc) in self.buffer.iter().enumerate() {
            // offset >= len means that this block should be completely skipped
            let len = desc.len() as usize;
            if offset >= len {
                offset -= len;
                continue;
            }

            self.slice_idx = i;
            self.slice_offset = offset;
            return;
        }

        self.slice_idx = self.buffer.len() + 1;
        self.slice_offset = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<'a> Seek for BufferReader<'a> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let (base, offset) = match pos {
            SeekFrom::Start(n) => (0, n as i64),
            SeekFrom::End(n) => (self.len, n),
            SeekFrom::Current(n) => (self.pos, n),
        };
        let new_pos = (base as i64 + offset) as usize;

        // If the position changed, reset slice_idx and slice_offset
        if self.pos != new_pos {
            self.seek_slice(new_pos);
        }

        self.pos = new_pos;
        Ok(new_pos as u64)
    }
}

impl<'a> Read for BufferReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.slice_idx >= self.buffer.len() {
            return Ok(0);
        }

        let slice = &self.buffer[self.slice_idx][self.slice_offset..];

        let len = if buf.len() >= slice.len() {
            buf[..slice.len()].copy_from_slice(slice);
            self.slice_idx += 1;
            self.slice_offset = 0;
            slice.len()
        } else {
            buf.copy_from_slice(&slice[..buf.len()]);
            self.slice_offset += buf.len();
            buf.len()
        };

        self.pos += len;
        Ok(len)
    }
}

pub struct BufferWriter<'a> {
    buffer: &'a mut [IoSliceMut<'static>],
    bytes_written: &'a mut usize,
    len: usize,
    pos: usize,
    slice_idx: usize,
    slice_offset: usize,
}

impl<'a> BufferWriter<'a> {
    pub fn new(buffer: &'a mut [IoSliceMut<'static>], bytes_written: &'a mut usize) -> Self {
        let len = buffer.iter().map(|x| x.len()).sum();
        Self { buffer, len, bytes_written, pos: 0, slice_idx: 0, slice_offset: 0 }
    }

    fn seek_slice(&mut self, mut offset: usize) {
        for (i, desc) in self.buffer.iter().enumerate() {
            // offset >= len means that this block should be completely skipped
            let len = desc.len() as usize;
            if offset >= len {
                offset -= len;
                continue;
            }

            self.slice_idx = i;
            self.slice_offset = offset;
            return;
        }

        self.slice_idx = self.buffer.len() + 1;
        self.slice_offset = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<'a> Seek for BufferWriter<'a> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let (base, offset) = match pos {
            SeekFrom::Start(n) => (0, n as i64),
            SeekFrom::End(n) => (self.len, n),
            SeekFrom::Current(n) => (self.pos, n),
        };
        let new_pos = (base as i64 + offset) as usize;

        // If the position changed, reset slice_idx and slice_offset
        if self.pos != new_pos {
            self.seek_slice(new_pos);
        }

        self.pos = new_pos;
        Ok(new_pos as u64)
    }
}

impl<'a> Write for BufferWriter<'a> {
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.slice_idx >= self.buffer.len() {
            return Ok(0);
        }

        let slice = &mut self.buffer[self.slice_idx][self.slice_offset..];

        let len = if buf.len() >= slice.len() {
            slice.copy_from_slice(&buf[..slice.len()]);
            self.slice_idx += 1;
            self.slice_offset = 0;
            slice.len()
        } else {
            slice[..buf.len()].copy_from_slice(buf);
            self.slice_offset += buf.len();
            buf.len()
        };

        self.pos += len;
        let bytes_written = usize::max(self.pos, *self.bytes_written);
        *self.bytes_written = bytes_written;
        Ok(len)
    }
}
