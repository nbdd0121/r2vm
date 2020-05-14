use parking_lot::Mutex;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use io::DmaContext;

const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;
// We don't support indirect yet
#[allow(dead_code)]
const VIRTQ_DESC_F_INDIRECT: u16 = 4;

pub struct QueueNotReady;

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
    pub last_avail_idx: u16,
    pub last_used_idx: u16,
    pub waker: Option<Waker>,
    pub dma_ctx: Arc<dyn DmaContext>,
}

impl QueueInner {
    pub fn new(dma_ctx: Arc<dyn DmaContext>, num_max: u16) -> Arc<Mutex<QueueInner>> {
        let inner = Arc::new(Mutex::new(QueueInner {
            ready: false,
            num: num_max,
            num_max,
            desc_addr: 0,
            avail_addr: 0,
            used_addr: 0,
            waker: None,
            last_avail_idx: 0,
            last_used_idx: 0,
            dma_ctx,
        }));
        inner
    }

    pub fn reset(&mut self) {
        self.ready = false;
        self.num = self.num_max;
        self.desc_addr = 0;
        self.avail_addr = 0;
        self.used_addr = 0;
        self.waker = None;
        self.last_avail_idx = 0;
        self.last_used_idx = 0;
    }

    /// Try to get a buffer from the available ring. If there are no new buffers, `None` will be
    /// returned.
    fn try_take(&mut self, arc: &Arc<Mutex<Self>>) -> Result<Option<Buffer>, QueueNotReady> {
        // If the queue is not ready, trying to take item from it can cause segfault.
        if !self.ready {
            return Err(QueueNotReady);
        }

        // Read the current index
        let avail_idx = self.dma_ctx.read_u16(self.avail_addr + 2);

        // No extra elements in this queue
        if self.last_avail_idx == avail_idx {
            return Ok(None);
        }

        // Obtain the corresponding descriptor index for a given index of available ring.
        // Each index is 2 bytes, and there are flags and idx (2 bytes each) before the ring, so
        // we have + 4 here.
        let idx_ptr = self.avail_addr + 4 + (self.last_avail_idx & (self.num - 1)) as u64 * 2;
        let mut idx = self.dma_ctx.read_u16(idx_ptr);

        // Now we have obtained this descriptor, increment the index to skip over this.
        self.last_avail_idx = self.last_avail_idx.wrapping_add(1);

        let mut avail = Buffer {
            queue: arc.clone(),
            idx,
            bytes_written: 0,
            read: Vec::new(),
            write: Vec::new(),
            read_len: 0,
            write_len: 0,
            dma_ctx: self.dma_ctx.clone(),
        };

        loop {
            let mut desc = [0; std::mem::size_of::<VirtqDesc>()];
            self.dma_ctx.dma_read(self.desc_addr + (idx & (self.num - 1)) as u64 * 16, &mut desc);
            let desc: VirtqDesc = unsafe { std::mem::transmute(desc) };

            // Add to the corresponding buffer (read/write)
            if (desc.flags & VIRTQ_DESC_F_WRITE) == 0 {
                avail.read.push((desc.addr, desc.len as usize));
            } else {
                avail.write.push((desc.addr, desc.len as usize));
            };

            avail.read_len = avail.read.iter().map(|(_, len)| len).sum();
            avail.write_len = avail.write.iter().map(|(_, len)| len).sum();

            // Follow the linked list until we've see a descritpro without NEXT flag.
            if (desc.flags & VIRTQ_DESC_F_NEXT) == 0 {
                break;
            }
            idx = desc.next;
        }

        Ok(Some(avail))
    }

    /// Put back a buffer to the ring.
    fn put(&mut self, avail: &Buffer) {
        if !self.ready {
            return;
        }

        let elem_ptr = self.used_addr + 4 + (self.last_used_idx & (self.num - 1)) as u64 * 8;
        let mut buffer = [0; 8];
        buffer[0..4].copy_from_slice(&(avail.idx as u32).to_le_bytes());
        buffer[4..8].copy_from_slice(&(avail.bytes_written as u32).to_le_bytes());
        self.dma_ctx.dma_write(elem_ptr, &buffer);

        self.last_used_idx = self.last_used_idx.wrapping_add(1);
        self.dma_ctx.write_u16(self.used_addr + 2, self.last_used_idx);
    }
}

pub struct Queue {
    pub(super) inner: Arc<Mutex<QueueInner>>,
}

impl Queue {
    /// Try to get a buffer from the available ring. If there are no new buffers, `None` will be
    /// returned.
    pub fn try_take(&mut self) -> Result<Option<Buffer>, QueueNotReady> {
        self.inner.lock().try_take(&self.inner)
    }

    // This is to be changed to async fn once 1.39 arrives.
    pub fn take(&mut self) -> impl Future<Output = Result<Buffer, QueueNotReady>> + '_ {
        /// The future returned for calling async `wake` function of `Queue`.
        struct Take<'a> {
            queue: &'a mut Queue,
        }

        impl Future for Take<'_> {
            type Output = Result<Buffer, QueueNotReady>;

            fn poll(self: Pin<&mut Self>, ctx: &mut Context) -> Poll<Self::Output> {
                let mut inner = self.queue.inner.lock();
                match inner.try_take(&self.queue.inner) {
                    Err(v) => Poll::Ready(Err(v)),
                    Ok(Some(v)) => Poll::Ready(Ok(v)),
                    Ok(None) => {
                        inner.waker = Some(ctx.waker().clone());
                        Poll::Pending
                    }
                }
            }
        }

        Take { queue: self }
    }
}

pub struct Buffer {
    queue: Arc<Mutex<QueueInner>>,
    idx: u16,
    bytes_written: usize,
    read: Vec<(u64, usize)>,
    write: Vec<(u64, usize)>,
    read_len: usize,
    write_len: usize,
    dma_ctx: Arc<dyn DmaContext>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.queue.lock().put(self);
    }
}

impl Buffer {
    pub fn reader(&self) -> BufferReader {
        BufferReader {
            buffer: &self.read,
            len: self.read_len,
            pos: 0,
            slice_idx: 0,
            slice_offset: 0,
            dma_ctx: &*self.dma_ctx,
        }
    }

    pub fn writer(&mut self) -> BufferWriter {
        BufferWriter {
            buffer: &self.write,
            len: self.write_len,
            bytes_written: &mut self.bytes_written,
            pos: 0,
            slice_idx: 0,
            slice_offset: 0,
            dma_ctx: &*self.dma_ctx,
        }
    }

    pub fn reader_writer(&mut self) -> (BufferReader, BufferWriter) {
        (
            BufferReader {
                buffer: &self.read,
                len: self.read_len,
                pos: 0,
                slice_idx: 0,
                slice_offset: 0,
                dma_ctx: &*self.dma_ctx,
            },
            BufferWriter {
                buffer: &self.write,
                len: self.write_len,
                bytes_written: &mut self.bytes_written,
                pos: 0,
                slice_idx: 0,
                slice_offset: 0,
                dma_ctx: &*self.dma_ctx,
            },
        )
    }
}

pub struct BufferReader<'a> {
    buffer: &'a [(u64, usize)],
    len: usize,
    pos: usize,
    slice_idx: usize,
    slice_offset: usize,
    dma_ctx: &'a dyn DmaContext,
}

impl<'a> BufferReader<'a> {
    fn seek_slice(&mut self, mut offset: usize) {
        for (i, &(_, len)) in self.buffer.iter().enumerate() {
            // offset >= len means that this block should be completely skipped
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

        let (addr, len) = self.buffer[self.slice_idx];
        let slice_addr = addr + self.slice_offset as u64;
        let slice_len = len - self.slice_offset;

        let len = if buf.len() >= slice_len {
            self.dma_ctx.dma_read(slice_addr, &mut buf[..slice_len]);
            self.slice_idx += 1;
            self.slice_offset = 0;
            slice_len
        } else {
            self.dma_ctx.dma_read(slice_addr, buf);
            self.slice_offset += buf.len();
            buf.len()
        };

        self.pos += len;
        Ok(len)
    }
}

pub struct BufferWriter<'a> {
    buffer: &'a [(u64, usize)],
    bytes_written: &'a mut usize,
    len: usize,
    pos: usize,
    slice_idx: usize,
    slice_offset: usize,
    dma_ctx: &'a dyn DmaContext,
}

impl<'a> BufferWriter<'a> {
    fn seek_slice(&mut self, mut offset: usize) {
        for (i, &(_, len)) in self.buffer.iter().enumerate() {
            // offset >= len means that this block should be completely skipped
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

        let (addr, len) = self.buffer[self.slice_idx];
        let slice_addr = addr + self.slice_offset as u64;
        let slice_len = len - self.slice_offset;

        let len = if buf.len() >= slice_len {
            self.dma_ctx.dma_write(slice_addr, &buf[..slice_len]);
            self.slice_idx += 1;
            self.slice_offset = 0;
            slice_len
        } else {
            self.dma_ctx.dma_write(slice_addr, buf);
            self.slice_offset += buf.len();
            buf.len()
        };

        self.pos += len;
        let bytes_written = usize::max(self.pos, *self.bytes_written);
        *self.bytes_written = bytes_written;
        Ok(len)
    }
}
