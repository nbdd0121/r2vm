use crate::DmaContext;
use futures::channel::mpsc::{channel, Receiver, Sender};
use futures::StreamExt;
use parking_lot::Mutex;
use std::sync::Arc;

const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;
// We don't support indirect yet
#[allow(dead_code)]
const VIRTQ_DESC_F_INDIRECT: u16 = 4;

/// Error when trying to take buffers from a virtio queue that is not ready.
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
    pub send: Option<Sender<()>>,
    pub recv: Option<Receiver<()>>,
}

impl QueueInner {
    pub fn new(num_max: u16) -> Arc<Mutex<QueueInner>> {
        let (send, recv) = channel(1);
        Arc::new(Mutex::new(QueueInner {
            ready: false,
            num: num_max,
            num_max,
            desc_addr: 0,
            avail_addr: 0,
            used_addr: 0,
            send: Some(send),
            recv: Some(recv),
        }))
    }

    pub fn reset(&mut self) {
        self.ready = false;
    }
}

/// Safe abstraction of a virtio queue.
pub struct Queue {
    inner: Arc<Mutex<QueueInner>>,
    dma_ctx: Arc<dyn DmaContext>,
    recv: Receiver<()>,
    last_avail_idx: u16,
    last_used_idx: u16,
}

impl Queue {
    pub(super) fn new(dma_ctx: Arc<dyn DmaContext>, inner: Arc<Mutex<QueueInner>>) -> Option<Self> {
        let recv = inner.lock().recv.take();
        if let Some(recv) = recv {
            Some(Self { inner, dma_ctx, recv, last_avail_idx: 0, last_used_idx: 0 })
        } else {
            None
        }
    }

    /// Try to get a buffer from the available ring.
    ///
    /// If there are no new buffers, `None` will be returned. If the queue is not ready,
    /// trying to take an item from it will cause `Err(QueueNotReady)` to be returned.
    pub async fn try_take(&mut self) -> Result<Option<Buffer>, QueueNotReady> {
        let (num, desc_addr, avail_addr) = {
            let guard = self.inner.lock();

            // If the queue is not ready, trying to take item from it can cause segfault.
            if !guard.ready {
                return Err(QueueNotReady);
            }

            (guard.num, guard.desc_addr, guard.avail_addr)
        };

        // Read the current index
        let avail_idx = self.dma_ctx.read_u16(avail_addr + 2).await;

        // No extra elements in this queue
        if self.last_avail_idx == avail_idx {
            return Ok(None);
        }

        // Obtain the corresponding descriptor index for a given index of available ring.
        // Each index is 2 bytes, and there are flags and idx (2 bytes each) before the ring, so
        // we have + 4 here.
        let idx_ptr = avail_addr + 4 + (self.last_avail_idx & (num - 1)) as u64 * 2;
        let mut idx = self.dma_ctx.read_u16(idx_ptr).await;

        // Now we have obtained this descriptor, increment the index to skip over this.
        self.last_avail_idx = self.last_avail_idx.wrapping_add(1);

        let mut avail = BufferInner {
            queue: self.inner.clone(),
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
            self.dma_ctx.dma_read(desc_addr + (idx & (num - 1)) as u64 * 16, &mut desc).await;
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

        Ok(Some(Buffer(avail)))
    }

    /// Get a buffer from the available ring.
    ///
    /// The future returned will only resolve when there is an buffer available.
    /// If the queue is not ready,
    /// trying to take an item from it will cause `Err(QueueNotReady)` to be returned.
    pub async fn take(&mut self) -> Result<Buffer, QueueNotReady> {
        loop {
            match self.try_take().await? {
                Some(v) => return Ok(v),
                _ => (),
            }
            if self.recv.next().await.is_none() {
                return Err(QueueNotReady);
            }
        }
    }

    pub async fn put(&mut self, buffer: Buffer) {
        // Get inner without invoking the `Buffer::drop`.
        let buffer = unsafe { std::mem::transmute::<_, BufferInner>(buffer) };

        if !Arc::ptr_eq(&buffer.queue, &self.inner) {
            panic!("Buffer can only be put back to the originating queue");
        }

        let (num, used_addr) = {
            let guard = self.inner.lock();
            if !guard.ready {
                return;
            }

            (guard.num, guard.used_addr)
        };

        let elem_ptr = used_addr + 4 + (self.last_used_idx & (num - 1)) as u64 * 8;
        let mut descriptor = [0; 8];
        descriptor[0..4].copy_from_slice(&(buffer.idx as u32).to_le_bytes());
        descriptor[4..8].copy_from_slice(&(buffer.bytes_written as u32).to_le_bytes());
        self.dma_ctx.dma_write(elem_ptr, &descriptor).await;

        self.last_used_idx = self.last_used_idx.wrapping_add(1);
        self.dma_ctx.write_u16(used_addr + 2, self.last_used_idx).await;
    }
}

struct BufferInner {
    queue: Arc<Mutex<QueueInner>>,
    idx: u16,
    bytes_written: usize,
    read: Vec<(u64, usize)>,
    write: Vec<(u64, usize)>,
    read_len: usize,
    write_len: usize,
    dma_ctx: Arc<dyn DmaContext>,
}

/// A buffer passed from the kernel to the virtio device.
#[repr(transparent)]
pub struct Buffer(BufferInner);

impl Drop for Buffer {
    fn drop(&mut self) {
        // Buffer shouldn't leak in normal operation but is possible during reset.
        warn!(target: "Mmio", "buffer leaked");
    }
}

impl Buffer {
    /// Get the readonly part of this buffer.
    pub fn reader(&self) -> BufferReader<'_> {
        BufferReader {
            buffer: &self.0.read,
            len: self.0.read_len,
            pos: 0,
            slice_idx: 0,
            slice_offset: 0,
            dma_ctx: &*self.0.dma_ctx,
        }
    }

    /// Get the write-only part of this buffer.
    pub fn writer(&mut self) -> BufferWriter<'_> {
        BufferWriter {
            buffer: &self.0.write,
            len: self.0.write_len,
            bytes_written: &mut self.0.bytes_written,
            pos: 0,
            slice_idx: 0,
            slice_offset: 0,
            dma_ctx: &*self.0.dma_ctx,
        }
    }

    /// Split this buffer into two halves.
    pub fn reader_writer(&mut self) -> (BufferReader<'_>, BufferWriter<'_>) {
        (
            BufferReader {
                buffer: &self.0.read,
                len: self.0.read_len,
                pos: 0,
                slice_idx: 0,
                slice_offset: 0,
                dma_ctx: &*self.0.dma_ctx,
            },
            BufferWriter {
                buffer: &self.0.write,
                len: self.0.write_len,
                bytes_written: &mut self.0.bytes_written,
                pos: 0,
                slice_idx: 0,
                slice_offset: 0,
                dma_ctx: &*self.0.dma_ctx,
            },
        )
    }
}

/// Reader half of the buffer.
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

    /// Get the length of this buffer half.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn seek(&mut self, new_pos: usize) -> std::io::Result<usize> {
        // If the position changed, reset slice_idx and slice_offset
        if self.pos != new_pos {
            self.seek_slice(new_pos);
        }

        self.pos = new_pos;
        Ok(new_pos)
    }

    pub async fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.slice_idx >= self.buffer.len() {
            return Ok(0);
        }

        let (addr, len) = self.buffer[self.slice_idx];
        let slice_addr = addr + self.slice_offset as u64;
        let slice_len = len - self.slice_offset;

        let len = if buf.len() >= slice_len {
            self.dma_ctx.dma_read(slice_addr, &mut buf[..slice_len]).await;
            self.slice_idx += 1;
            self.slice_offset = 0;
            slice_len
        } else {
            self.dma_ctx.dma_read(slice_addr, buf).await;
            self.slice_offset += buf.len();
            buf.len()
        };

        self.pos += len;
        Ok(len)
    }

    pub async fn read_exact(&mut self, mut buf: &mut [u8]) -> std::io::Result<()> {
        while !buf.is_empty() {
            match self.read(buf).await {
                Ok(0) => break,
                Ok(n) => {
                    buf = &mut buf[n..];
                }
                Err(e) => return Err(e),
            }
        }
        if !buf.is_empty() {
            Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "failed to fill whole buffer",
            ))
        } else {
            Ok(())
        }
    }
}

/// Writer half of the buffer.
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

    /// Get the length of this buffer half.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn seek(&mut self, new_pos: usize) -> std::io::Result<usize> {
        // If the position changed, reset slice_idx and slice_offset
        if self.pos != new_pos {
            self.seek_slice(new_pos);
        }

        self.pos = new_pos;
        Ok(new_pos)
    }

    pub async fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.slice_idx >= self.buffer.len() {
            return Ok(0);
        }

        let (addr, len) = self.buffer[self.slice_idx];
        let slice_addr = addr + self.slice_offset as u64;
        let slice_len = len - self.slice_offset;

        let len = if buf.len() >= slice_len {
            self.dma_ctx.dma_write(slice_addr, &buf[..slice_len]).await;
            self.slice_idx += 1;
            self.slice_offset = 0;
            slice_len
        } else {
            self.dma_ctx.dma_write(slice_addr, buf).await;
            self.slice_offset += buf.len();
            buf.len()
        };

        self.pos += len;
        let bytes_written = usize::max(self.pos, *self.bytes_written);
        *self.bytes_written = bytes_written;
        Ok(len)
    }

    pub async fn write_all(&mut self, mut buf: &[u8]) -> std::io::Result<()> {
        while !buf.is_empty() {
            match self.write(buf).await {
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::WriteZero,
                        "failed to write whole buffer",
                    ));
                }
                Ok(n) => buf = &buf[n..],
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}
