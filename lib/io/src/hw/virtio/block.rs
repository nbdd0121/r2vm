use super::{Device, DeviceId, Queue};
use crate::block::Block as BlockDevice;
use crate::{IrqPin, RuntimeContext};
use std::sync::Arc;

#[allow(dead_code)]
const VIRTIO_BLK_F_RO: usize = 5;

const VIRTIO_BLK_T_IN: u32 = 0;
const VIRTIO_BLK_T_OUT: u32 = 1;
/// This is an un-documented.
const VIRTIO_BLK_T_GET_ID: u32 = 8;

#[repr(C)]
struct VirtioBlkReqHeader {
    r#type: u32,
    reserved: u32,
    sector: u64,
}

/// A virtio block device.
pub struct Block {
    status: u32,
    config: [u8; 8],
    ctx: Arc<dyn RuntimeContext>,
    inner: Arc<Inner>,
}

struct Inner {
    file: Box<dyn BlockDevice + Send + Sync>,
    irq: Box<dyn IrqPin>,
}

impl Block {
    /// Create a new virtio block device.
    pub fn new(
        ctx: Arc<dyn RuntimeContext>,
        irq: Box<dyn IrqPin>,
        file: Box<dyn BlockDevice + Send + Sync>,
    ) -> Block {
        let len = file.len();
        if len % 512 != 0 {
            panic!("Size of block device must be multiple of 512 bytes");
        }
        let inner = Arc::new(Inner { file, irq });
        Block { status: 0, config: (len / 512).to_le_bytes(), ctx, inner }
    }

    fn start_task(&self, mut queue: Queue) {
        let inner = self.inner.clone();
        self.ctx.spawn_blocking("virtio_blk", Box::pin(async move {
            while let Ok(mut buffer) = queue.take().await {
                let (mut reader, mut writer) = buffer.reader_writer();

                let header: VirtioBlkReqHeader = unsafe {
                    let mut header = [0u8; 16];
                    reader.read_exact(&mut header).await.unwrap();
                    std::mem::transmute(header)
                };

                match header.r#type {
                    VIRTIO_BLK_T_IN => {
                        let mut io_buffer = Vec::with_capacity(writer.len());
                        unsafe { io_buffer.set_len(io_buffer.capacity() - 1) };

                        inner.file.read_exact_at(&mut io_buffer, header.sector * 512).unwrap();
                        trace!(target: "VirtioBlk", "read {} bytes from sector {:x}", io_buffer.len(), header.sector);

                        io_buffer.push(0);
                        writer.write_all(&io_buffer).await.unwrap();
                    }
                    VIRTIO_BLK_T_OUT => {
                        let mut io_buffer = Vec::with_capacity(reader.len() - 16);
                        unsafe { io_buffer.set_len(io_buffer.capacity()) };
                        reader.read_exact(&mut io_buffer).await.unwrap();

                        inner.file.write_all_at(&io_buffer, header.sector * 512).unwrap();
                        // We must make sure the data has been flushed into the disk before returning
                        inner.file.flush().unwrap();
                        trace!(target: "VirtioBlk", "write {} bytes from sector {:x}", io_buffer.len(), header.sector);

                        writer.write_all(&[0]).await.unwrap();
                    }
                    VIRTIO_BLK_T_GET_ID => {
                        // Fill in a dummy ID for now.
                        let len = writer.len();
                        writer.write_all(&vec![0; len]).await.unwrap();
                    }
                    _ => {
                        error!(target: "VirtioBlk", "unsupported block operation type {}", header.r#type);
                        continue;
                    }
                }

                queue.put(buffer).await;
                inner.irq.pulse();
            }
        }));
    }
}

impl Device for Block {
    fn device_id(&self) -> DeviceId {
        DeviceId::Block
    }
    fn device_feature(&self) -> u32 {
        0
    }
    fn driver_feature(&mut self, _value: u32) {}
    fn get_status(&self) -> u32 {
        self.status
    }
    fn set_status(&mut self, status: u32) {
        self.status = status
    }
    fn config_space(&self) -> &[u8] {
        &self.config
    }
    fn num_queues(&self) -> usize {
        1
    }
    fn reset(&mut self) {
        self.status = 0;
    }
    fn queue_ready(&mut self, _idx: usize, queue: Queue) {
        self.start_task(queue)
    }
}
