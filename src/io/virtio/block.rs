use super::super::block::Block as BlockDevice;
use super::super::{IoContext, IrqPin};
use super::{Device, DeviceId, Queue};
use parking_lot::Mutex;
use std::io::{Read, Write};
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

pub struct Block {
    status: u32,
    config: [u8; 8],
    file: Arc<Mutex<Box<dyn BlockDevice + Send>>>,
    irq: Arc<dyn IrqPin>,
    io_ctx: Arc<dyn IoContext>,
}

impl Block {
    pub fn new(
        io_ctx: Arc<dyn IoContext>,
        irq: Arc<dyn IrqPin>,
        mut file: Box<dyn BlockDevice + Send>,
    ) -> Block {
        let len = file.len().unwrap();
        if len % 512 != 0 {
            panic!("Size of block device must be multiple of 512 bytes");
        }
        Block {
            status: 0,
            config: (len / 512).to_le_bytes(),
            file: Arc::new(Mutex::new(file)),
            irq,
            io_ctx,
        }
    }
}

fn start_task(
    mut queue: Queue,
    io_ctx: &dyn IoContext,
    file: Arc<Mutex<Box<dyn BlockDevice + Send>>>,
    irq: Arc<dyn IrqPin>,
) {
    let task = async move {
        while let Ok(mut buffer) = queue.take().await {
            let (mut reader, mut writer) = buffer.reader_writer();

            let header: VirtioBlkReqHeader = unsafe {
                let mut header: [u8; 16] = std::mem::MaybeUninit::uninit().assume_init();
                reader.read_exact(&mut header).unwrap();
                std::mem::transmute(header)
            };

            let mut file = file.lock();
            match header.r#type {
                VIRTIO_BLK_T_IN => {
                    let mut io_buffer = Vec::with_capacity(writer.len());
                    unsafe { io_buffer.set_len(io_buffer.capacity() - 1) };
                    (*file).read_exact_at(&mut io_buffer, header.sector * 512).unwrap();
                    trace!(target: "VirtioBlk", "read {} bytes from sector {:x}", io_buffer.len(), header.sector);

                    io_buffer.push(0);
                    writer.write_all(&io_buffer).unwrap();
                }
                VIRTIO_BLK_T_OUT => {
                    let mut io_buffer = Vec::with_capacity(reader.len() - 16);
                    unsafe { io_buffer.set_len(io_buffer.capacity()) };
                    reader.read_exact(&mut io_buffer).unwrap();

                    file.write_all_at(&io_buffer, header.sector * 512).unwrap();
                    // We must make sure the data has been flushed into the disk before returning
                    file.flush().unwrap();
                    trace!(target: "VirtioBlk", "write {} bytes from sector {:x}", io_buffer.len(), header.sector);

                    writer.write_all(&[0]).unwrap();
                }
                VIRTIO_BLK_T_GET_ID => {
                    // Fill in a dummy ID for now.
                    let len = writer.len();
                    writer.write_all(&vec![0; len]).unwrap();
                }
                _ => {
                    error!(target: "VirtioBlk", "unsupported block operation type {}", header.r#type);
                    continue;
                }
            }

            drop(buffer);
            irq.pulse();
        }
    };
    io_ctx.spawn_blocking("virtio_blk", Box::pin(task));
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
        start_task(queue, &*self.io_ctx, self.file.clone(), self.irq.clone())
    }
}
