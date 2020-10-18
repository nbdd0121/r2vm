use super::{Device, DeviceId, Queue};
use crate::fs::FileSystem;
use crate::{IrqPin, RuntimeContext};
use byteorder::{ByteOrder, LE};
use p9::serialize::{Fcall, Serializable};
use p9::P9Handler;
use parking_lot::Mutex;
use std::sync::Arc;

/// Feature bit indicating presence of mount tag
const VIRTIO_9P_MOUNT_TAG: u32 = 1;

/// A virtio P9 file share device.
pub struct P9<FS: FileSystem> {
    status: u32,
    config: Box<[u8]>,
    ctx: Arc<dyn RuntimeContext>,
    inner: Arc<Inner<FS>>,
}

struct Inner<FS: FileSystem> {
    handler: Arc<Mutex<P9Handler<FS>>>,
    irq: Arc<Box<dyn IrqPin>>,
}

impl<FS> P9<FS>
where
    FS: FileSystem + Send + 'static,
    <FS as FileSystem>::File: Send,
{
    /// Create a new virtio P9 file share device with given mount tag and file system.
    pub fn new(
        ctx: Arc<dyn RuntimeContext>,
        irq: Box<dyn IrqPin>,
        mount_tag: &str,
        fs: FS,
    ) -> Self {
        // Config space is composed of u16 length followed by the tag bytes
        let config = {
            let tag_len = mount_tag.len();
            assert!(tag_len <= u16::max_value() as usize);
            let mut config = Vec::with_capacity(tag_len + 2);
            config.push(tag_len as u8);
            config.push((tag_len >> 8) as u8);
            config.extend_from_slice(mount_tag.as_bytes());
            config
        };

        let inner = Arc::new(Inner {
            handler: Arc::new(Mutex::new(P9Handler::new(fs))),
            irq: Arc::new(irq),
        });

        P9 { status: 0, config: config.into_boxed_slice(), ctx, inner }
    }

    fn start_task(&self, mut queue: Queue) {
        let inner = self.inner.clone();
        self.ctx.spawn_blocking(
            "virtio-p9",
            Box::pin(async move {
                while let Ok(mut buffer) = queue.take().await {
                    let (mut reader, mut writer) = buffer.reader_writer();

                    let size = {
                        let mut size_buf = [0u8; 4];
                        reader.read_exact(&mut size_buf).await.unwrap();
                        LE::read_u32(&size_buf) as usize - 4
                    };

                    let mut msg_buf = Vec::with_capacity(size);
                    unsafe {
                        msg_buf.set_len(size);
                    }
                    reader.read_exact(&mut msg_buf).await.unwrap();
                    let (tag, fcall) =
                        <(u16, Fcall)>::decode(&mut std::io::Cursor::new(&msg_buf)).unwrap();

                    trace!(target: "9p", "received {}, {:?}", tag, fcall);
                    let resp = inner.handler.lock().handle_fcall(fcall);
                    trace!(target: "9p", "send {}, {:?}", tag, resp);

                    msg_buf.resize(4, 0);
                    (tag, resp).encode(&mut msg_buf).unwrap();
                    let size = msg_buf.len();
                    LE::write_u32(&mut msg_buf, size as u32);

                    writer.write_all(&mut msg_buf).await.unwrap();

                    queue.put(buffer).await;
                    inner.irq.pulse();
                }
            }),
        );
    }
}

impl<FS> Device for P9<FS>
where
    FS: FileSystem + Send + 'static,
    <FS as FileSystem>::File: Send,
{
    fn device_id(&self) -> DeviceId {
        DeviceId::P9
    }
    fn device_feature(&self) -> u32 {
        VIRTIO_9P_MOUNT_TAG
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
        self.start_task(queue);
    }
}
