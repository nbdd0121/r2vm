use super::{Device, DeviceId, Queue};
use byteorder::{WriteBytesExt, LE};
use p9::serialize::{Fcall, Serializable};
use p9::{P9Handler, Passthrough};
use parking_lot::Mutex;
use std::sync::Arc;

use std::io::{Seek, SeekFrom};

/// Feature bit indicating presence of mount tag
const VIRTIO_9P_MOUNT_TAG: u32 = 1;

pub struct P9 {
    status: u32,
    config: Box<[u8]>,
    handler: Arc<Mutex<P9Handler<Passthrough>>>,
    irq: u32,
}

impl P9 {
    pub fn new(irq: u32, mount_tag: &str, path: &std::path::Path) -> P9 {
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

        P9 {
            status: 0,
            config: config.into_boxed_slice(),
            handler: Arc::new(Mutex::new(P9Handler::new(Passthrough::new(path).unwrap()))),
            irq,
        }
    }
}

fn start_task(mut queue: Queue, handler: Arc<Mutex<P9Handler<Passthrough>>>, irq: u32) {
    let task = async move {
        while let Ok(mut buffer) = queue.take().await {
            let (mut reader, mut writer) = buffer.reader_writer();

            reader.seek(SeekFrom::Start(4)).unwrap();
            let (tag, fcall) = <(u16, Fcall)>::decode(&mut reader).unwrap();

            trace!(target: "9p", "received {}, {:?}", tag, fcall);
            let resp = handler.lock().handle_fcall(fcall);
            trace!(target: "9p", "send {}, {:?}", tag, resp);

            writer.seek(SeekFrom::Start(4)).unwrap();
            (tag, resp).encode(&mut writer).unwrap();
            let size = writer.seek(SeekFrom::Current(0)).unwrap();
            writer.seek(SeekFrom::Start(0)).unwrap();
            writer.write_u32::<LE>(size as u32).unwrap();

            drop(buffer);
            crate::emu::PLIC.lock().trigger(irq);
        }
    };
    if crate::threaded() {
        std::thread::Builder::new()
            .name("virtio-p9".to_owned())
            .spawn(move || futures::executor::block_on(task))
            .unwrap();
    } else {
        crate::event_loop().spawn(task);
    }
}

impl Device for P9 {
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
        start_task(queue, self.handler.clone(), self.irq);
    }
}
