use super::{Device, DeviceId, Queue, BufferReader, BufferWriter};
use byteorder::{LE, WriteBytesExt};
use p9::serialize::{Fcall, Serializable};
use p9::{P9Handler, Passthrough};

use std::io::Seek;

/// Feature bit indicating presence of mount tag
const VIRTIO_9P_MOUNT_TAG: u32 = 1;

pub struct P9 {
    status: u32,
    queue: Queue,
    config: Box<[u8]>,
    handler: P9Handler<Passthrough>,
}

impl P9 {
    pub fn new(mount_tag: &str, path: &std::path::Path) -> P9 {
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
            queue: Queue::new(),
            config: config.into_boxed_slice(),
            handler: P9Handler::new(Passthrough::new(path).unwrap()),
        }
    }
}

impl Device for P9 {
    fn device_id(&self) -> DeviceId { DeviceId::P9 }
    fn device_feature(&self) -> u32 { VIRTIO_9P_MOUNT_TAG }
    fn driver_feature(&mut self, _value: u32) {}
    fn get_status(&self) -> u32 { self.status }
    fn set_status(&mut self, status: u32) { self.status = status }
    fn config_space(&self) -> &[u8] { &self.config }
    fn queues(&mut self) -> &mut [Queue] {
        std::slice::from_mut(&mut self.queue)
    }
    fn reset(&mut self) {
        self.status = 0;
        self.queue.reset();
    }
    fn notify(&mut self, _idx: usize) {
        while let Some(mut buffer) = self.queue.take() {
            let (tag, fcall) = {
                let mut reader = BufferReader::new(&mut buffer);
                reader.seek(std::io::SeekFrom::Start(4)).unwrap();
                <(u16, Fcall)>::decode(&mut reader).unwrap()
            };

            trace!(target: "9p", "received {}, {:?}", tag, fcall);
            let resp = self.handler.handle_fcall(fcall);
            trace!(target: "9p", "send {}, {:?}", tag, resp);

            let mut writer = BufferWriter::new(&mut buffer);
            writer.seek(std::io::SeekFrom::Start(4)).unwrap();
            (tag, resp).encode(&mut writer).unwrap();
            let size = writer.pos;
            writer.seek(std::io::SeekFrom::Start(0)).unwrap();
            writer.write_u32::<LE>(size as u32).unwrap();

            unsafe { self.queue.put(buffer); }
        }

        // TODO
        crate::emu::PLIC.lock().trigger(3);
    }

}
