use super::{Device, DeviceId, Queue};
use std::io::{Read, Write, Seek, SeekFrom};
use std::sync::Arc;
use parking_lot::Mutex;
use crate::io::network::{Network as NetworkDevice};

const VIRTIO_BLK_F_MAC: usize = 5;

#[repr(C)]
struct VirtioNetHeader {
    flags: u8,
    gso_type: u8,
    hdr_len: u16,
    gso_size: u16,
    csum_start: u16,
    csum_offset: u16,
    num_buffers: u16,
}

/// A virtio entropy source device.
pub struct Network {
    net: Arc<dyn NetworkDevice>,
    status: u32,
    rx: Arc<Mutex<Queue>>,
    tx: Queue,
    mac: [u8; 6],
    irq: u32,
}

fn send_packet(queue: &Arc<Mutex<Queue>>, buf: &[u8], irq: u32) {
    let mut queue = queue.lock();
    if let Some(mut buffer) = queue.take() {
        let mut writer = buffer.writer();
        let header: [u8; std::mem::size_of::<VirtioNetHeader>()] = {
            let header = VirtioNetHeader {
                flags: 0,
                gso_type: 0,
                hdr_len: 0,
                gso_size: 0,
                csum_start: 0,
                csum_offset: 0,
                num_buffers: 1,
            };
            unsafe { std::mem::transmute(header) }
        };
        if header.len() + buf.len() > writer.len() {
            info!(
                target: "VirtioNet",
                "discard packet of size {:x} because it does not fit into buffer of size {:x}",
                buf.len(), writer.len()
            );
            return;
        }
        writer.write_all(&header).unwrap();
        writer.write_all(buf).unwrap();
        unsafe { queue.put(buffer) };

        crate::emu::PLIC.lock().trigger(irq);
    } else {
        info!(
            target: "VirtioNet",
            "discard packet of size {:x} because there is no buffer in receiver queue",
            buf.len()
        );
    }
}

fn thread_run(iface: Arc<dyn NetworkDevice>, queue: Arc<Mutex<Queue>>, irq: u32) {
    // There's no stop mechanism, but we don't destroy devices anyway, so that's okay.
    std::thread::Builder::new().name("virtio-network".to_owned()).spawn(move || {
        let mut buffer = [0; 2048];
        loop {
            let len = iface.recv(&mut buffer).unwrap();
            send_packet(&queue, &buffer[..len], irq);
        }
    }).unwrap();
}

impl Network {
    pub fn new(irq: u32, net: impl NetworkDevice + 'static, mac: [u8; 6]) -> Network {
        let queue = Arc::new(Mutex::new(Queue::new()));
        let net = Arc::new(net);
        Network {
            net,
            status: 0,
            rx: queue,
            tx: Queue::new(),
            mac,
            irq,
        }
    }
}

impl Device for Network {
    fn device_id(&self) -> DeviceId { DeviceId::Network }
    fn device_feature(&self) -> u32 { 1 << VIRTIO_BLK_F_MAC }
    fn driver_feature(&mut self, _value: u32) {}
    fn get_status(&self) -> u32 { self.status }
    fn set_status(&mut self, status: u32) { self.status = status }
    fn config_space(&self) -> &[u8] { &self.mac }
    fn num_queues(&self) -> usize { 2 } 
    fn with_queue(&mut self, idx: usize, f: &mut dyn FnMut(&mut Queue)) {
        if idx == 0 {
            f(&mut self.rx.lock())
        } else {
            f(&mut self.tx)
        }
        // TODO: If thread is running, we should terminate it before return here
    }
    fn reset(&mut self) {
        self.status = 0;
        self.rx.lock().reset();
        self.tx.reset();
    }
    fn notify(&mut self, idx: usize) {
        if idx == 0 {
            eprintln!("Filling receving queue");
            return;
        }
        while let Some(buffer) = self.tx.take() {
            let mut reader = buffer.reader();

            let hdr_len = std::mem::size_of::<VirtioNetHeader>();
            if reader.len() <= hdr_len {
                // Unexpected packet
                error!(target: "VirtioNet", "illegal transmission with size {} smaller than header {}", reader.len(), hdr_len);
            }

            // We don't need any of the fields of the header, so just skip it.
            let packet_len = reader.len() - hdr_len;
            reader.seek(SeekFrom::Start(hdr_len as u64)).unwrap();

            let mut io_buffer = Vec::with_capacity(packet_len);
            unsafe { io_buffer.set_len(io_buffer.capacity()) };
            reader.read_exact(&mut io_buffer).unwrap();
            unsafe { self.tx.put(buffer); }

            self.net.send(&io_buffer).unwrap();
        }

        crate::emu::PLIC.lock().trigger(self.irq);
    }

    fn queue_ready(&mut self, idx: usize) {
        if idx == 0 {
            thread_run(self.net.clone(), self.rx.clone(), self.irq);
        }
    }
}
