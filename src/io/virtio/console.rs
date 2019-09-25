use super::{Device, DeviceId, Queue};
use std::io::{Read, Write};
use std::sync::Arc;
use parking_lot::Mutex;

/// A virtio entropy source device.
pub struct Console {
    status: u32,
    rx: Arc<Mutex<Queue>>,
    tx: Queue,
    irq: u32,
}

fn put(queue: &Arc<Mutex<Queue>>, buf: &[u8], irq: u32) {
    let mut queue = queue.lock();
    if let Some(mut buffer) = queue.take() {
        let mut writer = buffer.writer();
        writer.write_all(buf).unwrap();
        unsafe { queue.put(buffer) };

        crate::emu::PLIC.lock().trigger(irq);
    } else {
        info!(
            target: "VirtioConsole",
            "discard packet of size {:x} because there is no buffer in receiver queue",
            buf.len()
        );
    }
}

fn thread_run(queue: Arc<Mutex<Queue>>, irq: u32) {
    std::thread::Builder::new().name("virtio-console".to_owned()).spawn(move || {
        let mut buffer = [0; 2048];
        loop {
            let len = crate::io::console::CONSOLE.recv(&mut buffer).unwrap();
            put(&queue, &buffer[..len], irq);
        }
    }).unwrap();
}

impl Console {
    pub fn new(irq: u32) -> Console {
        let queue = Arc::new(Mutex::new(Queue::new_with_max(128)));
        Console {
            status: 0,
            rx: queue,
            tx: Queue::new(),
            irq,
        }
    }
}

impl Device for Console {
    fn device_id(&self) -> DeviceId { DeviceId::Console }
    fn device_feature(&self) -> u32 { 0 }
    fn driver_feature(&mut self, _value: u32) {}
    fn get_status(&self) -> u32 { self.status }
    fn set_status(&mut self, status: u32) { self.status = status }
    fn config_space(&self) -> &[u8] { &[] }
    fn num_queues(&self) -> usize { 2 } 
    fn with_queue(&mut self, idx: usize, f: &mut dyn FnMut(&mut Queue)) {
        if idx == 0 {
            f(&mut self.rx.lock())
        } else {
            f(&mut self.tx)
        }
    }
    fn reset(&mut self) {
        self.status = 0;
        self.rx.lock().reset();
        self.tx.reset();
        // TODO: If thread is running, we should terminate it before return here
    }
    fn notify(&mut self, idx: usize) {
        if idx == 0 {
            eprintln!("Filling receving queue");
            return;
        }
        while let Some(buffer) = self.tx.take() {
            let mut reader = buffer.reader();

            let mut io_buffer = Vec::with_capacity(reader.len());
            unsafe { io_buffer.set_len(io_buffer.capacity()) };
            reader.read_exact(&mut io_buffer).unwrap();
            unsafe { self.tx.put(buffer); }

            crate::io::console::CONSOLE.send(&io_buffer).unwrap();
        }

        crate::emu::PLIC.lock().trigger(self.irq);
    }
    fn queue_ready(&mut self, idx: usize) {
        if idx == 0 {
            thread_run(self.rx.clone(), self.irq);
        }
    }
}
