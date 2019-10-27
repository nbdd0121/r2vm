use super::{Device, DeviceId, Queue};
use parking_lot::Mutex;
use std::io::{Read, Write};
use std::sync::Arc;

const VIRTIO_CONSOLE_F_SIZE: usize = 0;

fn size_to_config(col: u16, row: u16) -> [u8; 4] {
    let col = col.to_le_bytes();
    let row = row.to_le_bytes();
    [col[0], col[1], row[0], row[1]]
}

/// A virtio entropy source device.
pub struct Console {
    status: u32,
    tx: Queue,
    irq: u32,
    /// Whether sizing feature should be available. We allow it to be turned off because sometimes
    /// STDIN is not tty.
    resize: bool,
    // Keep config with whether it has changed
    config: Arc<Mutex<([u8; 4], bool)>>,
}

fn put(queue: &mut Queue, buf: &[u8], irq: u32) {
    if let Ok(Some(mut buffer)) = queue.try_take() {
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

fn thread_run(mut queue: Queue, irq: u32) {
    std::thread::Builder::new()
        .name("virtio-console".to_owned())
        .spawn(move || {
            let mut buffer = [0; 2048];
            loop {
                let len = crate::io::console::CONSOLE.recv(&mut buffer).unwrap();
                put(&mut queue, &buffer[..len], irq);
            }
        })
        .unwrap();
}

impl Drop for Console {
    fn drop(&mut self) {
        crate::io::console::CONSOLE.on_size_change(None);
    }
}

impl Console {
    pub fn new(irq: u32, mut resize: bool) -> Console {
        let (col, row) = if resize {
            match crate::io::console::CONSOLE.get_size() {
                Ok(v) => v,
                Err(_) => {
                    warn!(
                        target: "VirtioConsole",
                        "Cannot query the size of the console. If you are not working with tty, consider turn console.resize to false in the config to suppress this warning."
                    );
                    resize = false;
                    (0, 0)
                }
            }
        } else {
            (0, 0)
        };

        // Mark the config changed by default so the driver will poll
        // the size from the very beginning.
        let config = Arc::new(Mutex::new((size_to_config(col, row), resize)));

        if resize {
            let callback = {
                let config = config.clone();
                Box::new(move || {
                    let (col, row) = match crate::io::console::CONSOLE.get_size() {
                        Err(_) => return,
                        Ok(v) => v,
                    };
                    let mut guard = config.lock();
                    guard.0 = size_to_config(col, row);
                    guard.1 = true;
                    crate::emu::PLIC.lock().trigger(irq);
                })
            };
            crate::io::console::CONSOLE.on_size_change(Some(callback));
        }

        Console { status: 0, tx: Queue::new(), irq, resize, config }
    }
}

impl Device for Console {
    fn device_id(&self) -> DeviceId {
        DeviceId::Console
    }
    fn device_feature(&self) -> u32 {
        if self.resize { 1 << VIRTIO_CONSOLE_F_SIZE } else { 0 }
    }

    fn driver_feature(&mut self, _value: u32) {}
    fn get_status(&self) -> u32 {
        self.status
    }
    fn set_status(&mut self, status: u32) {
        self.status = status
    }
    fn with_config_space(&self, f: &mut dyn FnMut(&[u8])) {
        f(&self.config.lock().0)
    }
    fn num_queues(&self) -> usize {
        2
    }
    fn max_queue_len(&self, idx: usize) -> u16 {
        if idx == 0 { 128 } else { 32768 }
    }
    fn reset(&mut self) {
        self.status = 0;
        // TODO: If thread is running, we should terminate it before return here
    }
    fn notify(&mut self, idx: usize) {
        if idx == 0 {
            return;
        }
        while let Ok(Some(buffer)) = self.tx.try_take() {
            let mut reader = buffer.reader();

            let mut io_buffer = Vec::with_capacity(reader.len());
            unsafe { io_buffer.set_len(io_buffer.capacity()) };
            reader.read_exact(&mut io_buffer).unwrap();
            unsafe {
                self.tx.put(buffer);
            }

            crate::io::console::CONSOLE.send(&io_buffer).unwrap();
        }

        crate::emu::PLIC.lock().trigger(self.irq);
    }
    fn queue_ready(&mut self, idx: usize, queue: Queue) {
        if idx == 0 {
            thread_run(queue, self.irq);
        } else {
            self.tx = queue;
        }
    }

    fn interrupt_status(&mut self) -> u32 {
        if self.config.lock().1 { 3 } else { 1 }
    }

    fn interrupt_ack(&mut self, ack: u32) {
        if ack & 2 != 0 {
            self.config.lock().1 = false
        }
    }
}
