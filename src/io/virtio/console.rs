use super::{Device, DeviceId, Queue};
use futures::future::AbortHandle;
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
    irq: u32,
    /// Whether sizing feature should be available. We allow it to be turned off because sometimes
    /// STDIN is not tty.
    resize: bool,
    // Keep config with whether it has changed
    config: Arc<Mutex<([u8; 4], bool)>>,
    rx_handle: Option<AbortHandle>,
}

fn start_rx(mut rx: Queue, irq: u32) -> AbortHandle {
    crate::event_loop().spawn_abortable(async move {
        let mut buffer = [0; 2048];
        loop {
            let len = crate::io::console::CONSOLE.recv(&mut buffer).await.unwrap();
            if let Ok(Some(mut dma_buffer)) = rx.try_take() {
                let mut writer = dma_buffer.writer();
                writer.write_all(&buffer[..len]).unwrap();
                unsafe { rx.put(dma_buffer) };

                crate::emu::PLIC.lock().trigger(irq);
            } else {
                info!(
                    target: "VirtioConsole",
                    "discard packet of size {:x} because there is no buffer in receiver queue",
                    len
                );
            }
        }
    })
}

fn start_tx(mut tx: Queue, irq: u32) {
    crate::event_loop().spawn(async move {
        while let Ok(buffer) = tx.take().await {
            let mut reader = buffer.reader();

            let mut io_buffer = Vec::with_capacity(reader.len());
            unsafe { io_buffer.set_len(io_buffer.capacity()) };
            reader.read_exact(&mut io_buffer).unwrap();
            unsafe {
                tx.put(buffer);
            }

            crate::io::console::CONSOLE.send(&io_buffer).unwrap();
            crate::emu::PLIC.lock().trigger(irq);
        }
    });
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

        Console { status: 0, irq, resize, config, rx_handle: None }
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
        self.rx_handle.take().map(|x| x.abort());
    }
    fn queue_ready(&mut self, idx: usize, queue: Queue) {
        if idx == 0 {
            self.rx_handle.take().map(|x| x.abort());
            self.rx_handle = Some(start_rx(queue, self.irq));
        } else {
            start_tx(queue, self.irq);
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
