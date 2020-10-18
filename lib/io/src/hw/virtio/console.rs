use super::{Device, DeviceId, Queue};
use crate::serial::Serial;
use crate::{IrqPin, RuntimeContext};
use futures::future::{AbortHandle, Abortable};
use parking_lot::Mutex;
use std::sync::Arc;

const VIRTIO_CONSOLE_F_SIZE: usize = 0;

fn size_to_config(col: u16, row: u16) -> [u8; 4] {
    let col = col.to_le_bytes();
    let row = row.to_le_bytes();
    [col[0], col[1], row[0], row[1]]
}

/// A virtio console device.
pub struct Console {
    status: u32,
    /// Whether sizing feature should be available. We allow it to be turned off because sometimes
    /// STDIN is not tty.
    resize: bool,
    rx_handle: Option<AbortHandle>,
    resize_handle: Option<AbortHandle>,
    ctx: Arc<dyn RuntimeContext>,
    inner: Arc<Inner>,
}

struct Inner {
    console: Box<dyn Serial>,
    irq: Box<dyn IrqPin>,
    // Keep config with whether it has changed
    config: Mutex<([u8; 4], bool)>,
}

impl Drop for Console {
    fn drop(&mut self) {
        self.rx_handle.take().map(|x| x.abort());
        self.resize_handle.take().map(|x| x.abort());
    }
}

impl Console {
    /// Create a virtio console device.
    ///
    /// Parameter `resize` may be used to turn resizing detection on or off.
    pub fn new(
        ctx: Arc<dyn RuntimeContext>,
        irq: Box<dyn IrqPin>,
        console: Box<dyn Serial>,
        mut resize: bool,
    ) -> Console {
        let (col, row) = if resize {
            match console.get_window_size() {
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
        let config = Mutex::new((size_to_config(col, row), resize));
        let inner = Arc::new(Inner { console, irq, config });
        let mut ret =
            Console { status: 0, resize, rx_handle: None, resize_handle: None, ctx, inner };

        ret.resize_handle = if resize { Some(ret.start_resize()) } else { None };

        ret
    }

    fn start_rx(&self, mut rx: Queue) -> AbortHandle {
        let inner = self.inner.clone();
        let (handle, reg) = AbortHandle::new_pair();
        self.ctx.spawn(Box::pin(async move {
            let _ = Abortable::new(async move {
                let mut buffer = [0; 2048];
                loop {
                    let len = inner.console.read(&mut buffer).await.unwrap();
                    if let Ok(Some(mut dma_buffer)) = rx.try_take().await {
                        let mut writer = dma_buffer.writer();
                        writer.write_all(&buffer[..len]).await.unwrap();
                        rx.put(dma_buffer).await;

                        inner.irq.pulse();
                    } else {
                        info!(
                            target: "VirtioConsole",
                            "discard packet of size {:x} because there is no buffer in receiver queue",
                            len
                        );
                    }
                }
            }, reg).await;
        }));
        handle
    }

    fn start_tx(&self, mut tx: Queue) {
        let inner = self.inner.clone();
        self.ctx.spawn(Box::pin(async move {
            while let Ok(buffer) = tx.take().await {
                let mut reader = buffer.reader();

                let mut io_buffer = Vec::with_capacity(reader.len());
                unsafe { io_buffer.set_len(io_buffer.capacity()) };
                reader.read_exact(&mut io_buffer).await.unwrap();
                tx.put(buffer).await;

                inner.console.write(&io_buffer).await.unwrap();
                inner.irq.pulse();
            }
        }));
    }

    fn start_resize(&self) -> AbortHandle {
        let inner = self.inner.clone();
        let (handle, reg) = AbortHandle::new_pair();
        self.ctx.spawn(Box::pin(async move {
            let _ = Abortable::new(
                async move {
                    loop {
                        inner.console.wait_window_size_changed().await.unwrap();
                        let (col, row) = inner.console.get_window_size().unwrap();
                        let mut guard = inner.config.lock();
                        guard.0 = size_to_config(col, row);
                        guard.1 = true;
                        inner.irq.pulse();
                    }
                },
                reg,
            )
            .await;
        }));
        handle
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
        f(&self.inner.config.lock().0)
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
            self.rx_handle = Some(self.start_rx(queue));
        } else {
            self.start_tx(queue);
        }
    }

    fn interrupt_status(&mut self) -> u32 {
        if self.inner.config.lock().1 { 3 } else { 1 }
    }

    fn interrupt_ack(&mut self, ack: u32) {
        if ack & 2 != 0 {
            self.inner.config.lock().1 = false
        }
    }
}
