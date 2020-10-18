use super::{Device, DeviceId, Queue};
use crate::{IrqPin, RuntimeContext};
use parking_lot::Mutex;
use std::sync::Arc;

/// A virtio entropy source device.
pub struct Rng {
    status: u32,
    ctx: Arc<dyn RuntimeContext>,
    inner: Arc<Mutex<Inner>>,
}

/// struct used by task
struct Inner {
    rng: Box<dyn crate::entropy::Entropy + Send>,
    irq: Box<dyn IrqPin>,
}

impl Rng {
    /// Create a virtio entropy source device using a given random number generator.
    pub fn new(
        ctx: Arc<dyn RuntimeContext>,
        irq: Box<dyn IrqPin>,
        rng: Box<dyn crate::entropy::Entropy + Send>,
    ) -> Rng {
        let inner = Arc::new(Mutex::new(Inner { rng, irq }));
        Rng { status: 0, inner, ctx }
    }

    fn start_task(&self, mut queue: Queue) {
        let inner = self.inner.clone();
        self.ctx.spawn(Box::pin(async move {
            while let Ok(mut buffer) = queue.take().await {
                let mut writer = buffer.writer();
                let mut buf = Vec::with_capacity(writer.len());
                {
                    let mut inner = inner.lock();
                    let rng: &mut dyn rand::RngCore = &mut inner.rng;
                    unsafe {
                        buf.set_len(writer.len());
                    }
                    rng.fill_bytes(&mut buf);
                }
                writer.write_all(&buf).await.unwrap();
                queue.put(buffer).await;
                inner.lock().irq.pulse();
            }
        }))
    }
}

impl Device for Rng {
    fn device_id(&self) -> DeviceId {
        DeviceId::Entropy
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
        &[]
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
