use super::{Device, DeviceId, Queue};
use io::IrqPin;
use parking_lot::Mutex;
use std::io::Read;
use std::sync::Arc;

/// A virtio entropy source device.
pub struct Rng {
    status: u32,
    inner: Arc<Mutex<Inner>>,
}

/// struct used by task
struct Inner {
    rng: Box<dyn io::entropy::Entropy + Send>,
    irq: Box<dyn IrqPin>,
}

impl Rng {
    /// Create a virtio entropy source device using a given random number generator.
    pub fn new(irq: Box<dyn IrqPin>, rng: Box<dyn io::entropy::Entropy + Send>) -> Rng {
        let inner = Arc::new(Mutex::new(Inner { rng, irq }));
        Rng { status: 0, inner }
    }
}

fn start_task(inner: Arc<Mutex<Inner>>, mut queue: Queue) {
    crate::event_loop().spawn(async move {
        while let Ok(mut buffer) = queue.take().await {
            let mut inner = inner.lock();
            let rng: &mut dyn rand::RngCore = &mut inner.rng;
            let mut writer = buffer.writer();
            std::io::copy(&mut rng.take(writer.len() as u64), &mut writer).unwrap();
            drop(buffer);
            inner.irq.pulse();
        }
    })
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
        start_task(self.inner.clone(), queue);
    }
}
