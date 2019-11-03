use super::{Device, DeviceId, Queue};
use parking_lot::Mutex;
use rand::SeedableRng;
use std::io::Read;
use std::sync::Arc;

/// A virtio entropy source device.
pub struct Rng {
    status: u32,
    inner: Arc<Mutex<Inner>>,
}

/// struct used by task
struct Inner {
    rng: Box<dyn rand::RngCore + Send>,
    irq: u32,
}

impl Rng {
    /// Create a virtio entropy source device using a given random number generator.
    pub fn new(irq: u32, rng: Box<dyn rand::RngCore + Send>) -> Rng {
        let inner = Arc::new(Mutex::new(Inner { rng, irq }));
        Rng { status: 0, inner }
    }

    /// Create a virtio entropy source device, fulfilled by OS's entropy source.
    pub fn new_os(irq: u32) -> Rng {
        Self::new(irq, Box::new(rand::rngs::OsRng))
    }

    /// Create a virtio entropy source device with a fixed seed.
    /// **This is not cryptographically secure!!!**
    pub fn new_seeded(irq: u32, seed: u64) -> Rng {
        Self::new(irq, Box::new(rand::rngs::StdRng::seed_from_u64(seed)))
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
            crate::emu::PLIC.lock().trigger(inner.irq);
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
