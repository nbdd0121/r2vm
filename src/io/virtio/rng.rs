use super::{Device, DeviceId, Queue};
use rand::SeedableRng;
use std::io::Read;

/// A virtio entropy source device.
pub struct Rng {
    status: u32,
    queue: Queue,
    rng: Box<dyn rand::RngCore + Send>,
    irq: u32,
}

impl Rng {
    /// Create a virtio entropy source device using a given random number generator.
    pub fn new(irq: u32, rng: Box<dyn rand::RngCore + Send>) -> Rng {
        Rng {
            status: 0,
            queue: Queue::new(),
            rng,
            irq,
        }
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

impl Device for Rng {
    fn device_id(&self) -> DeviceId { DeviceId::Entropy }
    fn device_feature(&self) -> u32 { 0 }
    fn driver_feature(&mut self, _value: u32) {}
    fn get_status(&self) -> u32 { self.status }
    fn set_status(&mut self, status: u32) { self.status = status }
    fn config_space(&self) -> &[u8] { &[] }
    fn num_queues(&self) -> usize { 1 }
    fn with_queue(&mut self, _idx: usize, f: &mut dyn FnMut(&mut Queue)) { f(&mut self.queue) }
    fn reset(&mut self) {
        self.status = 0;
        self.queue.reset();
    }
    fn notify(&mut self, _idx: usize) {
        while let Some(mut buffer) = self.queue.try_take() {
            let rng: &mut dyn rand::RngCore = &mut self.rng;
            let mut writer = buffer.writer();
            std::io::copy(&mut rng.take(writer.len() as u64), &mut writer).unwrap();
            unsafe { self.queue.put(buffer); }
        }

        crate::emu::PLIC.lock().trigger(self.irq);
    }
}
