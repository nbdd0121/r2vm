use super::{Device, DeviceId, Queue};
use rand::SeedableRng;

/// A virtio entropy source device.
pub struct Rng {
    status: u32,
    queue: Queue,
    rng: Box<dyn rand::RngCore>,
}

impl Rng {
    /// Create a virtio entropy source device using a given random number generator.
    pub fn new(rng: Box<dyn rand::RngCore>) -> Rng {
        Rng {
            status: 0,
            queue: Queue::new(),
            rng,
        }
    }

    /// Create a virtio entropy source device, fulfilled by OS's entropy source.
    pub fn new_os() -> Rng {
        Self::new(Box::new(rand::rngs::OsRng::new().unwrap()))
    }

    /// Create a virtio entropy source device with a fixed seed.
    /// **This is not cryptographically secure!!!**
    pub fn new_seeded() -> Rng {
        Self::new(Box::new(rand::rngs::SmallRng::seed_from_u64(0xcafebabedeadbeef)))
    }
}

impl Device for Rng {
    fn device_id(&self) -> DeviceId { DeviceId::Entropy }
    fn device_feature(&self) -> u32 { 0 }
    fn driver_feature(&mut self, _value: u32) {}
    fn get_status(&self) -> u32 { self.status }
    fn set_status(&mut self, status: u32) { self.status = status }
    fn config_space(&self) -> &[u8] { &[] }
    fn queues(&mut self) -> &mut [Queue] {
        std::slice::from_mut(&mut self.queue)
    }
    fn reset(&mut self) {
        self.status = 0;
        self.queue.reset();
    }
    fn notify(&mut self, _idx: usize) {
        while let Some(mut buffer) = self.queue.take() {
            let mut io_buffer = Vec::with_capacity(buffer.write_len());
            unsafe { io_buffer.set_len(io_buffer.capacity()) };
            self.rng.fill_bytes(&mut io_buffer);
            buffer.write(0, &io_buffer);
            unsafe { self.queue.put(buffer); }
        }

        unsafe { crate::emu::PLIC.as_mut().unwrap().trigger(2) };
    }
}
