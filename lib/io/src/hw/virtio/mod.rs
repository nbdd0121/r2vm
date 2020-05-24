use std::convert::TryInto;

mod mmio;
mod queue;
pub use mmio::Mmio;
pub use queue::{Buffer, BufferReader, BufferWriter, Queue, QueueNotReady};

#[cfg(feature = "virtio-network")]
mod network;
#[cfg(feature = "virtio-network")]
pub use network::Network;

#[cfg(feature = "virtio-block")]
mod block;
#[cfg(feature = "virtio-block")]
pub use block::Block;

#[cfg(feature = "virtio-rng")]
mod rng;
#[cfg(feature = "virtio-rng")]
pub use rng::Rng;

#[cfg(feature = "virtio-p9")]
mod p9;
#[cfg(feature = "virtio-p9")]
pub use self::p9::P9;

#[cfg(feature = "virtio-console")]
mod console;
#[cfg(feature = "virtio-console")]
pub use console::Console;

/// Types of virtio devices.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum DeviceId {
    Reserved = 0,
    Network = 1,
    Block = 2,
    Console = 3,
    Entropy = 4,
    P9 = 9,
}

/// A transport-agnostic abstraction of virtio devices.
pub trait Device: Send {
    /// Indicate what kind of device it is.
    fn device_id(&self) -> DeviceId;

    /// Indicate a list of supported features.
    fn device_feature(&self) -> u32 {
        0
    }

    /// Signal to the device that a feature is selected by the driver.
    fn driver_feature(&mut self, _value: u32) {}

    /// Retrieve the status field.
    fn get_status(&self) -> u32;

    /// Update the status by the driver.
    fn set_status(&mut self, status: u32);

    /// Get the configuration space.
    fn config_space(&self) -> &[u8] {
        unimplemented!()
    }

    /// Get the configuration space, callback form.
    fn with_config_space(&self, f: &mut dyn FnMut(&[u8])) {
        f(self.config_space())
    }

    /// Read from config space. Reading might have a side-effect, therefore it takes `&mut self`.
    fn config_read(&mut self, offset: usize, size: u32) -> u64 {
        let mut value = 0;
        self.with_config_space(&mut |config| {
            if offset + size as usize > config.len() {
                error!(target: "Mmio", "out-of-bound config register read 0x{:x}", offset);
                return;
            }
            let slice = &config[offset..offset + size as usize];
            value = match size {
                8 => u64::from_le_bytes(slice.try_into().unwrap()) as u64,
                4 => u32::from_le_bytes(slice.try_into().unwrap()) as u64,
                2 => u16::from_le_bytes(slice.try_into().unwrap()) as u64,
                _ => slice[0] as u64,
            };
        });
        value
    }

    /// Write to the config space.
    fn config_write(&mut self, offset: usize, value: u64, _size: u32) {
        error!(target: "Mmio", "config register write 0x{:x} = 0x{:x}", offset, value);
    }

    /// Get number of queues of this device
    fn num_queues(&self) -> usize;

    /// Get the maximum length of a queue.
    fn max_queue_len(&self, _idx: usize) -> u16 {
        32768
    }

    /// Reset a device
    fn reset(&mut self);

    /// Notify the device that the queue is ready
    fn queue_ready(&mut self, idx: usize, queue: Queue);

    /// Query what has caused the interrupt to be sent.
    fn interrupt_status(&mut self) -> u32 {
        1
    }

    /// Answer the interrupt.
    fn interrupt_ack(&mut self, _ack: u32) {}
}
