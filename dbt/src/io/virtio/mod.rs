mod queue;
mod mmio;

pub use queue::{Queue, Buffer};
pub use mmio::Mmio;

#[derive(Clone, Copy)]
pub enum DeviceId {
    Reserved = 0,
    Network = 1,
    Block = 2,
    Entropy = 4,
    #[doc(hidden)]
    __Nonexhaustive,
}

pub trait Device {
    /// Indicate what kind of device it is.
    fn device_id(&self) -> DeviceId;

    /// Indicate a list of supported features.
    fn device_feature(&self) -> u32;

    /// Signal to the device that a feature is selected by the driver.
    fn driver_feature(&mut self, value: u32);

    /// Retrieve the status field.
    fn get_status(&self) -> u32;

    /// Update the status by the driver.
    fn set_status(&mut self, status: u32);

    /// Get the configuration space. In current implementation this is readonly.
    fn config_space(&self) -> &[u8];

    /// Get queues associated with the device
    fn queues(&mut self) -> &mut [Queue];

    /// Reset a device
    fn reset(&mut self);

    /// Notify that a buffer has been queued
    fn notify(&mut self, idx: usize);
}
