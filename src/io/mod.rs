pub mod console;
pub mod network;
pub mod plic;
pub mod rtc;
pub mod virtio;
pub trait IoContext: io::DmaContext + io::RuntimeContext {}
use io::IrqPin;
