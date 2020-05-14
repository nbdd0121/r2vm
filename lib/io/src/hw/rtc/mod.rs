//! Real-time clock devices.

#[cfg(feature = "rtc-zyncmp")]
mod zyncmp;
#[cfg(feature = "rtc-zyncmp")]
pub use zyncmp::ZyncMp;
