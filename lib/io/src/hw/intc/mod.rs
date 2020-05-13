//! Interrupt controllers.

#[cfg(feature = "hw-clint")]
mod clint;
#[cfg(feature = "hw-clint")]
pub use clint::Clint;

#[cfg(feature = "hw-plic")]
mod plic;
#[cfg(feature = "hw-plic")]
pub use plic::Plic;
