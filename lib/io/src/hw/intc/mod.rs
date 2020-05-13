//! Interrupt controllers.

#[cfg(feature = "hw-clint")]
mod clint;
#[cfg(feature = "hw-clint")]
pub use clint::Clint;
