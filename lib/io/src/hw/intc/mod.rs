//! Interrupt controllers.

#[cfg(feature = "intc-clint")]
mod clint;
#[cfg(feature = "intc-clint")]
pub use clint::Clint;

#[cfg(feature = "intc-plic")]
mod plic;
#[cfg(feature = "intc-plic")]
pub use plic::Plic;
