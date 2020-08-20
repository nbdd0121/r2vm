//! Console devices.

#[cfg(feature = "console-ns16550")]
mod ns16550;
#[cfg(feature = "console-ns16550")]
pub use ns16550::NS16550;
