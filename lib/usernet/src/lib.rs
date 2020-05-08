//! An userspace network emulation stack for Rust.
//!
//! Currently the underlying implementation is `slirp`, the same library used in QEMU and other
//! full-system emulators.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

mod config;
mod slirp;
mod util;

pub use config::*;
pub use slirp::Network;

/// An async context which provides timer and spawning services.
///
/// This library aims to handle as much heavy-lifting as possible, while maintain a flexible
/// interface exposed to the user. Therefore which async runtime and timing service to use is
/// decided by the user, and not bundled to the dependency of this library.
pub trait Context {
    /// Get the current time, in nanoseconds, relative to an arbitary reference point.
    fn now(&mut self) -> Duration;

    /// Create a timer that fires at specified time point relative to the reference.
    fn create_timer(&mut self, time: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>>;

    /// Spawn a task to run until completion.
    fn spawn(&mut self, future: Pin<Box<dyn Future<Output = ()> + Send>>);
}
