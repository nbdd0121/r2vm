#[cfg(feature = "block")]
pub mod block;
#[cfg(feature = "entropy")]
pub mod entropy;
#[cfg(feature = "network")]
pub mod network;

use futures::future::BoxFuture;
use std::time::Duration;

/// Context for device DMA operations.
///
/// This is required to be [`Send`] + [`Sync`] so devices that use them can be `Send`.
pub trait DmaContext: Send + Sync {
    /// Perform a DMA read at given address.
    fn dma_read(&self, addr: u64, buf: &mut [u8]);

    /// Perform a DMA write at given address.
    fn dma_write(&self, addr: u64, buf: &[u8]);

    /// Read a half word atomically
    fn read_u16(&self, addr: u64) -> u16;

    /// Write a half word atomically
    fn write_u16(&self, addr: u64, value: u16);
}

/// Context for I/O event loop runtime.
///
/// `RuntimeContext` abstracts away the underlying event loop's runtime, regardless whether it's tokio,
/// async_std or custom implementation. This is supported by `spawn` and `spawn_blocking` functions.
/// We also require `now` and `create_timer` function so the runtime can time in its own fashion.
///
/// This is required to be [`Send`] + [`Sync`] so devices that use them can be `Send`.
pub trait RuntimeContext: Send + Sync {
    /// Get the current time since an arbitary epoch.
    fn now(&self) -> Duration;

    /// Get a [`Future`] that is triggered at the supplied time since the epoch.
    ///
    /// [`Future`]: std::future::Future
    fn create_timer(&self, time: Duration) -> BoxFuture<'static, ()>;

    /// Spawn a task.
    ///
    /// Most devices need to run a event loop, which we models into a Rust task. The I/O context
    /// therefore must provide a runtime for this to be carried out.
    fn spawn(&self, task: BoxFuture<'static, ()>);

    /// Spawn a task which may possibly block to perform IO.
    fn spawn_blocking(&self, name: &str, task: BoxFuture<'static, ()>);
}
