#[macro_use]
extern crate log;

pub mod hw;

pub mod block;
pub mod network;
pub mod serial;

#[cfg(feature = "entropy")]
pub mod entropy;
#[cfg(feature = "fs")]
pub mod fs;
#[cfg(feature = "system")]
pub mod system;

use futures::future::BoxFuture;
use std::time::Duration;

/// Context for device DMA operations.
///
/// This is required to be [`Send`] + [`Sync`] so devices that use them can be `Send`.
pub trait DmaContext: Send + Sync {
    /// Perform a DMA read at given address.
    fn dma_read<'asyn>(&'asyn self, addr: u64, buf: &'asyn mut [u8]) -> BoxFuture<'asyn, ()>;

    /// Perform a DMA write at given address.
    fn dma_write<'asyn>(&'asyn self, addr: u64, buf: &'asyn [u8]) -> BoxFuture<'asyn, ()>;

    /// Read a half word atomically
    fn read_u16<'asyn>(&'asyn self, addr: u64) -> BoxFuture<'asyn, u16>;

    /// Write a half word atomically
    fn write_u16<'asyn>(&'asyn self, addr: u64, value: u16) -> BoxFuture<'asyn, ()>;
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

/// An interrupt pin.
///
/// This is required to be [`Send`] + [`Sync`] so devices that use them can be `Send`.
pub trait IrqPin: Send + Sync {
    /// Set the IRQ level.
    fn set_level(&self, level: bool);

    /// Set the IRQ level to high.
    fn raise(&self) {
        self.set_level(true);
    }

    /// Set the IRQ level to low.
    fn lower(&self) {
        self.set_level(false);
    }

    /// Send a pulse that raise the IRQ level to high and then to low.
    fn pulse(&self) {
        self.raise();
        self.lower();
    }
}

/// An I/O memory region.
///
/// IoMemory represents a region of physically continuous I/O memory.
///
/// `IoMemory` requires [`Send`] + [`Sync`] so it is suitable for multi-threaded access.
pub trait IoMemory: Send + Sync {
    /// Read from I/O memory. `size` can be either 1, 2, 4 or 8. `addr` must be aligned properly,
    /// e.g. when `size` is 4, the least significant 2 bits of `addr` should be zero.
    ///
    /// `addr` has type `usize`, which depends on the architecture. However, we believe that there
    /// is never a use-case for continuous memory region with more than 4GiB size on a 32-bit
    /// machine on 32-bit machine. Use `usize` over `u64` makes it much easier to handle indexes.
    fn read(&self, addr: usize, size: u32) -> u64;

    /// Write to I/O memory. `size` can be either 1, 2, 4 or 8. `addr` must be aligned properly,
    /// e.g. when `size` is 4, the least significant 2 bits of `addr` should be zero.
    fn write(&self, addr: usize, value: u64, size: u32);
}

/// An I/O memory region requiring mutable reference.
///
/// IoMemory represents a region of physically continuous I/O memory.
pub trait IoMemoryMut {
    /// Read from I/O memory. `size` can be either 1, 2, 4 or 8. `addr` must be aligned properly,
    /// e.g. when `size` is 4, the least significant 2 bits of `addr` should be zero.
    fn read_mut(&mut self, addr: usize, size: u32) -> u64;

    /// Write to I/O memory. `size` can be either 1, 2, 4 or 8. `addr` must be aligned properly,
    /// e.g. when `size` is 4, the least significant 2 bits of `addr` should be zero.
    fn write_mut(&mut self, addr: usize, value: u64, size: u32);
}

impl<T: IoMemory> IoMemory for &'_ T {
    fn read(&self, addr: usize, size: u32) -> u64 {
        (**self).read(addr, size)
    }

    fn write(&self, addr: usize, value: u64, size: u32) {
        (**self).write(addr, value, size)
    }
}

impl<T: IoMemory + ?Sized> IoMemoryMut for T {
    fn read_mut(&mut self, addr: usize, size: u32) -> u64 {
        self.read(addr, size)
    }
    fn write_mut(&mut self, addr: usize, value: u64, size: u32) {
        self.write(addr, value, size)
    }
}

impl<R: lock_api::RawMutex + Send + Sync, T: IoMemoryMut + Send> IoMemory
    for lock_api::Mutex<R, T>
{
    fn read(&self, addr: usize, size: u32) -> u64 {
        self.lock().read_mut(addr, size)
    }
    fn write(&self, addr: usize, value: u64, size: u32) {
        self.lock().write_mut(addr, value, size)
    }
}
