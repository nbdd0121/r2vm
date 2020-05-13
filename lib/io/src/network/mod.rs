//! Network devices.
//!
//! This module provides a [`Network`] trait which bridges underlying network device implementation and
//! I/O devices that behaves as MACs.

use futures::future::poll_fn;
use std::io::Result;
use std::task::{Context, Poll};

#[cfg(feature = "network-logger")]
mod logger;
#[cfg(feature = "network-logger")]
pub use logger::Logger;
#[cfg(feature = "network-usernet")]
mod usernet;
#[cfg(feature = "network-usernet")]
pub use self::usernet::Usernet;

/// Abstraction of a network device.
pub trait Network: Send + Sync {
    /// Attempt to send a packet to the device.
    fn poll_send(&self, ctx: &mut Context, buf: &[u8]) -> Poll<Result<usize>>;

    /// Attempt to receive a packet from the device.
    fn poll_recv(&self, ctx: &mut Context, buf: &mut [u8]) -> Poll<Result<usize>>;
}

impl dyn Network {
    /// Send a packet to the device.
    pub async fn send(&self, buf: &[u8]) -> Result<usize> {
        poll_fn(|cx| self.poll_send(cx, buf)).await
    }

    /// Receive a packet from the device.
    pub async fn recv(&self, buf: &mut [u8]) -> Result<usize> {
        poll_fn(|cx| self.poll_recv(cx, buf)).await
    }
}
