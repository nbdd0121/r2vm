//! Serial devices.
//!
//! This module provides a [`Serial`] trait which bridges underlying serial device implementation and
//! I/O console devices.

use futures::future::poll_fn;
use std::io::{Error, ErrorKind, Result};
use std::task::{Context, Poll};

#[cfg(feature = "serial-console")]
mod console;
#[cfg(feature = "serial-console")]
pub use console::Console;

/// Abstraction of a serial device.
///
/// Unlike [`Network`] devices, data read from and written to serial devices are not packetised.
///
/// [`Network`]: super::network::Network
pub trait Serial: Send + Sync {
    /// Attempt to write to the device.
    fn poll_write(&self, cx: &mut Context, buf: &[u8]) -> Poll<Result<usize>>;

    /// Attempt to read from the device.
    fn poll_read(&self, cx: &mut Context, buf: &mut [u8]) -> Poll<Result<usize>>;

    /// Retrieve the current window size.
    ///
    /// Returned value are of format (columns, rows).
    ///
    /// This might not be supported for all `Serial`s. Those implementations may return an
    /// error instead.
    fn get_window_size(&self) -> Result<(u16, u16)> {
        Err(Error::new(ErrorKind::Other, "get_window_size not supported"))
    }

    /// Check if window size has been changed.
    fn poll_window_size_changed(&self, cx: &mut Context) -> Poll<Result<()>> {
        let _ = cx;
        Poll::Ready(Err(Error::new(ErrorKind::Other, "poll_window_size_changed not supported")))
    }
}

// Allow sharing of a Serial.
impl<T: Serial + ?Sized, P: std::ops::Deref<Target = T> + Send + Sync> Serial for P {
    fn poll_write(&self, cx: &mut Context, buf: &[u8]) -> Poll<Result<usize>> {
        (**self).poll_write(cx, buf)
    }

    fn poll_read(&self, cx: &mut Context, buf: &mut [u8]) -> Poll<Result<usize>> {
        (**self).poll_read(cx, buf)
    }

    fn get_window_size(&self) -> Result<(u16, u16)> {
        (**self).get_window_size()
    }

    fn poll_window_size_changed(&self, cx: &mut Context) -> Poll<Result<()>> {
        (**self).poll_window_size_changed(cx)
    }
}

impl dyn Serial {
    /// Send a packet to the device.
    pub async fn write(&self, buf: &[u8]) -> Result<usize> {
        poll_fn(|cx| self.poll_write(cx, buf)).await
    }

    /// Receive a packet from the device.
    pub async fn read(&self, buf: &mut [u8]) -> Result<usize> {
        poll_fn(|cx| self.poll_read(cx, buf)).await
    }

    /// Wait for window size to be changed.
    pub async fn wait_window_size_changed(&self) -> Result<()> {
        poll_fn(|cx| self.poll_window_size_changed(cx)).await
    }

    /// Attempt to write from the device.
    pub fn try_write(&self, buf: &[u8]) -> Result<usize> {
        let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
        match self.poll_write(&mut cx, buf) {
            Poll::Ready(x) => x,
            Poll::Pending => Err(Error::new(ErrorKind::WouldBlock, "WouldBlock")),
        }
    }

    /// Attempt to read from the device.
    pub fn try_read(&self, buf: &mut [u8]) -> Result<usize> {
        let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
        match self.poll_read(&mut cx, buf) {
            Poll::Ready(x) => x,
            Poll::Pending => Err(Error::new(ErrorKind::WouldBlock, "WouldBlock")),
        }
    }
}
