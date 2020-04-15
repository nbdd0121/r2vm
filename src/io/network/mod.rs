use futures::future::poll_fn;
use std::io::Result;
use std::task::{Context, Poll};

mod logger;
#[cfg(feature = "usernet")]
mod usernet;
pub mod xemaclite;

#[cfg(feature = "usernet")]
pub use self::usernet::Usernet;
pub use logger::Logger;

pub trait Network: Send + Sync {
    fn poll_send(&self, ctx: &mut Context, buf: &[u8]) -> Poll<Result<usize>>;

    fn poll_recv(&self, ctx: &mut Context, buf: &mut [u8]) -> Poll<Result<usize>>;
}

impl dyn Network {
    pub async fn send(&self, buf: &[u8]) -> Result<usize> {
        poll_fn(|cx| self.poll_send(cx, buf)).await
    }

    pub async fn recv(&self, buf: &mut [u8]) -> Result<usize> {
        poll_fn(|cx| self.poll_recv(cx, buf)).await
    }
}
