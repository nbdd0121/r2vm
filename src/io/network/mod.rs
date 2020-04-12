use async_trait::async_trait;
use std::io::Result;

mod logger;
#[cfg(feature = "usernet")]
mod usernet;
pub mod xemaclite;

#[cfg(feature = "usernet")]
pub use self::usernet::Usernet;
pub use logger::Logger;

#[async_trait]
pub trait Network: Send + Sync {
    async fn send(&self, buf: &[u8]) -> Result<usize>;
    async fn recv(&self, buf: &mut [u8]) -> Result<usize>;
}
