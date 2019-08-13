use std::io::Result;

#[cfg(feature = "slirp")]
mod slirp;
#[cfg(feature = "slirp")]
pub use slirp::Slirp;

pub trait Network: Send + Sync {
    fn send(&self, buf: &[u8]) -> Result<usize>;
    fn recv(&self, buf: &mut [u8]) -> Result<usize>;
}
