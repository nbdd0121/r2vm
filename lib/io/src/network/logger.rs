use super::Network;
use byteorder::{WriteBytesExt, LE};
use std::fs::File;
use std::io::{Result, Write};
use std::task::{Context, Poll};

/// A logger network device that captures all traffics and write to a pcap dump.
pub struct Logger<T> {
    file: File,
    network: T,
}

impl<T> Logger<T> {
    /// Create a new logger that captures traffic to the given file in pcap dump format.
    pub fn new(mut file: File, network: T) -> Self {
        // Write the header of pcap dump format.
        file.write_u32::<LE>(0xa1b2c3d4).unwrap();
        file.write_u16::<LE>(2).unwrap();
        file.write_u16::<LE>(4).unwrap();
        file.write_u32::<LE>(0).unwrap();
        file.write_u32::<LE>(0).unwrap();
        file.write_u32::<LE>(0xffffffff).unwrap();
        file.write_u32::<LE>(1).unwrap();

        Self { file, network }
    }

    fn log(&self, buf: &[u8]) -> Result<()> {
        let now =
            std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap();
        let mut file = &self.file;
        file.write_u32::<LE>(now.as_secs() as u32)?;
        file.write_u32::<LE>(now.subsec_micros())?;
        file.write_u32::<LE>(buf.len() as u32)?;
        file.write_u32::<LE>(buf.len() as u32)?;
        file.write_all(&buf)?;
        file.sync_data()
    }
}

impl<T: Network> Network for Logger<T> {
    fn poll_send(&self, ctx: &mut Context, buf: &[u8]) -> Poll<Result<usize>> {
        match self.network.poll_send(ctx, buf) {
            Poll::Ready(v) => {
                self.log(buf)?;
                Poll::Ready(v)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn poll_recv(&self, ctx: &mut Context, buf: &mut [u8]) -> Poll<Result<usize>> {
        match self.network.poll_recv(ctx, buf) {
            Poll::Ready(v) => {
                self.log(buf)?;
                Poll::Ready(v)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
