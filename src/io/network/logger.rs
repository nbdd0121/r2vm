use std::sync::{Arc, Mutex};
use byteorder::{LE, WriteBytesExt};
use std::io::{Write, Result};
use std::fs::File;

use super::Network;

pub struct Logger<T: Network> {
    file: Arc<Mutex<File>>,
    network: T,
}

impl<T: Network> Logger<T> {
    pub fn new(mut file: File, network: T) -> Self {
        // Write the header of pcap dump format.
        file.write_u32::<LE>(0xa1b2c3d4).unwrap();
        file.write_u16::<LE>(2).unwrap();
        file.write_u16::<LE>(4).unwrap();
        file.write_u32::<LE>(0).unwrap();
        file.write_u32::<LE>(0).unwrap();
        file.write_u32::<LE>(0xffffffff).unwrap();
        file.write_u32::<LE>(1).unwrap();

        Self {
            file: Arc::new(Mutex::new(file)),
            network,
        }
    }

    fn log(&self, buf: &[u8]) -> Result<()> {
        let now = std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap();
        let mut file = self.file.lock().unwrap();
        file.write_u32::<LE>(now.as_secs() as u32)?;
        file.write_u32::<LE>(now.subsec_micros())?;
        file.write_u32::<LE>(buf.len() as u32)?;
        file.write_u32::<LE>(buf.len() as u32)?;
        file.write_all(&buf)?;
        file.sync_data()
    }
}

impl<T: Network> Network for Logger<T> {
    fn send(&self, buf: &[u8]) -> Result<usize> {
        self.log(buf)?;
        self.network.send(buf)
    }

    fn recv(&self, buf: &mut [u8]) -> Result<usize> {
        let len = self.network.recv(buf)?;
        self.log(&buf[..len])?;
        Ok(len)
    }
}
