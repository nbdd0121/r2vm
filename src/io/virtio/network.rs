use super::super::network::Network as NetworkDevice;
use super::super::{IoContext, IrqPin};
use super::{Device, DeviceId, Queue};
use futures::future::AbortHandle;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

const VIRTIO_BLK_F_MAC: usize = 5;

#[repr(C)]
struct VirtioNetHeader {
    flags: u8,
    gso_type: u8,
    hdr_len: u16,
    gso_size: u16,
    csum_start: u16,
    csum_offset: u16,
    num_buffers: u16,
}

/// A virtio entropy source device.
pub struct Network {
    net: Arc<dyn NetworkDevice>,
    status: u32,
    mac: [u8; 6],
    irq: Arc<dyn IrqPin>,
    rx_handle: Option<AbortHandle>,
    io_ctx: Arc<dyn IoContext>,
}

impl Network {
    pub fn new(
        io_ctx: Arc<dyn IoContext>,
        irq: Arc<dyn IrqPin>,
        net: impl NetworkDevice + 'static,
        mac: [u8; 6],
    ) -> Network {
        let net = Arc::new(net);
        Network { net, status: 0, mac, irq, rx_handle: None, io_ctx }
    }

    fn start_tx(&self, mut tx: Queue) {
        let iface = self.net.clone();
        let irq = self.irq.clone();
        // There's no stop mechanism, but we don't destroy devices anyway, so that's okay.
        self.io_ctx.spawn(Box::pin(async move {
            while let Ok(buffer) = tx.take().await {
                let mut reader = buffer.reader();

                let hdr_len = std::mem::size_of::<VirtioNetHeader>();
                if reader.len() <= hdr_len {
                    // Unexpected packet
                    error!(target: "VirtioNet", "illegal transmission with size {} smaller than header {}", reader.len(), hdr_len);
                }

                // We don't need any of the fields of the header, so just skip it.
                let packet_len = reader.len() - hdr_len;
                reader.seek(SeekFrom::Start(hdr_len as u64)).unwrap();

                let mut io_buffer = Vec::with_capacity(packet_len);
                unsafe { io_buffer.set_len(io_buffer.capacity()) };
                reader.read_exact(&mut io_buffer).unwrap();
                drop(buffer);

                iface.send(&io_buffer).await.unwrap();
                irq.pulse();
            }
        }));
    }

    fn start_rx(&self, mut rx: Queue) -> AbortHandle {
        let iface = self.net.clone();
        let irq = self.irq.clone();
        let (handle, reg) = futures::future::AbortHandle::new_pair();
        self.io_ctx.spawn(Box::pin(async move {
            let _ = futures::future::Abortable::new(async move {
                let mut buffer = [0; 2048];
                loop {
                    let len = iface.recv(&mut buffer).await.unwrap();
                    match rx.try_take() {
                        // Queue shutdown, terminate gracefully
                        Err(_) => return,
                        Ok(Some(mut dma_buffer)) => {
                            let mut writer = dma_buffer.writer();
                            let header: [u8; std::mem::size_of::<VirtioNetHeader>()] = {
                                let header = VirtioNetHeader {
                                    flags: 0,
                                    gso_type: 0,
                                    hdr_len: 0,
                                    gso_size: 0,
                                    csum_start: 0,
                                    csum_offset: 0,
                                    num_buffers: 1,
                                };
                                unsafe { std::mem::transmute(header) }
                            };
                            if header.len() + len > writer.len() {
                                info!(
                                    target: "VirtioNet",
                                    "discard packet of size {:x} because it does not fit into buffer of size {:x}",
                                    len, writer.len()
                                );
                                return;
                            }
                            writer.write_all(&header).unwrap();
                            writer.write_all(&buffer[..len]).unwrap();
                            drop(dma_buffer);

                            irq.pulse();
                        }
                        Ok(None) => info!(
                            target: "VirtioNet",
                            "discard packet of size {:x} because there is no buffer in receiver queue",
                            len
                        ),
                    }
                }
            }, reg).await;
        }));
        handle
    }
}

impl Device for Network {
    fn device_id(&self) -> DeviceId {
        DeviceId::Network
    }
    fn device_feature(&self) -> u32 {
        1 << VIRTIO_BLK_F_MAC
    }
    fn driver_feature(&mut self, _value: u32) {}
    fn get_status(&self) -> u32 {
        self.status
    }
    fn set_status(&mut self, status: u32) {
        self.status = status
    }
    fn config_space(&self) -> &[u8] {
        &self.mac
    }
    fn num_queues(&self) -> usize {
        2
    }
    fn reset(&mut self) {
        self.status = 0;
        self.rx_handle.take().map(|x| x.abort());
    }
    fn queue_ready(&mut self, idx: usize, queue: Queue) {
        if idx == 0 {
            self.rx_handle.take().map(|x| x.abort());
            self.rx_handle = Some(self.start_rx(queue));
        } else {
            self.start_tx(queue);
        }
    }
}
