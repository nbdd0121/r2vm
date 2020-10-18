use super::Device;
use crate::{DmaContext, IoMemoryMut};
use parking_lot::Mutex;
use std::sync::Arc;

const ADDR_MAGIC_VALUE: usize = 0x000;
const ADDR_VERSION: usize = 0x004;
const ADDR_DEVICE_ID: usize = 0x008;
const ADDR_VENDOR_ID: usize = 0x00c;
const ADDR_DEVICE_FEATURES: usize = 0x010;
const ADDR_DEVICE_FEATURES_SEL: usize = 0x014;
const ADDR_DRIVER_FEATURES: usize = 0x020;
const ADDR_DRIVER_FEATURES_SEL: usize = 0x024;
const ADDR_QUEUE_SEL: usize = 0x030;
const ADDR_QUEUE_NUM_MAX: usize = 0x034;
const ADDR_QUEUE_NUM: usize = 0x038;
const ADDR_QUEUE_READY: usize = 0x044;
const ADDR_QUEUE_NOTIFY: usize = 0x050;
const ADDR_INTERRUPT_STATUS: usize = 0x060;
const ADDR_INTERRUPT_ACK: usize = 0x064;
const ADDR_STATUS: usize = 0x070;
const ADDR_QUEUE_DESC_LOW: usize = 0x080;
const ADDR_QUEUE_DESC_HIGH: usize = 0x084;
const ADDR_QUEUE_AVAIL_LOW: usize = 0x090;
const ADDR_QUEUE_AVAIL_HIGH: usize = 0x094;
const ADDR_QUEUE_USED_LOW: usize = 0x0a0;
const ADDR_QUEUE_USED_HIGH: usize = 0x0a4;
const ADDR_CONFIG_GENERATION: usize = 0x0fc;
const ADDR_CONFIG: usize = 0x100;

/// Virtio device with MMIO transport.
///
/// Note: Currently drop is not properly implemented and may cause memory and resource leak.
pub struct Mmio {
    device: Box<dyn Device>,
    queues: Vec<Arc<Mutex<super::queue::QueueInner>>>,
    device_features_sel: bool,
    driver_features_sel: bool,
    queue_sel: usize,
    dma_ctx: Arc<dyn DmaContext>,
}

impl Mmio {
    /// Create a virtio device with MMIO transport.
    pub fn new(dma_ctx: Arc<dyn DmaContext>, dev: Box<dyn Device>) -> Mmio {
        let num_queues = dev.num_queues();
        let mut queues = Vec::with_capacity(num_queues);
        for i in 0..num_queues {
            let len_max = dev.max_queue_len(i);
            let queue = super::queue::QueueInner::new(len_max);
            queues.push(queue);
        }
        Mmio {
            device: dev,
            queues,
            device_features_sel: false,
            driver_features_sel: false,
            queue_sel: 0,
            dma_ctx,
        }
    }

    pub fn build_dt(base: usize) -> fdt::Node {
        let mut node = fdt::Node::new(format!("virtio@{:x}", base));
        node.add_prop("compatible", "virtio,mmio");
        node
    }
}

impl IoMemoryMut for Mmio {
    fn read_mut(&mut self, addr: usize, size: u32) -> u64 {
        if addr >= ADDR_CONFIG {
            let value = self.device.config_read(addr - ADDR_CONFIG, size);
            trace!(target: "Mmio", "config register read 0x{:x} = 0x{:x}", addr, value);
            return value;
        }
        if size != 4 {
            error!(target: "Mmio", "illegal register read 0x{:x}", addr);
            return 0;
        }
        let ret = match addr {
            ADDR_MAGIC_VALUE => 0x74726976,
            ADDR_VERSION => 2,
            ADDR_DEVICE_ID => self.device.device_id() as u32,
            // This field is a PCI vendor, we use 0xFFFF because it indicates invalid (N/A)
            ADDR_VENDOR_ID => 0xffff,
            ADDR_DEVICE_FEATURES => {
                if self.device_features_sel {
                    // VIRTIO_F_VERSION_1 is always set
                    1
                } else {
                    self.device.device_feature()
                }
            }
            ADDR_DEVICE_FEATURES_SEL => self.device_features_sel as u32,
            ADDR_DRIVER_FEATURES_SEL => self.driver_features_sel as u32,
            ADDR_QUEUE_SEL => self.queue_sel as u32,
            ADDR_QUEUE_NUM_MAX => match self.queues.get(self.queue_sel) {
                None => 0,
                Some(queue) => queue.lock().num_max as u32,
            },
            ADDR_QUEUE_NUM | ADDR_QUEUE_READY | ADDR_QUEUE_DESC_LOW..=ADDR_QUEUE_USED_HIGH => {
                if self.queue_sel >= self.device.num_queues() {
                    error!(target: "Mmio", "attempting to access unavailable queue {}", self.queue_sel);
                    return 0;
                }
                let queue = self.queues[self.queue_sel].lock();
                match addr {
                    ADDR_QUEUE_NUM => queue.num as u32,
                    ADDR_QUEUE_READY => queue.ready as u32,
                    ADDR_QUEUE_DESC_LOW => queue.desc_addr as u32,
                    ADDR_QUEUE_DESC_HIGH => (queue.desc_addr >> 32) as u32,
                    ADDR_QUEUE_AVAIL_LOW => queue.avail_addr as u32,
                    ADDR_QUEUE_AVAIL_HIGH => (queue.avail_addr >> 32) as u32,
                    ADDR_QUEUE_USED_LOW => queue.used_addr as u32,
                    ADDR_QUEUE_USED_HIGH => (queue.used_addr >> 32) as u32,
                    _ => unreachable!(),
                }
            }
            // As currently config space is readonly, the interrupt status must be an used buffer.
            ADDR_INTERRUPT_STATUS => self.device.interrupt_status(),
            ADDR_STATUS => self.device.get_status(),
            ADDR_CONFIG_GENERATION => 0,
            _ => {
                error!(target: "Mmio", "illegal register read 0x{:x}", addr);
                0
            }
        };
        trace!(target: "Mmio", "Read {:x} => {:x}", addr, ret);
        ret as u64
    }

    fn write_mut(&mut self, addr: usize, value: u64, size: u32) {
        if addr >= ADDR_CONFIG {
            self.device.config_write(addr - ADDR_CONFIG, value, size);
            trace!(target: "Mmio", "config register write 0x{:x} = 0x{:x}", addr, value);
            return;
        }
        if size != 4 {
            error!(target: "Mmio", "illegal register write 0x{:x} = 0x{:x}", addr, value);
            return;
        }
        let value = value as u32;
        trace!(target: "Mmio", "register write 0x{:x} = 0x{:x}", addr, value);
        match addr {
            ADDR_DEVICE_FEATURES_SEL => {
                if value == 0 {
                    self.device_features_sel = false
                } else if value == 1 {
                    self.device_features_sel = true
                } else {
                    error!(target: "Mmio", "DriverFeaturesSel register is set to {}", value)
                }
            }
            ADDR_DRIVER_FEATURES => {
                if self.driver_features_sel {
                    if value != 1 {
                        error!(target: "Mmio", "DriverFeatures do not have VIRTIO_F_VERSION_1 set")
                    }
                } else {
                    // Only the lowest 24-bits are for the device.
                    self.device.driver_feature(value & 0xffffff);
                    trace!(target: "Mmio", "DriverFeatures set to {:24b}", value);
                }
            }
            ADDR_DRIVER_FEATURES_SEL => {
                if value == 0 {
                    self.driver_features_sel = false
                } else if value == 1 {
                    self.driver_features_sel = true
                } else {
                    error!(target: "Mmio", "DriverFeaturesSel register is set to {}", value)
                }
            }
            ADDR_QUEUE_SEL => self.queue_sel = value as usize,
            ADDR_QUEUE_NOTIFY => {
                if self.queue_sel >= self.device.num_queues() {
                    error!(target: "Mmio", "attempting to access unavailable queue {}", self.queue_sel);
                    return;
                }
                if let Some(ref mut send) = self.queues[self.queue_sel].lock().send {
                    let _ = send.try_send(());
                }
            }
            ADDR_QUEUE_NUM..=ADDR_QUEUE_READY | ADDR_QUEUE_DESC_LOW..=ADDR_QUEUE_USED_HIGH => {
                if self.queue_sel >= self.device.num_queues() {
                    error!(target: "Mmio", "attempting to access unavailable queue {}", self.queue_sel);
                    return;
                }
                let mut queue = self.queues[self.queue_sel].lock();
                match addr {
                    ADDR_QUEUE_NUM => {
                        if value.is_power_of_two() && value <= queue.num_max as u32 {
                            queue.num = value as u16
                        } else {
                            error!(target: "Mmio", "invalid queue size {}", value)
                        }
                    }
                    ADDR_QUEUE_READY => queue.ready = (value & 1) != 0,
                    ADDR_QUEUE_DESC_LOW => {
                        queue.desc_addr = (queue.desc_addr & !0xffffffff) | value as u64
                    }
                    ADDR_QUEUE_DESC_HIGH => {
                        queue.desc_addr = (queue.desc_addr & 0xffffffff) | (value as u64) << 32
                    }
                    ADDR_QUEUE_AVAIL_LOW => {
                        queue.avail_addr = (queue.avail_addr & !0xffffffff) | value as u64
                    }
                    ADDR_QUEUE_AVAIL_HIGH => {
                        queue.avail_addr = (queue.avail_addr & 0xffffffff) | (value as u64) << 32
                    }
                    ADDR_QUEUE_USED_LOW => {
                        queue.used_addr = (queue.used_addr & !0xffffffff) | value as u64
                    }
                    ADDR_QUEUE_USED_HIGH => {
                        queue.used_addr = (queue.used_addr & 0xffffffff) | (value as u64) << 32
                    }
                    _ => unreachable!(),
                }
                std::mem::drop(queue);

                if addr == ADDR_QUEUE_READY && value & 1 != 0 {
                    let inner = self.queues[self.queue_sel].clone();
                    if let Some(queue) = super::Queue::new(self.dma_ctx.clone(), inner) {
                        self.device.queue_ready(self.queue_sel, queue);
                    }
                }
            }
            ADDR_INTERRUPT_ACK => self.device.interrupt_ack(value),
            ADDR_STATUS => {
                if value == 0 {
                    self.device.reset();
                    // Upon reset, reset all queues, and replace them with new queue instances.
                    // Replacing them can hopefully allow devices to gracefully terminate tasks.
                    for (i, queue) in self.queues.iter_mut().enumerate() {
                        {
                            let mut lock = queue.lock();
                            lock.reset();
                            lock.send.take();
                        }
                        let inner = super::queue::QueueInner::new(self.device.max_queue_len(i));
                        *queue = inner;
                    }
                    self.queue_sel = 0;
                    self.device_features_sel = false;
                    self.driver_features_sel = false;
                } else {
                    self.device.set_status(value);
                }
            }
            _ => error!(target: "Mmio", "illegal register write 0x{:x} = 0x{:x}", addr, value),
        }
    }
}
