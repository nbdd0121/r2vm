use super::Device;
use super::super::IoMemory;

use std::convert::TryInto;
use crate::util::SplitInt;

const ADDR_MAGIC_VALUE         : usize = 0x000;
const ADDR_VERSION             : usize = 0x004;
const ADDR_DEVICE_ID           : usize = 0x008;
const ADDR_VENDOR_ID           : usize = 0x00c;
const ADDR_DEVICE_FEATURES     : usize = 0x010;
const ADDR_DEVICE_FEATURES_SEL : usize = 0x014;
const ADDR_DRIVER_FEATURES     : usize = 0x020;
const ADDR_DRIVER_FEATURES_SEL : usize = 0x024;
const ADDR_QUEUE_SEL           : usize = 0x030;
const ADDR_QUEUE_NUM_MAX       : usize = 0x034;
const ADDR_QUEUE_NUM           : usize = 0x038;
const ADDR_QUEUE_READY         : usize = 0x044;
const ADDR_QUEUE_NOTIFY        : usize = 0x050;
const ADDR_INTERRUPT_STATUS    : usize = 0x060;
const ADDR_INTERRUPT_ACK       : usize = 0x064;
const ADDR_STATUS              : usize = 0x070;
const ADDR_QUEUE_DESC_LOW      : usize = 0x080;
const ADDR_QUEUE_DESC_HIGH     : usize = 0x084;
const ADDR_QUEUE_AVAIL_LOW     : usize = 0x090;
const ADDR_QUEUE_AVAIL_HIGH    : usize = 0x094;
const ADDR_QUEUE_USED_LOW      : usize = 0x0a0;
const ADDR_QUEUE_USED_HIGH     : usize = 0x0a4;
const ADDR_CONFIG_GENERATION   : usize = 0x0fc;
const ADDR_CONFIG              : usize = 0x100;

pub struct Mmio {
    device: Box<dyn Device>,
    device_features_sel: bool,
    driver_features_sel: bool,
    queue_sel: usize,
}

impl Mmio {
    pub fn new(dev: Box<dyn Device>) -> Mmio {
        Mmio {
            device: dev,
            device_features_sel: false,
            driver_features_sel: false,
            queue_sel: 0,
        }
    }
}

impl IoMemory for Mmio {
    fn read(&mut self, addr: usize, size: u32) -> u64 {
        if size != 4 {
            error!(target: "Mmio", "illegal register read 0x{:x}", addr);
            return 0
        }
        if addr >= ADDR_CONFIG {
            let offset = (addr - ADDR_CONFIG) as usize;
            let config = self.device.config_space();
            if offset + 4 > config.len() {
                error!(target: "Mmio", "out-of-bound config register read 0x{:x}", offset);
                return 0
            }
            // TODO: Config-space read can be byte-sized
            let value = u32::from_le_bytes(config[offset .. offset + 4].try_into().unwrap());
            trace!(target: "Mmio", "config register read 0x{:x} = 0x{:x}", addr, value);
            return value as u64;
        }
        let ret = match addr {
            ADDR_MAGIC_VALUE         => 0x74726976,
            ADDR_VERSION             => 2,
            ADDR_DEVICE_ID           => self.device.device_id() as u32,
            // This field is a PCI vendor, we use 0xFFFF because it indicates invalid (N/A)
            ADDR_VENDOR_ID           => 0xffff,
            ADDR_DEVICE_FEATURES     =>
                if self.device_features_sel {
                    // VIRTIO_F_VERSION_1 is always set
                    1
                } else {
                    self.device.device_feature()
                },
            ADDR_DEVICE_FEATURES_SEL => self.device_features_sel as u32,
            ADDR_DRIVER_FEATURES_SEL => self.driver_features_sel as u32,
            ADDR_QUEUE_SEL           => self.queue_sel as u32,
            ADDR_QUEUE_NUM_MAX       => {
                let queues = self.device.queues();
                if self.queue_sel >= queues.len() {
                    0
                } else {
                    32768
                }
            }
            ADDR_QUEUE_NUM | ADDR_QUEUE_READY | ADDR_QUEUE_DESC_LOW ... ADDR_QUEUE_USED_HIGH => {
                let queues = self.device.queues();
                if self.queue_sel >= queues.len() {
                    error!(target: "Mmio", "attempting to access unavailable queue {}", self.queue_sel);
                    return 0;
                }
                let queue = &mut queues[self.queue_sel];
                match addr {
                    ADDR_QUEUE_NUM        => queue.num as u32,
                    ADDR_QUEUE_READY      => queue.ready as u32,
                    ADDR_QUEUE_DESC_LOW   => queue.desc_addr.lo(),
                    ADDR_QUEUE_DESC_HIGH  => queue.desc_addr.hi(),
                    ADDR_QUEUE_AVAIL_LOW  => queue.avail_addr.lo(),
                    ADDR_QUEUE_AVAIL_HIGH => queue.avail_addr.hi(),
                    ADDR_QUEUE_USED_LOW   => queue.used_addr.lo(),
                    ADDR_QUEUE_USED_HIGH  => queue.used_addr.hi(),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                }
            }
            // As currently config space is readonly, the interrupt status must be an used buffer.
            ADDR_INTERRUPT_STATUS    => 1,
            ADDR_STATUS              => self.device.get_status(),
            ADDR_CONFIG_GENERATION   => 0,
            _                        => {
                error!(target: "Mmio", "illegal register read 0x{:x}", addr);
                0
            }
        };
        trace!(target: "Mmio", "Read {:x} => {:x}", addr, ret);
        ret as u64
    }

    fn write(&mut self, addr: usize, value: u64, size: u32) {
        if addr >= ADDR_CONFIG {
            error!(target: "Mmio", "config register write 0x{:x} = 0x{:x}", addr, value);
            return;
        }
        if size != 4 {
            error!(target: "Mmio", "illegal register write 0x{:x} = 0x{:x}", addr, value);
            return
        }
        let value = value as u32;
        trace!(target: "Mmio", "register write 0x{:x} = 0x{:x}", addr, value);
        match addr {
            ADDR_DEVICE_FEATURES_SEL =>
                if value == 0 {
                    self.device_features_sel = false
                } else if value == 1 {
                    self.device_features_sel = true
                } else {
                    error!(target: "Mmio", "DriverFeaturesSel register is set to {}", value)
                },
            ADDR_DRIVER_FEATURES     => {
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
            ADDR_DRIVER_FEATURES_SEL =>
                if value == 0 {
                    self.driver_features_sel = false
                } else if value == 1 {
                    self.driver_features_sel = true
                } else {
                    error!(target: "Mmio", "DriverFeaturesSel register is set to {}", value)
                },
            ADDR_QUEUE_SEL           => self.queue_sel = value as usize,
            ADDR_QUEUE_NUM ... ADDR_QUEUE_NOTIFY | ADDR_QUEUE_DESC_LOW ... ADDR_QUEUE_USED_HIGH => {
                let queues = self.device.queues();
                if self.queue_sel >= queues.len() {
                    error!(target: "Mmio", "attempting to access unavailable queue {}", self.queue_sel);
                    return;
                }
                let queue = &mut queues[self.queue_sel];
                match addr {
                    ADDR_QUEUE_NUM           => {
                        if value.is_power_of_two() && value <= 32768 {
                            queue.num = value as u16
                        } else {
                            error!(target: "Mmio", "invalid queue size {}", value)
                        }
                    }
                    ADDR_QUEUE_READY      => queue.ready = (value & 1) != 0,
                    ADDR_QUEUE_NOTIFY     => self.device.notify(self.queue_sel),
                    ADDR_QUEUE_DESC_LOW   => queue.desc_addr.set_lo(value),
                    ADDR_QUEUE_DESC_HIGH  => queue.desc_addr.set_hi(value),
                    ADDR_QUEUE_AVAIL_LOW  => queue.avail_addr.set_lo(value),
                    ADDR_QUEUE_AVAIL_HIGH => queue.avail_addr.set_hi(value),
                    ADDR_QUEUE_USED_LOW   => queue.used_addr.set_lo(value),
                    ADDR_QUEUE_USED_HIGH  => queue.used_addr.set_hi(value),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                }
            }
            ADDR_INTERRUPT_ACK       => (),
            ADDR_STATUS              =>
                if value == 0 {
                    self.device.reset();
                    self.queue_sel = 0;
                    self.device_features_sel = false;
                    self.driver_features_sel = false;
                } else {
                    self.device.set_status(value);
                },
            _ => error!(target: "Mmio", "illegal register write 0x{:x} = 0x{:x}", addr, value),
        }
    }
}
