use crate::{IoMemory, IrqPin};

const ADDR_SET_TM_WR: usize = 0x00;
const ADDR_SET_TM_RD: usize = 0x04;
const ADDR_CALIB_WR: usize = 0x08;
const ADDR_CALIB_RD: usize = 0x0C;
const ADDR_CUR_TM: usize = 0x10;
const ADDR_CUR_TICK: usize = 0x14;
const ADDR_ALRM: usize = 0x18;
const ADDR_INT_STS: usize = 0x20;
const ADDR_INT_MASK: usize = 0x24;
const ADDR_INT_EN: usize = 0x28;
const ADDR_INT_DIS: usize = 0x2C;
const ADDR_CTRL: usize = 0x40;

const CTRL_BATT_EN: u32 = 1 << 31;

/// An implementation of Xilinx Zynq Ultrascale+ MPSoC RTC.
///
/// Currently this implementation is read only.
pub struct ZyncMp {}

impl ZyncMp {
    pub fn new(_alarm_irq: Box<dyn IrqPin>, _sec_irq: Box<dyn IrqPin>) -> ZyncMp {
        ZyncMp {}
    }

    pub fn build_dt(base: usize) -> fdt::Node {
        let mut node = fdt::Node::new(format!("rtc@{:x}", base));
        node.add_prop("compatible", "xlnx,zynqmp-rtc");
        node.add_prop("interrupt-names", &["alarm", "sec"][..]);
        node
    }
}

impl IoMemory for ZyncMp {
    fn read(&self, addr: usize, size: u32) -> u64 {
        // This I/O memory region supports 32-bit memory access only
        if size != 4 {
            error!(target: "RTC", "illegal register read 0x{:x}", addr);
            return 0;
        }
        let val = match addr {
            ADDR_SET_TM_RD | ADDR_CUR_TM => std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            ADDR_CALIB_RD => 0x198233,
            ADDR_CUR_TICK => 0xffff,
            ADDR_ALRM => 0,
            ADDR_INT_STS => 3,
            ADDR_INT_MASK => 3,
            ADDR_CTRL => CTRL_BATT_EN as u64,
            _ => {
                error!(target: "RTC", "illegal register read 0x{:x}", addr);
                0
            }
        };
        trace!(target: "RTC", "read register 0x{:x} as 0x{:x}", addr, val);
        val
    }

    fn write(&self, addr: usize, value: u64, size: u32) {
        // This I/O memory region supports 32-bit memory access only
        if size != 4 {
            error!(target: "RTC", "illegal register write 0x{:x} = 0x{:x}", addr, value);
            return;
        }
        trace!(target: "RTC", "write register 0x{:x} = 0x{:x}", addr, value);
        match addr {
            ADDR_SET_TM_WR => (),
            ADDR_CALIB_WR => (),
            ADDR_ALRM => (),
            ADDR_INT_STS => (),
            ADDR_INT_EN => (),
            ADDR_INT_DIS => (),
            ADDR_CTRL => (),
            _ => {
                error!(target: "RTC", "illegal register write 0x{:x} = 0x{:x}", addr, value);
            }
        }
    }
}
