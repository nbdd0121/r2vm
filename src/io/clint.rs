use parking_lot::Mutex;
use std::sync::{Arc, Weak};
use std::time::Duration;

use super::{IoContext, IoMemorySync, IrqPin};

pub struct Clint {
    inner: Mutex<Inner>,
}

struct Inner {
    msip: Box<[bool]>,
    mtimecmp: Box<[u64]>,
    msip_irqs: Box<[Arc<dyn IrqPin>]>,
    mtip_irqs: Box<[Arc<dyn IrqPin>]>,
    io_ctx: Arc<dyn IoContext>,
    self_ref: Weak<Clint>,
}

impl Clint {
    pub fn new(
        io_ctx: Arc<dyn IoContext>,
        msip_irqs: Vec<Arc<dyn IrqPin>>,
        mtip_irqs: Vec<Arc<dyn IrqPin>>,
    ) -> Arc<Self> {
        assert_eq!(msip_irqs.len(), mtip_irqs.len());
        let inner = Inner {
            msip: vec![false; msip_irqs.len()].into_boxed_slice(),
            mtimecmp: vec![u64::max_value(); mtip_irqs.len()].into_boxed_slice(),
            msip_irqs: msip_irqs.into_boxed_slice(),
            mtip_irqs: mtip_irqs.into_boxed_slice(),
            io_ctx,
            self_ref: Weak::new(),
        };
        let arc = Arc::new(Self { inner: Mutex::new(inner) });
        arc.inner.lock().self_ref = Arc::downgrade(&arc);
        arc
    }
}

impl IoMemorySync for Clint {
    fn read_sync(&self, addr: usize, size: u32) -> u64 {
        let inner = self.inner.lock();
        if addr == 0xBFF8 {
            inner.io_ctx.time().as_micros() as u64
        } else if addr < 0x4000 {
            let hart = addr / 4;
            if size != 4 || hart >= inner.msip.len() {
                error!(target: "CLINT", "illegal register read 0x{:x}", addr);
                return 0;
            }
            inner.msip[hart] as u64
        } else {
            let hart = (addr - 0x4000) / 8;
            if size != 8 || hart >= inner.mtimecmp.len() {
                error!(target: "CLINT", "illegal register read 0x{:x}", addr);
                return 0;
            }
            inner.mtimecmp[hart]
        }
    }

    fn write_sync(&self, addr: usize, value: u64, size: u32) {
        let mut inner = self.inner.lock();
        if addr < 0x4000 {
            let hart = addr / 4;
            if size != 4 || hart >= inner.msip.len() {
                error!(target: "CLINT", "illegal register write 0x{:x} = 0x{:x}", addr, value);
                return;
            }
            let value = value & 1 != 0;
            inner.msip[hart] = value;
            inner.msip_irqs[hart].set_level(value);
        } else {
            let hart = (addr - 0x4000) / 8;
            if size != 8 || hart >= inner.mtimecmp.len() {
                error!(target: "CLINT", "illegal register write 0x{:x} = 0x{:x}", addr, value);
                return;
            }
            inner.mtimecmp[hart] = value;

            let new_time = Duration::from_micros(value);
            let triggered = new_time <= inner.io_ctx.time();
            inner.mtip_irqs[hart].set_level(triggered);
            if !triggered {
                let timer = inner.io_ctx.on_time(new_time);
                let self_arc = inner.self_ref.upgrade().unwrap();
                inner.io_ctx.spawn(Box::pin(async move {
                    timer.await;
                    let inner = self_arc.inner.lock();
                    inner.mtip_irqs[hart]
                        .set_level(inner.mtimecmp[hart] <= inner.io_ctx.time().as_micros() as u64);
                }));
            }
        }
    }
}
