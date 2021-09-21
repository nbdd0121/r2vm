use crate::{IoMemory, IrqPin, RuntimeContext};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Implemention of RISC-V core-local interrupt controller.
pub struct Clint(Arc<Inner>);

struct Inner {
    msip: Box<[AtomicBool]>,
    mtimecmp: Box<[AtomicU64]>,
    msip_irqs: Box<[Box<dyn IrqPin>]>,
    mtip_irqs: Box<[Box<dyn IrqPin>]>,
    ctx: Arc<dyn RuntimeContext>,
}

impl Clint {
    /// Create a new `Clint` with MSIP and MTIP interrupt pins.
    ///
    /// # Panics
    /// The function will panic if `msip_irqs` and `mtip_irqs` are not of the same length.
    pub fn new(
        ctx: Arc<dyn RuntimeContext>,
        msip_irqs: Vec<Box<dyn IrqPin>>,
        mtip_irqs: Vec<Box<dyn IrqPin>>,
    ) -> Self {
        assert_eq!(msip_irqs.len(), mtip_irqs.len());
        let inner = Arc::new(Inner {
            msip: (0..msip_irqs.len()).map(|_| AtomicBool::new(false)).collect(),
            mtimecmp: (0..mtip_irqs.len()).map(|_| AtomicU64::new(u64::MAX)).collect(),
            msip_irqs: msip_irqs.into_boxed_slice(),
            mtip_irqs: mtip_irqs.into_boxed_slice(),
            ctx,
        });
        Self(inner)
    }
}

impl IoMemory for Clint {
    fn read(&self, addr: usize, size: u32) -> u64 {
        // Guard against access narrower than 32-bit
        if size < 4 {
            error!(target: "CLINT", "illegal register read 0x{:x}", addr);
            return 0;
        }

        match addr & !4 {
            0..=0x3FFF => {
                let hart = addr / 4;
                if size != 4 || hart >= self.0.msip.len() {
                    error!(target: "CLINT", "illegal register read 0x{:x}", addr);
                    return 0;
                }
                self.0.msip[hart].load(Ordering::Relaxed) as u64
            }
            0x4000..=0xBFF7 => {
                let hart = (addr - 0x4000) / 8;
                if hart >= self.0.mtimecmp.len() {
                    error!(target: "CLINT", "illegal register read 0x{:x}", addr);
                    return 0;
                }
                let mtimecmp = self.0.mtimecmp[hart].load(Ordering::Relaxed);
                if size == 8 {
                    mtimecmp
                } else {
                    // Narrow read
                    if addr & 4 == 0 { mtimecmp } else { mtimecmp >> 32 }
                }
            }
            0xBFF8 => {
                let time = self.0.ctx.now().as_micros() as u64;
                if size == 8 {
                    time
                } else {
                    // Narrow read
                    if addr & 4 == 0 { time } else { time >> 32 }
                }
            }
            _ => {
                error!(target: "CLINT", "illegal register read 0x{:x}", addr);
                0
            }
        }
    }

    fn write(&self, addr: usize, value: u64, size: u32) {
        // Guard against access narrower than 32-bit
        if size < 4 {
            error!(target: "CLINT", "illegal register write 0x{:x} = 0x{:x}", addr, value);
            return;
        }

        match addr & !4 {
            0..=0x3FFF => {
                let hart = addr / 4;
                if size != 4 || hart >= self.0.msip.len() {
                    error!(target: "CLINT", "illegal register write 0x{:x} = 0x{:x}", addr, value);
                    return;
                }
                let value = value & 1 != 0;
                self.0.msip[hart].store(value, Ordering::Relaxed);
                self.0.msip_irqs[hart].set_level(value);
            }
            0x4000..=0xBFF7 => {
                let hart = (addr - 0x4000) / 8;
                if hart >= self.0.mtimecmp.len() {
                    error!(target: "CLINT", "illegal register write 0x{:x} = 0x{:x}", addr, value);
                    return;
                }
                let value = if size == 8 {
                    value
                } else {
                    let oldvalue = self.0.mtimecmp[hart].load(Ordering::Relaxed);
                    if addr & 4 == 0 {
                        oldvalue & 0xffffffff00000000 | value as u32 as u64
                    } else {
                        oldvalue & 0xffffffff | (value as u32 as u64) << 32
                    }
                };
                self.0.mtimecmp[hart].store(value, Ordering::Relaxed);

                let new_time = Duration::from_micros(value);
                let triggered = new_time <= self.0.ctx.now();
                self.0.mtip_irqs[hart].set_level(triggered);
                if !triggered {
                    let timer = self.0.ctx.create_timer(new_time);
                    let self_ref = Arc::downgrade(&self.0);
                    self.0.ctx.spawn(Box::pin(async move {
                        timer.await;
                        if let Some(inner) = self_ref.upgrade() {
                            inner.mtip_irqs[hart].set_level(
                                inner.mtimecmp[hart].load(Ordering::Relaxed)
                                    <= inner.ctx.now().as_micros() as u64,
                            );
                        }
                    }));
                }
            }
            _ => {
                error!(target: "CLINT", "illegal register write 0x{:x} = 0x{:x}", addr, value);
            }
        }
    }
}
