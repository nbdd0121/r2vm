use super::IoMemory;

// The ranges here are all inclusive, and they are calculated under maximum permitted number
// of interrupts (1024).
const ADDR_PRIORITY_START: usize = 0x000;
const ADDR_PRIORITY_END: usize = 0xFFC;

const ADDR_PENDING_START: usize = 0x1000;
const ADDR_PENDING_END: usize = 0x107C;

const ADDR_ENABLE_START: usize = 0x2000;
const ADDR_ENABLE_SIZE: usize = 0x80;

const ADDR_CONTEXT_START: usize = 0x200000;
const ADDR_CONTEXT_SIZE: usize = 0x1000;
const OFFSET_PRIORITY_THRESHOLD: usize = 0x000;
const OFFSET_INTERRUPT_CLAIM: usize = 0x004;

/// An implementation of SiFive's PLIC controller. We support 31 interrupts at the moment.
pub struct Plic {
    priority: [u8; 32],
    pending: u32,
    claimed: u32,
    enable: Box<[u32]>,
    threshold: Box<[u8]>,
}

impl Plic {
    /// Create a SiFive-compatible PLIC, with `ctx` number of contexts.
    pub fn new(ctx: usize) -> Plic {
        Plic {
            priority: [0; 32],
            pending: 0,
            enable: vec![0; ctx].into_boxed_slice(),
            threshold: vec![0; ctx].into_boxed_slice(),
            claimed: 0,
        }
    }

    pub fn trigger(&mut self, irq: u32) {
        assert!(irq > 0 && irq < 32);
        self.pending |= 1 << irq;
        self.recompute_pending();
    }

    /// Check if there are IRQs pending for a context
    fn pending(&self, ctx: usize) -> bool {
        let claimable = (self.pending & !self.claimed) & self.enable[ctx];
        for irq in 1..32 {
            let prio = self.priority[irq];
            let enabled = (claimable & (1 << irq)) != 0;
            if enabled && prio > self.threshold[ctx] {
                return true;
            }
        }
        false
    }

    fn recompute_pending(&mut self) {
        for ctx in 0..self.enable.len() {
            if self.pending(ctx) {
                crate::shared_context(ctx).assert(512);
            } else {
                crate::shared_context(ctx).deassert(512);
            }
        }
    }

    fn claim(&mut self, ctx: usize) -> u32 {
        let claimable = (self.pending & !self.claimed) & self.enable[ctx];
        let mut cur_irq = 0;
        // According to document, the claim operation is not affected by the setting of priority
        // threshold register, and can claim even if EIP is low.
        let mut cur_prio = 0;
        for irq in 1..32 {
            let prio = self.priority[irq];
            let enabled = (claimable & (1 << irq)) != 0;
            if enabled && prio > cur_prio {
                cur_irq = irq;
                cur_prio = prio;
            }
        }
        if cur_irq != 0 {
            self.pending &= !(1 << cur_irq);
            self.claimed |= 1 << cur_irq;
            // If something is claimed, then the pending state will change, so recompute
            self.recompute_pending();
        }
        cur_irq as u32
    }
}

impl IoMemory for Plic {
    fn read(&mut self, addr: usize, size: u32) -> u64 {
        // This I/O memory region supports 32-bit memory access only
        if size != 4 {
            error!(target: "PLIC", "illegal register read 0x{:x}", addr);
            return 0;
        }
        (match addr {
            // Interrupt priority
            ADDR_PRIORITY_START..=ADDR_PRIORITY_END => {
                let irq = (addr - ADDR_PRIORITY_START) / 4;
                // Out of bound read
                if irq == 0 || irq >= 32 {
                    error!(target: "PLIC", "access to interrupt priority of {} is out of bound", irq);
                    return 0;
                }
                let value = self.priority[irq];
                trace!(target: "PLIC", "read interrupt priority of {} as {}", irq, value);
                value as u32
            }
            // Pending bits
            ADDR_PENDING_START..=ADDR_PENDING_END => {
                let irq = (addr - ADDR_PENDING_START) * 8;
                // Out of bound read
                if irq >= 32 {
                    error!(target: "PLIC", "access to pending of {}-{} is out of bound", irq, irq + 31);
                    return 0;
                }
                let value = self.pending;
                trace!(target: "PLIC", "read pending of {}-{} as {}", irq, irq + 31, value);
                value
            }
            // Interrupt enable bits
            ADDR_ENABLE_START..=0x1FFFFF => {
                let ctx = (addr - ADDR_ENABLE_START) / ADDR_ENABLE_SIZE;
                let irq = (addr & (ADDR_ENABLE_SIZE - 1)) * 8;
                // Out of bound write
                if ctx >= self.enable.len() || irq >= 32 {
                    error!(target: "PLIC", "{} access to interrupt enable of {}-{} is out of bound", ctx, irq, irq + 31);
                    return 0;
                }
                let value = self.enable[ctx];
                trace!(target: "PLIC", "{} read interrupt enable of {}-{} as {:b}", ctx, irq, irq + 31, value);
                value
            }
            // Threshold and claim bits
            addr if addr >= ADDR_CONTEXT_SIZE => {
                let ctx = (addr - ADDR_CONTEXT_START) / ADDR_CONTEXT_SIZE;
                let offset = addr & (ADDR_CONTEXT_SIZE - 1);
                // Out of bound write
                if ctx >= self.enable.len() {
                    error!(target: "PLIC", "access to context id {} is out of bound", ctx);
                    return 0;
                }
                match offset {
                    OFFSET_PRIORITY_THRESHOLD => {
                        let value = self.threshold[ctx];
                        trace!(target: "PLIC", "{} read priority threshold as {}", ctx, value);
                        value as u32
                    }
                    OFFSET_INTERRUPT_CLAIM => {
                        let value = self.claim(ctx);
                        trace!(target: "PLIC", "{} claimed interrupt {}", ctx, value);
                        value
                    }
                    // Out of bound write
                    _ => {
                        error!(target: "PLIC", "illegal register read 0x{:x}", addr);
                        return 0;
                    }
                }
            }
            _ => {
                error!(target: "PLIC", "illegal register read 0x{:x}", addr);
                return 0;
            }
        }) as u64
    }

    fn write(&mut self, addr: usize, value: u64, size: u32) {
        // This I/O memory region supports 32-bit memory access only
        if size != 4 {
            error!(target: "PLIC", "illegal register write 0x{:x} = 0x{:x}", addr, value);
            return;
        }
        let value = value as u32;

        match addr {
            // Interrupt priority
            ADDR_PRIORITY_START..=ADDR_PRIORITY_END => {
                let irq = (addr - ADDR_PRIORITY_START) / 4;
                // Out of bound write
                if irq == 0 || irq >= 32 {
                    error!(target: "PLIC", "access to interrupt priority of {} is out of bound", irq);
                    return;
                }
                self.priority[irq] = (value & 7) as u8;
                trace!(target: "PLIC", "set interrupt priority of {} to {}", irq, value & 7);
            }
            // Pending bits are not writable
            ADDR_PENDING_START..=ADDR_PENDING_END => {
                error!(target: "PLIC", "illegal write to interrupt pending 0x{:x} = 0x{:x}", addr, value);
                return;
            }
            // Interrupt enable bits
            ADDR_ENABLE_START..=0x1FFFFF => {
                let ctx = (addr - ADDR_ENABLE_START) / ADDR_ENABLE_SIZE;
                let irq = (addr & (ADDR_ENABLE_SIZE - 1)) * 8;
                // Out of bound write
                if ctx >= self.enable.len() || irq >= 32 {
                    error!(target: "PLIC", "{} access to interrupt enable of {}-{} is out of bound", ctx, irq, irq + 31);
                    return;
                }
                self.enable[ctx] = value;
                trace!(target: "PLIC", "{} set interrupt enable of {}-{} to {:b}", ctx, irq, irq + 31, value);
            }
            // Threshold and claim bits
            addr if addr >= ADDR_CONTEXT_START => {
                let ctx = (addr - ADDR_CONTEXT_START) / ADDR_CONTEXT_SIZE;
                let offset = addr & (ADDR_CONTEXT_SIZE - 1);
                // Out of bound write
                if ctx >= self.enable.len() {
                    error!(target: "PLIC", "access to context id {} is out of bound", ctx);
                    return;
                }
                match offset {
                    OFFSET_PRIORITY_THRESHOLD => {
                        self.threshold[ctx] = (value & 7) as u8;
                        trace!(target: "PLIC", "{} set priority threshold to {}", ctx, self.threshold[ctx]);
                    }
                    OFFSET_INTERRUPT_CLAIM => {
                        self.claimed &= !(1 << value);
                        trace!(target: "PLIC", "{} completed interrupt {}", ctx, value);
                    }
                    // Out of bound write
                    _ => {
                        error!(target: "PLIC", "illegal register write 0x{:x} = 0x{}", addr, value);
                        return;
                    }
                }
            }
            _ => {
                error!(target: "PLIC", "illegal register write 0x{:x} = 0x{:x}", addr, value);
                return;
            }
        }

        // Re-compute the interrupt status after each successful register write
        self.recompute_pending();
    }
}
