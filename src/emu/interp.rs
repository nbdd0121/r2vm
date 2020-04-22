use crate::io::IoMemorySync;
use crate::sim::get_memory_model;
use crate::util::AtomicExt;
use lazy_static::lazy_static;
use parking_lot::{Condvar, Mutex, MutexGuard};
use riscv::{mmu::*, Csr, Op};
use softfp::{self, F32, F64};
use std::collections::{BTreeMap, BTreeSet};
use std::convert::TryInto;
use std::sync::atomic::Ordering as MemOrder;
use std::sync::atomic::{AtomicI32, AtomicI64, AtomicU32, AtomicU64};

/// A cache line. `{CacheLine}` is composed of atomic variables because we sometimes need cross-
/// thread invalidation. Note that usually paddr isn't touched, but by keeping tag and paddr
/// together we can exploit better cache locality.
#[repr(C)]
pub struct CacheLine {
    /// Lowest bit is used to store whether this cache line is non-writable
    /// It actually stores (tag << 1) | non-writable
    pub tag: AtomicU64,
    /// It actually stores vaddr ^ paddr
    pub paddr: AtomicU64,
}

impl Default for CacheLine {
    fn default() -> Self {
        Self { tag: AtomicU64::new(i64::max_value() as u64), paddr: AtomicU64::new(0) }
    }
}

impl CacheLine {
    pub fn invalidate(&self) {
        self.tag.store(i64::max_value() as u64, MemOrder::Relaxed);
    }
}

/// Most fields of `{Context}` can only be safely accessed by the execution thread. However we do
/// ocassionally need to communicate between these threads. This `{SharedContext}` are parts that
/// can be safely accessed both from the execution thread and other harts.
#[repr(C)]
pub struct SharedContext {
    /// This field stored incoming message alarms. This field will be periodically checked by the
    /// running hart, and the running hart will enter slow path when any bit is set, therefore
    /// receive and process these messages. Alarm message include:
    /// * Whether there are interrupts not yet seen by the current hart. Interrupts
    ///   can be masked by setting SIE register or SSTATUS.SIE, yet we couldn't atomically fetch
    ///   these registers and determine if the hart should be interrupted if we are not the unique
    ///   owner of `Context`. Our solution is to use message alarms. One bit is asserted when an
    ///   external interrupt arrives. The message handler will check if it actually should take an
    ///   interrupt, or the interrupt is indeed masked out. It will clear the bit regardless SIE,
    ///   By doing so, it wouldn't need to check interrupts later.
    /// * Shutdown notice.
    pub alarm: AtomicU64,

    /// Interrupts currently pending. SIP is a very special register when doing simulation,
    /// because it could be updated from outside a running hart.
    /// * When an external interrupt asserts, it should be **OR**ed with corresponding bit.
    /// * When an external interrupt deasserts, it should be **AND**ed with a mask.
    pub mip: AtomicU64,

    /// This is the L0 cache used to accelerate simulation. If a memory request hits the cache line
    /// here, then it will not go through virtual address translation nor cache simulation.
    /// Therefore this should only contain entries that are neither in the TLB nor in the cache.
    ///
    /// The cache line should only contain valid entries for the current privilege level and ASID.
    /// Upon privilege-level switch or address space switch all entries here should be cleared.
    pub line: [CacheLine; 1024],
    pub i_line: [CacheLine; 1024],

    /// Floating point rounding mode. This will be used by softfp's get_rounding_mode callback.
    /// As instructions can specify their own rounding mode, this is not the same as frm register.
    pub rm: AtomicU32,
    /// Floating point exceptions.
    pub fflags: AtomicU32,

    /// Mutex for supporting WFI in threaded execution mode. In threaded execution mode, when WFI
    /// is executed we want to wait until an alarm is available. Ideally we protect alarm with a
    /// mutex and then we can use Condvar and other OS thread primitives. However as `alarm` is
    /// also used in assembly, we cannot put it in the Mutex. Therefore this mutex has content ()
    /// and thus is only used when paired with the Condvar. Box are used for C-layout compat.
    pub wfi_mutex: Box<Mutex<()>>,

    /// Condvar for supporting WFI. Used in pair with the mutex.
    pub wfi_condvar: Box<Condvar>,

    /// Tasks that needs to be running on the hart-specific thread, such as manipulating I-Cache.
    pub tasks: Mutex<Vec<Box<dyn FnOnce() + Send>>>,
}

impl SharedContext {
    pub fn new() -> Self {
        // Check the constant used in helper.s
        assert_eq!(offset_of!(Context, pc), 0x100);
        assert_eq!(offset_of!(Context, instret), 0x108);

        SharedContext {
            mip: AtomicU64::new(0),
            alarm: AtomicU64::new(0),
            line: unsafe {
                let mut arr: [CacheLine; 1024] = std::mem::MaybeUninit::uninit().assume_init();
                for item in arr.iter_mut() {
                    std::ptr::write(item, Default::default());
                }
                arr
            },
            i_line: unsafe {
                let mut arr: [CacheLine; 1024] = std::mem::MaybeUninit::uninit().assume_init();
                for item in arr.iter_mut() {
                    std::ptr::write(item, Default::default());
                }
                arr
            },
            rm: AtomicU32::new(0),
            fflags: AtomicU32::new(0),
            wfi_mutex: Box::new(Mutex::new(())),
            wfi_condvar: Box::new(Condvar::new()),
            tasks: Mutex::new(Vec::new()),
        }
    }

    /// Assert interrupt using the given mask.
    pub fn assert(&self, mask: u64) {
        if self.mip.fetch_or(mask, MemOrder::Relaxed) & mask != mask {
            self.alert();
        }
    }

    /// Deassert interrupt using the given mask.
    pub fn deassert(&self, mask: u64) {
        self.mip.fetch_and(!mask, MemOrder::Relaxed);
    }

    /// Trigger alarms. Will also wake up WFI.
    pub fn fire_alarm(&self, mask: u64) {
        // Necessary to order with wait_alarm
        let _guard = self.wfi_mutex.lock();
        self.alarm.fetch_or(mask, MemOrder::Relaxed);
        self.wfi_condvar.notify_one();
    }

    /// Perform a WFI operation. Return after an alarm is fired.
    pub fn wait_alarm(&self) {
        let mut guard = self.wfi_mutex.lock();
        if self.alarm.load(MemOrder::Relaxed) != 0 {
            return;
        }
        self.wfi_condvar.wait(&mut guard);
    }

    /// Inform the hart that there might be a pending interrupt, but without actually touching
    /// `mip`. This should be called, e.g. if SIE or SSTATUS is modified.
    pub fn alert(&self) {
        self.fire_alarm(1);
    }

    pub fn shutdown(&self) {
        self.fire_alarm(2);
    }

    /// Do a task on this hart's thread.
    pub fn run_on(&self, task: impl FnOnce() + Send + 'static) {
        self.tasks.lock().push(Box::new(task));
        self.alert();
    }

    pub fn clear_local_cache(&self) {
        for line in self.line.iter() {
            line.invalidate();
        }
    }

    pub fn clear_local_icache(&self) {
        for line in self.i_line.iter() {
            line.invalidate();
        }
    }

    pub fn invalidate_cache_virtual(&self, addr: u64) {
        let cache_line_size_log2 = get_memory_model().cache_line_size_log2();
        let idx = (addr >> cache_line_size_log2) as usize & 1023;
        self.line[idx].invalidate();
    }

    pub fn invalidate_cache_physical(&self, addr: u64) {
        let cache_line_size_log2 = get_memory_model().cache_line_size_log2();
        let interval = 1 << (12 - cache_line_size_log2);
        let start_idx = (addr >> cache_line_size_log2) as usize & (interval - 1);
        for i in (start_idx..1024).step_by(interval) {
            self.line[i].invalidate();
        }
    }

    pub fn invalidate_icache_virtual(&self, addr: u64) {
        let cache_line_size_log2 = get_memory_model().cache_line_size_log2();
        let idx = (addr >> cache_line_size_log2) as usize & 1023;
        self.i_line[idx].invalidate();
    }

    pub fn invalidate_icache_physical(&self, addr: u64) {
        let cache_line_size_log2 = get_memory_model().cache_line_size_log2();
        let interval = 1 << (12 - cache_line_size_log2);
        let start_idx = (addr >> cache_line_size_log2) as usize & (interval - 1);
        for i in (start_idx..1024).step_by(interval) {
            self.i_line[i].invalidate();
        }
    }

    pub fn protect_code(&self, page: u64) {
        let cache_line_size_log2 = get_memory_model().cache_line_size_log2();
        for line in self.line.iter() {
            let _ = line.tag.fetch_update_stable(
                |value| {
                    let paddr =
                        (value >> 1 << cache_line_size_log2) ^ line.paddr.load(MemOrder::Relaxed);
                    if paddr & !4095 == page { Some(value | 1) } else { None }
                },
                MemOrder::Relaxed,
                MemOrder::Relaxed,
            );
        }
    }
}

/// Context representing the CPU state of a RISC-V hart.
///
/// # Memory Layout
/// As this structure is going to be accessed directly by assembly, it is important that the fields
/// are ordered well, so it can produce more optimal assembly code. We use the following order:
/// * Firstly, most commonly accessed fields are placed at the front, for example, registers,
///   counters, etc.
/// * After that, we place a shared contexts.
/// * All other items are placed at the back.
#[repr(C)]
pub struct Context {
    pub registers: [u64; 32],
    pub pc: u64,
    pub instret: u64,

    pub cause: u64,
    pub tval: u64,

    // Performance counters
    pub minstret: u64,
    /// We want each hart to own its own mcycle counter. This is to account WFI instructin where
    /// cycles are effectively stopped, or when lock-step execution is not enforced.
    pub cycle_offset: i64,

    pub shared: SharedContext,

    // Floating point states
    pub fp_registers: [u64; 32],
    pub frm: u32,

    // For load reservation
    pub lr_addr: u64,
    pub lr_value: u64,

    // S-mode CSRs
    pub scause: u64,
    pub stval: u64,
    pub stvec: u64,
    pub sscratch: u64,
    pub sepc: u64,
    pub satp: u64,
    pub scounteren: u64,

    // M-mode CSRs
    pub mideleg: u64,
    pub medeleg: u64,
    pub mcause: u64,
    pub mtval: u64,
    pub mstatus: u64,
    pub mie: u64,
    pub mtvec: u64,
    pub mscratch: u64,
    pub mepc: u64,
    pub mcounteren: u64,

    // Current privilege level
    pub prv: u64,

    pub hartid: u64,
}

impl Context {
    pub fn test_and_set_fs(&mut self) -> Result<(), ()> {
        if cfg!(not(feature = "float")) {
            self.cause = 2;
            self.tval = 0;
            return Err(());
        }
        if self.mstatus & 0x6000 == 0 {
            self.cause = 2;
            self.tval = 0;
            return Err(());
        }
        self.mstatus |= 0x6000;
        Ok(())
    }

    pub fn test_counter(&mut self, id: u32) -> Result<(), ()> {
        if self.prv < 3 && self.mcounteren & (1 << id) == 0
            || self.prv < 1 && self.scounteren & (1 << id) == 0
        {
            self.cause = 2;
            self.tval = 0;
            return Err(());
        }
        Ok(())
    }

    /// Obtaining a bitmask of pending interrupts
    pub fn interrupt_pending(&mut self) -> u64 {
        let int_ready = self.shared.mip.load(MemOrder::Relaxed) & self.mie;
        (if self.prv != 3 || self.mstatus & 0x8 != 0 { int_ready & !self.mideleg } else { 0 })
            | (if self.prv == 0 || self.prv == 1 && self.mstatus & 0x2 != 0 {
                int_ready & self.mideleg
            } else {
                0
            })
    }

    /// Get the mcycle register of this hart.
    pub fn get_mcycle(&self) -> u64 {
        crate::event_loop().get_lockstep_cycles().wrapping_add(self.cycle_offset as u64)
    }

    /// Re-check interrupt status, potentially wake up WFI
    pub fn recheck_interrupt(&mut self) {
        let int_ready = self.shared.mip.load(MemOrder::Relaxed) & self.mie;
        if int_ready != 0 {
            if self.interrupt_pending() != 0 {
                self.shared.fire_alarm(1);
            } else {
                self.shared.fire_alarm(0);
            }
        }
    }

    /// Copy memory from the hart-specific virtual address.
    pub fn copy_from_virt_addr(&mut self, addr: u64, slice: &mut [u8]) -> Result<(), ()> {
        let mut virt_addr = addr;
        let mut phys_addr = 0;
        let mut virt_line = !addr;
        let cache_line_size_log2 = get_memory_model().cache_line_size_log2();
        for i in 0..slice.len() {
            // Enter a new cache line
            if virt_addr >> cache_line_size_log2 != virt_line {
                virt_line = virt_addr >> cache_line_size_log2;
                phys_addr = translate_read(self, virt_addr)?;
            }
            slice[i] = unsafe { *(phys_addr as *const u8) };
            virt_addr += 1;
            phys_addr += 1;
        }
        Ok(())
    }

    /// Copy memory to the hart-specific virtual address.
    pub fn copy_to_virt_addr(&mut self, addr: u64, slice: &[u8]) -> Result<(), ()> {
        let mut virt_addr = addr;
        let mut phys_addr = 0;
        let mut virt_line = !addr;
        let cache_line_size_log2 = get_memory_model().cache_line_size_log2();
        for i in 0..slice.len() {
            // Enter a new cache line
            if virt_addr >> cache_line_size_log2 != virt_line {
                virt_line = virt_addr >> cache_line_size_log2;
                phys_addr = translate_write(self, virt_addr)?;
            }
            unsafe { *(phys_addr as *mut u8) = slice[i] };
            virt_addr += 1;
            phys_addr += 1;
        }
        Ok(())
    }

    /// Translate a virtual address into physical address. When exception happens `cause` and
    /// `tval` are set and `Err` is returned.
    pub fn translate_vaddr(&mut self, addr: u64, access: AccessType) -> Result<u64, ()> {
        // Respect MPRV
        let mut prv = self.prv;
        if prv == 3 && self.mstatus & 0x20000 != 0 && access != AccessType::Execute {
            prv = (self.mstatus >> 11) & 3;
        }

        // MMU off
        if (self.satp >> 60) == 0 || prv == 3 {
            return Ok(addr);
        }

        let pte = walk_page(self.satp, addr >> 12, |addr| crate::emu::read_memory(addr as usize));
        match check_permission(pte, access, prv as u8, self.mstatus) {
            Ok(_) => Ok(pte >> 10 << 12 | addr & 4095),
            Err(_) => {
                self.cause = match access {
                    AccessType::Read => 13,
                    AccessType::Write => 15,
                    AccessType::Execute => 12,
                };
                self.tval = addr;
                Err(())
            }
        }
    }

    /// Insert a cache line into the L0 instruction cache.
    pub fn insert_instruction_cache_line(&mut self, vaddr: u64, paddr: u64) {
        let idx = vaddr >> get_memory_model().cache_line_size_log2();
        let line: &CacheLine = &self.shared.i_line[(idx & 1023) as usize];
        line.tag.store(idx, MemOrder::Relaxed);
        line.paddr.store(paddr ^ vaddr, MemOrder::Relaxed);
    }

    /// Insert a cache line into the L0 data cache.
    pub fn insert_data_cache_line(&mut self, vaddr: u64, paddr: u64, writable: bool) {
        let idx = vaddr >> get_memory_model().cache_line_size_log2();
        let line: &CacheLine = &self.shared.line[(idx & 1023) as usize];
        let tag = (idx << 1) | if writable { 0 } else { 1 };
        line.tag.store(tag, MemOrder::Relaxed);
        line.paddr.store(paddr ^ vaddr, MemOrder::Relaxed);
    }
}

/// Perform a CSR read on a context. Note that this operation performs no checks before accessing
/// them.
/// The caller should ensure:
/// * The current privilege level has enough permission to access the CSR. CSR is nicely partition
///   into regions, so privilege check can be easily done.
/// * U-mode code does not access floating point CSRs with FS == Off.
#[no_mangle]
fn read_csr(ctx: &mut Context, csr: Csr) -> Result<u64, ()> {
    Ok(match csr {
        Csr::Fflags => {
            ctx.test_and_set_fs()?;
            ctx.shared.fflags.load(MemOrder::Relaxed) as u64
        }
        Csr::Frm => {
            ctx.test_and_set_fs()?;
            ctx.frm as u64
        }
        Csr::Fcsr => {
            ctx.test_and_set_fs()?;
            ((ctx.frm << 5) | ctx.shared.fflags.load(MemOrder::Relaxed)) as u64
        }
        Csr::Cycle => {
            ctx.test_counter(0)?;
            ctx.get_mcycle()
        }
        Csr::Time => {
            ctx.test_counter(1)?;
            crate::event_loop().time()
        }
        // We assume the instret is incremented already
        Csr::Instret => {
            ctx.test_counter(2)?;
            ctx.instret - 1
        }
        Csr::Sstatus => {
            let mut value = ctx.mstatus & 0xC6122;
            // SSTATUS.FS = dirty, also set SD
            if value & 0x6000 == 0x6000 {
                value |= 0x8000000000000000
            }
            // Hard-wire UXL to 0b10, i.e. 64-bit.
            value |= 0x200000000;
            value
        }
        Csr::Sie => ctx.mie & ctx.mideleg,
        Csr::Stvec => ctx.stvec,
        Csr::Scounteren => ctx.scounteren,
        Csr::Sscratch => ctx.sscratch,
        Csr::Sepc => ctx.sepc,
        Csr::Scause => ctx.scause,
        Csr::Stval => ctx.stval,
        Csr::Sip => ctx.shared.mip.load(MemOrder::Relaxed) & ctx.mideleg,
        Csr::Satp => ctx.satp,
        Csr::Mvendorid | Csr::Marchid | Csr::Mimpid => 0,
        Csr::Mhartid => ctx.hartid,
        Csr::Mstatus => {
            let mut value = ctx.mstatus;
            // MSTATUS.FS = dirty, also set SD
            if value & 0x6000 == 0x6000 {
                value |= 0x8000000000000000
            }
            // Hard-wire UXL to 0b10, i.e. 64-bit.
            value |= 0x200000000;
            value
        }
        Csr::Misa => unimplemented!(),
        Csr::Medeleg => ctx.medeleg,
        Csr::Mideleg => ctx.mideleg,
        Csr::Mie => ctx.mie,
        Csr::Mtvec => ctx.mtvec,
        Csr::Mcounteren => ctx.mcounteren,
        Csr::Mscratch => ctx.mscratch,
        Csr::Mepc => ctx.mepc,
        Csr::Mcause => ctx.mcause,
        Csr::Mtval => ctx.mtval,
        Csr::Mip => ctx.shared.mip.load(MemOrder::Relaxed),
        Csr::Mcycle => ctx.get_mcycle(),
        Csr::Mtime => crate::event_loop().time(),
        Csr::Minstret => ctx.instret - 1,
        _ => {
            error!("read illegal csr {:x}", csr.0);
            ctx.cause = 2;
            ctx.tval = 0;
            return Err(());
        }
    })
}

/// This function does not check privilege level, so it must be checked ahead of time.
/// This function also does not check for readonly CSRs, which is handled by decoder.
#[no_mangle]
fn write_csr(ctx: &mut Context, csr: Csr, value: u64) -> Result<(), ()> {
    match csr {
        Csr::Fflags => {
            ctx.test_and_set_fs()?;
            ctx.shared.fflags.store((value & 0b11111) as u32, MemOrder::Relaxed)
        }
        Csr::Frm => {
            ctx.test_and_set_fs()?;
            ctx.frm = ((value & 0b111) as u32).min(4);
        }
        Csr::Fcsr => {
            ctx.test_and_set_fs()?;
            ctx.shared.fflags.store((value & 0b11111) as u32, MemOrder::Relaxed);
            ctx.frm = (((value >> 5) & 0b111) as u32).min(4);
        }
        Csr::Sstatus => {
            // Mask-out non-writable bits
            let old_value = ctx.mstatus;
            ctx.mstatus = old_value & !0xC6122 | value & 0xC6122;
            if ctx.interrupt_pending() != 0 {
                ctx.shared.alert()
            }
            // If MXR/SUM changed, need to flush local cache
            if old_value & 0xC0000 != ctx.mstatus & 0xC0000 {
                ctx.shared.clear_local_cache();
                ctx.shared.clear_local_icache();
            }
        }
        Csr::Sie => {
            ctx.mie = (ctx.mie & !ctx.mideleg) | (value & ctx.mideleg);
            ctx.recheck_interrupt();
        }
        Csr::Stvec => {
            // We support MODE 0 only at the moment
            if (value & 2) == 0 {
                ctx.stvec = value;
            }
        }
        Csr::Scounteren => ctx.scounteren = value & 0b111,
        Csr::Sscratch => ctx.sscratch = value,
        Csr::Sepc => ctx.sepc = value & !1,
        Csr::Scause => ctx.scause = value,
        Csr::Stval => ctx.stval = value,
        Csr::Sip => {
            if ctx.mideleg & 0x2 != 0 {
                // Only SSIP flag can be cleared by software
                if value & 0x2 != 0 {
                    ctx.shared.assert(2);
                } else {
                    ctx.shared.deassert(2);
                }
            }
        }
        Csr::Satp => {
            match value >> 60 {
                // No paging
                0 => ctx.satp = 0,
                // ASID not yet supported
                8 => ctx.satp = value,
                // We only support SV39 at the moment.
                _ => (),
            }
            ctx.shared.clear_local_cache();
            ctx.shared.clear_local_icache();
        }
        Csr::Mstatus => {
            // Mask-out non-writable bits
            let old_value = ctx.mstatus;
            ctx.mstatus = old_value & !0x7E79AA | value & 0x7E79AA;
            if ctx.interrupt_pending() != 0 {
                ctx.shared.alert();
            }

            // If MPRV/MPP/MXR/SUM changed, need to flush local cache
            if old_value & 0xE1800 != ctx.mstatus & 0xE1800 {
                ctx.shared.clear_local_cache();
                ctx.shared.clear_local_icache();
            }
        }
        Csr::Misa => (),
        Csr::Medeleg => ctx.medeleg = value & 0xB35D,
        Csr::Mideleg => {
            ctx.mideleg = value & 0x222;
            if ctx.interrupt_pending() != 0 {
                ctx.shared.alert()
            }
        }
        Csr::Mie => {
            ctx.mie = value & 0xAAA;
            ctx.recheck_interrupt();
        }
        Csr::Mtvec => {
            // We support MODE 0 only at the moment
            if (value & 2) == 0 {
                ctx.mtvec = value;
            }
        }
        Csr::Mcounteren => ctx.mcounteren = value & 0b111,
        Csr::Mscratch => ctx.mscratch = value,
        Csr::Mepc => ctx.mepc = value & !1,
        Csr::Mcause => ctx.mcause = value,
        Csr::Mtval => ctx.mtval = value,
        Csr::Mip => {
            ctx.shared.deassert(0x222 & !value);
            ctx.shared.assert(0x222 & value);
        }
        Csr::Minstret => ctx.instret = value,
        Csr(0x800) if cfg!(feature = "simcsr") => {
            crate::shutdown(crate::ExitReason::SwitchModel(value as usize));
        }
        Csr(0x801) if cfg!(feature = "simcsr") => {
            crate::shutdown(if value == 0 {
                crate::ExitReason::ClearStats
            } else {
                crate::ExitReason::PrintStats
            });
        }
        _ => {
            error!("write illegal csr {:x} = {:x}", csr.0, value);
            ctx.cause = 2;
            ctx.tval = 0;
            return Err(());
        }
    }
    Ok(())
}

pub fn icache_invalidate(start: usize, end: usize) {
    let start = (start & !4095) as u64;
    let end = ((end + 4095) & !4095) as u64;

    let mut prot = CODE_PROT.lock();

    // Go through the CODE_PROT to see if we actually need invalidation
    let mut need_inv = false;
    for page in (start..end).step_by(4096) {
        if prot.remove(&(page >> 12)) {
            need_inv = true;
        }
    }

    if need_inv {
        trace!(target: "CodeProt", "invalidate {:x} to {:x}", start, end);
        for i in 0..crate::core_count() {
            // `run_on` will queue the closure for execution the next time interrupt is checked
            // on the target hart. As SFENCE.VMA/FENCE.I will terminate a basic block and thus
            // imply a cache check, this guarantees that the closure is definitely executed before
            // before the software expects a coherence instruction cache.
            crate::shared_context(i).run_on(move || {
                let mut icache = icache(i as u64);
                let keys: Vec<u64> = icache.u_map.range(start..end).map(|(k, _)| *k).collect();
                for key in keys {
                    let blk = icache.u_map.remove(&key).unwrap();
                    unsafe { *(blk.1 as *mut u8) = 0xC3 }
                }
                let keys: Vec<u64> = icache.s_map.range(start..end).map(|(k, _)| *k).collect();
                for key in keys {
                    let blk = icache.s_map.remove(&key).unwrap();
                    unsafe { *(blk.1 as *mut u8) = 0xC3 }
                }
                let keys: Vec<u64> = icache.m_map.range(start..end).map(|(k, _)| *k).collect();
                for key in keys {
                    let blk = icache.m_map.remove(&key).unwrap();
                    unsafe { *(blk.1 as *mut u8) = 0xC3 }
                }
            });
        }
    }
}

#[inline(never)]
#[no_mangle]
fn insn_translate_cache_miss(ctx: &mut Context, addr: u64) -> Result<u64, ()> {
    get_memory_model().instruction_access(ctx, addr)
}

fn insn_translate(ctx: &mut Context, addr: u64) -> Result<u64, ()> {
    let idx = addr >> get_memory_model().cache_line_size_log2();
    let line = &ctx.shared.i_line[(idx & 1023) as usize];
    let paddr = if line.tag.load(MemOrder::Relaxed) != idx {
        insn_translate_cache_miss(ctx, addr)?
    } else {
        line.paddr.load(MemOrder::Relaxed) ^ addr
    };
    Ok(paddr)
}

#[inline(never)]
#[export_name = "translate_cache_miss"]
fn translate_cache_miss(ctx: &mut Context, addr: u64, write: bool) -> Result<u64, ()> {
    let out = get_memory_model().data_access(ctx, addr, write)?;
    if write {
        icache_invalidate(out as usize, out as usize + 1);
    }
    Ok(out)
}

fn translate_read(ctx: &mut Context, addr: u64) -> Result<usize, ()> {
    let idx = addr >> get_memory_model().cache_line_size_log2();
    let line = &ctx.shared.line[(idx & 1023) as usize];
    let paddr = if (line.tag.load(MemOrder::Relaxed) >> 1) != idx {
        translate_cache_miss(ctx, addr, false)?
    } else {
        line.paddr.load(MemOrder::Relaxed) ^ addr
    };
    Ok(paddr as usize)
}

fn read_vaddr<T>(ctx: &mut Context, addr: u64) -> Result<&'static T, ()> {
    ctx.minstret += 1;
    Ok(unsafe { &*(translate_read(ctx, addr)? as *const T) })
}

fn translate_write(ctx: &mut Context, addr: u64) -> Result<usize, ()> {
    let idx = addr >> get_memory_model().cache_line_size_log2();
    let line = &ctx.shared.line[(idx & 1023) as usize];
    let paddr = if line.tag.load(MemOrder::Relaxed) != (idx << 1) {
        translate_cache_miss(ctx, addr, true)?
    } else {
        line.paddr.load(MemOrder::Relaxed) ^ addr
    };
    Ok(paddr as usize)
}

fn ptr_vaddr_x<T>(ctx: &mut Context, addr: u64) -> Result<&'static mut T, ()> {
    ctx.minstret += 1;
    Ok(unsafe { &mut *(translate_write(ctx, addr)? as *mut T) })
}

/// DBT-ed instruction cache
/// ========================
///
/// It is vital that we make keep instruction cache coherent with the main memory. Alternatively we
/// can make use of the fence.i/sfence.vma instruction, but we would not like to flush the entire
/// cache when we see them because flushing the cache is very expensive, and modifying code in
/// icache is relatively rare.
///
/// It is very difficult to remove entries from the code cache, as there might be another hart
/// actively executing the code. To avoid messing around this scenario, we does not allow individual
/// cached blocks to be removed. Instead, we simply discard the pointer into the code cache so the
/// invalidated block will no longer be used in the future.
///
/// To avoid infinite growth of the cache, we will flush the cache if the amount of DBT-ed code
/// get large. This is achieved by partitioning the whole memory into two halves. Whenever we
/// cross the boundary and start allocating on the other half, we flush all pointers into the
/// code cache. The code currently executing will return after their current basic block is
/// finished, so we don't have to worry about overwriting code that is currently executing (
/// we cannot fill the entire half in a basic block's time). The allocating block will span two
/// partitions, but we don't have to worry about this, as it uses the very end of one half, so
/// next flush when crossing boundary again will invalidate it while not overwriting it.
///
/// Things may be a lot more complicated if we start to implement basic block chaining for extra
/// speedup. In that case we probably need some pseudo-IPI stuff to make sure nobody is executing
/// flushed or overwritten basic blocks.
const HEAP_SIZE: usize = 1024 * 1024 * 32;

struct ICache {
    // The tuple stores (start, non-speculative start)
    u_map: BTreeMap<u64, (usize, usize)>,
    s_map: BTreeMap<u64, (usize, usize)>,
    m_map: BTreeMap<u64, (usize, usize)>,
    heap_start: usize,
    heap_offset: usize,
}

impl ICache {
    fn new(ptr: usize) -> ICache {
        ICache {
            u_map: BTreeMap::default(),
            s_map: BTreeMap::default(),
            m_map: BTreeMap::default(),
            heap_start: ptr,
            heap_offset: 0,
        }
    }

    // Get the space left in I-Cache.
    fn space(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                (self.heap_offset + self.heap_start) as *mut u8,
                HEAP_SIZE - self.heap_offset,
            )
        }
    }

    fn rollover(&mut self) {
        self.heap_offset = 0;
        self.u_map.clear();
        self.s_map.clear();
        self.m_map.clear();
        debug!("icache {:x} rollover", self.heap_start);
    }

    // Commit space of some size
    fn commit(&mut self, size: usize) {
        self.heap_offset += size;
        assert!(self.heap_offset <= HEAP_SIZE);
    }
}

lazy_static! {
    static ref ICACHE: Vec<Mutex<ICache>> = {
        let core_count = crate::core_count();
        let size = HEAP_SIZE * core_count;
        let ptr = unsafe { libc::mmap(0x7ffec0000000 as *mut _, size as _, libc::PROT_READ|libc::PROT_WRITE|libc::PROT_EXEC, libc::MAP_ANONYMOUS | libc::MAP_PRIVATE, -1, 0) };
        assert_eq!(ptr, 0x7ffec0000000 as *mut _);
        let ptr = ptr as usize;
        let mut vec = Vec::with_capacity(core_count);
        for i in 0..core_count {
            vec.push(Mutex::new(ICache::new(ptr + HEAP_SIZE * i)));
        }

        if crate::get_flags().perf {
            use std::io::Write;
            let mut perf_map = std::fs::File::create(format!("/tmp/perf-{}.map", std::process::id())).unwrap();
            writeln!(perf_map, "{:x} {:x} (dbt code)", ptr, size).unwrap();
        }

        vec
    };

    /// To prevent needing to flush the entire translation cache when SFENCE.VMA/FENCE.I is
    /// executed, we instead guarantee that the entries active in translation caches truthfully
    /// represent the contents in RAM. i.e. we guarantee that whenever there is a write, the
    /// relevant entries in code cache are invalidated.
    ///
    /// This field is to provide this guarantee. There are a few requirements for the correct
    /// usage:
    /// (1) A writable entry should be inserted into D-Cache with CODE_PROT locked.
    /// (2) When inserting a I-Cache entry, CODE_PROT must be locked and updated while
    ///     (a) invalidating D-Cache entries
    ///     (b) before the I-Cache insertion happens.
    /// (3) I-Cache invalidation must be queued with CODE_PROT locked.
    ///
    /// (1) and (2)(a) guarantee that when there is a new piece of code being translated, next memory
    /// access to that location will always hit the slow path, because CODE_PROT serialises them.
    /// (2)(b) and (3) guarantee that after a write can be serialised with a concurrent code
    /// translation. In such situation, as CODE_PROT is updated before actual translation, the
    /// queued invalidation callback will be executed after the translated code is inserted into
    /// the I-Cache.
    static ref CODE_PROT: Mutex<BTreeSet<u64>> = {
        Mutex::new(BTreeSet::default())
    };
}

fn icache(hartid: u64) -> MutexGuard<'static, ICache> {
    ICACHE[hartid as usize].lock()
}

pub fn icache_reset() {
    for icache in ICACHE.iter() {
        let mut icache = icache.lock();
        icache.u_map.clear();
        icache.s_map.clear();
        icache.m_map.clear();
        icache.heap_offset = 0;
    }
}

/// Broadcast sfence
fn global_sfence(ctx: &mut Context, mask: u64, asid: Option<u16>, vpn: Option<u64>) {
    get_memory_model().before_sfence_vma(ctx, mask, asid, vpn);
    for i in 0..crate::core_count() {
        if mask & (1 << i) == 0 {
            continue;
        }
        let ctx = crate::shared_context(i);
        ctx.clear_local_cache();
        ctx.clear_local_icache();
    }
}

fn sbi_call(ctx: &mut Context, nr: u64, arg0: u64, arg1: u64, arg2: u64, arg3: u64) -> u64 {
    match nr {
        0 => {
            (*super::CLINT).write_sync(0x4000 + ctx.hartid as usize * 8, arg0, 8);
            0
        }
        1 => {
            crate::io::console::console_putchar(arg0 as u8);
            0
        }
        2 => crate::io::console::console_getchar() as u64,
        3 => {
            ctx.shared.deassert(2);
            0
        }
        4 => {
            let mask: u64 = crate::emu::read_memory(
                ctx.translate_vaddr(arg0, AccessType::Read).unwrap() as usize,
            );
            for i in 0..crate::core_count() {
                if mask & (1 << i) == 0 {
                    continue;
                }
                crate::shared_context(i).assert(2);
            }
            0
        }
        5 => {
            let mask: u64 = if arg0 == 0 {
                u64::max_value()
            } else {
                crate::emu::read_memory(
                    ctx.translate_vaddr(arg0, AccessType::Read).unwrap() as usize
                )
            };
            get_memory_model().before_fence_i(ctx, mask);
            for i in 0..crate::core_count() {
                if mask & (1 << i) == 0 {
                    continue;
                }
                crate::shared_context(i).clear_local_icache();
            }
            0
        }
        6 => {
            let mask: u64 = if arg0 == 0 {
                u64::max_value()
            } else {
                crate::emu::read_memory(
                    ctx.translate_vaddr(arg0, AccessType::Read).unwrap() as usize
                )
            };
            global_sfence(ctx, mask, None, if arg2 == 4096 { Some(arg1 & !4095) } else { None });
            0
        }
        7 => {
            let mask: u64 = if arg0 == 0 {
                u64::max_value()
            } else {
                crate::emu::read_memory(
                    ctx.translate_vaddr(arg0, AccessType::Read).unwrap() as usize
                )
            };
            global_sfence(
                ctx,
                mask,
                Some(arg3 as u16),
                if arg2 == 4096 { Some(arg1 & !4095) } else { None },
            );
            0
        }
        8 => {
            crate::shutdown(crate::ExitReason::Exit(0));
            0
        }
        _ => {
            warn!("unknown sbi call {}", nr);
            (-2i64) as u64
        }
    }
}

// #region softfp init
//

fn get_rounding_mode() -> softfp::RoundingMode {
    let ctx = unsafe { &(*crate::fiber::Fiber::scratchpad::<Context>()).shared };
    ctx.rm.load(MemOrder::Relaxed).try_into().unwrap()
}

fn set_exception_flags(flags: softfp::ExceptionFlags) {
    let ctx = unsafe { &(*crate::fiber::Fiber::scratchpad::<Context>()).shared };
    ctx.fflags.fetch_or(flags.bits(), MemOrder::Relaxed);
}

/// Initialise softfp environemnt
pub fn init_fp() {
    softfp::register_get_rounding_mode(get_rounding_mode);
    softfp::register_set_exception_flag(set_exception_flags);
}

//
// #endregion

/// Perform a single step of instruction.
/// This function assumes a post-incremented PC.
/// This function does not check privilege level, so it must be checked ahead of time.
fn step(ctx: &mut Context, op: &Op, compressed: bool) -> Result<(), ()> {
    macro_rules! read_reg {
        ($rs: expr) => {{
            let rs = $rs as usize;
            if rs >= 32 {
                unsafe { std::hint::unreachable_unchecked() }
            }
            ctx.registers[rs]
        }};
    }
    macro_rules! read_32 {
        ($rs: expr) => {{
            let rs = $rs as usize;
            if rs >= 32 {
                unsafe { std::hint::unreachable_unchecked() }
            }
            ctx.registers[rs] as u32
        }};
    }
    macro_rules! write_reg {
        ($rd: expr, $expression:expr) => {{
            let rd = $rd as usize;
            let value: u64 = $expression;
            if rd >= 32 {
                unsafe { std::hint::unreachable_unchecked() }
            }
            if rd != 0 {
                ctx.registers[rd] = value
            }
        }};
    }
    macro_rules! write_32 {
        ($rd: expr, $expression:expr) => {{
            let rd = $rd as usize;
            let value: u32 = $expression;
            if rd >= 32 {
                unsafe { std::hint::unreachable_unchecked() }
            }
            if rd != 0 {
                ctx.registers[rd] = value as i32 as u64
            }
        }};
    }
    macro_rules! read_fs {
        ($rs: expr) => {{
            let rs = $rs as usize;
            if rs >= 32 {
                unsafe { std::hint::unreachable_unchecked() }
            }
            F32::new(ctx.fp_registers[rs] as u32)
        }};
    }
    macro_rules! read_fd {
        ($rs: expr) => {{
            let rs = $rs as usize;
            if rs >= 32 {
                unsafe { std::hint::unreachable_unchecked() }
            }
            F64::new(ctx.fp_registers[rs])
        }};
    }
    macro_rules! write_fs {
        ($frd: expr, $expression:expr) => {{
            let frd = $frd as usize;
            let value: F32 = $expression;
            if frd >= 32 {
                unsafe { std::hint::unreachable_unchecked() }
            }
            ctx.fp_registers[frd] = value.0 as u64 | 0xffffffff00000000
        }};
    }
    macro_rules! write_fd {
        ($frd: expr, $expression:expr) => {{
            let frd = $frd as usize;
            let value: F64 = $expression;
            if frd >= 32 {
                unsafe { std::hint::unreachable_unchecked() }
            }
            ctx.fp_registers[frd] = value.0
        }};
    }
    macro_rules! set_rm {
        ($rm: expr) => {{
            ctx.test_and_set_fs()?;
            let rm = if $rm == 0b111 { ctx.frm } else { $rm as u32 };
            assert!(rm <= 4);
            ctx.shared.rm.store(rm, MemOrder::Relaxed);
        }};
    }
    macro_rules! clear_flags {
        () => {};
    }
    macro_rules! update_flags {
        () => {};
    }
    macro_rules! pc_pre {
        () => {
            ctx.pc.wrapping_sub(if compressed { 2 } else { 4 })
        };
    }
    macro_rules! trap {
        ($cause: expr, $tval: expr) => {{
            ctx.cause = $cause;
            ctx.tval = $tval;
            return Err(());
        }};
    }

    match *op {
        Op::Illegal => trap!(2, 0),
        /* LOAD */
        Op::Lb { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            write_reg!(rd, *read_vaddr::<u8>(ctx, vaddr)? as i8 as u64);
        }
        Op::Lh { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 1 != 0 {
                trap!(4, vaddr)
            }
            write_reg!(rd, *read_vaddr::<u16>(ctx, vaddr)? as i16 as u64);
        }
        Op::Lw { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 3 != 0 {
                trap!(4, vaddr)
            }
            write_reg!(rd, *read_vaddr::<u32>(ctx, vaddr)? as i32 as u64);
        }
        Op::Ld { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 7 != 0 {
                trap!(4, vaddr)
            }
            write_reg!(rd, *read_vaddr::<u64>(ctx, vaddr)?);
        }
        Op::Lbu { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            write_reg!(rd, *read_vaddr::<u8>(ctx, vaddr)? as u64);
        }
        Op::Lhu { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 1 != 0 {
                trap!(4, vaddr)
            }
            write_reg!(rd, *read_vaddr::<u16>(ctx, vaddr)? as u64);
        }
        Op::Lwu { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 3 != 0 {
                trap!(4, vaddr)
            }
            write_reg!(rd, *read_vaddr::<u32>(ctx, vaddr)? as u64);
        }
        /* OP-IMM */
        Op::Addi { rd, rs1, imm } => write_reg!(rd, read_reg!(rs1).wrapping_add(imm as u64)),
        Op::Slli { rd, rs1, imm } => write_reg!(rd, read_reg!(rs1) << imm),
        Op::Slti { rd, rs1, imm } => {
            write_reg!(rd, ((read_reg!(rs1) as i64) < (imm as i64)) as u64)
        }
        Op::Sltiu { rd, rs1, imm } => write_reg!(rd, (read_reg!(rs1) < (imm as u64)) as u64),
        Op::Xori { rd, rs1, imm } => write_reg!(rd, read_reg!(rs1) ^ (imm as u64)),
        Op::Srli { rd, rs1, imm } => write_reg!(rd, read_reg!(rs1) >> imm),
        Op::Srai { rd, rs1, imm } => write_reg!(rd, ((read_reg!(rs1) as i64) >> imm) as u64),
        Op::Ori { rd, rs1, imm } => write_reg!(rd, read_reg!(rs1) | (imm as u64)),
        Op::Andi { rd, rs1, imm } => write_reg!(rd, read_reg!(rs1) & (imm as u64)),
        /* MISC-MEM */
        Op::Fence => std::sync::atomic::fence(MemOrder::SeqCst),
        Op::FenceI => {
            get_memory_model().before_fence_i(ctx, 1 << ctx.hartid);
            ctx.shared.clear_local_icache();
        }
        /* OP-IMM-32 */
        Op::Addiw { rd, rs1, imm } => {
            write_reg!(rd, ((read_reg!(rs1) as i32).wrapping_add(imm)) as u64)
        }
        Op::Slliw { rd, rs1, imm } => write_reg!(rd, ((read_reg!(rs1) as i32) << imm) as u64),
        Op::Srliw { rd, rs1, imm } => {
            write_reg!(rd, (((read_reg!(rs1) as u32) >> imm) as i32) as u64)
        }
        Op::Sraiw { rd, rs1, imm } => write_reg!(rd, ((read_reg!(rs1) as i32) >> imm) as u64),
        /* STORE */
        Op::Sb { rs1, rs2, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            let paddr = ptr_vaddr_x(ctx, vaddr)?;
            *paddr = read_reg!(rs2) as u8;
        }
        Op::Sh { rs1, rs2, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 1 != 0 {
                trap!(5, vaddr)
            }
            let paddr = ptr_vaddr_x(ctx, vaddr)?;
            *paddr = read_reg!(rs2) as u16;
        }
        Op::Sw { rs1, rs2, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 3 != 0 {
                trap!(5, vaddr)
            }
            let paddr = ptr_vaddr_x(ctx, vaddr)?;
            *paddr = read_reg!(rs2) as u32;
        }
        Op::Sd { rs1, rs2, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 7 != 0 {
                trap!(5, vaddr)
            }
            let paddr = ptr_vaddr_x(ctx, vaddr)?;
            *paddr = read_reg!(rs2) as u64;
        }
        /* OP */
        Op::Add { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1).wrapping_add(read_reg!(rs2))),
        Op::Sub { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1).wrapping_sub(read_reg!(rs2))),
        Op::Sll { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1) << (read_reg!(rs2) & 63)),
        Op::Slt { rd, rs1, rs2 } => {
            write_reg!(rd, ((read_reg!(rs1) as i64) < (read_reg!(rs2) as i64)) as u64)
        }
        Op::Sltu { rd, rs1, rs2 } => write_reg!(rd, (read_reg!(rs1) < read_reg!(rs2)) as u64),
        Op::Xor { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1) ^ read_reg!(rs2)),
        Op::Srl { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1) >> (read_reg!(rs2) & 63)),
        Op::Sra { rd, rs1, rs2 } => {
            write_reg!(rd, ((read_reg!(rs1) as i64) >> (read_reg!(rs2) & 63)) as u64)
        }
        Op::Or { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1) | read_reg!(rs2)),
        Op::And { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1) & read_reg!(rs2)),
        /* LUI */
        Op::Lui { rd, imm } => write_reg!(rd, imm as u64),
        Op::Addw { rd, rs1, rs2 } => {
            write_reg!(rd, ((read_reg!(rs1) as i32).wrapping_add(read_reg!(rs2) as i32)) as u64)
        }
        Op::Subw { rd, rs1, rs2 } => {
            write_reg!(rd, ((read_reg!(rs1) as i32).wrapping_sub(read_reg!(rs2) as i32)) as u64)
        }
        Op::Sllw { rd, rs1, rs2 } => {
            write_reg!(rd, ((read_reg!(rs1) as i32) << (read_reg!(rs2) & 31)) as u64)
        }
        Op::Srlw { rd, rs1, rs2 } => {
            write_reg!(rd, (((read_reg!(rs1) as u32) >> (read_reg!(rs2) & 31)) as i32) as u64)
        }
        Op::Sraw { rd, rs1, rs2 } => {
            write_reg!(rd, ((read_reg!(rs1) as i32) >> (read_reg!(rs2) & 31)) as u64)
        }
        /* AUIPC */
        Op::Auipc { rd, imm } => write_reg!(rd, pc_pre!().wrapping_add(imm as u64)),
        /* BRANCH */
        // Same as auipc, PC-relative instructions are relative to the origin pc instead of the incremented one.
        Op::Beq { rs1, rs2, imm } => {
            if read_reg!(rs1) == read_reg!(rs2) {
                ctx.pc = pc_pre!().wrapping_add(imm as u64);
            }
        }
        Op::Bne { rs1, rs2, imm } => {
            if read_reg!(rs1) != read_reg!(rs2) {
                ctx.pc = pc_pre!().wrapping_add(imm as u64);
            }
        }
        Op::Blt { rs1, rs2, imm } => {
            if (read_reg!(rs1) as i64) < (read_reg!(rs2) as i64) {
                ctx.pc = pc_pre!().wrapping_add(imm as u64);
            }
        }
        Op::Bge { rs1, rs2, imm } => {
            if (read_reg!(rs1) as i64) >= (read_reg!(rs2) as i64) {
                ctx.pc = pc_pre!().wrapping_add(imm as u64);
            }
        }
        Op::Bltu { rs1, rs2, imm } => {
            if read_reg!(rs1) < read_reg!(rs2) {
                ctx.pc = pc_pre!().wrapping_add(imm as u64);
            }
        }
        Op::Bgeu { rs1, rs2, imm } => {
            if read_reg!(rs1) >= read_reg!(rs2) {
                ctx.pc = pc_pre!().wrapping_add(imm as u64);
            }
        }
        /* JALR */
        Op::Jalr { rd, rs1, imm } => {
            let new_pc = (read_reg!(rs1).wrapping_add(imm as u64)) & !1;
            write_reg!(rd, ctx.pc);
            ctx.pc = new_pc;
        }
        /* JAL */
        Op::Jal { rd, imm } => {
            write_reg!(rd, ctx.pc);
            ctx.pc = pc_pre!().wrapping_add(imm as u64);
        }
        /* SYSTEM */
        Op::Ecall => match ctx.prv {
            0 => {
                if crate::get_flags().prv == 0 {
                    ctx.registers[10] = unsafe {
                        crate::emu::syscall(
                            ctx.registers[17],
                            ctx.registers[10],
                            ctx.registers[11],
                            ctx.registers[12],
                            ctx.registers[13],
                            ctx.registers[14],
                            ctx.registers[15],
                        )
                    };
                } else {
                    trap!(8, 0)
                }
            }
            1 => {
                if crate::get_flags().prv == 1 {
                    ctx.registers[10] = sbi_call(
                        ctx,
                        ctx.registers[17],
                        ctx.registers[10],
                        ctx.registers[11],
                        ctx.registers[12],
                        ctx.registers[13],
                    )
                } else {
                    trap!(9, 0)
                }
            }
            3 => {
                // Note: We provide SBI interface to our M-mode, so the firmware can simple a few things down.
                // This is non-standard.
                ctx.registers[10] = sbi_call(
                    ctx,
                    ctx.registers[17],
                    ctx.registers[10],
                    ctx.registers[11],
                    ctx.registers[12],
                    ctx.registers[13],
                );
            }
            _ => unreachable!(),
        },
        Op::Ebreak => trap!(3, 0),
        Op::Csrrw { rd, rs1, csr } => {
            let result = if rd != 0 { read_csr(ctx, csr)? } else { 0 };
            write_csr(ctx, csr, read_reg!(rs1))?;
            write_reg!(rd, result);
        }
        Op::Csrrs { rd, rs1, csr } => {
            let result = read_csr(ctx, csr)?;
            if rs1 != 0 {
                write_csr(ctx, csr, result | read_reg!(rs1))?
            }
            write_reg!(rd, result);
        }
        Op::Csrrc { rd, rs1, csr } => {
            let result = read_csr(ctx, csr)?;
            if rs1 != 0 {
                write_csr(ctx, csr, result & !read_reg!(rs1))?
            }
            write_reg!(rd, result);
        }
        Op::Csrrwi { rd, imm, csr } => {
            let result = if rd != 0 { read_csr(ctx, csr)? } else { 0 };
            write_csr(ctx, csr, imm as u64)?;
            write_reg!(rd, result);
        }
        Op::Csrrsi { rd, imm, csr } => {
            let result = read_csr(ctx, csr)?;
            if imm != 0 {
                write_csr(ctx, csr, result | imm as u64)?
            }
            write_reg!(rd, result);
        }
        Op::Csrrci { rd, imm, csr } => {
            let result = read_csr(ctx, csr)?;
            if imm != 0 {
                write_csr(ctx, csr, result & !(imm as u64))?
            }
            write_reg!(rd, result);
        }

        /* F-extension */
        Op::Flw { frd, rs1, imm } => {
            ctx.test_and_set_fs()?;
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 3 != 0 {
                trap!(4, vaddr)
            }
            write_fs!(frd, F32::new(*read_vaddr::<u32>(ctx, vaddr)?));
        }
        Op::Fsw { rs1, frs2, imm } => {
            ctx.test_and_set_fs()?;
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 3 != 0 {
                trap!(5, vaddr)
            }
            let paddr = ptr_vaddr_x(ctx, vaddr)?;
            *paddr = read_fs!(frs2).0;
        }
        Op::FaddS { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, read_fs!(frs1) + read_fs!(frs2));
            update_flags!();
        }
        Op::FsubS { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, read_fs!(frs1) - read_fs!(frs2));
            update_flags!();
        }
        Op::FmulS { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, read_fs!(frs1) * read_fs!(frs2));
            update_flags!();
        }
        Op::FdivS { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, read_fs!(frs1) / read_fs!(frs2));
            update_flags!();
        }
        Op::FsqrtS { frd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, read_fs!(frs1).square_root());
            update_flags!();
        }
        Op::FsgnjS { frd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            write_fs!(frd, read_fs!(frs1).copy_sign(read_fs!(frs2)))
        }
        Op::FsgnjnS { frd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            write_fs!(frd, read_fs!(frs1).copy_sign_negated(read_fs!(frs2)))
        }
        Op::FsgnjxS { frd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            write_fs!(frd, read_fs!(frs1).copy_sign_xored(read_fs!(frs2)))
        }
        Op::FminS { frd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            clear_flags!();
            write_fs!(frd, F32::min(read_fs!(frs1), read_fs!(frs2)));
            update_flags!();
        }
        Op::FmaxS { frd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            clear_flags!();
            write_fs!(frd, F32::max(read_fs!(frs1), read_fs!(frs2)));
            update_flags!();
        }
        Op::FcvtWS { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_32!(rd, read_fs!(frs1).convert_to_sint::<u32>());
            update_flags!();
        }
        Op::FcvtWuS { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_32!(rd, read_fs!(frs1).convert_to_uint::<u32>());
            update_flags!();
        }
        Op::FcvtLS { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_reg!(rd, read_fs!(frs1).convert_to_sint::<u64>());
            update_flags!();
        }
        Op::FcvtLuS { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_reg!(rd, read_fs!(frs1).convert_to_uint::<u64>());
            update_flags!();
        }
        Op::FmvXW { rd, frs1 } => {
            ctx.test_and_set_fs()?;
            write_32!(rd, read_fs!(frs1).0);
        }
        Op::FclassS { rd, frs1 } => {
            ctx.test_and_set_fs()?;
            write_reg!(rd, 1 << read_fs!(frs1).classify() as u32);
        }
        Op::FeqS { rd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            write_reg!(rd, (read_fs!(frs1) == read_fs!(frs2)) as u64)
        }
        Op::FltS { rd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            clear_flags!();
            write_reg!(rd, (read_fs!(frs1) < read_fs!(frs2)) as u64);
            update_flags!();
        }
        Op::FleS { rd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            clear_flags!();
            write_reg!(rd, (read_fs!(frs1) <= read_fs!(frs2)) as u64);
            update_flags!();
        }
        Op::FcvtSW { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, F32::convert_from_sint::<u32>(read_32!(rs1)));
            update_flags!();
        }
        Op::FcvtSWu { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, F32::convert_from_uint::<u32>(read_32!(rs1)));
            update_flags!();
        }
        Op::FcvtSL { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, F32::convert_from_sint::<u64>(read_reg!(rs1)));
            update_flags!();
        }
        Op::FcvtSLu { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, F32::convert_from_uint::<u64>(read_reg!(rs1)));
            update_flags!();
        }
        Op::FmvWX { frd, rs1 } => {
            ctx.test_and_set_fs()?;
            write_fs!(frd, F32::new(read_32!(rs1)));
        }
        Op::FmaddS { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, F32::fused_multiply_add(read_fs!(frs1), read_fs!(frs2), read_fs!(frs3)));
            update_flags!();
        }
        Op::FmsubS { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(
                frd,
                F32::fused_multiply_add(read_fs!(frs1), read_fs!(frs2), -read_fs!(frs3))
            );
            update_flags!();
        }
        Op::FnmsubS { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(
                frd,
                F32::fused_multiply_add(-read_fs!(frs1), read_fs!(frs2), read_fs!(frs3))
            );
            update_flags!();
        }
        Op::FnmaddS { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(
                frd,
                -F32::fused_multiply_add(read_fs!(frs1), read_fs!(frs2), read_fs!(frs3))
            );
            update_flags!();
        }

        /* D-extension */
        Op::Fld { frd, rs1, imm } => {
            ctx.test_and_set_fs()?;
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 3 != 0 {
                trap!(4, vaddr)
            }
            write_fd!(frd, F64::new(*read_vaddr::<u64>(ctx, vaddr)?));
        }
        Op::Fsd { rs1, frs2, imm } => {
            ctx.test_and_set_fs()?;
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 7 != 0 {
                trap!(5, vaddr)
            }
            let paddr = ptr_vaddr_x(ctx, vaddr)?;
            *paddr = read_fd!(frs2).0;
        }
        Op::FaddD { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, read_fd!(frs1) + read_fd!(frs2));
            update_flags!();
        }
        Op::FsubD { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, read_fd!(frs1) - read_fd!(frs2));
            update_flags!();
        }
        Op::FmulD { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, read_fd!(frs1) * read_fd!(frs2));
            update_flags!();
        }
        Op::FdivD { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, read_fd!(frs1) / read_fd!(frs2));
            update_flags!();
        }
        Op::FsqrtD { frd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, read_fd!(frs1).square_root());
            update_flags!();
        }
        Op::FsgnjD { frd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            write_fd!(frd, read_fd!(frs1).copy_sign(read_fd!(frs2)))
        }
        Op::FsgnjnD { frd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            write_fd!(frd, read_fd!(frs1).copy_sign_negated(read_fd!(frs2)))
        }
        Op::FsgnjxD { frd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            write_fd!(frd, read_fd!(frs1).copy_sign_xored(read_fd!(frs2)))
        }
        Op::FminD { frd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            clear_flags!();
            write_fd!(frd, F64::min(read_fd!(frs1), read_fd!(frs2)));
            update_flags!();
        }
        Op::FmaxD { frd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            clear_flags!();
            write_fd!(frd, F64::max(read_fd!(frs1), read_fd!(frs2)));
            update_flags!();
        }
        Op::FcvtSD { frd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, read_fd!(frs1).convert_format());
            update_flags!();
        }
        Op::FcvtDS { frd, frs1, .. } => {
            ctx.test_and_set_fs()?;
            clear_flags!();
            write_fd!(frd, read_fs!(frs1).convert_format());
            update_flags!();
        }
        Op::FcvtWD { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_32!(rd, read_fd!(frs1).convert_to_sint::<u32>());
            update_flags!();
        }
        Op::FcvtWuD { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_32!(rd, read_fd!(frs1).convert_to_uint::<u32>());
            update_flags!();
        }
        Op::FcvtLD { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_reg!(rd, read_fd!(frs1).convert_to_sint::<u64>());
            update_flags!();
        }
        Op::FcvtLuD { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_reg!(rd, read_fd!(frs1).convert_to_uint::<u64>());
            update_flags!();
        }
        Op::FmvXD { rd, frs1 } => {
            ctx.test_and_set_fs()?;
            write_reg!(rd, read_fd!(frs1).0);
        }
        Op::FclassD { rd, frs1 } => {
            ctx.test_and_set_fs()?;
            write_reg!(rd, 1 << read_fd!(frs1).classify() as u32);
        }
        Op::FeqD { rd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            write_reg!(rd, (read_fd!(frs1) == read_fd!(frs2)) as u64)
        }
        Op::FltD { rd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            clear_flags!();
            write_reg!(rd, (read_fd!(frs1) < read_fd!(frs2)) as u64);
            update_flags!();
        }
        Op::FleD { rd, frs1, frs2 } => {
            ctx.test_and_set_fs()?;
            clear_flags!();
            write_reg!(rd, (read_fd!(frs1) <= read_fd!(frs2)) as u64);
            update_flags!();
        }
        Op::FcvtDW { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, F64::convert_from_sint::<u32>(read_32!(rs1)));
            update_flags!();
        }
        Op::FcvtDWu { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, F64::convert_from_uint::<u32>(read_32!(rs1)));
            update_flags!();
        }
        Op::FcvtDL { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, F64::convert_from_sint::<u64>(read_reg!(rs1)));
            update_flags!();
        }
        Op::FcvtDLu { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, F64::convert_from_uint::<u64>(read_reg!(rs1)));
            update_flags!();
        }
        Op::FmvDX { frd, rs1 } => {
            ctx.test_and_set_fs()?;
            write_fd!(frd, F64::new(read_reg!(rs1)));
        }
        Op::FmaddD { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, F64::fused_multiply_add(read_fd!(frs1), read_fd!(frs2), read_fd!(frs3)));
            update_flags!();
        }
        Op::FmsubD { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(
                frd,
                F64::fused_multiply_add(read_fd!(frs1), read_fd!(frs2), -read_fd!(frs3))
            );
            update_flags!();
        }
        Op::FnmsubD { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(
                frd,
                F64::fused_multiply_add(-read_fd!(frs1), read_fd!(frs2), read_fd!(frs3))
            );
            update_flags!();
        }
        Op::FnmaddD { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(
                frd,
                -F64::fused_multiply_add(read_fd!(frs1), read_fd!(frs2), read_fd!(frs3))
            );
            update_flags!();
        }

        /* M-extension */
        Op::Mul { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1).wrapping_mul(read_reg!(rs2))),
        Op::Mulh { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as i64 as i128;
            let b = read_reg!(rs2) as i64 as i128;
            write_reg!(rd, ((a * b) >> 64) as u64)
        }
        Op::Mulhsu { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as i64;
            let b = read_reg!(rs2);

            // First multiply as uint128_t. This will give compiler chance to optimize better.
            let exta = a as u64 as u128;
            let extb = b as u128;
            let mut r = ((exta * extb) >> 64) as u64;

            // If rs1 < 0, then the high bits of a should be all one, but the actual bits in exta
            // is all zero. Therefore we need to compensate this error by adding multiplying
            // 0xFFFFFFFF and b, which is effective -b.
            if a < 0 {
                r = r.wrapping_sub(b)
            }
            write_reg!(rd, r)
        }
        Op::Mulhu { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as u128;
            let b = read_reg!(rs2) as u128;
            write_reg!(rd, ((a * b) >> 64) as u64)
        }
        Op::Div { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as i64;
            let b = read_reg!(rs2) as i64;
            let r = if b == 0 { -1 } else { a.wrapping_div(b) };
            write_reg!(rd, r as u64);
        }
        Op::Divu { rd, rs1, rs2 } => {
            let a = read_reg!(rs1);
            let b = read_reg!(rs2);
            let r = if b == 0 { (-1i64) as u64 } else { a / b };
            write_reg!(rd, r);
        }
        Op::Rem { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as i64;
            let b = read_reg!(rs2) as i64;
            let r = if b == 0 { a } else { a.wrapping_rem(b) };
            write_reg!(rd, r as u64);
        }
        Op::Remu { rd, rs1, rs2 } => {
            let a = read_reg!(rs1);
            let b = read_reg!(rs2);
            let r = if b == 0 { a } else { a % b };
            write_reg!(rd, r);
        }
        Op::Mulw { rd, rs1, rs2 } => {
            write_reg!(rd, ((read_reg!(rs1) as i32).wrapping_mul(read_reg!(rs2) as i32)) as u64)
        }
        Op::Divw { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as i32;
            let b = read_reg!(rs2) as i32;
            let r = if b == 0 { -1 } else { a.wrapping_div(b) };
            write_reg!(rd, r as u64);
        }
        Op::Divuw { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as u32;
            let b = read_reg!(rs2) as u32;
            let r = if b == 0 { (-1i32) as u32 } else { a / b };
            write_reg!(rd, r as i32 as u64);
        }
        Op::Remw { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as i32;
            let b = read_reg!(rs2) as i32;
            let r = if b == 0 { a } else { a.wrapping_rem(b) };
            write_reg!(rd, r as u64);
        }
        Op::Remuw { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as u32;
            let b = read_reg!(rs2) as u32;
            let r = if b == 0 { a } else { a % b };
            write_reg!(rd, r as i32 as u64);
        }

        /* A-extension */
        Op::LrW { rd, rs1, .. } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 {
                trap!(5, addr)
            }
            let ptr = ptr_vaddr_x::<AtomicU32>(ctx, addr)?;
            let value = ptr.load(MemOrder::SeqCst) as i32 as u64;
            write_reg!(rd, value);
            ctx.lr_addr = addr;
            ctx.lr_value = value;
        }
        Op::LrD { rd, rs1, .. } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 {
                trap!(5, addr)
            }
            let ptr = ptr_vaddr_x::<AtomicU64>(ctx, addr)?;
            let value = ptr.load(MemOrder::SeqCst);
            write_reg!(rd, value);
            ctx.lr_addr = addr;
            ctx.lr_value = value;
        }
        Op::ScW { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2) as u32;
            let result = if addr != ctx.lr_addr {
                1
            } else {
                let ptr = ptr_vaddr_x::<AtomicU32>(ctx, addr)?;
                match ptr.compare_exchange(
                    ctx.lr_value as u32,
                    src,
                    MemOrder::SeqCst,
                    MemOrder::SeqCst,
                ) {
                    Ok(_) => 0,
                    Err(_) => 1,
                }
            };
            write_reg!(rd, result);
        }
        Op::ScD { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2);
            let result = if addr != ctx.lr_addr {
                1
            } else {
                let ptr = ptr_vaddr_x::<AtomicU64>(ctx, addr)?;
                match ptr.compare_exchange(ctx.lr_value, src, MemOrder::SeqCst, MemOrder::SeqCst) {
                    Ok(_) => 0,
                    Err(_) => 1,
                }
            };
            write_reg!(rd, result)
        }
        Op::AmoswapW { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<AtomicU32>(ctx, addr)?;
            let current = ptr.swap(src, MemOrder::SeqCst);
            write_32!(rd, current);
        }
        Op::AmoswapD { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<AtomicU64>(ctx, addr)?;
            let current = ptr.swap(src, MemOrder::SeqCst);
            write_reg!(rd, current);
        }
        Op::AmoaddW { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<AtomicU32>(ctx, addr)?;
            let current = ptr.fetch_add(src, MemOrder::SeqCst);
            write_32!(rd, current);
        }
        Op::AmoaddD { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<AtomicU64>(ctx, addr)?;
            let current = ptr.fetch_add(src, MemOrder::SeqCst);
            write_reg!(rd, current);
        }
        Op::AmoandW { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<AtomicU32>(ctx, addr)?;
            let current = ptr.fetch_and(src, MemOrder::SeqCst);
            write_32!(rd, current);
        }
        Op::AmoandD { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<AtomicU64>(ctx, addr)?;
            let current = ptr.fetch_and(src, MemOrder::SeqCst);
            write_reg!(rd, current);
        }
        Op::AmoorW { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<AtomicU32>(ctx, addr)?;
            let current = ptr.fetch_or(src, MemOrder::SeqCst);
            write_32!(rd, current);
        }
        Op::AmoorD { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<AtomicU64>(ctx, addr)?;
            let current = ptr.fetch_or(src, MemOrder::SeqCst);
            write_reg!(rd, current);
        }
        Op::AmoxorW { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<AtomicU32>(ctx, addr)?;
            let current = ptr.fetch_xor(src, MemOrder::SeqCst);
            write_32!(rd, current);
        }
        Op::AmoxorD { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<AtomicU64>(ctx, addr)?;
            let current = ptr.fetch_xor(src, MemOrder::SeqCst);
            write_reg!(rd, current);
        }
        Op::AmominW { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<AtomicI32>(ctx, addr)?;
            let current = ptr.fetch_min_stable(src as i32, MemOrder::SeqCst);
            write_32!(rd, current as u32);
        }
        Op::AmominD { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<AtomicI64>(ctx, addr)?;
            let current = ptr.fetch_min_stable(src as i64, MemOrder::SeqCst);
            write_reg!(rd, current as u64);
        }
        Op::AmomaxW { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<AtomicI32>(ctx, addr)?;
            let current = ptr.fetch_max_stable(src as i32, MemOrder::SeqCst);
            write_32!(rd, current as u32);
        }
        Op::AmomaxD { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<AtomicI64>(ctx, addr)?;
            let current = ptr.fetch_max_stable(src as i64, MemOrder::SeqCst);
            write_reg!(rd, current as u64);
        }
        Op::AmominuW { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<AtomicU32>(ctx, addr)?;
            let current = ptr.fetch_min_stable(src, MemOrder::SeqCst);
            write_32!(rd, current);
        }
        Op::AmominuD { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<AtomicU64>(ctx, addr)?;
            let current = ptr.fetch_min_stable(src, MemOrder::SeqCst);
            write_reg!(rd, current);
        }
        Op::AmomaxuW { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<AtomicU32>(ctx, addr)?;
            let current = ptr.fetch_max_stable(src, MemOrder::SeqCst);
            write_32!(rd, current);
        }
        Op::AmomaxuD { rd, rs1, rs2, .. } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 {
                trap!(5, addr)
            }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<AtomicU64>(ctx, addr)?;
            let current = ptr.fetch_max_stable(src, MemOrder::SeqCst);
            write_reg!(rd, current);
        }

        /* Privileged */
        Op::Mret => {
            ctx.pc = ctx.mepc;

            // Set privilege according to MPP
            let new_prv = (ctx.mstatus & 0x1800) >> 11;
            assert_ne!(new_prv, 2);
            ctx.prv = new_prv;
            if new_prv != 3 {
                // Switch from M-mode to S/U-mode, clear local cache
                ctx.shared.clear_local_cache();
                ctx.shared.clear_local_icache();
            }

            // Set MIE according to MPIE
            ctx.mstatus = (ctx.mstatus & !0x8) | (ctx.mstatus & 0x80) >> 4;
            // Set MPIE to 1
            ctx.mstatus |= 0x80;
            // Set MPP to U
            ctx.mstatus &= !0x1800;

            if ctx.interrupt_pending() != 0 {
                ctx.shared.alert()
            }
        }
        Op::Sret => {
            ctx.pc = ctx.sepc;

            // Set privilege according to SPP
            if (ctx.mstatus & 0x100) != 0 {
                ctx.prv = 1;
            } else {
                ctx.prv = 0;
                // Switch from S-mode to U-mode, clear local cache
                ctx.shared.clear_local_cache();
                ctx.shared.clear_local_icache();
            }

            // Set SIE according to SPIE
            if (ctx.mstatus & 0x20) != 0 {
                ctx.mstatus |= 0x2;
            } else {
                ctx.mstatus &= !0x2;
            }

            // Set SPIE to 1
            ctx.mstatus |= 0x20;
            // Set SPP to U
            ctx.mstatus &= !0x100;

            if ctx.interrupt_pending() != 0 {
                ctx.shared.alert()
            }
        }
        Op::Wfi => {
            if crate::threaded() {
                ctx.shared.wait_alarm()
            }
        }
        Op::SfenceVma { rs1, rs2 } => {
            let asid = if rs2 == 0 { None } else { Some(read_reg!(rs2) as u16) };
            let vpn = if rs1 == 0 { None } else { Some(read_reg!(rs1) & !4095) };
            global_sfence(ctx, 1 << ctx.hartid, asid, vpn)
        }
    }
    Ok(())
}

#[no_mangle]
pub fn riscv_step(ctx: &mut Context, op: u64) -> Result<(), ()> {
    let op: Op = unsafe { std::mem::transmute(op) };
    // The compressed bit is mainly used for PC offset calculation for PC-related operations,
    // which we have taken care in DBT, so it won't matter.
    step(ctx, &op, false)
}

fn translate_code(
    ctx: &mut Context,
    icache: &mut ICache,
    prv: u64,
    phys_pc: u64,
) -> (usize, usize) {
    let mut phys_pc_end = phys_pc;

    if crate::get_flags().disassemble {
        eprintln!("Decoding {:x}", phys_pc);
    }

    // Reserve some space for the DBT compiler.
    // This uses a very relax upper bound, enough for an entire page.
    let mut code = icache.space();
    let rollover = code.len() < 256 * 1024;
    // Rollover if the space is not sufficient for next allocation.
    if rollover {
        icache.rollover();
        code = icache.space();
    }

    let mut compiler = super::dbt::DbtCompiler::new(ctx, code);
    compiler.begin(phys_pc);

    loop {
        /// Decode a instruction at given location. If it will cross a page boundary, then Err is
        /// returned.
        fn read_insn(pc: usize) -> Result<(Op, bool, u32), u16> {
            let bits = crate::emu::read_memory::<u16>(pc);
            if bits & 3 == 3 {
                // The instruction will cross page boundary.
                if pc & 4095 == 4094 {
                    return Err(bits);
                }
                let hi_bits = crate::emu::read_memory::<u16>(pc + 2);
                let bits = (hi_bits as u32) << 16 | bits as u32;
                let op = riscv::decode(bits);
                Ok((op, false, bits))
            } else {
                let op = riscv::decode_compressed(bits);
                Ok((op, true, bits as u32))
            }
        }

        let (mut op, c, bits) = match read_insn(phys_pc_end as usize) {
            Ok(v) => v,
            Err(bits) => {
                compiler.end_cross(bits);
                break;
            }
        };
        if crate::get_flags().disassemble {
            eprintln!("{}", op.pretty_print(phys_pc_end, bits));
        }
        phys_pc_end += if c { 2 } else { 4 };

        // We must not emit code for protected ops
        if (prv as u8) < op.min_prv_level() {
            op = Op::Illegal
        }

        // The way we generate code is a bit slow for branches. The mini-optimisation here
        // captures conditional execution patterns.
        // Note that this shouldn't be done if the fused macro-op can cross page boundary.
        if phys_pc_end & 4095 != 0 && compiler.model.as_ref().unwrap().can_fuse_cond_op() {
            match op {
                Op::Beq { imm, .. }
                | Op::Bne { imm, .. }
                | Op::Blt { imm, .. }
                | Op::Bge { imm, .. }
                | Op::Bltu { imm, .. }
                | Op::Bgeu { imm, .. } => {
                    if imm == if c { 4 } else { 6 } {
                        if let Ok((next_op, true, bits)) = read_insn(phys_pc_end as usize) {
                            if crate::get_flags().disassemble {
                                eprintln!("{}", next_op.pretty_print(phys_pc_end, bits));
                            }
                            phys_pc_end += 2;
                            compiler.compile_cond_op(&op, c, &next_op, true);
                            if phys_pc_end & 4095 == 0 {
                                compiler.end();
                                break;
                            }
                            continue;
                        }
                    }
                    if imm == if c { 6 } else { 8 } {
                        if let Ok((next_op, false, bits)) = read_insn(phys_pc_end as usize) {
                            if crate::get_flags().disassemble {
                                eprintln!("{}", next_op.pretty_print(phys_pc_end, bits));
                            }
                            phys_pc_end += 4;
                            compiler.compile_cond_op(&op, c, &next_op, false);
                            if phys_pc_end & 4095 == 0 {
                                compiler.end();
                                break;
                            }
                            continue;
                        }
                    }
                }
                _ => (),
            }
        }

        match op {
            Op::Beq { .. }
            | Op::Bne { .. }
            | Op::Blt { .. }
            | Op::Bge { .. }
            | Op::Bltu { .. }
            | Op::Bgeu { .. } => {
                compiler.compile_op(&op, c, bits);
                compiler.end();
                break;
            }
            op => {
                compiler.compile_op(&op, c, bits);
                if op.can_change_control_flow() {
                    compiler.end_unreachable();
                    break;
                }
            }
        }

        // Need to stop when crossing page boundary
        if phys_pc_end & 4095 == 0 {
            compiler.end();
            break;
        }
    }

    // We must not cross page boundary in absolutely any cases.
    assert_eq!(phys_pc & !4095, (phys_pc_end - 1) & !4095, "op crosses page boundary");

    let func_len = compiler.len;
    assert!(func_len <= 256 * 1024);
    let spec_len = compiler.speculative_len;
    let code_fn = code.as_ptr() as usize;
    let nonspec_fn = code_fn + spec_len;
    let map = match prv {
        0 => &mut icache.u_map,
        1 => &mut icache.s_map,
        3 => &mut icache.m_map,
        _ => unreachable!(),
    };
    map.insert(phys_pc, (code_fn, nonspec_fn));

    // Actually commit the space we allocated
    icache.commit(func_len);

    // We use 0 to indicate that a rollover has happened during the translation, and therefore
    // no code should be patched, but the execution should resume from nonspec_fn instead.
    (if rollover { 0 } else { code_fn }, nonspec_fn)
}

extern "C" {
    fn helper_check_interrupt();
}

#[no_mangle]
extern "C" fn find_block(ctx: &mut Context) -> (usize, usize) {
    let pc = ctx.pc;
    let phys_pc = match insn_translate(ctx, pc) {
        Ok(pc) => pc,
        Err(_) => {
            trap(ctx);
            return (helper_check_interrupt as usize, helper_check_interrupt as usize);
        }
    };
    let mut prot = CODE_PROT.lock();
    let mut icache = icache(ctx.hartid);
    let map = match ctx.prv {
        0 => &mut icache.u_map,
        1 => &mut icache.s_map,
        3 => &mut icache.m_map,
        _ => unreachable!(),
    };
    match map.get(&phys_pc).copied() {
        Some(v) => v,
        None => {
            if prot.insert(phys_pc >> 12) {
                trace!(target: "CodeProt", "protecting {:x} to {:x}", phys_pc &! 4095, (phys_pc &! 4095) + 4096);
                for i in 0..crate::core_count() {
                    crate::shared_context(i).protect_code(phys_pc & !4095);
                }
            }
            std::mem::drop(prot);
            translate_code(ctx, &mut icache, ctx.prv, phys_pc)
        }
    }
}

#[no_mangle]
/// Check if an enabled interrupt is pending, and take it if so.
/// If `{Err}` is returned, the running fiber will exit.
pub fn check_interrupt(ctx: &mut Context) -> Result<(), ()> {
    let alarm = ctx.shared.alarm.swap(0, MemOrder::Acquire);

    if alarm & 2 != 0 {
        return Err(());
    }

    {
        let mut guard = ctx.shared.tasks.lock();
        for task in guard.drain(..) {
            task();
        }
    }

    // Find out which interrupts can be taken
    let interrupt_mask = ctx.interrupt_pending();
    // No interrupt pending
    if interrupt_mask == 0 {
        return Ok(());
    }
    // Find the highest priority interrupt
    let pending = 63 - interrupt_mask.leading_zeros() as u64;
    // Interrupts have the highest bit set
    ctx.cause = (1 << 63) | pending;
    ctx.tval = 0;
    trap(ctx);
    Ok(())
}

/// Trigger a trap. pc must be already adjusted properly before calling.
#[no_mangle]
pub fn trap(ctx: &mut Context) {
    if crate::get_flags().prv == 0 {
        eprintln!("unhandled trap {:x}, tval = {:x}", ctx.cause, ctx.tval);
        eprintln!("pc  = {:16x}  ra  = {:16x}", ctx.pc, ctx.registers[1]);
        for i in (2..32).step_by(2) {
            eprintln!(
                "{:-3} = {:16x}  {:-3} = {:16x}",
                riscv::register_name(i as u8),
                ctx.registers[i],
                riscv::register_name((i + 1) as u8),
                ctx.registers[i + 1]
            );
        }
        std::process::exit(1);
    }

    let deleg_reg = if ctx.cause >> 63 != 0 { ctx.mideleg } else { ctx.medeleg };
    let deleg_to_s = ctx.prv != 3 && (deleg_reg >> (ctx.cause & 15)) & 1 != 0;

    if deleg_to_s {
        ctx.scause = ctx.cause;
        ctx.stval = ctx.tval;
        ctx.sepc = ctx.pc;

        // Clear or set SPP bit
        if ctx.prv != 0 {
            ctx.mstatus |= 0x100;
        } else {
            ctx.mstatus &= !0x100;
            // Switch from U-mode to S-mode, clear local cache
            ctx.shared.clear_local_cache();
            ctx.shared.clear_local_icache();
        }

        // Clear of set SPIE bit
        if (ctx.mstatus & 0x2) != 0 {
            ctx.mstatus |= 0x20;
        } else {
            ctx.mstatus &= !0x20;
        }

        // Clear SIE
        ctx.mstatus &= !0x2;

        // Switch to S-mode
        ctx.prv = 1;
        ctx.pc = ctx.stvec;
    } else {
        assert_eq!(crate::get_flags().prv, 3);

        ctx.mcause = ctx.cause;
        ctx.mtval = ctx.tval;
        ctx.mepc = ctx.pc;

        // Set MPP bits
        ctx.mstatus = (ctx.mstatus & !0x1800) | (ctx.prv << 11);
        if ctx.prv != 3 {
            ctx.shared.clear_local_cache();
            ctx.shared.clear_local_icache();
        }

        // Clear of set MPIE bit
        if (ctx.mstatus & 0x8) != 0 {
            ctx.mstatus |= 0x80;
        } else {
            ctx.mstatus &= !0x80;
        }

        // Clear MIE
        ctx.mstatus &= !0x8;

        // Switch to M-mode
        ctx.prv = 3;
        ctx.pc = ctx.mtvec;
    }

    crate::fiber::Fiber::sleep(1)
}

/// Handle a misaligned load/store.
#[no_mangle]
pub fn handle_misalign(ctx: &mut Context, addr: u64) -> Result<(), ()> {
    // Decode the misaligned instruction.
    let (op, compressed) = {
        let bits = unsafe { *(insn_translate(ctx, ctx.pc)? as *mut u16) };
        if bits & 3 == 3 {
            let hi_bits = unsafe { *(insn_translate(ctx, ctx.pc + 2)? as *mut u16) };
            let bits = (hi_bits as u32) << 16 | bits as u32;
            (riscv::decode(bits), false)
        } else {
            (riscv::decode_compressed(bits), true)
        }
    };
    match op {
        Op::Lh { rd, .. } => {
            let mut bytes = [0; 2];
            ctx.copy_from_virt_addr(addr, &mut bytes)?;
            ctx.registers[rd as usize] = i16::from_le_bytes(bytes) as u64;
        }
        Op::Lw { rd, .. } => {
            let mut bytes = [0; 4];
            ctx.copy_from_virt_addr(addr, &mut bytes)?;
            ctx.registers[rd as usize] = i32::from_le_bytes(bytes) as u64;
        }
        Op::Ld { rd, .. } => {
            let mut bytes = [0; 8];
            ctx.copy_from_virt_addr(addr, &mut bytes)?;
            ctx.registers[rd as usize] = u64::from_le_bytes(bytes);
        }
        Op::Lhu { rd, .. } => {
            let mut bytes = [0; 2];
            ctx.copy_from_virt_addr(addr, &mut bytes)?;
            ctx.registers[rd as usize] = u16::from_le_bytes(bytes) as u64;
        }
        Op::Lwu { rd, .. } => {
            let mut bytes = [0; 4];
            ctx.copy_from_virt_addr(addr, &mut bytes)?;
            ctx.registers[rd as usize] = u32::from_le_bytes(bytes) as u64;
        }
        Op::Sh { rs2, .. } => {
            let bytes = (ctx.registers[rs2 as usize] as u16).to_le_bytes();
            ctx.copy_to_virt_addr(addr, &bytes)?;
        }
        Op::Sw { rs2, .. } => {
            let bytes = (ctx.registers[rs2 as usize] as u32).to_le_bytes();
            ctx.copy_to_virt_addr(addr, &bytes)?;
        }
        Op::Sd { rs2, .. } => {
            let bytes = (ctx.registers[rs2 as usize] as u64).to_le_bytes();
            ctx.copy_to_virt_addr(addr, &bytes)?;
        }
        _ => {
            // Otherwise it's misaligned atomic
            ctx.cause = 6;
            ctx.tval = addr;
            return Err(());
        }
    }

    // Advance PC past misaligned instruction.
    ctx.pc += if compressed { 2 } else { 4 };
    ctx.instret += 1;
    ctx.minstret += 1;
    crate::fiber::Fiber::sleep(1);

    Ok(())
}
