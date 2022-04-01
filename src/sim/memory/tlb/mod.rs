use super::cache::Cache;
use crate::emu::interp::Context;
use riscv::mmu::{check_permission, AccessType, PageWalkResult, PTE_D, PTE_W};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

mod assoc;
pub use assoc::SetAssocTLB;

pub(super) fn prv_of_ctx(ctx: &Context, exec: bool) -> u8 {
    // Respect MPRV
    let mut prv = ctx.prv;
    if prv == 3 && ctx.mstatus & 0x20000 != 0 && !exec {
        prv = (ctx.mstatus >> 11) & 3;
    }

    prv as u8
}

pub(super) fn asid_of_ctx(ctx: &Context, exec: bool) -> Asid {
    let prv = prv_of_ctx(ctx, exec);

    // MMU off
    if (ctx.satp >> 60) == 0 || prv == 3 {
        return Asid::Physical;
    }

    let asid = (ctx.satp >> 44) as u16;
    Asid::Local(if asid != 0 { asid } else { ctx.hartid as u16 })
}

pub(super) fn prv_asid_of_ctx(ctx: &Context, exec: bool) -> (u8, Asid) {
    let prv = prv_of_ctx(ctx, exec);
    let asid = asid_of_ctx(ctx, exec);
    (prv, asid)
}

/// A representation of all ASIDs.
///
/// For requests, TLB should only care about `Local`. For storage, it may need to use `Global`
/// as well. For internal handling purposes, `Physical` is already treated as ASID that always
/// do identity mapping.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Asid {
    Physical,
    Global,
    Local(u16),
}

impl From<u16> for Asid {
    fn from(other: u16) -> Asid {
        Asid::Local(other)
    }
}

/// Basic performance statistics gathered from the TLB.
///
/// Note that total number of accesses is not included. It should rather be calculated from
/// the previous level's miss count.
pub struct Statistics {
    pub miss: AtomicU64,
    pub evict: AtomicU64,
    pub flush: AtomicU64,
}

impl Statistics {
    pub const fn new() -> Self {
        Self { miss: AtomicU64::new(0), evict: AtomicU64::new(0), flush: AtomicU64::new(0) }
    }

    pub fn clear(&self) {
        self.miss.store(0, Ordering::Relaxed);
        self.evict.store(0, Ordering::Relaxed);
        self.flush.store(0, Ordering::Relaxed);
    }
}

/// Basic performance model of a TLB.
pub struct PerformanceModel {
    /// `access_latency` indicates the latency of accessing this TLB. This is the latency
    /// between access request to the TLB and the response from the TLB. This models the
    /// latency of SRAM access and tag check. For L1 TLB, due to the design
    /// of R2VM in which not all memory requests reach memory model, the access latency must be zero
    /// and modelled within the pipeline model instead.
    pub access_latency: usize,

    /// `miss_penalty_before` indicates the number of penalty cyclces if the entry to looked up
    /// does not exist in the TLB. This models the latency of preparing the bus request to
    /// next-level TLB or the page walker.
    pub miss_penalty_before: usize,

    /// `miss_penalty_after` indicates the number of penalty cycles after the response is being
    /// received from the next-level TLB or the page walker.
    pub miss_penalty_after: usize,
}

/// TLB model.
///
/// This model could be useful if you only care about hit rate, or only need roughly
/// accurate simulation. If you are trying to simulate a specific system, it's suggested
/// to supplement a custom implementation of [`MemoryModel`].
///
/// [`MemoryModel`]: super::MemoryModel
pub trait TLB: Send + Sync {
    /// Perform a TLB access at given virtual address.
    ///
    /// The leaf PTE is returned.
    fn access(&self, ctx: &mut Context, asid: Asid, addr: u64) -> PageWalkResult;

    /// Get the next level TLB.
    ///
    /// If this is the last-level, it should refer to a page walker. For a page-walker, calling
    /// this method may panic.
    fn parent(&self) -> &dyn TLB;

    /// Flush this TLB. If any of the argument is `None` it means a wildcard match.
    fn flush_local(&self, asid: Option<Asid>, vaddr: Option<u64>) {
        let _ = (asid, vaddr);
    }

    /// Flush this TLB. If the TLB is chained, do cascade flushing instead.
    fn flush(&self, asid: Option<Asid>, vaddr: Option<u64>) {
        self.flush_local(asid, vaddr);
        self.parent().flush(asid, vaddr);
    }
}

pub struct PageWalkerPerformanceModel {
    pub start_delay: usize,
    pub walk_delay_before: usize,
    pub walk_delay_after: usize,
    pub end_delay: usize,
}

pub struct PageWalker {
    parent: Arc<dyn Cache>,
    perf: PageWalkerPerformanceModel,
}

impl TLB for PageWalker {
    fn parent(&self) -> &dyn TLB {
        unimplemented!()
    }

    fn flush(&self, _asid: Option<Asid>, _vpn: Option<u64>) {}

    fn access(&self, ctx: &mut Context, _asid: Asid, addr: u64) -> PageWalkResult {
        fiber::sleep(self.perf.start_delay);
        let ret = riscv::mmu::walk_page(ctx.satp, addr >> 12, |addr| {
            fiber::sleep(self.perf.walk_delay_before);
            self.parent.access(ctx, addr, false);
            let ret = crate::emu::read_memory(addr as usize);
            fiber::sleep(self.perf.walk_delay_after);
            ret
        });
        fiber::sleep(self.perf.end_delay);
        ret
    }
}

impl dyn TLB {
    pub fn translate(
        &self,
        ctx: &mut Context,
        addr: u64,
        ty: AccessType,
    ) -> Result<(u64, bool), ()> {
        let (prv, asid) = prv_asid_of_ctx(ctx, ty == AccessType::Execute);
        if asid == Asid::Physical {
            return Ok((addr, true));
        }

        let pte = self.access(ctx, asid, addr).synthesise_4k(addr).pte;
        match check_permission(pte, ty, prv, ctx.mstatus) {
            Ok(_) => Ok(((pte >> 10 << 12) | (addr & 4095), pte & PTE_W != 0 && pte & PTE_D != 0)),
            Err(_) => {
                ctx.cause = match ty {
                    AccessType::Execute => 12,
                    AccessType::Read => 13,
                    AccessType::Write => 15,
                };
                ctx.tval = addr;
                Err(())
            }
        }
    }
}

impl PageWalker {
    pub fn new(parent: Arc<dyn Cache>, perf: PageWalkerPerformanceModel) -> Self {
        PageWalker { parent, perf }
    }
}

pub struct TLBModel {
    i_tlbs: Box<[Arc<dyn TLB>]>,
    d_tlbs: Box<[Arc<dyn TLB>]>,
    i_stats: Arc<Statistics>,
    d_stats: Arc<Statistics>,
}

impl TLBModel {
    pub fn example() -> Self {
        let i_stats = Arc::new(Statistics::new());
        let d_stats = Arc::new(Statistics::new());
        let memory = Arc::new(super::cache::MemoryController::new(
            super::cache::MemoryControllerPerformanceModel { read_latency: 0, write_latency: 0 },
        ));
        let walker = Arc::new(PageWalker::new(
            memory,
            PageWalkerPerformanceModel {
                start_delay: 0,
                walk_delay_before: 0,
                walk_delay_after: 0,
                end_delay: 0,
            },
        ));
        let core_count = crate::core_count();
        let mut i_tlbs = Vec::<Arc<dyn TLB>>::with_capacity(core_count);
        let mut d_tlbs = Vec::<Arc<dyn TLB>>::with_capacity(core_count);
        for _ in 0..core_count {
            let i_tlb = Arc::new(SetAssocTLB::new(
                walker.clone(),
                i_stats.clone(),
                PerformanceModel {
                    access_latency: 0,
                    miss_penalty_before: 0,
                    miss_penalty_after: 0,
                },
                true,
                4,
                8,
            ));
            let d_tlb = Arc::new(SetAssocTLB::new(
                walker.clone(),
                d_stats.clone(),
                PerformanceModel {
                    access_latency: 0,
                    miss_penalty_before: 0,
                    miss_penalty_after: 0,
                },
                false,
                4,
                8,
            ));

            i_tlbs.push(i_tlb);
            d_tlbs.push(d_tlb);
        }

        Self { i_tlbs: i_tlbs.into(), d_tlbs: d_tlbs.into(), i_stats, d_stats }
    }
}

impl super::MemoryModel for TLBModel {
    fn cache_line_size_log2(&self) -> u32 {
        12
    }

    fn instruction_access(&self, ctx: &mut Context, addr: u64) -> Result<u64, ()> {
        let out = self.i_tlbs[ctx.hartid as usize].translate(ctx, addr, AccessType::Execute)?.0;

        // TODO: ReplacementPolicy probably should be consulted first.
        ctx.insert_instruction_cache_line(addr, out);
        Ok(out)
    }

    fn data_access(&self, ctx: &mut Context, addr: u64, write: bool) -> Result<u64, ()> {
        let (out, w) = self.i_tlbs[ctx.hartid as usize].translate(
            ctx,
            addr,
            if write { AccessType::Write } else { AccessType::Read },
        )?;

        // TODO: ReplacementPolicy probably should be consulted first.
        ctx.insert_data_cache_line(addr, out, w);
        Ok(out)
    }

    fn before_sfence_vma(
        &self,
        _ctx: &mut Context,
        mask: u64,
        asid: Option<u16>,
        vaddr: Option<u64>,
    ) {
        let asid = asid.map(Into::into);
        for i in 0..crate::core_count() {
            if mask & (1 << i) == 0 {
                continue;
            }
            self.i_tlbs[i].flush(asid, vaddr);
            self.d_tlbs[i].flush(asid, vaddr);
        }
    }

    fn reset_stats(&self) {
        self.i_stats.clear();
        self.d_stats.clear();
    }

    fn print_stats(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        writeln!(writer, "ITLB Miss  {}", self.i_stats.miss.load(Ordering::Relaxed))?;
        writeln!(writer, "ITLB Evict {}", self.i_stats.evict.load(Ordering::Relaxed))?;
        writeln!(writer, "ITLB Flush {}", self.i_stats.flush.load(Ordering::Relaxed))?;
        writeln!(writer)?;
        writeln!(writer, "DTLB Miss  {}", self.d_stats.miss.load(Ordering::Relaxed))?;
        writeln!(writer, "DTLB Evict {}", self.d_stats.evict.load(Ordering::Relaxed))?;
        writeln!(writer, "DTLB Flush {}", self.d_stats.flush.load(Ordering::Relaxed))?;
        writeln!(writer)?;
        Ok(())
    }
}
