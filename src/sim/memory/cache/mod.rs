use super::tlb::{PageWalker, PageWalkerPerformanceModel, SetAssocTLB, TLB};
use crate::emu::interp::Context;
use riscv::mmu::AccessType;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

mod assoc;
pub use assoc::SetAssocCache;

/// Basic performance statistics gathered from cache or TLB.
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

/// Basic performance model of a cache.
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

/// Simple cache model.
///
/// This model could be useful if you only care about hit rate, or only need roughly
/// accurate simulation. If you are trying to simulate a specific system, it's suggested
/// to supplement a custom implementation of [`MemoryModel`].
///
/// The simple model has no cache coherency simulated.
///
/// [`MemoryModel`]: super::MemoryModel
pub trait Cache: Send + Sync {
    /// Perform a cache access at given physical address.
    fn access(&self, ctx: &mut Context, addr: u64, write: bool);

    /// Flush all entries from the cache.
    fn flush_all(&self);
}

pub struct MemoryControllerPerformanceModel {
    pub read_latency: usize,
    pub write_latency: usize,
}

/// Simulate a memory controller that spends a fixed number of cycles for an memory access.
pub struct MemoryController {
    perf: MemoryControllerPerformanceModel,
}

impl Cache for MemoryController {
    fn access(&self, _ctx: &mut Context, _addr: u64, write: bool) {
        fiber::sleep(if write { self.perf.write_latency } else { self.perf.read_latency });
    }

    fn flush_all(&self) {}
}

impl MemoryController {
    pub fn new(perf: MemoryControllerPerformanceModel) -> Self {
        Self { perf }
    }
}

pub struct SimpleCacheModel {
    i_tlbs: Box<[Arc<dyn TLB>]>,
    d_tlbs: Box<[Arc<dyn TLB>]>,
    i_caches: Box<[Arc<dyn Cache>]>,
    d_caches: Box<[Arc<dyn Cache>]>,
    i_tlb_stats: Arc<super::tlb::Statistics>,
    d_tlb_stats: Arc<super::tlb::Statistics>,
    i_cache_stats: Arc<Statistics>,
    d_cache_stats: Arc<Statistics>,
    l2_cache_stats: Arc<Statistics>,
}

impl SimpleCacheModel {
    pub fn example() -> Self {
        let i_tlb_stats = Arc::new(super::tlb::Statistics::new());
        let d_tlb_stats = Arc::new(super::tlb::Statistics::new());
        let i_cache_stats = Arc::new(Statistics::new());
        let d_cache_stats = Arc::new(Statistics::new());
        let l2_cache_stats = Arc::new(Statistics::new());

        let memory = Arc::new(MemoryController::new(MemoryControllerPerformanceModel {
            read_latency: 0,
            write_latency: 0,
        }));

        let l2 = Arc::new(SetAssocCache::new(
            memory,
            l2_cache_stats.clone(),
            PerformanceModel { access_latency: 0, miss_penalty_before: 0, miss_penalty_after: 0 },
            None,
            crate::core_count() * 256,
            8,
        ));

        let walker = Arc::new(PageWalker::new(
            l2.clone(),
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
        let mut i_caches = Vec::<Arc<dyn Cache>>::with_capacity(core_count);
        let mut d_caches = Vec::<Arc<dyn Cache>>::with_capacity(core_count);
        for _ in 0..core_count {
            let i_tlb = Arc::new(SetAssocTLB::new(
                walker.clone(),
                i_tlb_stats.clone(),
                super::tlb::PerformanceModel {
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
                d_tlb_stats.clone(),
                super::tlb::PerformanceModel {
                    access_latency: 0,
                    miss_penalty_before: 0,
                    miss_penalty_after: 0,
                },
                false,
                4,
                8,
            ));
            let i_cache = Arc::new(SetAssocCache::new(
                l2.clone(),
                i_cache_stats.clone(),
                PerformanceModel {
                    access_latency: 0,
                    miss_penalty_before: 0,
                    miss_penalty_after: 0,
                },
                Some(true),
                64,
                8,
            ));
            let d_cache = Arc::new(SetAssocCache::new(
                l2.clone(),
                d_cache_stats.clone(),
                PerformanceModel {
                    access_latency: 0,
                    miss_penalty_before: 0,
                    miss_penalty_after: 0,
                },
                Some(false),
                64,
                8,
            ));

            i_tlbs.push(i_tlb);
            d_tlbs.push(d_tlb);
            i_caches.push(i_cache);
            d_caches.push(d_cache);
        }

        Self {
            i_tlbs: i_tlbs.into(),
            d_tlbs: d_tlbs.into(),
            i_caches: i_caches.into(),
            d_caches: d_caches.into(),
            i_tlb_stats,
            d_tlb_stats,
            i_cache_stats,
            d_cache_stats,
            l2_cache_stats,
        }
    }
}

impl super::MemoryModel for SimpleCacheModel {
    fn cache_line_size_log2(&self) -> u32 {
        6
    }

    fn instruction_access(&self, ctx: &mut Context, addr: u64) -> Result<u64, ()> {
        let out = self.i_tlbs[ctx.hartid as usize].translate(ctx, addr, AccessType::Execute)?.0;
        self.i_caches[ctx.hartid as usize].access(ctx, out, false);

        // TODO: ReplacementPolicy probably should be consulted first.
        ctx.insert_instruction_cache_line(addr, out);
        Ok(out)
    }

    fn data_access(&self, ctx: &mut Context, addr: u64, write: bool) -> Result<u64, ()> {
        let out = self.i_tlbs[ctx.hartid as usize]
            .translate(ctx, addr, if write { AccessType::Write } else { AccessType::Read })?
            .0;
        self.d_caches[ctx.hartid as usize].access(ctx, out, write);

        // TODO: ReplacementPolicy probably should be consulted first.
        ctx.insert_data_cache_line(addr, out, write);
        Ok(out)
    }

    fn before_fence_i(&self, _ctx: &mut Context, mask: u64) {
        for i in 0..crate::core_count() {
            if mask & (1 << i) == 0 {
                continue;
            }
            self.i_caches[i].flush_all();
        }
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
            self.i_caches[i].flush_all();
        }
    }

    fn reset_stats(&self) {
        self.i_tlb_stats.clear();
        self.d_tlb_stats.clear();
        self.i_cache_stats.clear();
        self.d_cache_stats.clear();
        self.l2_cache_stats.clear();
    }

    fn print_stats(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        writeln!(writer, "ITLB Miss  {}", self.i_tlb_stats.miss.load(Ordering::Relaxed))?;
        writeln!(writer, "ITLB Evict {}", self.i_tlb_stats.evict.load(Ordering::Relaxed))?;
        writeln!(writer, "ITLB Flush {}", self.i_tlb_stats.flush.load(Ordering::Relaxed))?;
        writeln!(writer)?;
        writeln!(writer, "DTLB Miss  {}", self.d_tlb_stats.miss.load(Ordering::Relaxed))?;
        writeln!(writer, "DTLB Evict {}", self.d_tlb_stats.evict.load(Ordering::Relaxed))?;
        writeln!(writer, "DTLB Flush {}", self.d_tlb_stats.flush.load(Ordering::Relaxed))?;
        writeln!(writer)?;
        writeln!(writer, "I$   Miss  {}", self.i_cache_stats.miss.load(Ordering::Relaxed))?;
        writeln!(writer, "I$   Evict {}", self.i_cache_stats.evict.load(Ordering::Relaxed))?;
        writeln!(writer, "I$   Flush {}", self.i_cache_stats.flush.load(Ordering::Relaxed))?;
        writeln!(writer)?;
        writeln!(writer, "L1   Miss  {}", self.d_cache_stats.miss.load(Ordering::Relaxed))?;
        writeln!(writer, "L1   Evict {}", self.d_cache_stats.evict.load(Ordering::Relaxed))?;
        writeln!(writer, "L1   Flush {}", self.d_cache_stats.flush.load(Ordering::Relaxed))?;
        writeln!(writer)?;
        writeln!(writer, "L2   Miss  {}", self.l2_cache_stats.miss.load(Ordering::Relaxed))?;
        writeln!(writer, "L2   Evict {}", self.l2_cache_stats.evict.load(Ordering::Relaxed))?;
        writeln!(writer, "L2   Flush {}", self.l2_cache_stats.flush.load(Ordering::Relaxed))?;
        Ok(())
    }
}
