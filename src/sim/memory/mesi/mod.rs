pub mod cache_model;
pub mod coherence;

use cache_model::*;
use coherence::{Capability, Sharer};

use super::cache_set::CacheSet;
use crate::emu::interp::Context;
use crate::util::ILog2;
use fiber::{Mutex, MutexGuard};
use once_cell::sync::Lazy;
use riscv::mmu::AccessType;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const EXCLUSIVE_STATE: bool = true;

pub static L1_STATS: Stat = Stat::new();
pub static I_STATS: Stat = Stat::new();
pub static L2_STATS: Stat = Stat::new();

pub static I_ACCESS: AtomicU64 = AtomicU64::new(0);
pub static D_ACCESS: AtomicU64 = AtomicU64::new(0);

pub static BUS_GET: AtomicU64 = AtomicU64::new(0);
pub static BUS_PUT: AtomicU64 = AtomicU64::new(0);
pub static BUS_INV: AtomicU64 = AtomicU64::new(0);

pub static L1_INSTANCE: Lazy<Box<[L1]>> =
    Lazy::new(|| (0..crate::core_count()).map(|hartid| L1::new(32 * 1024, 8, hartid)).collect());
pub static I_INSTANCE: Lazy<Box<[ICache]>> = Lazy::new(|| {
    (0..crate::core_count()).map(|hartid| ICache::new(32 * 1024, 8, hartid)).collect()
});
pub static L2_INSTANCE: Lazy<Arc<L2>> =
    Lazy::new(|| Arc::new(L2::new(crate::core_count() * 128 * 1024, 8)));

// #region Statistics manipulation and printing
//

pub fn stats_clear() {
    BUS_GET.store(0, Ordering::Relaxed);
    BUS_PUT.store(0, Ordering::Relaxed);
    BUS_INV.store(0, Ordering::Relaxed);
}

pub fn stats_print(write: &mut dyn std::io::Write) -> std::io::Result<()> {
    writeln!(write, "Get = {}", BUS_GET.load(Ordering::Relaxed))?;
    writeln!(write, "Put = {}", BUS_PUT.load(Ordering::Relaxed))?;
    writeln!(write, "Inv = {}", BUS_INV.load(Ordering::Relaxed))?;
    Ok(())
}

//
// #region

pub struct Stat {
    pub miss: AtomicU64,
    pub evict: AtomicU64,
    pub flush: AtomicU64,
}

impl Stat {
    pub const fn new() -> Stat {
        Stat { miss: AtomicU64::new(0), evict: AtomicU64::new(0), flush: AtomicU64::new(0) }
    }

    pub fn clear(&self) {
        self.miss.store(0, Ordering::Relaxed);
        self.evict.store(0, Ordering::Relaxed);
        self.flush.store(0, Ordering::Relaxed);
    }
}

// #region Protocol
//

struct PutMsg {
    addr: u64,
    /// Permission to transit into
    to: Capability,
}

//
// #endregion

#[derive(Clone, Copy)]
struct Entry {
    tag: u64,
    state: Capability,
    dirty: bool,
}

pub struct L1 {
    acq_lock: Mutex<()>,
    sets: Mutex<Box<[CacheSet<Entry>]>>,
    idx_bits: u32,
    hartid: usize,
}

impl L1 {
    pub fn access(
        &self,
        ctx: &mut Context,
        vaddr: u64,
        addr: u64,
        write: bool,
        ty: AccessType,
    ) -> Result<u64, ()> {
        assert_ne!(ty, AccessType::Execute);

        let idx = self.index(addr);
        D_ACCESS.fetch_add(1, Ordering::Relaxed);
        let _acq_lock_guard = self.acq_lock.lock();

        let mut lock = self.sets.lock();

        // Find a matching cache line if any
        let (ptr, insert_ptr) = lock[idx].find(|entry| entry.tag == addr & !63);
        // Cache hit only if we got right permission
        if let Some(v) = ptr {
            if ty != AccessType::Write {
                ctx.insert_data_cache_line(vaddr, addr, write);
                return Ok(addr);
            }
            match v.state {
                Capability::Write => {
                    v.dirty = true;
                    ctx.insert_data_cache_line(vaddr, addr, write);
                    return Ok(addr);
                }
                Capability::Read => (),
                Capability::None => unreachable!(),
            }
        }

        // Pretend that we are waiting for L2.
        L1_STATS.miss.fetch_add(1, Ordering::Relaxed);
        std::mem::drop(lock);

        // Simulate latency sending "Get" message to L2.
        fiber::sleep(L1_GET_DELAY + L1_L2_NET_DELAY);
        let requested = if ty == AccessType::Write { Capability::Write } else { Capability::Read };
        let (granted, acq_lock) =
            L2_INSTANCE.access_lock(Some(ctx.hartid as usize), addr, requested);
        assert!(granted >= requested);

        let mut lock = self.sets.lock();
        let set = &mut lock[idx];

        // eprintln!("{}: Insert {:x} to {:x}[{:x}]", crate::event_loop().cycle(), addr, idx, set.insert_ptr);
        let evict = set.insert(
            insert_ptr,
            Entry { tag: addr & !63, state: granted, dirty: ty == AccessType::Write },
        );
        std::mem::drop(lock);

        if let Some(evict) = evict {
            if evict.tag != addr & !63 {
                L1_STATS.evict.fetch_add(1, Ordering::Relaxed);
                ctx.shared.invalidate_cache_physical(evict.tag);
                L2_INSTANCE
                    .put(ctx.hartid as usize, &PutMsg { addr: evict.tag, to: Capability::None });
            }
        }
        ctx.insert_data_cache_line(vaddr, addr, write);
        std::mem::drop(acq_lock);

        // Simulate latency receiving "Get-Ack" message from L2.
        // And a cycle of processing delay.
        fiber::sleep(L1_L2_NET_DELAY + L1_GET_ACK_DELAY);

        Ok(addr)
    }
}

impl L1 {
    pub fn new(size: usize, assoc: usize, hartid: usize) -> Self {
        let set = size / 64 / assoc;
        let mut sets = Vec::with_capacity(set);
        for _ in 0..set {
            sets.push(CacheSet::new(assoc));
        }
        Self {
            acq_lock: Mutex::new(()),
            sets: Mutex::new(sets.into_boxed_slice()),
            idx_bits: 31 - (set as u32).leading_zeros(),
            hartid,
        }
    }

    /// Find out which set to use for a given address
    fn index(&self, addr: u64) -> usize {
        ((addr >> 6) & ((1 << self.idx_bits) - 1)) as usize
    }

    /// Process invalidation probe.
    /// `line` is the starting address of the cache line.
    fn invalidate(&self, line: u64, cap: Capability) {
        BUS_INV.fetch_add(1, Ordering::Relaxed);
        let idx = self.index(line);
        self.sets.lock()[idx].retain(|entry| {
            if entry.tag != line {
                return true;
            }
            if cap == Capability::None {
                false
            } else {
                entry.state = entry.state.min(cap);
                true
            }
        });

        let ctx = crate::shared_context(self.hartid);
        ctx.invalidate_cache_physical(line);

        // eprintln!("invalidate {:x}", line)
    }
}

/// An L1 instruction cache, we don't need acqurie lock here, as I$ has no backward invalidation.
pub struct ICache {
    sets: Mutex<Box<[CacheSet<u64>]>>,
    idx_bits: usize,
    hartid: usize,
}

impl ICache {
    /// Create a new ICache. `size` should be specified in bytes.
    pub fn new(size: usize, assoc: usize, hartid: usize) -> Self {
        let num_sets = size / 64 / assoc;
        let sets: Vec<_> = (0..num_sets).map(|_| CacheSet::new(assoc)).collect();
        Self { sets: Mutex::new(sets.into_boxed_slice()), idx_bits: num_sets.log2(), hartid }
    }

    /// Find out which set to use for a given address
    fn index(&self, addr: u64) -> usize {
        (addr >> 6) as usize & ((1 << self.idx_bits) - 1)
    }

    /// Remove all entries in the cache. This should be invoked when `fence.i` or `sfence.vma`
    /// is executed.
    pub fn flush_all(&self) {
        I_STATS.evict.fetch_add(1, Ordering::Relaxed);
        for set in self.sets.lock().iter_mut() {
            set.retain(|_| false);
        }
    }

    fn access(&self, ctx: &mut Context, addr: u64, _: AccessType) -> Result<u64, ()> {
        I_ACCESS.fetch_add(1, Ordering::Relaxed);

        let mut sets = self.sets.lock();
        let set = &mut sets[self.index(addr)];

        // Find a matching cache line if any
        let (ptr, insert_ptr) = set.find(|entry| *entry == addr >> 6);
        if ptr.is_some() {
            return Ok(addr);
        }

        // Pretend that we are waiting for L2.
        I_STATS.miss.fetch_add(1, Ordering::Relaxed);

        // Simulate latency sending "Get" message to L2.
        fiber::sleep(L1_GET_DELAY + L1_L2_NET_DELAY);
        L2_INSTANCE.access(None, addr, Capability::None);
        // Simulate latency receiving "Get-Ack" message from L2.
        // And a cycle of processing delay.
        fiber::sleep(L1_L2_NET_DELAY + L1_GET_ACK_DELAY);

        let evict = set.insert(insert_ptr, addr >> 6);
        if let Some(evict) = evict {
            I_STATS.evict.fetch_add(1, Ordering::Relaxed);
            ctx.shared.invalidate_icache_physical(evict << 6);
        }

        Ok(addr)
    }
}

#[derive(Clone, Copy)]
struct L2Entry {
    /// Physical tag
    tag: u64,
    /// Whether this entry is writable
    owned: bool,
    /// The L1 currently owning/sharing this block
    mask: Sharer,
}

pub struct L2 {
    acq_lock: Mutex<()>,
    sets: Mutex<Box<[CacheSet<L2Entry>]>>,
    idx_bits: u32,
}

impl L2 {
    pub fn new(size: usize, assoc: usize) -> L2 {
        let set = size / 64 / assoc;
        let mut sets = Vec::with_capacity(set);
        for _ in 0..set {
            sets.push(CacheSet::new(assoc));
        }
        Self {
            acq_lock: Mutex::new(()),
            sets: Mutex::new(sets.into_boxed_slice()),
            idx_bits: 31 - (set as u32).leading_zeros(),
        }
    }

    /// Find out which set to use for a given address
    fn index(&self, addr: u64) -> usize {
        ((addr >> 6) & ((1 << self.idx_bits) - 1)) as usize
    }

    /// Send invalidation message to L1 caches.
    fn do_inv(&self, entry: &L2Entry, cap: Capability) {
        trace!("{}: INV {}: {:x} {:?}", crate::event_loop().cycle(), entry.mask, entry.tag, cap);

        // Simulate net latency for sending "Inv" message.
        // Note that this is vital!!! Our net does not allow reordering of
        // "Inv" and "Get-Ack" message, otherwise the result might be wrong.
        fiber::sleep(L2_INV_DELAY + L1_L2_NET_DELAY);

        for i in 0..crate::core_count() {
            if entry.mask.test(i) {
                L1_INSTANCE[i].invalidate(entry.tag, cap);
            }
        }

        // Simulate process latency of "Inv" and net latency of "Inv-Ack".
        fiber::sleep(L1_INV_DELAY + L1_L2_NET_DELAY + L2_INV_ACK_DELAY);
    }

    /// Voluntary returning a granted permission from master to slave.
    /// We does not simulate the latency of put here, because:
    /// 1. It is much more complex to do so
    /// 2. Put is usually only issued when the result of Get would overflow
    ///    the cache thus eviction is needed.
    fn put(&self, client: usize, msg: &PutMsg) {
        trace!(
            "{}: PUT {:4b}: {:x} {:?}",
            crate::event_loop().cycle(),
            1 << client,
            msg.addr,
            msg.to
        );
        BUS_PUT.fetch_add(1, Ordering::Relaxed);

        let mut lock = self.sets.lock();
        let set = &mut lock[self.index(msg.addr)];
        let mut accessed = false;
        set.find(|entry| {
            if entry.tag != msg.addr {
                return false;
            }
            accessed = true;
            if msg.to == Capability::None {
                entry.mask.clear(client);
            } else {
                // If request downgrade to Shared, then it must be owned!
                assert_eq!(msg.to, Capability::Read);
                assert!(entry.owned);
                entry.owned = false;
            }
            true
        });
        assert!(accessed);
    }

    /// `req` represents the target state that the requester would like to transit to.
    /// If the requester is not caching, it can be set to `Capability::None`.
    ///
    /// # Locking
    /// To maintain correctness and order, the acquire lock of this cache must be held while
    /// calling this function. Failure to do so will not cause undefined behaviour, but will
    /// cause missed invalidation and correctness issue.
    ///
    /// The lock is not acquired within this function because accessing the L2 and placing the
    /// granted cache line into L1 must be atomic. Failure to do so would cause L2 to invalidate
    /// an entry before it is inserted into L1 properly.
    pub fn access_raw(&self, client: Option<usize>, addr: u64, req: Capability) -> Capability {
        BUS_GET.fetch_add(1, Ordering::Relaxed);
        let idx = self.index(addr);

        trace!("{}: GET {:?}: {:x} {:?}", crate::event_loop().cycle(), client, addr, req);
        let mut lock = self.sets.lock();
        let set = &mut lock[idx];

        // Processing delay
        fiber::sleep(L2_GET_DELAY);

        let (ptr, insert_ptr) = set.find(|entry| entry.tag == addr & !63);

        let v = match ptr {
            None => {
                // L2 miss, fetch from main memory.
                L2_STATS.miss.fetch_add(1, Ordering::Relaxed);
                fiber::sleep(MEM_DELAY);

                let evict = set.insert(
                    insert_ptr,
                    L2Entry {
                        tag: addr & !63,
                        owned: if EXCLUSIVE_STATE { true } else { req == Capability::Write },
                        mask: {
                            let mut sharer = Sharer::new();
                            if let Some(v) = client {
                                sharer.set(v)
                            }
                            sharer
                        },
                    },
                );
                if let Some(evict) = evict {
                    L2_STATS.evict.fetch_add(1, Ordering::Relaxed);
                    self.do_inv(&evict, Capability::None);
                    // Write back to main memory. This can be async, so no need to worry about latency sim
                }

                if EXCLUSIVE_STATE { return Capability::Write } else { return req }
            }
            Some(v) => v,
        };

        // Unowned
        if v.mask.empty() {
            if EXCLUSIVE_STATE {
                v.owned = true;
                v.mask.reset();
                if let Some(client) = client {
                    v.mask.set(client)
                }
                return Capability::Write;
            } else {
                v.owned = req == Capability::Write;
                v.mask.reset();
                if let Some(client) = client {
                    v.mask.set(client)
                }
                return req;
            }
        }

        // Move into owned state
        if req == Capability::Write {
            v.mask.clear(client.unwrap());
            self.do_inv(v, Capability::None);
            v.mask.set(client.unwrap());
            v.owned = true;
            return Capability::Write;
        }

        // Downgrade M to S
        if v.owned {
            self.do_inv(v, Capability::Read);
            v.owned = false;
            if let Some(client) = client {
                v.mask.set(client)
            }
            return Capability::Read;
        }

        // Already in S
        if let Some(client) = client {
            v.mask.set(client)
        }
        Capability::Read
    }

    pub fn access(&self, client: Option<usize>, addr: u64, req: Capability) -> Capability {
        let _acq_lock_guard = self.acq_lock.lock();
        self.access_raw(client, addr, req)
    }

    pub fn access_lock(
        &self,
        client: Option<usize>,
        addr: u64,
        req: Capability,
    ) -> (Capability, MutexGuard<()>) {
        let acq_lock_guard = self.acq_lock.lock();
        (self.access_raw(client, addr, req), acq_lock_guard)
    }
}

impl super::cache::Cache for L2 {
    fn access(&self, _ctx: &mut Context, addr: u64, write: bool) {
        assert!(!write);
        self.access(None, addr, Capability::None);
    }

    fn flush_all(&self) {}
}

// #region MESIModel
//

pub struct MESIModel {
    i_tlbs: Box<[Arc<dyn super::tlb::TLB>]>,
    d_tlbs: Box<[Arc<dyn super::tlb::TLB>]>,
    i_tlb_stats: Arc<super::tlb::Statistics>,
    d_tlb_stats: Arc<super::tlb::Statistics>,
}

impl MESIModel {
    pub fn default() -> Self {
        let i_tlb_stats = Arc::new(super::tlb::Statistics::new());
        let d_tlb_stats = Arc::new(super::tlb::Statistics::new());

        let walker = Arc::new(super::tlb::PageWalker::new(
            L2_INSTANCE.clone(),
            super::tlb::PageWalkerPerformanceModel {
                start_delay: TLB_WALK_START_DELAY,
                walk_delay_before: TLB_GET_DELAY + L1_L2_NET_DELAY,
                walk_delay_after: L1_L2_NET_DELAY + TLB_GET_ACK_DELAY,
                end_delay: TLB_WALK_END_DELAY,
            },
        ));

        let core_count = crate::core_count();
        let mut i_tlbs = Vec::<Arc<dyn super::tlb::TLB>>::with_capacity(core_count);
        let mut d_tlbs = Vec::<Arc<dyn super::tlb::TLB>>::with_capacity(core_count);
        for _ in 0..core_count {
            let i_tlb = Arc::new(super::tlb::SetAssocTLB::new(
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
            let d_tlb = Arc::new(super::tlb::SetAssocTLB::new(
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

            i_tlbs.push(i_tlb);
            d_tlbs.push(d_tlb);
        }

        Self { i_tlbs: i_tlbs.into(), d_tlbs: d_tlbs.into(), i_tlb_stats, d_tlb_stats }
    }
}

impl super::MemoryModel for MESIModel {
    /// Called when an instruction memory access occurs and the cache line does not reside in the
    /// L0 instruction cache.
    fn instruction_access(&self, ctx: &mut Context, addr: u64) -> Result<u64, ()> {
        let out = self.i_tlbs[ctx.hartid as usize].translate(ctx, addr, AccessType::Execute)?.0;
        I_INSTANCE[ctx.hartid as usize].access(ctx, out, AccessType::Execute)?;
        ctx.insert_instruction_cache_line(addr, out);
        Ok(out)
    }

    /// Called when a data memory access occurs and the cache line does not reside in the L0 data
    /// cache.
    fn data_access(&self, ctx: &mut Context, addr: u64, write: bool) -> Result<u64, ()> {
        let out = self.d_tlbs[ctx.hartid as usize]
            .translate(ctx, addr, if write { AccessType::Write } else { AccessType::Read })?
            .0;
        L1_INSTANCE[ctx.hartid as usize].access(
            ctx,
            addr,
            out,
            write,
            if write { AccessType::Write } else { AccessType::Read },
        )?;
        Ok(out)
    }

    /// Hook to execute before a fence.i instruction. `mask` provides bit mask of harts for remote invalidation.
    fn before_fence_i(&self, _ctx: &mut Context, mask: u64) {
        for i in 0..crate::core_count() {
            if mask & (1 << i) == 0 {
                continue;
            }
            I_INSTANCE[i].flush_all();
        }
    }

    // Hook to execute before a sfence.vma instruction. `mask` provides bit mask of harts for remote invalidation.
    fn before_sfence_vma(
        &self,
        _ctx: &mut Context,
        mask: u64,
        asid: Option<u16>,
        vaddr: Option<u64>,
    ) {
        let asid = asid.map(Into::into);
        let vpn = vaddr.map(|x| x >> 12);
        for i in 0..crate::core_count() {
            if mask & (1 << i) == 0 {
                continue;
            }
            I_INSTANCE[i].flush_all();
            self.i_tlbs[i].flush(asid, vpn);
            self.d_tlbs[i].flush(asid, vpn);
        }
    }

    /// Reset all statistics
    fn reset_stats(&self) {
        self.i_tlb_stats.clear();
        self.d_tlb_stats.clear();
        I_ACCESS.store(0, Ordering::Relaxed);
        D_ACCESS.store(0, Ordering::Relaxed);
        L1_STATS.clear();
        I_STATS.clear();
        L2_STATS.clear();
        stats_clear();
    }

    /// Print relevant statistics counter
    fn print_stats(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        writeln!(writer, "ITLB Miss  {}", self.i_tlb_stats.miss.load(Ordering::Relaxed))?;
        writeln!(writer, "ITLB Evict {}", self.i_tlb_stats.evict.load(Ordering::Relaxed))?;
        writeln!(writer, "ITLB Flush {}", self.i_tlb_stats.flush.load(Ordering::Relaxed))?;
        writeln!(writer)?;
        writeln!(writer, "DTLB Miss  {}", self.d_tlb_stats.miss.load(Ordering::Relaxed))?;
        writeln!(writer, "DTLB Evict {}", self.d_tlb_stats.evict.load(Ordering::Relaxed))?;
        writeln!(writer, "DTLB Flush {}", self.d_tlb_stats.flush.load(Ordering::Relaxed))?;
        writeln!(writer)?;
        let l1 = &L1_STATS;
        writeln!(writer, "L1   SAcc  {}", D_ACCESS.load(Ordering::Relaxed))?;
        writeln!(writer, "L1   Miss  {}", l1.miss.load(Ordering::Relaxed))?;
        writeln!(writer, "L1   Evict {}", l1.evict.load(Ordering::Relaxed))?;
        writeln!(writer, "L1   Flush {}", l1.flush.load(Ordering::Relaxed))?;
        writeln!(writer)?;
        let i = &I_STATS;
        writeln!(writer, "I$   SAcc  {}", I_ACCESS.load(Ordering::Relaxed))?;
        writeln!(writer, "I$   Miss  {}", i.miss.load(Ordering::Relaxed))?;
        writeln!(writer, "I$   Evict {}", i.evict.load(Ordering::Relaxed))?;
        writeln!(writer, "I$   Flush {}", i.flush.load(Ordering::Relaxed))?;
        writeln!(writer)?;
        let l2 = &L2_STATS;
        writeln!(writer, "L2   Miss  {}", l2.miss.load(Ordering::Relaxed))?;
        writeln!(writer, "L2   Evict {}", l2.evict.load(Ordering::Relaxed))?;
        writeln!(writer, "L2   Flush {}", l2.flush.load(Ordering::Relaxed))?;
        stats_print(writer)?;
        Ok(())
    }
}

//
// #endregion
