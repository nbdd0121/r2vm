use super::super::cache_set::CacheSet;
use super::{Asid, PerformanceModel, Statistics, TLB};
use crate::emu::interp::Context;
use parking_lot::Mutex;
use riscv::mmu::{PageWalkResult, PTE_A, PTE_G, PTE_V};
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Clone, Copy)]
struct Entry {
    vpn: u64,
    pte: u64,
    asid: Asid,
}

pub struct SetAssocTLB {
    sets: Box<[Mutex<CacheSet<Entry>>]>,
    idx_bits: u32,
    perf: PerformanceModel,
    parent: Arc<dyn TLB>,
    stats: Arc<Statistics>,
    icache: bool,
}

impl TLB for SetAssocTLB {
    fn parent(&self) -> &dyn TLB {
        &*self.parent
    }

    fn access(&self, ctx: &mut Context, asid: Asid, addr: u64) -> PageWalkResult {
        let vpn = addr >> 12;
        let idx = self.index(vpn);

        fiber::sleep(self.perf.access_latency);

        // Access the TLB
        let mut set = self.sets[idx].lock();
        let (ptr, insert_ptr) = set.find(|entry| {
            if entry.vpn != vpn {
                return false;
            }
            match (entry.asid, asid) {
                (Asid::Global, Asid::Local(_)) => true,
                (Asid::Local(a), Asid::Local(b)) if a == b => true,
                _ => false,
            }
        });

        match ptr {
            Some(v) => return PageWalkResult::from_4k_pte(v.pte),
            None => (),
        }

        let evict = set.remove(insert_ptr);
        drop(set);

        // Handle entry eviction
        if let Some(evict) = evict {
            self.stats.evict.fetch_add(1, Ordering::Relaxed);
            if self.icache {
                ctx.shared.invalidate_icache_virtual_page(evict.vpn << 12);
            } else {
                ctx.shared.invalidate_cache_virtual_page(evict.vpn << 12);
            }
        }

        fiber::sleep(self.perf.miss_penalty_before);
        self.stats.miss.fetch_add(1, Ordering::Relaxed);

        let pte = self.parent.access(ctx, asid, addr).synthesise_4k(addr).pte;
        let mut set = self.sets[idx].lock();

        // Only insert if the entry is valid
        if pte & PTE_V != 0 && pte & PTE_A != 0 {
            // TODO: Consult ReplacementPolicy as sometimes doing this would require invalidation.
            set.insert(
                insert_ptr,
                Entry { vpn, pte, asid: if pte & PTE_G != 0 { Asid::Global } else { asid } },
            )
            .and_then::<(), _>(|_| unreachable!());
        }
        drop(set);

        fiber::sleep(self.perf.miss_penalty_after);

        PageWalkResult::from_4k_pte(pte)
    }

    fn flush_local(&self, asid: Option<Asid>, vaddr: Option<u64>) {
        let vpn = vaddr.map(|x| x >> 12);
        let mut num_flush = 0;
        match vpn {
            None => {
                for set in self.sets.iter() {
                    set.lock().retain(|entry| {
                        let result = match asid {
                            // Wildcard removal
                            None => false,
                            Some(v) => v != entry.asid,
                        };
                        if result {
                            num_flush += 1
                        }
                        result
                    })
                }
            }
            Some(vpn) => {
                let idx = self.index(vpn);
                self.sets[idx].lock().retain(|entry| {
                    match asid {
                        // Wildcard removal
                        None => (),
                        Some(v) => {
                            if v != entry.asid {
                                return true;
                            }
                        }
                    };
                    let result = entry.vpn != vpn;
                    if result {
                        num_flush += 1
                    }
                    result
                })
            }
        }
        self.stats.flush.fetch_add(num_flush, Ordering::Relaxed);
    }
}

impl SetAssocTLB {
    /// Create a new set associative TLB.
    pub fn new(
        parent: Arc<dyn TLB>,
        stats: Arc<Statistics>,
        perf: PerformanceModel,
        icache: bool,
        set: usize,
        assoc: usize,
    ) -> Self {
        let mut sets = Vec::with_capacity(set);
        for _ in 0..set {
            sets.push(Mutex::new(CacheSet::new(assoc)));
        }
        Self {
            sets: sets.into_boxed_slice(),
            idx_bits: 31 - (set as u32).leading_zeros(),
            perf,
            parent,
            stats,
            icache,
        }
    }

    /// Find out which set to use for a given address
    fn index(&self, vpn: u64) -> usize {
        (vpn & ((1 << self.idx_bits) - 1)) as usize
    }
}
