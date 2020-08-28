use super::super::cache_set::CacheSet;
use super::{Cache, PerformanceModel, Statistics};
use crate::emu::interp::Context;
use fiber::Mutex;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Clone, Copy)]
struct Entry {
    tag: u64,
}

pub struct SetAssocCache {
    sets: Box<[Mutex<CacheSet<Entry>>]>,
    idx_bits: u32,
    acq_lock: Mutex<()>,
    perf: PerformanceModel,
    parent: Arc<dyn Cache>,
    stats: Arc<Statistics>,
    icache: Option<bool>,
}

impl Cache for SetAssocCache {
    fn access(&self, ctx: &mut Context, addr: u64, write: bool) {
        let _acq_lock = self.acq_lock.lock();
        let tag = addr >> 6;
        let idx = self.index(tag);

        fiber::sleep(self.perf.access_latency);

        // Access the cache
        let mut set = self.sets[idx].lock();
        let (ptr, insert_ptr) = set.find(|entry| entry.tag == tag);

        match ptr {
            Some(_v) => return,
            None => (),
        }

        let evict = set.remove(insert_ptr);
        drop(set);

        // Handle entry eviction
        if let Some(evict) = evict {
            self.stats.evict.fetch_add(1, Ordering::Relaxed);
            if let Some(icache) = self.icache {
                if icache {
                    ctx.shared.invalidate_icache_physical(evict.tag << 6);
                } else {
                    ctx.shared.invalidate_cache_physical(evict.tag << 6);
                }
            }
        }

        fiber::sleep(self.perf.miss_penalty_before);
        self.stats.miss.fetch_add(1, Ordering::Relaxed);

        self.parent.access(ctx, addr, write);

        let mut set = self.sets[idx].lock();
        // TODO: Consult ReplacementPolicy as sometimes doing this would require invalidation.
        set.insert(insert_ptr, Entry { tag }).and_then::<(), _>(|_| unreachable!());
        drop(set);

        fiber::sleep(self.perf.miss_penalty_after);
    }

    fn flush_all(&self) {
        let mut num_flush = 0;
        for set in self.sets.iter() {
            set.lock().retain(|_entry| {
                num_flush += 1;
                false
            })
        }
        self.stats.flush.fetch_add(num_flush, Ordering::Relaxed);
    }
}

impl SetAssocCache {
    /// Create a new set associative cache.
    pub fn new(
        parent: Arc<dyn Cache>,
        stats: Arc<Statistics>,
        perf: PerformanceModel,
        icache: Option<bool>,
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
            acq_lock: Mutex::new(()),
            perf,
            parent,
            stats,
            icache,
        }
    }

    /// Find out which set to use for a given address
    fn index(&self, tag: u64) -> usize {
        (tag & ((1 << self.idx_bits) - 1)) as usize
    }
}
