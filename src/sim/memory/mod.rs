use crate::emu::interp::Context;
use riscv::mmu::AccessType;

pub trait MemoryModel: Sync {
    /// Whether lock-step execution is required for this model's simulation.
    /// For cycle-level simulation you would want this to be true, but if no cache coherency is
    /// simulated **and** only rough metrics are needed it's okay to set it to false.
    fn require_lockstep(&self) -> bool {
        true
    }

    /// log2 of the cache line size. By default it is 6 (64B). This should at least be 3 (8B) and
    /// at most be 12 (4096B, 1 page).
    fn cache_line_size_log2(&self) -> u32 {
        6
    }

    /// Called when an instruction memory access occurs and the cache line does not reside in the
    /// L0 instruction cache.
    fn instruction_access(&self, ctx: &mut Context, addr: u64) -> Result<u64, ()> {
        let paddr = ctx.translate_vaddr(addr, AccessType::Execute)?;
        ctx.insert_instruction_cache_line(addr, paddr);
        Ok(paddr)
    }

    /// Called when a data memory access occurs and the cache line does not reside in the L0 data
    /// cache.
    fn data_access(&self, ctx: &mut Context, addr: u64, write: bool) -> Result<u64, ()> {
        let paddr =
            ctx.translate_vaddr(addr, if write { AccessType::Write } else { AccessType::Read })?;
        ctx.insert_data_cache_line(addr, paddr, write);
        Ok(paddr)
    }
}

#[derive(Default)]
pub struct AtomicModel;

impl MemoryModel for AtomicModel {
    fn require_lockstep(&self) -> bool {
        false
    }

    fn cache_line_size_log2(&self) -> u32 {
        12
    }
}

#[derive(Default)]
pub struct SimpleModel;

impl MemoryModel for SimpleModel {}
