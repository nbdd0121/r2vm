use crate::emu::interp::Context;
use riscv::mmu::AccessType;

pub use super::pipeline::{AtomicModel, PipelineModel};

#[inline]
pub fn get_model() -> &'static dyn Model {
    &DefaultModel
}

pub trait Model {
    /// log2 of the cache line size
    fn cache_line_size_log2(&self) -> u32 {
        12
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

    /// Create a pipeline model associated with this model.
    fn pipeline_model(&self) -> Box<dyn PipelineModel>;
}

pub struct DefaultModel;

impl Model for DefaultModel {
    fn pipeline_model(&self) -> Box<dyn PipelineModel> {
        Box::new(AtomicModel::default())
    }
}
