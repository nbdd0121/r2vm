use crate::emu::dbt::DbtCompiler;
use riscv::Op;

mod in_order;
pub use in_order::InOrderModel;

pub trait PipelineModel {
    /// Indicates whether a branch over a single instruction can be treated as conditional executed
    /// operation. This can be true for inaccurate models, or when branching over a single instruction
    /// is optimised in hardware.
    ///
    /// Technically this can be a static method but we need it to be object-safe.
    fn can_fuse_cond_op(&self) -> bool {
        true
    }

    /// Hook to be called when the binary translator starts translating a block.
    fn begin_block(&mut self, _compiler: &mut DbtCompiler, _pc: u64) {}

    /// Hook to be called when the binary translator starts translating an instruction.
    fn before_instruction(&mut self, _compiler: &mut DbtCompiler, _op: &Op, _compressed: bool) {}

    /// Hook to be called after the binary translator generates code for an instruction.
    /// For a branch instruction, this method is only called when the branch is not taken.
    fn after_instruction(&mut self, _compiler: &mut DbtCompiler, _op: &Op, _compressed: bool) {}

    /// Hook to be called when the binary translator generates code for an taken branch instruction.
    fn after_taken_branch(&mut self, _compiler: &mut DbtCompiler, _op: &Op, _compressed: bool) {}
}

/// An atomic execution model, where all instructions except for memory access takes no time to execute.
#[derive(Default)]
pub struct AtomicModel;

impl PipelineModel for AtomicModel {}

/// An simple timing execution model, where each instruction takes 1 cycle to execute.
#[derive(Default)]
pub struct SimpleModel;

impl PipelineModel for SimpleModel {
    fn after_instruction(&mut self, compiler: &mut DbtCompiler, _op: &Op, _compressed: bool) {
        compiler.insert_cycle_count(1);
    }

    fn after_taken_branch(&mut self, compiler: &mut DbtCompiler, _op: &Op, _compressed: bool) {
        compiler.insert_cycle_count(1);
    }
}
