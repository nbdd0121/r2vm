use super::PipelineModel;
use crate::emu::dbt::DbtCompiler;
use riscv::Op;

/// A simple in-order 5-stage pipeline model.
/// It has a static branch predictor with backward branches predicted taken. A mispredicted branch takes 4 cycles.
/// A load-use dependency would cause a 1-cycle stall.
/// Multiplication and division are iterative and executed in MEM stage.
/// Misaligned uncompressed instruction takes 1 extra cycle to load.
pub struct InOrderModel {
    stall_reg: u8,
}

impl Default for InOrderModel {
    fn default() -> Self {
        InOrderModel { stall_reg: 0 }
    }
}

impl PipelineModel for InOrderModel {
    fn can_fuse_cond_op(&self) -> bool {
        false
    }

    fn begin_block(&mut self, compiler: &mut DbtCompiler, pc: u64) {
        if pc % 4 != 0 {
            compiler.insert_cycle_count(1);
        }
    }

    fn before_instruction(&mut self, compiler: &mut DbtCompiler, op: &Op, _compressed: bool) {
        let (rd, rs1, rs2) = op.get_regs();
        if self.stall_reg != 0 && (self.stall_reg == rs1 || self.stall_reg == rs2) {
            compiler.insert_cycle_count(1);
        }
        self.stall_reg = rd;
    }

    fn after_instruction(&mut self, compiler: &mut DbtCompiler, op: &Op, _compressed: bool) {
        let rd = self.stall_reg;
        self.stall_reg = 0;
        let op_cycle = match op {
            Op::Illegal => 1,
            Op::Lui { .. } | Op::Auipc { .. } => 1,
            Op::Jal { .. } => 1,
            Op::Beq { imm, .. }
            | Op::Bne { imm, .. }
            | Op::Blt { imm, .. }
            | Op::Bge { imm, .. }
            | Op::Bltu { imm, .. }
            | Op::Bgeu { imm, .. } => {
                if *imm >= 0 {
                    // Correctly predicted non-taken branch
                    1
                } else {
                    // Predicted taken, but not taken
                    4
                }
            }
            Op::Lb { .. }
            | Op::Lh { .. }
            | Op::Lw { .. }
            | Op::Ld { .. }
            | Op::Lbu { .. }
            | Op::Lhu { .. }
            | Op::Lwu { .. } => {
                self.stall_reg = rd;
                1
            }
            Op::Jalr { .. } => 4,
            Op::Fence => 1,
            Op::FenceI => 5,
            Op::Ecall | Op::Ebreak => 1,
            Op::Mret | Op::Sret => 5,
            Op::Wfi => 1,
            Op::SfenceVma { .. } => 1,
            Op::Sb { .. } | Op::Sh { .. } | Op::Sw { .. } | Op::Sd { .. } => 1,
            Op::Addi { .. }
            | Op::Slti { .. }
            | Op::Sltiu { .. }
            | Op::Xori { .. }
            | Op::Ori { .. }
            | Op::Andi { .. }
            | Op::Addiw { .. }
            | Op::Slli { .. }
            | Op::Srli { .. }
            | Op::Srai { .. }
            | Op::Slliw { .. }
            | Op::Srliw { .. }
            | Op::Sraiw { .. } => 1,
            Op::Add { .. }
            | Op::Sub { .. }
            | Op::Sll { .. }
            | Op::Slt { .. }
            | Op::Sltu { .. }
            | Op::Xor { .. }
            | Op::Srl { .. }
            | Op::Sra { .. }
            | Op::Or { .. }
            | Op::And { .. }
            | Op::Addw { .. }
            | Op::Subw { .. }
            | Op::Sllw { .. }
            | Op::Srlw { .. }
            | Op::Sraw { .. } => 1,
            Op::Mul { .. } => {
                self.stall_reg = rd;
                11
            }
            Op::Mulh { .. } | Op::Mulhsu { .. } | Op::Mulhu { .. } => {
                self.stall_reg = rd;
                17
            }
            Op::Div { .. } | Op::Divu { .. } | Op::Rem { .. } | Op::Remu { .. } => {
                self.stall_reg = rd;
                65
            }
            Op::Mulw { .. } => {
                self.stall_reg = rd;
                4
            }
            Op::Divw { .. } | Op::Divuw { .. } | Op::Remw { .. } | Op::Remuw { .. } => {
                self.stall_reg = rd;
                33
            }
            Op::Csrrw { .. }
            | Op::Csrrs { .. }
            | Op::Csrrc { .. }
            | Op::Csrrwi { .. }
            | Op::Csrrsi { .. }
            | Op::Csrrci { .. } => {
                self.stall_reg = rd;
                1
            }
            Op::LrW { .. } | Op::LrD { .. } => {
                self.stall_reg = rd;
                1
            }
            Op::ScW { .. }
            | Op::ScD { .. }
            | Op::AmoswapW { .. }
            | Op::AmoswapD { .. }
            | Op::AmoaddW { .. }
            | Op::AmoaddD { .. }
            | Op::AmoxorW { .. }
            | Op::AmoxorD { .. }
            | Op::AmoandW { .. }
            | Op::AmoandD { .. }
            | Op::AmoorW { .. }
            | Op::AmoorD { .. }
            | Op::AmominW { .. }
            | Op::AmominD { .. }
            | Op::AmomaxW { .. }
            | Op::AmomaxD { .. }
            | Op::AmominuW { .. }
            | Op::AmominuD { .. }
            | Op::AmomaxuW { .. }
            | Op::AmomaxuD { .. } => {
                self.stall_reg = rd;
                1
            }
            Op::Flw { .. } | Op::Fld { .. } => 1000,
            Op::Fsw { .. } | Op::Fsd { .. } => 1000,
            Op::FaddS { .. }
            | Op::FsubS { .. }
            | Op::FmulS { .. }
            | Op::FdivS { .. }
            | Op::FsgnjS { .. }
            | Op::FsgnjnS { .. }
            | Op::FsgnjxS { .. }
            | Op::FminS { .. }
            | Op::FmaxS { .. }
            | Op::FaddD { .. }
            | Op::FsubD { .. }
            | Op::FmulD { .. }
            | Op::FdivD { .. }
            | Op::FsgnjD { .. }
            | Op::FsgnjnD { .. }
            | Op::FsgnjxD { .. }
            | Op::FminD { .. }
            | Op::FmaxD { .. } => 1000,
            Op::FsqrtS { .. } | Op::FsqrtD { .. } | Op::FcvtSD { .. } | Op::FcvtDS { .. } => 1000,
            Op::FcvtWS { .. }
            | Op::FcvtWuS { .. }
            | Op::FcvtLS { .. }
            | Op::FcvtLuS { .. }
            | Op::FmvXW { .. }
            | Op::FclassS { .. }
            | Op::FcvtWD { .. }
            | Op::FcvtWuD { .. }
            | Op::FcvtLD { .. }
            | Op::FcvtLuD { .. }
            | Op::FmvXD { .. }
            | Op::FclassD { .. } => 1000,
            Op::FcvtSW { .. }
            | Op::FcvtSWu { .. }
            | Op::FcvtSL { .. }
            | Op::FcvtSLu { .. }
            | Op::FmvWX { .. }
            | Op::FcvtDW { .. }
            | Op::FcvtDWu { .. }
            | Op::FcvtDL { .. }
            | Op::FcvtDLu { .. }
            | Op::FmvDX { .. } => 1000,
            Op::FeqS { .. }
            | Op::FltS { .. }
            | Op::FleS { .. }
            | Op::FeqD { .. }
            | Op::FltD { .. }
            | Op::FleD { .. } => 1000,
            Op::FmaddS { .. }
            | Op::FmsubS { .. }
            | Op::FnmsubS { .. }
            | Op::FnmaddS { .. }
            | Op::FmaddD { .. }
            | Op::FmsubD { .. }
            | Op::FnmsubD { .. }
            | Op::FnmaddD { .. } => 1000,
        };
        compiler.insert_cycle_count(op_cycle);
    }

    fn after_taken_branch(&mut self, compiler: &mut DbtCompiler, op: &Op, _compressed: bool) {
        let imm = match *op {
            Op::Beq { imm, .. }
            | Op::Bne { imm, .. }
            | Op::Blt { imm, .. }
            | Op::Bge { imm, .. }
            | Op::Bltu { imm, .. }
            | Op::Bgeu { imm, .. } => imm,
            _ => unreachable!(),
        };
        compiler.insert_cycle_count(if imm < 0 { 1 } else { 4 })
    }
}
