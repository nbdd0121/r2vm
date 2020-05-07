use super::Csr;

/// Ordering semantics for atomics.
#[derive(Clone, Copy, PartialEq)]
pub enum Ordering {
    Relaxed = 0,
    Release = 1,
    Acquire = 2,
    SeqCst = 3,
}

use core::sync::atomic::Ordering as MemOrder;
impl From<Ordering> for MemOrder {
    fn from(ord: Ordering) -> Self {
        match ord {
            Ordering::Relaxed => MemOrder::Relaxed,
            Ordering::Acquire => MemOrder::Acquire,
            Ordering::Release => MemOrder::Release,
            Ordering::SeqCst => MemOrder::SeqCst,
        }
    }
}

/// This includes all supported RISC-V ops.
/// Ops are sorted in the following order
/// * Canonical order of extension
/// * Increasing base opcode number
/// * Increasing funct3 and then funct7, or their ordering in RISC-V spec
#[rustfmt::skip]
#[derive(Clone, Copy, PartialEq)]
pub enum Op {
    Illegal,
    /* RV64I */
    /* Base Opcode = LOAD */
    Lb { rd: u8, rs1: u8, imm: i32 },
    Lh { rd: u8, rs1: u8, imm: i32 },
    Lw { rd: u8, rs1: u8, imm: i32 },
    Ld { rd: u8, rs1: u8, imm: i32 },
    Lbu { rd: u8, rs1: u8, imm: i32 },
    Lhu { rd: u8, rs1: u8, imm: i32 },
    Lwu { rd: u8, rs1: u8, imm: i32 },
    /* Base Opcode = LOAD-FP */
    /* Base Opcode = MISC-MEM */
    Fence,
    FenceI,
    /* Base Opcode = OP-IMM */
    Addi { rd: u8, rs1: u8, imm: i32 },
    Slli { rd: u8, rs1: u8, imm: i32 },
    Slti { rd: u8, rs1: u8, imm: i32 },
    Sltiu { rd: u8, rs1: u8, imm: i32 },
    Xori { rd: u8, rs1: u8, imm: i32 },
    Srli { rd: u8, rs1: u8, imm: i32 },
    Srai { rd: u8, rs1: u8, imm: i32 },
    Ori { rd: u8, rs1: u8, imm: i32 },
    Andi { rd: u8, rs1: u8, imm: i32 },
    /* Base Opcode = AUIPC */
    Auipc { rd: u8, imm: i32 },
    /* Base Opcode = OP-IMM-32 */
    Addiw { rd: u8, rs1: u8, imm: i32 },
    Slliw { rd: u8, rs1: u8, imm: i32 },
    Srliw { rd: u8, rs1: u8, imm: i32 },
    Sraiw { rd: u8, rs1: u8, imm: i32 },
    /* Base Opcode = STORE */
    Sb { rs1: u8, rs2: u8, imm: i32 },
    Sh { rs1: u8, rs2: u8, imm: i32 },
    Sw { rs1: u8, rs2: u8, imm: i32 },
    Sd { rs1: u8, rs2: u8, imm: i32 },
    /* Base Opcode = STORE-FP */
    /* Base Opcode = AMO */
    /* Base Opcode = OP */
    Add { rd: u8, rs1: u8, rs2: u8 },
    Sub { rd: u8, rs1: u8, rs2: u8 },
    Sll { rd: u8, rs1: u8, rs2: u8 },
    Slt { rd: u8, rs1: u8, rs2: u8 },
    Sltu { rd: u8, rs1: u8, rs2: u8 },
    Xor { rd: u8, rs1: u8, rs2: u8 },
    Srl { rd: u8, rs1: u8, rs2: u8 },
    Sra { rd: u8, rs1: u8, rs2: u8 },
    Or { rd: u8, rs1: u8, rs2: u8 },
    And { rd: u8, rs1: u8, rs2: u8 },
    /* Base Opcode = LUI */
    Lui { rd: u8, imm: i32 },
    /* Base Opcode = OP-32 */
    Addw { rd: u8, rs1: u8, rs2: u8 },
    Subw { rd: u8, rs1: u8, rs2: u8 },
    Sllw { rd: u8, rs1: u8, rs2: u8 },
    Srlw { rd: u8, rs1: u8, rs2: u8 },
    Sraw { rd: u8, rs1: u8, rs2: u8 },
    /* Base Opcode = MADD */
    /* Base Opcode = MSUB */
    /* Base Opcode = NMSUB */
    /* Base Opcode = NMADD */
    /* Base Opcode = OP-FP */
    /* Base Opcode = BRANCH */
    // Note: the immediate here is the offset from `pc after - 4`, instead of
    // `pc before`. This will make `step` function simpler by not needing a
    // op length as input.
    // Similar reasoning applies to JAL as well.
    Beq { rs1: u8, rs2: u8, imm: i32 },
    Bne { rs1: u8, rs2: u8, imm: i32 },
    Blt { rs1: u8, rs2: u8, imm: i32 },
    Bge { rs1: u8, rs2: u8, imm: i32 },
    Bltu { rs1: u8, rs2: u8, imm: i32 },
    Bgeu { rs1: u8, rs2: u8, imm: i32 },
    /* Base Opcode = JALR */
    Jalr { rd: u8, rs1: u8, imm: i32 },
    /* Base Opcode = JAL */
    Jal { rd: u8, imm: i32 },
    /* Base Opcode = SYSTEM */
    Ecall,
    Ebreak,
    Csrrw { rd: u8, rs1: u8, csr: Csr },
    Csrrs { rd: u8, rs1: u8, csr: Csr },
    Csrrc { rd: u8, rs1: u8, csr: Csr },
    Csrrwi { rd: u8, imm: u8, csr: Csr },
    Csrrsi { rd: u8, imm: u8, csr: Csr },
    Csrrci { rd: u8, imm: u8, csr: Csr },

    /* M extension */
    /* Base Opcode = OP */
    Mul { rd: u8, rs1: u8, rs2: u8 },
    Mulh { rd: u8, rs1: u8, rs2: u8 },
    Mulhsu { rd: u8, rs1: u8, rs2: u8 },
    Mulhu { rd: u8, rs1: u8, rs2: u8 },
    Div { rd: u8, rs1: u8, rs2: u8 },
    Divu { rd: u8, rs1: u8, rs2: u8 },
    Rem { rd: u8, rs1: u8, rs2: u8 },
    Remu { rd: u8, rs1: u8, rs2: u8 },
    /* Base Opcode = OP-32 */
    Mulw { rd: u8, rs1: u8, rs2: u8 },
    Divw { rd: u8, rs1: u8, rs2: u8 },
    Divuw { rd: u8, rs1: u8, rs2: u8 },
    Remw { rd: u8, rs1: u8, rs2: u8 },
    Remuw { rd: u8, rs1: u8, rs2: u8 },

    /* A extension */
    /* Base Opcode = AMO */
    LrW { rd: u8, rs1: u8, aqrl: Ordering },
    LrD { rd: u8, rs1: u8, aqrl: Ordering },
    ScW { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    ScD { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmoswapW { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmoswapD { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmoaddW { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmoaddD { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmoxorW { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmoxorD { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmoandW { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmoandD { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmoorW { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmoorD { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmominW { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmominD { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmomaxW { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmomaxD { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmominuW { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmominuD { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmomaxuW { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },
    AmomaxuD { rd: u8, rs1: u8, rs2: u8, aqrl: Ordering },

    /* F extension */
    /* Base Opcode = LOAD-FP */
    Flw { frd: u8, rs1: u8, imm: i32 },
    /* Base Opcode = STORE-FP */
    Fsw { rs1: u8, frs2: u8, imm: i32 },
    /* Base Opcode = OP-FP */
    FaddS { frd: u8, frs1: u8, frs2: u8, rm: u8 },
    FsubS { frd: u8, frs1: u8, frs2: u8, rm: u8 },
    FmulS { frd: u8, frs1: u8, frs2: u8, rm: u8 },
    FdivS { frd: u8, frs1: u8, frs2: u8, rm: u8 },
    FsqrtS { frd: u8, frs1: u8, rm: u8 },
    FsgnjS { frd: u8, frs1: u8, frs2: u8 },
    FsgnjnS { frd: u8, frs1: u8, frs2: u8 },
    FsgnjxS { frd: u8, frs1: u8, frs2: u8 },
    FminS { frd: u8, frs1: u8, frs2: u8 },
    FmaxS { frd: u8, frs1: u8, frs2: u8 },
    FcvtWS { rd: u8, frs1: u8, rm: u8 },
    FcvtWuS { rd: u8, frs1: u8, rm: u8 },
    FcvtLS { rd: u8, frs1: u8, rm: u8 },
    FcvtLuS { rd: u8, frs1: u8, rm: u8 },
    FmvXW { rd: u8, frs1: u8 },
    FclassS { rd: u8, frs1: u8 },
    FeqS { rd: u8, frs1: u8, frs2: u8 },
    FltS { rd: u8, frs1: u8, frs2: u8 },
    FleS { rd: u8, frs1: u8, frs2: u8 },
    FcvtSW { frd: u8, rs1: u8, rm: u8 },
    FcvtSWu { frd: u8, rs1: u8, rm: u8 },
    FcvtSL { frd: u8, rs1: u8, rm: u8 },
    FcvtSLu { frd: u8, rs1: u8, rm: u8 },
    FmvWX { frd: u8, rs1: u8 },
    /* Base Opcode = MADD */
    FmaddS { frd: u8, frs1: u8, frs2: u8, frs3: u8, rm: u8 },
    /* Base Opcode = MSUB */
    FmsubS { frd: u8, frs1: u8, frs2: u8, frs3: u8, rm: u8 },
    /* Base Opcode = NMSUB */
    FnmsubS { frd: u8, frs1: u8, frs2: u8, frs3: u8, rm: u8 },
    /* Base Opcode = NMADD */
    FnmaddS { frd: u8, frs1: u8, frs2: u8, frs3: u8, rm: u8 },

    /* D extension */
    /* Base Opcode = LOAD-FP */
    Fld { frd: u8, rs1: u8, imm: i32 },
    /* Base Opcode = STORE-FP */
    Fsd { rs1: u8, frs2: u8, imm: i32 },
    /* Base Opcode = OP-FP */
    FaddD { frd: u8, frs1: u8, frs2: u8, rm: u8 },
    FsubD { frd: u8, frs1: u8, frs2: u8, rm: u8 },
    FmulD { frd: u8, frs1: u8, frs2: u8, rm: u8 },
    FdivD { frd: u8, frs1: u8, frs2: u8, rm: u8 },
    FsqrtD { frd: u8, frs1: u8, rm: u8 },
    FsgnjD { frd: u8, frs1: u8, frs2: u8 },
    FsgnjnD { frd: u8, frs1: u8, frs2: u8 },
    FsgnjxD { frd: u8, frs1: u8, frs2: u8 },
    FminD { frd: u8, frs1: u8, frs2: u8 },
    FmaxD { frd: u8, frs1: u8, frs2: u8 },
    FcvtSD { frd: u8, frs1: u8, rm: u8 },
    FcvtDS { frd: u8, frs1: u8, rm: u8 },
    FcvtWD { rd: u8, frs1: u8, rm: u8 },
    FcvtWuD { rd: u8, frs1: u8, rm: u8 },
    FcvtLD { rd: u8, frs1: u8, rm: u8 },
    FcvtLuD { rd: u8, frs1: u8, rm: u8 },
    FmvXD { rd: u8, frs1: u8 },
    FclassD { rd: u8, frs1: u8 },
    FeqD { rd: u8, frs1: u8, frs2: u8 },
    FltD { rd: u8, frs1: u8, frs2: u8 },
    FleD { rd: u8, frs1: u8, frs2: u8 },
    FcvtDW { frd: u8, rs1: u8, rm: u8 },
    FcvtDWu { frd: u8, rs1: u8, rm: u8 },
    FcvtDL { frd: u8, rs1: u8, rm: u8 },
    FcvtDLu { frd: u8, rs1: u8, rm: u8 },
    FmvDX { frd: u8, rs1: u8 },
    /* Base Opcode = MADD */
    FmaddD { frd: u8, frs1: u8, frs2: u8, frs3: u8, rm: u8 },
    /* Base Opcode = MSUB */
    FmsubD { frd: u8, frs1: u8, frs2: u8, frs3: u8, rm: u8 },
    /* Base Opcode = NMSUB */
    FnmsubD { frd: u8, frs1: u8, frs2: u8, frs3: u8, rm: u8 },
    /* Base Opcode = NMADD */
    FnmaddD { frd: u8, frs1: u8, frs2: u8, frs3: u8, rm: u8 },

    /* Privileged */
    Mret,
    Sret,
    Wfi,
    SfenceVma { rs1: u8, rs2: u8 },
}

impl Op {
    pub fn can_change_control_flow(&self) -> bool {
        match self {
            // Branch and jump instructions will definitely disrupt the control flow.
            Op::Beq {..} |
            Op::Bne {..} |
            Op::Blt {..} |
            Op::Bge {..} |
            Op::Bltu {..} |
            Op::Bgeu {..} |
            Op::Jalr {..} |
            Op::Jal {..} |
            // Return from ecall also changes control flow.
            Op::Mret |
            Op::Sret |
            // They always trigger faults
            Op::Ecall |
            Op::Ebreak |
            Op::Illegal |
            // fence.i might cause instruction cache to be invalidated. If the code executing is invalidated, then we need
            // to stop executing, so it is safer to treat it as special instruction at the moment.
            // sfence.vma has similar effects.
            Op::FenceI |
            Op::SfenceVma {..} => true,
            // Some CSRs need special treatment
            Op::Csrrw { csr, .. } |
            Op::Csrrs { csr, .. } |
            Op::Csrrc { csr, .. } |
            Op::Csrrwi { csr, .. } |
            Op::Csrrsi { csr, .. } |
            Op::Csrrci { csr, .. } => match *csr {
                // A common way of using basic blocks is to `batch' instret and pc increment. So if CSR to be accessed is
                // instret, consider it as special.
                Csr::Instret |
                Csr::Instreth |
                // SATP shouldn't belong here, but somehow Linux assumes setting SATP changes
                // addressing mode immediately...
                Csr::Satp => true,
                _ => false,
            }
            _ => false,
        }
    }

    /// Get the minimal privilege level required to execute the op
    pub fn min_prv_level(self) -> u8 {
        match self {
            Op::Csrrw { csr, .. }
            | Op::Csrrs { csr, .. }
            | Op::Csrrc { csr, .. }
            | Op::Csrrwi { csr, .. }
            | Op::Csrrsi { csr, .. }
            | Op::Csrrci { csr, .. } => csr.min_prv_level(),
            Op::Mret => 3,
            Op::Sret | Op::Wfi | Op::SfenceVma { .. } => 1,
            _ => 0,
        }
    }

    /// Retrieve the RD, RS1, RS2 from the op. If the operation does not use RD, RS1 or RS2, 0 is returned.
    pub fn get_regs(self) -> (u8, u8, u8) {
        match self {
            Op::Illegal => (0, 0, 0),
            Op::Lui { rd, .. } | Op::Auipc { rd, .. } => (rd, 0, 0),
            Op::Jal { rd, .. } => (rd, 0, 0),
            Op::Beq { rs1, rs2, .. }
            | Op::Bne { rs1, rs2, .. }
            | Op::Blt { rs1, rs2, .. }
            | Op::Bge { rs1, rs2, .. }
            | Op::Bltu { rs1, rs2, .. }
            | Op::Bgeu { rs1, rs2, .. } => (0, rs1, rs2),
            Op::Lb { rd, rs1, .. }
            | Op::Lh { rd, rs1, .. }
            | Op::Lw { rd, rs1, .. }
            | Op::Ld { rd, rs1, .. }
            | Op::Lbu { rd, rs1, .. }
            | Op::Lhu { rd, rs1, .. }
            | Op::Lwu { rd, rs1, .. } => (rd, rs1, 0),
            Op::Jalr { rd, rs1, .. } => (rd, rs1, 0),
            Op::Fence => (0, 0, 0),
            Op::FenceI => (0, 0, 0),
            Op::Ecall | Op::Ebreak => (0, 0, 0),
            Op::Mret | Op::Sret => (0, 0, 0),
            Op::Wfi => (0, 0, 0),
            Op::SfenceVma { rs1, rs2 } => (0, rs1, rs2),
            Op::Sb { rs1, rs2, .. }
            | Op::Sh { rs1, rs2, .. }
            | Op::Sw { rs1, rs2, .. }
            | Op::Sd { rs1, rs2, .. } => (0, rs1, rs2),
            Op::Addi { rd, rs1, .. }
            | Op::Slti { rd, rs1, .. }
            | Op::Sltiu { rd, rs1, .. }
            | Op::Xori { rd, rs1, .. }
            | Op::Ori { rd, rs1, .. }
            | Op::Andi { rd, rs1, .. }
            | Op::Addiw { rd, rs1, .. }
            | Op::Slli { rd, rs1, .. }
            | Op::Srli { rd, rs1, .. }
            | Op::Srai { rd, rs1, .. }
            | Op::Slliw { rd, rs1, .. }
            | Op::Srliw { rd, rs1, .. }
            | Op::Sraiw { rd, rs1, .. } => (rd, rs1, 0),
            Op::Add { rd, rs1, rs2 }
            | Op::Sub { rd, rs1, rs2 }
            | Op::Sll { rd, rs1, rs2 }
            | Op::Slt { rd, rs1, rs2 }
            | Op::Sltu { rd, rs1, rs2 }
            | Op::Xor { rd, rs1, rs2 }
            | Op::Srl { rd, rs1, rs2 }
            | Op::Sra { rd, rs1, rs2 }
            | Op::Or { rd, rs1, rs2 }
            | Op::And { rd, rs1, rs2 }
            | Op::Addw { rd, rs1, rs2 }
            | Op::Subw { rd, rs1, rs2 }
            | Op::Sllw { rd, rs1, rs2 }
            | Op::Srlw { rd, rs1, rs2 }
            | Op::Sraw { rd, rs1, rs2 } => (rd, rs1, rs2),
            Op::Mul { rd, rs1, rs2 } => (rd, rs1, rs2),
            Op::Mulh { rd, rs1, rs2 }
            | Op::Mulhsu { rd, rs1, rs2 }
            | Op::Mulhu { rd, rs1, rs2 } => (rd, rs1, rs2),
            Op::Div { rd, rs1, rs2 }
            | Op::Divu { rd, rs1, rs2 }
            | Op::Rem { rd, rs1, rs2 }
            | Op::Remu { rd, rs1, rs2 } => (rd, rs1, rs2),
            Op::Mulw { rd, rs1, rs2 } => (rd, rs1, rs2),
            Op::Divw { rd, rs1, rs2 }
            | Op::Divuw { rd, rs1, rs2 }
            | Op::Remw { rd, rs1, rs2 }
            | Op::Remuw { rd, rs1, rs2 } => (rd, rs1, rs2),
            Op::Csrrw { rd, rs1, .. } | Op::Csrrs { rd, rs1, .. } | Op::Csrrc { rd, rs1, .. } => {
                (rd, rs1, 0)
            }
            Op::Csrrwi { rd, .. } | Op::Csrrsi { rd, .. } | Op::Csrrci { rd, .. } => (rd, 0, 0),
            Op::LrW { rd, rs1, .. } | Op::LrD { rd, rs1, .. } => (rd, rs1, 0),
            Op::ScW { rd, rs1, rs2, .. }
            | Op::ScD { rd, rs1, rs2, .. }
            | Op::AmoswapW { rd, rs1, rs2, .. }
            | Op::AmoswapD { rd, rs1, rs2, .. }
            | Op::AmoaddW { rd, rs1, rs2, .. }
            | Op::AmoaddD { rd, rs1, rs2, .. }
            | Op::AmoxorW { rd, rs1, rs2, .. }
            | Op::AmoxorD { rd, rs1, rs2, .. }
            | Op::AmoandW { rd, rs1, rs2, .. }
            | Op::AmoandD { rd, rs1, rs2, .. }
            | Op::AmoorW { rd, rs1, rs2, .. }
            | Op::AmoorD { rd, rs1, rs2, .. }
            | Op::AmominW { rd, rs1, rs2, .. }
            | Op::AmominD { rd, rs1, rs2, .. }
            | Op::AmomaxW { rd, rs1, rs2, .. }
            | Op::AmomaxD { rd, rs1, rs2, .. }
            | Op::AmominuW { rd, rs1, rs2, .. }
            | Op::AmominuD { rd, rs1, rs2, .. }
            | Op::AmomaxuW { rd, rs1, rs2, .. }
            | Op::AmomaxuD { rd, rs1, rs2, .. } => (rd, rs1, rs2),
            Op::Flw { rs1, .. } | Op::Fld { rs1, .. } => (0, rs1, 0),
            Op::Fsw { rs1, .. } | Op::Fsd { rs1, .. } => (0, rs1, 0),
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
            | Op::FmaxD { .. } => (0, 0, 0),
            Op::FsqrtS { .. } | Op::FsqrtD { .. } | Op::FcvtSD { .. } | Op::FcvtDS { .. } => {
                (0, 0, 0)
            }
            Op::FcvtWS { rd, .. }
            | Op::FcvtWuS { rd, .. }
            | Op::FcvtLS { rd, .. }
            | Op::FcvtLuS { rd, .. }
            | Op::FmvXW { rd, .. }
            | Op::FclassS { rd, .. }
            | Op::FcvtWD { rd, .. }
            | Op::FcvtWuD { rd, .. }
            | Op::FcvtLD { rd, .. }
            | Op::FcvtLuD { rd, .. }
            | Op::FmvXD { rd, .. }
            | Op::FclassD { rd, .. } => (rd, 0, 0),
            Op::FcvtSW { rs1, .. }
            | Op::FcvtSWu { rs1, .. }
            | Op::FcvtSL { rs1, .. }
            | Op::FcvtSLu { rs1, .. }
            | Op::FmvWX { rs1, .. }
            | Op::FcvtDW { rs1, .. }
            | Op::FcvtDWu { rs1, .. }
            | Op::FcvtDL { rs1, .. }
            | Op::FcvtDLu { rs1, .. }
            | Op::FmvDX { rs1, .. } => (0, rs1, 0),
            Op::FeqS { rd, .. }
            | Op::FltS { rd, .. }
            | Op::FleS { rd, .. }
            | Op::FeqD { rd, .. }
            | Op::FltD { rd, .. }
            | Op::FleD { rd, .. } => (rd, 0, 0),
            Op::FmaddS { .. }
            | Op::FmsubS { .. }
            | Op::FnmsubS { .. }
            | Op::FnmaddS { .. }
            | Op::FmaddD { .. }
            | Op::FmsubD { .. }
            | Op::FnmsubD { .. }
            | Op::FnmaddD { .. } => (0, 0, 0),
        }
    }
}
