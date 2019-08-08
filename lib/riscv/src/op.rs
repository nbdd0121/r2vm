use super::Csr;

/// This includes all supported RISC-V ops.
/// Ops are sorted in the following order
/// * Canonical order of extension
/// * Increasing base opcode number
/// * Increasing funct3 and then funct7, or their ordering in RISC-V spec
#[derive(Clone, Copy)]
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
    Addw { rd: u8, rs1: u8, rs2: u8},
    Subw { rd: u8, rs1: u8, rs2: u8},
    Sllw { rd: u8, rs1: u8, rs2: u8},
    Srlw { rd: u8, rs1: u8, rs2: u8},
    Sraw { rd: u8, rs1: u8, rs2: u8},
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
    LrW { rd: u8, rs1: u8 },
    LrD { rd: u8, rs1: u8 },
    ScW { rd: u8, rs1: u8, rs2: u8 },
    ScD { rd: u8, rs1: u8, rs2: u8 },
    AmoswapW { rd: u8, rs1: u8, rs2: u8 },
    AmoswapD { rd: u8, rs1: u8, rs2: u8 },
    AmoaddW { rd: u8, rs1: u8, rs2: u8 },
    AmoaddD { rd: u8, rs1: u8, rs2: u8 },
    AmoxorW { rd: u8, rs1: u8, rs2: u8 },
    AmoxorD { rd: u8, rs1: u8, rs2: u8 },
    AmoandW { rd: u8, rs1: u8, rs2: u8 },
    AmoandD { rd: u8, rs1: u8, rs2: u8 },
    AmoorW { rd: u8, rs1: u8, rs2: u8 },
    AmoorD { rd: u8, rs1: u8, rs2: u8 },
    AmominW { rd: u8, rs1: u8, rs2: u8 },
    AmominD { rd: u8, rs1: u8, rs2: u8 },
    AmomaxW { rd: u8, rs1: u8, rs2: u8 },
    AmomaxD { rd: u8, rs1: u8, rs2: u8 },
    AmominuW { rd: u8, rs1: u8, rs2: u8 },
    AmominuD { rd: u8, rs1: u8, rs2: u8 },
    AmomaxuW { rd: u8, rs1: u8, rs2: u8 },
    AmomaxuD { rd: u8, rs1: u8, rs2: u8 },

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
            Op::Csrrw { csr, .. } |
            Op::Csrrs { csr, .. } |
            Op::Csrrc { csr, .. } |
            Op::Csrrwi { csr, .. } |
            Op::Csrrsi { csr, .. } |
            Op::Csrrci { csr, .. } => {
                csr.min_prv_level()
            }
            Op::Sret |
            Op::Wfi |
            Op::SfenceVma {..} => 1,
            _ => 0,
        }
    }
}
