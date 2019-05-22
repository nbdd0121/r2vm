use super::Csr;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LegacyOp {
    opcode: u8,
    rd: u8,
    rs1: u8,
    rs2: u8,
    imm: i32,
}

/// This includes all supported RISC-V ops.
/// Ops are sorted in the following order
/// * Canonical order of extension
/// * Increasing base opcode number
/// * Increasing funct3 and then funct7, or their ordering in RISC-V spec
#[derive(Clone, Copy)]
pub enum Op {
    Legacy(LegacyOp),
    Illegal,
    /* Base Opcode = MISC-MEM */
    Fence,
    FenceI,
    /* Base Opcode = AUIPC */
    Auipc { rd: u8, imm: i32 },
    /* Base Opcode = BRANCH */
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

    /* Privileged */
    Sret,
    Wfi,
    SfenceVma { rs1: u8, rs2: u8 },
}

extern {
    fn legacy_can_change_control_flow(op: &LegacyOp) -> bool;
}

impl Op {
    pub fn can_change_control_flow(&self) -> bool {
        match self {
            Op::Legacy(op) => unsafe { legacy_can_change_control_flow(op) },
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
            Op::Csrrci { csr, .. } => match csr {
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
}
