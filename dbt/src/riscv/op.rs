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
}

extern {
    fn legacy_can_change_control_flow(op: &LegacyOp) -> bool;
}

impl Op {
    pub fn can_change_control_flow(&self) -> bool {
        match self {
            Op::Legacy(op) => unsafe { legacy_can_change_control_flow(op) },
            Op::Illegal => true,
            _ => false,
        }
    }
}
