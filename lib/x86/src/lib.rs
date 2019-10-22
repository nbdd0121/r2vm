#![no_std]

// Currently our encoder makes use of Vec, so we need to depend on alloc
// In the future this dependency could be removed.
extern crate alloc;

mod decode;
mod disasm;
mod encode;
mod op;

pub use decode::{decode, Decoder};
pub use encode::{encode, Encoder};
pub use op::{ConditionCode, Location, Memory, Op, Operand, Register, Size};

/// Prelude for easy assembly
pub mod builder {
    pub use super::Location::*;
    pub use super::Operand::{Imm, Mem as OpMem, Reg as OpReg};
}
