#![cfg_attr(not(feature = "std"), no_std)]

// Currently our encoder makes use of Vec, so we need to depend on alloc
// In the future this dependency could be removed.
extern crate alloc;

mod op;
mod encode;
mod decode;
pub mod disasm;

pub use encode::Encoder;
pub use decode::Decoder;
pub use op::{ConditionCode, Register, Memory, Location, Operand, Op, Size};

/// Prelude for easy assembly
pub mod builder {
    pub use super::Location::*;
    pub use super::Operand::{Reg as OpReg, Mem as OpMem, Imm};
}
