use super::Op;
use super::Operand;
use core::fmt;

impl Op {
    /// Get the mnemonic for this op. This method looks at discriminant only, so it will print
    /// cmovcc/jcc/setcc instead of actual condition code.
    pub fn mnemonic(&self) -> &'static str {
        match *self {
            Op::Illegal { .. } => "illegal",
            Op::Adc { .. } => "adc",
            Op::Add { .. } => "add",
            Op::And { .. } => "and",
            Op::Call { .. } => "call",
            Op::Cdqe { .. } => "cdqe",
            Op::Cmovcc { .. } => "cmovcc",
            Op::Cmp { .. } => "cmp",
            Op::Cmpxchg { .. } => "cmpxchg",
            Op::Cdq { .. } => "cdq",
            Op::Cqo { .. } => "cqo",
            Op::Div { .. } => "div",
            Op::Hlt { .. } => "hlt",
            Op::Idiv { .. } => "idiv",
            Op::Imul1 { .. } => "imul",
            Op::Imul2 { .. } => "imul",
            Op::Jcc { .. } => "jcc",
            Op::Jmp { .. } => "jmp",
            Op::Lock { .. } => "lock",
            Op::Lea { .. } => "lea",
            Op::Mfence { .. } => "mfence",
            Op::Mov { .. } => "mov",
            Op::Movabs { .. } => "movabs",
            Op::Movsx { .. } => "movsx",
            Op::Movzx { .. } => "movzx",
            Op::Mul { .. } => "mul",
            Op::Neg { .. } => "neg",
            Op::Nop { .. } => "nop",
            Op::Not { .. } => "not",
            Op::Or { .. } => "or",
            Op::Pop { .. } => "pop",
            Op::Push { .. } => "push",
            Op::Ret { .. } => "ret",
            Op::Sar { .. } => "sar",
            Op::Sbb { .. } => "sbb",
            Op::Setcc { .. } => "setcc",
            Op::Shl { .. } => "shl",
            Op::Shr { .. } => "shr",
            Op::Sub { .. } => "sub",
            Op::Test { .. } => "test",
            Op::Xadd { .. } => "xadd",
            Op::Xchg { .. } => "xchg",
            Op::Xor { .. } => "xor",
        }
    }

    /// Print the instruction with optional next pc information.
    fn print(&self, fmt: &mut fmt::Formatter, npc: Option<u64>) -> fmt::Result {
        // Print mnemonic
        match self {
            Op::Cmovcc(_, _, cc) => write!(fmt, "cmov{:-4}", cc),
            Op::Jcc(_, cc) => write!(fmt, "j{:-7}", cc),
            Op::Setcc(_, cc) => write!(fmt, "set{:-5}", cc),
            _ => write!(fmt, "{:-8}", self.mnemonic()),
        }?;

        match self {
            Op::Illegal | Op::Cdqe | Op::Cdq | Op::Cqo | Op::Hlt | Op::Lock | Op::Mfence => (),
            Op::Nop => (),
            Op::Adc(dst, src)
            | Op::Add(dst, src)
            | Op::And(dst, src)
            | Op::Cmp(dst, src)
            | Op::Mov(dst, src)
            | Op::Or(dst, src)
            | Op::Sar(dst, src)
            | Op::Sbb(dst, src)
            | Op::Shl(dst, src)
            | Op::Shr(dst, src)
            | Op::Sub(dst, src)
            | Op::Test(dst, src)
            | Op::Xor(dst, src) => write!(fmt, "{}, {}", dst, src)?,
            &Op::Call(Operand::Imm(imm)) | &Op::Jmp(Operand::Imm(imm)) => {
                let (sign, uimm) = if imm < 0 { ('-', -imm) } else { ('+', imm) };
                write!(fmt, "pc {} {:#x}", sign, uimm)?;
                if let Some(npc) = npc {
                    let target_pc = npc.wrapping_add(imm as u64);
                    write!(fmt, " <{:x}>", target_pc)?;
                }
            }
            Op::Call(src) | Op::Jmp(src) | Op::Push(src) => write!(fmt, "{}", src)?,
            Op::Cmovcc(dst, src, _) | Op::Imul2(dst, src) => write!(fmt, "{}, {}", dst, src)?,
            Op::Div(dst)
            | Op::Idiv(dst)
            | Op::Imul1(dst)
            | Op::Mul(dst)
            | Op::Neg(dst)
            | Op::Not(dst)
            | Op::Pop(dst)
            | Op::Setcc(dst, _) => write!(fmt, "{}", dst)?,
            &Op::Jcc(imm, _) => {
                let (sign, uimm) = if imm < 0 { ('-', -imm) } else { ('+', imm) };
                write!(fmt, "pc {} {:#x}", sign, uimm)?;
                if let Some(npc) = npc {
                    let target_pc = npc.wrapping_add(imm as u64);
                    write!(fmt, " <{:x}>", target_pc)?;
                }
            }
            Op::Lea(dst, src) => write!(fmt, "{}, {}", dst, src)?,
            Op::Movabs(dst, src) => write!(fmt, "{}, {}", dst, src)?,
            Op::Movsx(dst, src) | Op::Movzx(dst, src) => write!(fmt, "{}, {}", dst, src)?,
            &Op::Ret(pop) => {
                if pop != 0 {
                    write!(fmt, "{}", pop)?
                }
            }
            Op::Cmpxchg(dst, src) | Op::Xadd(dst, src) => write!(fmt, "{}, {}", dst, src)?,
            Op::Xchg(dst, src) => write!(fmt, "{}, {}", dst, src)?,
        }

        Ok(())
    }

    /// Pretty-print the assembly with program counter and binary instrumentation
    pub fn pretty_print<'a>(&'a self, pc: u64, code: &'a [u8]) -> impl fmt::Display + 'a {
        Disasm { pc, code, op: self }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.print(fmt, None)
    }
}

struct Disasm<'a> {
    pc: u64,
    code: &'a [u8],
    op: &'a Op,
}

impl<'a> fmt::Display for Disasm<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        if (self.pc & 0xFFFF_FFFF) == self.pc {
            write!(fmt, "{:8x}:       ", self.pc)?;
        } else {
            write!(fmt, "{:16x}:       ", self.pc)?;
        }

        for i in 0..8 {
            if i < self.code.len() {
                write!(fmt, "{:02x}", self.code[i])?;
            } else {
                write!(fmt, "  ")?;
            }
        }
        write!(fmt, "        ")?;

        self.op.print(fmt, Some(self.pc + self.code.len() as u64))?;

        if self.code.len() > 8 {
            writeln!(fmt)?;
            let pc = self.pc + 8;
            if (pc & 0xFFFF_FFFF) == pc {
                write!(fmt, "{:8x}:       ", pc)?;
            } else {
                write!(fmt, "{:16x}:       ", pc)?;
            }

            for i in 8..self.code.len() {
                write!(fmt, "{:02x}", self.code[i])?;
            }
        }

        Ok(())
    }
}
