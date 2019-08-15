use super::{Op};
use super::Operand;

pub fn mnemonic(op: &Op) -> &'static str {
    match *op {
        Op::Illegal {..} => "illegal",
        Op::Adc {..} => "adc",
        Op::Add {..} => "add",
        Op::And {..} => "and",
        Op::Call {..} => "call",
        Op::Cdqe {..} => "cdqe",
        Op::Cmovcc {..} => "cmovcc",
        Op::Cmp {..} => "cmp",
        Op::Cdq {..} => "cdq",
        Op::Cqo {..} => "cqo",
        Op::Div {..} => "div",
        Op::Idiv {..} => "idiv",
        Op::Imul1 {..} => "imul",
        Op::Imul2 {..} => "imul",
        Op::Jcc {..} => "jcc",
        Op::Jmp {..} => "jmp",
        Op::Lock {..} => "lock",
        Op::Lea {..} => "lea",
        Op::Mfence {..} => "mfence",
        Op::Mov {..} => "mov",
        Op::Movabs {..} => "movabs",
        Op::Movsx {..} => "movsx",
        Op::Movzx {..} => "movzx",
        Op::Mul {..} => "mul",
        Op::Neg {..} => "neg",
        Op::Nop {..} => "nop",
        Op::Not {..} => "not",
        Op::Or {..} => "or",
        Op::Pop {..} => "pop",
        Op::Push {..} => "push",
        Op::Ret {..} => "ret",
        Op::Sar {..} => "sar",
        Op::Sbb {..} => "sbb",
        Op::Setcc {..} => "setcc",
        Op::Shl {..} => "shl",
        Op::Shr {..} => "shr",
        Op::Sub {..} => "sub",
        Op::Test {..} => "test",
        Op::Xadd {..} => "xadd",
        Op::Xchg {..} => "xchg",
        Op::Xor {..} => "xor",
    }
}

#[cfg(feature = "std")]
pub fn print_instr(pc: u64, code: &[u8], inst: &Op) {
    let mnemonic = mnemonic(inst);

    if (pc & 0xFFFF_FFFF) == pc {
        eprint!("{:8x}:       ", pc);
    } else {
        eprint!("{:16x}:       ", pc);
    }

    for i in 0..8 {
        if i < code.len() {
            eprint!("{:02x}", code[i]);
        } else {
            eprint!("  ");
        }
    }
    eprint!("        ");

    // Print mnemonic
    match inst {
        Op::Cmovcc(_, _, cc) => eprint!("cmov{:-4}", cc),
        Op::Jcc(_, cc) => eprint!("j{:-7}", cc),
        Op::Setcc(_, cc) => eprint!("set{:-5}", cc),
        _ => eprint!("{:-8}", mnemonic),
    }

    match inst {
        Op::Illegal |
        Op::Cdqe |
        Op::Cdq |
        Op::Cqo |
        Op::Lock |
        Op::Mfence => (),
        Op::Nop => (),
        Op::Adc(dst, src) |
        Op::Add(dst, src) |
        Op::And(dst, src) |
        Op::Cmp(dst, src) |
        Op::Mov(dst, src) |
        Op::Or(dst, src) |
        Op::Sar(dst, src) |
        Op::Sbb(dst, src) |
        Op::Shl(dst, src) |
        Op::Shr(dst, src) |
        Op::Sub(dst, src) |
        Op::Test(dst, src) |
        Op::Xor(dst, src) => eprint!("{}, {}", dst, src),
        &Op::Call(Operand::Imm(imm)) |
        &Op::Jmp(Operand::Imm(imm)) => {
            let target_pc = pc.wrapping_add(code.len() as u64).wrapping_add(imm as u64);
            let (sign, imm) = if imm < 0 {
                ('-', -imm)
            } else {
                ('+', imm)
            };
            eprint!("pc {} {:#x} <{:x}>", sign, imm, target_pc)
        }
        Op::Call(src) |
        Op::Jmp(src) |
        Op::Push(src) => eprint!("{}", src),
        Op::Cmovcc(dst, src, _) |
        Op::Imul2(dst, src) => eprint!("{}, {}", dst, src),
        Op::Div(dst) |
        Op::Idiv(dst) |
        Op::Imul1(dst) |
        Op::Mul(dst) |
        Op::Neg(dst) |
        Op::Not(dst) |
        Op::Pop(dst) |
        Op::Setcc(dst, _) => eprint!("{}", dst),
        &Op::Jcc(imm, _) => {
            let target_pc = pc.wrapping_add(code.len() as u64).wrapping_add(imm as u64);
            let (sign, imm) = if imm < 0 {
                ('-', -imm)
            } else {
                ('+', imm)
            };
            eprint!("pc {} {:#x} <{:x}>", sign, imm, target_pc)
        }
        Op::Lea(dst, src) => eprint!("{}, {}", dst, src),
        Op::Movabs(dst, src) => eprint!("{}, {}", dst, src),
        Op::Movsx(dst, src) |
        Op::Movzx(dst, src) => eprint!("{}, {}", dst, src),
        &Op::Ret(pop) => if pop != 0 { eprint!("{}", pop) }
        Op::Xadd(dst, src) => eprint!("{}, {}", dst, src),
        Op::Xchg(dst, src) => eprint!("{}, {}", dst, src),
    }
    eprintln!();

    if code.len() > 8 {
        let pc = pc + 8;
        if (pc & 0xFFFF_FFFF) == pc {
            eprint!("{:8x}:       ", pc);
        } else {
            eprint!("{:16x}:       ", pc);
        }

        for i in 8..code.len() {
            eprint!("{:02x}", code[i]);
        }

        eprintln!()
    }
}
