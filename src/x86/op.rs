use std::fmt::{self, Write};

/// Helper for displaying signed hex
struct Signed(i64);

impl fmt::LowerHex for Signed {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let value = if self.0 < 0 {
            let value = self.0.wrapping_neg() as u64;
            f.write_char('-')?;
            value
        } else {
            if f.sign_plus() { f.write_char('+')? }
            self.0 as u64
        };
        if f.alternate() {
            f.write_str("0x")?;
        }
        write!(f, "{:x}", value)
    }
}

// The register name is represented using an integer. The lower 4-bit represents the index, and the highest bits
// represents types of the register.
pub const REG_GPB: u8 = 0x10;
pub const REG_GPW: u8 = 0x20;
pub const REG_GPD: u8 = 0x30;
pub const REG_GPQ: u8 = 0x40;
// This is for special spl, bpl, sil and dil
pub const REG_GPB2: u8 = 0x50;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Register {
    // General purpose registers
    AL   = 0  | REG_GPB, AX   = 0  | REG_GPW, EAX  = 0  | REG_GPD, RAX = 0  | REG_GPQ,
    CL   = 1  | REG_GPB, CX   = 1  | REG_GPW, ECX  = 1  | REG_GPD, RCX = 1  | REG_GPQ,
    DL   = 2  | REG_GPB, DX   = 2  | REG_GPW, EDX  = 2  | REG_GPD, RDX = 2  | REG_GPQ,
    BL   = 3  | REG_GPB, BX   = 3  | REG_GPW, EBX  = 3  | REG_GPD, RBX = 3  | REG_GPQ,
    AH   = 4  | REG_GPB, SP   = 4  | REG_GPW, ESP  = 4  | REG_GPD, RSP = 4  | REG_GPQ,
    CH   = 5  | REG_GPB, BP   = 5  | REG_GPW, EBP  = 5  | REG_GPD, RBP = 5  | REG_GPQ,
    DH   = 6  | REG_GPB, SI   = 6  | REG_GPW, ESI  = 6  | REG_GPD, RSI = 6  | REG_GPQ,
    BH   = 7  | REG_GPB, DI   = 7  | REG_GPW, EDI  = 7  | REG_GPD, RDI = 7  | REG_GPQ,
    R8B  = 8  | REG_GPB, R8W  = 8  | REG_GPW, R8D  = 8  | REG_GPD, R8  = 8  | REG_GPQ,
    R9B  = 9  | REG_GPB, R9W  = 9  | REG_GPW, R9D  = 9  | REG_GPD, R9  = 9  | REG_GPQ,
    R10B = 10 | REG_GPB, R10W = 10 | REG_GPW, R10D = 10 | REG_GPD, R10 = 10 | REG_GPQ,
    R11B = 11 | REG_GPB, R11W = 11 | REG_GPW, R11D = 11 | REG_GPD, R11 = 11 | REG_GPQ,
    R12B = 12 | REG_GPB, R12W = 12 | REG_GPW, R12D = 12 | REG_GPD, R12 = 12 | REG_GPQ,
    R13B = 13 | REG_GPB, R13W = 13 | REG_GPW, R13D = 13 | REG_GPD, R13 = 13 | REG_GPQ,
    R14B = 14 | REG_GPB, R14W = 14 | REG_GPW, R14D = 14 | REG_GPD, R14 = 14 | REG_GPQ,
    R15B = 15 | REG_GPB, R15W = 15 | REG_GPW, R15D = 15 | REG_GPD, R15 = 15 | REG_GPQ,
    // Special register that requires REX prefix to access.
    SPL = 4 | REG_GPB2, BPL = 5 | REG_GPB2, SIL = 6 | REG_GPB2, DIL = 7 | REG_GPB2,
}

impl Register {
    pub fn size(self) -> u8 {
        let num = self as u8;
        match num & 0xF0 {
            REG_GPB | REG_GPB2 => 1,
            REG_GPW => 2,
            REG_GPD => 4,
            REG_GPQ => 8,
            _ => unreachable!(),
        }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", super::disasm::register_name(*self as u8))
    }
}

#[derive(Clone, Copy)]
pub struct Memory {
    // We don't need to worry about the size, Rust can optimise it to 1 bytes
    pub base: Option<Register>,
    pub index: Option<(Register, u8)>,
    pub displacement: i32,
    pub size: u8,
}

impl fmt::Display for Memory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let qualifier = match self.size {
            1 => "byte",
            2 => "word",
            4 => "dword",
            8 => "qword",
            _ => "(unknown)",
        };
        write!(f, "{} [", qualifier)?;
        let mut first = true;

        if let Some(base) = self.base {
            write!(f, "{}", base)?;
            first = false;
        }

        if let Some((index, scale)) = self.index {
            if first {
                first = false;
            } else {
                write!(f, "+")?;
            }
            write!(f, "{}", index)?;
            if scale != 1 {
                write!(f, "*{}", scale)?;
            }
        }

        if first {
            // Write out the full address in this case.
            write!(f, "{:#x}", self.displacement as u64)?;
        } else if self.displacement != 0 {
            write!(f, "{:+#x}", Signed(self.displacement as i64))?;
        }

        write!(f, "]")
    }
}

/// Represent a register or memory location. Can be used as left-value operand.
#[derive(Clone, Copy)]
pub enum Location {
    Reg(Register),
    Mem(Memory),
}

impl Location {
    pub fn size(&self) -> u8 {
        match self {
            Location::Reg(reg) => reg.size(),
            Location::Mem(mem) => mem.size,
        }
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Location::Reg(it) => it.fmt(f),
            Location::Mem(it) => it.fmt(f),
        }
    }
}

/// Represent a register, memory or immediate value. Can be used as right-value operand.
#[derive(Clone, Copy)]
pub enum Operand {
    Reg(Register),
    Mem(Memory),
    Imm(i64),
}

impl Operand {
    pub fn size(&self) -> u8 {
        match self {
            Operand::Reg(reg) => reg.size(),
            Operand::Mem(mem) => mem.size,
            Operand::Imm(_) => unreachable!(),
        }
    }

    pub fn as_loc(self) -> Result<Location, i64> {
        match self {
            Operand::Reg(it) => Ok(Location::Reg(it)),
            Operand::Mem(it) => Ok(Location::Mem(it)),
            Operand::Imm(it) => Err(it),
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Operand::Reg(it) => it.fmt(f),
            Operand::Mem(it) => it.fmt(f),
            &Operand::Imm(it) => write!(f, "{:+#x}", Signed(it)),
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum ConditionCode {
    Overflow = 0x0,
    NotOverflow = 0x1,
    Below = 0x2, // Carry = 0x2, NotAboveEqual = 0x2,
    AboveEqual = 0x3, // NotBelow = 0x3, NotCarry = 0x3,
    Equal = 0x4, // Zero = 0x4,
    NotEqual = 0x5, // NotZero = 0x5,
    BelowEqual = 0x6, // NotAbove = 0x6,
    Above = 0x7, // NotBelowEqual = 0x7,
    Sign = 0x8,
    NotSign = 0x9,
    Parity = 0xA, // ParityEven = 0xA,
    NotParity = 0xB, // ParityOdd = 0xB,
    Less = 0xC, // NotGreaterEqual = 0xC,
    GreaterEqual = 0xD, // NotLess = 0xD,
    LessEqual = 0xE, // NotGreater = 0xE,
    Greater = 0xF, // NotLessEqual = 0xF,
}

impl fmt::Display for ConditionCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let str = match self {
            ConditionCode::Overflow => "o",
            ConditionCode::NotOverflow => "no",
            ConditionCode::Below => "b",
            ConditionCode::AboveEqual => "ae",
            ConditionCode::Equal => "e",
            ConditionCode::NotEqual => "ne",
            ConditionCode::BelowEqual => "be",
            ConditionCode::Above => "a",
            ConditionCode::Sign => "s",
            ConditionCode::NotSign => "ns",
            ConditionCode::Parity => "p",
            ConditionCode::NotParity => "np",
            ConditionCode::Less => "l",
            ConditionCode::GreaterEqual => "ge",
            ConditionCode::LessEqual => "le",
            ConditionCode::Greater => "g",
        };
        str.fmt(f)
    }
}

#[derive(Clone, Copy)]
pub enum Op {
    Illegal,
    Add(Location, Operand),
    And(Location, Operand),
    Call(Operand),
    Cdqe,
    Cmovcc(Register, Location, ConditionCode),
    Cmp(Location, Operand),
    Cdq,
    Cqo,
    Div(Location),
    Idiv(Location),
    Imul1(Location),
    Imul2(Register, Location),
    Jcc(i32, ConditionCode),
    Jmp(Operand),
    Lea(Register, Memory),
    Mov(Location, Operand),
    /// mov instruction with absolute address as src or dst
    Movabs(Operand, Operand),
    Movsx(Register, Location),
    Movzx(Register, Location),
    Mul(Location),
    Neg(Location),
    Nop,
    Not(Location),
    Or(Location, Operand),
    Pop(Location),
    Push(Operand),
    // ret with stack pop. If the instruction shouldn't pop stack, set pop to 0.
    Ret(u16),
    Sar(Location, Operand),
    Sbb(Location, Operand),
    Setcc(Location, ConditionCode),
    Shl(Location, Operand),
    Shr(Location, Operand),
    Sub(Location, Operand),
    Test(Location, Operand),
    Xchg(Location, Location),
    Xor(Location, Operand),
}

// index * scale
impl std::ops::Mul<u8> for Register {
    type Output = Memory;
    fn mul(self, rhs: u8) -> Memory {
        Memory {
            displacement: 0,
            base: None,
            index: Some((self, rhs)),
            size: 0,
        }
    }
}

// base + index * scale
impl std::ops::Add<Memory> for Register {
    type Output = Memory;
    fn add(self, mut rhs: Memory) -> Memory {
        rhs.base = Some(self);
        rhs
    }
}

// base + index
impl std::ops::Add<Register> for Register {
    type Output = Memory;
    fn add(self, rhs: Register) -> Memory {
        Memory {
            displacement: 0,
            base: Some(self),
            index: Some((rhs, 1)),
            size: 0,
        }
    }
}

// base + displacement
impl std::ops::Add<i32> for Register {
    type Output = Memory;
    fn add(self, rhs: i32) -> Memory {
        Memory {
            displacement: rhs,
            base: Some(self),
            index: None,
            size: 0,
        }
    }
}

// base - displacement
impl std::ops::Sub<i32> for Register {
    type Output = Memory;
    fn sub(self, rhs: i32) -> Memory {
        Memory {
            displacement: -rhs,
            base: Some(self),
            index: None,
            size: 0,
        }
    }
}

// [base +] index * scale + displacement
impl std::ops::Add<i32> for Memory {
    type Output = Memory;
    fn add(mut self, rhs: i32) -> Memory {
        self.displacement = rhs;
        self
    }
}

impl std::ops::Sub<i32> for Memory {
    type Output = Memory;
    fn sub(mut self, rhs: i32) -> Memory {
        self.displacement = -rhs;
        self
    }
}

impl Memory {
    pub fn qword(mut self) -> Self {
        self.size = 8;
        self
    }

    pub fn dword(mut self) -> Self {
        self.size = 4;
        self
    }

    pub fn word(mut self) -> Self {
        self.size = 2;
        self
    }

    pub fn byte(mut self) -> Self {
        self.size = 1;
        self
    }
}
