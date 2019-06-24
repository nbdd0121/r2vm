// The register name is represented using an integer. The lower 4-bit represents the index, and the highest bits
// represents types of the register.
const REG_GPB: u8 = 0x10;
const REG_GPW: u8 = 0x20;
const REG_GPD: u8 = 0x30;
const REG_GPQ: u8 = 0x40;
// This is for special spl, bpl, sil and dil
const REG_GPB2: u8 = 0x50;

#[repr(u8)]
#[derive(Clone, Copy)]
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

    None = 0,
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

pub struct Memory {
    pub displacement: u32,
    pub base: Register,
    pub index: Register,
    pub scale: u8,
    pub size: u8,
}

pub enum Operand {
    Reg(Register),
    Mem(Memory),
    Imm(u64),
}

impl Operand {
    pub fn size(&self) -> u8 {
        match self {
            Operand::Reg(reg) => reg.size(),
            Operand::Mem(mem) => mem.size,
            Operand::Imm(_) => unreachable!(),
        }
    }
}

#[repr(u8)]
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

pub enum Op {
    Illegal,
    Add { dst: Operand, src: Operand },
    And { dst: Operand, src: Operand },
    Call { src: Operand },
    Cdqe,
    Cmp { dst: Operand, src: Operand },
    Cdq,
    Cqo,
    Div { src: Operand },
    Idiv { src: Operand },
    // imul with implicit AX
    ImulA { src: Operand },
    Imul { dst: Register, src: Operand },
    Jcc { src: Operand, cc: ConditionCode },
    Jmp { src: Operand },
    Lea { dst: Register, src: Memory },
    Mov { dst: Operand, src: Operand },
    Movabs { dst: Operand, src: Operand },
    Movsx { dst: Operand, src: Operand },
    Movzx { dst: Operand, src: Operand },
    Mul { src: Operand },
    Neg { src: Operand },
    Nop,
    Not { src: Operand },
    Or { dst: Operand, src: Operand },
    Pop { dst: Operand },
    Push { src: Operand },
    Ret,
    // ret with stack pop
    RetI { dst: u16 },
    Sar { dst: Operand, src: Operand },
    Sbb { dst: Operand, src: Operand },
    Setcc { dst: Operand, cc: ConditionCode },
    Shl { dst: Operand, src: Operand },
    Shr { dst: Operand, src: Operand },
    Sub { dst: Operand, src: Operand },
    Test { dst: Operand, src: Operand },
    Xchg { dst: Operand, src: Operand },
    Xor { dst: Operand, src: Operand },
}

// index * scale
impl std::ops::Mul<u8> for Register {
    type Output = Memory;
    fn mul(self, rhs: u8) -> Memory {
        Memory {
            displacement: 0,
            base: Register::None,
            index: self,
            scale: rhs,
            size: 0,
        }
    }
}

// base + index * scale
impl std::ops::Add<Memory> for Register {
    type Output = Memory;
    fn add(self, mut rhs: Memory) -> Memory {
        rhs.base = self;
        rhs
    }
}

// base + index
impl std::ops::Add<Register> for Register {
    type Output = Memory;
    fn add(self, rhs: Register) -> Memory {
        Memory {
            displacement: 0,
            base: self,
            index: rhs,
            scale: 1,
            size: 0,
        }
    }
}

// base + displacement
impl std::ops::Add<u32> for Register {
    type Output = Memory;
    fn add(self, rhs: u32) -> Memory {
        Memory {
            displacement: rhs,
            base: self,
            index: Register::None,
            scale: 0,
            size: 0,
        }
    }
}

// base - displacement
impl std::ops::Sub<u32> for Register {
    type Output = Memory;
    fn sub(self, rhs: u32) -> Memory {
        Memory {
            displacement: -(rhs as i32) as u32,
            base: self,
            index: Register::None,
            scale: 0,
            size: 0,
        }
    }
}

// [base +] index * scale + displacement
impl std::ops::Add<u32> for Memory {
    type Output = Memory;
    fn add(mut self, rhs: u32) -> Memory {
        self.displacement = rhs;
        self
    }
}

impl std::ops::Sub<u32> for Memory {
    type Output = Memory;
    fn sub(mut self, rhs: u32) -> Memory {
        self.displacement = -(rhs as i32) as u32;
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
