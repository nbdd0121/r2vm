use super::{Location, Operand, Register, Memory, Size, Op};
use super::builder::*;
use super::op::{REG_GPB, REG_GPB2, REG_GPW, REG_GPD, REG_GPQ};

pub struct Decoder<'a> {
    iter: &'a mut dyn Iterator<Item=u8>,
}

impl<'a> Decoder<'a> {
    pub fn new(iter: &'a mut dyn Iterator<Item=u8>) -> Self {
        Decoder {
            iter,
        }
    }

    /// Decode a single byte
    pub fn byte(&mut self) -> u8 {
        self.iter.next().unwrap()
    }

    /// Decode a dword
    pub fn dword(&mut self) -> u32 {
        let mut result = 0;
        for i in 0..4 {
            result |= (self.byte() as u32) << (i * 8);
        }
        result
    }

    pub fn register_of_size(reg: u8, size: Size, rex: u8) -> Register {
        // Make sure this will always be a valid register
        let reg = reg & 15;

        let mask = match size {
            Size::Byte => if rex & 0x40 != 0 && reg >= 4 && reg <= 7 { REG_GPB2 } else { REG_GPB },
            Size::Word => REG_GPW,
            Size::Dword => REG_GPD,
            Size::Qword => REG_GPQ,
        };

        // This will always be valid
        unsafe { core::mem::transmute(reg | mask) }
    }

    /// Decode a ModR/M sequence
    pub fn modrm(&mut self, rex: u8, size: Size) -> (Location, Register) {
        let first_byte = self.byte();
        let mut modb = first_byte >> 6;
        let rm = first_byte & 0b111;

        // Decode register and prefix with REX.R.
        let mut reg_id = (first_byte >> 3) & 0b111;
        if rex & 0x4 != 0 { reg_id += 8 }

        // For register with size 1 and no REX set, dil and sil etc are not accessible.
        let reg = Self::register_of_size(reg_id, size, rex);

        // Operand is a register.
        if modb == 0b11 {
            let op_id = rm | if rex & 0x1 != 0 { 8 } else { 0 };
            let operand = Self::register_of_size(op_id, size, rex);
            return (Location::Reg(operand), reg)
        }

        if modb == 0b00 && rm == 0b100 {
            // rip-relative addressing not yet supported
            unimplemented!()
        }

        let mut mem = Memory {
            base: None,
            index: None,
            displacement: 0,
            size,
        };

        // No SIB bytes.
        if rm != 0b100 {
            let base_id = rm | if rex & 0x1 != 0 { 8 } else { 0 };
            mem.base = Some(Self::register_of_size(base_id, Size::Qword, 0));
        } else {
            let sib = self.byte();
            let ss = sib >> 6;
            let index = if rex & 0x2 != 0 { 8 } else { 0 } | ((sib >> 3) & 0b111);
            let base = if rex & 0x1 != 0 { 8 } else { 0 } | (sib & 0b111);

            // RSP cannot be index. index = RSP means there is no index.
            if index != 0b100 {
                mem.index = Some((Self::register_of_size(index, Size::Qword, 0), 1 << ss));
            }

            // If mod = 0, and base = RBP or R13, then we have no base register.
            if modb == 0 && (base & 7) == 0b101 {
                // a tiny trick, so we have 32-bit displacement
                modb = 0b10;
            } else {
                mem.base = Some(Self::register_of_size(base, Size::Qword, 0));
            }
        }

        if modb == 0b00 {
        } else if modb == 0b01 {
            mem.displacement = self.byte() as i8 as i32
        } else {
            mem.displacement = self.dword() as i32
        }
        (Location::Mem(mem), reg)
    }

    fn decode_alu(dst: Location, src: Operand, id: Register) -> Op {
        match (id as u8) & 7 {
            0 => Op::Add(dst, src),
            1 => Op::Or (dst, src),
            2 => Op::Adc(dst, src),
            3 => Op::Sbb(dst, src),
            4 => Op::And(dst, src),
            5 => Op::Sub(dst, src),
            6 => Op::Xor(dst, src),
            7 => Op::Cmp(dst, src),
            _ => unsafe { std::hint::unreachable_unchecked() }
        }
    }

    pub fn op(&mut self) -> Op {
        let mut rex = 0;
        let mut opsize = Size::Dword;

        // Keep reading prefixes.
        let mut opcode: u32 = loop {
            let prefix = self.byte();
            if (prefix & 0xF0) == 0x40 {
                // REX prefix
                rex = prefix;
                // REX.W
                if rex & 0x08 != 0 { opsize = Size::Qword }
            } else if prefix == 0x66 {
                opsize = Size::Word;
            } else {
                // Unread the byte
                break prefix as u32;
            }
        };

        // Handle escape sequences
        if opcode == 0xF {
            opcode = opcode << 8 | self.byte() as u32;
        }

        // Handling byte-sized ops
        match opcode {
            0x88 | 0x8A | 0xF6 => {
                opcode += 1;
                opsize = Size::Byte;
            }
            _ => (),
        }

        match opcode {
            0x0FB6 => {
                let (src, dst) = self.modrm(rex, opsize);
                Op::Movzx(dst, src.resize(Size::Byte))
            }
            0x0FB7 => {
                let (src, dst) = self.modrm(rex, opsize);
                Op::Movzx(dst, src.resize(Size::Word))
            }
            0x0FBE => {
                let (src, dst) = self.modrm(rex, opsize);
                Op::Movsx(dst, src.resize(Size::Byte))
            }
            0x0FBF => {
                let (src, dst) = self.modrm(rex, opsize);
                Op::Movsx(dst, src.resize(Size::Word))
            }
            0x63 => {
                let (src, dst) = self.modrm(rex, opsize);
                Op::Movsx(dst, src.resize(Size::Dword))
            }
            0x83 => {
                let (operand, reg) = self.modrm(rex, opsize);
                Self::decode_alu(operand, Imm(self.byte() as i8 as i64), reg)
            }
            0x89 => {
                let (operand, reg) = self.modrm(rex, opsize);
                Op::Mov(operand, OpReg(reg))
            }
            0x8B => {
                let (operand, reg) = self.modrm(rex, opsize);
                Op::Mov(reg.into(), operand.into())
            }
            0xF7 => {
                let (operand, reg) = self.modrm(rex, opsize);
                match reg as u8 & 7 {
                    6 => Op::Div(operand),
                    7 => Op::Idiv(operand),
                    _ => unimplemented!(),
                }
            }
            _ => unimplemented!("opcode {:x}", opcode),
        }
    }
}
