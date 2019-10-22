use super::builder::*;
use super::op::{REG_GPB, REG_GPB2, REG_GPD, REG_GPQ, REG_GPW};
use super::{Location, Memory, Op, Operand, Register, Size};
use core::convert::TryInto;

pub struct Decoder<'a> {
    pub iter: &'a mut dyn FnMut() -> u8,
}

impl<'a> Decoder<'a> {
    /// Decode a single byte
    pub fn byte(&mut self) -> u8 {
        (self.iter)()
    }

    /// Decode a word
    pub fn word(&mut self) -> u16 {
        let mut result = 0;
        for i in 0..2 {
            result |= (self.byte() as u16) << (i * 8);
        }
        result
    }

    /// Decode a dword
    pub fn dword(&mut self) -> u32 {
        let mut result = 0;
        for i in 0..4 {
            result |= (self.byte() as u32) << (i * 8);
        }
        result
    }

    /// Decode a dword
    pub fn qword(&mut self) -> u64 {
        let mut result = 0;
        for i in 0..8 {
            result |= (self.byte() as u64) << (i * 8);
        }
        result
    }

    /// Decode an immediate
    pub fn immediate(&mut self, size: Size) -> i64 {
        match size {
            Size::Byte => self.byte() as i8 as i64,
            Size::Word => self.word() as i16 as i64,
            Size::Dword => self.dword() as i32 as i64,
            Size::Qword => self.qword() as i64,
        }
    }

    pub fn register_of_size(reg: u8, size: Size, rex: u8) -> Register {
        // Make sure this will always be a valid register
        let reg = reg & 15;

        let mask = match size {
            Size::Byte => {
                if rex & 0x40 != 0 && reg >= 4 && reg <= 7 {
                    REG_GPB2
                } else {
                    REG_GPB
                }
            }
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
        if rex & 0x4 != 0 {
            reg_id += 8
        }

        // For register with size 1 and no REX set, dil and sil etc are not accessible.
        let reg = Self::register_of_size(reg_id, size, rex);

        // Operand is a register.
        if modb == 0b11 {
            let op_id = rm | if rex & 0x1 != 0 { 8 } else { 0 };
            let operand = Self::register_of_size(op_id, size, rex);
            return (Location::Reg(operand), reg);
        }

        if modb == 0b00 && rm == 0b101 {
            // rip-relative addressing not yet supported
            let displacement = self.dword() as i32;
            return (Location::Mem(Register::RIP + displacement), reg);
        }

        let mut mem = Memory { base: None, index: None, displacement: 0, size };

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

    fn decode_alu(dst: Location, src: Operand, id: u8) -> Op {
        match id & 7 {
            0 => Op::Add(dst, src),
            1 => Op::Or(dst, src),
            2 => Op::Adc(dst, src),
            3 => Op::Sbb(dst, src),
            4 => Op::And(dst, src),
            5 => Op::Sub(dst, src),
            6 => Op::Xor(dst, src),
            7 => Op::Cmp(dst, src),
            _ => unreachable!(),
        }
    }

    fn decode_shift(dst: Location, src: Operand, id: Register) -> Op {
        match (id as u8) & 7 {
            // 0 => Op::Rol(dst, src),
            // 1 => Op::Ror(dst, src),
            // 2 => Op::Rcl(dst, src),
            // 3 => Op::Rcr(dst, src),
            4 => Op::Shl(dst, src),
            5 => Op::Shr(dst, src),
            7 => Op::Sar(dst, src),
            _ => unimplemented!(),
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
                if rex & 0x08 != 0 {
                    opsize = Size::Qword
                }
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
            // These are all INST r/m, r ALU ops
            0x00 | 0x08 | 0x10 | 0x18 | 0x20 | 0x28 | 0x30 | 0x38 |
            // These are all INST r, r/m ALU ops
            0x02 | 0x0A | 0x12 | 0x1A | 0x22 | 0x2A | 0x32 | 0x3A |
            // These are all RAX short encoded ALU ops
            0x04 | 0x0C | 0x14 | 0x1C | 0x24 | 0x2C | 0x34 | 0x3C |
            0x80 | 0x84 | 0x86 | 0x88 | 0x8A |
            0xC0 | 0xC6 | 0xD0 | 0xD2 |
            0xF6 |
            0x0FB0 | 0x0FC0 => {
                opcode += 1;
                opsize = Size::Byte;
            }
            0xB0 => {
                opcode += 8;
                opsize = Size::Byte;
            }
            _ => (),
        }

        match opcode {
            0x0F0B => Op::Illegal,
            0x0F40..=0x0F4F => {
                let cc = (opcode as u8 & 0xF).try_into().unwrap();
                let (src, dst) = self.modrm(rex, opsize);
                Op::Cmovcc(dst, src, cc)
            }
            0x0F80..=0x0F8F => {
                let cc = (opcode as u8 & 0xF).try_into().unwrap();
                Op::Jcc(self.dword() as i32, cc)
            }
            0x0F90..=0x0F9F => {
                let cc = (opcode as u8 & 0xF).try_into().unwrap();
                let (dst, reg) = self.modrm(rex, Size::Byte);
                assert!(reg as u8 & 7 == 0);
                Op::Setcc(dst, cc)
            }
            0x0FAE => {
                let next = self.byte();
                assert_eq!(next, 0xF0);
                Op::Mfence
            }
            0x0FAF => {
                let (src, dst) = self.modrm(rex, opsize);
                Op::Imul2(dst, src)
            }
            0x0FB1 => {
                let (dst, src) = self.modrm(rex, opsize);
                Op::Cmpxchg(dst, src)
            }
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
            0x0FC1 => {
                let (dst, src) = self.modrm(rex, opsize);
                Op::Xadd(dst, src)
            }
            // These are all INST r/m, r ALU ops
            0x01 | 0x09 | 0x11 | 0x19 | 0x21 | 0x29 | 0x31 | 0x39 => {
                let (dst, src) = self.modrm(rex, opsize);
                Self::decode_alu(dst, OpReg(src), (opcode as u8) >> 3)
            }
            // These are all INST r, r/m ALU ops
            0x03 | 0x0B | 0x13 | 0x1B | 0x23 | 0x2B | 0x33 | 0x3B => {
                let (src, dst) = self.modrm(rex, opsize);
                Self::decode_alu(Reg(dst), src.into(), (opcode as u8) >> 3)
            }
            // These are all RAX short encoded ALU ops
            0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 | 0x3D => {
                let imm = self.immediate(opsize.cap_to_dword());
                Self::decode_alu(
                    Reg(Self::register_of_size(0, opsize, rex)),
                    Imm(imm),
                    (opcode as u8) >> 3,
                )
            }
            0x50..=0x57 => {
                let reg_id = if rex & 0x1 != 0 { 8 } else { 0 } | opcode as u8 & 7;
                let reg = Self::register_of_size(
                    reg_id,
                    if opsize == Size::Dword { Size::Qword } else { opsize },
                    rex,
                );
                Op::Push(reg.into())
            }
            0x58..=0x5F => {
                let reg_id = if rex & 0x1 != 0 { 8 } else { 0 } | opcode as u8 & 7;
                let reg = Self::register_of_size(
                    reg_id,
                    if opsize == Size::Dword { Size::Qword } else { opsize },
                    rex,
                );
                Op::Pop(reg.into())
            }
            0x63 => {
                let (src, dst) = self.modrm(rex, opsize);
                Op::Movsx(dst, src.resize(Size::Dword))
            }
            0x70..=0x7F => {
                let cc = (opcode as u8 & 0xF).try_into().unwrap();
                Op::Jcc(self.byte() as i8 as i32, cc)
            }
            0x81 => {
                let (operand, reg) = self.modrm(rex, opsize);
                let imm = self.immediate(opsize.cap_to_dword());
                Self::decode_alu(operand, Imm(imm), reg as u8)
            }
            0x83 => {
                let (operand, reg) = self.modrm(rex, opsize);
                Self::decode_alu(operand, Imm(self.byte() as i8 as i64), reg as u8)
            }
            0x85 => {
                let (operand, reg) = self.modrm(rex, opsize);
                Op::Test(operand, OpReg(reg))
            }
            0x87 => {
                let (operand, reg) = self.modrm(rex, opsize);
                Op::Xchg(Reg(reg), operand)
            }
            0x89 => {
                let (operand, reg) = self.modrm(rex, opsize);
                Op::Mov(operand, OpReg(reg))
            }
            0x8B => {
                let (operand, reg) = self.modrm(rex, opsize);
                Op::Mov(reg.into(), operand.into())
            }
            0x8D => {
                if let (Mem(operand), reg) = self.modrm(rex, opsize) {
                    Op::Lea(reg, operand)
                } else {
                    // LEA with register src is illegal
                    Op::Illegal
                }
            }
            0x90 => Op::Nop,
            0x98 => match opsize {
                Size::Qword => Op::Cdqe,
                _ => unimplemented!(),
            },
            0x99 => match opsize {
                Size::Dword => Op::Cdq,
                Size::Qword => Op::Cqo,
                _ => unimplemented!(),
            },
            0xB8..=0xBF => {
                let reg_id = if rex & 0x1 != 0 { 8 } else { 0 } | opcode as u8 & 7;
                let reg = Self::register_of_size(reg_id, opsize, rex);
                let imm = self.immediate(opsize);
                Op::Mov(Reg(reg), Imm(imm))
            }
            0xC1 => {
                let (operand, reg) = self.modrm(rex, opsize);
                Self::decode_shift(operand, Imm(self.byte() as i64), reg)
            }
            0xC2 => Op::Ret(self.word()),
            0xC3 => Op::Ret(0),
            0xC7 => {
                let (operand, reg) = self.modrm(rex, opsize);
                if reg as u8 & 7 != 0 {
                    unimplemented!()
                }
                Op::Mov(operand, Imm(self.immediate(opsize.cap_to_dword())))
            }
            0xD1 => {
                let (operand, reg) = self.modrm(rex, opsize);
                Self::decode_shift(operand, Imm(1), reg)
            }
            0xD3 => {
                let (operand, reg) = self.modrm(rex, opsize);
                Self::decode_shift(operand, OpReg(Register::CL), reg)
            }
            0xE8 => Op::Call(Imm(self.dword() as i32 as i64)),
            0xE9 => Op::Jmp(Imm(self.dword() as i32 as i64)),
            0xEB => Op::Jmp(Imm(self.byte() as i8 as i64)),
            0xF0 => Op::Lock,
            0xF4 => Op::Hlt,
            0xF7 => {
                let (operand, reg) = self.modrm(rex, opsize);
                match reg as u8 & 7 {
                    0 => Op::Test(operand, Imm(self.immediate(opsize.cap_to_dword()))),
                    2 => Op::Not(operand),
                    3 => Op::Neg(operand),
                    4 => Op::Mul(operand),
                    5 => Op::Imul1(operand),
                    6 => Op::Div(operand),
                    7 => Op::Idiv(operand),
                    _ => unimplemented!(),
                }
            }
            0xFF => {
                let (operand, reg) = self.modrm(rex, Size::Qword);
                match reg as u8 & 7 {
                    0 => Op::Push(operand.into()),
                    2 => Op::Call(operand.into()),
                    4 => Op::Jmp(operand.into()),
                    _ => unimplemented!(),
                }
            }
            _ => unimplemented!("opcode {:x}", opcode),
        }
    }
}

pub fn decode(iter: &mut dyn FnMut() -> u8) -> Op {
    let mut decoder = Decoder { iter };
    decoder.op()
}
