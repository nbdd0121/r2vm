use super::op::{LegacyOp, Op};
use super::Csr;

extern {
    fn legacy_decode(inst: u32) -> LegacyOp;
}

// #region: decoding helpers for 32-bit instructions
//

fn rd(bits: u32) -> u8 {
    ((bits >> 7) & 0b11111) as u8
}

fn rs1(bits: u32) -> u8 {
    ((bits >> 15) & 0b11111) as u8
}

fn rs2(bits: u32) -> u8 {
    ((bits >> 20) & 0b11111) as u8
}

fn rs3(bits: u32) -> u8 {
    ((bits >> 27) & 0b11111) as u8
}

fn funct3(bits: u32) -> u32 {
    (bits >> 12) & 0b111
}

fn funct7(bits: u32) -> u32 {
    (bits >> 25) & 0b1111111
}

fn csr(bits: u32) -> Csr {
    ((bits >> 20) as u16).into()
}

fn i_imm(bits: u32) -> i32 {
    ((bits as i32) >> 20)
}

fn s_imm(bits: u32) -> i32 {
    (((bits & 0b11111110_00000000_00000000_00000000) as i32) >> 20) |
    (((bits & 0b00000000_00000000_00001111_10000000) as i32) >> 7)
}

fn b_imm(bits: u32) -> i32 {
    (((bits & 0b10000000_00000000_00000000_00000000) as i32) >> 19) |
    (((bits & 0b00000000_00000000_00000000_10000000) as i32) << 4) |
    (((bits & 0b01111110_00000000_00000000_00000000) as i32) >> 20) |
    (((bits & 0b00000000_00000000_00001111_00000000) as i32) >> 7)
}

fn u_imm(bits: u32) -> i32 {
    (bits & 0xfffff000) as i32
}

fn j_imm(instr: u32) -> i32 {
    (((instr & 0b10000000_00000000_00000000_00000000) as i32) >> 11) |
    (((instr & 0b00000000_00001111_11110000_00000000) as i32) >> 0) |
    (((instr & 0b00000000_00010000_00000000_00000000) as i32) >> 9) |
    (((instr & 0b01111111_11100000_00000000_00000000) as i32) >> 20)
}

//
// #endregion

pub fn decode(bits: u32) -> Op {
    // Compressed ops
    if bits & 3 != 3 {
        return Op::Legacy(unsafe { legacy_decode(bits) })
    }

    // Longer ops, treat them as illegal ops
    if bits & 0x1f == 0x1f { return Op::Illegal }

    let function = funct3(bits);
    let rd = rd(bits);
    let rs1 = rs1(bits);
    let rs2 = rs2(bits);

    match bits & 0b1111111 {
        /* MISC-MEM */
        0b0001111 => {
            match function {
                0b000 => {
                    // TODO Multiple types of fence
                    Op::Fence
                }
                0b001 => Op::FenceI,
                _ => Op::Illegal,
            }
        }

        /* AUIPC */
        0b0010111 => Op::Auipc {
            rd: rd,
            imm: u_imm(bits),
        },

        /* BRANCH */
        0b1100011 => {
            let imm = b_imm(bits);
            match function {
                0b000 => Op::Beq { rs1, rs2, imm },
                0b001 => Op::Bne { rs1, rs2, imm },
                0b100 => Op::Blt { rs1, rs2, imm },
                0b101 => Op::Bge { rs1, rs2, imm },
                0b110 => Op::Bltu { rs1, rs2, imm },
                0b111 => Op::Bgeu { rs1, rs2, imm },
                _ => Op::Illegal,
            }
        }

        /* JALR */
        0b1100111 => Op::Jalr { rd, rs1, imm: i_imm(bits) },

        /* JAL */
        0b1101111 => Op::Jal { rd, imm: j_imm(bits) },

        /* SYSTEM */
        0b1110011 => {
            // CSR is encoded in the same place as I-imm.
            let csr = csr(bits);

            match function {
                0b000 => {
                    match bits {
                        0x73 => Op::Ecall,
                        0x100073 => Op::Ebreak,
                        0x10200073 => Op::Sret,
                        0x10500073 => Op::Wfi,
                        bits if rd == 0 && funct7(bits) == 0b0001001 => Op::SfenceVma { rs1, rs2 },
                        _ => Op::Illegal,
                    }
                }
                0b001 => Op::Csrrw { rd, rs1, csr },
                0b010 => Op::Csrrs { rd, rs1, csr },
                0b011 => Op::Csrrc { rd, rs1, csr },
                0b101 => Op::Csrrwi { rd, imm: rs1, csr },
                0b110 => Op::Csrrsi { rd, imm: rs1, csr },
                0b111 => Op::Csrrci { rd, imm: rs1, csr },
                _ => Op::Illegal,
            }
        }
        _ => Op::Legacy(unsafe { legacy_decode(bits) }),
    }
}

pub fn decode_instr(pc: &mut u64, pc_next: u64) -> (Op, bool) {
    let mut bits = unsafe { crate::emu::read_memory::<u16>(*pc) as u32 };
    *pc += 2;
    if bits & 3 == 3 {
        if *pc & 4095 == 0 {
            bits |= (unsafe { crate::emu::read_memory::<u16>(pc_next) as u32 }) << 16;
        } else {
            bits |= (unsafe { crate::emu::read_memory::<u16>(*pc) as u32 }) << 16;
        }
        *pc += 2;
    }
    (decode(bits), bits & 3 != 3)
}

pub fn decode_block(mut pc: u64, pc_next: u64) -> (Vec<(Op, bool)>, u64, u64) {
    let start_pc = pc;
    let mut vec = Vec::new();
    loop {
        let (op, c) = decode_instr(&mut pc, pc_next);
        if op.can_change_control_flow() || (pc &! 4095) != (start_pc &! 4095) {
            vec.push((op, c));
            break
        }
        vec.push((op, c));
    }
    (vec, start_pc, pc)
}
