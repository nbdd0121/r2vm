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
    ((bits & 0b11111110_00000000_00000000_00000000) as i32) >> 20 |
    ((bits & 0b00000000_00000000_00001111_10000000) as i32) >> 7
}

fn b_imm(bits: u32) -> i32 {
    ((bits & 0b10000000_00000000_00000000_00000000) as i32) >> 19 |
    ((bits & 0b00000000_00000000_00000000_10000000) as i32) << 4 |
    ((bits & 0b01111110_00000000_00000000_00000000) as i32) >> 20 |
    ((bits & 0b00000000_00000000_00001111_00000000) as i32) >> 7
}

fn u_imm(bits: u32) -> i32 {
    (bits & 0xfffff000) as i32
}

fn j_imm(instr: u32) -> i32 {
    ((instr & 0b10000000_00000000_00000000_00000000) as i32) >> 11 |
    ((instr & 0b00000000_00001111_11110000_00000000) as i32) >> 0 |
    ((instr & 0b00000000_00010000_00000000_00000000) as i32) >> 9 |
    ((instr & 0b01111111_11100000_00000000_00000000) as i32) >> 20
}

//
// #endregion

// #region: decoding helpers for compressed 16-bit instructions
//

fn c_funct3(bits: u16) -> u32 {
    ((bits >> 13) & 0b111) as u32
}

fn c_rd(bits: u16) -> u8 {
    ((bits >> 7) & 0b11111) as u8
}

fn c_rs1(bits: u16) -> u8 { c_rd(bits) }

fn c_rs2(bits: u16) -> u8 {
    ((bits >> 2) & 0b11111) as u8
}

fn c_rds(bits: u16) -> u8 {
    ((bits >> 2) & 0b111) as u8 + 8
}

fn c_rs1s(bits: u16) -> u8 {
    ((bits >> 7) & 0b111) as u8 + 8
}

fn c_rs2s(bits: u16) -> u8 { c_rds(bits) }

fn ci_imm(bits: u16) -> i32 {
    ((bits & 0b00010000_00000000) as i32) << (31 - 12) >> (31 - 5) |
    ((bits & 0b00000000_01111100) as i32) >> 2
}

fn ci_addi16sp_imm(bits: u16) -> i32 {
    ((bits & 0b00010000_00000000) as i32) << (31 - 12) >> (31 - 9) |
    ((bits & 0b00000000_00011000) as i32) << 4 |
    ((bits & 0b00000000_00100000) as i32) << 1 |
    ((bits & 0b00000000_00000100) as i32) << 3 |
    ((bits & 0b00000000_01000000) as i32) >> 2
}

fn ciw_imm(bits: u16) -> i32 {
    ((bits & 0b00000111_10000000) as i32) >> 1 |
    ((bits & 0b00011000_00000000) as i32) >> 7 |
    ((bits & 0b00000000_00100000) as i32) >> 2 |
    ((bits & 0b00000000_01000000) as i32) >> 4
}

fn cb_imm(bits: u16) -> i32 {
    ((bits & 0b00010000_00000000) as i32) << (31 - 12) >> (31 - 8) |
    ((bits & 0b00000000_01100000) as i32) << 1 |
    ((bits & 0b00000000_00000100) as i32) << 3 |
    ((bits & 0b00001100_00000000) as i32) >> 7 |
    ((bits & 0b00000000_00011000) as i32) >> 2
}

fn cj_imm(bits: u16) -> i32 {
    ((bits & 0b00010000_00000000) as i32) << (31 - 12) >> (31 - 11) |
    ((bits & 0b00000001_00000000) as i32) << 2 |
    ((bits & 0b00000110_00000000) as i32) >> 1 |
    ((bits & 0b00000000_01000000) as i32) << 1 |
    ((bits & 0b00000000_10000000) as i32) >> 1 |
    ((bits & 0b00000000_00000100) as i32) << 3 |
    ((bits & 0b00001000_00000000) as i32) >> 7 |
    ((bits & 0b00000000_00111000) as i32) >> 2
}

//
// #endregion

pub fn decode_compressed(bits: u16) -> Op {
    let function = c_funct3(bits);

    match bits & 0b11 {
        0b00 => {
            match function {
                0b000 => {
                    let imm = ciw_imm(bits);
                    if imm == 0 {
                        // Illegal instruction
                        return Op::Illegal
                    }
                    // C.ADDI4SPN
                    // translate to addi rd', x2, imm
                    Op::Addi { rd: c_rds(bits), rs1: 2, imm }
                }
                _ => Op::Illegal,
            }
        }
        0b01 => {
            match function {
                0b000 => {
                    // rd = x0 is HINT
                    // r0 = 0 is C.NOP
                    // C.ADDI
                    // translate to addi rd, rd, imm
                    let rd = c_rd(bits);
                    Op::Addi { rd, rs1: rd, imm: ci_imm(bits) }
                }
                0b001 => {
                    let rd = c_rd(bits);
                    if rd == 0 {
                        // Reserved
                        return Op::Illegal
                    }
                    // C.ADDIW
                    // translate to addiw rd, rd, imm
                    Op::Addiw { rd, rs1: rd, imm: ci_imm(bits) }
                }
                0b010 => {
                    // rd = x0 is HINT
                    // C.LI
                    // translate to addi rd, x0, imm
                    Op::Addi { rd: c_rd(bits), rs1: 0, imm: ci_imm(bits) }
                }
                0b011 => {
                    let rd = c_rd(bits);
                    if rd == 2 {
                        let imm = ci_addi16sp_imm(bits);
                        if imm == 0 {
                            // Reserved
                            return Op::Illegal
                        }
                        // C.ADDI16SP
                        // translate to addi x2, x2, imm
                        Op::Addi { rd: 2, rs1: 2, imm }
                    } else {
                        // rd = x0 is HINT
                        // C.LUI
                        // translate to lui rd, imm
                        Op::Lui { rd, imm: ci_imm(bits) << 12 }
                    }
                }
                0b100 => {
                    let rs1 = c_rs1s(bits);
                    match (bits >> 10) & 0b11 {
                        0b00 => {
                            // imm = 0 is HINT
                            // C.SRLI
                            // translate to srli rs1', rs1', imm
                            Op::Srli { rd: rs1, rs1, imm: ci_imm(bits) & 63 }
                        }
                        0b01 => {
                            // imm = 0 is HINT
                            // C.SRAI
                            // translate to srai rs1', rs1', imm
                            Op::Srai { rd: rs1, rs1, imm: ci_imm(bits) & 63 }
                        }
                        0b10 => {
                            // C.ANDI
                            // translate to andi rs1', rs1', imm
                            Op::Andi { rd: rs1, rs1, imm: ci_imm(bits) }
                        }
                        0b11 => if (bits & 0x1000) == 0 {
                            // C.SUB
                            // C.XOR
                            // C.OR
                            // C.AND
                            // translates to [OP] rs1', rs1', rs2'
                            let rs2 = c_rs2s(bits);
                            match (bits >> 5) & 0b11 {
                                0b00 => Op::Sub { rd: rs1, rs1, rs2 },
                                0b01 => Op::Xor { rd: rs1, rs1, rs2 },
                                0b10 => Op::Or { rd: rs1, rs1, rs2 },
                                0b11 => Op::And { rd: rs1, rs1, rs2 },
                                // full case
                                _ => unsafe { std::hint::unreachable_unchecked() },
                            }
                        } else {
                            Op::Illegal
                        }
                        // full case
                        _ => unsafe { std::hint::unreachable_unchecked() },
                    }
                }
                0b101 => {
                    // C.J
                    // translate to jal x0, imm
                    Op::Jal { rd: 0, imm: cj_imm(bits) }
                }
                0b110 => {
                    // C.BEQZ
                    // translate to beq rs1', x0, imm
                    Op::Beq { rs1: c_rs1s(bits), rs2: 0, imm: cb_imm(bits) }
                }
                0b111 => {
                    // C.BNEZ
                    // translate to bne rs1', x0, imm
                    Op::Bne { rs1: c_rs1s(bits), rs2: 0, imm: cb_imm(bits) }
                }
                // full case
                _ => unsafe { std::hint::unreachable_unchecked() },
            }
        }
        0b10 => {
            match function {
                0b000 => {
                    // imm = 0 is HINT
                    // rd = 0 is HINT
                    // C.SLLI
                    // translates to slli rd, rd, imm
                    let rd = c_rd(bits);
                    Op::Slli { rd, rs1: rd, imm: ci_imm(bits) & 63 }
                }
                0b100 => {
                    let rs2 = c_rs2(bits);
                    if (bits & 0x1000) == 0 {
                        if rs2 == 0 {
                            let rs1 = c_rs1(bits);
                            if rs1 == 0 {
                                // Reserved
                                return Op::Illegal
                            }
                            // C.JR
                            // translate to jalr x0, rs1, 0
                            Op::Jalr { rd: 0, rs1, imm: 0 }
                        } else {
                            // rd = 0 is HINT
                            // C.MV
                            // translate to add rd, x0, rs2
                            Op::Add { rd: c_rd(bits), rs1: 0, rs2 }
                        }
                    } else {
                        let rs1 = c_rs1(bits);
                        if rs1 == 0 {
                            // C.EBREAK
                            Op::Ebreak
                        } else if rs2 == 0 {
                            // C.JALR
                            // translate to jalr x1, rs1, 0
                            Op::Jalr { rd: 1, rs1, imm: 0 }
                        } else {
                            // rd = 0 is HINT
                            // C.ADD
                            // translate to add rd, rd, rs2
                            let rd = c_rd(bits);
                            Op::Add { rd, rs1: rd, rs2 }
                        }
                    }
                }
                _ => Op::Illegal
            }
        }
        _ => unreachable!(),
    }
}

pub fn decode(bits: u32) -> Op {
    // We shouldn't see compressed ops here
    assert!(bits & 3 == 3);

    // Longer ops, treat them as illegal ops
    if bits & 0x1f == 0x1f { return Op::Illegal }

    let function = funct3(bits);
    let rd = rd(bits);
    let rs1 = rs1(bits);
    let rs2 = rs2(bits);

    match bits & 0b1111111 {
        /* OP-IMM */
        0b0010011 => {
            let imm = i_imm(bits);
            match function {
                0b000 => Op::Addi { rd, rs1, imm },
                0b001 =>
                    if imm >= 64 {
                        Op::Illegal
                    } else {
                        Op::Slli { rd, rs1, imm }
                    }
                0b010 => Op::Slti { rd, rs1, imm },
                0b011 => Op::Sltiu { rd, rs1, imm },
                0b100 => Op::Xori { rd, rs1, imm },
                0b101 =>
                    if imm &! 0x400 >= 64 {
                        Op::Illegal
                    } else if (imm & 0x400) != 0 {
                        Op::Srai { rd, rs1, imm: imm &! 0x400 }
                    } else {
                        Op::Srli { rd, rs1, imm }
                    }
                0b110 => Op::Ori { rd, rs1, imm },
                0b111 => Op::Andi { rd, rs1, imm },
                _ => unsafe { std::hint::unreachable_unchecked() }
            }
        }

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

        /* OP-IMM-32 */
        0b0011011 => {
            let imm = i_imm(bits);
            match function {
                0b000 => Op::Addiw { rd, rs1, imm },
                0b001 =>
                    if imm >= 32 {
                        Op::Illegal
                    } else {
                        Op::Slliw { rd, rs1, imm }
                    }
                0b101 =>
                    if imm &! 0x400 >= 32 {
                        Op::Illegal
                    } else if (imm & 0x400) != 0 {
                        Op::Sraiw { rd, rs1, imm: imm &! 0x400 }
                    } else {
                        Op::Srliw { rd, rs1, imm }
                    }
                _ => Op::Illegal,
            }
        }

        /* OP */
        0b0110011 => {
            match funct7(bits) {
                // M-extension
                0b0000001 => {
                    return Op::Legacy(unsafe { legacy_decode(bits) })
                }
                0b0000000 => match function {
                    0b000 => Op::Add { rd, rs1, rs2 },
                    0b001 => Op::Sll { rd, rs1, rs2 },
                    0b010 => Op::Slt { rd, rs1, rs2 },
                    0b011 => Op::Sltu { rd, rs1, rs2 },
                    0b100 => Op::Xor { rd, rs1, rs2 },
                    0b101 => Op::Srl { rd, rs1, rs2 },
                    0b110 => Op::Or { rd, rs1, rs2 },
                    0b111 => Op::And { rd, rs1, rs2 },
                    _ => unsafe { std::hint::unreachable_unchecked() },
                }
                0b0100000 => match function {
                    0b000 => Op::Sub { rd, rs1, rs2 },
                    0b101 => Op::Sra { rd, rs1, rs2 },
                    _ => Op::Illegal
                }
                _ => Op::Illegal
            }
        }

        /* LUI */
        0b0110111 => Op::Lui { rd, imm: u_imm(bits) },

        /* AUIPC */
        0b0010111 => Op::Auipc { rd, imm: u_imm(bits) },

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
        (decode(bits), false)
    } else {
        let op = decode_compressed(bits as u16);
        if let Op::Illegal = op {
            return (Op::Legacy(unsafe { legacy_decode(bits) }), true)
        }
        (op, true)
    }
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
