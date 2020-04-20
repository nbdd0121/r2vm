use super::op::{Op, Ordering};

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

fn csr(bits: u32) -> u16 {
    (bits >> 20) as u16
}

fn i_imm(bits: u32) -> i32 {
    (bits as i32) >> 20
}

fn s_imm(bits: u32) -> i32 {
    ((bits & 0b11111110_00000000_00000000_00000000) as i32) >> 20
        | ((bits & 0b00000000_00000000_00001111_10000000) as i32) >> 7
}

fn b_imm(bits: u32) -> i32 {
    ((bits & 0b10000000_00000000_00000000_00000000) as i32) >> 19
        | ((bits & 0b00000000_00000000_00000000_10000000) as i32) << 4
        | ((bits & 0b01111110_00000000_00000000_00000000) as i32) >> 20
        | ((bits & 0b00000000_00000000_00001111_00000000) as i32) >> 7
}

fn u_imm(bits: u32) -> i32 {
    (bits & 0xfffff000) as i32
}

fn j_imm(instr: u32) -> i32 {
    ((instr & 0b10000000_00000000_00000000_00000000) as i32) >> 11
        | ((instr & 0b00000000_00001111_11110000_00000000) as i32) >> 0
        | ((instr & 0b00000000_00010000_00000000_00000000) as i32) >> 9
        | ((instr & 0b01111111_11100000_00000000_00000000) as i32) >> 20
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

fn c_rs1(bits: u16) -> u8 {
    c_rd(bits)
}

fn c_rs2(bits: u16) -> u8 {
    ((bits >> 2) & 0b11111) as u8
}

fn c_rds(bits: u16) -> u8 {
    ((bits >> 2) & 0b111) as u8 + 8
}

fn c_rs1s(bits: u16) -> u8 {
    ((bits >> 7) & 0b111) as u8 + 8
}

fn c_rs2s(bits: u16) -> u8 {
    c_rds(bits)
}

fn ci_imm(bits: u16) -> i32 {
    ((bits & 0b00010000_00000000) as i32) << (31 - 12) >> (31 - 5)
        | ((bits & 0b00000000_01111100) as i32) >> 2
}

fn ci_lwsp_imm(bits: u16) -> i32 {
    ((bits & 0b00000000_00001100) as i32) << 4
        | ((bits & 0b00010000_00000000) as i32) >> 7
        | ((bits & 0b00000000_01110000) as i32) >> 2
}

fn ci_ldsp_imm(bits: u16) -> i32 {
    ((bits & 0b00000000_00011100) as i32) << 4
        | ((bits & 0b00010000_00000000) as i32) >> 7
        | ((bits & 0b00000000_01100000) as i32) >> 2
}

fn ci_addi16sp_imm(bits: u16) -> i32 {
    ((bits & 0b00010000_00000000) as i32) << (31 - 12) >> (31 - 9)
        | ((bits & 0b00000000_00011000) as i32) << 4
        | ((bits & 0b00000000_00100000) as i32) << 1
        | ((bits & 0b00000000_00000100) as i32) << 3
        | ((bits & 0b00000000_01000000) as i32) >> 2
}

fn css_swsp_imm(bits: u16) -> i32 {
    ((bits & 0b00000001_10000000) as i32) >> 1 | ((bits & 0b00011110_00000000) as i32) >> 7
}

fn css_sdsp_imm(bits: u16) -> i32 {
    ((bits & 0b00000011_10000000) as i32) >> 1 | ((bits & 0b00011100_00000000) as i32) >> 7
}

fn ciw_imm(bits: u16) -> i32 {
    ((bits & 0b00000111_10000000) as i32) >> 1
        | ((bits & 0b00011000_00000000) as i32) >> 7
        | ((bits & 0b00000000_00100000) as i32) >> 2
        | ((bits & 0b00000000_01000000) as i32) >> 4
}

fn cl_lw_imm(bits: u16) -> i32 {
    ((bits & 0b00000000_00100000) as i32) << 1
        | ((bits & 0b00011100_00000000) as i32) >> 7
        | ((bits & 0b00000000_01000000) as i32) >> 4
}

fn cl_ld_imm(bits: u16) -> i32 {
    ((bits & 0b00000000_01100000) as i32) << 1 | ((bits & 0b00011100_00000000) as i32) >> 7
}

fn cs_sw_imm(bits: u16) -> i32 {
    cl_lw_imm(bits)
}

fn cs_sd_imm(bits: u16) -> i32 {
    cl_ld_imm(bits)
}

fn cb_imm(bits: u16) -> i32 {
    ((bits & 0b00010000_00000000) as i32) << (31 - 12) >> (31 - 8)
        | ((bits & 0b00000000_01100000) as i32) << 1
        | ((bits & 0b00000000_00000100) as i32) << 3
        | ((bits & 0b00001100_00000000) as i32) >> 7
        | ((bits & 0b00000000_00011000) as i32) >> 2
}

fn cj_imm(bits: u16) -> i32 {
    ((bits & 0b00010000_00000000) as i32) << (31 - 12) >> (31 - 11)
        | ((bits & 0b00000001_00000000) as i32) << 2
        | ((bits & 0b00000110_00000000) as i32) >> 1
        | ((bits & 0b00000000_01000000) as i32) << 1
        | ((bits & 0b00000000_10000000) as i32) >> 1
        | ((bits & 0b00000000_00000100) as i32) << 3
        | ((bits & 0b00001000_00000000) as i32) >> 7
        | ((bits & 0b00000000_00111000) as i32) >> 2
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
                        return Op::Illegal;
                    }
                    // C.ADDI4SPN
                    // translate to addi rd', x2, imm
                    Op::Addi { rd: c_rds(bits), rs1: 2, imm }
                }
                0b001 => {
                    // C.FLD
                    // translate to fld rd', rs1', offset
                    Op::Fld { frd: c_rds(bits), rs1: c_rs1s(bits), imm: cl_ld_imm(bits) }
                }
                0b010 => {
                    // C.LW
                    // translate to lw rd', rs1', offset
                    Op::Lw { rd: c_rds(bits), rs1: c_rs1s(bits), imm: cl_lw_imm(bits) }
                }
                0b011 => {
                    // C.LD
                    // translate to ld rd', rs1', offset
                    Op::Ld { rd: c_rds(bits), rs1: c_rs1s(bits), imm: cl_ld_imm(bits) }
                }
                0b100 => {
                    // Reserved
                    Op::Illegal
                }
                0b101 => {
                    // C.FSD
                    // translate to fsd rs2', rs1', offset
                    Op::Fsd { rs1: c_rs1s(bits), frs2: c_rs2s(bits), imm: cs_sd_imm(bits) }
                }
                0b110 => {
                    // C.SW
                    // translate to sw rs2', rs1', offset
                    Op::Sw { rs1: c_rs1s(bits), rs2: c_rs2s(bits), imm: cs_sw_imm(bits) }
                }
                0b111 => {
                    // C.SD
                    // translate to sd rs2', rs1', offset
                    Op::Sd { rs1: c_rs1s(bits), rs2: c_rs2s(bits), imm: cs_sd_imm(bits) }
                }
                // full case
                _ => unreachable!(),
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
                        return Op::Illegal;
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
                            return Op::Illegal;
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
                        0b11 => {
                            if (bits & 0x1000) == 0 {
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
                                    _ => unreachable!(),
                                }
                            } else {
                                // C.SUBW
                                // C.ADDW
                                let rs2 = c_rs2s(bits);
                                match (bits >> 5) & 0b11 {
                                    0b00 => Op::Subw { rd: rs1, rs1, rs2 },
                                    0b01 => Op::Addw { rd: rs1, rs1, rs2 },
                                    _ => Op::Illegal,
                                }
                            }
                        }
                        // full case
                        _ => unreachable!(),
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
                _ => unreachable!(),
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
                0b001 => {
                    // C.FLDSP
                    // translate to fld rd, x2, imm
                    Op::Fld { frd: c_rd(bits), rs1: 2, imm: ci_ldsp_imm(bits) }
                }
                0b010 => {
                    let rd = c_rd(bits);
                    if rd == 0 {
                        // Reserved
                        return Op::Illegal;
                    }
                    // C.LWSP
                    // translate to lw rd, x2, imm
                    Op::Lw { rd, rs1: 2, imm: ci_lwsp_imm(bits) }
                }
                0b011 => {
                    let rd = c_rd(bits);
                    if rd == 0 {
                        // Reserved
                        return Op::Illegal;
                    }
                    // C.LDSP
                    // translate to ld rd, x2, imm
                    Op::Ld { rd, rs1: 2, imm: ci_ldsp_imm(bits) }
                }
                0b100 => {
                    let rs2 = c_rs2(bits);
                    if (bits & 0x1000) == 0 {
                        if rs2 == 0 {
                            let rs1 = c_rs1(bits);
                            if rs1 == 0 {
                                // Reserved
                                return Op::Illegal;
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
                0b101 => {
                    // C.FSDSP
                    // translate to fsd rs2, x2, imm
                    Op::Fsd { rs1: 2, frs2: c_rs2(bits), imm: css_sdsp_imm(bits) }
                }
                0b110 => {
                    // C.SWSP
                    // translate to sw rs2, x2, imm
                    Op::Sw { rs1: 2, rs2: c_rs2(bits), imm: css_swsp_imm(bits) }
                }
                0b111 => {
                    // C.SDSP
                    // translate to sd rs2, x2, imm
                    Op::Sd { rs1: 2, rs2: c_rs2(bits), imm: css_sdsp_imm(bits) }
                }
                // full case
                _ => unreachable!(),
            }
        }
        _ => unreachable!(),
    }
}

pub fn decode(bits: u32) -> Op {
    macro_rules! rm {
        ($rm: expr) => {{
            let rm = $rm as u8;
            if rm > 4 && rm != 0b111 {
                return Op::Illegal;
            }
            rm
        }};
    }

    // We shouldn't see compressed ops here
    assert!(bits & 3 == 3);

    // Longer ops, treat them as illegal ops
    if bits & 0x1f == 0x1f {
        return Op::Illegal;
    }

    let function = funct3(bits);
    let rd = rd(bits);
    let rs1 = rs1(bits);
    let rs2 = rs2(bits);

    match bits & 0b1111111 {
        /* LOAD */
        0b0000011 => {
            let imm = i_imm(bits);
            match function {
                0b000 => Op::Lb { rd, rs1, imm },
                0b001 => Op::Lh { rd, rs1, imm },
                0b010 => Op::Lw { rd, rs1, imm },
                0b011 => Op::Ld { rd, rs1, imm },
                0b100 => Op::Lbu { rd, rs1, imm },
                0b101 => Op::Lhu { rd, rs1, imm },
                0b110 => Op::Lwu { rd, rs1, imm },
                _ => Op::Illegal,
            }
        }

        /* LOAD-FP */
        0b0000111 => {
            let imm = i_imm(bits);
            match function {
                0b010 => Op::Flw { frd: rd, rs1, imm },
                0b011 => Op::Fld { frd: rd, rs1, imm },
                _ => Op::Illegal,
            }
        }

        /* OP-IMM */
        0b0010011 => {
            let imm = i_imm(bits);
            match function {
                0b000 => Op::Addi { rd, rs1, imm },
                0b001 => {
                    if imm >= 64 {
                        Op::Illegal
                    } else {
                        Op::Slli { rd, rs1, imm }
                    }
                }
                0b010 => Op::Slti { rd, rs1, imm },
                0b011 => Op::Sltiu { rd, rs1, imm },
                0b100 => Op::Xori { rd, rs1, imm },
                0b101 => {
                    if imm & !0x400 >= 64 {
                        Op::Illegal
                    } else if (imm & 0x400) != 0 {
                        Op::Srai { rd, rs1, imm: imm & !0x400 }
                    } else {
                        Op::Srli { rd, rs1, imm }
                    }
                }
                0b110 => Op::Ori { rd, rs1, imm },
                0b111 => Op::Andi { rd, rs1, imm },
                // full case
                _ => unreachable!(),
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
                0b001 => {
                    if imm >= 32 {
                        Op::Illegal
                    } else {
                        Op::Slliw { rd, rs1, imm }
                    }
                }
                0b101 => {
                    if imm & !0x400 >= 32 {
                        Op::Illegal
                    } else if (imm & 0x400) != 0 {
                        Op::Sraiw { rd, rs1, imm: imm & !0x400 }
                    } else {
                        Op::Srliw { rd, rs1, imm }
                    }
                }
                _ => Op::Illegal,
            }
        }

        /* STORE */
        0b0100011 => {
            let imm = s_imm(bits);
            match function {
                0b000 => Op::Sb { rs1, rs2, imm },
                0b001 => Op::Sh { rs1, rs2, imm },
                0b010 => Op::Sw { rs1, rs2, imm },
                0b011 => Op::Sd { rs1, rs2, imm },
                _ => Op::Illegal,
            }
        }

        /* STORE-FP */
        0b0100111 => {
            let imm = s_imm(bits);
            match function {
                0b010 => Op::Fsw { rs1, frs2: rs2, imm },
                0b011 => Op::Fsd { rs1, frs2: rs2, imm },
                _ => Op::Illegal,
            }
        }

        /* Base Opcode AMO */
        0b0101111 => {
            /* A-Extension */
            let func = funct7(bits) >> 2;
            let aqrl = match funct7(bits) & 3 {
                0 => Ordering::Relaxed,
                1 => Ordering::Release,
                2 => Ordering::Acquire,
                3 => Ordering::SeqCst,
                _ => unreachable!(),
            };
            if function == 0b010 {
                match func {
                    0b00010 => {
                        if rs2 != 0 {
                            Op::Illegal
                        } else {
                            Op::LrW { rd, rs1, aqrl }
                        }
                    }
                    0b00011 => Op::ScW { rd, rs1, rs2, aqrl },
                    0b00001 => Op::AmoswapW { rd, rs1, rs2, aqrl },
                    0b00000 => Op::AmoaddW { rd, rs1, rs2, aqrl },
                    0b00100 => Op::AmoxorW { rd, rs1, rs2, aqrl },
                    0b01100 => Op::AmoandW { rd, rs1, rs2, aqrl },
                    0b01000 => Op::AmoorW { rd, rs1, rs2, aqrl },
                    0b10000 => Op::AmominW { rd, rs1, rs2, aqrl },
                    0b10100 => Op::AmomaxW { rd, rs1, rs2, aqrl },
                    0b11000 => Op::AmominuW { rd, rs1, rs2, aqrl },
                    0b11100 => Op::AmomaxuW { rd, rs1, rs2, aqrl },
                    _ => Op::Illegal,
                }
            } else if function == 0b011 {
                match func {
                    0b00010 => {
                        if rs2 != 0 {
                            Op::Illegal
                        } else {
                            Op::LrD { rd, rs1, aqrl }
                        }
                    }
                    0b00011 => Op::ScD { rd, rs1, rs2, aqrl },
                    0b00001 => Op::AmoswapD { rd, rs1, rs2, aqrl },
                    0b00000 => Op::AmoaddD { rd, rs1, rs2, aqrl },
                    0b00100 => Op::AmoxorD { rd, rs1, rs2, aqrl },
                    0b01100 => Op::AmoandD { rd, rs1, rs2, aqrl },
                    0b01000 => Op::AmoorD { rd, rs1, rs2, aqrl },
                    0b10000 => Op::AmominD { rd, rs1, rs2, aqrl },
                    0b10100 => Op::AmomaxD { rd, rs1, rs2, aqrl },
                    0b11000 => Op::AmominuD { rd, rs1, rs2, aqrl },
                    0b11100 => Op::AmomaxuD { rd, rs1, rs2, aqrl },
                    _ => Op::Illegal,
                }
            } else {
                Op::Illegal
            }
        }

        /* OP */
        0b0110011 => {
            match funct7(bits) {
                // M-extension
                0b0000001 => match function {
                    0b000 => Op::Mul { rd, rs1, rs2 },
                    0b001 => Op::Mulh { rd, rs1, rs2 },
                    0b010 => Op::Mulhsu { rd, rs1, rs2 },
                    0b011 => Op::Mulhu { rd, rs1, rs2 },
                    0b100 => Op::Div { rd, rs1, rs2 },
                    0b101 => Op::Divu { rd, rs1, rs2 },
                    0b110 => Op::Rem { rd, rs1, rs2 },
                    0b111 => Op::Remu { rd, rs1, rs2 },
                    // full case
                    _ => unreachable!(),
                },
                0b0000000 => match function {
                    0b000 => Op::Add { rd, rs1, rs2 },
                    0b001 => Op::Sll { rd, rs1, rs2 },
                    0b010 => Op::Slt { rd, rs1, rs2 },
                    0b011 => Op::Sltu { rd, rs1, rs2 },
                    0b100 => Op::Xor { rd, rs1, rs2 },
                    0b101 => Op::Srl { rd, rs1, rs2 },
                    0b110 => Op::Or { rd, rs1, rs2 },
                    0b111 => Op::And { rd, rs1, rs2 },
                    // full case
                    _ => unreachable!(),
                },
                0b0100000 => match function {
                    0b000 => Op::Sub { rd, rs1, rs2 },
                    0b101 => Op::Sra { rd, rs1, rs2 },
                    _ => Op::Illegal,
                },
                _ => Op::Illegal,
            }
        }

        /* LUI */
        0b0110111 => Op::Lui { rd, imm: u_imm(bits) },

        /* OP-32 */
        0b0111011 => {
            match funct7(bits) {
                // M-extension
                0b0000001 => match function {
                    0b000 => Op::Mulw { rd, rs1, rs2 },
                    0b100 => Op::Divw { rd, rs1, rs2 },
                    0b101 => Op::Divuw { rd, rs1, rs2 },
                    0b110 => Op::Remw { rd, rs1, rs2 },
                    0b111 => Op::Remuw { rd, rs1, rs2 },
                    _ => Op::Illegal,
                },
                0b0000000 => match function {
                    0b000 => Op::Addw { rd, rs1, rs2 },
                    0b001 => Op::Sllw { rd, rs1, rs2 },
                    0b101 => Op::Srlw { rd, rs1, rs2 },
                    _ => Op::Illegal,
                },
                0b0100000 => match function {
                    0b000 => Op::Subw { rd, rs1, rs2 },
                    0b101 => Op::Sraw { rd, rs1, rs2 },
                    _ => Op::Illegal,
                },
                _ => Op::Illegal,
            }
        }

        /* MADD */
        0b1000011 => match funct7(bits) & 3 {
            0b00 => {
                Op::FmaddS { frd: rd, frs1: rs1, frs2: rs2, frs3: rs3(bits), rm: rm!(function) }
            }
            0b01 => {
                Op::FmaddD { frd: rd, frs1: rs1, frs2: rs2, frs3: rs3(bits), rm: rm!(function) }
            }
            _ => Op::Illegal,
        },

        /* MSUB */
        0b1000111 => match funct7(bits) & 3 {
            0b00 => {
                Op::FmsubS { frd: rd, frs1: rs1, frs2: rs2, frs3: rs3(bits), rm: rm!(function) }
            }
            0b01 => {
                Op::FmsubD { frd: rd, frs1: rs1, frs2: rs2, frs3: rs3(bits), rm: rm!(function) }
            }
            _ => Op::Illegal,
        },

        /* NMSUB */
        0b1001011 => match funct7(bits) & 3 {
            0b00 => {
                Op::FnmsubS { frd: rd, frs1: rs1, frs2: rs2, frs3: rs3(bits), rm: rm!(function) }
            }
            0b01 => {
                Op::FnmsubD { frd: rd, frs1: rs1, frs2: rs2, frs3: rs3(bits), rm: rm!(function) }
            }
            _ => Op::Illegal,
        },

        /* NMADD */
        0b1001111 => match funct7(bits) & 3 {
            0b00 => {
                Op::FnmaddS { frd: rd, frs1: rs1, frs2: rs2, frs3: rs3(bits), rm: rm!(function) }
            }
            0b01 => {
                Op::FnmaddD { frd: rd, frs1: rs1, frs2: rs2, frs3: rs3(bits), rm: rm!(function) }
            }
            _ => Op::Illegal,
        },

        /* AUIPC */
        0b0010111 => Op::Auipc { rd, imm: u_imm(bits) },

        /* OP-FP */
        0b1010011 => {
            let function7 = funct7(bits);
            match function7 {
                /* F-extension and D-extension */
                0b0000000 => Op::FaddS { frd: rd, frs1: rs1, frs2: rs2, rm: rm!(function) },
                0b0000001 => Op::FaddD { frd: rd, frs1: rs1, frs2: rs2, rm: rm!(function) },
                0b0000100 => Op::FsubS { frd: rd, frs1: rs1, frs2: rs2, rm: rm!(function) },
                0b0000101 => Op::FsubD { frd: rd, frs1: rs1, frs2: rs2, rm: rm!(function) },
                0b0001000 => Op::FmulS { frd: rd, frs1: rs1, frs2: rs2, rm: rm!(function) },
                0b0001001 => Op::FmulD { frd: rd, frs1: rs1, frs2: rs2, rm: rm!(function) },
                0b0001100 => Op::FdivS { frd: rd, frs1: rs1, frs2: rs2, rm: rm!(function) },
                0b0001101 => Op::FdivD { frd: rd, frs1: rs1, frs2: rs2, rm: rm!(function) },
                0b0101100 => match rs2 {
                    0b00000 => Op::FsqrtS { frd: rd, frs1: rs1, rm: rm!(function) },
                    _ => Op::Illegal,
                },
                0b0101101 => match rs2 {
                    0b00000 => Op::FsqrtD { frd: rd, frs1: rs1, rm: rm!(function) },
                    _ => Op::Illegal,
                },
                0b0010000 => match function {
                    0b000 => Op::FsgnjS { frd: rd, frs1: rs1, frs2: rs2 },
                    0b001 => Op::FsgnjnS { frd: rd, frs1: rs1, frs2: rs2 },
                    0b010 => Op::FsgnjxS { frd: rd, frs1: rs1, frs2: rs2 },
                    _ => Op::Illegal,
                },
                0b0010001 => match function {
                    0b000 => Op::FsgnjD { frd: rd, frs1: rs1, frs2: rs2 },
                    0b001 => Op::FsgnjnD { frd: rd, frs1: rs1, frs2: rs2 },
                    0b010 => Op::FsgnjxD { frd: rd, frs1: rs1, frs2: rs2 },
                    _ => Op::Illegal,
                },
                0b0010100 => match function {
                    0b000 => Op::FminS { frd: rd, frs1: rs1, frs2: rs2 },
                    0b001 => Op::FmaxS { frd: rd, frs1: rs1, frs2: rs2 },
                    _ => Op::Illegal,
                },
                0b0010101 => match function {
                    0b000 => Op::FminD { frd: rd, frs1: rs1, frs2: rs2 },
                    0b001 => Op::FmaxD { frd: rd, frs1: rs1, frs2: rs2 },
                    _ => Op::Illegal,
                },
                0b0100000 => match rs2 {
                    0b00001 => Op::FcvtSD { frd: rd, frs1: rs1, rm: rm!(function) },
                    _ => Op::Illegal,
                },
                0b0100001 => match rs2 {
                    0b00000 => Op::FcvtDS { frd: rd, frs1: rs1, rm: rm!(function) },
                    _ => Op::Illegal,
                },
                0b1100000 => match rs2 {
                    0b00000 => Op::FcvtWS { rd, frs1: rs1, rm: rm!(function) },
                    0b00001 => Op::FcvtWuS { rd, frs1: rs1, rm: rm!(function) },
                    0b00010 => Op::FcvtLS { rd, frs1: rs1, rm: rm!(function) },
                    0b00011 => Op::FcvtLuS { rd, frs1: rs1, rm: rm!(function) },
                    _ => Op::Illegal,
                },
                0b1100001 => match rs2 {
                    0b00000 => Op::FcvtWD { rd, frs1: rs1, rm: rm!(function) },
                    0b00001 => Op::FcvtWuD { rd, frs1: rs1, rm: rm!(function) },
                    0b00010 => Op::FcvtLD { rd, frs1: rs1, rm: rm!(function) },
                    0b00011 => Op::FcvtLuD { rd, frs1: rs1, rm: rm!(function) },
                    _ => Op::Illegal,
                },
                0b1110000 => match (rs2, function) {
                    (0b00000, 0b000) => Op::FmvXW { rd, frs1: rs1 },
                    (0b00000, 0b001) => Op::FclassS { rd, frs1: rs1 },
                    _ => Op::Illegal,
                },
                0b1110001 => match (rs2, function) {
                    (0b00000, 0b000) => Op::FmvXD { rd, frs1: rs1 },
                    (0b00000, 0b001) => Op::FclassD { rd, frs1: rs1 },
                    _ => Op::Illegal,
                },
                0b1010000 => match function {
                    0b000 => Op::FleS { rd, frs1: rs1, frs2: rs2 },
                    0b001 => Op::FltS { rd, frs1: rs1, frs2: rs2 },
                    0b010 => Op::FeqS { rd, frs1: rs1, frs2: rs2 },
                    _ => Op::Illegal,
                },
                0b1010001 => match function {
                    0b000 => Op::FleD { rd, frs1: rs1, frs2: rs2 },
                    0b001 => Op::FltD { rd, frs1: rs1, frs2: rs2 },
                    0b010 => Op::FeqD { rd, frs1: rs1, frs2: rs2 },
                    _ => Op::Illegal,
                },
                0b1101000 => match rs2 {
                    0b00000 => Op::FcvtSW { frd: rd, rs1, rm: rm!(function) },
                    0b00001 => Op::FcvtSWu { frd: rd, rs1, rm: rm!(function) },
                    0b00010 => Op::FcvtSL { frd: rd, rs1, rm: rm!(function) },
                    0b00011 => Op::FcvtSLu { frd: rd, rs1, rm: rm!(function) },
                    _ => Op::Illegal,
                },
                0b1101001 => match rs2 {
                    0b00000 => Op::FcvtDW { frd: rd, rs1, rm: rm!(function) },
                    0b00001 => Op::FcvtDWu { frd: rd, rs1, rm: rm!(function) },
                    0b00010 => Op::FcvtDL { frd: rd, rs1, rm: rm!(function) },
                    0b00011 => Op::FcvtDLu { frd: rd, rs1, rm: rm!(function) },
                    _ => Op::Illegal,
                },
                0b1111000 => match (rs2, function) {
                    (0b00000, 0b000) => Op::FmvWX { frd: rd, rs1 },
                    _ => Op::Illegal,
                },
                0b1111001 => match (rs2, function) {
                    (0b00000, 0b000) => Op::FmvDX { frd: rd, rs1 },
                    _ => Op::Illegal,
                },
                _ => Op::Illegal,
            }
        }

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
            match function {
                0b000 => match bits {
                    0x73 => Op::Ecall,
                    0x100073 => Op::Ebreak,
                    0x30200073 => Op::Mret,
                    0x10200073 => Op::Sret,
                    0x10500073 => Op::Wfi,
                    bits if rd == 0 && funct7(bits) == 0b0001001 => Op::SfenceVma { rs1, rs2 },
                    _ => Op::Illegal,
                },
                0b100 => Op::Illegal,
                _ => {
                    // Otherwise this is CSR instruction
                    let csr = super::Csr(csr(bits));
                    // For CSRRS, CSRRC, CSRRSI, CSRRCI, rs1 = 0 means readonly.
                    // If the CSR is readonly while we try to write it, it is an exception.
                    let readonly = function & 0b010 != 0 && rs1 == 0;
                    if csr.readonly() && !readonly {
                        return Op::Illegal;
                    }
                    match function {
                        0b001 => Op::Csrrw { rd, rs1, csr },
                        0b010 => Op::Csrrs { rd, rs1, csr },
                        0b011 => Op::Csrrc { rd, rs1, csr },
                        0b101 => Op::Csrrwi { rd, imm: rs1, csr },
                        0b110 => Op::Csrrsi { rd, imm: rs1, csr },
                        0b111 => Op::Csrrci { rd, imm: rs1, csr },
                        _ => unreachable!(),
                    }
                }
            }
        }
        _ => Op::Illegal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fence_future_compatibility() {
        // Unsupported FM, RS1, RD must be ignore for future compatiblity.
        let future = decode(0b1111_1111_1111_11111_000_11111_0001111);
        let current = decode(0b0000_1111_1111_00000_000_00000_0001111);
        assert_eq!(unsafe { std::mem::transmute::<_, u64>(future) }, unsafe {
            std::mem::transmute::<_, u64>(current)
        });
    }

    #[test]
    fn test_addi4spn_signedness() {
        let op = decode_compressed(0b000_10_1010_0_1_000_00);
        assert!(match op {
            // Make sure the immediate is correctly decoded as unsigned.
            Op::Addi { rd: 8, rs1: 2, imm: 0b1010101000 } => true,
            _ => false,
        });
    }
}
