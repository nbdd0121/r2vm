#include "emu/mmu.h"
#include "emu/state.h"
#include "riscv/basic_block.h"
#include "riscv/csr.h"
#include "riscv/decoder.h"
#include "riscv/disassembler.h"
#include "riscv/opcode.h"
#include "riscv/instruction.h"
#include "util/assert.h"
#include "util/format.h"

namespace riscv {

Instruction Decoder::decode(uint32_t bits) {
    Instruction ret;
    Opcode opcode = Opcode::illegal;

    // 2-byte compressed instructions
    if ((bits & 0x03) != 0x03) {

        // Fields definition
        using C_funct3_field = util::Bitfield<uint32_t, 15, 13>;
        using C_rd_field = util::Bitfield<uint32_t, 11, 7>;
        using C_rs1_field = C_rd_field;
        using C_rs2_field = util::Bitfield<uint32_t, 6, 2>;
        using C_rds_field = util::Bitfield<uint32_t, 4, 2>;
        using C_rs1s_field = util::Bitfield<uint32_t, 9, 7>;
        using C_rs2s_field = C_rds_field;

        using Ci_imm_field = util::Bitfield<int64_t, 12, 12, 6, 2>;
        using Ci_lwsp_imm_field = util::Bitfield<uint32_t, 3, 2, 12, 12, 6, 4, -1, 2>;
        using Ci_ldsp_imm_field = util::Bitfield<uint32_t, 4, 2, 12, 12, 6, 5, -1, 3>;
        using Ci_addi16sp_imm_field = util::Bitfield<int64_t, 12, 12, 4, 3, 5, 5, 2, 2, 6, 6, -1, 4>;
        using Css_swsp_imm_field = util::Bitfield<uint32_t, 8, 7, 12, 9, -1, 2>;
        using Css_sdsp_imm_field = util::Bitfield<uint32_t, 9, 7, 12, 10, -1, 3>;
        using Ciw_imm_field = util::Bitfield<uint32_t, 10, 7, 12, 11, 5, 5, 6, 6, -1, 2>;
        using Cl_lw_imm_field = util::Bitfield<uint32_t, 5, 5, 12, 10, 6, 6, -1, 2>;
        using Cl_ld_imm_field = util::Bitfield<uint32_t, 6, 5, 12, 10, -1, 3>;
        using Cs_sw_imm_field = Cl_lw_imm_field;
        using Cs_sd_imm_field = Cl_ld_imm_field;
        using Cb_imm_field = util::Bitfield<int64_t, 12, 12, 6, 5, 2, 2, 11, 10, 4, 3, -1, 1>;
        using Cj_imm_field = util::Bitfield<int64_t, 12, 12, 8, 8, 10, 9, 6, 6, 7, 7, 2, 2, 11, 11, 5, 3, -1, 1>;

        int function = C_funct3_field::extract(bits);

        ret.length(2);

        switch (bits & 0b11) {
            case 0b00: {
                switch (function) {
                    case 0b000: {
                        reg_t imm = Ciw_imm_field::extract(bits);
                        if (imm == 0) {
                            // Illegal instruction. At this point ret is all zero, so return directly.
                            return ret;
                        }
                        // C.ADDI4SPN
                        // translate to addi rd', x2, imm
                        ret.opcode(Opcode::addi);
                        ret.rd(C_rds_field::extract(bits) + 8);
                        ret.rs1(2);
                        ret.imm(imm);
                        return ret;
                    }
                    case 0b001: {
                        // C.FLD
                        // translate to fld rd', rs1', offset
                        ret.opcode(Opcode::fld);
                        ret.rd(C_rds_field::extract(bits) + 8);
                        ret.rs1(C_rs1s_field::extract(bits) + 8);
                        ret.imm(Cl_ld_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b010: {
                        // C.LW
                        // translate to lw rd', rs1', offset
                        ret.opcode(Opcode::lw);
                        ret.rd(C_rds_field::extract(bits) + 8);
                        ret.rs1(C_rs1s_field::extract(bits) + 8);
                        ret.imm(Cl_lw_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b011: {
                        // C.LD
                        // translate to ld rd', rs1', offset
                        ret.opcode(Opcode::ld);
                        ret.rd(C_rds_field::extract(bits) + 8);
                        ret.rs1(C_rs1s_field::extract(bits) + 8);
                        ret.imm(Cl_ld_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b100:
                        // Reserved
                        goto illegal_compressed;
                    case 0b101: {
                        // C.FSD
                        // translate to fsd rs2', rs1', offset
                        ret.opcode(Opcode::fsd);
                        ret.rs1(C_rs1s_field::extract(bits) + 8);
                        ret.rs2(C_rs2s_field::extract(bits) + 8);
                        ret.imm(Cs_sd_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b110: {
                        // C.SW
                        // translate to sw rs2', rs1', offset
                        ret.opcode(Opcode::sw);
                        ret.rs1(C_rs1s_field::extract(bits) + 8);
                        ret.rs2(C_rs2s_field::extract(bits) + 8);
                        ret.imm(Cs_sw_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b111: {
                        // C.SD
                        // translate to sd rs2', rs1', offset
                        ret.opcode(Opcode::sd);
                        ret.rs1(C_rs1s_field::extract(bits) + 8);
                        ret.rs2(C_rs2s_field::extract(bits) + 8);
                        ret.imm(Cs_sd_imm_field::extract(bits));
                        return ret;
                    }
                    // full case
                    default: UNREACHABLE();
                }
            }
            case 0b01: {
                switch (function) {
                    case 0b000: {
                        // rd = x0 is HINT
                        // r0 = 0 is C.NOP
                        // C.ADDI
                        // translate to addi rd, rd, imm
                        int rd = C_rd_field::extract(bits);
                        ret.opcode(Opcode::addi);
                        ret.rd(rd);
                        ret.rs1(rd);
                        ret.imm(Ci_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b001: {
                        int rd = C_rd_field::extract(bits);
                        if (rd == 0) {
                            // Reserved
                            goto illegal_compressed;
                        }
                        // C.ADDIW
                        // translate to addiw rd, rd, imm
                        ret.opcode(Opcode::addiw);
                        ret.rd(rd);
                        ret.rs1(rd);
                        ret.imm(Ci_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b010: {
                        // rd = x0 is HINT
                        // C.LI
                        // translate to addi rd, x0, imm
                        ret.opcode(Opcode::addi);
                        ret.rd(C_rd_field::extract(bits));
                        ret.rs1(0);
                        ret.imm(Ci_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b011: {
                        int rd = C_rd_field::extract(bits);
                        if (rd == 2) {
                            reg_t imm = Ci_addi16sp_imm_field::extract(bits);
                            if (imm == 0) {
                                // Reserved
                                goto illegal_compressed;
                            }
                            // C.ADDI16SP
                            // translate to addi x2, x2, imm
                            ret.opcode(Opcode::addi);
                            ret.rd(2);
                            ret.rs1(2);
                            ret.imm(imm);
                            return ret;
                        } else {
                            // rd = x0 is HINT
                            // C.LUI
                            // translate to lui rd, imm
                            ret.opcode(Opcode::lui);
                            ret.rd(rd);
                            ret.imm(Ci_imm_field::extract(bits) << 12);
                            return ret;
                        }
                    }
                    case 0b100: {
                        int rs1 = C_rs1s_field::extract(bits) + 8;
                        switch (util::Bitfield<uint32_t, 11, 10>::extract(bits)) {
                            case 0b00: {
                                // imm = 0 is HINT
                                // C.SRLI
                                // translate to srli rs1', rs1', imm
                                ret.opcode(Opcode::srli);
                                ret.rd(rs1);
                                ret.rs1(rs1);
                                ret.imm(Ci_imm_field::extract(bits) & 63);
                                return ret;
                            }
                            case 0b01: {
                                // imm = 0 is HINT
                                // C.SRAI
                                // translate to srai rs1', rs1', imm
                                ret.opcode(Opcode::srai);
                                ret.rs1(rs1);
                                ret.rd(rs1);
                                ret.imm(Ci_imm_field::extract(bits) & 63);
                                return ret;
                            }
                            case 0b10: {
                                // C.ANDI
                                // translate to andi rs1', rs1', imm
                                ret.opcode(Opcode::andi);
                                ret.rs1(rs1);
                                ret.rd(rs1);
                                ret.imm(Ci_imm_field::extract(bits));
                                return ret;
                            }
                            case 0b11: {
                                if ((bits & 0x1000) == 0) {
                                    // C.SUB
                                    // C.XOR
                                    // C.OR
                                    // C.AND
                                    // translates to [OP] rs1', rs1', rs2'
                                    switch (util::Bitfield<uint32_t, 6, 5>::extract(bits)) {
                                        case 0b00: opcode = Opcode::sub; break;
                                        case 0b01: opcode = Opcode::i_xor; break;
                                        case 0b10: opcode = Opcode::i_or; break;
                                        case 0b11: opcode = Opcode::i_and; break;
                                        // full case
                                        default: UNREACHABLE();
                                    }
                                    ret.opcode(opcode);
                                    ret.rd(rs1);
                                    ret.rs1(rs1);
                                    ret.rs2(C_rs2s_field::extract(bits) + 8);
                                    return ret;
                                } else {
                                    switch (util::Bitfield<uint32_t, 6, 5>::extract(bits)) {
                                        case 0b00: {
                                            // C.SUBW
                                            // translates to subw rs1', rs1', rs2'
                                            ret.opcode(Opcode::subw);
                                            ret.rd(rs1);
                                            ret.rs1(rs1);
                                            ret.rs2(C_rs2s_field::extract(bits) + 8);
                                            return ret;
                                        }
                                        case 0b01: {
                                            // C.ADDW
                                            // translates to addw rs1', rs1', rs2'
                                            ret.opcode(Opcode::addw);
                                            ret.rd(rs1);
                                            ret.rs1(rs1);
                                            ret.rs2(C_rs2s_field::extract(bits) + 8);
                                            return ret;
                                        }
                                        default:
                                            // Reserved
                                            goto illegal_compressed;
                                    }
                                }
                            }
                            // full case
                            default: UNREACHABLE();
                        }
                    }
                    case 0b101: {
                        // C.J
                        // translate to jal x0, imm
                        ret.opcode(Opcode::jal);
                        ret.rd(0);
                        ret.imm(Cj_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b110: {
                        // C.BEQZ
                        // translate to beq rs1', x0, imm
                        ret.opcode(Opcode::beq);
                        ret.rs2(0);
                        ret.rs1(C_rs1s_field::extract(bits) + 8);
                        ret.imm(Cb_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b111: {
                        // C.BNEZ
                        // translate to bne rs1', x0, imm
                        ret.opcode(Opcode::bne);
                        ret.rs2(0);
                        ret.rs1(C_rs1s_field::extract(bits) + 8);
                        ret.imm(Cb_imm_field::extract(bits));
                        return ret;
                    }
                    // full case
                    default: UNREACHABLE();
                }
            }
            case 0b10: {
                switch (function) {
                    case 0b000: {
                        // imm = 0 is HINT
                        // rd = 0 is HINT
                        // C.SLLI
                        // translates to slli rd, rd, imm
                        int rd = C_rd_field::extract(bits);
                        ret.opcode(Opcode::slli);
                        ret.rd(rd);
                        ret.rs1(rd);
                        ret.imm(Ci_imm_field::extract(bits) & 63);
                        return ret;
                    }
                    case 0b001: {
                        // C.FLDSP
                        // translate to fld rd, x2, imm
                        int rd = C_rd_field::extract(bits);
                        ret.opcode(Opcode::fld);
                        ret.rd(rd);
                        ret.rs1(2);
                        ret.imm(Ci_ldsp_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b010: {
                        int rd = C_rd_field::extract(bits);
                        if (rd == 0) {
                            // Reserved
                            goto illegal_compressed;
                        }
                        // C.LWSP
                        // translate to lw rd, x2, imm
                        ret.opcode(Opcode::lw);
                        ret.rd(rd);
                        ret.rs1(2);
                        ret.imm(Ci_lwsp_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b011: {
                        int rd = C_rd_field::extract(bits);
                        if (rd == 0) {
                            // Reserved
                            goto illegal_compressed;
                        }
                        // C.LDSP
                        // translate to ld rd, x2, imm
                        ret.opcode(Opcode::ld);
                        ret.rd(rd);
                        ret.rs1(2);
                        ret.imm(Ci_ldsp_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b100: {
                        int rs2 = C_rs2_field::extract(bits);
                        if ((bits & 0x1000) == 0) {
                            if (rs2 == 0) {
                                int rs1 = C_rs1_field::extract(bits);
                                if (rs1 == 0) {
                                    // Reserved
                                    goto illegal_compressed;
                                }
                                // C.JR
                                // translate to jalr x0, rs1, 0
                                ret.opcode(Opcode::jalr);
                                ret.rd(0);
                                ret.rs1(rs1);
                                ret.imm(0);
                                return ret;
                            } else {
                                // rd = 0 is HINT
                                // C.MV
                                // translate to add rd, x0, rs2
                                ret.opcode(Opcode::add);
                                ret.rd(C_rd_field::extract(bits));
                                ret.rs1(0);
                                ret.rs2(rs2);
                                return ret;
                            }
                        } else {
                            int rs1 = C_rs1_field::extract(bits);
                            if (rs1 == 0) {
                                // C.EBREAK
                                ret.opcode(Opcode::ebreak);
                                return ret;
                            } else if (rs2 == 0) {
                                // C.JALR
                                // translate to jalr x1, rs1, 0
                                ret.opcode(Opcode::jalr);
                                ret.rd(1);
                                ret.rs1(rs1);
                                ret.imm(0);
                                return ret;
                            } else {
                                // rd = 0 is HINT
                                // C.ADD
                                // translate to add rd, rd, rs2
                                int rd = C_rd_field::extract(bits);
                                ret.opcode(Opcode::add);
                                ret.rd(rd);
                                ret.rs1(rd);
                                ret.rs2(rs2);
                                return ret;
                            }
                        }
                    }
                    case 0b101: {
                        // C.FSDSP
                        // translate to fsd rs2, x2, imm
                        ret.opcode(Opcode::fsd);
                        ret.rs1(2);
                        ret.rs2(C_rs2_field::extract(bits));
                        ret.imm(Css_sdsp_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b110: {
                        // C.SWSP
                        // translate to sw rs2, x2, imm
                        ret.opcode(Opcode::sw);
                        ret.rs1(2);
                        ret.rs2(C_rs2_field::extract(bits));
                        ret.imm(Css_swsp_imm_field::extract(bits));
                        return ret;
                    }
                    case 0b111: {
                        // C.SDSP
                        // translate to sd rs2, x2, imm
                        ret.opcode(Opcode::sd);
                        ret.rs1(2);
                        ret.rs2(C_rs2_field::extract(bits));
                        ret.imm(Css_sdsp_imm_field::extract(bits));
                        return ret;
                    }
                    // full case
                    default: UNREACHABLE();
                }
                break;
            }
            // full case
            default: UNREACHABLE();
        }

    illegal_compressed:
        // All illegal instructions landed here. Since ret.opcode() is illegal by default, we can just return it.
        return ret;
    }

    if ((bits & 0x1F) != 0x1F) {

        // Field definitions
        using Funct7_field = util::Bitfield<uint32_t, 31, 25>;
        using Rs2_field = util::Bitfield<uint32_t, 24, 20>;
        using Rs1_field = util::Bitfield<uint32_t, 19, 15>;
        using Funct3_field = util::Bitfield<uint32_t, 14, 12>;
        using Rd_field = util::Bitfield<uint32_t, 11, 7>;
        using Csr_field = util::Bitfield<uint32_t, 31, 20>;

        using I_imm_field = util::Bitfield<int64_t, 31, 20>;
        using S_imm_field = util::Bitfield<int64_t, 31, 25, 11, 7>;
        using B_imm_field = util::Bitfield<int64_t, 31, 31, 7, 7, 30, 25, 11, 8, -1, 1>;
        using U_imm_field = util::Bitfield<int64_t, 31, 12, -1, 12>;
        using J_imm_field = util::Bitfield<int64_t, 31, 31, 19, 12, 20, 20, 30, 21, -1, 1>;

        // Almost all functions use funct3
        int function = Funct3_field::extract(bits);

        // First fill all rd, rs1, rs2 as they are common.
        ret.rd(Rd_field::extract(bits));
        ret.rs1(Rs1_field::extract(bits));
        int rs2 = Rs2_field::extract(bits);
        ret.rs2(rs2);
        ret.length(4);

        switch (bits & 0b1111111) {
            /* Base Opcode LOAD */
            case 0b0000011: {
                switch (function) {
                    case 0b000: opcode = Opcode::lb; break;
                    case 0b001: opcode = Opcode::lh; break;
                    case 0b010: opcode = Opcode::lw; break;
                    case 0b011: opcode = Opcode::ld; break;
                    case 0b100: opcode = Opcode::lbu; break;
                    case 0b101: opcode = Opcode::lhu; break;
                    case 0b110: opcode = Opcode::lwu; break;
                    goto illegal;
                }
                ret.opcode(opcode);
                ret.imm(I_imm_field::extract(bits));
                return ret;
            }

            /* Base Opcode LOAD-FP */
            case 0b0000111: {
                /* F-extension */
                switch (function) {
                    case 0b010: opcode = Opcode::flw; break;
                    case 0b011: opcode = Opcode::fld; break;
                    goto illegal;
                }
                ret.opcode(opcode);
                ret.imm(I_imm_field::extract(bits));
                return ret;
            }

            /* Base Opcode MISC-MEM */
            case 0b0001111: {
                switch (function) {
                    case 0b000: {
                        reg_t imm = I_imm_field::extract(bits);
                        ret.opcode(Opcode::fence);
                        ret.imm(imm & 0xFF);
                        return ret;
                    }
                    case 0b001: {
                        ret.opcode(Opcode::fence_i);
                        return ret;
                    }
                    default: goto illegal;
                }
            }

            /* Base Opcode OP-IMM */
            case 0b0010011: {
                reg_t imm = I_imm_field::extract(bits);
                switch (function) {
                    case 0b000: opcode = Opcode::addi; break;
                    case 0b001:
                        if (imm >= 64) goto illegal;
                        opcode = Opcode::slli;
                        break;
                    case 0b010: opcode = Opcode::slti; break;
                    case 0b011: opcode = Opcode::sltiu; break;
                    case 0b100: opcode = Opcode::xori; break;
                    case 0b101:
                        if (imm & 0x400) {
                            opcode = Opcode::srai;
                            imm &=~ 0x400;
                        } else {
                            opcode = Opcode::srli;
                        }
                        if (imm >= 64) goto illegal;
                        break;
                    case 0b110: opcode = Opcode::ori; break;
                    case 0b111: opcode = Opcode::andi; break;
                    // full case
                }
                ret.opcode(opcode);
                ret.imm(imm);
                return ret;
            }

            /* Base Opcode AUIPC */
            case 0b0010111: {
                ret.opcode(Opcode::auipc);
                ret.imm(U_imm_field::extract(bits));
                return ret;
            }

            /* Base Opcode OP-IMM-32 */
            case 0b0011011: {
                reg_t imm = I_imm_field::extract(bits);
                switch (function) {
                    case 0b000: opcode = Opcode::addiw; break;
                    case 0b001: {
                        if (imm >= 32) goto illegal;
                        opcode = Opcode::slliw;
                        break;
                    }
                    case 0b101: {
                        if (imm & 0x400) {
                            opcode = Opcode::sraiw;
                            imm &=~ 0x400;
                        } else {
                            opcode = Opcode::srliw;
                        }
                        if (imm >= 32) goto illegal;
                        break;
                    }
                    goto illegal;
                }
                ret.opcode(opcode);
                ret.imm(imm);
                return ret;
            }

            /* Base Opcode STORE */
            case 0b0100011: {
                switch (function) {
                    case 0b000: opcode = Opcode::sb; break;
                    case 0b001: opcode = Opcode::sh; break;
                    case 0b010: opcode = Opcode::sw; break;
                    case 0b011: opcode = Opcode::sd; break;
                    goto illegal;
                }
                ret.opcode(opcode);
                ret.imm(S_imm_field::extract(bits));
                return ret;
            }

            /* Base Opcode STORE-FP */
            case 0b0100111: {
                /* F-extension */
                switch (function) {
                    case 0b010: opcode = Opcode::fsw; break;
                    case 0b011: opcode = Opcode::fsd; break;
                    default: goto illegal;
                }
                ret.opcode(opcode);
                ret.imm(S_imm_field::extract(bits));
                return ret;
            }

            /* Base Opcode AMO */
            case 0b0101111: {
                /* A-Extension */
                int function7 = Funct7_field::extract(bits);
                if (function == 0b010) {
                    switch (function7 >> 2) {
                        case 0b00010: if (rs2 != 0) goto illegal; opcode = Opcode::lr_w; break;
                        case 0b00011: opcode = Opcode::sc_w; break;
                        case 0b00001: opcode = Opcode::amoswap_w; break;
                        case 0b00000: opcode = Opcode::amoadd_w; break;
                        case 0b00100: opcode = Opcode::amoxor_w; break;
                        case 0b01100: opcode = Opcode::amoand_w; break;
                        case 0b01000: opcode = Opcode::amoor_w; break;
                        case 0b10000: opcode = Opcode::amomin_w; break;
                        case 0b10100: opcode = Opcode::amomax_w; break;
                        case 0b11000: opcode = Opcode::amominu_w; break;
                        case 0b11100: opcode = Opcode::amomaxu_w; break;
                        default: goto illegal;
                    }
                } else if (function == 0b011) {
                    switch (function7 >> 2) {
                        case 0b00010: if (rs2 != 0) goto illegal; opcode = Opcode::lr_d; break;
                        case 0b00011: opcode = Opcode::sc_d; break;
                        case 0b00001: opcode = Opcode::amoswap_d; break;
                        case 0b00000: opcode = Opcode::amoadd_d; break;
                        case 0b00100: opcode = Opcode::amoxor_d; break;
                        case 0b01100: opcode = Opcode::amoand_d; break;
                        case 0b01000: opcode = Opcode::amoor_d; break;
                        case 0b10000: opcode = Opcode::amomin_d; break;
                        case 0b10100: opcode = Opcode::amomax_d; break;
                        case 0b11000: opcode = Opcode::amominu_d; break;
                        case 0b11100: opcode = Opcode::amomaxu_d; break;
                        default: goto illegal;
                    }
                } else {
                    goto illegal;
                }
                ret.opcode(opcode);
                // aq and rl
                ret.imm(function7 & 3);
                return ret;
            }

            /* Base Opcode OP */
            case 0b0110011: {
                int function7 = Funct7_field::extract(bits);

                // M-extension
                if (function7 == 0b0000001) {
                    switch (function) {
                        case 0b000: opcode = Opcode::mul; break;
                        case 0b001: opcode = Opcode::mulh; break;
                        case 0b010: opcode = Opcode::mulhsu; break;
                        case 0b011: opcode = Opcode::mulhu; break;
                        case 0b100: opcode = Opcode::div; break;
                        case 0b101: opcode = Opcode::divu; break;
                        case 0b110: opcode = Opcode::rem; break;
                        case 0b111: opcode = Opcode::remu; break;
                        // full case
                    }
                    ret.opcode(opcode);
                    return ret;
                }

                switch (function) {
                    case 0b000:
                        if (function7 == 0b0000000) opcode = Opcode::add;
                        else if (function7 == 0b0100000) opcode = Opcode::sub;
                        else goto illegal;
                        break;
                    case 0b001:
                        if (function7 == 0b0000000) opcode = Opcode::sll;
                        else goto illegal;
                        break;
                    case 0b010:
                        if (function7 == 0b0000000) opcode = Opcode::slt;
                        else goto illegal;
                        break;
                    case 0b011:
                        if (function7 == 0b0000000) opcode = Opcode::sltu;
                        else goto illegal;
                        break;
                    case 0b100:
                        if (function7 == 0b0000000) opcode = Opcode::i_xor;
                        else goto illegal;
                        break;
                    case 0b101:
                        if (function7 == 0b0000000) opcode = Opcode::srl;
                        else if (function7 == 0b0100000) opcode = Opcode::sra;
                        else goto illegal;
                        break;
                    case 0b110:
                        if (function7 == 0b0000000) opcode = Opcode::i_or;
                        else goto illegal;
                        break;
                    case 0b111:
                        if (function7 == 0b0000000) opcode = Opcode::i_and;
                        else goto illegal;
                        break;
                    // full case
                }
                ret.opcode(opcode);
                return ret;
            }

            /* Base Opcode LUI */
            case 0b0110111: {
                ret.opcode(Opcode::lui);
                ret.imm(U_imm_field::extract(bits));
                return ret;
            }

            /* Base Opcode OP-32 */
            case 0b0111011: {
                int function7 = Funct7_field::extract(bits);

                // M-extension
                if (function7 == 0b0000001) {
                    switch (function) {
                        case 0b000: opcode = Opcode::mulw; break;
                        case 0b100: opcode = Opcode::divw; break;
                        case 0b101: opcode = Opcode::divuw; break;
                        case 0b110: opcode = Opcode::remw; break;
                        case 0b111: opcode = Opcode::remuw; break;
                        default: goto illegal;
                    }
                    ret.opcode(opcode);
                    return ret;
                }

                switch (function) {
                    case 0b000:
                        if (function7 == 0b0000000) opcode = Opcode::addw;
                        else if (function7 == 0b0100000) opcode = Opcode::subw;
                        else goto illegal;
                        break;
                    case 0b001:
                        if (function7 == 0b0000000) opcode = Opcode::sllw;
                        else goto illegal;
                        break;
                    case 0b101:
                        if (function7 == 0b0000000) opcode = Opcode::srlw;
                        else if (function7 == 0b0100000) opcode = Opcode::sraw;
                        else goto illegal;
                        break;
                    default: goto illegal;
                }
                ret.opcode(opcode);
                return ret;
            }

            /* Base Opcode MADD */
            case 0b1000011: {
                int function7 = Funct7_field::extract(bits);
                switch (function7 & 3) {
                    case 0b00: opcode = Opcode::fmadd_s; break;
                    case 0b01: opcode = Opcode::fmadd_d; break;
                    default: goto illegal;
                }
                ret.opcode(opcode);
                ret.rs3(function7 >> 2);
                ret.rm(function);
                return ret;
            }

            /* Base Opcode MSUB */
            case 0b1000111: {
                int function7 = Funct7_field::extract(bits);
                switch (function7 & 3) {
                    case 0b00: opcode = Opcode::fmsub_s; break;
                    case 0b01: opcode = Opcode::fmsub_d; break;
                    default: goto illegal;
                }
                ret.opcode(opcode);
                ret.rs3(function7 >> 2);
                ret.rm(function);
                return ret;
            }

            /* Base Opcode NMSUB */
            case 0b1001011: {
                int function7 = Funct7_field::extract(bits);
                switch (function7 & 3) {
                    case 0b00: opcode = Opcode::fnmsub_s; break;
                    case 0b01: opcode = Opcode::fnmsub_d; break;
                    default: goto illegal;
                }
                ret.opcode(opcode);
                ret.rs3(function7 >> 2);
                ret.rm(function);
                return ret;
            }

            /* Base Opcode NMADD */
            case 0b1001111: {
                int function7 = Funct7_field::extract(bits);
                switch (function7 & 3) {
                    case 0b00: opcode = Opcode::fnmadd_s; break;
                    case 0b01: opcode = Opcode::fnmadd_d; break;
                    default: goto illegal;
                }
                ret.opcode(opcode);
                ret.rs3(function7 >> 2);
                ret.rm(function);
                return ret;
            }

            /* Base Opcode OP-FP */
            case 0b1010011: {
                int function7 = Funct7_field::extract(bits);
                switch (function7) {
                    /* F-extension and D-extension */
                    case 0b0000000: opcode = Opcode::fadd_s; break;
                    case 0b0000001: opcode = Opcode::fadd_d; break;
                    case 0b0000100: opcode = Opcode::fsub_s; break;
                    case 0b0000101: opcode = Opcode::fsub_d; break;
                    case 0b0001000: opcode = Opcode::fmul_s; break;
                    case 0b0001001: opcode = Opcode::fmul_d; break;
                    case 0b0001100: opcode = Opcode::fdiv_s; break;
                    case 0b0001101: opcode = Opcode::fdiv_d; break;
                    case 0b0101100:
                        if (rs2 == 0b00000) opcode = Opcode::fsqrt_s;
                        else goto illegal;
                        break;
                    case 0b0101101:
                        if (rs2 == 0b00000) opcode = Opcode::fsqrt_d;
                        else goto illegal;
                        break;
                    case 0b0010000:
                        if (function == 0b000) opcode = Opcode::fsgnj_s;
                        else if (function == 0b001) opcode = Opcode::fsgnjn_s;
                        else if (function == 0b010) opcode = Opcode::fsgnjx_s;
                        else goto illegal;
                        break;
                    case 0b0010001:
                        if (function == 0b000) opcode = Opcode::fsgnj_d;
                        else if (function == 0b001) opcode = Opcode::fsgnjn_d;
                        else if (function == 0b010) opcode = Opcode::fsgnjx_d;
                        else goto illegal;
                        break;
                    case 0b0010100:
                        if (function == 0b000) opcode = Opcode::fmin_s;
                        else if (function == 0b001) opcode = Opcode::fmax_s;
                        else goto illegal;
                        break;
                    case 0b0010101:
                        if (function == 0b000) opcode = Opcode::fmin_d;
                        else if (function == 0b001) opcode = Opcode::fmax_d;
                        else goto illegal;
                        break;
                    case 0b0100000:
                        if (rs2 == 0b00001) opcode = Opcode::fcvt_s_d;
                        else goto illegal;
                        break;
                    case 0b0100001:
                        if (rs2 == 0b00000) opcode = Opcode::fcvt_d_s;
                        else goto illegal;
                        break;
                    case 0b1100000:
                        switch (rs2) {
                            case 0b00000: opcode = Opcode::fcvt_w_s; break;
                            case 0b00001: opcode = Opcode::fcvt_wu_s; break;
                            case 0b00010: opcode = Opcode::fcvt_l_s; break;
                            case 0b00011: opcode = Opcode::fcvt_lu_s; break;
                            default: goto illegal;
                        }
                        break;
                    case 0b1100001:
                        switch (rs2) {
                            case 0b00000: opcode = Opcode::fcvt_w_d; break;
                            case 0b00001: opcode = Opcode::fcvt_wu_d; break;
                            case 0b00010: opcode = Opcode::fcvt_l_d; break;
                            case 0b00011: opcode = Opcode::fcvt_lu_d; break;
                            default: goto illegal;
                        }
                        break;
                    case 0b1110000:
                        if (rs2 == 0b00000 && function == 0b000) opcode = Opcode::fmv_x_w;
                        else if (rs2 == 0b00000 && function == 0b001) opcode = Opcode::fclass_s;
                        else goto illegal;
                        break;
                    case 0b1110001:
                        if (rs2 == 0b00000 && function == 0b000) opcode = Opcode::fmv_x_d;
                        else if (rs2 == 0b00000 && function == 0b001) opcode = Opcode::fclass_d;
                        else goto illegal;
                        break;
                    case 0b1010000:
                        if (function == 0b000) opcode = Opcode::fle_s;
                        else if (function == 0b001) opcode = Opcode::flt_s;
                        else if (function == 0b010) opcode = Opcode::feq_s;
                        else goto illegal;
                        break;
                    case 0b1010001:
                        if (function == 0b000) opcode = Opcode::fle_d;
                        else if (function == 0b001) opcode = Opcode::flt_d;
                        else if (function == 0b010) opcode = Opcode::feq_d;
                        else goto illegal;
                        break;
                    case 0b1101000:
                        switch (rs2) {
                            case 0b00000: opcode = Opcode::fcvt_s_w; break;
                            case 0b00001: opcode = Opcode::fcvt_s_wu; break;
                            case 0b00010: opcode = Opcode::fcvt_s_l; break;
                            case 0b00011: opcode = Opcode::fcvt_s_lu; break;
                            default: goto illegal;
                        }
                        break;
                    case 0b1101001:
                        switch (rs2) {
                            case 0b00000: opcode = Opcode::fcvt_d_w; break;
                            case 0b00001: opcode = Opcode::fcvt_d_wu; break;
                            case 0b00010: opcode = Opcode::fcvt_d_l; break;
                            case 0b00011: opcode = Opcode::fcvt_d_lu; break;
                            default: goto illegal;
                        }
                        break;
                    case 0b1111000:
                        if (rs2 == 0b00000 && function == 0b000) opcode = Opcode::fmv_w_x;
                        else goto illegal;
                        break;
                    case 0b1111001:
                        if (rs2 == 0b00000 && function == 0b000) opcode = Opcode::fmv_d_x;
                        else goto illegal;
                        break;
                    default: goto illegal;
                }
                ret.opcode(opcode);
                ret.rm(function);
                return ret;
            }

             /* Base Opcode BRANCH */
            case 0b1100011: {
                switch (function) {
                    case 0b000: opcode = Opcode::beq; break;
                    case 0b001: opcode = Opcode::bne; break;
                    case 0b100: opcode = Opcode::blt; break;
                    case 0b101: opcode = Opcode::bge; break;
                    case 0b110: opcode = Opcode::bltu; break;
                    case 0b111: opcode = Opcode::bgeu; break;
                    default: goto illegal;
                }
                ret.opcode(opcode);
                ret.imm(B_imm_field::extract(bits));
                return ret;
            }

            /* Base Opcode JALR */
            case 0b1100111: {
                if (function != 0b000) goto illegal;
                ret.opcode(Opcode::jalr);
                ret.imm(I_imm_field::extract(bits));
                return ret;
            }

            /* Base Opcode JAL */
            case 0b1101111: {
                ret.opcode(Opcode::jal);
                ret.imm(J_imm_field::extract(bits));
                return ret;
            }

            /* Base Opcode SYSTEM */
            case 0b1110011: {
                switch (function) {
                    case 0b000:
                        if (bits == 0x73) {
                            // All other bits cleared
                            ret.opcode(Opcode::ecall);
                            return ret;
                        } else if (bits == 0x100073) {
                            ret.opcode(Opcode::ebreak);
                            return ret;
                        }
                        goto illegal;
                    case 0b001: opcode = Opcode::csrrw; break;
                    case 0b010: opcode = Opcode::csrrs; break;
                    case 0b011: opcode = Opcode::csrrc; break;
                    case 0b101: opcode = Opcode::csrrwi; break;
                    case 0b110: opcode = Opcode::csrrsi; break;
                    case 0b111: opcode = Opcode::csrrci; break;
                    default: goto illegal;
                }
                // CSR instructions
                // In both I and non-I cases we put immediate in RS1, so we don't have to deal with that specially.
                // csr fields are similar to I-type but unsigned.
                ret.opcode(opcode);
                ret.imm(Csr_field::extract(bits));
                return ret;
            }

            default: goto illegal;
        }

    illegal:
        // All illegal instructions landed here. Since ret.opcode() is illegal by default, we can just return it.
        return ret;
    }

    // Long instructions are not supported yet. For now just treat it as a 2-bit illegal instruction.
    ret.length(2);
    return ret;
}

// Determine whether an instruction can change control flow (excluding exceptional scenario).
bool Decoder::can_change_control_flow(Instruction inst) {
    switch (inst.opcode()) {
        // Branch and jump instructions will definitely disrupt the control flow.
        case Opcode::beq:
        case Opcode::bne:
        case Opcode::blt:
        case Opcode::bge:
        case Opcode::bltu:
        case Opcode::bgeu:
        case Opcode::jalr:
        case Opcode::jal:
        // ecall and illegal logically does not interrupt control flow, but as they trigger fault, the control flow
        // will eventually be redirected to the signal handler.
        case Opcode::ebreak:
        case Opcode::illegal:
        // fence.i might cause instruction cache to be invalidated. If the code executing is invalidated, then we need
        // to stop executing, so it is safer to treat it as special instruction at the moment.
        case Opcode::fence_i:
        // ecall usually does not change control flow, but due to existence of syscall such as exit(), it is safer to
        // treat it as specially at the moment, and maybe considering optimizing later.
        case Opcode::ecall:
            return true;
        // A common way of using basic blocks is to `batch' instret and pc increment. So if CSR to be accessed is
        // instret, consider it as special.
        case Opcode::csrrw:
        case Opcode::csrrs:
        case Opcode::csrrc:
        case Opcode::csrrwi:
        case Opcode::csrrsi:
        case Opcode::csrrci: {
            Csr csr = static_cast<Csr>(inst.imm());
            return csr == Csr::instret || csr == Csr::instreth;
        }
        default:
            return false;
    }
}

Instruction Decoder::decode_instruction() {
    uint32_t bits = emu::load_memory<uint32_t>(pc_);
    Instruction inst = decode(bits);
    if (emu::state::disassemble) {
        Disassembler::print_instruction(pc_, bits, inst);
    }
    pc_ += inst.length();
    return inst;
}

Basic_block Decoder::decode_basic_block() {
    Basic_block block;

    if (emu::state::disassemble) {
        util::log("Decoding {:x}\n", pc_);
    }

    block.start_pc = pc_;

    // Scan util a branching instruction is encountered
    while (true) {
        Instruction inst = decode_instruction();
        block.instructions.push_back(inst);

        // TODO: We should also consider breaking when this gets large.
        if (can_change_control_flow(inst)) {
            break;
        }
    }

    block.end_pc = pc_;
    return block;
}

}
