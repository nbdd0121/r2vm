#include "emu/mmu.h"
#include "emu/state.h"
#include "riscv/basic_block.h"
#include "riscv/csr.h"
#include "riscv/opcode.h"
#include "riscv/instruction.h"
#include "util/assert.h"
#include "util/format.h"

using namespace riscv;

extern "C" Instruction legacy_decode(uint32_t bits) {
    Instruction ret;
    Opcode opcode = Opcode::illegal;

    // 2-byte compressed instructions
    if ((bits & 0x03) != 0x03) throw "moved to rust";

    if ((bits & 0x1F) != 0x1F) {

        // Field definitions
        using Funct7_field = util::Bitfield<uint32_t, 31, 25>;
        using Rs2_field = util::Bitfield<uint32_t, 24, 20>;
        using Rs1_field = util::Bitfield<uint32_t, 19, 15>;
        using Funct3_field = util::Bitfield<uint32_t, 14, 12>;
        using Rd_field = util::Bitfield<uint32_t, 11, 7>;

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
            case 0b0000011:
            /* Base Opcode LOAD-FP */
            case 0b0000111:
            /* Base Opcode MISC-MEM */
            case 0b0001111:
            /* Base Opcode OP-IMM */
            case 0b0010011:
            /* Base Opcode AUIPC */
            case 0b0010111:
            /* Base Opcode OP-IMM-32 */
            case 0b0011011:
            /* Base Opcode STORE */
            case 0b0100011:
            /* Base Opcode STORE-FP */
            case 0b0100111:
            /* Base Opcode AMO */
            case 0b0101111:
            /* Base Opcode OP */
            case 0b0110011:
            /* Base Opcode LUI */
            case 0b0110111:
            /* Base Opcode OP-32 */
            case 0b0111011: throw "moved to rust";

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
            case 0b1100011:
            /* Base Opcode JALR */
            case 0b1100111:
            /* Base Opcode JAL */
            case 0b1101111:
            /* Base Opcode SYSTEM */
            case 0b1110011: throw "moved to rust";

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
