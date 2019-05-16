#include <iostream>
#include <iomanip>

#include "riscv/disassembler.h"
#include "riscv/opcode.h"
#include "riscv/csr.h"
#include "riscv/decoder.h"
#include "riscv/instruction.h"

#include "util/assert.h"
#include "util/format.h"

using namespace riscv;

const char* Disassembler::register_name(int reg) {
    static std::array<const char*, 32> reg_names = {
        "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
        "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
        "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
        "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"
    };
    return reg_names[reg];
}

const char* Disassembler::csr_name(Csr csr) {
    switch (csr) {
#define CASE(x) case Csr::x: return #x;
        CASE(fflags)
        CASE(frm)
        CASE(fcsr)
        CASE(cycle)
        CASE(time)
        CASE(instret)
        CASE(cycleh)
        CASE(timeh)
        CASE(instreth)
#undef CASE
        default:
            return "(unknown)";
    }
}

const char* Disassembler::opcode_name(Opcode opcode) {
    switch (opcode) {
        case Opcode::illegal: return "illegal";
        case Opcode::lb: return "lb";
        case Opcode::lh: return "lh";
        case Opcode::lw: return "lw";
        case Opcode::ld: return "ld";
        case Opcode::lbu: return "lbu";
        case Opcode::lhu: return "lhu";
        case Opcode::lwu: return "lwu";
        case Opcode::fence: return "fence";
        case Opcode::fence_i: return "fence.i";
        case Opcode::addi: return "addi";
        case Opcode::slli: return "slli";
        case Opcode::slti: return "slti";
        case Opcode::sltiu: return "sltiu";
        case Opcode::xori: return "xori";
        case Opcode::srli: return "srli";
        case Opcode::srai: return "srai";
        case Opcode::ori: return "ori";
        case Opcode::andi: return "andi";
        case Opcode::auipc: return "auipc";
        case Opcode::addiw: return "addiw";
        case Opcode::slliw: return "slliw";
        case Opcode::srliw: return "srliw";
        case Opcode::sraiw: return "sraiw";
        case Opcode::sb: return "sb";
        case Opcode::sh: return "sh";
        case Opcode::sw: return "sw";
        case Opcode::sd: return "sd";
        case Opcode::add: return "add";
        case Opcode::sub: return "sub";
        case Opcode::sll: return "sll";
        case Opcode::slt: return "slt";
        case Opcode::sltu: return "sltu";
        case Opcode::i_xor: return "xor";
        case Opcode::srl: return "srl";
        case Opcode::sra: return "sra";
        case Opcode::i_or: return "or";
        case Opcode::i_and: return "and";
        case Opcode::lui: return "lui";
        case Opcode::addw: return "addw";
        case Opcode::subw: return "subw";
        case Opcode::sllw: return "sllw";
        case Opcode::srlw: return "srlw";
        case Opcode::sraw: return "sraw";
        case Opcode::beq: return "beq";
        case Opcode::bne: return "bne";
        case Opcode::blt: return "blt";
        case Opcode::bge: return "bge";
        case Opcode::bltu: return "bltu";
        case Opcode::bgeu: return "bgeu";
        case Opcode::jalr: return "jalr";
        case Opcode::jal: return "jal";
        case Opcode::ecall: return "ecall";
        case Opcode::ebreak: return "ebreak";
        case Opcode::csrrw: return "csrrw";
        case Opcode::csrrs: return "csrrs";
        case Opcode::csrrc: return "csrrc";
        case Opcode::csrrwi: return "csrrwi";
        case Opcode::csrrsi: return "csrrsi";
        case Opcode::csrrci: return "csrrci";
        case Opcode::mul: return "mul";
        case Opcode::mulh: return "mulh";
        case Opcode::mulhsu: return "mulhsu";
        case Opcode::mulhu: return "mulhu";
        case Opcode::div: return "div";
        case Opcode::divu: return "divu";
        case Opcode::rem: return "rem";
        case Opcode::remu: return "remu";
        case Opcode::mulw: return "mulw";
        case Opcode::divw: return "divw";
        case Opcode::divuw: return "divuw";
        case Opcode::remw: return "remw";
        case Opcode::remuw: return "remuw";
        case Opcode::lr_w: return "lr.w";
        case Opcode::lr_d: return "lr.d";
        case Opcode::sc_w: return "sc.w";
        case Opcode::sc_d: return "sc.d";
        case Opcode::amoswap_w: return "amoswap.w";
        case Opcode::amoswap_d: return "amoswap.d";
        case Opcode::amoadd_w: return "amoadd.w";
        case Opcode::amoadd_d: return "amoadd.d";
        case Opcode::amoxor_w: return "amoxor.w";
        case Opcode::amoxor_d: return "amoxor.d";
        case Opcode::amoand_w: return "amoand.w";
        case Opcode::amoand_d: return "amoand.d";
        case Opcode::amoor_w: return "amoor.w";
        case Opcode::amoor_d: return "amoor.d";
        case Opcode::amomin_w: return "amomin.w";
        case Opcode::amomin_d: return "amomin.d";
        case Opcode::amomax_w: return "amomax.w";
        case Opcode::amomax_d: return "amomax.d";
        case Opcode::amominu_w: return "amominu.w";
        case Opcode::amominu_d: return "amominu.d";
        case Opcode::amomaxu_w: return "amomaxu.w";
        case Opcode::amomaxu_d: return "amomaxu.d";
        case Opcode::flw: return "flw";
        case Opcode::fsw: return "fsw";
        case Opcode::fadd_s: return "fadd.s";
        case Opcode::fsub_s: return "fsub.s";
        case Opcode::fmul_s: return "fmul.s";
        case Opcode::fdiv_s: return "fdiv.s";
        case Opcode::fsqrt_s: return "fsqrt.s";
        case Opcode::fsgnj_s: return "fsgnj.s";
        case Opcode::fsgnjn_s: return "fsgnjn.s";
        case Opcode::fsgnjx_s: return "fsgnjx.s";
        case Opcode::fmin_s: return "fmin.s";
        case Opcode::fmax_s: return "fmax.s";
        case Opcode::fcvt_w_s: return "fcvt.w.s";
        case Opcode::fcvt_wu_s: return "fcvt.wu.s";
        case Opcode::fcvt_l_s: return "fcvt.l.s";
        case Opcode::fcvt_lu_s: return "fcvt.lu.s";
        case Opcode::fmv_x_w: return "fmv.x.w";
        case Opcode::feq_s: return "feq.s";
        case Opcode::flt_s: return "flt.s";
        case Opcode::fle_s: return "fle.s";
        case Opcode::fclass_s: return "fclass.s";
        case Opcode::fcvt_s_w: return "fcvt.s.w";
        case Opcode::fcvt_s_wu: return "fcvt.s.wu";
        case Opcode::fcvt_s_l: return "fcvt.s.l";
        case Opcode::fcvt_s_lu: return "fcvt.s.lu";
        case Opcode::fmv_w_x: return "fmv.w.x";
        case Opcode::fmadd_s: return "fmadd.s";
        case Opcode::fmsub_s: return "fmsub.s";
        case Opcode::fnmsub_s: return "fnmsub.s";
        case Opcode::fnmadd_s: return "fnmadd.s";
        case Opcode::fld: return "fld";
        case Opcode::fsd: return "fsd";
        case Opcode::fadd_d: return "fadd.d";
        case Opcode::fsub_d: return "fsub.d";
        case Opcode::fmul_d: return "fmul.d";
        case Opcode::fdiv_d: return "fdiv.d";
        case Opcode::fsqrt_d: return "fsqrt.d";
        case Opcode::fsgnj_d: return "fsgnj.d";
        case Opcode::fsgnjn_d: return "fsgnjn.d";
        case Opcode::fsgnjx_d: return "fsgnjx.d";
        case Opcode::fmin_d: return "fmin.d";
        case Opcode::fmax_d: return "fmax.d";
        case Opcode::fcvt_s_d: return "fcvt.s.d";
        case Opcode::fcvt_d_s: return "fcvt.d.s";
        case Opcode::feq_d: return "feq.d";
        case Opcode::flt_d: return "flt.d";
        case Opcode::fle_d: return "fle.d";
        case Opcode::fclass_d: return "fclass.d";
        case Opcode::fcvt_w_d: return "fcvt.w.d";
        case Opcode::fcvt_wu_d: return "fcvt.wu.d";
        case Opcode::fcvt_l_d: return "fcvt.l.d";
        case Opcode::fcvt_lu_d: return "fcvt.lu.d";
        case Opcode::fmv_x_d: return "fmv.x.d";
        case Opcode::fcvt_d_w: return "fcvt.d.w";
        case Opcode::fcvt_d_wu: return "fcvt.d.wu";
        case Opcode::fcvt_d_l: return "fcvt.d.l";
        case Opcode::fcvt_d_lu: return "fcvt.d.lu";
        case Opcode::fmv_d_x: return "fmv.d.x";
        case Opcode::fmadd_d: return "fmadd.d";
        case Opcode::fmsub_d: return "fmsub.d";
        case Opcode::fnmsub_d: return "fnmsub.d";
        case Opcode::fnmadd_d: return "fnmadd.d";
        default: return "(unknown)";
    }
}

void Disassembler::print_instruction(reg_t pc, uint32_t bits, Instruction inst) {
    Opcode opcode = inst.opcode();
    const char *opcode_name = Disassembler::opcode_name(opcode);

    if ((pc & 0xFFFFFFFF) == pc) {
        util::log("{:8x}:       ", pc);
    } else {
        util::log("{:16x}:       ", pc);
    }

    if (inst.length() == 4) {
        util::log("{:08x}", bits);
    } else {
        util::log("{:04x}    ", bits & 0xFFFF);
    }

    util::log("        {:-7} ", opcode_name);

    // Since most instructions use sign-extension, convert it to signed beforehand.
    sreg_t imm = inst.imm();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();
    int rd = inst.rd();

    switch (opcode) {
        case Opcode::lui:
        case Opcode::auipc:
            util::log("{}, {:#x}",  register_name(rd), static_cast<uint32_t>(imm) >> 12);
            break;
        case Opcode::jal: {
            reg_t target_pc = pc + imm;
            char sign = '+';
            if (imm < 0) {
                imm = -imm;
                sign = '-';
            }
            util::log("{}, pc {} {} <{:x}>",  register_name(rd), sign, imm, target_pc);
            break;
        }
        case Opcode::beq:
        case Opcode::bne:
        case Opcode::blt:
        case Opcode::bge:
        case Opcode::bltu:
        case Opcode::bgeu: {
            reg_t target_pc = pc + imm;
            char sign = '+';
            if (imm < 0) {
                imm = -imm;
                sign = '-';
            }
            util::log("{}, {}, pc {} {} <{:x}>",  register_name(rs1), register_name(rs2), sign, imm, target_pc);
            break;
        }
        case Opcode::lb:
        case Opcode::lh:
        case Opcode::lw:
        case Opcode::ld:
        case Opcode::lbu:
        case Opcode::lhu:
        case Opcode::lwu:
        // jalr has same string representation as load instructions.
        case Opcode::jalr:
            util::log("{}, {}({})", register_name(rd), imm, register_name(rs1));
            break;
        // TODO: display the arguments of fence?
        case Opcode::fence:
        case Opcode::fence_i:
        case Opcode::ecall:
        case Opcode::ebreak:
            break;
        case Opcode::sb:
        case Opcode::sh:
        case Opcode::sw:
        case Opcode::sd:
            util::log("{}, {}({})", register_name(rs2), imm, register_name(rs1));
            break;
        case Opcode::addi:
        case Opcode::slti:
        case Opcode::sltiu:
        case Opcode::xori:
        case Opcode::ori:
        case Opcode::andi:
        case Opcode::addiw:
        // The shifts technically should have a unsigned argument, but since immediates for shifts are small numbers,
        // converting to sreg_t does not hurt.
        case Opcode::slli:
        case Opcode::srli:
        case Opcode::srai:
        case Opcode::slliw:
        case Opcode::srliw:
        case Opcode::sraiw:
            util::log("{}, {}, {}", register_name(rd), register_name(rs1), imm);
            break;
        case Opcode::add:
        case Opcode::sub:
        case Opcode::sll:
        case Opcode::slt:
        case Opcode::sltu:
        case Opcode::i_xor:
        case Opcode::srl:
        case Opcode::sra:
        case Opcode::i_or:
        case Opcode::i_and:
        case Opcode::addw:
        case Opcode::subw:
        case Opcode::sllw:
        case Opcode::srlw:
        case Opcode::sraw:
        case Opcode::mul:
        case Opcode::mulh:
        case Opcode::mulhsu:
        case Opcode::mulhu:
        case Opcode::div:
        case Opcode::divu:
        case Opcode::rem:
        case Opcode::remu:
        case Opcode::mulw:
        case Opcode::divw:
        case Opcode::divuw:
        case Opcode::remw:
        case Opcode::remuw:
            util::log("{}, {}, {}", register_name(rd), register_name(rs1), register_name(rs2));
            break;
        // CSR instructions store immediates differently.
        case Opcode::csrrw:
        case Opcode::csrrs:
        case Opcode::csrrc:
            util::log("{}, #{}, {}", register_name(rd), csr_name(static_cast<Csr>(imm)), register_name(rs1));
            break;
        case Opcode::csrrwi:
        case Opcode::csrrsi:
        case Opcode::csrrci:
            util::log("{}, #{}, {}", register_name(rd), csr_name(static_cast<Csr>(imm)), rs1);
            break;
        // TODO: For atomic instructions we may want to display their aq, rl arguments?
        case Opcode::lr_w:
        case Opcode::lr_d:
            util::log("{}, ({})", register_name(rd), register_name(rs1));
            break;
        case Opcode::sc_w:
        case Opcode::sc_d:
        case Opcode::amoswap_w:
        case Opcode::amoswap_d:
        case Opcode::amoadd_w:
        case Opcode::amoadd_d:
        case Opcode::amoxor_w:
        case Opcode::amoxor_d:
        case Opcode::amoand_w:
        case Opcode::amoand_d:
        case Opcode::amoor_w:
        case Opcode::amoor_d:
        case Opcode::amomin_w:
        case Opcode::amomin_d:
        case Opcode::amomax_w:
        case Opcode::amomax_d:
        case Opcode::amominu_w:
        case Opcode::amominu_d:
        case Opcode::amomaxu_w:
        case Opcode::amomaxu_d:
            util::log("{}, {}, ({})", register_name(rd), register_name(rs2), register_name(rs1));
            break;
        // TODO: For floating point arguments we may want to display their r/m arguments?
        case Opcode::flw:
        case Opcode::fld:
            util::log("f{}, {}({})", rd, imm, register_name(rs1));
            break;
        case Opcode::fsw:
        case Opcode::fsd:
            util::log("f{}, {}({})", rs2, imm, register_name(rs1));
            break;
        case Opcode::fadd_s:
        case Opcode::fsub_s:
        case Opcode::fmul_s:
        case Opcode::fdiv_s:
        case Opcode::fsgnj_s:
        case Opcode::fsgnjn_s:
        case Opcode::fsgnjx_s:
        case Opcode::fmin_s:
        case Opcode::fmax_s:
        case Opcode::fadd_d:
        case Opcode::fsub_d:
        case Opcode::fmul_d:
        case Opcode::fdiv_d:
        case Opcode::fsgnj_d:
        case Opcode::fsgnjn_d:
        case Opcode::fsgnjx_d:
        case Opcode::fmin_d:
        case Opcode::fmax_d:
            util::log("f{}, f{}, f{}", rd, rs1, rs2);
            break;
        case Opcode::fsqrt_s:
        case Opcode::fsqrt_d:
        case Opcode::fcvt_s_d:
        case Opcode::fcvt_d_s:
            util::log("f{}, f{}", rd, rs1);
            break;
        case Opcode::fcvt_w_s:
        case Opcode::fcvt_wu_s:
        case Opcode::fcvt_l_s:
        case Opcode::fcvt_lu_s:
        case Opcode::fmv_x_w:
        case Opcode::fclass_s:
        case Opcode::fcvt_w_d:
        case Opcode::fcvt_wu_d:
        case Opcode::fcvt_l_d:
        case Opcode::fcvt_lu_d:
        case Opcode::fmv_x_d:
        case Opcode::fclass_d:
            util::log("{}, f{}", register_name(rd), rs1);
            break;
        case Opcode::fcvt_s_w:
        case Opcode::fcvt_s_wu:
        case Opcode::fcvt_s_l:
        case Opcode::fcvt_s_lu:
        case Opcode::fmv_w_x:
        case Opcode::fcvt_d_w:
        case Opcode::fcvt_d_wu:
        case Opcode::fcvt_d_l:
        case Opcode::fcvt_d_lu:
        case Opcode::fmv_d_x:
            util::log("f{}, {}", rd, register_name(rs1));
            break;
        case Opcode::feq_s:
        case Opcode::flt_s:
        case Opcode::fle_s:
        case Opcode::feq_d:
        case Opcode::flt_d:
        case Opcode::fle_d:
            util::log("{}, f{}, f{}", register_name(rd), rs1, rs2);
            break;
        case Opcode::fmadd_s:
        case Opcode::fmsub_s:
        case Opcode::fnmsub_s:
        case Opcode::fnmadd_s:
        case Opcode::fmadd_d:
        case Opcode::fmsub_d:
        case Opcode::fnmsub_d:
        case Opcode::fnmadd_d:
            util::log("f{}, f{}, f{}, f{}", rd, rs1, rs2, inst.rs3());
            break;

        case Opcode::illegal:
            break;
    }

    std::clog << std::endl;
}
