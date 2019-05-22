#include <iostream>

#include "emu/mmu.h"
#include "emu/state.h"
#include "riscv/context.h"
#include "riscv/csr.h"
#include "riscv/instruction.h"
#include "riscv/opcode.h"
#include "softfp/float.h"
#include "util/memory.h"

namespace riscv::abi { 
    enum class Syscall_number;
}

namespace emu {
    reg_t syscall(riscv::abi::Syscall_number, reg_t, reg_t, reg_t, reg_t, reg_t, reg_t);
}

namespace riscv {

// Helper functions for interpreter

static inline uint64_t sign_ext8(uint8_t value) {
    return static_cast<int64_t>(static_cast<int8_t>(value));
}

static inline uint64_t sign_ext16(uint16_t value) {
    return static_cast<int64_t>(static_cast<int16_t>(value));
}

static inline uint64_t sign_ext(uint32_t value) {
    return static_cast<int64_t>(static_cast<int32_t>(value));
}

static inline uint64_t zero_ext8(uint8_t value) {
    return value;
}

static inline uint64_t zero_ext16(uint16_t value) {
    return value;
}

static inline uint64_t zero_ext(uint32_t value) {
    return value;
}

template<typename T>
static inline T load_memory(reg_t address) {
    if (address >= 0x80000000) return mmio_read(address - 0x80000000);
    return *(T*)(emu::translate_address(address));
}

static_assert(sizeof(freg_t) == 8);

static inline void write_double(freg_t& target, softfp::Double value) {
    target = util::read_as<uint64_t>(&value);
}

static inline void write_single(freg_t& target, softfp::Single value) {
    target = util::read_as<uint32_t>(&value) | 0xFFFFFFFF00000000;
}

static inline softfp::Double read_double(freg_t& target) {
    return util::read_as<softfp::Double>(&target);
}

static inline softfp::Single read_single(freg_t& target) {
    return util::read_as<softfp::Single>(&target);
}

static inline void set_rm_real(int rm) {
    if (rm >= 5) {
        throw Trap { Cause::illegal_inst };
    }
    softfp::rounding_mode = static_cast<softfp::Rounding_mode>(rm);
}

#define read_rs1() context->registers[inst.rs1()]
#define read_rs2() context->registers[inst.rs2()]
#define write_rd(_value) do { \
        int rd = inst.rd(); \
        uint64_t _saved_value = _value; \
        if (rd != 0) context->registers[rd] = _saved_value; \
    } while(0);

#define read_frs1_d() read_double(context->fp_registers[inst.rs1()])
#define read_frs2_d() read_double(context->fp_registers[inst.rs2()])
#define write_frd_d(value) do { \
        write_double(context->fp_registers[inst.rd()], value); \
    } while(0);

#define read_frs1_s() read_single(context->fp_registers[inst.rs1()])
#define read_frs2_s() read_single(context->fp_registers[inst.rs2()])
#define write_frd_s(value) do { \
        write_single(context->fp_registers[inst.rd()], value); \
    } while(0);

#define read_frs3_s() read_single(context->fp_registers[inst.rs3()])
#define read_frs3_d() read_double(context->fp_registers[inst.rs3()])

#define set_rm() do { \
        set_rm_real(inst.rm() == 0b111 ? (context->fcsr >> 5) : inst.rm()); \
    } while (0);

#define clear_flags() do {\
        softfp::exception_flags = softfp::Exception_flag::none; \
    } while (0);

#define update_flags() do { \
        context->fcsr |= static_cast<int>(softfp::exception_flags); \
    } while (0);

// Instruction pointers are assumed to move *past* the instruction already.
void step(Context *context, Instruction inst) {

    // IMPORTANT: All bit pattern must be represented using 
    // unsigned integers. Signed integer overflows are considered
    // undefined behavior. If we need signedness, convert them
    // to signed when necessary.
    switch (inst.opcode()) {
        /* LOAD */
        case Opcode::lb:
            write_rd(sign_ext8(load_memory<uint8_t>(translate(context, read_rs1() + inst.imm(), false))));
            break;
        case Opcode::lh:
            write_rd(sign_ext16(load_memory<uint16_t>(translate(context, read_rs1() + inst.imm(), false))));
            break;
        case Opcode::lw:
            write_rd(sign_ext(load_memory<uint32_t>(translate(context, read_rs1() + inst.imm(), false))));
            break;
        case Opcode::ld:
            write_rd(load_memory<uint64_t>(translate(context, read_rs1() + inst.imm(), false)));
            break;
        case Opcode::lbu:
            write_rd(zero_ext8(load_memory<uint8_t>(translate(context, read_rs1() + inst.imm(), false))));
            break;
        case Opcode::lhu:
            write_rd(zero_ext16(load_memory<uint16_t>(translate(context, read_rs1() + inst.imm(), false))));
            break;
        case Opcode::lwu:
            write_rd(zero_ext(load_memory<uint32_t>(translate(context, read_rs1() + inst.imm(), false))));
            break;
        /* MISC-MEM */
        /* OP-IMM */
        /* AUIPC */
        /* OP-IMM-32 */
        /* STORE */
        case Opcode::sb:
            emu::store_memory<uint8_t>(translate(context, read_rs1() + inst.imm(), true), read_rs2());
            break;
        case Opcode::sh:
            emu::store_memory<uint16_t>(translate(context, read_rs1() + inst.imm(), true), read_rs2());
            break;
        case Opcode::sw:
            emu::store_memory<uint32_t>(translate(context, read_rs1() + inst.imm(), true), read_rs2());
            break;
        case Opcode::sd:
            emu::store_memory<uint64_t>(translate(context, read_rs1() + inst.imm(), true), read_rs2());
            break;
        /* OP */
        /* LUI */
        /* OP-32 */
        /* BRANCH */
        /* JALR */
        /* JAL */
        /* SYSTEM */

        /* M-extension */

        /* A-extension */
        // Stub implementations. Single thread only.
        case Opcode::lr_w: {
            reg_t addr = read_rs1();
            write_rd(sign_ext(load_memory<uint32_t>(translate(context, addr, true))));
            context->lr = addr;
            break;
        }
        case Opcode::lr_d: {
            reg_t addr = read_rs1();
            write_rd(load_memory<uint64_t>(translate(context, addr, true)));
            context->lr = addr;
            break;
        }
        case Opcode::sc_w: {
            reg_t addr = read_rs1();
            if (addr != context->lr) {
                write_rd(1);
                return;
            }
            emu::store_memory<uint32_t>(translate(context, addr, true), read_rs2());
            write_rd(0);
            break;
        }
        case Opcode::sc_d: {
            reg_t addr = read_rs1();
            if (addr != context->lr) {
                write_rd(1);
                return;
            }
            emu::store_memory<uint64_t>(translate(context, addr, true), read_rs2());
            write_rd(0);
            break;
        }
        case Opcode::amoswap_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            if (inst.rd() != 0) {
                write_rd(sign_ext(load_memory<uint32_t>(translate(context, addr, true))));
            }
            emu::store_memory<uint32_t>(translate(context, addr, true), src);
            break;
        }
        case Opcode::amoswap_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            if (inst.rd() != 0) {
                write_rd(load_memory<uint64_t>(translate(context, addr, true)));
            }
            emu::store_memory<uint64_t>(translate(context, addr, true), src);
            break;
        }
        case Opcode::amoadd_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = load_memory<uint32_t>(translate(context, addr, true));
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(translate(context, addr, true), src + mem);
            break;
        }
        case Opcode::amoadd_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = load_memory<uint64_t>(translate(context, addr, true));
            write_rd(mem);
            emu::store_memory<uint64_t>(translate(context, addr, true), src + mem);
            break;
        }
        case Opcode::amoand_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = load_memory<uint32_t>(translate(context, addr, true));
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(translate(context, addr, true), src & mem);
            break;
        }
        case Opcode::amoand_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = load_memory<uint64_t>(translate(context, addr, true));
            write_rd(mem);
            emu::store_memory<uint64_t>(translate(context, addr, true), src & mem);
            break;
        }
        case Opcode::amoor_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = load_memory<uint32_t>(translate(context, addr, true));
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(translate(context, addr, true), src | mem);
            break;
        }
        case Opcode::amoor_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = load_memory<uint64_t>(translate(context, addr, true));
            write_rd(mem);
            emu::store_memory<uint64_t>(translate(context, addr, true), src | mem);
            break;
        }
        case Opcode::amoxor_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = load_memory<uint32_t>(translate(context, addr, true));
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(translate(context, addr, true), src ^ mem);
            break;
        }
        case Opcode::amoxor_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = load_memory<uint64_t>(translate(context, addr, true));
            write_rd(mem);
            emu::store_memory<uint64_t>(translate(context, addr, true), src ^ mem);
            break;
        }
        case Opcode::amomin_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = load_memory<uint32_t>(translate(context, addr, true));
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(translate(context, addr, true), std::min<int32_t>(src, mem));
            break;
        }
        case Opcode::amomin_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = load_memory<uint64_t>(translate(context, addr, true));
            write_rd(mem);
            emu::store_memory<uint64_t>(translate(context, addr, true), std::min<int64_t>(src, mem));
            break;
        }
        case Opcode::amomax_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = load_memory<uint32_t>(translate(context, addr, true));
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(translate(context, addr, true), std::max<int32_t>(src, mem));
            break;
        }
        case Opcode::amomax_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = load_memory<uint64_t>(translate(context, addr, true));
            write_rd(mem);
            emu::store_memory<uint64_t>(translate(context, addr, true), std::max<int64_t>(src, mem));
            break;
        }
        case Opcode::amominu_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = load_memory<uint32_t>(translate(context, addr, true));
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(translate(context, addr, true), std::min<uint32_t>(src, mem));
            break;
        }
        case Opcode::amominu_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = load_memory<uint64_t>(translate(context, addr, true));
            write_rd(mem);
            emu::store_memory<uint64_t>(translate(context, addr, true), std::min<uint64_t>(src, mem));
            break;
        }
        case Opcode::amomaxu_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = load_memory<uint32_t>(translate(context, addr, true));
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(translate(context, addr, true), std::max<uint32_t>(src, mem));
            break;
        }
        case Opcode::amomaxu_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = load_memory<uint64_t>(translate(context, addr, true));
            write_rd(mem);
            emu::store_memory<uint64_t>(translate(context, addr, true), std::max<uint64_t>(src, mem));
            break;
        }

        /* F-extension */
        case Opcode::flw: {
            uint32_t value = load_memory<uint32_t>(translate(context, read_rs1() + inst.imm(), false));
            write_frd_s(util::read_as<softfp::Single>(&value));
            break;
        }
        case Opcode::fsw: {
            softfp::Single value = read_frs2_s();
            emu::store_memory<uint32_t>(translate(context, read_rs1() + inst.imm(), true), util::read_as<uint32_t>(&value));
            break;
        }
        case Opcode::fadd_s:
            set_rm();
            clear_flags();
            write_frd_s(read_frs1_s() + read_frs2_s());
            update_flags();
            break;
        case Opcode::fsub_s:
            set_rm();
            clear_flags();
            write_frd_s(read_frs1_s() - read_frs2_s());
            update_flags();
            break;
        case Opcode::fmul_s:
            set_rm();
            clear_flags();
            write_frd_s(read_frs1_s() * read_frs2_s());
            update_flags();
            break;
        case Opcode::fdiv_s:
            set_rm();
            clear_flags();
            write_frd_s(read_frs1_s() / read_frs2_s());
            update_flags();
            break;
        case Opcode::fsqrt_s:
            set_rm();
            clear_flags();
            write_frd_s(read_frs1_s().square_root());
            update_flags();
            break;
        case Opcode::fsgnj_s: 
            write_frd_s(read_frs1_s().copy_sign(read_frs2_s()));
            break;
        case Opcode::fsgnjn_s:
            write_frd_s(read_frs1_s().copy_sign_negated(read_frs2_s()));
            break;
        case Opcode::fsgnjx_s:
            write_frd_s(read_frs1_s().copy_sign_xored(read_frs2_s()));
            break;
        case Opcode::fmin_s:
            clear_flags();
            write_frd_s(softfp::Single::min(read_frs1_s(), read_frs2_s()));
            update_flags();
            break;
        case Opcode::fmax_s:
            clear_flags();
            write_frd_s(softfp::Single::max(read_frs1_s(), read_frs2_s()));
            update_flags();
            break;
        case Opcode::fcvt_w_s:
            set_rm();
            clear_flags();
            write_rd(sign_ext(read_frs1_s().convert_to_int<int32_t>()));
            update_flags();
            break;
        case Opcode::fcvt_wu_s:
            set_rm();
            clear_flags();
            write_rd(sign_ext(read_frs1_s().convert_to_int<uint32_t>()));
            update_flags();
            break;
        case Opcode::fcvt_l_s:
            set_rm();
            clear_flags();
            write_rd(read_frs1_s().convert_to_int<int64_t>());
            update_flags();
            break;
        case Opcode::fcvt_lu_s:
            set_rm();
            clear_flags();
            write_rd(read_frs1_s().convert_to_int<uint64_t>());
            update_flags();
            break;
        case Opcode::fmv_x_w: {
            softfp::Single value = read_frs1_s();
            write_rd(sign_ext(util::read_as<uint32_t>(&value)));
            break;
        }
        case Opcode::fcvt_s_w:
            set_rm();
            clear_flags();
            write_frd_s(softfp::Single::convert_from_int<int32_t>(read_rs1()));
            update_flags();
            break;
        case Opcode::fcvt_s_wu:
            set_rm();
            clear_flags();
            write_frd_s(softfp::Single::convert_from_int<uint32_t>(read_rs1()));
            update_flags();
            break;
        case Opcode::fcvt_s_l:
            set_rm();
            clear_flags();
            write_frd_s(softfp::Single::convert_from_int<int64_t>(read_rs1()));
            update_flags();
            break;
        case Opcode::fcvt_s_lu:
            set_rm();
            clear_flags();
            write_frd_s(softfp::Single::convert_from_int<uint64_t>(read_rs1()));
            update_flags();
            break;
        case Opcode::feq_s:
            write_rd(read_frs1_s() == read_frs2_s());
            break;
        case Opcode::flt_s:
            clear_flags();
            write_rd(read_frs1_s() < read_frs2_s());
            update_flags();
            break;
        case Opcode::fle_s:
            clear_flags();
            write_rd(read_frs1_s() <= read_frs2_s());
            update_flags();
            break;
        case Opcode::fclass_s: {
            softfp::Class category = read_frs1_s().classify();
        
            // Class is a number in [0, 9] where the expected result is a bit mask.
            write_rd(1 << static_cast<int>(category));
            break;
        }
        case Opcode::fmv_w_x: {
            reg_t value = read_rs1();
            write_frd_s(util::read_as<softfp::Single>(&value));
            break;
        }
        case Opcode::fmadd_s:
            set_rm();
            clear_flags();
            write_frd_s(softfp::Single::fused_multiply_add(read_frs1_s(), read_frs2_s(), read_frs3_s()));
            update_flags();
            break;
        case Opcode::fmsub_s:
            set_rm();
            clear_flags();
            write_frd_s(softfp::Single::fused_multiply_add(read_frs1_s(), read_frs2_s(), -read_frs3_s()));
            update_flags();
            break;
        case Opcode::fnmsub_s:
            set_rm();
            clear_flags();
            write_frd_s(-softfp::Single::fused_multiply_add(read_frs1_s(), read_frs2_s(), -read_frs3_s()));
            update_flags();
            break;
        case Opcode::fnmadd_s:
            set_rm();
            clear_flags();
            write_frd_s(-softfp::Single::fused_multiply_add(read_frs1_s(), read_frs2_s(), read_frs3_s()));
            update_flags();
            break;

        /* D-extension */
        case Opcode::fld: {
            uint64_t value = load_memory<uint64_t>(translate(context, read_rs1() + inst.imm(), false));
            write_frd_d(util::read_as<softfp::Double>(&value));
            break;
        }
        case Opcode::fsd: {
            softfp::Double value = read_frs2_d();
            emu::store_memory<uint64_t>(translate(context, read_rs1() + inst.imm(), true), util::read_as<uint64_t>(&value));
            break;
        }
        case Opcode::fadd_d:
            set_rm();
            clear_flags();
            write_frd_d(read_frs1_d() + read_frs2_d());
            update_flags();
            break;
        case Opcode::fsub_d:
            set_rm();
            clear_flags();
            write_frd_d(read_frs1_d() - read_frs2_d());
            update_flags();
            break;
        case Opcode::fmul_d:
            set_rm();
            clear_flags();
            write_frd_d(read_frs1_d() * read_frs2_d());
            update_flags();
            break;
        case Opcode::fdiv_d:
            set_rm();
            clear_flags();
            write_frd_d(read_frs1_d() / read_frs2_d());
            update_flags();
            break;
        case Opcode::fsqrt_d:
            set_rm();
            clear_flags();
            write_frd_d(read_frs1_d().square_root());
            update_flags();
            break;
        case Opcode::fsgnj_d:
            write_frd_d(read_frs1_d().copy_sign(read_frs2_d()));
            break;
        case Opcode::fsgnjn_d:
            write_frd_d(read_frs1_d().copy_sign_negated(read_frs2_d()));
            break;
        case Opcode::fsgnjx_d:
            write_frd_d(read_frs1_d().copy_sign_xored(read_frs2_d()));
            break;
        case Opcode::fmin_d:
            clear_flags();
            write_frd_d(softfp::Double::min(read_frs1_d(), read_frs2_d()));
            update_flags();
            break;
        case Opcode::fmax_d:
            clear_flags();
            write_frd_d(softfp::Double::max(read_frs1_d(), read_frs2_d()));
            update_flags();
            break;
        case Opcode::fcvt_s_d:
            set_rm();
            clear_flags();
            write_frd_s(read_frs1_d().convert_format<softfp::Single>());
            update_flags();
            break;
        case Opcode::fcvt_d_s:
            clear_flags();
            write_frd_d(read_frs1_s().convert_format<softfp::Double>());
            update_flags();
            break;
        case Opcode::fcvt_w_d:
            set_rm();
            clear_flags();
            write_rd(sign_ext(read_frs1_d().convert_to_int<int32_t>()));
            update_flags();
            break;
        case Opcode::fcvt_wu_d:
            set_rm();
            clear_flags();
            write_rd(sign_ext(read_frs1_d().convert_to_int<uint32_t>()));
            update_flags();
            break;
        case Opcode::fcvt_l_d:
            set_rm();
            clear_flags();
            write_rd(read_frs1_d().convert_to_int<int64_t>());
            update_flags();
            break;
        case Opcode::fcvt_lu_d:
            set_rm();
            clear_flags();
            write_rd(read_frs1_d().convert_to_int<uint64_t>());
            update_flags();
            break;
        case Opcode::fmv_x_d: {
            softfp::Double value = read_frs1_d();
            write_rd(util::read_as<uint64_t>(&value));
            break;
        }
        case Opcode::fcvt_d_w:
            set_rm();
            clear_flags();
            write_frd_d(softfp::Double::convert_from_int<int32_t>(read_rs1()));
            update_flags();
            break;
        case Opcode::fcvt_d_wu:
            set_rm();
            clear_flags();
            write_frd_d(softfp::Double::convert_from_int<uint32_t>(read_rs1()));
            update_flags();
            break;
        case Opcode::fcvt_d_l:
            set_rm();
            clear_flags();
            write_frd_d(softfp::Double::convert_from_int<int64_t>(read_rs1()));
            update_flags();
            break;
        case Opcode::fcvt_d_lu:
            set_rm();
            clear_flags();
            write_frd_d(softfp::Double::convert_from_int<uint64_t>(read_rs1()));
            update_flags();
            break;
        case Opcode::feq_d:
            write_rd(read_frs1_d() == read_frs2_d());
            break;
        case Opcode::flt_d:
            clear_flags();
            write_rd(read_frs1_d() < read_frs2_d());
            update_flags();
            break;
        case Opcode::fle_d:
            clear_flags();
            write_rd(read_frs1_d() <= read_frs2_d());
            update_flags();
            break;
        case Opcode::fclass_d: {
            softfp::Class category = read_frs1_d().classify();
        
            // Class is a number in [0, 9] where the expected result is a bit mask.
            write_rd(1 << static_cast<int>(category));
            break;
        }
        case Opcode::fmv_d_x: {
            reg_t value = read_rs1();
            write_frd_d(util::read_as<softfp::Double>(&value));
            break;
        }
        case Opcode::fmadd_d:
            set_rm();
            clear_flags();
            write_frd_d(softfp::Double::fused_multiply_add(read_frs1_d(), read_frs2_d(), read_frs3_d()));
            update_flags();
            break;
        case Opcode::fmsub_d:
            set_rm();
            clear_flags();
            write_frd_d(softfp::Double::fused_multiply_add(read_frs1_d(), read_frs2_d(), -read_frs3_d()));
            update_flags();
            break;
        case Opcode::fnmsub_d:
            set_rm();
            clear_flags();
            write_frd_d(-softfp::Double::fused_multiply_add(read_frs1_d(), read_frs2_d(), -read_frs3_d()));
            update_flags();
            break;
        case Opcode::fnmadd_d:
            set_rm();
            clear_flags();
            write_frd_d(-softfp::Double::fused_multiply_add(read_frs1_d(), read_frs2_d(), read_frs3_d()));
            update_flags();
            break;
        case Opcode::illegal: {
            auto bin = load_memory<uint32_t>(context->pc - inst.length());
            std::cerr << "Illegal opcode " << std::hex << bin << std::endl;
            throw Trap { Cause::illegal_inst };
        }
    }
}

extern "C" uint64_t legacy_step(riscv::Context* ctx, riscv::Instruction* inst)  {
    try {
        step(ctx, *inst);
        return 0;
    } catch (riscv::Trap& trap){
        ctx->stval = trap.tval;
        return (uint64_t)trap.cause;
    }
}

#define CACHE_LINE_LOG2_SIZE 12

extern "C" reg_t rs_translate(Context* context, reg_t addr, bool write, reg_t& out);

reg_t translate(Context* context, reg_t addr, bool write) {
    auto idx = addr >> CACHE_LINE_LOG2_SIZE;
    auto& line = context->line[idx & 1023];
    if (LIKELY(line.tag == idx)) {
        if (!write || (line.paddr & 1)) {
            return (line.paddr &~ 1) | (addr & ((1 << CACHE_LINE_LOG2_SIZE) - 1));
        }
    }
    reg_t out;
    reg_t ex = rs_translate(context, addr, write, out);
    if (ex) throw Trap { (Cause)ex, addr };
    line.tag = idx;
    line.paddr = out &~ ((1 << CACHE_LINE_LOG2_SIZE) - 1);
    if (write) line.paddr |= 1;
    return out;
}

}
