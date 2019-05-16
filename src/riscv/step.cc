#include <iostream>

#include "emu/mmu.h"
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
        throw "Illegal rounding mode";
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

reg_t read_csr(Context *context, int csr) {
    switch (static_cast<Csr>(csr)) {
        case Csr::fflags:
            return context->fcsr & 0b11111;
        case Csr::frm:
            return (context->fcsr >> 5) & 0b111;
        case Csr::fcsr:
            return context->fcsr;
        case Csr::instret:
            // Assume that instret is incremented already.
            return context->instret - 1;
        default:
            std::cerr << "READ CSR " << csr << std::endl;
            throw "Illegal CSR";
    }
}

void write_csr(Context *context, int csr, reg_t value) {
    switch (static_cast<Csr>(csr)) {
        case Csr::fflags:
            context->fcsr = (context->fcsr &~ 0b11111) | (value & 0b11111);
            break;
        case Csr::frm:
            context->fcsr = (context->fcsr &~ (0b111 << 5)) | ((value & 0b111) << 5);
            break;
        case Csr::fcsr:
            context->fcsr = value & ((1 << 8) - 1);
            break;
        case Csr::instret:
            context->instret = value;
            break;
        default:
            std::cerr << "WRITE CSR " << csr << std::endl;
            throw "Illegal CSR";
    }
}

// Instruction pointers are assumed to move *past* the instruction already.
void step(Context *context, Instruction inst) {

    // IMPORTANT: All bit pattern must be represented using 
    // unsigned integers. Signed integer overflows are considered
    // undefined behavior. If we need signedness, convert them
    // to signed when necessary.
    switch (inst.opcode()) {
        /* LOAD */
        case Opcode::lb:
            write_rd(sign_ext8(emu::load_memory<uint8_t>(read_rs1() + inst.imm())));
            break;
        case Opcode::lh:
            write_rd(sign_ext16(emu::load_memory<uint16_t>(read_rs1() + inst.imm())));
            break;
        case Opcode::lw:
            write_rd(sign_ext(emu::load_memory<uint32_t>(read_rs1() + inst.imm())));
            break;
        case Opcode::ld:
            write_rd(emu::load_memory<uint64_t>(read_rs1() + inst.imm()));
            break;
        case Opcode::lbu:
            write_rd(zero_ext8(emu::load_memory<uint8_t>(read_rs1() + inst.imm())));
            break;
        case Opcode::lhu:
            write_rd(zero_ext16(emu::load_memory<uint16_t>(read_rs1() + inst.imm())));
            break;
        case Opcode::lwu:
            write_rd(zero_ext(emu::load_memory<uint32_t>(read_rs1() + inst.imm())));
            break;
        /* MISC-MEM */
        case Opcode::fence:
            break;
        case Opcode::fence_i:
            context->executor->flush_cache();
            break;
        /* OP-IMM */
        case Opcode::addi:
            write_rd(read_rs1() + inst.imm());
            break;
        case Opcode::slli:
            write_rd(read_rs1() << inst.imm());
            break;
        case Opcode::slti:
            write_rd(static_cast<int64_t>(read_rs1()) < static_cast<int64_t>(inst.imm()));
            break;
        case Opcode::sltiu:
            write_rd(read_rs1() < inst.imm());
            break;
        case Opcode::xori:
            write_rd(read_rs1() ^ inst.imm());
            break;
        case Opcode::srli:
            write_rd(read_rs1() >> inst.imm());
            break;
        case Opcode::srai:
            write_rd(static_cast<int64_t>(read_rs1()) >> inst.imm());
            break;
        case Opcode::ori:
            write_rd(read_rs1() | inst.imm());
            break;
        case Opcode::andi:
            write_rd(read_rs1() & inst.imm());
            break;
        /* AUIPC */
        case Opcode::auipc:
            // PC-relative instructions are relative to the origin pc instead of the incremented one.
            write_rd(context->pc - inst.length() + inst.imm());
            break;
        /* OP-IMM-32 */
        case Opcode::addiw:
            write_rd(sign_ext(read_rs1() + inst.imm()));
            break;
        case Opcode::slliw:
            write_rd(sign_ext(read_rs1() << inst.imm()));
            break;
        case Opcode::srliw:
            write_rd(sign_ext(static_cast<uint32_t>(read_rs1()) >> inst.imm()));
            break;
        case Opcode::sraiw:
            write_rd(sign_ext(static_cast<int32_t>(read_rs1()) >> inst.imm()));
            break;
        /* STORE */
        case Opcode::sb:
            emu::store_memory<uint8_t>(read_rs1() + inst.imm(), read_rs2());
            break;
        case Opcode::sh:
            emu::store_memory<uint16_t>(read_rs1() + inst.imm(), read_rs2());
            break;
        case Opcode::sw:
            emu::store_memory<uint32_t>(read_rs1() + inst.imm(), read_rs2());
            break;
        case Opcode::sd:
            emu::store_memory<uint64_t>(read_rs1() + inst.imm(), read_rs2());
            break;
        /* OP */
        case Opcode::add:
            write_rd(read_rs1() + read_rs2());
            break;
        case Opcode::sub:
            write_rd(read_rs1() - read_rs2());
            break;
        case Opcode::sll:
            write_rd(read_rs1() << (read_rs2() & 63));
            break;
        case Opcode::slt:
            write_rd(static_cast<int64_t>(read_rs1()) < static_cast<int64_t>(read_rs2()));
            break;
        case Opcode::sltu:
            write_rd(read_rs1() < read_rs2());
            break;
        case Opcode::i_xor:
            write_rd(read_rs1() ^ read_rs2());
            break;
        case Opcode::srl:
            write_rd(read_rs1() >> (read_rs2() & 63));
            break;
        case Opcode::sra:
            write_rd(static_cast<int64_t>(read_rs1()) >> (read_rs2() & 63));
            break;
        case Opcode::i_or:
            write_rd(read_rs1() | read_rs2());
            break;
        case Opcode::i_and:
            write_rd(read_rs1() & read_rs2());
            break;
        /* LUI */
        case Opcode::lui:
            write_rd(inst.imm());
            break;
        /* OP-32 */
        case Opcode::addw:
            write_rd(sign_ext(read_rs1() + read_rs2()));
            break;
        case Opcode::subw:
            write_rd(sign_ext(read_rs1() - read_rs2()));
            break;
        case Opcode::sllw:
            write_rd(sign_ext(read_rs1() << (read_rs2() & 31)));
            break;
        case Opcode::srlw:
            write_rd(sign_ext(static_cast<uint32_t>(read_rs1()) >> (read_rs2() & 31)));
            break;
        case Opcode::sraw:
            write_rd(sign_ext(static_cast<int32_t>(read_rs1()) >> (read_rs2() & 31)));
            break;
        /* BRANCH */
        // Same as auipc, PC-relative instructions are relative to the origin pc instead of the incremented one.
        case Opcode::beq:
            if (read_rs1() == read_rs2()) {
                context->pc += -inst.length() + inst.imm();
            }
            break;
        case Opcode::bne:
            if (read_rs1() != read_rs2()) {
                context->pc += -inst.length() + inst.imm();
            }
            break;
        case Opcode::blt:
            if (static_cast<int64_t>(read_rs1()) < static_cast<int64_t>(read_rs2())) {
                context->pc += -inst.length() + inst.imm();
            }
            break;
        case Opcode::bge:
            if (static_cast<int64_t>(read_rs1()) >= static_cast<int64_t>(read_rs2())) {
                context->pc += -inst.length() + inst.imm();
            }
            break;
        case Opcode::bltu:
            if (read_rs1() < read_rs2()) {
                context->pc += -inst.length() + inst.imm();
            }
            break;
        case Opcode::bgeu:
            if (read_rs1() >= read_rs2()) {
                context->pc += -inst.length() + inst.imm();
            }
            break;
        /* JALR */
        case Opcode::jalr: {
            uint64_t new_pc = (read_rs1() + inst.imm()) &~ 1;
            write_rd(context->pc);
            context->pc = new_pc;
            break;
        }
        /* JAL */
        case Opcode::jal:
            write_rd(context->pc);
            context->pc += -inst.length() + inst.imm();
            break;
        /* SYSTEM */
        /* Environment operations */
        case Opcode::ecall:
            context->registers[10] = emu::syscall(
                static_cast<abi::Syscall_number>(context->registers[17]),
                context->registers[10],
                context->registers[11],
                context->registers[12],
                context->registers[13],
                context->registers[14],
                context->registers[15]
            );
            break;
        case Opcode::ebreak:
            throw "Break point";
        /* CSR operations */
        case Opcode::csrrw: {
            int csr = inst.imm();
            uint64_t result = 0;
            if (inst.rd() != 0) result = read_csr(context, csr);
            write_csr(context, csr, read_rs1());
            write_rd(result);
            break;
        }
        case Opcode::csrrs: {
            int csr = inst.imm();
            uint64_t result = read_csr(context, csr);
            write_rd(result);
            if (inst.rs1() != 0) write_csr(context, csr, result | read_rs1());
            break;
        }
        case Opcode::csrrc: {
            int csr = inst.imm();
            uint64_t result = read_csr(context, csr);
            write_rd(result);
            if (inst.rs1() != 0) write_csr(context, csr, result &~ read_rs1());
            break;
        }
        case Opcode::csrrwi: {
            int csr = inst.imm();
            if (inst.rd() != 0) write_rd(read_csr(context, csr));
            write_csr(context, csr, inst.rs1());
            break;
        }
        case Opcode::csrrsi: {
            int csr = inst.imm();
            uint64_t result = read_csr(context, csr);
            write_rd(result);
            if (inst.rs1() != 0) write_csr(context, csr, result | inst.rs1());
            break;
        }
        case Opcode::csrrci: {
            int csr = inst.imm();
            uint64_t result = read_csr(context, csr);
            write_rd(result);
            if (inst.rs1() != 0) write_csr(context, csr, result &~ inst.rs1());
            break;
        }

        /* M-extension */
        case Opcode::mul:
            write_rd(read_rs1() * read_rs2());
            break;
        case Opcode::mulh: {
            util::int128_t a = static_cast<sreg_t>(read_rs1());
            util::int128_t b = static_cast<sreg_t>(read_rs2());
            write_rd((a * b) >> 64);
            break;
        }
        case Opcode::mulhsu: {
            sreg_t rs1 = read_rs1();
            reg_t rs2 = read_rs2();

            // First multiply as uint128_t. This will give compiler chance to optimize better.
            util::uint128_t a = static_cast<reg_t>(rs1);
            util::uint128_t b = rs2;
            reg_t r = (a * b) >> 64;

            // If rs1 < 0, then the high bits of a should be all one, but the actual bits in a is all zero. Therefore
            // we need to compensate this error by adding multiplying 0xFFFFFFFF and b, which is effective -b.
            if (rs1 < 0) r -= rs2;
            write_rd(r);
            break;
        }
        case Opcode::mulhu: {
            util::uint128_t a = read_rs1();
            util::uint128_t b = read_rs2();
            write_rd((a * b) >> 64);
            break;
        }
        case Opcode::div: {
            int64_t operand1 = read_rs1();
            int64_t operand2 = read_rs2();
            if (operand2 == 0) {
                write_rd(-1);
            } else if (operand1 == std::numeric_limits<int64_t>::min() && operand2 == -1) {
                write_rd(operand1);
            } else {
                write_rd(operand1 / operand2);
            }
            break;
        }
        case Opcode::divu: {
            uint64_t operand1 = read_rs1();
            uint64_t operand2 = read_rs2();
            if (operand2 == 0) {
                write_rd(-1);
            } else {
                write_rd(operand1 / operand2);
            }
            break;
        }
        case Opcode::rem: {
            int64_t operand1 = read_rs1();
            int64_t operand2 = read_rs2();
            if (operand2 == 0) {
                write_rd(operand1);
            } else if (operand1 == std::numeric_limits<int64_t>::min() && operand2 == -1) {
                write_rd(0);
            } else {
                write_rd(operand1 % operand2);
            }
            break;
        }
        case Opcode::remu: {
            uint64_t operand1 = read_rs1();
            uint64_t operand2 = read_rs2();
            if (operand2 == 0) {
                write_rd(operand1);
            } else {
                write_rd(operand1 % operand2);
            }
            break;
        }
        case Opcode::mulw:
            write_rd(sign_ext(read_rs1() * read_rs2()));
            break;
        case Opcode::divw: {
            int32_t operand1 = read_rs1();
            int32_t operand2 = read_rs2();
            if (operand2 == 0) {
                write_rd(-1);
            } else if (operand1 == std::numeric_limits<int32_t>::min() && operand2 == -1) {
                write_rd(operand1);
            } else {
                write_rd(operand1 / operand2);
            }
            break;
        }
        case Opcode::divuw: {
            uint32_t operand1 = read_rs1();
            uint32_t operand2 = read_rs2();
            if (operand2 == 0) {
                write_rd(-1);
            } else {
                write_rd(sign_ext(operand1 / operand2));
            }
            break;
        }
        case Opcode::remw: {
            int32_t operand1 = read_rs1();
            int32_t operand2 = read_rs2();
            if (operand2 == 0) {
                write_rd(operand1);
            } else if (operand1 == std::numeric_limits<int32_t>::min() && operand2 == -1) {
                write_rd(0);
            } else {
                write_rd(operand1 % operand2);
            }
            break;
        }
        case Opcode::remuw: {
            uint32_t operand1 = read_rs1();
            uint32_t operand2 = read_rs2();
            if (operand2 == 0) {
                write_rd(sign_ext(operand1));
            } else {
                write_rd(sign_ext(operand1 % operand2));
            }
            break;
        }
        
        /* A-extension */
        // Stub implementations. Single thread only.
        case Opcode::lr_w: {
            reg_t addr = read_rs1();
            write_rd(sign_ext(emu::load_memory<uint32_t>(addr)));
            context->lr = addr;
            break;
        }
        case Opcode::lr_d: {
            reg_t addr = read_rs1();
            write_rd(emu::load_memory<uint64_t>(addr));
            context->lr = addr;
            break;
        }
        case Opcode::sc_w: {
            reg_t addr = read_rs1();
            if (addr != context->lr) {
                write_rd(1);
                return;
            }
            emu::store_memory<uint32_t>(addr, read_rs2());
            write_rd(0);
            break;
        }
        case Opcode::sc_d: {
            reg_t addr = read_rs1();
            if (addr != context->lr) {
                write_rd(1);
                return;
            }
            emu::store_memory<uint64_t>(addr, read_rs2());
            write_rd(0);
            break;
        }
        case Opcode::amoswap_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            if (inst.rd() != 0) {
                write_rd(sign_ext(emu::load_memory<uint32_t>(addr)));
            }
            emu::store_memory<uint32_t>(addr, src);
            break;
        }
        case Opcode::amoswap_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            if (inst.rd() != 0) {
                write_rd(emu::load_memory<uint64_t>(addr));
            }
            emu::store_memory<uint64_t>(addr, src);
            break;
        }
        case Opcode::amoadd_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = emu::load_memory<uint32_t>(addr);
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(addr, src + mem);
            break;
        }
        case Opcode::amoadd_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = emu::load_memory<uint64_t>(addr);
            write_rd(mem);
            emu::store_memory<uint64_t>(addr, src + mem);
            break;
        }
        case Opcode::amoand_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = emu::load_memory<uint32_t>(addr);
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(addr, src & mem);
            break;
        }
        case Opcode::amoand_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = emu::load_memory<uint64_t>(addr);
            write_rd(mem);
            emu::store_memory<uint64_t>(addr, src & mem);
            break;
        }
        case Opcode::amoor_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = emu::load_memory<uint32_t>(addr);
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(addr, src | mem);
            break;
        }
        case Opcode::amoor_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = emu::load_memory<uint64_t>(addr);
            write_rd(mem);
            emu::store_memory<uint64_t>(addr, src | mem);
            break;
        }
        case Opcode::amoxor_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = emu::load_memory<uint32_t>(addr);
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(addr, src ^ mem);
            break;
        }
        case Opcode::amoxor_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = emu::load_memory<uint64_t>(addr);
            write_rd(mem);
            emu::store_memory<uint64_t>(addr, src ^ mem);
            break;
        }
        case Opcode::amomin_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = emu::load_memory<uint32_t>(addr);
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(addr, std::min<int32_t>(src, mem));
            break;
        }
        case Opcode::amomin_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = emu::load_memory<uint64_t>(addr);
            write_rd(mem);
            emu::store_memory<uint64_t>(addr, std::min<int64_t>(src, mem));
            break;
        }
        case Opcode::amomax_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = emu::load_memory<uint32_t>(addr);
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(addr, std::max<int32_t>(src, mem));
            break;
        }
        case Opcode::amomax_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = emu::load_memory<uint64_t>(addr);
            write_rd(mem);
            emu::store_memory<uint64_t>(addr, std::max<int64_t>(src, mem));
            break;
        }
        case Opcode::amominu_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = emu::load_memory<uint32_t>(addr);
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(addr, std::min<uint32_t>(src, mem));
            break;
        }
        case Opcode::amominu_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = emu::load_memory<uint64_t>(addr);
            write_rd(mem);
            emu::store_memory<uint64_t>(addr, std::min<uint64_t>(src, mem));
            break;
        }
        case Opcode::amomaxu_w: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint32_t mem = emu::load_memory<uint32_t>(addr);
            write_rd(sign_ext(mem));
            emu::store_memory<uint32_t>(addr, std::max<uint32_t>(src, mem));
            break;
        }
        case Opcode::amomaxu_d: {
            reg_t addr = read_rs1();
            reg_t src = read_rs2();
            uint64_t mem = emu::load_memory<uint64_t>(addr);
            write_rd(mem);
            emu::store_memory<uint64_t>(addr, std::max<uint64_t>(src, mem));
            break;
        }

        /* F-extension */
        case Opcode::flw: {
            uint32_t value = emu::load_memory<uint32_t>(read_rs1() + inst.imm());
            write_frd_s(util::read_as<softfp::Single>(&value));
            break;
        }
        case Opcode::fsw: {
            softfp::Single value = read_frs2_s();
            emu::store_memory<uint32_t>(read_rs1() + inst.imm(), util::read_as<uint32_t>(&value));
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
            uint64_t value = emu::load_memory<uint64_t>(read_rs1() + inst.imm());
            write_frd_d(util::read_as<softfp::Double>(&value));
            break;
        }
        case Opcode::fsd: {
            softfp::Double value = read_frs2_d();
            emu::store_memory<uint64_t>(read_rs1() + inst.imm(), util::read_as<uint64_t>(&value));
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

        case Opcode::illegal:
            std::cerr << "Illegal opcode " << std::endl;
            throw "Illegal";
    }
}

}
