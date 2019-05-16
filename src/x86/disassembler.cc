#include <array>
#include <utility>

#include "util/assert.h"
#include "util/format.h"
#include "x86/disassembler.h"
#include "x86/instruction.h"
#include "x86/opcode.h"

namespace {

// std::ostream does not support print hexical values as signed. The following code is a helper that works around the
// issue.
template<typename T>
struct Signed_hex { T value; };

template<typename T>
Signed_hex<T> as_signed(T value) {
    return {value};
}

template<typename T>
std::ostream& operator <<(std::ostream& stream, Signed_hex<T> hex) {
    std::make_unsigned_t<T> value = hex.value;
    if (hex.value < 0) {
        value = -value;
        stream << '-';
    } else {
        if (stream.flags() & std::ios::showpos) {
            stream << '+';
        }
    }
    stream << value;
    return stream;
}

}

namespace x86::disassembler {

const char *register_name(Register reg) {
    static std::array<const char*, 0x48> reg_names = {
        "al", "cl", "dl", "bl", "ah", "ch", "dh", "bh",
        "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
        "ax", "cx", "dx", "bx", "sp", "bp", "si", "di",
        "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w",
        "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi",
        "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
        "(unknown)", "(unknown)", "(unknown)", "(unknown)", "spl", "bpl", "sil", "dil"
    };

    uint8_t reg_num = static_cast<uint8_t>(reg);
    if (reg_num < 0x10 || reg_num >= 0x58) return "(unknown)";
    return reg_names[reg_num - 0x10];
}

const char *opcode_name(Opcode opcode) {
    switch (opcode) {
#define CASE(x) case Opcode::x: return #x;
        CASE(add)
        case Opcode::i_and: return "and";
        CASE(cdqe)
        CASE(call)
        CASE(cmovcc)
        CASE(cmp)
        CASE(cdq) CASE(cqo)
        CASE(div)
        CASE(idiv)
        CASE(imul)
        CASE(jmp)
        CASE(lea)
        CASE(mov)
        CASE(movabs)
        CASE(movsx)
        CASE(movzx)
        CASE(mul)
        CASE(neg)
        CASE(nop)
        case Opcode::i_not: return "not";
        case Opcode::i_or: return "or";
        CASE(push)
        CASE(pop)
        CASE(ret)
        CASE(sar)
        CASE(sbb)
        CASE(setcc)
        CASE(shl)
        CASE(shr)
        CASE(sub)
        CASE(test)
        CASE(xchg)
        case Opcode::i_xor: return "xor";
#undef CASE
        default: return "(unknown)";
    }
}

const char *condition_code_name(Condition_code cc) {
    switch (cc) {
        case Condition_code::overflow: return "o";
        case Condition_code::not_overflow: return "no";
        case Condition_code::below: return "b";
        case Condition_code::above_equal: return "ae";
        case Condition_code::zero: return "z";
        case Condition_code::not_zero: return "nz";
        case Condition_code::below_equal: return "be";
        case Condition_code::above: return "a";
        case Condition_code::sign: return "s";
        case Condition_code::not_sign: return "ns";
        case Condition_code::parity: return "p";
        case Condition_code::not_parity: return "np";
        case Condition_code::less: return "l";
        case Condition_code::greater_equal: return "ge";
        case Condition_code::less_equal: return "le";
        case Condition_code::greater: return "g";
        default: return "(unknown)";
    }
}

void print_operand(const Operand& operand) {
    if (operand.is_register()) {
        std::clog << register_name(operand.as_register());
    } else if (operand.is_memory()) {
        const Memory& it = operand.as_memory();

        const char *qualifier = it.size == 1 ? "byte" :
                                it.size == 2 ? "word" :
                                it.size == 4 ? "dword" :
                                it.size == 8 ? "qword" : "(unknown)";

        std::clog << qualifier << " [";
        bool first = true;

        if (it.base != Register::none) {
            std::clog << register_name(it.base);
            first = false;
        }

        if (it.index != Register::none) {
            if (first) {
                first = false;
            } else {
                std::clog << '+';
            }
            std::clog << register_name(it.index);
            if (it.scale != 1) {
                std::clog << '*' << static_cast<int>(it.scale);
            }
        }

        if (first) {
            // Write out the full address in this case.
            util::log("{:#x}", static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(it.displacement))));
        } else if (it.displacement) {
            util::log("{:+#x}", as_signed<int32_t>(it.displacement));
        }

        std::clog << ']';

    } else if (operand.is_immediate()) {
        util::log("{:#x}", as_signed<int64_t>(operand.as_immediate()));
    } else {
        ASSERT(0);
    }
}

void print_instruction(uint64_t pc, const char *code, size_t length, const Instruction& inst) {
    if ((pc & 0xFFFFFFFF) == pc) {
        util::log("{:8x}:       ", pc);
    } else {
        util::log("{:16x}:       ", pc);
    }

    for (size_t i = 0; i < 8; i++) {
        if (i < length) {
            util::log("{:02x}", code[i]);
        } else {
            std::clog << "  ";
        }
    }

    if (inst.opcode == Opcode::cmovcc) {
        util::log("        cmov{:-4}", condition_code_name(inst.cond));
    } else if (inst.opcode == Opcode::jcc) {
        util::log("        j{:-7}", condition_code_name(inst.cond));
    } else if (inst.opcode == Opcode::setcc) {
        util::log("        set{:-5}", condition_code_name(inst.cond));
    } else {
        util::log("        {:-8}", opcode_name(inst.opcode));
    }

    for (int i = 0; i < 2; i++) {
        if (inst.operands[i].is_empty()) break;
        if (i != 0) std::clog << ", ";
        print_operand(inst.operands[i]);
    }

    std::clog << std::endl;

    if (length > 8) {
        pc += 8;
        if ((pc & 0xFFFFFFFF) == pc) {
            util::log("{:8x}:       ", pc);
        } else {
            util::log("{:16x}:       ", pc);
        }

        for (size_t i = 8; i < length; i++) {
            util::log("{:02x}", code[i]);
        }

        std::clog << std::endl;
    }
}

}
