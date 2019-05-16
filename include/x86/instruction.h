#ifndef X86_INSTRUCTION_H
#define X86_INSTRUCTION_H

#include <cstdint>

#include "util/assert.h"

namespace x86 {

enum class Opcode: uint16_t;
enum class Condition_code: uint8_t;

// The register name is represented using an integer. The lower 4-bit represents the index, and the highest bits
// represents types of the register.
constexpr uint8_t reg_gpb = 0x10;
constexpr uint8_t reg_gpw = 0x20;
constexpr uint8_t reg_gpd = 0x30;
constexpr uint8_t reg_gpq = 0x40;
// This is for special spl, bpl, sil and dil
constexpr uint8_t reg_gpb2 = 0x50;

// These registers are given their 64-bit variant's name.
enum class Register: uint8_t {
    // General purpose registers
    al   = 0  | reg_gpb, ax   = 0  | reg_gpw, eax  = 0  | reg_gpd, rax = 0  | reg_gpq,
    cl   = 1  | reg_gpb, cx   = 1  | reg_gpw, ecx  = 1  | reg_gpd, rcx = 1  | reg_gpq,
    dl   = 2  | reg_gpb, dx   = 2  | reg_gpw, edx  = 2  | reg_gpd, rdx = 2  | reg_gpq,
    bl   = 3  | reg_gpb, bx   = 3  | reg_gpw, ebx  = 3  | reg_gpd, rbx = 3  | reg_gpq,
    ah   = 4  | reg_gpb, sp   = 4  | reg_gpw, esp  = 4  | reg_gpd, rsp = 4  | reg_gpq,
    ch   = 5  | reg_gpb, bp   = 5  | reg_gpw, ebp  = 5  | reg_gpd, rbp = 5  | reg_gpq,
    dh   = 6  | reg_gpb, si   = 6  | reg_gpw, esi  = 6  | reg_gpd, rsi = 6  | reg_gpq,
    bh   = 7  | reg_gpb, di   = 7  | reg_gpw, edi  = 7  | reg_gpd, rdi = 7  | reg_gpq,
    r8b  = 8  | reg_gpb, r8w  = 8  | reg_gpw, r8d  = 8  | reg_gpd, r8  = 8  | reg_gpq,
    r9b  = 9  | reg_gpb, r9w  = 9  | reg_gpw, r9d  = 9  | reg_gpd, r9  = 9  | reg_gpq,
    r10b = 10 | reg_gpb, r10w = 10 | reg_gpw, r10d = 10 | reg_gpd, r10 = 10 | reg_gpq,
    r11b = 11 | reg_gpb, r11w = 11 | reg_gpw, r11d = 11 | reg_gpd, r11 = 11 | reg_gpq,
    r12b = 12 | reg_gpb, r12w = 12 | reg_gpw, r12d = 12 | reg_gpd, r12 = 12 | reg_gpq,
    r13b = 13 | reg_gpb, r13w = 13 | reg_gpw, r13d = 13 | reg_gpd, r13 = 13 | reg_gpq,
    r14b = 14 | reg_gpb, r14w = 14 | reg_gpw, r14d = 14 | reg_gpd, r14 = 14 | reg_gpq,
    r15b = 15 | reg_gpb, r15w = 15 | reg_gpw, r15d = 15 | reg_gpd, r15 = 15 | reg_gpq,
    // Special register that requires REX prefix to access.
    spl = 4 | reg_gpb2, bpl = 5 | reg_gpb2, sil = 6 | reg_gpb2, dil = 7 | reg_gpb2,

    // Ideally we represent this using optional<Register>, but currently std::optional is too expensive.
    none = 0,
};

struct Memory {
    uint32_t displacement;
    Register base;
    Register index;
    uint8_t scale;
    uint8_t size;
};

// Previous std::variant is used here. However it turns out to have huge performance penalty, so a traditional union is
// used instead. The performance penalty comes from the fact that variant is not trivally copyable even though all its
// type parameters are.
class Operand {
    union {
        Register reg_;
        Memory mem_;
        uint64_t imm_;
    };
    uint8_t tag_;

public:
    Operand() noexcept: tag_{0} {};
    Operand(Register reg) noexcept: reg_{reg}, tag_{1} {}
    Operand(const Memory& mem) noexcept: mem_{mem}, tag_{2} {}
    Operand(uint64_t imm) noexcept: imm_{imm}, tag_{3} {}

    bool is_empty() const noexcept { return tag_ == 0; }
    bool is_register() const noexcept { return tag_ == 1; }
    bool is_memory() const noexcept { return tag_ == 2; }
    bool is_immediate() const noexcept { return tag_ == 3; }

    Register as_register() const {
        ASSERT(tag_ == 1);
        return reg_;
    }

    const Memory& as_memory() const {
        ASSERT(tag_ == 2);
        return mem_;
    }

    uint64_t as_immediate() const {
        ASSERT(tag_ == 3);
        return imm_;
    }
};

struct Instruction {
    Operand operands[2];
    Opcode opcode;
    // Only used for *cc opcodes.
    Condition_code cond;
};

}

#endif
