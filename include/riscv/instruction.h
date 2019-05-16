#ifndef RISCV_INSTRUCTION_H
#define RISCV_INSTRUCTION_H

#include <cstdint>
#include <type_traits>

#include "util/assert.h"
#include "util/bitfield.h"
#include "riscv/typedef.h"

namespace riscv {

enum class Opcode;

// The size of Instruction class has to be smaller than 64-bit, so it can fit naturally in a machine word, or 2 in
// 32-bit architecture. The memory layout of this class is subject to change, so this class should be accessible via
// member functions only.
class Instruction {

    /* The following fields are subdivision of rs2 field. */
    using Rs2_field = util::Bitfield<uint8_t, 4, 0>;

    // 0 indicates 32-bit, 1 indicates 16-bit.
    using Length_field = util::Bitfield<uint8_t, 7, 7>;

    /* The following fields are subdivision of immediate field. */
    using Rs3_field = util::Bitfield<uint32_t, 7, 3>;
    using Rm_field = util::Bitfield<uint32_t, 2, 0>;

    // Opcode. Currently we limit it to 256 instructions. If it goes
    // over it, then this class has to go through a major redesign.
    uint8_t opcode_;

    // Destination register.
    // 3 unused bits.
    uint8_t rd_;

    // Source register 1 for most instructions.
    // For CSR*I, immediate will be stored here.
    // 3 unused bits.
    uint8_t rs1_;

    // Source register 2 for most instructions.
    // The highest bit will indicate whether the instruction is 32-bit or 16-bit.
    // 2 unused bits.
    uint8_t rs2_;

    // Decoded immediates for most instructions.
    // For FENCE instruction, pred and succ will be stored here.
    // For CSR*I, csr will be stored here.
    // For atomic instructions, aq and rl will be stored here.
    // For floating point arithemtic instructions, rs3 (if any) and rm will be stored here.
    uint32_t immediate_;

public:
    Instruction() noexcept: opcode_{0}, rd_{0}, rs1_{0}, rs2_{0}, immediate_{0} {}

    /* Accessors */
    int rd() const noexcept { return rd_; }
    int rs1() const noexcept { return rs1_; }
    int rs2() const noexcept { return Rs2_field::extract(rs2_); }
    reg_t imm() const noexcept { return static_cast<sreg_t>(static_cast<int32_t>(immediate_)); }
    int length() const noexcept { return Length_field::extract(rs2_) ? 4 : 2; }
    Opcode opcode() const noexcept { return static_cast<Opcode>(opcode_); }
    int rs3() const noexcept { return Rs3_field::extract(immediate_); }
    int rm() const noexcept { return Rm_field::extract(immediate_); }

    /* Mutators */
    void rd(int rd) noexcept { rd_ = rd; }
    void rs1(int rs1) noexcept { rs1_ = rs1; }
    void rs2(int rs2) noexcept { rs2_ = Rs2_field::pack(rs2_, rs2); }
    void imm(reg_t imm) noexcept { immediate_ = imm; }
    void rs3(int rs3) noexcept { immediate_ = Rs3_field::pack(immediate_, rs3); }
    void rm(int rm) noexcept { immediate_ = Rm_field::pack(immediate_, rm); }

    void length(int len) {
        ASSERT(len == 2 || len == 4);
        rs2_ = Length_field::pack(rs2_, len == 4);
    }

    void opcode(Opcode opcode) noexcept { opcode_ = static_cast<uint16_t>(opcode); }
};

static_assert(std::is_standard_layout<Instruction>::value, "class Instruction must be of standard layout.");
static_assert(sizeof(Instruction) <= sizeof(uint64_t), "class Instruction must fit within 64-bits.");

} // riscv

#endif
