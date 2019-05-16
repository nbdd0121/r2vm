#ifndef X86_BUILDER_H
#define X86_BUILDER_H

#include "x86/instruction.h"
#include "x86/opcode.h"

namespace x86::builder {

/* Builder for memory operands */

namespace internal {

struct Memory_operand_builder {
    Memory value;
};

}

// index * scale
[[maybe_unused]]
static internal::Memory_operand_builder operator *(Register index, int scale) {
    internal::Memory_operand_builder ret;
    ret.value.base = Register::none;
    ret.value.index = index;
    ret.value.scale = scale;
    ret.value.displacement = 0;
    return ret;
}

// base + index * scale
[[maybe_unused]]
static internal::Memory_operand_builder&& operator +(Register base, internal::Memory_operand_builder&& builder) {
    builder.value.base = base;
    return std::move(builder);
}

// base + index
[[maybe_unused]]
static internal::Memory_operand_builder operator *(Register base, Register index) {
    internal::Memory_operand_builder ret;
    ret.value.base = base;
    ret.value.index = index;
    ret.value.scale = 1;
    ret.value.displacement = 0;
    return ret;
}

// base + displacement
[[maybe_unused]]
static internal::Memory_operand_builder operator +(Register base, uint32_t displacement) {
    internal::Memory_operand_builder ret;
    ret.value.base = base;
    ret.value.index = Register::none;
    ret.value.scale = 0;
    ret.value.displacement = displacement;
    return ret;
}

// base - displacement
[[maybe_unused]]
static internal::Memory_operand_builder operator -(Register base, uint32_t displacement) {
    return base + (-displacement);
}

// [base +] index * scale + displacement
[[maybe_unused]]
static internal::Memory_operand_builder&& operator +(internal::Memory_operand_builder&& builder, uint32_t displacement) {
    builder.value.displacement = displacement;
    return std::move(builder);
}

[[maybe_unused]]
static internal::Memory_operand_builder&& operator -(internal::Memory_operand_builder&& builder, uint32_t displacement) {
    return std::move(builder) + (-displacement);
}

[[maybe_unused]]
static Memory&& qword(internal::Memory_operand_builder&& operand) {
    operand.value.size = 8;
    return std::move(operand.value);
}

[[maybe_unused]]
static Memory&& dword(internal::Memory_operand_builder&& operand) {
    operand.value.size = 4;
    return std::move(operand.value);
}

[[maybe_unused]]
static Memory&& word(internal::Memory_operand_builder&& operand) {
    operand.value.size = 2;
    return std::move(operand.value);
}

[[maybe_unused]]
static Memory&& byte(internal::Memory_operand_builder&& operand) {
    operand.value.size = 1;
    return std::move(operand.value);
}

/* Instruction builders */

static Instruction nullary(Opcode opcode) {
    Instruction ret;
    ret.opcode = opcode;
    return ret;
}

static Instruction unary(Opcode opcode, const Operand& op1) {
    Instruction ret;
    ret.opcode = opcode;
    ret.operands[0] = op1;
    return ret;
}

static Instruction binary(Opcode opcode, const Operand& op1, const Operand& op2) {
    Instruction ret;
    ret.opcode = opcode;
    ret.operands[0] = op1;
    ret.operands[1] = op2;
    return ret;
}

#define NULLARY(name) [[maybe_unused]] static Instruction name() { return nullary(Opcode::name); }

#define UNARY(name) [[maybe_unused]] \
static Instruction name(const Operand& op1) {\
    return unary(Opcode::name, op1);\
}

#define BINARY(name) [[maybe_unused]] \
static Instruction name(const Operand& op1, const Operand& op2) {\
    return binary(Opcode::name, op1, op2);\
}

BINARY(add)
BINARY(i_and)

[[maybe_unused]]
static Instruction cmovcc(Condition_code cc, const Operand& op1, const Operand& op2) {
    Instruction ret;
    ret.opcode = Opcode::cmovcc;
    ret.cond = cc;
    ret.operands[0] = op1;
    ret.operands[1] = op2;
    return ret;
}

UNARY(call)
NULLARY(cdqe)
BINARY(cmp)
NULLARY(cdq) NULLARY(cqo)
UNARY(div)
UNARY(idiv)
UNARY(imul) BINARY(imul)

[[maybe_unused]]
static Instruction jcc(Condition_code cc, const Operand& op) {
    Instruction ret;
    ret.opcode = Opcode::jcc;
    ret.cond = cc;
    ret.operands[0] = op;
    return ret;
}

UNARY(jmp)
BINARY(lea)
BINARY(mov)
BINARY(movabs)
BINARY(movsx)
BINARY(movzx)
UNARY(mul)
UNARY(neg)
NULLARY(nop)
UNARY(i_not)
BINARY(i_or)
UNARY(pop)
UNARY(push)
NULLARY(ret) UNARY(ret)
BINARY(sar)
BINARY(sbb)

[[maybe_unused]]
static Instruction setcc(Condition_code cc, const Operand& op) {
    Instruction ret;
    ret.opcode = Opcode::setcc;
    ret.cond = cc;
    ret.operands[0] = op;
    return ret;
}

BINARY(shl)
BINARY(shr)
BINARY(sub)
BINARY(test)
BINARY(xchg)
BINARY(i_xor)

#undef BINARY
#undef UNARY
#undef NULLARY

}

#endif
