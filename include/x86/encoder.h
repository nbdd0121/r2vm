#ifndef X86_ENCODER_H
#define X86_ENCODER_H

#include "util/code_buffer.h"
#include "x86/instruction.h"

namespace x86 {

enum class Register: uint8_t;

class Encoder {
private:
    util::Code_buffer& buffer_;

    // Internal helper functions
    void emit_byte(uint8_t byte);
    void emit_word(uint16_t word);
    void emit_dword(int32_t dword);
    void emit_qword(int64_t qword);
    void emit_immediate(int size, uint64_t imm);

    void emit_rex(const Operand& operand, Register reg, uint8_t rex);
    void emit_modrm(const Operand& operand, Register reg);

    void emit_r_rm(int op_size, const Operand& mem, Register reg, uint64_t opcode);
    void emit_plusr(int op_size, Register reg, uint64_t opcode);

    void emit_r_rm(const Operand& mem, const Operand& reg, uint64_t opcode);
    void emit_rm(const Instruction& inst, uint64_t opcode, int id);
    void emit_alu(const Instruction& inst, int id);
    void emit_shift(const Instruction& inst, int id);

    void emit_call(const Instruction& inst);
    void emit_imul(const Instruction& inst);
    void emit_jcc(const Instruction& inst);
    void emit_jmp(const Instruction& inst);
    void emit_lea(const Instruction& inst);
    void emit_mov(const Instruction& inst);
    void emit_movabs(const Instruction& inst);
    void emit_movsx(const Instruction& inst);
    void emit_movzx(const Instruction& inst);
    void emit_pop(const Instruction& inst);
    void emit_push(const Instruction& inst);
    void emit_ret(const Instruction& inst);
    void emit_setcc(const Instruction& inst);
    void emit_test(const Instruction& inst);
    void emit_xchg(const Instruction& inst);

public:
    Encoder(util::Code_buffer& buffer): buffer_ {buffer} {};

    util::Code_buffer& buffer() { return buffer_; }

    void encode(const Instruction& inst);
};

}

#endif
