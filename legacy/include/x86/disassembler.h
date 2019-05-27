#ifndef X86_DISASSEMBLER_H
#define X86_DISASSEMBLER_H

#include <cstdint>

#include "instruction.h"

namespace x86 {

namespace disassembler {

const char *register_name(Register reg);
const char *opcode_name(Opcode opcode);
void print_operand(const Operand& operand);
void print_instruction(uint64_t pc, const char *code, size_t length, const Instruction& inst);

}

}

#endif
