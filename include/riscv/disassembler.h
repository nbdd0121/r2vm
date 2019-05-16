#ifndef RISCV_DISASSEMBLER_H
#define RISCV_DISASSEMBLER_H

#include <cstdint>

#include "riscv/typedef.h"

namespace riscv {

enum class Opcode;
enum class Csr;
class Instruction;

class Disassembler {
public:
    static const char* register_name(int reg);
    static const char* csr_name(Csr csr);
    static const char* opcode_name(Opcode opcode);
    static void print_instruction(reg_t pc, uint32_t bits, Instruction inst);
};

} // riscv

#endif
