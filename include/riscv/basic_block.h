#ifndef RISCV_BASIC_BLOCK_H
#define RISCV_BASIC_BLOCK_H

#include <vector>

#include "riscv/typedef.h"

namespace riscv {

class Instruction;

struct Basic_block {
    
    // PC before first instruction.
    reg_t start_pc;

    // PC *past* last instruction.
    reg_t end_pc;

    // List of instructions in the basic block.
    std::vector<Instruction> instructions;
};

}

#endif
