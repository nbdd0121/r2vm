#ifndef RISCV_DECODER_H
#define RISCV_DECODER_H

#include <cstdint>

#include "riscv/typedef.h"

namespace riscv {

class Instruction;
struct Basic_block;

class Decoder {
    reg_t pc_;
    reg_t pc_next_;

public:
    static Instruction decode(uint32_t bits);
    static bool can_change_control_flow(Instruction inst);

public:
    Decoder(): pc_{0}, pc_next_{0} {}
    Decoder(reg_t pc): pc_{pc}, pc_next_{0} {}
    Decoder(reg_t pc, reg_t pc_next): pc_{pc}, pc_next_{pc_next} {}

    reg_t pc() const { return pc_; }
    void pc(reg_t pc) { pc_ = pc; }

    Instruction decode_instruction();
    Basic_block decode_basic_block();
};

} // riscv

#endif
