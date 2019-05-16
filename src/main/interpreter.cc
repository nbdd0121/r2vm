#include "main/interpreter.h"
#include "riscv/context.h"
#include "riscv/decoder.h"
#include "riscv/instruction.h"
#include "riscv/opcode.h"
#include "util/assert.h"

Interpreter::Interpreter() noexcept {}

Interpreter::~Interpreter() {}

void Interpreter::step(riscv::Context& context) {
    emu::reg_t pc = context.pc;
    riscv::Basic_block& basic_block = inst_cache_[pc];

    if (UNLIKELY(basic_block.instructions.size() == 0)) {
        riscv::Decoder decoder {pc};
        basic_block = decoder.decode_basic_block();

        // Function step will assume the pc is pre-incremented, but this is clearly not the case for auipc. Therfore we
        // preprocess all auipc instructions to compensate this.
        for (auto& inst: basic_block.instructions) {
            if (inst.opcode() == riscv::Opcode::auipc) {
                inst.imm(inst.imm() + (pc - basic_block.start_pc) + inst.length());
            }
            pc += inst.length();
        }
    }

    size_t block_size = basic_block.instructions.size() - 1;

    for (size_t i = 0; i < block_size; i++) {
        // Retrieve cached data
        riscv::Instruction inst = basic_block.instructions[i];
        try {
            riscv::step(&context, inst);
        } catch(...) {
            // In case an exception happens, we need to move the pc before the instruction.
            for (size_t j = 0; j < i; j++) {
                context.pc += basic_block.instructions[j].length();
            }
            context.instret += i;
            throw;
        }
    }

    context.pc = basic_block.end_pc;
    context.instret += block_size + 1;
    riscv::Instruction inst = basic_block.instructions[block_size];
    try {
        riscv::step(&context, inst);
    } catch(...) {
        context.pc -= inst.length();
        context.instret--;
        throw;
    }
}

void Interpreter::flush_cache() {
    inst_cache_.clear();
}
