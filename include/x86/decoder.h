#ifndef X86_DECODER_H
#define X86_DECODER_H

#include "x86/instruction.h"

namespace x86 {

class Decoder {
    uint64_t _pc;
    int _rex;
    int _opsize;

public:
    Decoder(): _pc{0} {}
    Decoder(uint64_t pc): _pc{pc} {}

    uint64_t pc() const { return _pc; }
    void pc(uint64_t pc) { _pc = pc; }

private:
    uint8_t read_byte();
    uint32_t read_dword();

    Register register_of_size(int reg, int size);

    void decode_modrm(Operand& operand, Register& reg, int size);

public:
    Instruction decode_instruction();
};

} // x86

#endif
