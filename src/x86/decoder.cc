#include "util/memory.h"
#include "x86/decoder.h"
#include "x86/instruction.h"
#include "x86/opcode.h"

namespace x86 {

uint8_t Decoder::read_byte() {
    return util::read_as<uint8_t>(reinterpret_cast<void*>(_pc++));
}

uint32_t Decoder::read_dword() {
    uint32_t ret = util::read_as<uint32_t>(reinterpret_cast<void*>(_pc));
    _pc += 4;
    return ret;
}

Register Decoder::register_of_size(int reg, int size) {
    switch (size) {
        case 1: return static_cast<Register>(reg | (reg >= 4 && reg <= 7 ? reg_gpb2 : reg_gpb));
        case 2: return static_cast<Register>(reg | reg_gpw);
        case 4: return static_cast<Register>(reg | reg_gpd);
        case 8: return static_cast<Register>(reg | reg_gpq);
        default: ASSERT(0);
    }
}

void Decoder::decode_modrm(Operand& operand, Register& reg, int size) {

    uint8_t first_byte = read_byte();
    int mod = first_byte >> 6;
    int rm = first_byte & 0b111;

    // Decode register and prefix with REX.R.
    int reg_id = (first_byte >> 3) & 0b111;
    if (_rex & 0x4) reg_id += 8;

    // For register with size 1 and no REX set, dil and sil etc are not accessible.
    if (size == 1 && !(_rex & 0x40)) {
        reg = static_cast<Register>(reg_id | reg_gpb);
    } else {
        reg = register_of_size(reg_id, size);
    }

    // Operand is a register.
    if (mod == 0b11) {
        int op_id = rm | (_rex & 0x1 ? 8 : 0);
        if (size == 1 && !(_rex & 0x40)) {
            operand = static_cast<Register>(op_id | reg_gpb);
        } else {
            operand = register_of_size(op_id, size);
        }
        return;
    }

    if (mod == 0b00 && rm == 0b100) {
        // rip-relative addressing not supported
        ASSERT(0);
    }

    Memory mem;
    mem.size = size;

    // No SIB bytes.
    if (rm != 0b100) {
        mem.base = static_cast<Register>(rm | (_rex & 0x1 ? 8 : 0) | reg_gpq);
        mem.index = Register::none;
        mem.scale = 0;

    } else {

        uint8_t sib = read_byte();
        int ss = sib >> 6;
        int index = (_rex & 0x2 ? 8 : 0) | ((sib >> 3) & 0b111);
        int base = (_rex & 0x1 ? 8 : 0) | (sib & 0b111);

        // RSP cannot be index. index = RSP means there is no index.
        if (index == 0b100) {
            mem.index = Register::none;
            mem.scale = 0;

        } else {
            mem.index = static_cast<Register>(index | reg_gpq);
            mem.scale = (1 << ss);
        }

        // If mod = 0, and base = RBP or R13, then we have no base register.
        if (mod == 0 && (base & 7) == 0b101) {
            mem.base = Register::none;

            // a tiny trick, so we have 32-bit displacement
            mod = 0b10;
        } else {
            mem.base = static_cast<Register>(base | reg_gpq);
        }
    }

    if (mod == 0b00) {
        mem.displacement = 0;
    } else if (mod == 0b01) {
        mem.displacement = static_cast<int32_t>(static_cast<int8_t>(read_byte()));
    } else {
        mem.displacement = read_dword();
    }
    operand = mem;
}

Instruction Decoder::decode_instruction() {
    _rex = 0;
    _opsize = 4;

    // Keep reading prefixes.
    while (true) {
        uint8_t prefix = read_byte();
        if ((prefix & 0xF0) == 0x40) {
            // REX prefix
            _rex = prefix;
            // REX.W
            if (_rex & 0x08) _opsize = 8;
        } else if (prefix == 0x66) {
            _opsize = 2;
        } else {
            // Unread the byte
            _pc--;
            break;
        }
    }

    uint8_t opcode = read_byte();
    Instruction ret;

    // 0xF6 = div/idiv, and that's the only thing we care at the moment!
    if (opcode != 0xF6 && opcode != 0xF7) {
        ret.opcode = Opcode::illegal;
        return ret;
    }

    if (opcode == 0xF6) _opsize = 1;

    Register reg;
    decode_modrm(ret.operands[0], reg, _opsize);
    int id = static_cast<int>(reg) & 7;
    if (id == 6) {
        ret.opcode = Opcode::div;
    } else if (id == 7) {
        ret.opcode = Opcode::idiv;
    } else {
        ret.opcode = Opcode::illegal;
        return ret;
    }

    return ret;
}

}
