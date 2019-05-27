#include "util/assert.h"
#include "util/int_size.h"
#include "util/memory.h"
#include "x86/encoder.h"
#include "x86/opcode.h"

using namespace x86;

namespace {

int get_size(const Operand& operand) {
    if (operand.is_register()) {
        switch (static_cast<uint8_t>(operand.as_register()) & 0xF0) {
            case reg_gpb: case reg_gpb2: return 1;
            case reg_gpw: return 2;
            case reg_gpd: return 4;
            case reg_gpq: return 8;
            default: ASSERT(0);
        }
    } else if (operand.is_memory()) {
        return operand.as_memory().size;
    } else {
        ASSERT(0);
    }
}

// For most instructions taking immediates:
// If operation size is 64-bit, then imm must be int32.
// If operation size is 8, 16 or 32-bit, then imm must be int8/int16/int32 or uint8/uint16/uint32.
void check_immediate_size(int size, uint64_t imm) {
    if (size == 1) {
        ASSERT(util::is_int8(imm) || util::is_uint8(imm));
    } else if (size == 2) {
        ASSERT(util::is_int16(imm) || util::is_uint16(imm));
    } else if (size == 4) {
        ASSERT(util::is_int32(imm) || util::is_uint32(imm));
    } else if (size == 8) {
        ASSERT(util::is_int32(imm));
    } else {
        ASSERT(0);
    }
}

}

namespace x86 {

inline void Encoder::emit_byte(uint8_t byte) {
    buffer_.push_back(static_cast<std::byte>(byte));
}

inline void Encoder::emit_word(uint16_t word) {
    size_t size = buffer_.size();
    buffer_.resize(size + 2);
    util::write_as<uint16_t>(buffer_.data() + size, word);
}

inline void Encoder::emit_dword(int32_t dword) {
    size_t size = buffer_.size();
    buffer_.resize(size + 4);
    util::write_as<uint32_t>(buffer_.data() + size, dword);
}

inline void Encoder::emit_qword(int64_t qword) {
    size_t size = buffer_.size();
    buffer_.resize(size + 8);
    util::write_as<uint64_t>(buffer_.data() + size, qword);
}

inline void Encoder::emit_immediate(int size, uint64_t imm) {
    if (size == 1) {
        emit_byte(imm);
    } else if (size == 2) {
        emit_word(imm);
    } else if (size == 4) {
        emit_dword(imm);
    } else if (size == 8) {
        emit_qword(imm);
    } else {
        ASSERT(0);
    }
}

// Emit REX prefix given r/m operand.
// Parameter rex specifies the bits that needs to be true in REX prefix.
// It can be 0x00 (no REX needed), 0x08 (need REX.W), or 0x40 (no bits needed, but need prefix)
void Encoder::emit_rex(const Operand& operand, Register reg, uint8_t rex) {
    uint8_t reg_num = static_cast<uint8_t>(reg);

    // For spl, bpl, sil, di, rex prefix is required.
    if ((reg_num & 0xF0) == reg_gpb2) {
        rex |= 0x40;
    }

    // REX.R
    if (reg_num & 8) {
        rex |= 0x4;
    }

    if (operand.is_register()) {
        uint8_t it_num = static_cast<uint8_t>(operand.as_register());

        // For spl, bpl, sil, di, rex prefix is required.
        if ((it_num & 0xF0) == reg_gpb2) {
            rex |= 0x40;
        }

        // With REX prefix, r/m8 cannot be encoded to access AH, BH, CH, DH
        ASSERT(!rex || !(it_num >= static_cast<uint8_t>(Register::ah) &&
                            it_num <= static_cast<uint8_t>(Register::bh)));

        // REX.B
        if (it_num & 8) {
            rex |= 0x1;
        }
    } else if (operand.is_memory()) {
        const Memory& it = operand.as_memory();

        if (it.base != Register::none) {

            // REX.B
            if (static_cast<uint8_t>(it.base) & 8) {
                rex |= 0x1;
            }
        }

        if (it.index != Register::none) {

            // REX.X
            if (static_cast<uint8_t>(it.index) & 8) {
                rex |= 0x2;
            }
        }

    } else {
        ASSERT(0);
    }

    if (rex) {
        ASSERT(!(reg_num >= static_cast<uint8_t>(Register::ah) && reg_num <= static_cast<uint8_t>(Register::bh)));
        emit_byte(rex | 0x40);
    }
}

// Emit ModR/M and SIB given r/m operand.
// This assumes 64-bit address size.
void Encoder::emit_modrm(const Operand& operand, Register reg) {
    uint8_t reg_num = static_cast<uint8_t>(reg) & 7;

    if (operand.is_register()) {

        // Take only the lowest 3 bit. 4th bit is encoded in REX and higher ones indicate register type.
        emit_byte(0xC0 | (reg_num << 3) | (static_cast<uint8_t>(operand.as_register()) & 7));

    } else if (operand.is_memory()) {
        const Memory& it = operand.as_memory();

        uint8_t base_reg = static_cast<uint8_t>(it.base) & 7;
        uint8_t index_reg = static_cast<uint8_t>(it.index) & 7;
        uint8_t shift = 0;

        // Sanity check that is valid and sp is not used as index register.
        if (it.index != Register::none) {

            ASSERT((static_cast<uint8_t>(it.index) & 0xF0) == reg_gpq);

            // index = RSP is invalid.
            ASSERT((static_cast<uint8_t>(it.index) & 0xF) != 0b101);

            shift = it.scale == 1 ? 0 :
                    it.scale == 2 ? 1 :
                    it.scale == 4 ? 2 :
                    it.scale == 8 ? 3 : 4;

            ASSERT(shift != 4);

        } else {
            ASSERT(it.scale == 0);
        }

        // No base, it can either be [disp32], or [index * scale + disp32]
        if (it.base == Register::none) {

            emit_byte((reg_num << 3) | 0b100);

            if (it.index != Register::none) {
                emit_byte((shift << 6) | (index_reg << 3) | 0b101);
            } else {
                emit_byte(0x25);
            }

            emit_dword(static_cast<uint32_t>(it.displacement));
            return;
        }

        ASSERT((static_cast<uint8_t>(it.base) & 0xF0) == reg_gpq);

        // [base + disp] No SIB byte if base is not RSP
        if (it.index == Register::none) {

            // [RSP/R12 + disp]. Unfortunately in this case we still need SIB byte.
            if (base_reg == 0b100) {

                // [RSP/R12]
                if (it.displacement == 0) {
                    emit_byte((reg_num << 3) | 0b100);
                    emit_byte(0x24);
                    return;
                }

                // [RSP/R12 + disp8]
                if (util::is_int8(it.displacement)) {
                    emit_byte(0x40 | (reg_num << 3) | 0b100);
                    emit_byte(0x24);
                    emit_byte(static_cast<uint8_t>(it.displacement));
                    return;
                }

                // [RSP/R12 + disp32]
                emit_byte(0x80 | (reg_num << 3) | 0b100);
                emit_byte(0x24);
                emit_dword(static_cast<uint32_t>(it.displacement));
                return;
            }

            // [base]. No direct encoding of [RBP/R13] however.
            if (it.displacement == 0 && base_reg != 0b101) {
                emit_byte((reg_num << 3) | base_reg);
                return;
            }

            // [base + disp8]
            if (util::is_int8(it.displacement)) {
                emit_byte(0x40 | (reg_num << 3) | base_reg);
                emit_byte(static_cast<uint8_t>(it.displacement));
                return;
            }

            // [base + disp32]
            emit_byte(0x80 | (reg_num << 3) | base_reg);
            emit_dword(static_cast<uint32_t>(it.displacement));
            return;
        }

        // [base + index * scale]. Similarly, base cannot be RBP/R13
        if (it.displacement == 0 && base_reg != 0b101) {
            emit_byte((reg_num << 3) | 0b100);
            emit_byte((shift << 6) | (index_reg << 3) | base_reg);
            return;
        }

        // [base + index * scale + disp8]
        if (util::is_int8(it.displacement)) {
            emit_byte(0x40 | (reg_num << 3) | 0b100);
            emit_byte((shift << 6) | (index_reg << 3) | base_reg);
            emit_byte(static_cast<uint8_t>(it.displacement));
            return;
        }

        // [base + index * scale + disp32]
        emit_byte(0x80 | (reg_num << 3) | 0b100);
        emit_byte((shift << 6) | (index_reg << 3) | base_reg);
        emit_dword(static_cast<uint32_t>(it.displacement));
        return;

    } else {
        ASSERT(0);
    }
}

// Generic helper function emitting all instructions that operate on r/rm, supporting 8, 16, 32 and 64-bit.
// Argument mem does not have to be a memory operand.
// Encoding of opcode: highest byte is reserved for potential mandatory prefix, and lower 3 bytes, if non-zero,
// will be emitted. The last byte will be emitted even if it's zero. Note that this works since 00 will not appear
// as an escape code.
void Encoder::emit_r_rm(int op_size, const Operand& mem, Register reg, uint64_t opcode) {
    // Operand size override prefix.
    if (op_size == 2) emit_byte(0x66);

    // TODO: If required, emit mandatory prefix here.

    // Emit REX prefix is necessary. Note that the prefix must comes immediately before opcode.
    emit_rex(mem, reg, op_size == 8 ? 0x08 : 0);

    // Emit opcode.
    if (opcode & 0xFF0000) emit_byte(opcode >> 16);
    if (opcode & 0xFF00) emit_byte(opcode >> 8);
    emit_byte(opcode);

    // Mod R/M comes last.
    emit_modrm(mem, reg);
}

// Generic helper function emitting all instructions that uses format +r, supporting 8, 16, 32 and 64-bit.
void Encoder::emit_plusr(int op_size, Register reg, uint64_t opcode) {
    uint8_t reg_num = static_cast<uint8_t>(reg);

    if (op_size == 2) emit_byte(0x66);

    // TODO: If required, emit mandatory prefix here.

    // Emit REX prefix is necessary.
    uint8_t rex = 0;
    if (op_size == 8) rex |= 0x08;
    if (reg_num & 8) rex |= 0x01;
    if ((reg_num & 0xF0) == reg_gpb2) rex |= 0x40;

    if (rex) {
        ASSERT(!(reg_num >= static_cast<uint8_t>(Register::ah) && reg_num <= static_cast<uint8_t>(Register::bh)));
        emit_byte(rex | 0x40);
    }

    // Emit opcode.
    if (opcode & 0xFF0000) emit_byte(opcode >> 16);
    if (opcode & 0xFF00) emit_byte(opcode >> 8);
    emit_byte(opcode | (reg_num & 7));
}

// Generate code for instructions with only R/RM encoding.
// It makes sure that reg is indeed register, and operand size match.
void Encoder::emit_r_rm(const Operand& mem, const Operand& reg, uint64_t opcode) {
    int op_size = get_size(mem);
    ASSERT(op_size != 1);
    ASSERT(reg.is_register());
    ASSERT(get_size(reg) == op_size);
    emit_r_rm(op_size, mem, reg.as_register(), opcode);
}

// Generate code for instructions with only RM encoding.
// It assumes the default opcode is for byte sized operands, and other sizes have opcode + 1.
void Encoder::emit_rm(const Instruction& inst, uint64_t opcode, int id) {
    ASSERT(inst.operands[1].is_empty());
    int op_size = get_size(inst.operands[0]);
    if (op_size != 1) {
        opcode += 1;
    }
    emit_r_rm(op_size, inst.operands[0], static_cast<Register>(id), opcode);
}

// Generate code for ALU instructions.
// ALU instructions include (ordered by their id): add, or, adc, sbb, and, sub, xor, cmp.
void Encoder::emit_alu(const Instruction& inst, int id) {
    const Operand& dst = inst.operands[0];
    const Operand& src = inst.operands[1];

    // Get size of dst. This also guards dst from holding immediate.
    int op_size = get_size(dst);

    // Add immediate.
    if (src.is_immediate()) {

        // Check that the immediate size is encodable.
        uint64_t imm = src.as_immediate();
        check_immediate_size(op_size, imm);

        // Short encoding available for 8-bit immediate.
        if (op_size != 1 && util::is_int8(imm)) {
            emit_r_rm(op_size, dst, static_cast<Register>(id), 0x83);
            emit_byte(imm);
            return;
        }

        // Short encoding is available for RAX
        if (dst.is_register() && (static_cast<uint8_t>(dst.as_register()) & 0xF) == 0) {

            if (op_size == 2) {
                emit_byte(0x66);
            } else if (op_size == 8) {
                emit_byte(0x48);
            }

            emit_byte((id << 3) | (op_size == 1 ? 0x04 : 0x05));
            emit_immediate(op_size == 8 ? 4 : op_size, imm);
            return;
        }

        emit_r_rm(op_size, dst, static_cast<Register>(id), op_size == 1 ? 0x80 : 0x81);
        emit_immediate(op_size == 8 ? 4 : op_size, imm);
        return;
    }

    // Make sure operand size matches. This also guards src from holding monostate.
    ASSERT(get_size(src) == op_size);

    // Prefer INST r/m, r to INST r, r/m in case of INST r, r
    if (src.is_register()) {
        emit_r_rm(op_size, dst, src.as_register(), (id << 3) | (op_size == 1 ? 0x00 : 0x01));
        return;
    }

    // Operands cannot both be memory.
    ASSERT(dst.is_register());
    emit_r_rm(op_size, src, dst.as_register(), (id << 3) | (op_size == 1 ? 0x02 : 0x03));
}

// Generate code for shift instructions.
// Shift instructions include (ordered by their id): rol, ror, rcl, rcr, shl, shr, _, sar.
void Encoder::emit_shift(const Instruction& inst, int id) {
    const Operand& dst = inst.operands[0];
    const Operand& src = inst.operands[1];

    // Get size of dst. This also guards dst from holding immediate.
    int op_size = get_size(dst);

    // Shift by CL
    if (src.is_register() && src.as_register() == x86::Register::cl) {
        emit_r_rm(op_size, dst, static_cast<Register>(id), op_size == 1 ? 0xD2 : 0xD3);
        return;
    }

    // Otherwise src must be uint8.
    ASSERT(src.is_immediate());
    uint64_t imm = src.as_immediate();
    ASSERT(util::is_uint8(imm));

    if (imm == 1) {
        emit_r_rm(op_size, dst, static_cast<Register>(id), op_size == 1 ? 0xD0 : 0xD1);
    } else {
        emit_r_rm(op_size, dst, static_cast<Register>(id), op_size == 1 ? 0xC0 : 0xC1);
        emit_byte(imm);
    }
}

// Emit code for call
void Encoder::emit_call(const Instruction& inst) {
    const Operand& dst = inst.operands[0];
    ASSERT(inst.operands[1].is_empty());

    // Indirect jmp
    if (!dst.is_immediate()) {
        ASSERT(get_size(dst) == 8);
        emit_rex(dst, Register{2}, 0);
        emit_byte(0xFF);
        emit_modrm(dst, Register{2});
        return;
    }

    uint64_t imm = dst.as_immediate();
    ASSERT(util::is_int32(imm));
    emit_byte(0xE8);
    emit_dword(imm);
}

// Emit code for imul. 3-operand form is not yet supported.
void Encoder::emit_imul(const Instruction& inst) {

    int op_size = get_size(inst.operands[0]);

    // d:a = a * r/m
    if (inst.operands[1].is_empty()) {
        emit_r_rm(op_size, inst.operands[0], static_cast<Register>(5), op_size == 1 ? 0xF6 : 0xF7);
        return;
    }

    ASSERT(op_size != 1 && get_size(inst.operands[1]) == op_size);
    ASSERT(inst.operands[0].is_register());
    emit_r_rm(op_size, inst.operands[1], inst.operands[0].as_register(), 0x0FAF);
}

// Emit code for jcc
void Encoder::emit_jcc(const Instruction& inst) {
    const Operand& dst = inst.operands[0];
    ASSERT(inst.operands[1].is_empty() && dst.is_immediate());

    uint64_t imm = dst.as_immediate();
    if (util::is_int8(imm)) {
        emit_byte(0x70 + static_cast<uint8_t>(inst.cond));
        emit_byte(imm);
        return;
    }

    ASSERT(util::is_int32(imm));
    emit_byte(0x0F);
    emit_byte(0x80 + static_cast<uint8_t>(inst.cond));
    emit_dword(imm);
}

// Emit code for jmp
void Encoder::emit_jmp(const Instruction& inst) {
    const Operand& dst = inst.operands[0];
    ASSERT(inst.operands[1].is_empty());

    // Indirect jmp
    if (!dst.is_immediate()) {
        ASSERT(get_size(dst) == 8);
        emit_rex(dst, Register{4}, 0);
        emit_byte(0xFF);
        emit_modrm(dst, Register{4});
        return;
    }

    uint64_t imm = dst.as_immediate();
    if (util::is_int8(imm)) {
        emit_byte(0xEB);
        emit_byte(imm);
        return;
    }

    ASSERT(util::is_int32(imm));
    emit_byte(0xE9);
    emit_dword(imm);
}

// Emit code for lea.
void Encoder::emit_lea(const Instruction& inst) {

    // lea is special: only LEA r,m is allowed.
    ASSERT(inst.operands[0].is_register() && inst.operands[1].is_memory());
    int op_size = get_size(inst.operands[0]);
    ASSERT(op_size != 1);
    emit_r_rm(op_size, inst.operands[1], inst.operands[0].as_register(), 0x8D);
}

// Emit code for mov.
void Encoder::emit_mov(const Instruction& inst) {
    const Operand& dst = inst.operands[0];
    const Operand& src = inst.operands[1];
    int op_size = get_size(dst);

    if (src.is_immediate()) {
        uint64_t imm = src.as_immediate();

        // Special encoding for mov r, imm.
        if (dst.is_register()) {

            // Special optimization for mov: mov rax, uint32 can be optimized to mov eax, uint32.
            if (op_size == 8 && util::is_uint32(imm)) op_size = 4;

            // If the above optimization is not possible, but imm is int32, then it is shorter to encode it using
            // mod r/m.
            if (op_size != 8 || !util::is_int32(imm)) {
                emit_plusr(op_size, dst.as_register(), op_size == 1 ? 0xB0 : 0xB8);
                emit_immediate(op_size, imm);
                return;
            }
        }

        check_immediate_size(op_size, imm);
        emit_r_rm(op_size, dst, static_cast<Register>(0), op_size == 1 ? 0xC6 : 0xC7);
        emit_immediate(op_size == 8 ? 4 : op_size, imm);
        return;
    }

    // Make sure operand size matches. This also guards src from holding monostate.
    ASSERT(get_size(src) == op_size);

    // Prefer INST r/m, r to INST r, r/m in case of INST r, r
    if (src.is_register()) {
        emit_r_rm(op_size, dst, src.as_register(), op_size == 1 ? 0x88 : 0x89);
        return;
    }

    // Operands cannot both be memory.
    ASSERT(dst.is_register());
    emit_r_rm(op_size, src, dst.as_register(), op_size == 1 ? 0x8A : 0x8B);
}

// Emit code for movabs.
void Encoder::emit_movabs(const Instruction& inst) {
    const Operand& dst = inst.operands[0];
    const Operand& src = inst.operands[1];
    uint64_t imm;
    int op_size;
    uint8_t opcode;

    if (src.is_immediate()) {
        ASSERT(dst.is_register() && (static_cast<uint8_t>(dst.as_register()) & 0xF) == 0);

        imm = src.as_immediate();
        op_size = get_size(dst);
        opcode = 0xA0;
    } else {
        ASSERT(dst.is_immediate() && src.is_register() && (static_cast<uint8_t>(src.as_register()) & 0xF) == 0);

        imm = dst.as_immediate();
        op_size = get_size(src);
        opcode = 0xA2;
    }

    if (op_size == 2) {
        emit_byte(0x66);
    } else if (op_size == 8) {
        emit_byte(0x48);
    }

    emit_byte(op_size == 1 ? opcode : opcode + 1);
    emit_immediate(op_size, imm);
}

// Emit code for movsx.
void Encoder::emit_movsx(const Instruction& inst) {
    const Operand& dst = inst.operands[0];
    const Operand& src = inst.operands[1];

    // Must be movsx r, r/m
    ASSERT(dst.is_register());
    int dst_size = get_size(dst);
    int src_size = get_size(src);
    ASSERT(dst_size > src_size);
    emit_r_rm(dst_size, src, dst.as_register(), src_size == 1 ? 0x0FBE : (src_size == 2 ? 0x0FBF : 0x63));
}

// Emit code for movzx.
void Encoder::emit_movzx(const Instruction& inst) {
    const Operand& dst = inst.operands[0];
    const Operand& src = inst.operands[1];

    // Must be movzx r, r/m
    ASSERT(dst.is_register());
    int dst_size = get_size(dst);
    int src_size = get_size(src);
    ASSERT(dst_size > src_size && src_size != 4);
    emit_r_rm(dst_size, src, dst.as_register(), src_size == 1 ? 0x0FB6 : 0x0FB7);
}

// Emit code for pop.
void Encoder::emit_pop(const Instruction& inst) {
    const Operand& dst = inst.operands[0];
    ASSERT(inst.operands[1].is_empty());
    int op_size = get_size(dst);

    // Only 16 and 64 bit pop are encodable.
    ASSERT(op_size == 2 || op_size == 8);

    // REX.W not needed
    if (op_size == 8) op_size = 4;

    if (dst.is_register()) {
        emit_plusr(op_size, dst.as_register(), 0x58);
        return;
    }

    emit_r_rm(op_size, dst, static_cast<Register>(0), 0x8F);
}

// Emit code for push. Largely identical to pop but also supports push immediate.
void Encoder::emit_push(const Instruction& inst) {
    const Operand& dst = inst.operands[0];
    ASSERT(inst.operands[1].is_empty());

    // push imm. Note that we does not support push word xxx.
    if (inst.operands[0].is_immediate()) {
        uint64_t imm = inst.operands[0].as_immediate();
        if (util::is_int8(imm)) {
            emit_byte(0x6A);
            emit_byte(imm);
        } else {
            ASSERT(util::is_int32(imm));
            emit_byte(0x68);
            emit_dword(imm);
        }
        return;
    }

    int op_size = get_size(dst);

    // Only 16 and 64 bit pop are encodable.
    ASSERT(op_size == 2 || op_size == 8);

    // REX.W not needed
    if (op_size == 8) op_size = 4;

    if (dst.is_register()) {
        emit_plusr(op_size, dst.as_register(), 0x50);
        return;
    }

    emit_r_rm(op_size, dst, static_cast<Register>(0), 0xFF);
}

// Emit code for ret.
void Encoder::emit_ret(const Instruction& inst) {
    uint64_t imm = 0;

    // It can optionally have an immediate argument.
    if (!inst.operands[0].is_empty()) {
        ASSERT(inst.operands[0].is_immediate() && inst.operands[1].is_empty());
        imm = inst.operands[1].as_immediate();
        ASSERT(util::is_uint16(imm));
    }

    if (imm == 0) {
        emit_byte(0xC3);
    } else {
        emit_byte(0xC2);
        emit_word(imm);
    }
}

void Encoder::emit_setcc(const Instruction& inst) {
    ASSERT(inst.operands[1].is_empty());
    ASSERT(get_size(inst.operands[0]) == 1);
    emit_r_rm(1, inst.operands[0], static_cast<Register>(0), 0x0F90 + static_cast<uint8_t>(inst.cond));
}

void Encoder::emit_test(const Instruction& inst) {
    const Operand& dst = inst.operands[0];
    const Operand& src = inst.operands[1];

    int op_size = get_size(dst);

    if (src.is_immediate()) {
        uint64_t imm = src.as_immediate();
        check_immediate_size(op_size, imm);

        // Short encoding is available for RAX
        if (dst.is_register() && (static_cast<uint8_t>(dst.as_register()) & 0xF) == 0) {

            if (op_size == 2) {
                emit_byte(0x66);
            } else if (op_size == 8) {
                emit_byte(0x48);
            }

            emit_byte(op_size == 1 ? 0xA8 : 0xA9);
            emit_immediate(op_size == 8 ? 4 : op_size, imm);
            return;
        }

        emit_r_rm(op_size, dst, static_cast<Register>(0), op_size == 1 ? 0xF6 : 0xF7);
        emit_immediate(op_size == 8 ? 4 : op_size, imm);
        return;
    }

    ASSERT(src.is_register() && get_size(src) == op_size);
    emit_r_rm(op_size, dst, src.as_register(), op_size == 1 ? 0x84 : 0x85);
}

void Encoder::emit_xchg(const Instruction& inst) {
    Operand dst = inst.operands[0];
    Operand src = inst.operands[1];

    // Normalise to make src always register.
    ASSERT(dst.is_register() || src.is_register());
    if (!src.is_register()) std::swap(dst, src);

    int op_size = get_size(dst);

    // Special encoding exists if either operand is AX, EAX or RAX.
    if (op_size != 1 && dst.is_register()) {
        if ((static_cast<uint8_t>(src.as_register()) & 0xF) == 0) {
            emit_plusr(op_size, dst.as_register(), 0x90);
            return;
        } else if ((static_cast<uint8_t>(dst.as_register()) & 0xF) == 0) {
            emit_plusr(op_size, src.as_register(), 0x90);
            return;
        }
    }

    emit_r_rm(op_size, dst, src.as_register(), op_size == 1 ? 0x86 : 0x87);
}

void Encoder::encode(const Instruction& inst) {

    // If operand0 is monostate, then operand1 must also be.
    ASSERT(!inst.operands[0].is_empty() || inst.operands[1].is_empty());

    switch (inst.opcode) {
        /* ALU instructions */
        case Opcode::add: emit_alu(inst, 0); break;
        case Opcode::i_or: emit_alu(inst, 1); break;
        case Opcode::sbb: emit_alu(inst, 3); break;
        case Opcode::i_and: emit_alu(inst, 4); break;
        case Opcode::sub: emit_alu(inst, 5); break;
        case Opcode::i_xor: emit_alu(inst, 6); break;
        case Opcode::cmp: emit_alu(inst, 7); break;

        /* Shift instructions */
        case Opcode::shl: emit_shift(inst, 4); break;
        case Opcode::shr: emit_shift(inst, 5); break;
        case Opcode::sar: emit_shift(inst, 7); break;

        case Opcode::call: emit_call(inst); break;
        case Opcode::cdqe: emit_byte(0x48); emit_byte(0x98); break;
        case Opcode::cmovcc: emit_r_rm(inst.operands[1], inst.operands[0], 0x0F40 + static_cast<uint8_t>(inst.cond)); break;
        case Opcode::cdq: emit_byte(0x99); break;
        case Opcode::cqo: emit_byte(0x48); emit_byte(0x99); break;
        case Opcode::div: emit_rm(inst, 0xF6, 6); break;
        case Opcode::idiv: emit_rm(inst, 0xF6, 7); break;
        case Opcode::imul: emit_imul(inst); break;
        case Opcode::jcc: emit_jcc(inst); break;
        case Opcode::jmp: emit_jmp(inst); break;
        case Opcode::lea: emit_lea(inst); break;
        case Opcode::mov: emit_mov(inst); break;
        case Opcode::movabs: emit_movabs(inst); break;
        case Opcode::movsx: emit_movsx(inst); break;
        case Opcode::movzx: emit_movzx(inst); break;
        case Opcode::mul: emit_rm(inst, 0xF6, 4); break;
        case Opcode::neg: emit_rm(inst, 0xF6, 3); break;
        case Opcode::nop: emit_byte(0x90); break;
        case Opcode::i_not: emit_rm(inst, 0xF6, 2); break;
        case Opcode::pop: emit_pop(inst); break;
        case Opcode::push: emit_push(inst); break;
        case Opcode::ret: emit_ret(inst); break;
        case Opcode::setcc: emit_setcc(inst); break;
        case Opcode::test: emit_test(inst); break;
        case Opcode::xchg: emit_xchg(inst); break;
        default: ASSERT(0);
    }
}

}

