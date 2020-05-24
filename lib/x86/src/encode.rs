use super::op::{ConditionCode, Location, Memory, Op, Operand, Register, Size, REG_GPB2, REG_GPQ};
use core::convert::TryFrom;

pub struct Encoder<'a> {
    pub emitter: &'a mut dyn FnMut(u8),
}

// For most instructions taking immediates:
// If operation size is 64-bit, then imm must be int32.
// If operation size is 8, 16 or 32-bit, then imm must be int8/int16/int32 or uint8/uint16/uint32.
fn check_imm_size(size: Size, imm: i64) {
    match size {
        Size::Byte => assert!(i8::try_from(imm).is_ok() || u8::try_from(imm).is_ok()),
        Size::Word => assert!(i16::try_from(imm).is_ok() || u16::try_from(imm).is_ok()),
        Size::Dword => assert!(i32::try_from(imm).is_ok() || u32::try_from(imm).is_ok()),
        Size::Qword => assert!(i32::try_from(imm).is_ok()),
    }
}

// #region Routines for emitting constants
//

impl<'a> Encoder<'a> {
    fn emit_u8(&mut self, value: u8) {
        (self.emitter)(value);
    }

    fn emit_u16(&mut self, value: u16) {
        (self.emitter)(value as u8);
        (self.emitter)((value >> 8) as u8);
    }

    fn emit_u32(&mut self, value: u32) {
        (self.emitter)(value as u8);
        (self.emitter)((value >> 8) as u8);
        (self.emitter)((value >> 16) as u8);
        (self.emitter)((value >> 24) as u8);
    }

    fn emit_u64(&mut self, value: u64) {
        (self.emitter)(value as u8);
        (self.emitter)((value >> 8) as u8);
        (self.emitter)((value >> 16) as u8);
        (self.emitter)((value >> 24) as u8);
        (self.emitter)((value >> 32) as u8);
        (self.emitter)((value >> 40) as u8);
        (self.emitter)((value >> 48) as u8);
        (self.emitter)((value >> 56) as u8);
    }

    fn emit_imm(&mut self, size: Size, value: i64) {
        match size {
            Size::Byte => self.emit_u8(value as u8),
            Size::Word => self.emit_u16(value as u16),
            Size::Dword => self.emit_u32(value as u32),
            Size::Qword => self.emit_u64(value as u64),
        }
    }
}

//
// #endregion

impl<'a> Encoder<'a> {
    /// Emit REX prefix given r/m operand.
    /// Parameter rex specifies the bits that needs to be true in REX prefix.
    /// It can be 0x00 (no REX needed), 0x08 (need REX.W), or 0x40 (no bits needed, but need prefix)
    fn emit_rex(&mut self, op: &Location, reg_num: u8, mut rex: u8) {
        // For spl, bpl, sil, di, rex prefix is required.
        if (reg_num & 0xF0) == REG_GPB2 {
            rex |= 0x40;
        }

        // REX.R
        if (reg_num & 8) != 0 {
            rex |= 0x4;
        }

        match op {
            &Location::Reg(it) => {
                let it_num = it as u8;

                // For spl, bpl, sil, di, rex prefix is required.
                if (it_num & 0xF0) == REG_GPB2 {
                    rex |= 0x40;
                }

                // With REX prefix, r/m8 cannot be encoded to access AH, BH, CH, DH
                assert!(
                    rex == 0 || !(it_num >= Register::AH as u8 && it_num <= Register::BH as u8)
                );

                // REX.B
                if (it_num & 8) != 0 {
                    rex |= 0x1;
                }
            }
            Location::Mem(it) => {
                if let Some(base) = it.base {
                    // REX.B
                    if base as u8 & 8 != 0 {
                        rex |= 0x1;
                    }
                }

                if let Some((index, _)) = it.index {
                    // REX.X
                    if index as u8 & 8 != 0 {
                        rex |= 0x2;
                    }
                }
            }
        }

        if rex != 0 {
            assert!(!(reg_num >= Register::AH as u8 && reg_num <= Register::BH as u8));
            self.emit_u8(rex | 0x40);
        }
    }

    /// Emit ModR/M and SIB given r/m operand.
    /// This assumes 64-bit address size.
    fn emit_modrm(&mut self, op: &Location, reg_num: u8) {
        let reg_num = reg_num & 7;
        match op {
            &Location::Reg(it) => {
                // Take only the lowest 3 bit. 4th bit is encoded in REX and higher ones indicate register type.
                self.emit_u8(0xC0 | (reg_num << 3) | (it as u8 & 7));
            }
            Location::Mem(it) => {
                let mut shift = 0;

                // Sanity check that is valid and sp is not used as index register.
                if let Some((index, scale)) = it.index {
                    assert!(index as u8 & 0xF0 == REG_GPQ);

                    // index = RSP is invalid.
                    assert!(index as u8 & 0xF != 0b101);

                    shift = match scale {
                        1 => 0,
                        2 => 1,
                        4 => 2,
                        8 => 3,
                        _ => unreachable!(),
                    };
                }

                match (it.base, it.index) {
                    // [disp32]
                    (None, None) => {
                        self.emit_u8((reg_num << 3) | 0b100);
                        self.emit_u8(0x25);
                        self.emit_u32(it.displacement as u32);
                    }

                    // [index * scale + disp32]
                    (None, Some((index, _))) => {
                        self.emit_u8((reg_num << 3) | 0b100);
                        self.emit_u8((shift << 6) | ((index as u8 & 7) << 3) | 0b101);
                        self.emit_u32(it.displacement as u32);
                    }

                    // [RIP + disp32]
                    (Some(Register::RIP), None) => {
                        self.emit_u8((reg_num << 3) | 0b101);
                        self.emit_u32(it.displacement as u32);
                    }

                    // [RSP/R12 + disp]. We need a SIB byte in this case
                    (Some(Register::RSP), None) | (Some(Register::R12), None) => {
                        // [RSP/R12]
                        if it.displacement == 0 {
                            self.emit_u8((reg_num << 3) | 0b100);
                            self.emit_u8(0x24);
                            return;
                        }

                        // [RSP/R12 + disp8]
                        if i8::try_from(it.displacement).is_ok() {
                            self.emit_u8(0x40 | (reg_num << 3) | 0b100);
                            self.emit_u8(0x24);
                            self.emit_u8(it.displacement as u8);
                            return;
                        }

                        // [RSP/R12 + disp32]
                        self.emit_u8(0x80 | (reg_num << 3) | 0b100);
                        self.emit_u8(0x24);
                        self.emit_u32(it.displacement as u32);
                    }

                    // [base + disp]. Excluding RSP/R12
                    (Some(base), None) => {
                        let base_reg = base as u8 & 7;
                        assert!(base as u8 & 0xF0 == REG_GPQ);

                        // [base]. No direct encoding of [RBP/R13] however.
                        if it.displacement == 0 && base_reg != 0b101 {
                            self.emit_u8((reg_num << 3) | base_reg);
                            return;
                        }

                        // [base + disp8]
                        if i8::try_from(it.displacement).is_ok() {
                            self.emit_u8(0x40 | (reg_num << 3) | base_reg);
                            self.emit_u8(it.displacement as u8);
                            return;
                        }

                        // [base + disp32]
                        self.emit_u8(0x80 | (reg_num << 3) | base_reg);
                        self.emit_u32(it.displacement as u32);
                    }

                    (Some(base), Some((index, _))) => {
                        let base_reg = base as u8 & 7;
                        let index_reg = index as u8 & 7;
                        assert!(base as u8 & 0xF0 == REG_GPQ);

                        // [base + index * scale]. Similarly, base cannot be RBP/R13
                        if it.displacement == 0 && base_reg != 0b101 {
                            self.emit_u8((reg_num << 3) | 0b100);
                            self.emit_u8((shift << 6) | (index_reg << 3) | base_reg);
                            return;
                        }

                        // [base + index * scale + disp8]
                        if i8::try_from(it.displacement).is_ok() {
                            self.emit_u8(0x40 | (reg_num << 3) | 0b100);
                            self.emit_u8((shift << 6) | (index_reg << 3) | base_reg);
                            self.emit_u8(it.displacement as u8);
                            return;
                        }

                        // [base + index * scale + disp32]
                        self.emit_u8(0x80 | (reg_num << 3) | 0b100);
                        self.emit_u8((shift << 6) | (index_reg << 3) | base_reg);
                        self.emit_u32(it.displacement as u32);
                    }
                }
            }
        }
    }

    /// Generic helper function emitting all instructions that operate on r/rm, supporting 8, 16, 32 and 64-bit.
    /// Argument mem does not have to be a memory operand.
    /// Encoding of opcode: highest byte is reserved for potential mandatory prefix, and lower 3 bytes, if non-zero,
    /// will be emitted. The last byte will be emitted even if it's zero. Note that this works since 00 will not appear
    /// as an escape code.
    fn emit_r_rm(&mut self, op_size: Size, mem: &Location, reg_num: u8, opcode: u32) {
        // Operand size override prefix.
        if op_size == Size::Word {
            self.emit_u8(0x66)
        }

        // TODO: If required, emit mandatory prefix here.

        // Emit REX prefix is necessary. Note that the prefix must comes immediately before opcode.
        self.emit_rex(mem, reg_num, if op_size == Size::Qword { 0x08 } else { 0 });

        // Emit opcode.
        if opcode & 0xFF_0000 != 0 {
            self.emit_u8((opcode >> 16) as u8)
        }
        if opcode & 0xFF00 != 0 {
            self.emit_u8((opcode >> 8) as u8)
        }
        self.emit_u8(opcode as u8);

        // Mod R/M comes last.
        self.emit_modrm(mem, reg_num);
    }

    /// Generic helper function emitting all instructions that uses format +r, supporting 8, 16, 32 and 64-bit.
    fn emit_plusr(&mut self, op_size: Size, reg: Register, opcode: u32) {
        let reg_num = reg as u8;

        if op_size == Size::Word {
            self.emit_u8(0x66)
        }

        // TODO: If required, emit mandatory prefix here.

        // Emit REX prefix is necessary.
        let mut rex = 0;
        if op_size == Size::Qword {
            rex |= 0x08
        }
        if reg_num & 8 != 0 {
            rex |= 0x01
        }
        if reg_num & 0xF0 == REG_GPB2 {
            rex |= 0x40
        }

        if rex != 0 {
            assert!(!(reg_num >= Register::AH as u8 && reg_num <= Register::BH as u8));
            self.emit_u8(rex | 0x40);
        }

        // Emit opcode.
        if opcode & 0xFF_0000 != 0 {
            self.emit_u8((opcode >> 16) as u8)
        }
        if opcode & 0xFF00 != 0 {
            self.emit_u8((opcode >> 8) as u8)
        }
        self.emit_u8(opcode as u8 | (reg_num & 7));
    }

    /// Generate code for instructions with only RM encoding.
    /// It assumes the default opcode is for byte sized operands, and other sizes have opcode + 1.
    fn emit_rm(&mut self, dst: Location, mut opcode: u32, id: u8) {
        let op_size = dst.size();
        if op_size != Size::Byte {
            opcode += 1;
        }
        self.emit_r_rm(op_size, &dst, id, opcode);
    }

    /// Generate code for ALU instructions.
    /// ALU instructions include (ordered by their id): add, or, adc, sbb, and, sub, xor, cmp.
    fn emit_alu(&mut self, dst: Location, src: Operand, id: u8) {
        let op_size = dst.size();

        match src.as_loc() {
            Err(imm) => {
                // Check that the immediate size is encodable.
                check_imm_size(op_size, imm);

                // Short encoding available for 8-bit immediate.
                if op_size != Size::Byte && i8::try_from(imm).is_ok() {
                    self.emit_r_rm(op_size, &dst, id, 0x83);
                    self.emit_u8(imm as u8);
                    return;
                }

                // Short encoding is available for RAX
                if let Location::Reg(reg) = dst {
                    if reg as u8 & 0xF == 0 {
                        if op_size == Size::Word {
                            self.emit_u8(0x66);
                        } else if op_size == Size::Qword {
                            self.emit_u8(0x48);
                        }

                        self.emit_u8((id << 3) | if op_size == Size::Byte { 0x04 } else { 0x05 });
                        self.emit_imm(op_size.cap_to_dword(), imm);
                        return;
                    }
                }

                self.emit_r_rm(op_size, &dst, id, if op_size == Size::Byte { 0x80 } else { 0x81 });
                self.emit_imm(op_size.cap_to_dword(), imm);
            }

            Ok(src) => {
                assert_eq!(src.size(), op_size);

                match (dst, src) {
                    // Prefer INST r/m, r to INST r, r/m in case of INST r, r
                    (dst, Location::Reg(reg)) => self.emit_r_rm(
                        op_size,
                        &dst,
                        reg as u8,
                        (id << 3) as u32 | if op_size == Size::Byte { 0x00 } else { 0x01 },
                    ),
                    (Location::Reg(reg), src) => self.emit_r_rm(
                        op_size,
                        &src,
                        reg as u8,
                        (id << 3) as u32 | if op_size == Size::Byte { 0x02 } else { 0x03 },
                    ),
                    // Operands cannot both be memory.
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Generate code for shift instructions.
    /// Shift instructions include (ordered by their id): rol, ror, rcl, rcr, shl, shr, _, sar.
    fn emit_shift(&mut self, dst: Location, src: Operand, id: u8) {
        let op_size = dst.size();

        match src {
            // Shift by CL
            Operand::Reg(Register::CL) => {
                self.emit_r_rm(op_size, &dst, id, if op_size == Size::Byte { 0xD2 } else { 0xD3 })
            }
            Operand::Imm(1) => {
                self.emit_r_rm(op_size, &dst, id, if op_size == Size::Byte { 0xD0 } else { 0xD1 })
            }
            Operand::Imm(imm) => {
                self.emit_r_rm(op_size, &dst, id, if op_size == Size::Byte { 0xC0 } else { 0xC1 });
                self.emit_u8(u8::try_from(imm).unwrap());
            }
            _ => unreachable!(),
        }
    }

    /// Emit code for call
    fn emit_call(&mut self, target: Operand) {
        match target.as_loc() {
            Ok(target) => {
                // Indirect jump
                assert_eq!(target.size(), Size::Qword);
                self.emit_rex(&target, 2, 0);
                self.emit_u8(0xFF);
                self.emit_modrm(&target, 2);
            }

            Err(imm) => {
                assert!(i32::try_from(imm).is_ok());
                self.emit_u8(0xE8);
                self.emit_u32(imm as u32);
            }
        }
    }

    /// Emit code for jcc
    fn emit_jcc(&mut self, target: i32, cond: ConditionCode) {
        if i8::try_from(target).is_ok() {
            self.emit_u8(0x70 + cond as u8);
            self.emit_u8(target as u8);
        } else {
            self.emit_u8(0x0F);
            self.emit_u8(0x80 + cond as u8);
            self.emit_u32(target as u32);
        }
    }

    /// Emit code for jmp
    fn emit_jmp(&mut self, target: Operand) {
        match target.as_loc() {
            Ok(target) => {
                // Indirect jump
                assert_eq!(target.size(), Size::Qword);
                self.emit_rex(&target, 4, 0);
                self.emit_u8(0xFF);
                self.emit_modrm(&target, 4);
            }

            Err(imm) => {
                if i8::try_from(imm).is_ok() {
                    self.emit_u8(0xEB);
                    self.emit_u8(imm as u8);
                } else {
                    assert!(i32::try_from(imm).is_ok());
                    self.emit_u8(0xE9);
                    self.emit_u32(imm as u32);
                }
            }
        }
    }

    /// Emit code for lea.
    fn emit_lea(&mut self, dst: Register, src: Memory) {
        let op_size = dst.size();
        assert_ne!(op_size, Size::Byte);
        self.emit_r_rm(op_size, &Location::Mem(src), dst as u8, 0x8D);
    }

    /// Emit code for mov.
    fn emit_mov(&mut self, dst: Location, src: Operand) {
        let mut op_size = dst.size();

        match src.as_loc() {
            Err(imm) => {
                // Special encoding for mov r, imm.
                if let Location::Reg(reg) = dst {
                    // Special optimization for mov: mov rax, uint32 can be optimized to mov eax, uint32.
                    if op_size == Size::Qword && u32::try_from(imm).is_ok() {
                        op_size = Size::Dword
                    }

                    // If the above optimization is not possible, but imm is int32, then it is shorter to encode it using
                    // mod r/m.
                    if op_size != Size::Qword || i32::try_from(imm).is_err() {
                        self.emit_plusr(
                            op_size,
                            reg,
                            if op_size == Size::Byte { 0xB0 } else { 0xB8 },
                        );
                        self.emit_imm(op_size, imm);
                        return;
                    }
                }

                check_imm_size(op_size, imm);
                self.emit_r_rm(op_size, &dst, 0, if op_size == Size::Byte { 0xC6 } else { 0xC7 });
                self.emit_imm(op_size.cap_to_dword(), imm);
            }

            Ok(src) => {
                // Make sure operand size matches
                assert_eq!(src.size(), op_size);

                match (dst, src) {
                    // Prefer INST r/m, r to INST r, r/m in case of INST r, r
                    (dst, Location::Reg(reg)) => self.emit_r_rm(
                        op_size,
                        &dst,
                        reg as u8,
                        if op_size == Size::Byte { 0x88 } else { 0x89 },
                    ),
                    (Location::Reg(reg), src) => self.emit_r_rm(
                        op_size,
                        &src,
                        reg as u8,
                        if op_size == Size::Byte { 0x8A } else { 0x8B },
                    ),
                    // Operands cannot both be memory.
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Emit code for movabs.
    fn emit_movabs(&mut self, dst: Operand, src: Operand) {
        let (imm, reg, opcode) = match (dst, src) {
            (Operand::Reg(reg), Operand::Imm(imm)) => (imm, reg, 0xA0),
            (Operand::Imm(imm), Operand::Reg(reg)) => (imm, reg, 0xA2),
            _ => unreachable!(),
        };

        // Register can only be RAX or subsize
        assert_eq!(reg as u8 & 0xF, 0);

        let op_size = reg.size();
        match op_size {
            Size::Word => self.emit_u8(0x66),
            Size::Qword => self.emit_u8(0x48),
            _ => (),
        }
        self.emit_u8(if op_size == Size::Byte { opcode } else { opcode + 1 });
        self.emit_imm(op_size, imm);
    }

    /// Emit code for movsx.
    fn emit_movsx(&mut self, dst: Register, src: Location) {
        let dst_size = dst.size();
        let src_size = src.size();
        assert!(dst_size > src_size);
        let opcode = match src_size {
            Size::Byte => 0x0FBE,
            Size::Word => 0x0FBF,
            Size::Dword => 0x63,
            _ => unreachable!(),
        };
        self.emit_r_rm(dst_size, &src, dst as u8, opcode);
    }

    /// Emit code for movzx.
    fn emit_movzx(&mut self, dst: Register, src: Location) {
        let dst_size = dst.size();
        let src_size = src.size();
        assert!(dst_size > src_size && src_size != Size::Dword);
        self.emit_r_rm(
            dst_size,
            &src,
            dst as u8,
            if src_size == Size::Byte { 0x0FB6 } else { 0x0FB7 },
        );
    }

    /// Emit code for pop.
    fn emit_pop(&mut self, dst: Location) {
        let mut op_size = dst.size();

        // Only 16 and 64 bit pop are encodable.
        assert!(op_size == Size::Word || op_size == Size::Qword);

        // REX.W not needed
        if op_size == Size::Qword {
            op_size = Size::Dword
        }

        if let Location::Reg(reg) = dst {
            self.emit_plusr(op_size, reg, 0x58);
            return;
        }

        self.emit_r_rm(op_size, &dst, 0, 0x8F);
    }

    /// Emit code for push. Largely identical to pop but also supports push immediate.
    fn emit_push(&mut self, src: Operand) {
        match src.as_loc() {
            Err(imm) => {
                // push imm. Note that we does not support push word xxx.
                if i8::try_from(imm).is_ok() {
                    self.emit_u8(0x6A);
                    self.emit_u8(imm as u8);
                } else {
                    assert!(i32::try_from(imm).is_ok());
                    self.emit_u8(0x68);
                    self.emit_u32(imm as u32);
                }
            }

            Ok(src) => {
                let mut op_size = src.size();

                // Only 16 and 64 bit pop are encodable.
                assert!(op_size == Size::Word || op_size == Size::Qword);

                // REX.W not needed
                if op_size == Size::Qword {
                    op_size = Size::Dword
                }

                if let Location::Reg(reg) = src {
                    self.emit_plusr(op_size, reg, 0x50);
                    return;
                }

                self.emit_r_rm(op_size, &src, 0, 0xFF);
            }
        }
    }

    /// Emit code for ret.
    fn emit_ret(&mut self, pop: u16) {
        if pop == 0 {
            self.emit_u8(0xC3);
        } else {
            self.emit_u8(0xC2);
            self.emit_u16(pop);
        }
    }

    fn emit_setcc(&mut self, dst: Location, cc: ConditionCode) {
        assert_eq!(dst.size(), Size::Byte);
        self.emit_r_rm(Size::Byte, &dst, 0, 0x0F90 + cc as u8 as u32);
    }

    fn emit_test(&mut self, dst: Location, src: Operand) {
        let op_size = dst.size();

        match src {
            Operand::Imm(imm) => {
                check_imm_size(op_size, imm);

                // Short encoding is available for RAX
                if let Location::Reg(reg) = dst {
                    if reg as u8 & 0xF == 0 {
                        if op_size == Size::Word {
                            self.emit_u8(0x66);
                        } else if op_size == Size::Qword {
                            self.emit_u8(0x48);
                        }

                        self.emit_u8(if op_size == Size::Byte { 0xA8 } else { 0xA9 });
                        self.emit_imm(op_size.cap_to_dword(), imm);
                        return;
                    }
                }

                self.emit_r_rm(op_size, &dst, 0, if op_size == Size::Byte { 0xF6 } else { 0xF7 });
                self.emit_imm(op_size.cap_to_dword(), imm);
            }

            Operand::Reg(reg) => {
                assert_eq!(reg.size(), op_size);
                self.emit_r_rm(
                    op_size,
                    &dst,
                    reg as u8,
                    if op_size == Size::Byte { 0x84 } else { 0x85 },
                );
            }

            _ => unreachable!(),
        }
    }

    fn emit_xchg(&mut self, dst: Location, src: Location) {
        // Normalise to make src always a register
        let (dst, src) = match (dst, src) {
            (Location::Reg(src), dst) | (dst, Location::Reg(src)) => (dst, src),
            _ => unreachable!(),
        };

        let op_size = dst.size();
        assert_eq!(op_size, src.size());

        // Special encoding exists if either operand is AX, EAX or RAX.
        if op_size != Size::Byte {
            if let Location::Reg(reg) = dst {
                if src as u8 & 0xF == 0 {
                    self.emit_plusr(op_size, reg, 0x90);
                    return;
                } else if reg as u8 & 0xF == 0 {
                    self.emit_plusr(op_size, src, 0x90);
                    return;
                }
            }
        }

        self.emit_r_rm(op_size, &dst, src as u8, if op_size == Size::Byte { 0x86 } else { 0x87 });
    }

    pub fn encode(&mut self, op: Op) {
        match op {
            // ALU instructions
            Op::Add(dst, src) => self.emit_alu(dst, src, 0),
            Op::Or(dst, src) => self.emit_alu(dst, src, 1),
            Op::Adc(dst, src) => self.emit_alu(dst, src, 2),
            Op::Sbb(dst, src) => self.emit_alu(dst, src, 3),
            Op::And(dst, src) => self.emit_alu(dst, src, 4),
            Op::Sub(dst, src) => self.emit_alu(dst, src, 5),
            Op::Xor(dst, src) => self.emit_alu(dst, src, 6),
            Op::Cmp(dst, src) => self.emit_alu(dst, src, 7),

            // Shift instructions
            Op::Shl(dst, src) => self.emit_shift(dst, src, 4),
            Op::Shr(dst, src) => self.emit_shift(dst, src, 5),
            Op::Sar(dst, src) => self.emit_shift(dst, src, 7),

            Op::Illegal => {
                self.emit_u8(0x0F);
                self.emit_u8(0x0B);
            }
            Op::Call(target) => self.emit_call(target),
            Op::Cdqe => {
                self.emit_u8(0x48);
                self.emit_u8(0x98);
            }
            Op::Cmovcc(dst, src, cc) => {
                let op_size = dst.size();
                assert_ne!(op_size, Size::Byte);
                assert_eq!(op_size, src.size());
                self.emit_r_rm(op_size, &src, dst as u8, 0x0F40 + cc as u8 as u32);
            }
            Op::Cdq => self.emit_u8(0x99),
            Op::Cmpxchg(dst, src) => {
                let op_size = dst.size();
                assert_eq!(op_size, src.size());
                self.emit_r_rm(
                    op_size,
                    &dst,
                    src as u8,
                    if op_size == Size::Byte { 0x0FB0 } else { 0x0FB1 },
                );
            }
            Op::Cqo => {
                self.emit_u8(0x48);
                self.emit_u8(0x99)
            }
            Op::Div(src) => self.emit_rm(src, 0xF6, 6),
            Op::Hlt => self.emit_u8(0xf4),
            Op::Idiv(src) => self.emit_rm(src, 0xF6, 7),
            Op::Imul1(src) => {
                let op_size = src.size();
                self.emit_r_rm(op_size, &src, 5, if op_size == Size::Byte { 0xF6 } else { 0xF7 });
            }
            Op::Imul2(dst, src) => {
                let op_size = dst.size();
                assert_ne!(op_size, Size::Byte);
                assert_eq!(op_size, src.size());
                self.emit_r_rm(op_size, &src, dst as u8, 0x0FAF);
            }
            Op::Jcc(target, cc) => self.emit_jcc(target, cc),
            Op::Lock => self.emit_u8(0xF0),
            Op::Jmp(target) => self.emit_jmp(target),
            Op::Lea(dst, src) => self.emit_lea(dst, src),
            Op::Mfence => {
                self.emit_u8(0x0F);
                self.emit_u8(0xAE);
                self.emit_u8(0xF0);
            }
            Op::Mov(dst, src) => self.emit_mov(dst, src),
            Op::Movabs(dst, src) => self.emit_movabs(dst, src),
            Op::Movsx(dst, src) => self.emit_movsx(dst, src),
            Op::Movzx(dst, src) => self.emit_movzx(dst, src),
            Op::Mul(src) => self.emit_rm(src, 0xF6, 4),
            Op::Neg(dst) => self.emit_rm(dst, 0xF6, 3),
            Op::Nop => self.emit_u8(0x90),
            Op::Not(dst) => self.emit_rm(dst, 0xF6, 2),
            Op::Pop(dst) => self.emit_pop(dst),
            Op::Push(src) => self.emit_push(src),
            Op::Ret(pop) => self.emit_ret(pop),
            Op::Setcc(dst, cc) => self.emit_setcc(dst, cc),
            Op::Test(dst, src) => self.emit_test(dst, src),
            Op::Xadd(dst, src) => {
                let op_size = dst.size();
                assert_eq!(op_size, src.size());
                self.emit_r_rm(
                    op_size,
                    &dst,
                    src as u8,
                    if op_size == Size::Byte { 0x0FC0 } else { 0x0FC1 },
                );
            }
            Op::Xchg(dst, src) => self.emit_xchg(dst, src),
        }
    }
}

pub fn encode(op: Op, emitter: &mut dyn FnMut(u8)) {
    let mut encoder = Encoder { emitter };
    encoder.encode(op);
}
