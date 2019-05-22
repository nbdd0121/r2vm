use super::op::{LegacyOp, Op};

extern {
    fn legacy_decode(inst: u32) -> LegacyOp;
}

fn rd(bits: u32) -> u8 {
    ((bits >> 7) & 0b11111) as u8
}

fn rs1(bits: u32) -> u8 {
    ((bits >> 15) & 0b11111) as u8
}

fn rs2(bits: u32) -> u8 {
    ((bits >> 20) & 0b11111) as u8
}

fn rs3(bits: u32) -> u8 {
    ((bits >> 27) & 0b11111) as u8
}

fn u_imm(bits: u32) -> i32 {
    (bits & 0xfffff000) as i32
}

pub fn decode(bits: u32) -> Op {
    // Compressed ops
    if bits & 3 != 3 {
        return Op::Legacy(unsafe { legacy_decode(bits) })
    }

    // Longer ops, treat them as illegal ops
    if bits & 0x1f == 0x1f { return Op::Illegal }

    let rd = rd(bits);

    match bits & 0b1111111 {
        /* AUIPC */
        0b0010111 => Op::Auipc {
            rd: rd,
            imm: u_imm(bits),
        },
        _ => Op::Legacy(unsafe { legacy_decode(bits) }),
    }
}

pub fn decode_instr(pc: &mut u64, pc_next: u64) -> (Op, bool) {
    let mut bits = unsafe { crate::emu::read_memory::<u16>(*pc) as u32 };
    *pc += 2;
    if bits & 3 == 3 {
        if *pc & 4095 == 0 {
            bits |= (unsafe { crate::emu::read_memory::<u16>(pc_next) as u32 }) << 16;
        } else {
            bits |= (unsafe { crate::emu::read_memory::<u16>(*pc) as u32 }) << 16;
        }
        *pc += 2;
    }
    (decode(bits), bits & 3 != 3)
}

pub fn decode_block(mut pc: u64, pc_next: u64) -> (Vec<(Op, bool)>, u64, u64) {
    let start_pc = pc;
    let mut vec = Vec::new();
    loop {
        let (op, c) = decode_instr(&mut pc, pc_next);
        if op.can_change_control_flow() || (pc &! 4095) != (start_pc &! 4095) {
            vec.push((op, c));
            break
        }
        vec.push((op, c));
    }
    (vec, start_pc, pc)
}
