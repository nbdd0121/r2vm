use super::op::{LegacyOp, Op};

extern {
    fn legacy_decode(inst: u32) -> LegacyOp;
}

pub fn decode(bits: u32) -> Op {
    // Compressed ops
    if bits & 3 != 3 {
        return Op::Legacy(unsafe { legacy_decode(bits) })
    }

    // Longer ops, treat them as illegal ops
    if bits & 0x1f == 0x1f { return Op::Illegal }
    match bits & 0b1111111 {
        _ => Op::Legacy(unsafe { legacy_decode(bits) }),
    }
}
