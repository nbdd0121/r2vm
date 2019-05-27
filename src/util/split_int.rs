pub trait SplitInt {
    type Half;
    fn lo(&self) -> Self::Half;
    fn hi(&self) -> Self::Half;
    fn set_lo(&mut self, lo: Self::Half);
    fn set_hi(&mut self, hi: Self::Half);
}

impl SplitInt for u64 {
    type Half = u32;
    fn lo(&self) -> u32 { *self as u32 }
    fn hi(&self) -> u32 { (*self >> 32) as u32 }
    fn set_lo(&mut self, lo: u32) { *self = (*self &! 0xFFFFFFFFu64) | (lo as u64) }
    fn set_hi(&mut self, hi: u32) { *self = (*self & 0xFFFFFFFFu64) | ((hi as u64) << 32) }
}
