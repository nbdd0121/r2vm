//! Cache coherency related definitions

/// Represent capability granted on a cache line.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Capability {
    None = 0,
    Read = 1,
    Write = 2,
}

impl PartialOrd for Capability {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Capability {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

/// Sharer bitmask utility
#[derive(Clone, Copy)]
pub struct BitmaskSharer(pub u64);

impl BitmaskSharer {
    pub fn new() -> Self {
        BitmaskSharer(0)
    }

    pub fn test(&self, hartid: usize) -> bool {
        self.0 & (1 << hartid) != 0
    }

    pub fn set(&mut self, hartid: usize) {
        self.0 |= 1 << hartid
    }

    pub fn clear(&mut self, hartid: usize) {
        self.0 &= !(1 << hartid)
    }

    pub fn reset(&mut self) {
        self.0 = 0
    }

    pub fn empty(&self) -> bool {
        self.0 == 0
    }
}

use std::fmt;
impl fmt::Display for BitmaskSharer {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:4b}", self.0)
    }
}

pub type Sharer = BitmaskSharer;
