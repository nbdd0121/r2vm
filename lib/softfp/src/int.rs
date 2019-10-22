//! Helper traits for templating over integer types.

use core::ops;

/// Trait for fixed-size integer types.
pub trait Int:
    Sized
    + Ord
    + Eq
    + Copy
    + ops::Shr<u32, Output = Self>
    + ops::Shl<u32, Output = Self>
    + ops::Add<Self, Output = Self>
    + ops::Sub<Self, Output = Self>
    + ops::Mul<Self, Output = Self>
    + ops::BitAnd<Self, Output = Self>
    + ops::BitOr<Self, Output = Self>
    + ops::ShrAssign<u32>
    + ops::ShlAssign<u32>
    + ops::AddAssign<Self>
    + ops::SubAssign<Self>
    + ops::BitAndAssign<Self>
    + ops::BitOrAssign<Self>
    + ops::Not<Output = Self>
    + core::fmt::Debug
{
    fn zero() -> Self;
    fn one() -> Self;
    fn max_value() -> Self;
    fn bit_width() -> u32 {
        core::mem::size_of::<Self>() as u32 * 8
    }
}

impl Int for u32 {
    fn zero() -> u32 {
        0
    }
    fn one() -> u32 {
        1
    }
    fn max_value() -> u32 {
        u32::max_value()
    }
}

impl Int for u64 {
    fn zero() -> u64 {
        0
    }
    fn one() -> u64 {
        1
    }
    fn max_value() -> u64 {
        u64::max_value()
    }
}

impl Int for u128 {
    fn zero() -> u128 {
        0
    }
    fn one() -> u128 {
        1
    }
    fn max_value() -> u128 {
        u128::max_value()
    }
}

/// Trait for unsigned fixed-size integer types.
pub trait UInt: Int {
    fn log2_floor(self) -> u32;
}

impl UInt for u32 {
    fn log2_floor(self) -> u32 {
        31 - self.leading_zeros()
    }
}

impl UInt for u64 {
    fn log2_floor(self) -> u32 {
        63 - self.leading_zeros()
    }
}

impl UInt for u128 {
    fn log2_floor(self) -> u32 {
        127 - self.leading_zeros()
    }
}

/// Numerical cast between types. Semantically identical to "as" operator, and can be lossy.
pub trait CastFrom<T> {
    fn cast_from(value: T) -> Self;
}

impl<T: Int> CastFrom<T> for T {
    fn cast_from(value: Self) -> Self {
        value
    }
}

impl CastFrom<u64> for u8 {
    fn cast_from(value: u64) -> u8 {
        value as u8
    }
}

impl CastFrom<u64> for u16 {
    fn cast_from(value: u64) -> u16 {
        value as u16
    }
}

impl CastFrom<u64> for u32 {
    fn cast_from(value: u64) -> u32 {
        value as u32
    }
}

impl CastFrom<u128> for u32 {
    fn cast_from(value: u128) -> u32 {
        value as u32
    }
}

impl CastFrom<u32> for u64 {
    fn cast_from(value: u32) -> u64 {
        value as u64
    }
}

impl CastFrom<u128> for u64 {
    fn cast_from(value: u128) -> u64 {
        value as u64
    }
}

impl CastFrom<u32> for u128 {
    fn cast_from(value: u32) -> u128 {
        value as u128
    }
}

impl CastFrom<u64> for u128 {
    fn cast_from(value: u64) -> u128 {
        value as u128
    }
}

/// Numerical cast between types. Semantically identical to "as" operator, and can be lossy.
pub trait CastTo<T> {
    fn cast_to(self) -> T;
}

impl<T, U: CastFrom<T>> CastTo<U> for T {
    fn cast_to(self) -> U {
        U::cast_from(self)
    }
}
