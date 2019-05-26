#![allow(dead_code)]

use std::ops;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU32, Ordering as MemOrder};

// #region Exception flags constant and manipulation

const EX_INEXACT           : u32 = 1;
const EX_UNDERFLOW         : u32 = 2;
const EX_OVERFLOW          : u32 = 4;
const EX_DIVIDE_BY_ZERO    : u32 = 8;
const EX_INVALID_OPERATION : u32 = 16;

thread_local!(
    /// We use atomic here merely to avoid the cost of RefCell. So we should use relaxed ordering.
    static EXCEPTION_FLAGS: AtomicU32 = AtomicU32::new(0)
);

#[inline]
pub fn clear_exception_flag() {
    EXCEPTION_FLAGS.with(|flags| {
        flags.store(0, MemOrder::Relaxed);
    })
}

#[inline]
pub fn get_exception_flag() -> u32 {
    EXCEPTION_FLAGS.with(|flags| {
        flags.load(MemOrder::Relaxed)
    })
}

#[inline]
fn set_exception_flag(flag: u32) {
    EXCEPTION_FLAGS.with(|flags| {
        flags.fetch_or(flag, MemOrder::Relaxed);
    })
}

//
// #endregion

const CLASS_NEGATIVE_INFINITY  : u32 = 0;
const CLASS_NEGATIVE_NORMAL    : u32 = 1;
const CLASS_NEGATIVE_SUBNORMAL : u32 = 2;
const CLASS_NEGATIVE_ZERO      : u32 = 3;
const CLASS_POSITIVE_ZERO      : u32 = 4;
const CLASS_POSITIVE_SUBNORMAL : u32 = 5;
const CLASS_POSITIVE_NORMAL    : u32 = 6;
const CLASS_POSITIVE_INFINITY  : u32 = 7;
const CLASS_SIGNALING_NAN      : u32 = 8;
const CLASS_QUIET_NAN          : u32 = 9;

pub trait Int: Sized + Ord + Eq + Copy +
    ops::Shr<u32, Output=Self> +
    ops::Shl<u32, Output=Self> +
    ops::Add<Self, Output=Self> +
    ops::Sub<Self, Output=Self> +
    ops::BitAnd<Self, Output=Self> +
    ops::BitOr<Self, Output=Self> +
    ops::BitAndAssign<Self> +
    ops::BitOrAssign<Self> +
    ops::Not<Output=Self> +
    std::fmt::LowerHex +
    From<u32> {

    fn zero() -> Self;
    fn one() -> Self;
    /// Convert into u32. Necessary for extracting exponent.
    fn as_u32(self) -> u32;

    fn log2_floor(self) -> u32;
}

pub trait FpDesc: Copy {
    const EXPONENT_WIDTH: u32;
    const SIGNIFICAND_WIDTH: u32;
    type Holder: Int;
}

#[derive(Clone, Copy)]
pub struct Fp<Desc: FpDesc>(pub Desc::Holder);

impl<Desc: FpDesc> Fp<Desc> {

    // Special exponent values
    const INFINITY_BIASED_EXPONENT: u32 = (1 << Desc::EXPONENT_WIDTH) - 1;
    const MAXIMUM_BIASED_EXPONENT : u32 = Self::INFINITY_BIASED_EXPONENT - 1;
    const EXPONENT_BIAS           : u32 = (1 << (Desc::EXPONENT_WIDTH - 1)) - 1;
    const MINIMUM_EXPONENT        : u32 = 1 - Self::EXPONENT_BIAS;
    const MAXIMUM_EXPONENT        : u32 = Self::MAXIMUM_BIASED_EXPONENT - Self::EXPONENT_BIAS;

    #[inline]
    pub fn new(value: Desc::Holder) -> Self {
        Fp(value)
    }

    // #region Component accessors
    //

    fn sign(&self) -> bool {
        self.0 >> (Desc::EXPONENT_WIDTH + Desc::SIGNIFICAND_WIDTH) != Desc::Holder::zero()
    }

    fn set_sign(&mut self, sign: bool) {
        let mask = Desc::Holder::one() << (Desc::EXPONENT_WIDTH + Desc::SIGNIFICAND_WIDTH);
        if sign {
            self.0 |= mask;
        } else {
            self.0 &=! mask;
        }
    }

    fn biased_exponent(&self) -> u32 {
        let mask = (Desc::Holder::one() << Desc::EXPONENT_WIDTH) - Desc::Holder::one();
        ((self.0 >> Desc::SIGNIFICAND_WIDTH) & mask).as_u32()
    }

    /// Set the biased exponent of the floating pointer number.
    /// Only up to exponent_width bits are respected and all other bits are ignored.
    fn set_biased_exponent(&mut self, exp: u32) {
        let mask = ((Desc::Holder::one() << Desc::EXPONENT_WIDTH) - Desc::Holder::one()) << Desc::SIGNIFICAND_WIDTH;
        self.0 = (self.0 &! mask) | ((Into::<Desc::Holder>::into(exp) << Desc::SIGNIFICAND_WIDTH) & mask);
    }

    fn trailing_significand(&self) -> Desc::Holder {
        let mask = (Desc::Holder::one() << Desc::SIGNIFICAND_WIDTH) - Desc::Holder::one();
        self.0 & mask
    }

    // Set the trailing significand of the floating pointer number.
    // Only up to significand_width bits are respected and all other bits are ignored.
    fn set_trailing_significand(&mut self, value: Desc::Holder) {
        let mask = (Desc::Holder::one() << Desc::SIGNIFICAND_WIDTH) - Desc::Holder::one();
        self.0 = (self.0 &! mask) | (value & mask);
    }

    //
    // #endregion

    fn get_normalized_significand(&self) -> (u32, Desc::Holder) {
        let biased_exponent = self.biased_exponent();
        let trailing_significand = self.trailing_significand();

        // We couldn't handle this
        if biased_exponent == Self::INFINITY_BIASED_EXPONENT ||
            (biased_exponent == 0 && trailing_significand == Desc::Holder::zero()) { panic!() }

        if biased_exponent == 0 {
            let width_diff = Desc::SIGNIFICAND_WIDTH - trailing_significand.log2_floor();
            return (Self::MINIMUM_EXPONENT - width_diff, trailing_significand << (width_diff + 2));
        }

        let exponent = biased_exponent - Self::EXPONENT_BIAS;
        let significand = trailing_significand | (Desc::Holder::one() << Desc::SIGNIFICAND_WIDTH);
        (exponent, significand << 2)
    }

    fn get_significand(&self) -> (u32, Desc::Holder)  {
        let biased_exponent = self.biased_exponent();
        let trailing_significand = self.trailing_significand();

        // We couldn't handle this
        if biased_exponent == Self::INFINITY_BIASED_EXPONENT  { panic!() }

        if biased_exponent == 0 {
            // Perform lvalue-rvalue conversion to get rid of ODR-use of minimum_exponent.
            return (Self::MINIMUM_EXPONENT, trailing_significand << 2);
        }

        let exponent = biased_exponent - Self::EXPONENT_BIAS;
        let significand = trailing_significand | (Desc::Holder::one() << Desc::SIGNIFICAND_WIDTH);
        (exponent, significand << 2)
    }

    // #region Classification
    //

    #[allow(dead_code)]
    pub fn is_normal(&self) -> bool {
        let exponent = self.biased_exponent();
        exponent != 0 && exponent != Self::INFINITY_BIASED_EXPONENT
    }

    #[allow(dead_code)]
    pub fn is_finite(&self) -> bool {
        self.biased_exponent() != Self::INFINITY_BIASED_EXPONENT
    }

    pub fn is_zero(&self) -> bool {
        self.biased_exponent() == 0 &&
        self.trailing_significand() == Desc::Holder::zero()
    }

    #[allow(dead_code)]
    pub fn is_subnormal(&self) -> bool {
        self.biased_exponent() == 0 &&
        self.trailing_significand() != Desc::Holder::zero()
    }

    pub fn is_infinite(&self) -> bool {
        self.biased_exponent() == Self::INFINITY_BIASED_EXPONENT &&
        self.trailing_significand() == Desc::Holder::zero()
    }

    pub fn is_nan(&self) -> bool {
        self.biased_exponent() == Self::INFINITY_BIASED_EXPONENT &&
        self.trailing_significand() != Desc::Holder::zero()
    }

    pub fn is_signaling(&self) -> bool {
        // Special exponent for Infinites and NaNs
        if self.biased_exponent() != Self::INFINITY_BIASED_EXPONENT { return false }

        let t = self.trailing_significand();

        // t == 0 is Infinity
        if t == Desc::Holder::zero() { return false }

        // Signaling NaN has MSB = 0, otherwise quiet NaN
        t & (Desc::Holder::one() << (Desc::SIGNIFICAND_WIDTH - 1)) == Desc::Holder::zero()
    }

    /* IEEE 754-2008 5.7.2 Non-computation operations > General operations */
    pub fn classify(&self) -> u32 {
        let sign = self.sign();
        let exponent = self.biased_exponent();
        let significand = self.trailing_significand();
        let positive_class = if exponent == 0 {
            if significand == Desc::Holder::zero() {
                CLASS_POSITIVE_ZERO
            } else {
                CLASS_POSITIVE_SUBNORMAL
            }
        } else if exponent == Self::INFINITY_BIASED_EXPONENT {
            if significand == Desc::Holder::zero() {
                CLASS_POSITIVE_INFINITY
            } else if (significand & (Desc::Holder::one() << (Desc::SIGNIFICAND_WIDTH - 1))) == Desc::Holder::zero() {
                return CLASS_SIGNALING_NAN;
            } else {
                return CLASS_QUIET_NAN;
            }
        } else {
            CLASS_POSITIVE_NORMAL
        };
        // We use the property that negative and positive classes add up to 7.
        if sign { 7 - positive_class } else { positive_class }
    }

    //
    // #endregion

    // #region Comparison
    //

    /* IEEE 754-2008 5.6.1 Signaling-computational operations > Comparisions */
    pub fn compare_quiet(a: Self, b: Self) -> Option<Ordering> {
        if a.is_nan() || b.is_nan() {
            return None;
        }
        if a.is_zero() && b.is_zero() { return Some(Ordering::Equal) }
        Some(Self::total_order(a, b))
    }

    pub fn compare_signaling(a: Self, b: Self) -> Option<Ordering> {
        if a.is_nan() || b.is_nan() {
            set_exception_flag(EX_INVALID_OPERATION);
            return None;
        }
        if a.is_zero() && b.is_zero() { return Some(Ordering::Equal) }
        Some(Self::total_order(a, b))
    }

    pub fn total_order(a: Self, b: Self) -> Ordering {
        if a.sign() == b.sign() {
            let ret = Self::total_order_magnitude(a, b);
            if a.sign() {
                ret.reverse()
            } else {
                ret
            }
        } else if a.sign() {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }

    /// Compare the total order between abs(a) and abs(b)
    pub fn total_order_magnitude(mut a: Self, mut b: Self) -> Ordering {
        a.set_sign(false);
        b.set_sign(false);
        a.0.cmp(&b.0)
    }

    //
    // #endregion
}

impl<Desc: FpDesc> std::cmp::PartialEq for Fp<Desc> {
    fn eq(&self, other: &Self) -> bool {
        Fp::compare_quiet(*self, *other) == Some(Ordering::Equal)
    }
}

impl<Desc: FpDesc> std::cmp::PartialOrd for Fp<Desc> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Fp::compare_signaling(*self, *other)
    }
}

#[derive(Clone, Copy)]
pub struct F32Desc;

impl FpDesc for F32Desc {
    const EXPONENT_WIDTH: u32 = 8;
    const SIGNIFICAND_WIDTH: u32 = 23;
    type Holder = u32;
}

impl Int for u32 {
    fn zero() -> u32 { 0 }
    fn one() -> u32 { 1 }
    fn as_u32(self) -> u32 { self }
    fn log2_floor(self) -> u32 { 31 - self.leading_zeros() }
}

#[derive(Clone, Copy)]
pub struct F64Desc;

impl FpDesc for F64Desc {
    const EXPONENT_WIDTH: u32 = 11;
    const SIGNIFICAND_WIDTH: u32 = 52;
    type Holder = u64;
}

impl Int for u64 {
    fn zero() -> u64 { 0 }
    fn one() -> u64 { 1 }
    fn as_u32(self) -> u32 { self as u32 }
    fn log2_floor(self) -> u32 { 63 - self.leading_zeros() }
}

pub type F32 = Fp<F32Desc>;
pub type F64 = Fp<F64Desc>;
