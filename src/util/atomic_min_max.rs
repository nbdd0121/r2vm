//! Currently atomic min/max is unstable. This is a stable implementation using CAS loop.

use core::sync::atomic::{AtomicI32, AtomicU32, AtomicI64, AtomicU64, Ordering};

pub trait AtomicMinMax {
    type Type;
    fn fetch_max_stable(&self, val: Self::Type, order: Ordering) -> Self::Type;
    fn fetch_min_stable(&self, val: Self::Type, order: Ordering) -> Self::Type;
}

macro_rules! generate {
    ($atype: ident, $itype: ident) => {
        impl AtomicMinMax for $atype {
            type Type = $itype;
            fn fetch_max_stable(&self, val: Self::Type, order: Ordering) -> Self::Type {
                let mut current = self.load(Ordering::Relaxed);
                loop {
                    let new = $itype::max(current, val);
                    match self.compare_exchange_weak(current, new, order, Ordering::Relaxed) {
                        Ok(v) => return v,
                        Err(v) => current = v,
                    }
                }
            }
            fn fetch_min_stable(&self, val: Self::Type, order: Ordering) -> Self::Type {
                let mut current = self.load(Ordering::Relaxed);
                loop {
                    let new = $itype::min(current, val);
                    match self.compare_exchange_weak(current, new, order, Ordering::Relaxed) {
                        Ok(v) => return v,
                        Err(v) => current = v,
                    }
                }
            }
        }
    };
}

generate!(AtomicI32, i32);
generate!(AtomicI64, i64);
generate!(AtomicU32, u32);
generate!(AtomicU64, u64);
