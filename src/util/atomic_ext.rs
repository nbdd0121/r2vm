//! Currently atomic min/max/update is unstable. This is a stable implementation using CAS loop.

use core::sync::atomic::{AtomicI32, AtomicI64, AtomicU32, AtomicU64, Ordering};

pub trait AtomicExt {
    type Type;
    fn fetch_max_stable(&self, val: Self::Type, order: Ordering) -> Self::Type;
    fn fetch_min_stable(&self, val: Self::Type, order: Ordering) -> Self::Type;
    fn fetch_update_stable(
        &self,
        f: impl FnMut(Self::Type) -> Option<Self::Type>,
        fetch_order: Ordering,
        set_order: Ordering,
    ) -> Result<Self::Type, Self::Type>;
}

macro_rules! generate {
    ($atype: ident, $itype: ident) => {
        impl AtomicExt for $atype {
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

            fn fetch_update_stable(
                &self,
                mut f: impl FnMut(Self::Type) -> Option<Self::Type>,
                fetch_order: Ordering,
                set_order: Ordering,
            ) -> Result<Self::Type, Self::Type> {
                let mut prev = self.load(fetch_order);
                while let Some(next) = f(prev) {
                    match self.compare_exchange_weak(prev, next, set_order, fetch_order) {
                        x @ Ok(_) => return x,
                        Err(next_prev) => prev = next_prev,
                    }
                }
                Err(prev)
            }
        }
    };
}

generate!(AtomicI32, i32);
generate!(AtomicI64, i64);
generate!(AtomicU32, u32);
generate!(AtomicU64, u64);
