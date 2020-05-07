//! Currently atomic min/max/update is unstable. This is a stable implementation using CAS loop.
//! Remove as soon as they are stablised.
use core::sync::atomic::{
    AtomicI16, AtomicI32, AtomicI64, AtomicI8, AtomicIsize, AtomicU16, AtomicU32, AtomicU64,
    AtomicU8, AtomicUsize, Ordering,
};

/// Extension trait to provide `fetch_max`, `fetch_min` and `fetch_update` to stable Rust.
pub trait AtomicExt {
    /// Associated primitive integer type.
    type Type;

    /// Maximum with the current value.
    ///
    /// Finds the maximum of the current value and the argument val, and sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// `fetch_max_stable` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// [`Relaxed`]: Ordering::Relaxed
    /// [`Release`]: Ordering::Release
    /// [`Acquire`]: Ordering::Acquire
    fn fetch_max_stable(&self, val: Self::Type, order: Ordering) -> Self::Type;

    /// Minimum with the current value.
    ///
    /// Finds the minimum of the current value and the argument val, and sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// `fetch_min_stable` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// [`Relaxed`]: Ordering::Relaxed
    /// [`Release`]: Ordering::Release
    /// [`Acquire`]: Ordering::Acquire
    fn fetch_min_stable(&self, val: Self::Type, order: Ordering) -> Self::Type;

    /// Fetches the value, and applies a function to it that returns an optional
    /// new value. Returns a `Result` of `Ok(previous_value)` if the function returned `Some(_)`, else
    /// `Err(previous_value)`.
    ///
    /// Note: This may call the function multiple times if the value has been changed from other threads in
    /// the meantime, as long as the function returns `Some(_)`, but the function will have been applied
    /// but once to the stored value.
    ///
    /// `fetch_update_stable` takes two [`Ordering`] arguments to describe the memory
    /// ordering of this operation. The second describes the required ordering for loads
    /// and failed updates while the first describes the required ordering when the
    /// operation finally succeeds.
    ///
    /// Using [`Acquire`] as success ordering makes the store part
    /// of this operation [`Relaxed`], and using [`Release`] makes the final successful load
    /// [`Relaxed`]. The (failed) load ordering can only be [`SeqCst`], [`Acquire`] or [`Relaxed`]
    /// and must be equivalent to or weaker than the success ordering.
    ///
    /// [`Relaxed`]: Ordering::Relaxed
    /// [`Release`]: Ordering::Release
    /// [`Acquire`]: Ordering::Acquire
    /// [`SeqCst`]: Ordering::SeqCst
    fn fetch_update_stable(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        f: impl FnMut(Self::Type) -> Option<Self::Type>,
    ) -> Result<Self::Type, Self::Type>;
}

macro_rules! generate {
    ($atype: ident, $itype: ident) => {
        impl AtomicExt for $atype {
            type Type = $itype;

            #[inline]
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

            #[inline]
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

            #[inline]
            fn fetch_update_stable(
                &self,
                set_order: Ordering,
                fetch_order: Ordering,
                mut f: impl FnMut(Self::Type) -> Option<Self::Type>,
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

generate!(AtomicI8, i8);
generate!(AtomicU8, u8);
generate!(AtomicI16, i16);
generate!(AtomicU16, u16);
generate!(AtomicI32, i32);
generate!(AtomicU32, u32);
generate!(AtomicI64, i64);
generate!(AtomicU64, u64);
generate!(AtomicIsize, isize);
generate!(AtomicUsize, usize);
