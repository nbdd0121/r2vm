use std::sync::atomic::{AtomicBool, Ordering};

pub struct RawMutex(AtomicBool);

unsafe impl lock_api::RawMutex for RawMutex {
    type GuardMarker = lock_api::GuardSend;

    const INIT: Self = Self(AtomicBool::new(false));

    #[inline]
    fn lock(&self) {
        while !self.try_lock() {
            std::sync::atomic::spin_loop_hint();
        }
    }

    #[inline]
    fn try_lock(&self) -> bool {
        !self.0.compare_and_swap(false, true, Ordering::Acquire)
    }

    #[inline]
    fn unlock(&self) {
        self.0.store(false, Ordering::Release);
    }
}

pub type Mutex<T> = lock_api::Mutex<RawMutex, T>;
pub type MutexGuard<'a, T> = lock_api::MutexGuard<'a, RawMutex, T>;
