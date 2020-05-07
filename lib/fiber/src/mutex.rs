use super::park::{park, unpark_all, unpark_one};
use atomic_ext::AtomicExt;
use lock_api::RawMutex as LRawMutex;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

const LOCKED_BIT: u8 = 0b01;
const PARKED_BIT: u8 = 0b10;

/// Raw mutex type.
pub struct RawMutex {
    locked: AtomicU8,
}

impl RawMutex {
    #[cold]
    fn lock_slow(&self) {
        let mut spinwait = 0;
        let mut state = self.locked.load(Ordering::Relaxed);
        loop {
            // Acquire the lock if not locked
            if state & LOCKED_BIT == 0 {
                match self.locked.compare_exchange_weak(
                    state,
                    state | LOCKED_BIT,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(x) => state = x,
                }
                continue;
            }

            // If no queue is there, wait a few times
            if state & PARKED_BIT == 0 && spinwait < 10 {
                spinwait += 1;
                state = self.locked.load(Ordering::Relaxed);
                continue;
            }

            // Set the parked bit
            if state & PARKED_BIT == 0 {
                if let Err(x) = self.locked.compare_exchange_weak(
                    state,
                    state | PARKED_BIT,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    state = x;
                    continue;
                }
            }

            // Park the fiber until woekn up by unlock again
            park(
                self as *const _ as usize,
                || {
                    // Another thread races to unlock
                    self.locked.load(Ordering::Relaxed) == LOCKED_BIT | PARKED_BIT
                },
                || (),
            );

            spinwait = 0;
            state = self.locked.load(Ordering::Relaxed);
        }
    }

    #[cold]
    fn unlock_slow(&self) {
        unpark_one(self as *const _ as usize, |have_more| {
            self.locked.store(if have_more { PARKED_BIT } else { 0 }, Ordering::Release);
        });
    }
}

unsafe impl LRawMutex for RawMutex {
    type GuardMarker = lock_api::GuardSend;

    const INIT: Self = Self { locked: AtomicU8::new(0) };

    #[inline]
    fn lock(&self) {
        if self
            .locked
            .compare_exchange_weak(0, LOCKED_BIT, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            self.lock_slow();
        }
    }

    #[inline]
    fn try_lock(&self) -> bool {
        self.locked
            .fetch_update_stable(Ordering::Acquire, Ordering::Relaxed, |state| {
                if state & LOCKED_BIT != 0 {
                    return None;
                }
                Some(state | LOCKED_BIT)
            })
            .is_ok()
    }

    #[inline]
    fn unlock(&self) {
        if self
            .locked
            .compare_exchange(LOCKED_BIT, 0, Ordering::Release, Ordering::Relaxed)
            .is_err()
        {
            self.unlock_slow();
        }
    }
}

/// A mutual exclusion primitive useful for protecting shared data with fiber support.
pub type Mutex<T> = lock_api::Mutex<RawMutex, T>;

/// An RAII implementation of a "scoped lock" of a mutex. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
///
/// The data protected by the mutex can be accessed through this guard via its
/// [`Deref`] and [`DerefMut`] implementations.
///
/// [`Deref`]: std::ops::Deref
/// [`DerefMut`]: std::ops::DerefMut
pub type MutexGuard<'a, T> = lock_api::MutexGuard<'a, RawMutex, T>;

/// A Condition Variable
///
/// Condition variables represent the ability to block a fiber such that it consumes no CPU time while waiting for an event to occur. Condition variables are typically associated with a boolean predicate (a condition) and a mutex. The predicate is always verified inside of the mutex before determining that a fiber must block.
pub struct Condvar {
    has_queue: AtomicBool,
}

impl Condvar {
    #[cold]
    fn notify_one_slow(&self) {
        unpark_one(self as *const _ as usize, |have_more| {
            if !have_more {
                self.has_queue.store(false, Ordering::Relaxed);
            }
        });
    }

    #[cold]
    fn notify_all_slow(&self) {
        self.has_queue.store(false, Ordering::Relaxed);
        unpark_all(self as *const _ as usize);
    }

    fn wait_slow(&self, mutex: &RawMutex) {
        park(
            self as *const _ as usize,
            || true,
            || {
                if !self.has_queue.load(Ordering::Relaxed) {
                    self.has_queue.store(true, Ordering::Relaxed);
                }
                mutex.unlock();
            },
        );
        mutex.lock();
    }
}

impl Condvar {
    /// Creates a new condition variable.
    #[inline]
    pub const fn new() -> Self {
        Condvar { has_queue: AtomicBool::new(false) }
    }

    /// Wakes up one blocked fiber on this condvar.
    #[inline]
    pub fn notify_one(&self) {
        if self.has_queue.load(Ordering::Relaxed) {
            self.notify_one_slow();
        }
    }

    /// Wakes up all blocked fibers on this condvar.
    #[inline]
    pub fn notify_all(&self) {
        if self.has_queue.load(Ordering::Relaxed) {
            self.notify_all_slow();
        }
    }

    /// Blocks the current fiber until this condition variable receives a notification.
    ///
    /// This function will atomically unlock the mutex guard and block the current fiber. Upon return, the lock will be re-acquired.
    #[inline]
    pub fn wait<T>(&self, guard: &mut MutexGuard<'_, T>) {
        self.wait_slow(unsafe { MutexGuard::mutex(&guard).raw() });
    }
}
