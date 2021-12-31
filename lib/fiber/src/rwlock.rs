use lock_api::{RawMutex as LRawMutex, RawRwLock as LRawRwLock};
use std::cell::Cell;

/// Raw reader-writer lock type.
pub struct RawRwLock {
    // This is a rough implementation so it just works. We could refer to parking_lot's
    // implementation to fit everything inside usize.
    read: super::RawMutex,
    write: super::RawMutex,
    // Use a cell to reduce RwLock size.
    readers: Cell<usize>,
}

unsafe impl Sync for RawRwLock {}

unsafe impl LRawRwLock for RawRwLock {
    type GuardMarker = lock_api::GuardSend;

    const INIT: Self =
        Self { read: super::RawMutex::INIT, write: super::RawMutex::INIT, readers: Cell::new(0) };

    #[inline]
    fn lock_exclusive(&self) {
        self.write.lock()
    }

    #[inline]
    fn try_lock_exclusive(&self) -> bool {
        self.write.try_lock()
    }

    #[inline]
    unsafe fn unlock_exclusive(&self) {
        self.write.unlock()
    }

    #[inline]
    fn lock_shared(&self) {
        self.read.lock();
        let readers = self.readers.get();
        if readers == 0 {
            self.write.lock();
        }
        self.readers.set(readers + 1);
        unsafe { self.read.unlock() };
    }

    #[inline]
    fn try_lock_shared(&self) -> bool {
        if !self.read.try_lock() {
            return false;
        }
        self.read.lock();
        let readers = self.readers.get();
        if readers == 0 {
            if !self.write.try_lock() {
                return false;
            }
        }
        self.readers.set(readers + 1);
        unsafe { self.read.unlock() };
        true
    }

    #[inline]
    unsafe fn unlock_shared(&self) {
        self.read.lock();
        let readers = self.readers.get() - 1;
        self.readers.set(readers);
        if readers == 0 {
            self.write.unlock();
        }
        self.read.unlock();
    }

    #[inline]
    fn is_locked(&self) -> bool {
        self.write.is_locked()
    }
}

/// A mutual exclusion primitive useful for protecting shared data with fiber support.
pub type RwLock<T> = lock_api::RwLock<RawRwLock, T>;

/// RAII structure used to release the shared read access of a lock when
/// dropped.
pub type RwLockReadGuard<'a, T> = lock_api::RwLockReadGuard<'a, RawRwLock, T>;

/// RAII structure used to release the exclusive write access of a lock when
/// dropped.
pub type RwLockWriteGuard<'a, T> = lock_api::RwLockWriteGuard<'a, RawRwLock, T>;

#[test]
fn test_size() {
    assert_eq!(std::mem::size_of::<RawRwLock>(), 2 * std::mem::size_of::<usize>());
}
