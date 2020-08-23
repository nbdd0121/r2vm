use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::ops::Deref;

/// A cell that is readonly.
/// It is expected to remain readonly for most time. Some use cases include set-once global
/// variables. Construction and mutation of RoCell are allowed in unsafe code, and the safety
/// must be ensured by the caller.
pub struct RoCell<T>(UnsafeCell<MaybeUninit<T>>);

unsafe impl<T: Sync> Sync for RoCell<T> {}

impl<T> Drop for RoCell<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe { std::mem::replace(&mut *(self.0.get()), MaybeUninit::uninit()).assume_init() };
    }
}

impl<T> RoCell<T> {
    /// Create a new RoCell that is initialised already.
    #[inline]
    pub const fn new(value: T) -> Self {
        RoCell(UnsafeCell::new(MaybeUninit::new(value)))
    }

    /// RoCell can be read in safe code. Therefore, we make its construction unsafe, therefore
    /// permitting uninit value. If `T` needs drop, the caller must ensure that RoCell is
    /// initialised or forgotten before it is dropped.
    #[inline]
    pub const unsafe fn new_uninit() -> Self {
        RoCell(UnsafeCell::new(MaybeUninit::uninit()))
    }

    /// Initialise a RoCell.
    ///
    /// No synchronisation is handled by RoCell.
    /// The caller must guarantee that no other threads are accessing this
    /// RoCell and other threads are properly synchronised after the call.
    #[inline]
    pub unsafe fn init(this: &Self, value: T) {
        std::ptr::write((*this.0.get()).as_mut_ptr(), value);
    }

    /// Replace a RoCell and return old content.
    ///
    /// No synchronisation is handled by RoCell.
    /// The caller must guarantee that no other threads are accessing this
    /// RoCell and other threads are properly synchronised after the call.
    #[inline]
    pub unsafe fn replace(this: &Self, value: T) -> T {
        std::mem::replace(RoCell::as_mut(this), value)
    }

    #[inline]
    pub unsafe fn as_mut(this: &Self) -> &mut T {
        &mut *(*this.0.get()).as_mut_ptr()
    }
}

impl<T> Deref for RoCell<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*(*self.0.get()).as_ptr() }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for RoCell<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.deref(), f)
    }
}
