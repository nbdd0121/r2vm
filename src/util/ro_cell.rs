use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

/// A cell that is readonly.
/// It is expected to remain readonly for most time. Some use cases include set-once global
/// variables. Construction and mutation of RoCell are allowed in unsafe code, and the safety
/// must be ensured by the caller.
pub struct RoCell<T>(UnsafeCell<MaybeUninit<T>>);

unsafe impl<T: Sync> Sync for RoCell<T> {}

impl<T> Drop for RoCell<T> {
    fn drop(&mut self) {
        unsafe { std::mem::replace(&mut *(self.0.get()), MaybeUninit::uninit()).assume_init() };
    }
}

impl<T> RoCell<T> {
    pub const fn new(value: T) -> Self {
        RoCell(UnsafeCell::new(MaybeUninit::new(value)))
    }

    /// RoCell can be read in safe code. Therefore, we make its construction unsafe, therefore
    /// permitting uninit value. If `T` needs drop, the caller must ensure that RoCell is
    /// initialised or forgotten before it is dropped.
    pub const unsafe fn new_uninit() -> Self {
        RoCell(UnsafeCell::new(MaybeUninit::uninit()))
    }

    pub unsafe fn init(this: &Self, value: T) {
        std::ptr::write((*this.0.get()).as_mut_ptr(), value);
    }

    pub unsafe fn replace(this: &Self, value: T) {
        std::mem::replace(&mut *(*this.0.get()).as_mut_ptr(), value);
    }
}

impl<T> std::ops::Deref for RoCell<T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*(*self.0.get()).as_ptr() }
    }
}
