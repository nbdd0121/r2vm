mod split_int;
pub use split_int::SplitInt;

mod atomic_min_max;
pub use atomic_min_max::AtomicMinMax;

macro_rules! offset_of {
    ($ty:ty, $field:ident) => {
        unsafe { &(*(0 as *const $ty)).$field as *const _ as usize }
    }
}

pub fn cpu_time() -> std::time::Duration {
    unsafe {
        let mut timespec = std::mem::uninitialized();
        let ret = libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut timespec);
        assert_eq!(ret, 0);
        std::time::Duration::new(timespec.tv_sec as u64, timespec.tv_nsec as u32)
    }
}
