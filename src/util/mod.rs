mod ro_cell;
pub mod spin;
pub use ro_cell::RoCell;

macro_rules! offset_of {
    ($ty:ty, $($field:ident).*) => {
        unsafe { &(*(std::ptr::null() as *const $ty)) $(.$field)* as *const _ as usize }
    }
}

pub fn cpu_time() -> std::time::Duration {
    unsafe {
        let mut timespec = std::mem::MaybeUninit::uninit();
        let ret = libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, timespec.as_mut_ptr());
        assert_eq!(ret, 0);
        let timespec = timespec.assume_init();
        std::time::Duration::new(timespec.tv_sec as u64, timespec.tv_nsec as u32)
    }
}
