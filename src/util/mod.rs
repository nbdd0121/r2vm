pub fn cpu_time() -> std::time::Duration {
    unsafe {
        let mut timespec = std::mem::MaybeUninit::uninit();
        let ret = libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, timespec.as_mut_ptr());
        assert_eq!(ret, 0);
        let timespec = timespec.assume_init();
        std::time::Duration::new(timespec.tv_sec as u64, timespec.tv_nsec as u32)
    }
}

pub trait ILog2 {
    fn log2(self) -> usize;
    fn clog2(self) -> usize;

    /// Assert that the value is a power of 2, and get log2 of the current value.
    ///
    /// If the value is not a power of 2, calling this method will cause an abort.
    fn log2_assert(self) -> usize;
}

impl ILog2 for usize {
    fn log2(self) -> usize {
        std::mem::size_of::<usize>() * 8 - 1 - self.leading_zeros() as usize
    }

    fn clog2(self) -> usize {
        ILog2::log2(self - 1) + 1
    }

    fn log2_assert(self) -> usize {
        let log2 = ILog2::log2(self);
        assert_eq!(1 << log2, self);
        log2
    }
}

impl ILog2 for u64 {
    fn log2(self) -> usize {
        std::mem::size_of::<u64>() * 8 - 1 - self.leading_zeros() as usize
    }

    fn clog2(self) -> usize {
        ILog2::log2(self - 1) + 1
    }

    fn log2_assert(self) -> usize {
        let log2 = ILog2::log2(self);
        assert_eq!(1 << log2, self);
        log2
    }
}
