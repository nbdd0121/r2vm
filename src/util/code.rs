pub struct Code(&'static mut [u8]);

impl Drop for Code {
    fn drop(&mut self) {
        let size = (self.0.len() + 4095) &! 4095;
        unsafe { libc::munmap(self.0.as_mut_ptr() as _, size); }
    }
}

impl Code {
    pub fn new(bytes: &[u8]) -> Code {
        let slice = unsafe {
            let size = (bytes.len() + 4095) &! 4095;
            let addr = libc::mmap(
                std::ptr::null_mut(), size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0
            );
            assert_ne!(addr, libc::MAP_FAILED);
            std::slice::from_raw_parts_mut(addr as *mut u8, bytes.len())
        };
        slice.copy_from_slice(bytes);
        Code(slice)
    }

    pub fn as_slice(&self) -> &[u8] {
        self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.0
    }

    /// T must be a fn() type
    pub unsafe fn as_func_ptr<T: Copy>(&self) -> T {
        let p = self.0.as_ptr();
        *(&p as *const *const u8 as *const T)
    }
}
