/// IoMemory represents a region of physically continuous I/O memory.
///
/// We currently expect only one guest core can access a region of I/O memory at a time. Usually as
/// I/O access are rare and considered expensive, having a global lock for all I/O access, or have
/// a lock for each region of I/O memory should be sufficient. It somehow it ended up being a
/// bottleneck then we can come back to optimise it.
pub trait IoMemory {
    /// Read from I/O memory. `size` can be either 1, 2, 4 or 8. `addr` must be aligned properly,
    /// e.g. when `size` is 4, the least significant 2 bits of `addr` should be zero.
    ///
    /// Note that we use `&mut self`, as read operation is not always nullpotent in I/O memory.
    ///
    /// `addr` has type `usize`, which depends on the architecture. However, we believe that there
    /// is never a use-case for continuous memory region with more than 4GiB size on a 32-bit
    /// machine on 32-bit machine. Use `usize` over `u64` makes it much easier to handle indexes.
    fn read(&mut self, addr: usize, size: u32) -> u64;

    /// Write to I/O memory. `size` can be either 1, 2, 4 or 8. `addr` must be aligned properly,
    /// e.g. when `size` is 4, the least significant 2 bits of `addr` should be zero.
    fn write(&mut self, addr: usize, value: u64, size: u32);

    /// Read a 64-bit value from the memory.
    fn read_u64(&mut self, addr: usize) -> u64 {
        self.read(addr, 8)
    }

    /// Read a 32-bit value from the memory.
    fn read_u32(&mut self, addr: usize) -> u32 {
        self.read(addr, 4) as u32
    }

    /// Read a 16-bit value from the memory.
    fn read_u16(&mut self, addr: usize) -> u16 {
        self.read(addr, 2) as u16
    }

    /// Read a 8-bit value from the memory.
    fn read_u8(&mut self, addr: usize) -> u8 {
        self.read(addr, 1) as u8
    }

    /// Write a 64-bit value to the memory.
    fn write_u64(&mut self, addr: usize, value: u64) {
        self.write(addr, value, 8)
    }

    /// Write a 32-bit value to the memory.
    fn write_u32(&mut self, addr: usize, value: u32) {
        self.write(addr, value as u64, 4)
    }

    /// Write a 16-bit value to the memory.
    fn write_u16(&mut self, addr: usize, value: u16) {
        self.write(addr, value as u64, 2)
    }

    /// Write a 8-bit value to the memory.
    fn write_u8(&mut self, addr: usize, value: u8) {
        self.write(addr, value as u64, 1)
    }
}
