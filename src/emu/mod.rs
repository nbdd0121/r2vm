use crate::io::IoMemory;
use crate::io::plic::Plic;
use crate::io::virtio::{Mmio, Block, Rng, P9};

mod abi;
mod event;
mod syscall;
pub mod signal;
pub mod loader;
pub use event::EventLoop;
pub use syscall::syscall;

// Type definition of guest sizes
#[allow(non_camel_case_types)]
pub type ureg = u64;
#[allow(non_camel_case_types)]
pub type sreg = i64;

// The global PLIC
pub static mut PLIC: Option<Plic> = None;

/// This governs the boundary between RAM and I/O memory. If an address is strictly above this
/// location, then it is considered I/O. For user-space applications, we consider all memory
/// locations as RAM, so the default value here is `ureg::max_value()`.
static mut IO_BOUNDARY: ureg = ureg::max_value();

static mut VIRTIO_BLK: Option<Mmio> = None;
static mut VIRTIO_RNG: Option<Mmio> = None;
static mut VIRTIO_P9: Option<Mmio> = None;

pub fn init() {
    let file = std::fs::OpenOptions::new()
                                    .read(true)
                                    .open("rootfs.img")
                                    .unwrap();
    let file = crate::io::block::Shadow::new(file);
    unsafe {
        PLIC = Some(Plic::new(4));
        VIRTIO_BLK = Some(Mmio::new(Box::new(Block::new(Box::new(file)))));
        
        VIRTIO_RNG = Some(Mmio::new(Box::new(Rng::new_seeded())));
        VIRTIO_P9 = Some(Mmio::new(Box::new(P9::new("/dev/root", std::path::Path::new("./shared")))));

        IO_BOUNDARY = 0x7fffffff;

        let result = libc::mmap(
            0x80000000 as *mut libc::c_void, 0x80000000,
            libc::PROT_NONE,
            libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_FIXED,
            -1, 0
        );
        assert_ne!(result, libc::MAP_FAILED);
    }
}

#[inline(always)]
pub unsafe fn read_memory_unsafe<T: Copy>(addr: u64) -> T {
    let ptr = addr as usize as *const T;
    std::ptr::read_volatile(ptr)
}

#[inline(always)]
pub unsafe fn write_memory_unsafe<T: Copy>(addr: u64, value: T) {
    let ptr = addr as usize as *mut T;
    std::ptr::write_volatile(ptr, value)
}

pub fn read_memory<T: Copy>(addr: u64) -> T {
    assert!(addr < unsafe { IO_BOUNDARY });
    unsafe { read_memory_unsafe(addr) }
}

pub fn write_memory<T: Copy>(addr: u64, value: T) {
    assert!(addr < unsafe { IO_BOUNDARY });
    unsafe { write_memory_unsafe(addr, value) }
}

pub fn phys_read(addr: ureg, size: u32) -> ureg {
    if addr >= unsafe { IO_BOUNDARY } {
        let addr = (addr - 0x80000000) as usize;
        unsafe {if addr >= 0x100000 {
            PLIC.as_mut().unwrap().read(addr - 0x100000, size)
        } else {
            if addr >= 0x2000 {
                VIRTIO_P9.as_mut().unwrap().read(addr - 0x2000, size)
            } else if addr >= 0x1000 {
                VIRTIO_RNG.as_mut().unwrap().read(addr - 4096, size)
            } else {
                VIRTIO_BLK.as_mut().unwrap().read(addr, size)
            }
        }}
    } else {
        match size {
            1 => unsafe { read_memory_unsafe::<u8>(addr) as u64 },
            2 => unsafe { read_memory_unsafe::<u16>(addr) as u64 },
            4 => unsafe { read_memory_unsafe::<u32>(addr) as u64 },
            8 => unsafe { read_memory_unsafe::<u64>(addr) },
            _ => unreachable!(),
        }
    }
}

pub fn phys_write(addr: ureg, value: ureg, size: u32) {
    if addr >= unsafe { IO_BOUNDARY } {
        let addr = (addr - 0x80000000) as usize;
        unsafe {if addr >= 0x100000 {
            PLIC.as_mut().unwrap().write(addr - 0x100000, value, size)
        } else {
            if addr >= 0x2000 {
                VIRTIO_P9.as_mut().unwrap().write(addr - 0x2000, value, size)
            } else if addr >= 0x1000 {
                VIRTIO_RNG.as_mut().unwrap().write(addr - 4096, value, size)
            } else {
                VIRTIO_BLK.as_mut().unwrap().write(addr, value, size)
            }
        }}
    } else {
        match size {
            1 => unsafe { write_memory_unsafe(addr, value as u8) },
            2 => unsafe { write_memory_unsafe(addr, value as u16) },
            4 => unsafe { write_memory_unsafe(addr, value as u32) },
            8 => unsafe { write_memory_unsafe(addr, value) },
            _ => unreachable!(),
        }
    }
}
