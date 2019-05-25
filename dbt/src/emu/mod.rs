use crate::io::IoMemory;
use crate::io::plic::Plic;
use crate::io::virtio::{Mmio, Block, Rng};

mod abi;
mod syscall;
pub mod safe_memory;
pub use syscall::syscall;

// The global PLIC
pub static mut PLIC: Option<Plic> = None;

static mut VIRTIO_BLK: Option<Mmio> = None;
static mut VIRTIO_RNG: Option<Mmio> = None;

pub fn init() {
    let file = std::fs::OpenOptions::new()
                                    .read(true)
                                    .open("rootfs.img")
                                    .unwrap();
    let file = crate::io::block::Shadow::new(file);
    unsafe {
        PLIC = Some(Plic::new(2));
        VIRTIO_BLK = Some(Mmio::new(Box::new(Block::new(Box::new(file)))));
        
        VIRTIO_RNG = Some(Mmio::new(Box::new(Rng::new_seeded())));
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

pub unsafe fn read_memory<T: Copy>(addr: u64) -> T {
    let ptr = addr as usize as *const T;
    safe_memory::probe_read(ptr).unwrap();
    *ptr
}

pub unsafe fn write_memory<T: Copy>(addr: u64, value: T) {
    let ptr = addr as usize as *mut T;
    safe_memory::probe_write(ptr).unwrap();
    *ptr = value;
}

#[no_mangle]
pub extern "C" fn phys_read(addr: u64, size: u32) -> u64 {
    if addr >= 0x80000000 {
        let addr = (addr - 0x80000000) as usize;
        unsafe {if addr >= 0x100000 {
            PLIC.as_mut().unwrap().read(addr - 0x100000, size)
        } else {
            if addr >= 4096 {
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

#[no_mangle]
pub extern "C" fn phys_write(addr: u64, value: u64, size: u32) {
    if addr >= 0x80000000 {
        let addr = (addr - 0x80000000) as usize;
        unsafe {if addr >= 0x100000 {
            PLIC.as_mut().unwrap().write(addr - 0x100000, value, size)
        } else {
            if addr >= 4096 {
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
