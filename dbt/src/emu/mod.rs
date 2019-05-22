use crate::io::IoMemory;
use crate::io::plic::Plic;
use crate::io::virtio::{Mmio, Block, Rng};

// The global PLIC
pub static mut PLIC: Option<Plic> = None;

static mut VIRTIO_BLK: Option<Mmio> = None;
static mut VIRTIO_RNG: Option<Mmio> = None;

pub fn init() {
    unsafe {
        PLIC = Some(Plic::new(2));
        VIRTIO_BLK = Some(Mmio::new(Box::new(Block::new("rootfs.img"))));
        VIRTIO_RNG = Some(Mmio::new(Box::new(Rng::new_os())));
    }
}

pub unsafe fn read_memory<T: Copy>(addr: u64) -> T {
    let ptr = addr as usize as *const T;
    crate::util::safe_memory::probe_read(ptr).unwrap();
    *ptr
}

pub unsafe fn write_memory<T: Copy>(addr: u64, value: T) {
    let ptr = addr as usize as *mut T;
    crate::util::safe_memory::probe_write(ptr).unwrap();
    *ptr = value;
}

#[no_mangle]
pub unsafe extern "C" fn mmio_read(addr: usize) -> u32 {
    if addr >= 0x100000 {
        PLIC.as_mut().unwrap().read_u32(addr - 0x100000)
    } else {
        if addr >= 4096 {
            VIRTIO_RNG.as_mut().unwrap().read_u32(addr - 4096)
        } else {
            VIRTIO_BLK.as_mut().unwrap().read_u32(addr)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn mmio_write(addr: usize, value: u32) {
    if addr >= 0x100000 {
        PLIC.as_mut().unwrap().write_u32(addr - 0x100000, value)
    } else {
        if addr >= 4096 {
            VIRTIO_RNG.as_mut().unwrap().write_u32(addr - 4096, value)
        } else {
            VIRTIO_BLK.as_mut().unwrap().write_u32(addr, value)
        }
    }
}
