use crate::io::IoMemory;
use crate::io::plic::Plic;

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
