use crate::io::IoMemory;
use crate::io::plic::Plic;

// The global PLIC
pub static mut PLIC: Option<Plic> = None;

pub fn init() {
    unsafe {
        PLIC = Some(Plic::new(2));
}
