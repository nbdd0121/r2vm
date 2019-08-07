use crate::io::IoMemorySync;
use crate::io::plic::Plic;
use crate::io::virtio::{Mmio, Block, Rng, P9};
use spin::Mutex;

pub mod interp;
mod abi;
mod event;
mod syscall;
mod dbt;
pub mod signal;
pub mod loader;
pub use event::EventLoop;
pub use syscall::syscall;

// Type definition of guest sizes
#[allow(non_camel_case_types)]
pub type ureg = u64;
#[allow(non_camel_case_types)]
pub type sreg = i64;

lazy_static! {
    /// The global PLIC
    pub static ref PLIC: Mutex<Plic> = {
        assert!(!crate::get_flags().user_only);
        Mutex::new(Plic::new(crate::core_count()))
    };
}

lazy_static! {
    pub static ref VIRTIO_BLK: Mutex<Mmio> = {
        assert!(!crate::get_flags().user_only);
        let file = std::fs::OpenOptions::new()
                                        .read(true)
                                        .open("rootfs.img")
                                        .unwrap();
        let file = crate::io::block::Shadow::new(file);
        Mutex::new(Mmio::new(Box::new(Block::new(Box::new(file)))))
    };

    pub static ref VIRTIO_RNG: Mutex<Mmio> = {
        assert!(!crate::get_flags().user_only);
        Mutex::new(Mmio::new(Box::new(Rng::new_seeded())))
    };

    pub static ref VIRTIO_9P: Mutex<Mmio> = {
        assert!(!crate::get_flags().user_only);
        Mutex::new(Mmio::new(Box::new(P9::new("share", std::path::Path::new("./share")))))
    };
}

/// This governs the boundary between RAM and I/O memory. If an address is strictly above this
/// location, then it is considered I/O. For user-space applications, we consider all memory
/// locations as RAM, so the default value here is `ureg::max_value()`.
static mut IO_BOUNDARY: ureg = ureg::max_value();

pub fn init() {
    unsafe {
        IO_BOUNDARY = 0x7fffffff;

        let result = libc::mmap(
            0x80000000 as *mut libc::c_void, 0x80000000,
            libc::PROT_NONE,
            libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_FIXED,
            -1, 0
        );
        assert_ne!(result, libc::MAP_FAILED);
    }
    lazy_static::initialize(&PLIC);
    lazy_static::initialize(&VIRTIO_BLK);
    lazy_static::initialize(&VIRTIO_RNG);
    lazy_static::initialize(&VIRTIO_9P);
}

pub fn device_tree() -> fdt::Node {
    let mut root = fdt::Node::new("");
    root.add_prop("model", "riscv-virtio,qemu");
    root.add_prop("compatible", "riscv-virtio");
    root.add_prop("#address-cells", 2u32);
    root.add_prop("#size-cells", 2u32);

    let chosen = root.add_node("chosen");
    chosen.add_prop("bootargs", match crate::get_flags().init {
        None => std::borrow::Cow::Borrowed("console=hvc0 root=/dev/vda rw"),
        Some(ref v) => std::borrow::Cow::Owned(format!("console=hvc0 root=/dev/vda rw init={}", v)),
    }.as_ref());

    let cpus = root.add_node("cpus");
    cpus.add_prop("timebase-frequency", 1000000u32);
    cpus.add_prop("#address-cells", 1u32);
    cpus.add_prop("#size-cells", 0u32);

    let core_count = crate::core_count() as u32;

    for i in 0..core_count {
        let cpu = cpus.add_node(format!("cpu@{:x}", i));
        cpu.add_prop("clock-frequency", 0u32);
        cpu.add_prop("mmu-type", "riscv,sv39");
        cpu.add_prop("riscv,isa", "rv64imafdc");
        cpu.add_prop("compatible", "riscv");
        cpu.add_prop("status", "okay");
        cpu.add_prop("reg", i);
        cpu.add_prop("device_type", "cpu");

        let intc = cpu.add_node("interrupt-controller");
        intc.add_prop("#interrupt-cells", 1u32);
        intc.add_prop("interrupt-controller", ());
        intc.add_prop("compatible", "riscv,cpu-intc");
        intc.add_prop("phandle", i + 1);
    }

    let soc = root.add_node("soc");
    soc.add_prop("ranges", ());
    soc.add_prop("compatible", "simple-bus");
    soc.add_prop("#address-cells", 2u32);
    soc.add_prop("#size-cells", 2u32);

    let plic = soc.add_node("plic@80100000");
    plic.add_prop("#interrupt-cells", 1u32);
    plic.add_prop("interrupt-controller", ());
    plic.add_prop("compatible", "sifive,plic-1.0.0");
    plic.add_prop("riscv,ndev", 31u32);
    plic.add_prop("reg", &[0x80100000u64, 0x400000][..]);
    let mut vec: Vec<u32> = Vec::with_capacity(8);
    for i in 0..core_count {
        vec.push(i + 1);
        vec.push(9);
    }
    plic.add_prop("interrupts-extended", vec.as_slice());
    plic.add_prop("phandle", core_count + 1);

    for i in 0..3 {
        let addr: u64 = 0x80000000 + (i as u64) * 0x1000;
        let virtio = soc.add_node(format!("virtio@{:x}", addr));
        virtio.add_prop("reg", &[addr, 0x1000][..]);
        virtio.add_prop("compatible", "virtio,mmio");
        virtio.add_prop("interrupts-extended", &[core_count + 1, i+1][..]);
    }

    let memory = root.add_node("memory@200000");
    memory.add_prop("reg", &[0x200000, 0x3FE00000u64][..]);
    memory.add_prop("device_type", "memory");

    root
}

// TODO: Remove these 4 functions

unsafe fn read_memory_unsafe<T: Copy>(addr: u64) -> T {
    let ptr = addr as usize as *const T;
    std::ptr::read_volatile(ptr)
}

unsafe fn write_memory_unsafe<T: Copy>(addr: u64, value: T) {
    let ptr = addr as usize as *mut T;
    std::ptr::write_volatile(ptr, value)
}

pub fn read_memory<T: Copy>(addr: u64) -> T {
    assert!(addr <= unsafe { IO_BOUNDARY });
    unsafe { read_memory_unsafe(addr) }
}

pub fn write_memory<T: Copy>(addr: u64, value: T) {
    assert!(addr <= unsafe { IO_BOUNDARY });
    unsafe { write_memory_unsafe(addr, value) }
}

pub fn phys_read(addr: ureg, size: u32) -> ureg {
    if addr > unsafe { IO_BOUNDARY } {
        let addr = (addr - 0x80000000) as usize;
        if addr >= 0x100000 {
            PLIC.read_sync(addr - 0x100000, size)
        } else {
            if addr >= 0x2000 {
                VIRTIO_9P.read_sync(addr - 0x2000, size)
            } else if addr >= 0x1000 {
                VIRTIO_RNG.read_sync(addr - 4096, size)
            } else {
                VIRTIO_BLK.read_sync(addr, size)
            }
        }
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
    if addr > unsafe { IO_BOUNDARY } {
        let addr = (addr - 0x80000000) as usize;
        if addr >= 0x100000 {
            PLIC.write_sync(addr - 0x100000, value, size)
        } else {
            if addr >= 0x2000 {
                VIRTIO_9P.write_sync(addr - 0x2000, value, size)
            } else if addr >= 0x1000 {
                VIRTIO_RNG.write_sync(addr - 4096, value, size)
            } else {
                VIRTIO_BLK.write_sync(addr, value, size)
            }
        }
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
