use crate::io::IoMemorySync;
use crate::io::plic::Plic;
use crate::io::virtio::{Mmio, Block, Rng, P9};
use parking_lot::Mutex;
use lazy_static::lazy_static;

#[cfg(feature = "slirp")]
use crate::io::virtio::Network;
#[cfg(feature = "slirp")]
use crate::io::network::Slirp;

pub mod interp;
mod abi;
mod event;
pub mod syscall;
mod dbt;
pub mod signal;
pub mod loader;
pub use event::EventLoop;
pub use syscall::syscall;

lazy_static! {
    /// The global PLIC
    pub static ref PLIC: Mutex<Plic> = {
        assert!(!crate::get_flags().user_only);
        Mutex::new(Plic::new(crate::core_count()))
    };
}

#[cfg(feature = "slirp")]
fn init_network(vec: &mut Vec<Mutex<Mmio>>) {
    for config in crate::CONFIG.network.iter() {
        let mac = eui48::MacAddress::parse_str(&config.mac).expect("unexpected mac address").to_array();
        let irq = (vec.len() + 1) as u32;
        vec.push(Mutex::new(Mmio::new(Box::new(Network::new(irq, Slirp::new(), mac)))));
    }
}

#[cfg(not(feature = "slirp"))]
fn init_network(_vec: &mut Vec<Mutex<Mmio>>) {}

lazy_static! {
    pub static ref VIRTIO: Vec<Mutex<Mmio>> = {
        assert!(!crate::get_flags().user_only);
        let mut vec = Vec::new();

        for config in crate::CONFIG.drive.iter() {
            let file = std::fs::OpenOptions::new()
                                            .read(true)
                                            .write(!config.shadow)
                                            .open(&config.path)
                                            .unwrap();
            let file: Box<dyn crate::io::block::Block + Send> = if config.shadow {
                Box::new(crate::io::block::Shadow::new(file))
            } else {
                Box::new(file)
            };
            let dev = Box::new(Block::new((vec.len() + 1) as u32, file));
            vec.push(Mutex::new(Mmio::new(dev)));
        }

        for config in crate::CONFIG.random.iter() {
            let rng = match config.r#type {
                crate::config::RandomType::Pseudo => Rng::new_seeded((vec.len() + 1) as u32, config.seed),
                crate::config::RandomType::OS => Rng::new_os((vec.len() + 1) as u32),
            };
            vec.push(Mutex::new(Mmio::new(Box::new(rng))));
        }

        for config in crate::CONFIG.share.iter() {
            let dev = Box::new(P9::new((vec.len() + 1) as u32, &config.tag, &config.path));
            vec.push(Mutex::new(Mmio::new(dev)));
        }

        init_network(&mut vec);

        vec
    };
}

/// This governs the boundary between RAM and I/O memory. If an address is strictly below this
/// location, then it is considered I/O. For user-space applications, we consider all memory
/// locations as RAM, so the default value here is 0.
static IO_BOUNDARY: crate::util::RoCell<usize> = crate::util::RoCell::new(0);

pub fn init() {
    unsafe {
        // The memory map looks like this:
        // 0 MiB - 2 MiB (reserved for null)
        // 2 MiB - 6 MiB PLIC
        // 6 MiB -       VIRTIO
        // 1 GiB -       main memory
        crate::util::RoCell::replace(&IO_BOUNDARY, 0x40000000);

        let phys_size = crate::CONFIG.memory * 1024 * 1024;
        let phys_limit = 0x40000000 + phys_size;

        // First allocate physical memory region, without making them accessible
        let result = libc::mmap(
            0x200000 as _, (phys_limit - 0x200000) as _,
            libc::PROT_NONE,
            libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_FIXED,
            -1, 0
        );
        if result == libc::MAP_FAILED {
            panic!("mmap failed while initing");
        }

        // Allocate wanted memory
        let result = libc::mprotect(
            0x40000000 as _, phys_size as _,
            libc::PROT_READ | libc::PROT_WRITE
        );
        if result != 0 {
            panic!("mmap failed while initing");
        }
    }
    lazy_static::initialize(&PLIC);
    lazy_static::initialize(&VIRTIO);
}

pub fn device_tree() -> fdt::Node {
    let mut root = fdt::Node::new("");
    root.add_prop("model", "riscv-virtio,qemu");
    root.add_prop("compatible", "riscv-virtio");
    root.add_prop("#address-cells", 2u32);
    root.add_prop("#size-cells", 2u32);

    let chosen = root.add_node("chosen");
    chosen.add_prop("bootargs", crate::CONFIG.cmdline.as_str());

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
    plic.add_prop("reg", &[0x200000u64, 0x400000][..]);
    let mut vec: Vec<u32> = Vec::with_capacity(8);
    for i in 0..core_count {
        vec.push(i + 1);
        vec.push(9);
    }
    plic.add_prop("interrupts-extended", vec.as_slice());
    plic.add_prop("phandle", core_count + 1);

    for i in 0..VIRTIO.len() {
        let addr: u64 = 0x600000 + (i as u64) * 0x1000;
        let virtio = soc.add_node(format!("virtio@{:x}", addr));
        virtio.add_prop("reg", &[addr, 0x1000][..]);
        virtio.add_prop("compatible", "virtio,mmio");
        virtio.add_prop("interrupts-extended", &[core_count + 1, (i+1) as u32][..]);
    }

    let memory = root.add_node("memory@0x40000000");
    memory.add_prop("reg", &[0x40000000, (crate::CONFIG.memory * 1024 * 1024) as u64][..]);
    memory.add_prop("device_type", "memory");

    root
}

// TODO: Remove these 2 functions
pub fn read_memory<T: Copy>(addr: usize) -> T {
    assert!(addr >= *IO_BOUNDARY);
    unsafe { std::ptr::read(addr as *const T) }
}

pub fn write_memory<T: Copy>(addr: usize, value: T) {
    assert!(addr >= *IO_BOUNDARY);
    unsafe { std::ptr::write_volatile(addr as *mut T, value) }
}

pub fn phys_read(addr: usize, size: u32) -> u64 {
    if addr < *IO_BOUNDARY {
        if addr < 0x600000 {
            PLIC.read_sync(addr - 0x200000, size)
        } else {
            let idx = (addr - 0x600000) >> 12;
            if idx > VIRTIO.len() {
                error!("out-of-bound I/O memory read 0x{:x}", addr);
                return 0;
            }
            VIRTIO[idx].read_sync(addr & 4095, size)
        }
    } else {
        match size {
            1 => unsafe { std::ptr::read_volatile(addr as *const u8) as u64 },
            2 => unsafe { std::ptr::read_volatile(addr as *const u16) as u64 },
            4 => unsafe { std::ptr::read_volatile(addr as *const u32) as u64 },
            8 => unsafe { std::ptr::read_volatile(addr as *const u64) },
            _ => unreachable!(),
        }
    }
}

pub fn phys_write(addr: usize, value: u64, size: u32) {
    if addr < *IO_BOUNDARY {
        if addr < 0x600000 {
            PLIC.write_sync(addr - 0x200000, value, size)
        } else {
            let idx = (addr - 0x600000) >> 12;
            if idx > VIRTIO.len() {
                error!("out-of-bound I/O memory write 0x{:x} = 0x{:x}", addr, value);
                return;
            }
            VIRTIO[idx].write_sync(addr & 4095, value, size)
        }
    } else {
        match size {
            1 => unsafe { std::ptr::write_volatile(addr as *mut _, value as u8) },
            2 => unsafe { std::ptr::write_volatile(addr as *mut _, value as u16) },
            4 => unsafe { std::ptr::write_volatile(addr as *mut _, value as u32) },
            8 => unsafe { std::ptr::write_volatile(addr as *mut _, value) },
            _ => unreachable!(),
        }
    }
}
