use std::collections::BTreeMap;

use crate::io::clint::Clint;
use crate::io::plic::{Plic, PlicIrq};
use crate::io::rtc::Rtc;
use crate::io::virtio::{Block, Console, Mmio, Rng, P9};
use crate::io::IoMemorySync;
use futures::future::BoxFuture;
use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "usernet")]
use crate::io::network::Usernet;
#[cfg(feature = "usernet")]
use crate::io::virtio::Network;

pub mod interp;
#[rustfmt::skip]
mod abi;
mod dbt;
mod event;
pub mod loader;
pub mod signal;
pub mod syscall;
pub use event::EventLoop;
pub use syscall::syscall;

struct DirectIoContext;

impl crate::io::IoContext for DirectIoContext {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) {
        unsafe {
            std::ptr::copy_nonoverlapping(addr as usize as *const u8, buf.as_mut_ptr(), buf.len())
        };
    }

    fn dma_write(&self, addr: u64, buf: &[u8]) {
        let addr = addr as usize;
        unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), addr as *mut u8, buf.len()) };
        crate::emu::interp::icache_invalidate(addr, addr + buf.len());
    }

    fn read_u16(&self, addr: u64) -> u16 {
        unsafe {
            (*(addr as *const std::sync::atomic::AtomicU16))
                .load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    fn write_u16(&self, addr: u64, value: u16) {
        unsafe {
            (*(addr as *const std::sync::atomic::AtomicU16))
                .store(value, std::sync::atomic::Ordering::SeqCst)
        }
    }

    fn time(&self) -> Duration {
        Duration::from_micros(crate::event_loop().time())
    }

    fn on_time(&self, time: Duration) -> BoxFuture<'static, ()> {
        Box::pin(crate::event_loop().on_time(time.as_micros() as u64))
    }

    fn spawn(
        &self,
        task: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'static + Send>>,
    ) {
        crate::event_loop().spawn(task);
    }
}

struct CoreIrq(usize, u64);

impl crate::io::IrqPin for CoreIrq {
    fn set_level(&self, level: bool) {
        if level {
            crate::shared_context(self.0).assert(self.1);
        } else {
            crate::shared_context(self.0).deassert(self.1);
        }
    }
}

/// This describes all I/O aspects of the system.
struct IoSystem {
    /// The IO memory map.
    map: BTreeMap<usize, (usize, Arc<dyn IoMemorySync>)>,

    /// The PLIC instance. It always exist.
    plic: Arc<Plic>,

    // Types below are useful only for initialisation
    next_irq: u32,
    boundary: usize,

    /// The "soc" node for
    fdt: fdt::Node,
}

impl IoSystem {
    pub fn new() -> IoSystem {
        assert_ne!(crate::get_flags().prv, 0);

        // Instantiate PLIC and corresponding device tre
        let core_count = crate::core_count();
        let plic = Plic::new(
            (0..core_count)
                .map(|i| -> Arc<dyn crate::io::IrqPin> { Arc::new(CoreIrq(i, 512)) })
                .collect(),
        );

        let mut soc = fdt::Node::new("soc");
        soc.add_prop("ranges", ());
        soc.add_prop("compatible", "simple-bus");
        soc.add_prop("#address-cells", 2u32);
        soc.add_prop("#size-cells", 2u32);

        let plic_node = soc.add_node("plic@200000");
        plic_node.add_prop("#interrupt-cells", 1u32);
        plic_node.add_prop("interrupt-controller", ());
        plic_node.add_prop("compatible", "sifive,plic-1.0.0");
        plic_node.add_prop("riscv,ndev", 31u32);
        plic_node.add_prop("reg", &[0x200000u64, 0x400000][..]);
        let mut vec: Vec<u32> = Vec::with_capacity(core_count * 2);
        for i in 0..(core_count as u32) {
            vec.push(i + 1);
            vec.push(9);
        }
        plic_node.add_prop("interrupts-extended", vec.as_slice());
        plic_node.add_prop("phandle", core_count as u32 + 1);

        let mut sys = IoSystem {
            map: BTreeMap::default(),
            plic: plic.clone(),
            next_irq: 1,
            boundary: 0x600000,
            fdt: soc,
        };

        sys.register_io_mem(0x200000, 0x400000, plic);

        if let Some(ref config) = crate::CONFIG.clint {
            let base = config.io_base.unwrap_or_else(|| {
                let mem = sys.boundary;
                sys.boundary += 0x10000;
                mem
            });
            sys.register_io_mem(base, 0x10000, CLINT.clone());
        }
        sys
    }

    pub fn register_io_mem(&mut self, base: usize, size: usize, mem: Arc<dyn IoMemorySync>) {
        if let Some((k, v)) = self.map.range(..(base + size)).next_back() {
            let last_end = *k + v.0;
            assert!(base >= last_end);
        }
        self.map.insert(base, (size, mem));
    }

    /// Add a virtio device
    pub fn add_virtio<T>(&mut self, f: impl FnOnce(PlicIrq) -> T)
    where
        T: crate::io::virtio::Device + Send + 'static,
    {
        let irq = self.next_irq;
        self.next_irq += 1;

        let mem = self.boundary;
        self.boundary += 4096;

        let device = Box::new(f(self.plic.clone().irq_pin(irq)));
        let virtio = Arc::new(Mutex::new(Mmio::new(Arc::new(DirectIoContext), device)));
        self.register_io_mem(mem, 4096, virtio.clone());

        let core_count = crate::core_count();
        let node = self.fdt.add_node(format!("virtio@{:x}", mem));
        node.add_prop("reg", &[mem as u64, 0x1000][..]);
        node.add_prop("compatible", "virtio,mmio");
        node.add_prop("interrupts-extended", &[core_count as u32 + 1, irq][..]);
    }

    pub fn find_io_mem<'a>(&'a self, ptr: usize) -> Option<(usize, &'a dyn IoMemorySync)> {
        if let Some((k, v)) = self.map.range(..=ptr).next_back() {
            let last_end = *k + v.0;
            if ptr >= last_end { None } else { Some((*k, &*v.1)) }
        } else {
            None
        }
    }
}

lazy_static! {
    static ref IO_SYSTEM: IoSystem = {
        let mut sys = IoSystem::new();
        init_virtio(&mut sys);
        if crate::CONFIG.rtc {
            init_rtc(&mut sys);
        }
        sys
    };

    /// The global PLIC
    pub static ref PLIC: &'static Arc<Plic> = {
        &IO_SYSTEM.plic
    };

    pub static ref CLINT: Arc<Clint> = {
        let core_count = crate::core_count();
        Clint::new(Arc::new(DirectIoContext),
            (0..core_count)
                .map(|i| -> Arc<dyn crate::io::IrqPin> { Arc::new(CoreIrq(i, if crate::get_flags().prv == 1 { 2 } else { 8 })) })
                .collect(),
            (0..core_count)
                .map(|i| -> Arc<dyn crate::io::IrqPin> { Arc::new(CoreIrq(i, if crate::get_flags().prv == 1 { 32 } else { 128 })) })
                .collect())
    };
}

#[cfg(feature = "usernet")]
fn init_network(sys: &mut IoSystem) {
    for config in crate::CONFIG.network.iter() {
        let mac = eui48::MacAddress::parse_str(&config.config.mac)
            .expect("unexpected mac address")
            .to_array();
        let usernet = Usernet::new();
        for fwd in config.config.forward.iter() {
            usernet
                .add_host_forward(
                    fwd.protocol == crate::config::ForwardProtocol::Udp,
                    fwd.host_addr,
                    fwd.host_port,
                    fwd.guest_port,
                )
                .expect("cannot establish port forwarding");
        }

        match config.config.r#type.as_str() {
            "virtio" => {
                sys.add_virtio(|irq| {
                    Network::new(Arc::new(DirectIoContext), Arc::new(irq), usernet, mac)
                });
            }
            "xemaclite" => {
                let irq = sys.next_irq;
                sys.next_irq += 1;

                let base = match config.io_base {
                    None => {
                        let mem = sys.boundary;
                        sys.boundary += 0x2000;
                        mem
                    }
                    Some(v) => v,
                };

                use crate::io::network::xemaclite::Xemaclite;
                let xemaclite = Xemaclite::new(
                    Arc::new(DirectIoContext),
                    Arc::new(sys.plic.clone().irq_pin(irq)),
                    Arc::new(usernet),
                );
                sys.register_io_mem(base, 0x2000, Arc::new(xemaclite));
                let core_count = crate::core_count();
                sys.fdt.child.push(Xemaclite::build_fdt(
                    (base as u64, 0x2000),
                    (core_count as u32 + 1, irq),
                ));
            }
            _ => panic!("unknown device type"),
        }
    }
}

#[cfg(not(feature = "usernet"))]
fn init_network(_sys: &mut IoSystem) {}

fn init_virtio(sys: &mut IoSystem) {
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
        sys.add_virtio(|irq| Block::new(Arc::new(irq), file));
    }

    for config in crate::CONFIG.random.iter() {
        sys.add_virtio(|irq| match config.r#type {
            crate::config::RandomType::Pseudo => Rng::new_seeded(Box::new(irq), config.seed),
            crate::config::RandomType::OS => Rng::new_os(Box::new(irq)),
        });
    }

    for config in crate::CONFIG.share.iter() {
        sys.add_virtio(|irq| P9::new(Arc::new(irq), &config.tag, &config.path));
    }

    init_network(sys);

    if crate::CONFIG.console.virtio {
        sys.add_virtio(|irq| Console::new(Arc::new(irq), crate::CONFIG.console.resize));
    }
}

fn init_rtc(sys: &mut IoSystem) {
    let irq = sys.next_irq;
    sys.next_irq += 2;

    let mem = sys.boundary;
    sys.boundary += 4096;

    let rtc = Arc::new(Rtc::new(irq, irq + 1));
    sys.register_io_mem(mem, 4096, rtc);

    let node = sys.fdt.add_node(format!("rtc@{:x}", mem));
    node.add_prop("compatible", "xlnx,zynqmp-rtc");
    node.add_prop("reg", &[mem as u64, 0x100][..]);
    let core_count = crate::core_count();
    node.add_prop("interrupt-parent", core_count as u32 + 1);
    node.add_prop("interrupts", &[irq, irq + 1][..]);
    node.add_prop("interrupt-names", &["alarm", "sec"][..]);
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

        // If firmware is present give it 2MiB of extra memory.
        let phys_size = (crate::CONFIG.memory
            + (if crate::CONFIG.firmware.is_some() { 2 } else { 0 }))
            * 1024
            * 1024;
        let phys_limit = 0x40000000 + phys_size;

        // First allocate physical memory region, without making them accessible
        let result = libc::mmap(
            0x200000 as _,
            (phys_limit - 0x200000) as _,
            libc::PROT_NONE,
            libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_FIXED,
            -1,
            0,
        );
        if result == libc::MAP_FAILED {
            panic!("mmap failed while initing");
        }

        // Allocate wanted memory
        let result =
            libc::mprotect(0x40000000 as _, phys_size as _, libc::PROT_READ | libc::PROT_WRITE);
        if result != 0 {
            panic!("mmap failed while initing");
        }
    }
    lazy_static::initialize(&PLIC);
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

    root.child.push(IO_SYSTEM.fdt.clone());

    let memory = root.add_node("memory@40000000");
    memory.add_prop("reg", &[0x40000000, (crate::CONFIG.memory * 1024 * 1024) as u64][..]);
    memory.add_prop("device_type", "memory");

    root
}

// TODO: Remove these 2 functions
pub fn read_memory<T: Copy>(addr: usize) -> T {
    assert!(addr >= *IO_BOUNDARY, "{:x} access out-of-bound", addr);
    unsafe { std::ptr::read(addr as *const T) }
}

pub fn write_memory<T: Copy>(addr: usize, value: T) {
    assert!(addr >= *IO_BOUNDARY, "{:x} access out-of-bound", addr);
    unsafe { std::ptr::write_volatile(addr as *mut T, value) }
}

pub fn io_read(addr: usize, size: u32) -> u64 {
    assert!(addr < *IO_BOUNDARY, "{:x} access out-of-bound", addr);
    match IO_SYSTEM.find_io_mem(addr) {
        Some((base, v)) => v.read_sync(addr - base, size),
        None => {
            error!("out-of-bound I/O memory read 0x{:x}", addr);
            0
        }
    }
}

pub fn io_write(addr: usize, value: u64, size: u32) {
    assert!(addr < *IO_BOUNDARY, "{:x} access out-of-bound", addr);
    match IO_SYSTEM.find_io_mem(addr) {
        Some((base, v)) => return v.write_sync(addr - base, value, size),
        None => {
            error!("out-of-bound I/O memory write 0x{:x} = 0x{:x}", addr, value);
        }
    }
}
