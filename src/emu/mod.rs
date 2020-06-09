use std::collections::BTreeMap;

use futures::future::BoxFuture;
use io::hw::intc::{Clint, Plic};
use io::hw::rtc::ZyncMp;
use io::hw::virtio::{Block, Console, Mmio, Rng, P9};
use io::{IoMemory, IrqPin};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

pub mod interp;
#[rustfmt::skip]
mod abi;
pub mod dbt;
mod event;
pub mod loader;
pub mod signal;
pub mod syscall;
pub use event::EventLoop;
pub use syscall::syscall;

struct DirectIoContext;

impl io::DmaContext for DirectIoContext {
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
}

impl io::RuntimeContext for DirectIoContext {
    fn now(&self) -> Duration {
        Duration::from_micros(crate::event_loop().time())
    }

    fn create_timer(&self, time: Duration) -> BoxFuture<'static, ()> {
        Box::pin(crate::event_loop().on_time(time.as_micros() as u64))
    }

    fn spawn(
        &self,
        task: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'static + Send>>,
    ) {
        crate::event_loop().spawn(task);
    }

    fn spawn_blocking(
        &self,
        name: &str,
        task: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'static + Send>>,
    ) {
        if crate::get_flags().blocking_io {
            crate::event_loop().spawn(task);
        } else {
            std::thread::Builder::new()
                .name(name.to_owned())
                .spawn(move || futures::executor::block_on(task))
                .unwrap();
        }
    }
}

struct CoreIrq(usize, u64);

impl IrqPin for CoreIrq {
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
    map: BTreeMap<usize, (usize, Arc<dyn IoMemory>)>,

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
        let plic = Arc::new(Plic::new(
            (0..core_count).map(|i| -> Box<dyn IrqPin> { Box::new(CoreIrq(i, 512)) }).collect(),
        ));

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
            sys.register_io_mem(base, 0x10000, Arc::new(&*CLINT));
        }
        sys
    }

    pub fn register_io_mem(&mut self, base: usize, size: usize, mem: Arc<dyn IoMemory>) {
        if let Some((k, v)) = self.map.range(..(base + size)).next_back() {
            let last_end = *k + v.0;
            assert!(base >= last_end);
        }
        self.map.insert(base, (size, mem));
    }

    /// Add a virtio device
    pub fn add_virtio<T>(&mut self, f: impl FnOnce(Box<dyn IrqPin>) -> T)
    where
        T: io::hw::virtio::Device + 'static,
    {
        let irq = self.next_irq;
        self.next_irq += 1;

        let mem = self.boundary;
        self.boundary += 4096;

        let device = Box::new(f(self.plic.irq_pin(irq, true)));
        let virtio = Arc::new(Mutex::new(Mmio::new(Arc::new(DirectIoContext), device)));
        self.register_io_mem(mem, 4096, virtio);

        let core_count = crate::core_count();
        let node = self.fdt.add_node(format!("virtio@{:x}", mem));
        node.add_prop("reg", &[mem as u64, 0x1000][..]);
        node.add_prop("compatible", "virtio,mmio");
        node.add_prop("interrupts-extended", &[core_count as u32 + 1, irq][..]);
    }

    pub fn find_io_mem(&self, ptr: usize) -> Option<(usize, &'_ dyn IoMemory)> {
        if let Some((k, v)) = self.map.range(..=ptr).next_back() {
            let last_end = *k + v.0;
            if ptr >= last_end { None } else { Some((*k, &*v.1)) }
        } else {
            None
        }
    }
}

static IO_SYSTEM: Lazy<IoSystem> = Lazy::new(|| {
    let mut sys = IoSystem::new();
    init_virtio(&mut sys);
    if crate::CONFIG.rtc {
        init_rtc(&mut sys);
    }
    sys
});

pub static CLINT: Lazy<Clint> = Lazy::new(|| {
    let core_count = crate::core_count();
    Clint::new(
        Arc::new(DirectIoContext),
        (0..core_count)
            .map(|i| -> Box<dyn IrqPin> {
                Box::new(CoreIrq(i, if crate::get_flags().prv == 1 { 2 } else { 8 }))
            })
            .collect(),
        (0..core_count)
            .map(|i| -> Box<dyn IrqPin> {
                Box::new(CoreIrq(i, if crate::get_flags().prv == 1 { 32 } else { 128 }))
            })
            .collect(),
    )
});

pub static CONSOLE: Lazy<io::serial::Console> = Lazy::new(|| {
    let mut console = io::serial::Console::new().unwrap();
    let mut escape_hit = false;
    console.set_processor(move |x| {
        if !escape_hit {
            if x == 1 {
                // Ctrl + A hit, wait for another byte to arrive
                escape_hit = true;
                return None;
            }
            return Some(x);
        }

        // Byte after Ctrl + A hit, do corresponding action
        match x {
            b't' => {
                let model_id = if crate::get_flags().model_id == 0 { 1 } else { 0 };
                crate::shutdown(crate::ExitReason::SwitchModel(model_id));
            }
            b'x' => {
                println!("Terminated");
                crate::shutdown(crate::ExitReason::Exit(0));
            }
            b'p' => {
                crate::shutdown(crate::ExitReason::PrintStats);
            }
            b'c' => unsafe {
                libc::raise(libc::SIGTRAP);
            },
            // Hit Ctrl + A twice, send Ctrl + A to guest
            1 => return Some(x),
            // Ignore all other characters
            _ => (),
        }
        None
    });
    console
});

#[cfg(feature = "usernet")]
fn init_network(sys: &mut IoSystem) {
    use io::hw::virtio::Network;
    use io::network::Usernet;

    for config in crate::CONFIG.network.iter() {
        let mac = eui48::MacAddress::parse_str(&config.config.mac).expect("unexpected mac address");
        let usernet = Usernet::new(Arc::new(DirectIoContext));
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
                sys.add_virtio(|irq| Network::new(Arc::new(DirectIoContext), irq, usernet, mac));
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

                use io::hw::network::XemacLite;
                let xemaclite = XemacLite::new(
                    Arc::new(DirectIoContext),
                    sys.plic.irq_pin(irq, true),
                    Box::new(usernet),
                );
                sys.register_io_mem(base, 0x2000, Arc::new(xemaclite));
                let core_count = crate::core_count();
                sys.fdt.child.push(XemacLite::build_dt(
                    (base as u64, 0x2000),
                    (core_count as u32 + 1, irq),
                    mac.to_array(),
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
        let file = io::block::File::new(file).unwrap();
        let file: Box<dyn io::block::Block + Send> =
            if config.shadow { Box::new(io::block::Shadow::new(file)) } else { Box::new(file) };
        sys.add_virtio(|irq| Block::new(Arc::new(DirectIoContext), irq, file));
    }

    for config in crate::CONFIG.random.iter() {
        sys.add_virtio(|irq| {
            use io::entropy::rand::SeedableRng;
            use io::entropy::{Entropy, Os, Seeded};
            let source: Box<dyn Entropy + Send + 'static> = match config.r#type {
                crate::config::RandomType::Pseudo => Box::new(Seeded::seed_from_u64(config.seed)),
                crate::config::RandomType::OS => Box::new(Os),
            };
            Rng::new(Arc::new(DirectIoContext), irq, source)
        });
    }

    for config in crate::CONFIG.share.iter() {
        use io::fs::Passthrough;
        sys.add_virtio(|irq| {
            P9::new(
                Arc::new(DirectIoContext),
                irq,
                &config.tag,
                Passthrough::new(&config.path).unwrap(),
            )
        });
    }

    init_network(sys);

    if crate::CONFIG.console.virtio {
        sys.add_virtio(|irq| {
            Console::new(
                Arc::new(DirectIoContext),
                irq,
                Box::new(&*CONSOLE),
                crate::CONFIG.console.resize,
            )
        });
    }
}

fn init_rtc(sys: &mut IoSystem) {
    let irq = sys.next_irq;
    sys.next_irq += 2;

    let mem = sys.boundary;
    sys.boundary += 4096;

    let rtc = Arc::new(ZyncMp::new(sys.plic.irq_pin(irq, true), sys.plic.irq_pin(irq + 1, true)));
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

/// Only memory addresses strictly below this location is accessible by the guest. For user-space
/// application, we consider all memory locations as RAM, so the default value here is usize::MAX.
static MEM_BOUNDARY: crate::util::RoCell<usize> = crate::util::RoCell::new(usize::MAX);

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

        crate::util::RoCell::replace(&MEM_BOUNDARY, phys_limit);

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
    Lazy::force(&IO_SYSTEM);
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

pub fn io_read(addr: usize, size: u32) -> u64 {
    assert!(addr < *IO_BOUNDARY, "{:x} access out-of-bound", addr);
    match IO_SYSTEM.find_io_mem(addr) {
        Some((base, v)) => v.read(addr - base, size),
        None => {
            error!("out-of-bound I/O memory read 0x{:x}", addr);
            0
        }
    }
}

pub fn io_write(addr: usize, value: u64, size: u32) {
    assert!(addr < *IO_BOUNDARY, "{:x} access out-of-bound", addr);
    match IO_SYSTEM.find_io_mem(addr) {
        Some((base, v)) => v.write(addr - base, value, size),
        None => {
            error!("out-of-bound I/O memory write 0x{:x} = 0x{:x}", addr, value);
        }
    }
}
