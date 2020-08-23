use futures::future::BoxFuture;
use io::hw::intc::Clint;
use io::system::IoSystem;
use io::{IoMemory, IrqPin};
use once_cell::sync::Lazy;
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

static IO_SYSTEM: Lazy<IoSystem> = Lazy::new(|| {
    assert_ne!(crate::get_flags().prv, 0);

    let mut sys = IoSystem::new(
        Arc::new(DirectIoContext),
        Some(Arc::new(DirectIoContext)),
        crate::core_count(),
        |i| Box::new(CoreIrq(i, 512)),
    );

    if let Some(ref config) = crate::CONFIG.clint {
        let base = config.io_base.unwrap_or_else(|| sys.allocate_mem(0x10000));
        sys.register_mem(base, 0x10000, Arc::new(&*CLINT));
    }

    for config in crate::CONFIG.drive.iter() {
        sys.instantiate_drive(config);
    }

    for config in crate::CONFIG.random.iter() {
        sys.instantiate_random(config);
    }

    for config in crate::CONFIG.share.iter() {
        sys.instantiate_share(config);
    }

    #[cfg(feature = "usernet")]
    for config in crate::CONFIG.network.iter() {
        sys.instantiate_network(config);
    }

    if let Some(ref config) = crate::CONFIG.console {
        sys.instantiate_console(config, Box::new(&*CONSOLE));
    }

    if let Some(ref config) = crate::CONFIG.rtc {
        sys.instantiate_rtc(config);
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

        escape_hit = false;

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

    root.child.push(IO_SYSTEM.device_tree().clone());

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
    IO_SYSTEM.read(addr, size)
}

pub fn io_write(addr: usize, value: u64, size: u32) {
    assert!(addr < *IO_BOUNDARY, "{:x} access out-of-bound", addr);
    IO_SYSTEM.write(addr, value, size)
}
