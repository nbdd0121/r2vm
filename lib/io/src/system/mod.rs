use crate::hw::intc::Plic;
use crate::{DmaContext, IoMemory, IrqPin, RuntimeContext};
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub mod config;
use config::*;

/// This describes all I/O aspects of the system.
pub struct IoSystem {
    /// Global contexts
    ctx: Arc<dyn RuntimeContext>,
    dma_ctx: Option<Arc<dyn DmaContext>>,

    /// The IO memory map.
    map: BTreeMap<usize, (usize, Arc<dyn IoMemory>)>,

    /// Allocated IRQs
    irq_set: BTreeSet<u32>,

    /// The PLIC instance. It always exist.
    plic: Arc<Plic>,
    plic_phandle: usize,

    // Types below are useful only for initialisation
    next_irq: u32,
    boundary: usize,

    /// The "soc" node for
    fdt: fdt::Node,
}

impl IoSystem {
    pub fn new(
        ctx: Arc<dyn RuntimeContext>,
        dma_ctx: Option<Arc<dyn DmaContext>>,
        core_count: usize,
        plic_base: Option<usize>,
        seip_irq: impl FnMut(usize) -> Box<dyn IrqPin>,
    ) -> IoSystem {
        // Instantiate PLIC and corresponding device tre
        let plic = Arc::new(Plic::new((0..core_count).map(seip_irq).collect()));

        let mut soc = fdt::Node::new("soc");
        soc.add_prop("ranges", ());
        soc.add_prop("compatible", "simple-bus");
        soc.add_prop("#address-cells", 2u32);
        soc.add_prop("#size-cells", 2u32);

        let plic_base = plic_base.unwrap_or(0x200000);
        let plic_node = soc.add_node(format!("plic@{:x}", plic_base));
        plic_node.add_prop("#interrupt-cells", 1u32);
        plic_node.add_prop("interrupt-controller", ());
        plic_node.add_prop("compatible", "sifive,plic-1.0.0");
        plic_node.add_prop("riscv,ndev", 31u32);
        plic_node.add_prop("reg", &[plic_base as u64, 0x400000][..]);
        let mut vec: Vec<u32> = Vec::with_capacity(core_count * 2);
        for i in 0..(core_count as u32) {
            vec.push(i + 1);
            vec.push(9);
        }
        plic_node.add_prop("interrupts-extended", vec.as_slice());
        plic_node.add_prop("phandle", core_count as u32 + 1);

        let mut sys = IoSystem {
            ctx,
            dma_ctx,
            map: BTreeMap::default(),
            irq_set: BTreeSet::default(),
            plic: plic.clone(),
            plic_phandle: core_count,
            next_irq: 1,
            boundary: 0x600000,
            fdt: soc,
        };

        // 0 is not a valid IRQ
        sys.irq_set.insert(0);

        sys.register_mem(plic_base, 0x400000, plic);
        sys
    }

    /// Allocate an unoccupied region of memory
    pub fn allocate_mem(&mut self, size: usize) -> usize {
        let base = self.boundary;
        if let Some((k, v)) = self.map.range(..(base + size)).next_back() {
            let last_end = *k + v.0;
            assert!(base >= last_end, "allocated memory region overlap");
        }
        self.boundary += size;
        base
    }

    /// Allocate an unoccupied IRQ
    pub fn allocate_irq(&mut self) -> u32 {
        let irq = self.next_irq;
        assert!(!self.irq_set.contains(&irq), "allocated irq overlap");
        self.next_irq += 1;
        irq
    }

    pub fn register_mem(&mut self, base: usize, size: usize, mem: Arc<dyn IoMemory>) {
        if let Some((k, v)) = self.map.range(..(base + size)).next_back() {
            let last_end = *k + v.0;
            assert!(base >= last_end);
        }
        self.map.insert(base, (size, mem));
    }

    pub fn register_irq(&mut self, irq: u32, edge_trigger: bool) -> Box<dyn IrqPin> {
        if !self.irq_set.insert(irq) {
            panic!("irq overlap");
        }
        self.plic.irq_pin(irq, edge_trigger)
    }

    pub fn find_by_mem(&self, ptr: usize) -> Option<(usize, &'_ dyn IoMemory)> {
        if let Some((k, v)) = self.map.range(..=ptr).next_back() {
            let last_end = *k + v.0;
            if ptr >= last_end { None } else { Some((*k, &*v.1)) }
        } else {
            None
        }
    }

    pub fn device_tree(&self) -> &fdt::Node {
        &self.fdt
    }
}

impl IoMemory for IoSystem {
    #[inline]
    fn read(&self, addr: usize, size: u32) -> u64 {
        match self.find_by_mem(addr) {
            Some((base, v)) => v.read(addr - base, size),
            None => {
                error!("out-of-bound I/O memory read 0x{:x}", addr);
                0
            }
        }
    }

    #[inline]
    fn write(&self, addr: usize, value: u64, size: u32) {
        match self.find_by_mem(addr) {
            Some((base, v)) => v.write(addr - base, value, size),
            None => {
                error!("out-of-bound I/O memory write 0x{:x} = 0x{:x}", addr, value);
            }
        }
    }
}

impl IoSystem {
    /// Add a virtio device
    #[cfg(feature = "virtio")]
    pub fn add_virtio<T>(
        &mut self,
        base: Option<usize>,
        irq: Option<u32>,
        f: impl FnOnce(&mut Self, Box<dyn IrqPin>) -> T,
    ) where
        T: crate::hw::virtio::Device + 'static,
    {
        let mem = base.unwrap_or_else(|| self.allocate_mem(4096));
        let irq = irq.unwrap_or_else(|| self.allocate_irq());

        let irq_pin = self.register_irq(irq, true);
        let device = Box::new(f(self, irq_pin));
        let virtio = Arc::new(Mutex::new(crate::hw::virtio::Mmio::new(
            self.dma_ctx
                .as_ref()
                .expect("Attempt to create DMA-capable device without DMA context")
                .clone(),
            device,
        )));
        self.register_mem(mem, 4096, virtio);

        let mut node = crate::hw::virtio::Mmio::build_dt(mem);
        node.add_prop("reg", &[mem as u64, 0x1000][..]);
        node.add_prop("interrupts-extended", &[self.plic_phandle as u32 + 1, irq][..]);
        self.fdt.child.push(node);
    }

    #[cfg(all(feature = "virtio-block", feature = "block-file", feature = "block-shadow"))]
    pub fn instantiate_drive(&mut self, config: &DeviceConfig<DriveConfig>) {
        use crate::hw::virtio::Block;

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(!config.config.shadow)
            .open(&config.config.path)
            .unwrap();
        let file = crate::block::File::new(file).unwrap();
        let file: Box<dyn crate::block::Block + Send + Sync> = if config.config.shadow {
            Box::new(crate::block::Shadow::new(file))
        } else {
            Box::new(file)
        };
        self.add_virtio(config.io_base, config.irq, |sys, irq| {
            Block::new(sys.ctx.clone(), irq, file)
        });
    }

    #[cfg(all(feature = "virtio-rng", feature = "entropy"))]
    pub fn instantiate_random(&mut self, config: &DeviceConfig<RandomConfig>) {
        use crate::entropy::rand::SeedableRng;
        use crate::entropy::{Entropy, Os, Seeded};
        use crate::hw::virtio::Rng;

        let source: Box<dyn Entropy + Send + 'static> = match config.config.r#type {
            RandomType::Pseudo => Box::new(Seeded::seed_from_u64(config.config.seed)),
            RandomType::OS => Box::new(Os),
        };
        self.add_virtio(config.io_base, config.irq, |sys, irq| {
            Rng::new(sys.ctx.clone(), irq, source)
        });
    }

    #[cfg(all(feature = "virtio-p9", feature = "fs"))]
    pub fn instantiate_share(&mut self, config: &DeviceConfig<ShareConfig>) {
        use crate::fs::Passthrough;
        use crate::hw::virtio::P9;

        self.add_virtio(config.io_base, config.irq, |sys, irq| {
            P9::new(
                sys.ctx.clone(),
                irq,
                &config.config.tag,
                Passthrough::new(&config.config.path).unwrap(),
            )
        });
    }

    #[cfg(feature = "network-usernet")]
    pub fn instantiate_network(&mut self, config: &DeviceConfig<NetworkConfig>) {
        use crate::network::Usernet;

        let mac = eui48::MacAddress::parse_str(&config.config.mac).expect("unexpected mac address");
        let usernet = Usernet::new(self.ctx.clone());
        for fwd in config.config.forward.iter() {
            usernet
                .add_host_forward(
                    fwd.protocol == ForwardProtocol::Udp,
                    fwd.host_addr,
                    fwd.host_port,
                    fwd.guest_port,
                )
                .expect("cannot establish port forwarding");
        }

        match config.config.r#type.as_str() {
            #[cfg(feature = "virtio-network")]
            "virtio" => {
                self.add_virtio(config.io_base, config.irq, |sys, irq| {
                    crate::hw::virtio::Network::new(sys.ctx.clone(), irq, usernet, mac)
                });
            }
            #[cfg(feature = "network-xemaclite")]
            "xemaclite" => {
                use crate::hw::network::XemacLite;

                let base = config.io_base.unwrap_or_else(|| self.allocate_mem(0x2000));
                let irq = config.irq.unwrap_or_else(|| self.allocate_irq());
                let irq_pin = self.register_irq(irq, true);
                let xemaclite = XemacLite::new(self.ctx.clone(), irq_pin, Box::new(usernet));
                self.register_mem(base, 0x2000, Arc::new(xemaclite));
                let mut node = XemacLite::build_dt(base, mac.to_array());
                node.add_prop("reg", &[base as u64, 0x2000][..]);
                node.add_prop("interrupts-extended", &[self.plic_phandle as u32 + 1, irq][..]);
                self.fdt.child.push(node);
            }
            _ => panic!("unknown device type"),
        }
    }

    pub fn instantiate_console(
        &mut self,
        config: &DeviceConfig<ConsoleConfig>,
        console: Box<dyn crate::serial::Serial>,
    ) {
        match config.config.r#type {
            #[cfg(feature = "virtio-console")]
            ConsoleType::Virtio => {
                self.add_virtio(config.io_base, config.irq, |sys, irq| {
                    crate::hw::virtio::Console::new(
                        sys.ctx.clone(),
                        irq,
                        console,
                        config.config.resize,
                    )
                });
            }
            #[cfg(feature = "console-ns16550")]
            ConsoleType::NS16550 => {
                use crate::hw::console::NS16550;

                let base = config.io_base.unwrap_or_else(|| self.allocate_mem(0x1000));
                let irq = config.irq.unwrap_or_else(|| self.allocate_irq());
                let irq_pin = self.register_irq(irq, false);

                let ns16550 = NS16550::new(self.ctx.clone(), irq_pin, console);
                self.register_mem(base, 0x1000, Arc::new(ns16550));

                let mut node = NS16550::build_dt(base);
                node.add_prop("reg", &[base as u64, 0x1000][..]);
                node.add_prop("interrupts-extended", &[self.plic_phandle as u32 + 1, irq][..]);
                self.fdt.child.push(node);
            }
            #[allow(unreachable_patterns)]
            _ => panic!("unknown device type"),
        }
    }

    #[cfg(feature = "rtc-zyncmp")]
    pub fn instantiate_rtc(&mut self, config: &DeviceConfig<RTCConfig>) {
        use crate::hw::rtc::ZyncMp;

        let mem = config.io_base.unwrap_or_else(|| self.allocate_mem(4096));
        let irq = config.irq.unwrap_or_else(|| self.allocate_irq());
        let irq_pin = self.register_irq(irq, true);
        let irq2 = config.config.irq2.unwrap_or_else(|| self.allocate_irq());
        let irq2_pin = self.register_irq(irq2, true);

        let rtc = Arc::new(ZyncMp::new(irq_pin, irq2_pin));
        self.register_mem(mem, 4096, rtc);

        let mut node = ZyncMp::build_dt(mem);
        node.add_prop("reg", &[mem as u64, 0x100][..]);
        node.add_prop("interrupt-parent", self.plic_phandle as u32 + 1);
        node.add_prop("interrupts", &[irq, irq + 1][..]);
        self.fdt.child.push(node);
    }
}
