use crate::serial::Serial;
use crate::{IoMemory, IrqPin, RuntimeContext};
use futures::channel::mpsc::UnboundedSender;
use futures::future::AbortHandle;
use futures::prelude::*;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

const ADDR_RBR_THR_DLL: usize = 0x00;
const ADDR_IER_DLM: usize = 0x04;
const ADDR_IIR_FCR: usize = 0x08;
const ADDR_LCR: usize = 0x0C;
const ADDR_MCR: usize = 0x10;
const ADDR_LSR: usize = 0x14;
const ADDR_MSR: usize = 0x18;
const ADDR_SCR: usize = 0x1C;

/// An implementation of ns16550.
///
/// This is a behaviour model. Both the send and receive buffer are infinite.
pub struct NS16550 {
    inner: Arc<Inner>,
    rx_handle: AbortHandle,
    // This sender is placed here so its dropping will cause tx task to stop.
    tx_sender: Mutex<UnboundedSender<u8>>,
}

struct Inner {
    state: Mutex<State>,
    irq: Box<dyn IrqPin>,
    console: Box<dyn Serial>,
    ctx: Arc<dyn RuntimeContext>,
}

struct State {
    divisor: u16,
    ier: u8,
    lcr: u8,
    mcr: u8,
    fcr: u8,
    scr: u8,
    // Receive buffer trigger level
    trigger: usize,
    // Current pending IRQ
    irq: Option<u8>,
    rx_buffer: VecDeque<u8>,
}

impl Drop for NS16550 {
    fn drop(&mut self) {
        self.rx_handle.abort();
    }
}

impl NS16550 {
    /// Create a NS16550 device.
    pub fn new(
        ctx: Arc<dyn RuntimeContext>,
        irq: Box<dyn IrqPin>,
        console: Box<dyn Serial>,
    ) -> NS16550 {
        let state = State {
            divisor: 260,
            ier: 0,
            lcr: 3,
            mcr: 0,
            fcr: 0,
            scr: 0,
            trigger: 1,
            irq: None,
            rx_buffer: VecDeque::new(),
        };
        let inner = Arc::new(Inner { state: Mutex::new(state), irq, console, ctx });
        let rx_handle = Self::start_rx(inner.clone());
        let tx_sender = Self::start_tx(inner.clone());
        Self { inner, rx_handle, tx_sender: Mutex::new(tx_sender) }
    }

    fn update_irq(inner: &Inner, state: &mut State) {
        fn compute_irq(state: &mut State) -> Option<u8> {
            if state.ier & 0b0001 != 0 && !state.rx_buffer.is_empty() {
                // Threshold reached, report "Received Data Available"
                if state.rx_buffer.len() >= state.trigger {
                    return Some(0b010);
                }
                // Otherwise report "Character Timeout"
                return Some(0b110);
            }
            // Always report "Transmitter Holding Register Empty"
            if state.ier & 0b0010 != 0 {
                return Some(0b001);
            }
            None
        }
        state.irq = compute_irq(state);
        inner.irq.set_level(state.irq.is_some());
    }

    fn start_tx(inner: Arc<Inner>) -> futures::channel::mpsc::UnboundedSender<u8> {
        let (sender, mut recv) = futures::channel::mpsc::unbounded::<u8>();
        let ctx = inner.ctx.clone();
        ctx.spawn(Box::pin(async move {
            while let Some(byte) = recv.next().await {
                inner.console.write(&[byte]).await.unwrap();
            }
        }));
        sender
    }

    fn start_rx(inner: Arc<Inner>) -> AbortHandle {
        let ctx = inner.ctx.clone();
        let (handle, reg) = futures::future::AbortHandle::new_pair();
        ctx.spawn(Box::pin(async move {
            let _ = futures::future::Abortable::new(
                async move {
                    let mut buffer = [0; 2048];
                    loop {
                        // Receive into ping
                        let len = inner.console.read(&mut buffer).await.unwrap();
                        let mut state = inner.state.lock();
                        state.rx_buffer.extend(&buffer[..len]);
                        Self::update_irq(&inner, &mut state);
                    }
                },
                reg,
            )
            .await;
        }));
        handle
    }

    /// Build the device tree for this device.
    ///
    /// Node name will include the given base address, but `reg` and `interrupts` fields will not be filled.
    pub fn build_dt(base: usize) -> fdt::Node {
        let mut node = fdt::Node::new(format!("serial@{:x}", base));
        node.add_prop("clock-frequency", 40_000_000u32);
        node.add_prop("reg-shift", 2u32);
        node.add_prop("reg-io-width", 4u32);
        node.add_prop("compatible", "ns16550a");
        node.add_prop("current-speed", 9600u32);
        node
    }
}

impl IoMemory for NS16550 {
    fn read(&self, addr: usize, _size: u32) -> u64 {
        let mut state = self.inner.state.lock();
        let val = match addr {
            ADDR_RBR_THR_DLL => {
                if state.lcr & 0x80 == 0 {
                    // RBR
                    let val = state.rx_buffer.pop_front().unwrap_or(0);
                    Self::update_irq(&self.inner, &mut state);
                    val
                } else {
                    // DLL
                    state.divisor as u8
                }
            }
            ADDR_IER_DLM => {
                if state.lcr & 0x80 == 0 {
                    // IER
                    state.ier
                } else {
                    // DLM
                    (state.divisor >> 8) as u8
                }
            }
            ADDR_IIR_FCR => {
                if state.lcr & 0x80 == 0 {
                    // IIR
                    let fifo_en = state.fcr & 0b1;
                    let irq_pending = if let Some(id) = state.irq { id << 1 } else { 1 };
                    fifo_en << 6 | irq_pending
                } else {
                    // FCR read
                    state.fcr
                }
            }
            ADDR_LCR => state.lcr,
            ADDR_MCR => state.mcr,
            ADDR_LSR => 0x60 | if state.rx_buffer.is_empty() { 0 } else { 1 },
            ADDR_MSR => 0,
            ADDR_SCR => state.scr,
            _ => {
                error!(target: "ns16550", "illegal register read 0x{:x}", addr);
                0
            }
        };
        trace!(target: "ns16550", "read register 0x{:x} as 0x{:x}", addr, val);
        val as u64
    }

    fn write(&self, addr: usize, value: u64, _size: u32) {
        let mut state = self.inner.state.lock();
        let value = value as u8;
        trace!(target: "ns16550", "write register 0x{:x} = 0x{:x}", addr, value);
        match addr {
            ADDR_RBR_THR_DLL => {
                if state.lcr & 0x80 == 0 {
                    // THR
                    // error!(target: "ns16550", "THR write");
                    self.tx_sender.lock().unbounded_send(value).unwrap();
                } else {
                    // DLL
                    state.divisor = (state.divisor & 0xff00) | value as u16;
                }
            }
            ADDR_IER_DLM => {
                if state.lcr & 0x80 == 0 {
                    // IER
                    state.ier = value & 0b1111;
                } else {
                    // DLM
                    state.divisor = (state.divisor & 0x00ff) | (value as u16) << 8;
                }
            }
            ADDR_IIR_FCR => {
                state.fcr = value & 0b11001001;
                if value & 0b10 != 0 {
                    // Reset receiver FIFO
                    state.rx_buffer.clear();
                }
                // Set trigger level
                state.trigger = match (value >> 6) & 0b11 {
                    0b00 => 1,
                    0b01 => 4,
                    0b10 => 8,
                    _ => 14,
                };
            }
            ADDR_LCR => {
                state.lcr = value;
            }
            ADDR_MCR => {
                state.mcr = value & 0b11111;
            }
            ADDR_LSR => {
                // Ignore LSR write
                info!(target: "ns16550", "LSR write");
            }
            ADDR_MSR => {
                // Ignore MSR write
                info!(target: "ns16550", "MSR write");
            }
            ADDR_SCR => {
                state.scr = value;
            }
            _ => {
                error!(target: "ns16550", "illegal register write 0x{:x} = 0x{:x}", addr, value);
            }
        }
        Self::update_irq(&self.inner, &mut state);
    }
}
