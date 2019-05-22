use super::csr::Csr;

#[repr(C)]
struct CacheLine {
    tag: u64,
    paddr: u64,
}

#[repr(C)]
struct Context {
    registers: [u64; 32],
    fp_registers: [u64; 32],
    pc: u64,
    instret: u64,
    fcsr: u64,

    // For load reservation
    lr: u64,

    // S-mode CSRs
    sstatus: u64,
    sie: u64,
    stvec: u64,
    sscratch: u64,
    sepc: u64,
    scause: u64,
    stval: u64,
    sip: u64,
    satp: u64,

    timecmp: u64,

    // Current privilege level
    prv: u64,

    // Pending exceptions: sstatus.sie ? sie & sip : 0
    pending: u64,

    line: [CacheLine; 1024],
}

type Trap = u64;

#[no_mangle]
extern "C" fn rs_translate(ctx: &mut Context, addr: u64, write: bool, out: &mut u64) -> Trap {
    let fault_type = if write { 15 } else { 13 };
    if (ctx.satp >> 60) == 0 {
        *out = addr;
        return 0;
    }
    let mut ppn = ctx.satp & ((1u64 << 44) - 1);
    let mut pte: u64 = unsafe { crate::emu::read_memory(ppn * 4096 + ((addr >> 30) & 511) * 8) };
    if (pte & 1) == 0 { return fault_type; }
    let ret = loop {
        ppn = pte >> 10;
        if (pte & 0xf) != 1 {
            break (ppn << 12) | (addr & ((1<<30)-1));
        }
        pte = unsafe { crate::emu::read_memory(ppn * 4096 + ((addr >> 21) & 511) * 8) };
        if (pte & 1) == 0 { return fault_type; }
        ppn = pte >> 10;
        if (pte & 0xf) != 1 {
            break (ppn << 12) | (addr & ((1<<21)-1));
        }
        pte = unsafe { crate::emu::read_memory(ppn * 4096 + ((addr >> 12) & 511) * 8) };
        if (pte & 1) == 0 { return fault_type; }

        ppn = pte >> 10;
        break (ppn << 12) | (addr & 4095);
    };
    if (pte & 0x40) == 0 || (write && ((pte & 0x4) == 0 || (pte & 0x80) == 0)) { return fault_type; }
    *out = ret;
    return 0;
}
