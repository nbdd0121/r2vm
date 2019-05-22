use super::csr::Csr;
use super::op::{LegacyOp, Op};

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

extern {
    fn legacy_step(ctx: &mut Context, op: &LegacyOp) -> Trap;
}

fn step(ctx: &mut Context, op: &Op) -> Trap {
    macro_rules! write_reg {
        ($rd: expr, $expression:expr) => ({
            let rd = $rd as usize;
            let value = $expression;
            if rd != 0 { ctx.registers[rd] = value }
        })
    }

    match op {
        Op::Legacy(op) => return unsafe{legacy_step(ctx, op)},
        Op::Illegal => return 2,
        /* AUIPC */
        &Op::Auipc { rd, imm } => write_reg!(rd, ctx.pc - 4 + imm as u64),
    0
}

fn run_block(ctx: &mut Context) -> u64 {
    let pc = ctx.pc;
    let mut phys_pc = 0;
    let ex = rs_translate(ctx, pc, false, &mut phys_pc);
    if ex != 0 {
        ctx.stval = pc;
        return ex
    }
    let mut phys_pc_next = 0;
    // Ignore error in this case
    rs_translate(ctx, (pc &! 4095) + 4096, false, &mut phys_pc_next);

    let (vec, start, end) = icache().entry(phys_pc).or_insert_with(|| {
        let (mut vec, start, end) = super::decode::decode_block(phys_pc, phys_pc_next);
        // Function step will assume the pc is just past the instruction, however we will reduce
        // change to instret by increment past the whole basic block. We preprocess the block to
        // handle the difference.
        for (op, c) in &mut vec {
            phys_pc += if *c { 2 } else { 4 };
            if let Op::Auipc { imm, .. } = op {
                *imm -= (end - phys_pc) as i32;
            }
        }
        (vec, start, end)
    });

    ctx.pc += *end - *start;
    ctx.instret += vec.len() as u64;

    for i in 0..vec.len() {
        let inst = &vec[i].0;
        let ex = step(ctx, inst);
        if ex != 0 {
            // Adjust pc and instret by iterating through remaining instructions.
            for j in i..vec.len() {
                ctx.pc -= if vec[j].1 { 2 } else { 4 };
            }
            ctx.instret -= (vec.len() - i) as u64;
            return ex;
        }
    }
    0
}

#[no_mangle]
extern "C" fn rust_emu_start(ctx: &mut Context) {
    loop {
        let ex = loop {
            let ex = run_block(ctx);
            if ex != 0 { break ex }
            if ctx.instret >= ctx.timecmp {
                ctx.sip |= 32;
                ctx.pending = if (ctx.sstatus & 0x2) != 0 { ctx.sip & ctx.sie } else { 0 };
            }
            if ctx.pending != 0 {
                // The highest set bit of ctx.pending
                let pending = 63 - ctx.pending.leading_zeros() as u64;
                ctx.sip &= !(1 << pending);
                // The highest bit of cause indicates this is an interrupt
                break (1 << 63) | pending
            }
        };

        // if user_only {
        //     eprintln!("unhandled trap {}", ex);
        //     eprintln!("pc  = {:16x}  ra  = {:16x}", ctx.pc, ctx.registers[1]);
        //     for i in (2..32).step_by(2) {
        //         eprintln!(
        //             "{:-3} = {:16x}  {:-3} = {:16x}",
        //             super::disasm::REG_NAMES[i], ctx.registers[i],
        //             super::disasm::REG_NAMES[i + 1], ctx.registers[i + 1]
        //         );
        //     }
        //     return;
        // }

        ctx.sepc = ctx.pc;
        ctx.scause = ex;

        // Clear or set SPP bit
        if ctx.prv != 0 {
            ctx.sstatus |= 0x100;
        } else {
            ctx.sstatus &=! 0x100;
        }
        // Clear of set SPIE bit
        if (ctx.sstatus & 0x2) != 0 {
            ctx.sstatus |= 0x20;
        } else {
            ctx.sstatus &=! 0x20;
        }
        // Clear SIE
        ctx.sstatus &= !0x2;
        ctx.pending = 0;
        // Switch to S-mode
        ctx.prv = 1;
        ctx.pc = ctx.stvec;
    }
}
