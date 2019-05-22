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

/// Perform a CSR read on a context. Note that this operation performs no checks before accessing
/// them.
/// The caller should ensure:
/// * The current privilege level has enough permission to access the CSR. CSR is nicely partition
///   into regions, so privilege check can be easily done.
/// * U-mode code does not access floating point CSRs with FS == Off.
/// * The CSR is a valid CSR.
fn read_csr(ctx: &mut Context, csr: Csr) -> u64 {
    match csr {
        Csr::Fflags => ctx.fcsr & 0b11111,
        Csr::Frm => (ctx.fcsr >> 5) & 0b111,
        Csr::Fcsr => ctx.fcsr,
        // Pretend that we're 100MHz
        Csr::Time => ctx.instret / 100,
        // We assume the instret is incremented already
        Csr::Instret => ctx.instret - 1,
        Csr::Sstatus => ctx.sstatus,
        Csr::Sie => ctx.sie,
        Csr::Stvec => ctx.stvec,
        Csr::Scounteren => 0,
        Csr::Sscratch => ctx.sscratch,
        Csr::Sepc => ctx.sepc,
        Csr::Scause => ctx.scause,
        Csr::Stval => ctx.stval,
        Csr::Sip => ctx.sip,
        Csr::Satp => ctx.satp,
        _ => {
           unreachable!("read illegal csr {}", csr as i32);
        }
    }
}

fn write_csr(ctx: &mut Context, csr: Csr, value: u64) {
    match csr {
        Csr::Fflags => {
            ctx.fcsr = (ctx.fcsr &! 0b11111) | (value & 0b11111);
            // Set SSTATUS.{FS, SD}
            ctx.sstatus |= 0x8000000000006000
        }
        Csr::Frm => {
            ctx.fcsr = (ctx.fcsr &! (0b111 << 5)) | ((value & 0b111) << 5);
            ctx.sstatus |= 0x8000000000006000
        }
        Csr::Fcsr => {
            ctx.fcsr = value;
            ctx.sstatus |= 0x8000000000006000
        }
        Csr::Instret => ctx.instret = value,
        Csr::Sstatus => {
            // Mask-out non-writable bits
            let mut value = value & 0xC6122;
            // SSTATUS.FS = dirty, also set SD
            if (value & 0x6000) == 0x6000 { value |= 0x8000000000000000 }
            // Hard-wire UXL to 0b10, i.e. 64-bit.
            value |= 0x200000000;
            ctx.sstatus = value;
            // Update ctx.pending. Important!
            ctx.pending = if (ctx.sstatus & 0x2) != 0 { ctx.sip & ctx.sie } else { 0 }
        }
        Csr::Sie => {
            ctx.sie = value;
            ctx.pending = if (ctx.sstatus & 0x2) != 0 { ctx.sip & ctx.sie } else { 0 }
        }
        Csr::Stvec => {
            // We support MODE 0 only at the moment
            if (value & 2) != 0 { return }
            ctx.stvec = value;
        }
        Csr::Scounteren => (),
        Csr::Sscratch => ctx.sscratch = value,
        Csr::Sepc => ctx.sepc = value &! 1,
        Csr::Scause => ctx.scause = value,
        Csr::Stval => ctx.stval = value,
        // Csr::Sip => ctx.sip = value,
        Csr::Satp => {
            match value >> 60 {
                // No paging
                0 => ctx.satp = 0,
                // ASID not yet supported
                8 => ctx.satp = value &! (0xffffu64 << 44),
                // We only support SV39 at the moment.
                _ => (),
            }
            for line in ctx.line.iter_mut() {
                line.tag = u64::max_value();
            }
        }
        _ => {
           unreachable!("write illegal csr {}", csr as i32);
        }
    }
}

type Trap = u64;

fn translate(ctx: &mut Context, addr: u64, write: bool) -> Result<u64, Trap> {
    let fault_type = if write { 15 } else { 13 };
    if (ctx.satp >> 60) == 0 {
        return Ok(addr);
    }
    let mut ppn = ctx.satp & ((1u64 << 44) - 1);
    let mut pte: u64 = unsafe { crate::emu::read_memory(ppn * 4096 + ((addr >> 30) & 511) * 8) };
    if (pte & 1) == 0 { return Err(fault_type); }
    let ret = loop {
        ppn = pte >> 10;
        if (pte & 0xf) != 1 {
            break (ppn << 12) | (addr & ((1<<30)-1));
        }
        pte = unsafe { crate::emu::read_memory(ppn * 4096 + ((addr >> 21) & 511) * 8) };
        if (pte & 1) == 0 { return Err(fault_type); }
        ppn = pte >> 10;
        if (pte & 0xf) != 1 {
            break (ppn << 12) | (addr & ((1<<21)-1));
        }
        pte = unsafe { crate::emu::read_memory(ppn * 4096 + ((addr >> 12) & 511) * 8) };
        if (pte & 1) == 0 { return Err(fault_type); }

        ppn = pte >> 10;
        break (ppn << 12) | (addr & 4095);
    };
    if (pte & 0x40) == 0 || (write && ((pte & 0x4) == 0 || (pte & 0x80) == 0)) { return Err(fault_type); }
    return Ok(ret);
}

#[no_mangle]
extern "C" fn rs_translate(ctx: &mut Context, addr: u64, write: bool, out: &mut u64) -> Trap {
    match translate(ctx, addr, write) {
        Ok(ret) => {
            *out = ret;
            0
        }
        Err(trap) => trap,
    }
}
}

extern {
    fn legacy_step(ctx: &mut Context, op: &LegacyOp) -> Trap;
}

fn sbi_call(ctx: &mut Context, nr: u64, arg0: u64) -> u64 {
    match nr {
        0 => {
            ctx.timecmp = arg0 * 100;
            0
        }
        1 => {
            crate::io::console::console_putchar(arg0 as u8);
            0
        }
        2 => crate::io::console::console_getchar() as u64,
        3 => panic!("Ignore clear_ipi"),
        4 => panic!("Ignore send_ipi"),
        5 | 6 | 7 => {
            for l in ctx.line.iter_mut() {
                l.tag = i64::max_value() as u64;
            }
            0
        }
        8 => std::process::exit(0),
        _ => {
            panic!("unknown sbi call {}", nr);
        }
    }
}

fn step(ctx: &mut Context, op: &Op, compressed: bool) -> Result<(), Trap> {
    macro_rules! read_reg {
        ($rs: expr) => {{
            let rs = $rs as usize;
            ctx.registers[rs]
        }}
    }
    macro_rules! write_reg {
        ($rd: expr, $expression:expr) => {{
            let rd = $rd as usize;
            let value = $expression;
            if rd != 0 { ctx.registers[rd] = value }
        }}
    }
    macro_rules! len {
        () => {
            if compressed { 2 } else { 4 }
        }
    }

    match *op {
        Op::Legacy(ref op) => {
            let ex = unsafe{legacy_step(ctx, op)};
            if ex != 0 { return Err(ex) }
        }
        Op::Illegal => return Err(2),
        /* OP-IMM */
        Op::Addi { rd, rs1, imm }=> write_reg!(rd, read_reg!(rs1).wrapping_add(imm as u64)),
        Op::Slli { rd, rs1, imm }=> write_reg!(rd, read_reg!(rs1) << imm),
        Op::Slti { rd, rs1, imm }=> write_reg!(rd, ((read_reg!(rs1) as i64) < (imm as i64)) as u64),
        Op::Sltiu { rd, rs1, imm }=> write_reg!(rd, (read_reg!(rs1) < (imm as u64)) as u64),
        Op::Xori { rd, rs1, imm }=> write_reg!(rd, read_reg!(rs1) ^ (imm as u64)),
        Op::Srli { rd, rs1, imm }=> write_reg!(rd, read_reg!(rs1) >> imm),
        Op::Srai { rd, rs1, imm }=> write_reg!(rd, ((read_reg!(rs1) as i64) >> imm) as u64),
        Op::Ori { rd, rs1, imm }=> write_reg!(rd, read_reg!(rs1) | (imm as u64)),
        Op::Andi { rd, rs1, imm }=> write_reg!(rd, read_reg!(rs1) & (imm as u64)),
        /* MISC-MEM */
        Op::Fence => (),
        Op::FenceI => (),
        /* OP-IMM-32 */
        Op::Addiw { rd, rs1, imm }=> write_reg!(rd, ((read_reg!(rs1) as i32).wrapping_add(imm)) as u64),
        Op::Slliw { rd, rs1, imm }=> write_reg!(rd, ((read_reg!(rs1) as i32) << imm) as u64),
        Op::Srliw { rd, rs1, imm }=> write_reg!(rd, (((read_reg!(rs1) as u32) >> imm) as i32) as u64),
        Op::Sraiw { rd, rs1, imm }=> write_reg!(rd, ((read_reg!(rs1) as i32) >> imm) as u64),
        /* OP */
        Op::Add { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1).wrapping_add(read_reg!(rs2))),
        Op::Sub { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1).wrapping_sub(read_reg!(rs2))),
        Op::Sll { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1) << (read_reg!(rs2) & 63)),
        Op::Slt { rd, rs1, rs2 } => write_reg!(rd, ((read_reg!(rs1) as i64) < (read_reg!(rs2) as i64)) as u64),
        Op::Sltu { rd, rs1, rs2 } => write_reg!(rd, (read_reg!(rs1) < read_reg!(rs2)) as u64),
        Op::Xor { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1) ^ read_reg!(rs2)),
        Op::Srl { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1) >> (read_reg!(rs2) & 63)),
        Op::Sra { rd, rs1, rs2 } => write_reg!(rd, ((read_reg!(rs1) as i64) >> (read_reg!(rs2) & 63)) as u64),
        Op::Or { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1) | read_reg!(rs2)),
        Op::And { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1) & read_reg!(rs2)),
        /* LUI */
        Op::Lui { rd, imm } => write_reg!(rd, imm as u64),
        Op::Addw { rd, rs1, rs2 }=> write_reg!(rd, ((read_reg!(rs1) as i32).wrapping_add(read_reg!(rs2) as i32)) as u64),
        Op::Subw { rd, rs1, rs2 }=> write_reg!(rd, ((read_reg!(rs1) as i32).wrapping_sub(read_reg!(rs2) as i32)) as u64),
        Op::Sllw { rd, rs1, rs2 }=> write_reg!(rd, ((read_reg!(rs1) as i32) << (read_reg!(rs2) & 31)) as u64),
        Op::Srlw { rd, rs1, rs2 }=> write_reg!(rd, (((read_reg!(rs1) as u32) >> (read_reg!(rs2) & 31)) as i32) as u64),
        Op::Sraw { rd, rs1, rs2 }=> write_reg!(rd, ((read_reg!(rs1) as i32) >> (read_reg!(rs2) & 31)) as u64),
        /* AUIPC */
        Op::Auipc { rd, imm } => write_reg!(rd, ctx.pc.wrapping_sub(len!()).wrapping_add(imm as u64)),
        /* BRANCH */
        // Same as auipc, PC-relative instructions are relative to the origin pc instead of the incremented one.
        Op::Beq { rs1, rs2, imm } => {
            if read_reg!(rs1) == read_reg!(rs2) {
                ctx.pc = ctx.pc.wrapping_sub(len!()).wrapping_add(imm as u64);
            }
        }
        Op::Bne { rs1, rs2, imm } => {
            if read_reg!(rs1) != read_reg!(rs2) {
                ctx.pc = ctx.pc.wrapping_sub(len!()).wrapping_add(imm as u64);
            }
        }
        Op::Blt { rs1, rs2, imm } => {
            if (read_reg!(rs1) as i64) < (read_reg!(rs2) as i64) {
                ctx.pc = ctx.pc.wrapping_sub(len!()).wrapping_add(imm as u64);
            }
        }
        Op::Bge { rs1, rs2, imm } => {
            if (read_reg!(rs1) as i64) >= (read_reg!(rs2) as i64) {
                ctx.pc = ctx.pc.wrapping_sub(len!()).wrapping_add(imm as u64);
            }
        }
        Op::Bltu { rs1, rs2, imm } => {
            if read_reg!(rs1) < read_reg!(rs2) {
                ctx.pc = ctx.pc.wrapping_sub(len!()).wrapping_add(imm as u64);
            }
        }
        Op::Bgeu { rs1, rs2, imm } => {
            if read_reg!(rs1) >= read_reg!(rs2) {
                ctx.pc = ctx.pc.wrapping_sub(len!()).wrapping_add(imm as u64);
            }
        }
        /* JALR */
        Op::Jalr { rd, rs1, imm } => {
            let new_pc = (read_reg!(rs1).wrapping_add(imm as u64)) &! 1;
            write_reg!(rd, ctx.pc);
            ctx.pc = new_pc;
        }
        /* JAL */
        Op::Jal { rd, imm } => {
            write_reg!(rd, ctx.pc);
            ctx.pc = ctx.pc.wrapping_sub(len!()).wrapping_add(imm as u64);
        }
        /* SYSTEM */
        Op::Ecall =>
            if ctx.prv == 0 {
                // if (emu::state::user_only) {
                //     context->registers[10] = emu::syscall(
                //         static_cast<abi::Syscall_number>(context->registers[17]),
                //         context->registers[10],
                //         context->registers[11],
                //         context->registers[12],
                //         context->registers[13],
                //         context->registers[14],
                //         context->registers[15]
                //     );
                // } else {
                    return Err(8);
                // }
            } else {
                ctx.registers[10] = sbi_call(
                    ctx,
                    ctx.registers[17],
                    ctx.registers[10],
                )
            }
        Op::Ebreak => return Err(3),
        Op::Csrrw { rd, rs1, csr } => {
            let result = if rd != 0 { read_csr(ctx, csr) } else { 0 };
            write_csr(ctx, csr, read_reg!(rs1));
            write_reg!(rd, result);
        }
        Op::Csrrs { rd, rs1, csr } => {
            let result = read_csr(ctx, csr);
            if rs1 != 0 { write_csr(ctx, csr, result | read_reg!(rs1)) }
            write_reg!(rd, result);
        }
        Op::Csrrc { rd, rs1, csr } => {
            let result = read_csr(ctx, csr);
            if rs1 != 0 { write_csr(ctx, csr, result &! read_reg!(rs1)) }
            write_reg!(rd, result);
        }
        Op::Csrrwi { rd, imm, csr } => {
            let result = if rd != 0 { read_csr(ctx, csr) } else { 0 };
            write_csr(ctx, csr, imm as u64);
            write_reg!(rd, result);
        }
        Op::Csrrsi { rd, imm, csr } => {
            let result = read_csr(ctx, csr);
            if imm != 0 { write_csr(ctx, csr, result | imm as u64) }
            write_reg!(rd, result);
        }
        Op::Csrrci { rd, imm, csr } => {
            let result = read_csr(ctx, csr);
            if imm != 0 { write_csr(ctx, csr, result &! imm as u64) }
            write_reg!(rd, result);
        }

        /* M-extension */
        Op::Mul { rd, rs1, rs2 } => write_reg!(rd, read_reg!(rs1).wrapping_mul(read_reg!(rs2))),
        Op::Mulh { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as i64 as i128;
            let b = read_reg!(rs2) as i64 as i128;
            write_reg!(rd, ((a * b) >> 64) as u64)
        }
        Op::Mulhsu { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as i64;
            let b = read_reg!(rs2);

            // First multiply as uint128_t. This will give compiler chance to optimize better.
            let exta = a as u64 as u128;
            let extb = b as u128;
            let mut r = ((exta * extb) >> 64) as u64;

            // If rs1 < 0, then the high bits of a should be all one, but the actual bits in exta
            // is all zero. Therefore we need to compensate this error by adding multiplying
            // 0xFFFFFFFF and b, which is effective -b.
            if a < 0 { r = r.wrapping_sub(b) }
            write_reg!(rd, r)
        }
        Op::Mulhu { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as u128;
            let b = read_reg!(rs2) as u128;
            write_reg!(rd, ((a * b) >> 64) as u64)
        }
        Op::Div { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as i64;
            let b = read_reg!(rs2) as i64;
            let r = if b == 0 { -1 } else { a.wrapping_div(b) };
            write_reg!(rd, r as u64);
        }
        Op::Divu { rd, rs1, rs2 } => {
            let a = read_reg!(rs1);
            let b = read_reg!(rs2);
            let r = if b == 0 { (-1i64) as u64 } else { a / b };
            write_reg!(rd, r);
        }
        Op::Rem { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as i64;
            let b = read_reg!(rs2) as i64;
            let r = if b == 0 { a } else { a.wrapping_rem(b) };
            write_reg!(rd, r as u64);
        }
        Op::Remu { rd, rs1, rs2 } => {
            let a = read_reg!(rs1);
            let b = read_reg!(rs2);
            let r = if b == 0 { a } else { a % b };
            write_reg!(rd, r);
        }
        Op::Mulw { rd, rs1, rs2 } => write_reg!(rd, ((read_reg!(rs1) as i32).wrapping_mul(read_reg!(rs2) as i32)) as u64),
        Op::Divw { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as i32;
            let b = read_reg!(rs2) as i32;
            let r = if b == 0 { -1 } else { a.wrapping_div(b) };
            write_reg!(rd, r as u64);
        }
        Op::Divuw { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as u32;
            let b = read_reg!(rs2) as u32;
            let r = if b == 0 { (-1i32) as u32 } else { a / b };
            write_reg!(rd, r as i32 as u64);
        }
        Op::Remw { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as i32;
            let b = read_reg!(rs2) as i32;
            let r = if b == 0 { a } else { a.wrapping_rem(b) };
            write_reg!(rd, r as u64);
        }
        Op::Remuw { rd, rs1, rs2 } => {
            let a = read_reg!(rs1) as u32;
            let b = read_reg!(rs2) as u32;
            let r = if b == 0 { a } else { a % b };
            write_reg!(rd, r as i32 as u64);
        }

        /* Privileged */
        Op::Sret => {
            ctx.pc = ctx.sepc;

            // Set privilege according to SPP
            if (ctx.sstatus & 0x100) != 0 {
                ctx.prv = 1;
            } else {
                ctx.prv = 0;
            }

            // Set SIE according to SPIE
            if (ctx.sstatus & 0x20) != 0 {
                ctx.sstatus |= 0x2;
            } else {
                ctx.sstatus &=! 0x2;
            }

            // Set SPIE to 1
            ctx.sstatus |= 0x20;
            // Set SPP to U
            ctx.sstatus &=! 0x100;
        }
        Op::Wfi => (),
        Op::SfenceVma {..} => {
            for l in ctx.line.iter_mut() {
                l.tag = i64::max_value() as u64;
            }
        }
    }
    Ok(())
}

fn run_block(ctx: &mut Context) -> Result<(), Trap> {
    let pc = ctx.pc;
    let mut phys_pc = match translate(ctx, pc, false) {
        Ok(pc) => pc,
        Err(ex) => {
            ctx.stval = pc;
            return Err(ex);
        }
    };
    // Ignore error in this case
    let phys_pc_next = match translate(ctx, (pc &! 4095) + 4096, false) {
        Ok(pc) => pc,
        Err(_) => 0,
    };

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
        let (ref inst, c) = vec[i];
        match step(ctx, inst, c) {
            Ok(()) => (),
            Err(ex) => {
                // Adjust pc and instret by iterating through remaining instructions.
                for j in i..vec.len() {
                    ctx.pc -= if vec[j].1 { 2 } else { 4 };
                }
                ctx.instret -= (vec.len() - i) as u64;
                return Err(ex);
            }
        }
    }
    Ok(())
}

#[no_mangle]
extern "C" fn rust_emu_start(ctx: &mut Context) {
    loop {
        let ex = loop {
            match run_block(ctx) {
                Ok(()) => (),
                Err(ex) => break ex,
            }
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
