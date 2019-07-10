use super::csr::Csr;
use super::op::Op;
use crate::util::int::{CastFrom};
use crate::util::softfp::{self, F32, F64};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CacheLine {
    /// Lowest bit is used to store whether this cache line is non-writable
    /// It actually stores (tag << 1) | non-writable
    pub tag: u64,
    /// It actually stores vaddr ^ paddr
    pub paddr: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Context {
    pub registers: [u64; 32],
    pub fp_registers: [u64; 32],
    pub pc: u64,
    pub instret: u64,
    pub fcsr: u64,

    // For load reservation
    pub lr_addr: u64,
    pub lr_value: u64,

    // S-mode CSRs
    pub sstatus: u64,
    pub sie: u64,
    pub stvec: u64,
    pub sscratch: u64,
    pub sepc: u64,
    pub scause: u64,
    pub stval: u64,
    pub sip: u64,
    pub satp: u64,

    pub cycle: u64,
    pub timecmp: u64,

    // Current privilege level
    pub prv: u64,

    // Pending exceptions: sstatus.sie ? sie & sip : 0
    pub pending: u64,

    pub line: [CacheLine; 1024],
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
        Csr::Time => ctx.cycle / 100,
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
           unreachable!("read illegal csr {:x}", csr as i32);
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
            ctx.fcsr = value & 0xff;
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
        Csr::Sip => {
            // Only SSIP flag can be cleared by software
            ctx.sip = ctx.sip &! 0x2 | value & 0x2;
            ctx.pending = if (ctx.sstatus & 0x2) != 0 { ctx.sip & ctx.sie } else { 0 }
        }
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
           unreachable!("write illegal csr {:x} = {:x}", csr as i32, value);
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
    let mut pte: u64 = crate::emu::read_memory(ppn * 4096 + ((addr >> 30) & 511) * 8);
    if (pte & 1) == 0 { return Err(fault_type); }
    let ret = loop {
        ppn = pte >> 10;
        if (pte & 0xf) != 1 {
            break (ppn << 12) | (addr & ((1<<30)-1));
        }
        pte = crate::emu::read_memory(ppn * 4096 + ((addr >> 21) & 511) * 8);
        if (pte & 1) == 0 { return Err(fault_type); }
        ppn = pte >> 10;
        if (pte & 0xf) != 1 {
            break (ppn << 12) | (addr & ((1<<21)-1));
        }
        pte = crate::emu::read_memory(ppn * 4096 + ((addr >> 12) & 511) * 8);
        if (pte & 1) == 0 { return Err(fault_type); }

        ppn = pte >> 10;
        break (ppn << 12) | (addr & 4095);
    };
    if (pte & 0x40) == 0 || (write && ((pte & 0x4) == 0 || (pte & 0x80) == 0)) { return Err(fault_type); }
    return Ok(ret);
}

const CACHE_LINE_LOG2_SIZE: usize = 12;

#[inline(never)]
#[export_name = "translate_cache_miss"]
fn translate_cache_miss(ctx: &mut Context, addr: u64, write: bool) -> Result<u64, ()> {
    let idx = addr >> CACHE_LINE_LOG2_SIZE;
    let out = unsafe { &mut *ctx.cache }.access(ctx, addr, if write { AccessType::Write } else { AccessType::Read })?;
        // Refill is only possible if reside in physical memory
        let line: &mut CacheLine = &mut ctx.line[(idx & 1023) as usize];
        line.tag = idx << 1;
        line.paddr = out ^ addr;
        if !write { line.tag |= 1 }
    Ok(out)
    }

fn read_vaddr<T: Copy + CastFrom<u64>>(ctx: &mut Context, addr: u64) -> Result<T, ()> {
    let idx = addr >> CACHE_LINE_LOG2_SIZE;
    let line = &ctx.line[(idx & 1023) as usize];
    let paddr = if (line.tag >> 1) != idx {
        translate_cache_miss(ctx, addr, false)?
    } else {
        line.paddr ^ addr
    };
    Ok(unsafe { crate::emu::read_memory_unsafe(paddr) })
}

fn ptr_vaddr_x<T: Copy>(ctx: &mut Context, addr: u64) -> Result<&'static mut T, ()> {
    let idx = addr >> CACHE_LINE_LOG2_SIZE;
    let line = &ctx.line[(idx & 1023) as usize];
    let paddr = if line.tag != (idx << 1) {
        translate_cache_miss(ctx, addr, true)?
    } else {
        line.paddr ^ addr
    };
    Ok(unsafe { &mut *(paddr as *mut T) })
}

fn write_vaddr<T: Copy + Into<u64>>(ctx: &mut Context, addr: u64, value: T) -> Result<(), ()> {
    let idx = addr >> CACHE_LINE_LOG2_SIZE;
    let line = &ctx.line[(idx & 1023) as usize];
    let paddr = if line.tag != (idx << 1) {
        translate_cache_miss(ctx, addr, true)?
    } else {
        line.paddr ^ addr
    };
    Ok(unsafe { crate::emu::write_memory_unsafe(paddr, value) })
}

use fnv::FnvHashMap;
type Block = (Vec<(Op, bool)>, u64, u64);

static mut ICACHE: Option<FnvHashMap<u64, Block>> = None;

fn icache() -> &'static mut FnvHashMap<u64, Block> {
    unsafe {
        if ICACHE.is_none() {
            ICACHE = Some(FnvHashMap::default())
        }
        ICACHE.as_mut().unwrap()
    }
}

extern {
    fn send_ipi(mask: u64);
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
        3 => 0, //panic!("Ignore clear_ipi"),
        4 => {
            let mask: u64 = crate::emu::read_memory(translate(ctx, arg0, false).unwrap());
            unsafe { send_ipi(mask) };
            0
        }
        5 | 6 | 7 => {
            for l in ctx.line.iter_mut() {
                l.tag = i64::max_value() as u64;
            }
            icache().clear();
            0
        }
        8 => std::process::exit(0),
        _ => {
            panic!("unknown sbi call {}", nr);
        }
    }
}

pub fn step(ctx: &mut Context, op: &Op) -> Result<(), ()> {
    macro_rules! read_reg {
        ($rs: expr) => {{
            let rs = $rs as usize;
            if rs >= 32 { unsafe { std::hint::unreachable_unchecked() } }
            ctx.registers[rs]
        }}
    }
    macro_rules! read_32 {
        ($rs: expr) => {{
            let rs = $rs as usize;
            if rs >= 32 { unsafe { std::hint::unreachable_unchecked() } }
            ctx.registers[rs] as u32
        }}
    }
    macro_rules! write_reg {
        ($rd: expr, $expression:expr) => {{
            let rd = $rd as usize;
            let value: u64 = $expression;
            if rd >= 32 { unsafe { std::hint::unreachable_unchecked() } }
            if rd != 0 { ctx.registers[rd] = value }
        }}
    }
    macro_rules! write_32 {
        ($rd: expr, $expression:expr) => {{
            let rd = $rd as usize;
            let value: u32 = $expression;
            if rd >= 32 { unsafe { std::hint::unreachable_unchecked() } }
            if rd != 0 { ctx.registers[rd] = value as i32 as u64 }
        }}
    }
    macro_rules! read_fs {
        ($rs: expr) => {{
            let rs = $rs as usize;
            if rs >= 32 { unsafe { std::hint::unreachable_unchecked() } }
            F32::new(ctx.fp_registers[rs] as u32)
        }}
    }
    macro_rules! read_fd {
        ($rs: expr) => {{
            let rs = $rs as usize;
            if rs >= 32 { unsafe { std::hint::unreachable_unchecked() } }
            F64::new(ctx.fp_registers[rs])
        }}
    }
    macro_rules! write_fs {
        ($frd: expr, $expression:expr) => {{
            let frd = $frd as usize;
            let value: F32 = $expression;
            if frd >= 32 { unsafe { std::hint::unreachable_unchecked() } }
            ctx.fp_registers[frd] = value.0 as u64 | 0xffffffff00000000
        }}
    }
    macro_rules! write_fd {
        ($frd: expr, $expression:expr) => {{
            let frd = $frd as usize;
            let value: F64 = $expression;
            if frd >= 32 { unsafe { std::hint::unreachable_unchecked() } }
            ctx.fp_registers[frd] = value.0
        }}
    }
    macro_rules! set_rm {
        ($rm: expr) => {{
            let rm = if $rm == 0b111 { (ctx.fcsr >> 5) as u32 } else { $rm as u32 };
            if rm >= 5 { trap!(2, 0) }
            softfp::set_rounding_mode(rm);
        }}
    }
    macro_rules! clear_flags {
        () => {
            softfp::clear_exception_flag()
        };
    }
    macro_rules! update_flags {
        () => {
            ctx.fcsr |= softfp::get_exception_flag() as u64;
        };
    }
    macro_rules! trap {
        ($cause: expr, $tval: expr) => {{
            ctx.scause = $cause;
            ctx.stval = $tval;
            return Err(())
        }}
    }

    match *op {
        Op::Illegal => { trap!(2, 0) }
        /* LOAD */
        Op::Lb { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            write_reg!(rd, read_vaddr::<u8>(ctx, vaddr)? as i8 as u64);
        }
        Op::Lh { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 1 != 0 { trap!(4, vaddr) }
            write_reg!(rd, read_vaddr::<u16>(ctx, vaddr)? as i16 as u64);
        }
        Op::Lw { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 3 != 0 { trap!(4, vaddr) }
            write_reg!(rd, read_vaddr::<u32>(ctx, vaddr)? as i32 as u64);
        }
        Op::Ld { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 7 != 0 { trap!(4, vaddr) }
            write_reg!(rd, read_vaddr::<u64>(ctx, vaddr)?);
        }
        Op::Lbu { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            write_reg!(rd, read_vaddr::<u8>(ctx, vaddr)? as u64);
        }
        Op::Lhu { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 1 != 0 { trap!(4, vaddr) }
            write_reg!(rd, read_vaddr::<u16>(ctx, vaddr)? as u64);
        }
        Op::Lwu { rd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 3 != 0 { trap!(4, vaddr) }
            write_reg!(rd, read_vaddr::<u32>(ctx, vaddr)? as u64);
        }
        /* OP-IMM */
        Op::Addi { rd, rs1, imm } => write_reg!(rd, read_reg!(rs1).wrapping_add(imm as u64)),
        Op::Slli { rd, rs1, imm } => write_reg!(rd, read_reg!(rs1) << imm),
        Op::Slti { rd, rs1, imm } => write_reg!(rd, ((read_reg!(rs1) as i64) < (imm as i64)) as u64),
        Op::Sltiu { rd, rs1, imm } => write_reg!(rd, (read_reg!(rs1) < (imm as u64)) as u64),
        Op::Xori { rd, rs1, imm } => write_reg!(rd, read_reg!(rs1) ^ (imm as u64)),
        Op::Srli { rd, rs1, imm } => write_reg!(rd, read_reg!(rs1) >> imm),
        Op::Srai { rd, rs1, imm } => write_reg!(rd, ((read_reg!(rs1) as i64) >> imm) as u64),
        Op::Ori { rd, rs1, imm } => write_reg!(rd, read_reg!(rs1) | (imm as u64)),
        Op::Andi { rd, rs1, imm } => write_reg!(rd, read_reg!(rs1) & (imm as u64)),
        /* MISC-MEM */
        Op::Fence => (),
        Op::FenceI => icache().clear(),
        /* OP-IMM-32 */
        Op::Addiw { rd, rs1, imm } => write_reg!(rd, ((read_reg!(rs1) as i32).wrapping_add(imm)) as u64),
        Op::Slliw { rd, rs1, imm } => write_reg!(rd, ((read_reg!(rs1) as i32) << imm) as u64),
        Op::Srliw { rd, rs1, imm } => write_reg!(rd, (((read_reg!(rs1) as u32) >> imm) as i32) as u64),
        Op::Sraiw { rd, rs1, imm } => write_reg!(rd, ((read_reg!(rs1) as i32) >> imm) as u64),
        /* STORE */
        Op::Sb { rs1, rs2, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            write_vaddr(ctx, vaddr, read_reg!(rs2) as u8)?
        }
        Op::Sh { rs1, rs2, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 1 != 0 { trap!(5, vaddr) }
            write_vaddr(ctx, vaddr, read_reg!(rs2) as u16)?
        }
        Op::Sw { rs1, rs2, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 3 != 0 { trap!(5, vaddr) }
            write_vaddr(ctx, vaddr, read_reg!(rs2) as u32)?
        }
        Op::Sd { rs1, rs2, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 7 != 0 { trap!(5, vaddr) }
            write_vaddr(ctx, vaddr, read_reg!(rs2) as u64)?
        }
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
        Op::Addw { rd, rs1, rs2 } => write_reg!(rd, ((read_reg!(rs1) as i32).wrapping_add(read_reg!(rs2) as i32)) as u64),
        Op::Subw { rd, rs1, rs2 } => write_reg!(rd, ((read_reg!(rs1) as i32).wrapping_sub(read_reg!(rs2) as i32)) as u64),
        Op::Sllw { rd, rs1, rs2 } => write_reg!(rd, ((read_reg!(rs1) as i32) << (read_reg!(rs2) & 31)) as u64),
        Op::Srlw { rd, rs1, rs2 } => write_reg!(rd, (((read_reg!(rs1) as u32) >> (read_reg!(rs2) & 31)) as i32) as u64),
        Op::Sraw { rd, rs1, rs2 } => write_reg!(rd, ((read_reg!(rs1) as i32) >> (read_reg!(rs2) & 31)) as u64),
        /* AUIPC */
        Op::Auipc { rd, imm } => write_reg!(rd, ctx.pc.wrapping_sub(4).wrapping_add(imm as u64)),
        /* BRANCH */
        // Same as auipc, PC-relative instructions are relative to the origin pc instead of the incremented one.
        Op::Beq { rs1, rs2, imm } => {
            if read_reg!(rs1) == read_reg!(rs2) {
                ctx.pc = ctx.pc.wrapping_sub(4).wrapping_add(imm as u64);
            }
        }
        Op::Bne { rs1, rs2, imm } => {
            if read_reg!(rs1) != read_reg!(rs2) {
                ctx.pc = ctx.pc.wrapping_sub(4).wrapping_add(imm as u64);
            }
        }
        Op::Blt { rs1, rs2, imm } => {
            if (read_reg!(rs1) as i64) < (read_reg!(rs2) as i64) {
                ctx.pc = ctx.pc.wrapping_sub(4).wrapping_add(imm as u64);
            }
        }
        Op::Bge { rs1, rs2, imm } => {
            if (read_reg!(rs1) as i64) >= (read_reg!(rs2) as i64) {
                ctx.pc = ctx.pc.wrapping_sub(4).wrapping_add(imm as u64);
            }
        }
        Op::Bltu { rs1, rs2, imm } => {
            if read_reg!(rs1) < read_reg!(rs2) {
                ctx.pc = ctx.pc.wrapping_sub(4).wrapping_add(imm as u64);
            }
        }
        Op::Bgeu { rs1, rs2, imm } => {
            if read_reg!(rs1) >= read_reg!(rs2) {
                ctx.pc = ctx.pc.wrapping_sub(4).wrapping_add(imm as u64);
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
            ctx.pc = ctx.pc.wrapping_sub(4).wrapping_add(imm as u64);
        }
        /* SYSTEM */
        Op::Ecall =>
            if ctx.prv == 0 {
                trap!(8, 0)
            } else {
                ctx.registers[10] = sbi_call(
                    ctx,
                    ctx.registers[17],
                    ctx.registers[10],
                )
            }
        Op::Ebreak => trap!(3, 0),
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

        /* F-extension */
        Op::Flw { frd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 3 != 0 { trap!(4, vaddr) }
            write_fs!(frd, F32::new(read_vaddr::<u32>(ctx, vaddr)?));
        }
        Op::Fsw { rs1, frs2, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 3 != 0 { trap!(5, vaddr) }
            write_vaddr(ctx, vaddr, read_fs!(frs2).0)?
        }
        Op::FaddS { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, read_fs!(frs1) + read_fs!(frs2));
            update_flags!();
        }
        Op::FsubS { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, read_fs!(frs1) - read_fs!(frs2));
            update_flags!();
        }
        Op::FmulS { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, read_fs!(frs1) * read_fs!(frs2));
            update_flags!();
        }
        Op::FdivS { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, read_fs!(frs1) / read_fs!(frs2));
            update_flags!();
        }
        Op::FsqrtS { frd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, read_fs!(frs1).square_root());
            update_flags!();
        }
        Op::FsgnjS { frd, frs1, frs2 } => write_fs!(frd, read_fs!(frs1).copy_sign(read_fs!(frs2))),
        Op::FsgnjnS { frd, frs1, frs2 } => write_fs!(frd, read_fs!(frs1).copy_sign_negated(read_fs!(frs2))),
        Op::FsgnjxS { frd, frs1, frs2 } => write_fs!(frd, read_fs!(frs1).copy_sign_xored(read_fs!(frs2))),
        Op::FminS { frd, frs1, frs2 } => {
            clear_flags!();
            write_fs!(frd, F32::min(read_fs!(frs1), read_fs!(frs2)));
            update_flags!();
        }
        Op::FmaxS { frd, frs1, frs2 } => {
            clear_flags!();
            write_fs!(frd, F32::max(read_fs!(frs1), read_fs!(frs2)));
            update_flags!();
        }
        Op::FcvtWS { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_32!(rd, read_fs!(frs1).convert_to_sint::<u32>());
            update_flags!();
        }
        Op::FcvtWuS { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_32!(rd, read_fs!(frs1).convert_to_uint::<u32>());
            update_flags!();
        }
        Op::FcvtLS { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_reg!(rd, read_fs!(frs1).convert_to_sint::<u64>());
            update_flags!();
        }
        Op::FcvtLuS { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_reg!(rd, read_fs!(frs1).convert_to_uint::<u64>());
            update_flags!();
        }
        Op::FmvXW { rd, frs1 } => {
            write_32!(rd, read_fs!(frs1).0);
        }
        Op::FclassS { rd, frs1 } => {
            write_reg!(rd, 1 << read_fs!(frs1).classify());
        }
        Op::FeqS { rd, frs1, frs2 } => {
            write_reg!(rd, (read_fs!(frs1) == read_fs!(frs2)) as u64)
        }
        Op::FltS { rd, frs1, frs2 } => {
            clear_flags!();
            write_reg!(rd, (read_fs!(frs1) < read_fs!(frs2)) as u64);
            update_flags!();
        }
        Op::FleS { rd, frs1, frs2 } => {
            clear_flags!();
            write_reg!(rd, (read_fs!(frs1) <= read_fs!(frs2)) as u64);
            update_flags!();
        }
        Op::FcvtSW { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, F32::convert_from_sint::<u32>(read_32!(rs1)));
            update_flags!();
        }
        Op::FcvtSWu { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, F32::convert_from_uint::<u32>(read_32!(rs1)));
            update_flags!();
        }
        Op::FcvtSL { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, F32::convert_from_sint::<u64>(read_reg!(rs1)));
            update_flags!();
        }
        Op::FcvtSLu { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, F32::convert_from_uint::<u64>(read_reg!(rs1)));
            update_flags!();
        }
        Op::FmvWX { frd, rs1 } => {
            write_fs!(frd, F32::new(read_32!(rs1)));
        }
        Op::FmaddS { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, F32::fused_multiply_add(read_fs!(frs1), read_fs!(frs2), read_fs!(frs3)));
            update_flags!();
        }
        Op::FmsubS { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, F32::fused_multiply_add(read_fs!(frs1), read_fs!(frs2), -read_fs!(frs3)));
            update_flags!();
        }
        Op::FnmsubS { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, F32::fused_multiply_add(-read_fs!(frs1), read_fs!(frs2), read_fs!(frs3)));
            update_flags!();
        }
        Op::FnmaddS { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, -F32::fused_multiply_add(read_fs!(frs1), read_fs!(frs2), read_fs!(frs3)));
            update_flags!();
        }

        /* D-extension */
        Op::Fld { frd, rs1, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 3 != 0 { trap!(4, vaddr) }
            write_fd!(frd, F64::new(read_vaddr::<u64>(ctx, vaddr)?));
        }
        Op::Fsd { rs1, frs2, imm } => {
            let vaddr = read_reg!(rs1).wrapping_add(imm as u64);
            if vaddr & 7 != 0 { trap!(5, vaddr) }
            write_vaddr(ctx, vaddr, read_fd!(frs2).0)?
        }
        Op::FaddD { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, read_fd!(frs1) + read_fd!(frs2));
            update_flags!();
        }
        Op::FsubD { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, read_fd!(frs1) - read_fd!(frs2));
            update_flags!();
        }
        Op::FmulD { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, read_fd!(frs1) * read_fd!(frs2));
            update_flags!();
        }
        Op::FdivD { frd, frs1, frs2, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, read_fd!(frs1) / read_fd!(frs2));
            update_flags!();
        }
        Op::FsqrtD { frd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, read_fd!(frs1).square_root());
            update_flags!();
        }
        Op::FsgnjD { frd, frs1, frs2 } => write_fd!(frd, read_fd!(frs1).copy_sign(read_fd!(frs2))),
        Op::FsgnjnD { frd, frs1, frs2 } => write_fd!(frd, read_fd!(frs1).copy_sign_negated(read_fd!(frs2))),
        Op::FsgnjxD { frd, frs1, frs2 } => write_fd!(frd, read_fd!(frs1).copy_sign_xored(read_fd!(frs2))),
        Op::FminD { frd, frs1, frs2 } => {
            clear_flags!();
            write_fd!(frd, F64::min(read_fd!(frs1), read_fd!(frs2)));
            update_flags!();
        }
        Op::FmaxD { frd, frs1, frs2 } => {
            clear_flags!();
            write_fd!(frd, F64::max(read_fd!(frs1), read_fd!(frs2)));
            update_flags!();
        }
        Op::FcvtSD { frd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fs!(frd, read_fd!(frs1).convert_format());
            update_flags!();
        }
        Op::FcvtDS { frd, frs1, .. } => {
            clear_flags!();
            write_fd!(frd, read_fs!(frs1).convert_format());
            update_flags!();
        }
        Op::FcvtWD { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_32!(rd, read_fd!(frs1).convert_to_sint::<u32>());
            update_flags!();
        }
        Op::FcvtWuD { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_32!(rd, read_fd!(frs1).convert_to_uint::<u32>());
            update_flags!();
        }
        Op::FcvtLD { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_reg!(rd, read_fd!(frs1).convert_to_sint::<u64>());
            update_flags!();
        }
        Op::FcvtLuD { rd, frs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_reg!(rd, read_fd!(frs1).convert_to_uint::<u64>());
            update_flags!();
        }
        Op::FmvXD { rd, frs1 } => {
            write_reg!(rd, read_fd!(frs1).0);
        }
        Op::FclassD { rd, frs1 } => {
            write_reg!(rd, 1 << read_fd!(frs1).classify());
        }
        Op::FeqD { rd, frs1, frs2 } => {
            write_reg!(rd, (read_fd!(frs1) == read_fd!(frs2)) as u64)
        }
        Op::FltD { rd, frs1, frs2 } => {
            clear_flags!();
            write_reg!(rd, (read_fd!(frs1) < read_fd!(frs2)) as u64);
            update_flags!();
        }
        Op::FleD { rd, frs1, frs2 } => {
            clear_flags!();
            write_reg!(rd, (read_fd!(frs1) <= read_fd!(frs2)) as u64);
            update_flags!();
        }
        Op::FcvtDW { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, F64::convert_from_sint::<u32>(read_32!(rs1)));
            update_flags!();
        }
        Op::FcvtDWu { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, F64::convert_from_uint::<u32>(read_32!(rs1)));
            update_flags!();
        }
        Op::FcvtDL { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, F64::convert_from_sint::<u64>(read_reg!(rs1)));
            update_flags!();
        }
        Op::FcvtDLu { frd, rs1, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, F64::convert_from_uint::<u64>(read_reg!(rs1)));
            update_flags!();
        }
        Op::FmvDX { frd, rs1 } => {
            write_fd!(frd, F64::new(read_reg!(rs1)));
        }
        Op::FmaddD { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, F64::fused_multiply_add(read_fd!(frs1), read_fd!(frs2), read_fd!(frs3)));
            update_flags!();
        }
        Op::FmsubD { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, F64::fused_multiply_add(read_fd!(frs1), read_fd!(frs2), -read_fd!(frs3)));
            update_flags!();
        }
        Op::FnmsubD { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, F64::fused_multiply_add(-read_fd!(frs1), read_fd!(frs2), read_fd!(frs3)));
            update_flags!();
        }
        Op::FnmaddD { frd, frs1, frs2, frs3, rm } => {
            set_rm!(rm);
            clear_flags!();
            write_fd!(frd, -F64::fused_multiply_add(read_fd!(frs1), read_fd!(frs2), read_fd!(frs3)));
            update_flags!();
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

        /* A-extension */
        // Stub implementations. Single thread only.
        Op::LrW { rd, rs1 } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 { trap!(5, addr) }
            let paddr = ptr_vaddr_x::<u32>(ctx, addr)?;
            let value = *paddr as i32 as u64;
            write_reg!(rd, value);
            ctx.lr_addr = addr;
            ctx.lr_value = value;
        }
        Op::LrD { rd, rs1 } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 { trap!(5, addr) }
            let paddr = ptr_vaddr_x::<u64>(ctx, addr)?;
            let value = *paddr;
            write_reg!(rd, value);
            ctx.lr_addr = addr;
            ctx.lr_value = value;
        }
        Op::ScW { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 { trap!(5, addr) }
            let paddr = ptr_vaddr_x::<u32>(ctx, addr)?;
            if addr != ctx.lr_addr || *paddr != ctx.lr_value as u32 {
                write_reg!(rd, 1)
            } else {
                *paddr = read_reg!(rs2) as u32;
                write_reg!(rd, 0)
            }
        }
        Op::ScD { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 { trap!(5, addr) }
            let paddr = ptr_vaddr_x::<u64>(ctx, addr)?;
            if addr != ctx.lr_addr || *paddr != ctx.lr_value {
                write_reg!(rd, 1)
            } else {
                *paddr = read_reg!(rs2);
                write_reg!(rd, 0)
            }
        }
        Op::AmoswapW { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<u32>(ctx, addr)?;
            if rd != 0 {
                write_reg!(rd, *ptr as i32 as u64);
            }
            *ptr = src;
        }
        Op::AmoswapD { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<u64>(ctx, addr)?;
            if rd != 0 {
                write_reg!(rd, *ptr);
            }
            *ptr = src;
        }
        Op::AmoaddW { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<u32>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current as i32 as u64);
            *ptr = current.wrapping_add(src);
        }
        Op::AmoaddD { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<u64>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current);
            *ptr = current.wrapping_add(src);
        }
        Op::AmoandW { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<u32>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current as i32 as u64);
            *ptr = current & src;
        }
        Op::AmoandD { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<u64>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current);
            *ptr = current & src;
        }
        Op::AmoorW { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<u32>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current as i32 as u64);
            *ptr = current | src;
        }
        Op::AmoorD { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<u64>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current);
            *ptr = current | src;
        }
        Op::AmoxorW { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<u32>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current as i32 as u64);
            *ptr = current ^ src;
        }
        Op::AmoxorD { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<u64>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current);
            *ptr = current ^ src;
        }
        Op::AmominW { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<u32>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current as i32 as u64);
            *ptr = i32::min(current as i32, src as i32) as u32;
        }
        Op::AmominD { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<u64>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current);
            *ptr = i64::min(current as i64, src as i64) as u64;
        }
        Op::AmomaxW { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<u32>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current as i32 as u64);
            *ptr = i32::max(current as i32, src as i32) as u32;
        }
        Op::AmomaxD { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<u64>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current);
            *ptr = i64::max(current as i64, src as i64) as u64;
        }
        Op::AmominuW { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<u32>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current as i32 as u64);
            *ptr = u32::min(current, src);
        }
        Op::AmominuD { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<u64>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current);
            *ptr = u64::min(current, src);
        }
        Op::AmomaxuW { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 3 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2) as u32;
            let ptr = ptr_vaddr_x::<u32>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current as i32 as u64);
            *ptr = u32::max(current, src);
        }
        Op::AmomaxuD { rd, rs1, rs2 } => {
            let addr = read_reg!(rs1);
            if addr & 7 != 0 { trap!(5, addr) }
            let src = read_reg!(rs2);
            let ptr = ptr_vaddr_x::<u64>(ctx, addr)?;
            let current = *ptr;
            write_reg!(rd, current);
            *ptr = u64::max(current, src);
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
            icache().clear()
        }
    }
    Ok(())
}

fn run_instr(ctx: &mut Context) -> Result<(), ()> {
    let pc = ctx.pc;
    let mut phys_pc = match translate(ctx, pc, false) {
        Ok(pc) => pc,
        Err(ex) => {
            ctx.scause = ex;
            ctx.stval = pc;
            ctx.cycle += 1;
            return Err(())
        }
    };
    // Ignore error in this case
    let phys_pc_next = match translate(ctx, (pc &! 4095) + 4096, false) {
        Ok(pc) => pc,
        Err(_) => 0,
    };
    let (op, c) = super::decode::decode_instr(&mut phys_pc, phys_pc_next);
    ctx.pc += if c { 2 } else { 4 };
    ctx.instret += 1;
    ctx.cycle += 1;
    match step(ctx, &op) {
        Ok(()) => (),
        Err(()) => {
            ctx.pc -= if c { 2 } else { 4 };
            ctx.instret -= 1;
            return Err(());
        }
    }
    Ok(())
}

fn run_block(ctx: &mut Context) -> Result<(), ()> {
    let pc = ctx.pc;
    let mut phys_pc = match translate(ctx, pc, false) {
        Ok(pc) => pc,
        Err(ex) => {
            ctx.scause = ex;
            ctx.stval = pc;
            return Err(())
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
    ctx.cycle += vec.len() as u64;

    for i in 0..vec.len() {
        let (ref inst, _) = vec[i];
        match step(ctx, inst) {
            Ok(()) => (),
            Err(()) => {
                // Adjust pc and instret by iterating through remaining instructions.
                for j in i..vec.len() {
                    ctx.pc -= if vec[j].1 { 2 } else { 4 };
                }
                ctx.instret -= (vec.len() - i) as u64;
                ctx.cycle -= (vec.len() - i - 1) as u64;
                return Err(());
            }
        }
    }
    Ok(())
}

/// Trigger a trap. pc must be already adjusted properly before calling.
pub fn trap(ctx: &mut Context) {
    if crate::get_flags().user_only {
        if ctx.scause == 8 {
            ctx.registers[10] = unsafe { crate::emu::syscall(
                ctx.registers[17],
                ctx.registers[10],
                ctx.registers[11],
                ctx.registers[12],
                ctx.registers[13],
                ctx.registers[14],
                ctx.registers[15],
            ) };
            ctx.pc += 4;
            return;
        }
        eprintln!("unhandled trap {}", ctx.scause);
        eprintln!("pc  = {:16x}  ra  = {:16x}", ctx.pc, ctx.registers[1]);
        for i in (2..32).step_by(2) {
            eprintln!(
                "{:-3} = {:16x}  {:-3} = {:16x}",
                super::disasm::REG_NAMES[i], ctx.registers[i],
                super::disasm::REG_NAMES[i + 1], ctx.registers[i + 1]
            );
        }
        std::process::exit(1);
    }

    ctx.sepc = ctx.pc;

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

pub fn run_instr_ex(ctx: &mut Context) {
    match run_instr(ctx) {
        Ok(()) => (),
        Err(()) => return trap(ctx),
    }
    if ctx.cycle >= ctx.timecmp {
        ctx.sip |= 32;
        ctx.pending = if (ctx.sstatus & 0x2) != 0 { ctx.sip & ctx.sie } else { 0 };
    }
    if ctx.pending != 0 {
        // The highest set bit of ctx.pending
        let pending = 63 - ctx.pending.leading_zeros() as u64;
        ctx.sip &= !(1 << pending);
        // The highest bit of cause indicates this is an interrupt
        ctx.scause = (1 << 63) | pending;
        ctx.stval = 0;
        trap(ctx);
    }
}

pub fn run_block_ex(ctx: &mut Context) {
    match run_block(ctx) {
        Ok(()) => (),
        Err(()) => return trap(ctx),
    }
    if ctx.cycle >= ctx.timecmp {
        ctx.sip |= 32;
        ctx.pending = if (ctx.sstatus & 0x2) != 0 { ctx.sip & ctx.sie } else { 0 };
    }
    if ctx.pending != 0 {
        // The highest set bit of ctx.pending
        let pending = 63 - ctx.pending.leading_zeros() as u64;
        ctx.sip &= !(1 << pending);
        // The highest bit of cause indicates this is an interrupt
        ctx.scause = (1 << 63) | pending;
        ctx.stval = 0;
        trap(ctx);
    }
}
