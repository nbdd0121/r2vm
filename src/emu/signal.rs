use crate::x86::{Register, Operand, Memory, Op, Location, Size};

const REG_RAX: usize = libc::REG_RAX as usize;
const REG_RDX: usize = libc::REG_RDX as usize;
const REG_RSP: usize = libc::REG_RSP as usize;
const REG_RIP: usize = libc::REG_RIP as usize;

const REG_LIST: [usize; 16] = [
    REG_RAX, libc::REG_RCX as usize, REG_RDX, libc::REG_RBX as usize,
    REG_RSP, libc::REG_RBP as usize, libc::REG_RSI as usize, libc::REG_RDI as usize,
    libc::REG_R8  as usize, libc::REG_R9  as usize, libc::REG_R10 as usize, libc::REG_R11 as usize,
    libc::REG_R12 as usize, libc::REG_R13 as usize, libc::REG_R14 as usize, libc::REG_R15 as usize,
];

struct MemReader(usize);

impl Iterator for MemReader {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        let ptr = self.0;
        self.0 += 1;
        Some(unsafe { *(ptr as *const u8) })
    }
}

/// Given a `Memory`, evaluate the memory location it points to.
fn eval_memory_location(ctx: &libc::ucontext_t, mem: &Memory) -> usize {
    // Sign-extend displacement to 64-bit.
    let mut address = mem.displacement as u64;
    if let Some(base) = mem.base {
        address = address.wrapping_add(ctx.uc_mcontext.gregs[REG_LIST[(base as u8 & 15) as usize]] as u64);
    }
    if let Some((index, scale)) = mem.index {
        address = address.wrapping_add(ctx.uc_mcontext.gregs[REG_LIST[(index as u8 & 15) as usize]] as u64) * scale as u64;
    }
    address as usize
}

/// Given a `Location`, read its content from register or memory and zero-extend to u64.
unsafe fn read_location(ctx: &libc::ucontext_t, loc: &Location) -> u64 {
    match loc {
        &Location::Reg(reg) => {
            let value = ctx.uc_mcontext.gregs[REG_LIST[(reg as u8 & 15) as usize]] as u64;
            match reg.size() {
                Size::Qword => value,
                Size::Dword => value as u32 as u64,
                Size::Word => value as u16 as u64,
                Size::Byte => (if (reg as u8) >= Register::AH as u8 && (reg as u8) <= Register::BH as u8 {
                    (ctx.uc_mcontext.gregs[REG_LIST[(reg as u8 & 7) as usize]] >> 8) as u8
                } else {
                    value as u8
                }) as u64,
            }
        }
        Location::Mem(mem) => {
            let address = eval_memory_location(ctx, mem);
            match mem.size {
                Size::Qword => *(address as *const u64),
                Size::Dword => *(address as *const u32) as u64,
                Size::Word => *(address as *const u16) as u64,
                Size::Byte => *(address as *const u8) as u64,
            }
        }
    }
}

/// Given a `Location`, write content to the register or memory.
unsafe fn write_location(ctx: &mut libc::ucontext_t, loc: &Location, value: u64) {
    match loc {
        &Location::Reg(reg) => {
            let slot = &mut ctx.uc_mcontext.gregs[REG_LIST[(reg as u8 & 15) as usize]];
            match reg.size() {
                Size::Qword => *slot = value as i64,
                // zero-extend when writing dword
                Size::Dword => *slot = value as u32 as i64,
                // do not alter higher values when writing word/byte
                Size::Word => *slot = (*slot &! 0xFFFF) | (value as i64 & 0xFFFF),
                Size::Byte => if (reg as u8) >= Register::AH as u8 && (reg as u8) <= Register::BH as u8 {
                    let slot = &mut ctx.uc_mcontext.gregs[REG_LIST[(reg as u8 & 7) as usize]];
                    *slot = (*slot &! 0xFF00) | (value as i64 & 0xFF) << 8
                } else {
                    *slot = (*slot &! 0xFF) | (value as i64 & 0xFF)
                },
            }
        }
        Location::Mem(mem) => {
            let address = eval_memory_location(ctx, mem);
            match mem.size {
                Size::Qword => *(address as *mut u64) = value,
                Size::Dword => *(address as *mut u32) = value as u32,
                Size::Word => *(address as *mut u16) = value as u16,
                Size::Byte => *(address as *mut u8) = value as u8,
            }
        }
    }
}

unsafe extern "C" fn handle_fpe(_: libc::c_int, _: &mut libc::siginfo_t, ctx: &mut libc::ucontext_t) {
    let current_ip = ctx.uc_mcontext.gregs[REG_RIP];

    // Decode the faulting instruction
    let mut reader = MemReader(current_ip as usize);
    let op = {
        let mut decoder = crate::x86::Decoder::new(&mut reader);
        decoder.op()
    };

    let opr = match op {
        Op::Div(opr) |
        Op::Idiv(opr) => opr,
        _ => unimplemented!(),
    };

    // Retrieve the value of the divisor.
    let opsize = opr.size();
    let divisor = read_location(&*ctx, &opr);

    // Retrive dividend. Note that technically RDX is also dividend, but we assume it is always sign/zero-extended.
    let mut dividend = ctx.uc_mcontext.gregs[REG_RAX] as u64;
    if opsize == Size::Dword { dividend = dividend as u32 as u64 }

    if divisor == 0 {
        // For divide by zero, per RISC-V we set quotient to -1 and remainder to dividend.
        ctx.uc_mcontext.gregs[REG_RAX] = if opsize == Size::Qword { -1 } else { -1i32 as u32 as i64 };
        ctx.uc_mcontext.gregs[REG_RDX] = dividend as i64;
    } else {
        // Integer division overflow. Per RISC-V we set quotient to dividend and remainder to 0.
        ctx.uc_mcontext.gregs[REG_RAX] = dividend as i64;
        ctx.uc_mcontext.gregs[REG_RDX] = 0;
    }

    // Advance to next ip.
    ctx.uc_mcontext.gregs[REG_RIP] = reader.0 as i64;
}

extern {
    static memory_probe_start: u8;
    static memory_probe_end: u8;
}

pub unsafe extern "C" fn handle_segv(_: libc::c_int, _: &mut libc::siginfo_t, ctx: &mut libc::ucontext_t) -> () {
    let current_ip = ctx.uc_mcontext.gregs[REG_RIP];

    // Fault within the probe
    if current_ip >= (&memory_probe_start as *const _ as usize as i64) &&
       current_ip < (&memory_probe_end as *const _ as usize as i64) {

        // If we fault with in the probe, let the probe return instead.
        // Pop out rip from the stack and set it.
        let rsp = ctx.uc_mcontext.gregs[REG_RSP];
        ctx.uc_mcontext.gregs[REG_RIP] = *(rsp as usize as *mut i64);
        ctx.uc_mcontext.gregs[REG_RSP] = rsp + 8;

        // Set rax to 1 to signal failure
        ctx.uc_mcontext.gregs[REG_RAX] = 1;
        return;
    }

    // Decode the faulting instruction
    let mut reader = MemReader(current_ip as usize);
    let op = {
        let mut decoder = crate::x86::Decoder::new(&mut reader);
        decoder.op()
    };

    // Replay the read/write, as if they are accessing directly to guest physical memory
    match op {
        Op::Mov(Location::Reg(reg), Operand::Mem(mem)) => {
            let address = eval_memory_location(ctx, &mem);
            let data = crate::emu::phys_read(address as _, reg.size().bytes() as u32);
            write_location(ctx, &Location::Reg(reg), data);
        }
        Op::Mov(Location::Mem(mem), Operand::Reg(reg)) => {
            let address = eval_memory_location(ctx, &mem);
            let data = read_location(ctx, &Location::Reg(reg));
            crate::emu::phys_write(address as _, data, reg.size().bytes() as u32);
        }
        _ => unimplemented!(),
    };

    // Advance to next ip.
    ctx.uc_mcontext.gregs[REG_RIP] = reader.0 as i64;
}

pub fn init() {
    unsafe {
        let mut act: libc::sigaction = std::mem::zeroed();
        act.sa_sigaction = handle_fpe as usize;
        act.sa_flags = libc::SA_SIGINFO;
        libc::sigaction(libc::SIGFPE, &act, std::ptr::null_mut());

        act.sa_sigaction = handle_segv as usize;
        act.sa_flags = libc::SA_SIGINFO;
        libc::sigaction(libc::SIGSEGV, &act, std::ptr::null_mut());
        libc::sigaction(libc::SIGBUS, &act, std::ptr::null_mut());
    }
}