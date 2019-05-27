extern {
    fn memory_probe_read(ptr: usize) -> bool;
    fn memory_probe_write(ptr: usize) -> bool;
}

pub fn probe_read<T>(ptr: *const T) -> Result<(), ()> {
    if unsafe { memory_probe_read(ptr as usize) } { Err(()) } else { Ok(()) }
}

pub fn probe_write<T>(ptr: *mut T) -> Result<(), ()> {
    if unsafe { memory_probe_write(ptr as usize) } { Err(()) } else { Ok(()) }
}


extern {
    static memory_probe_start: u8;
    static memory_probe_end: u8;
}

unsafe extern "C" fn handle_fault(_: libc::c_int, _: *mut libc::siginfo_t, ctx: *mut libc::ucontext_t) {
    // Fault within the probe
    let current_ip = (*ctx).uc_mcontext.gregs[libc::REG_RIP as usize];
    if current_ip >= (&memory_probe_start as *const _ as usize as i64) &&
       current_ip < (&memory_probe_end as *const _ as usize as i64) {

        // If we fault with in the probe, let the probe return instead.
        // Pop out rip from the stack and set it.
        let rsp = (*ctx).uc_mcontext.gregs[libc::REG_RSP as usize];
        (*ctx).uc_mcontext.gregs[libc::REG_RIP as usize] = *(rsp as usize as *mut i64);
        (*ctx).uc_mcontext.gregs[libc::REG_RSP as usize] = rsp + 8;

        // Set rax to 1 to signal failure
        (*ctx).uc_mcontext.gregs[libc::REG_RAX as usize] = 1;
        return;
    }

    panic!("unexpected illegal memory access");
}

/// Setup signal handler necessary for probes to work
pub fn init() {
    unsafe {
        let mut act: libc::sigaction = std::mem::zeroed();
        act.sa_sigaction = handle_fault as usize;
        act.sa_flags = libc::SA_SIGINFO;
        libc::sigaction(libc::SIGSEGV, &act, std::ptr::null_mut());
        libc::sigaction(libc::SIGBUS, &act, std::ptr::null_mut());
    }
}
