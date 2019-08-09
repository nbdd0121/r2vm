#[macro_use]
extern crate log;
#[macro_use]
extern crate lazy_static;
extern crate spin;
extern crate pretty_env_logger;
extern crate rand;
extern crate fnv;
extern crate byteorder;

extern crate softfp;
extern crate p9;
extern crate x86;
extern crate riscv;
extern crate fdt;

pub mod io;
#[macro_use]
pub mod util;
pub mod emu;
pub mod fiber;

use std::ffi::{CString};
use std::ptr;

macro_rules! usage_string {() => ("Usage: {} [options] program [arguments...]
Options:
  --no-direct-memory    Disable generation of memory access instruction, use
                        call to helper function instead.
  --strace              Log system calls.
  --disassemble         Log decoded instructions.
  --engine=interpreter  Use interpreter instead of dynamic binary translator.
  --engine=dbt          Use simple binary translator instead of IR-based
                        optimising binary translator.
  --with-instret        Enable precise instret updating in binary translated
                        code.
  --monitor-performance Display metrics about performance in compilation phase.
  --thread              Use non-accurate threaded mode for execution.
  --sysroot             Change the sysroot to a non-default value.
  --init                Specify the init program for full system emulation.
  --help                Display this help message.
")}

pub struct Flags {

    // Whether direct memory access or call to helper should be generated for guest memory access.
    no_direct_memory_access: bool,

    // A flag to determine whether to trace all system calls. If true then all guest system calls will be logged.
    strace: bool,

    // A flag to determine whether to print instruction out when it is decoded.
    disassemble: bool,

    // A flag to determine whether instret should be updated precisely in binary translated code.
    // XXX: Not currently used
    no_instret: bool,

    // Whether compilation performance counters should be enabled.
    // XXX: Not currently used
    monitor_performance: bool,

    // The actual path of the executable. Needed to redirect /proc/self/*
    exec_path: *const i8,

    // Path of sysroot. When the guest application tries to open a file, and the corresponding file exists in sysroot,
    // it will be redirected.
    sysroot: Option<String>,

    // If we are only emulating userspace code
    user_only: bool,

    // Path of init to execute. If supplied, it will be included in Linux bootcmd
    init: Option<String>,

    // Whether threaded mode should be used
    thread: std::sync::atomic::AtomicBool,
}

static mut FLAGS: Flags = Flags {
    no_direct_memory_access: true,
    strace: false,
    disassemble: false,
    no_instret: true,
    monitor_performance: false,
    exec_path: ptr::null(),
    sysroot: None,
    user_only: true,
    init: None,
    thread: std::sync::atomic::AtomicBool::new(false),
};

pub fn get_flags() -> &'static Flags {
    unsafe { &FLAGS }
}

pub fn threaded() -> bool {
    get_flags().thread.load(std::sync::atomic::Ordering::Relaxed)
}

pub static mut CONTEXTS: &'static mut [*mut emu::interp::Context] = &mut [];

pub fn shared_context(id: usize) -> &'static emu::interp::SharedContext {
    unsafe { &(*CONTEXTS[id]).shared }
}

pub fn core_count() -> usize {
    let cnt = unsafe { CONTEXTS.len() };
    assert_ne!(cnt, 0);
    cnt
}

static mut EVENT_LOOP: *const emu::EventLoop = std::ptr::null_mut();

pub fn event_loop() -> &'static emu::EventLoop {
    unsafe { &*EVENT_LOOP }
}

extern {
    fn fiber_interp_run();
}

pub fn main() {
    // Top priority: set up page fault handlers so safe_memory features will work.
    emu::signal::init();
    pretty_env_logger::init();

    let mut args = std::env::args();

    // Ignore interpreter name
    let mut item = args.next();
    let interp_name = item.unwrap();

    let mut sysroot = String::from("/opt/riscv/sysroot");

    item = args.next();
    while let Some(ref arg) = item {
        // We've parsed all arguments. This indicates the name of the executable.
        if !arg.starts_with('-') {
            break;
        }

        match arg.as_str() {
            "--no-direct-memory" => unsafe {
                FLAGS.no_direct_memory_access = true;
            }
            "--strace" => unsafe {
                FLAGS.strace = true;
            }
            "--disassemble" => unsafe {
                FLAGS.disassemble = true;
            }
            "--engine=dbt" => panic!("engine dbt"),
            "--engine=interpreter" => panic!("engine int"),
        	"--with-instret" => unsafe {
                FLAGS.no_instret = false;
            }
            "--monitor-performance" => unsafe {
                FLAGS.monitor_performance = true;
            }
            "--thread" => unsafe {
                FLAGS.thread.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            "--help" => {
                eprintln!(usage_string!(), interp_name);
                std::process::exit(0);
            }
            _ => if arg.starts_with("--sysroot=") {
                let path_slice = &arg["--sysroot=".len()..];
                sysroot = path_slice.to_owned();
            } else if arg.starts_with("--init=") {
                let path_slice = &arg["--init=".len()..];
                unsafe { FLAGS.init = Some(path_slice.to_owned()) }
            } else {
                eprintln!("{}: unrecognized option '{}'", interp_name, arg);
                std::process::exit(1);
            }
        }

        item = args.next();
    }

    if item.is_none() {
        eprintln!(usage_string!(), interp_name);
        std::process::exit(1);
    }

    let program_name = item.unwrap();

    let cprogram_name = CString::new(program_name.as_str()).unwrap();
    unsafe {
        FLAGS.exec_path = cprogram_name.as_ptr();
    }
    std::mem::forget(cprogram_name);
    unsafe { FLAGS.sysroot = Some(sysroot) };

    let loader = emu::loader::Loader::new(program_name.as_ref()).unwrap();
    // Simple guess: If not elf, then we load it as if it is a flat binary kernel
    unsafe { FLAGS.user_only = match loader.validate_elf() {
        Ok(_) => !loader.guess_kernel(),
        Err(_) => false,
    }; }

    // Create fibers for all threads
    let mut fibers = Vec::new();
    let mut contexts = Vec::new();

    let num_cores = if get_flags().user_only { 1 } else { 4 };

    // Create a fiber for event-driven simulation, e.g. timer, I/O
    let event_fiber = fiber::Fiber::new();
    unsafe { std::ptr::write(event_fiber.data_pointer(), emu::EventLoop::new()) };
    unsafe { EVENT_LOOP = event_fiber.data_pointer() }
    fibers.push(event_fiber);

    for i in 0..num_cores {
        let mut newctx = emu::interp::Context {
            shared: emu::interp::SharedContext::new(),
            registers: [0xCCCCCCCCCCCCCCCC; 32],
            fp_registers: [0xFFFFFFFFFFFFFFFF; 32],
            fcsr: 0,
            instret: 0,
            lr_addr: 0,
            lr_value: 0,
            // FPU turned on by default
            sstatus: 0x6000,
            scause: 0,
            sepc: 0,
            stval: 0,
            satp: 0,
            sie: 0,
            sscratch: 0,
            stvec: 0,
            timecmp: u64::max_value(),
            // These are set by setup_mem, so we don't really care now.
            pc: 0,
            prv: 0,
            hartid: i as u64,
            minstret: 0,
        };
        // x0 must always be 0
        newctx.registers[0] = 0;

        let fiber = fiber::Fiber::new();
        let ptr = fiber.data_pointer();
        unsafe { *ptr = newctx }
        contexts.push(ptr);
        fibers.push(fiber);
    }

    unsafe { CONTEXTS = Box::leak(contexts.into_boxed_slice()) }

    // These should only be initialised for full-system emulation
    if !get_flags().user_only {
        io::console::console_init();
        emu::init();
    }

    // Load the program
    unsafe { emu::loader::load(&loader, &mut std::iter::once(program_name).chain(args)) };
    std::mem::drop(loader);

    fibers[0].set_fn(|| {
        let this: &emu::EventLoop = unsafe { &*fiber::Fiber::scratchpad() };
        this.event_loop()
    });
    for fiber in &mut fibers[1..] {
        fiber.set_fn(|| unsafe{fiber_interp_run()});
    }

    if !crate::threaded() {
        // Run multiple fibers in the same group.
        let mut group = fiber::FiberGroup::new();
        for fiber in fibers {
            group.add(fiber);
        }
        group.run();
    } else {
        // Run one fiber per thread.
        let handles: Vec<_> = fibers.into_iter().map(|fiber| {
            std::thread::spawn(move || {
                let mut group = fiber::FiberGroup::new();
                group.add(fiber);
                group.run();
            })
        }).collect();
        for handle in handles { handle.join().unwrap(); }
    }
}

pub fn print_stats_and_exit(code: i32) -> ! {
    unsafe {
        println!("TIME = {:?}", crate::util::cpu_time());
        println!("CYCLE = {:x}", crate::event_loop().cycle());
        for i in 0..crate::CONTEXTS.len() {
            let ctx = &*crate::CONTEXTS[i];
            println!("Hart {}: INSTRET = {:x}, MINSTRET = {:x}", i, ctx.instret, ctx.minstret);
        }
    }
    std::process::exit(code)
}
