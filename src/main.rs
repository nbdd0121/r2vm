#[macro_use]
extern crate log;

pub mod io;
#[macro_use]
pub mod util;
pub mod emu;
pub mod fiber;
pub mod config;

use std::ffi::{CString};
use util::RoCell;

macro_rules! usage_string {() => ("Usage: {} [options] program [arguments...]
Options:
  --no-direct-memory    Disable generation of memory access instruction, use
                        call to helper function instead.
  --strace              Log system calls.
  --disassemble         Log decoded instructions.
  --perf                Generate /tmp/perf-<PID>.map for perf tool.
  --lockstep            Use lockstep non-threaded mode for execution.
  --sysroot             Change the sysroot to a non-default value.
  --help                Display this help message.
")}

pub struct Flags {

    // Whether direct memory access or call to helper should be generated for guest memory access.
    no_direct_memory_access: bool,

    // A flag to determine whether to print instruction out when it is decoded.
    disassemble: bool,

    // If we are only emulating userspace code
    user_only: bool,

    /// If perf map should be generated
    perf: bool,

    // Whether threaded mode should be used
    thread: std::sync::atomic::AtomicBool,
}

static mut FLAGS: Flags = Flags {
    no_direct_memory_access: true,
    disassemble: false,
    user_only: false,
    perf: false,
    thread: std::sync::atomic::AtomicBool::new(true),
};

pub fn get_flags() -> &'static Flags {
    unsafe { &FLAGS }
}

pub static mut CONTEXTS: &'static mut [*mut emu::interp::Context] = &mut [];
static SHARED_CONTEXTS: RoCell<Vec<&'static emu::interp::SharedContext>> = unsafe { RoCell::new_uninit() };

pub fn shared_context(id: usize) -> &'static emu::interp::SharedContext {
    SHARED_CONTEXTS[id]
}

pub fn core_count() -> usize {
    let cnt = SHARED_CONTEXTS.len();
    assert_ne!(cnt, 0);
    cnt
}

static EVENT_LOOP: RoCell<&'static emu::EventLoop> = unsafe { RoCell::new_uninit() };

pub fn event_loop() -> &'static emu::EventLoop {
    &EVENT_LOOP
}

pub fn threaded() -> bool {
    get_flags().thread.load(std::sync::atomic::Ordering::Relaxed)
}

/// Update the current threading mode
pub fn set_threaded(mode: bool) {
    // First shutdown event loop as it is prone to influence of this flag
    event_loop().shutdown();

    // Actually set the flag
    get_flags().thread.store(mode, std::sync::atomic::Ordering::Relaxed);

    // Shutdown all execution threads
    for i in 0..core_count() {
        shared_context(i).shutdown();
    }

    // The control is now handed to main, which will restart all of them.
}

static CONFIG: RoCell<config::Config> = unsafe { RoCell::new_uninit() };

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
    let interp_name = item.expect("program name should not be absent");

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
                RoCell::replace(&emu::syscall::STRACE, true);
            }
            "--disassemble" => unsafe {
                FLAGS.disassemble = true;
            }
            "--perf" => unsafe { FLAGS.perf = true },
            "--lockstep" => unsafe {
                FLAGS.thread.store(false, std::sync::atomic::Ordering::Relaxed);
            }
            "--help" => {
                eprintln!(usage_string!(), interp_name);
                std::process::exit(0);
            }
            _ => if arg.starts_with("--sysroot=") {
                let path_slice = &arg["--sysroot=".len()..];
                sysroot = path_slice.to_owned();
            } else {
                eprintln!("{}: unrecognized option '{}'", interp_name, arg);
                std::process::exit(1);
            }
        }

        item = args.next();
    }

    let program_name = item.unwrap_or_else(|| {
        eprintln!(usage_string!(), interp_name);
        std::process::exit(1);
    });

    unsafe {
        RoCell::init(&emu::syscall::EXEC_PATH, CString::new(program_name.as_str()).unwrap());
        RoCell::init(&emu::syscall::SYSROOT, sysroot.into());
    }

    let mut loader = emu::loader::Loader::new(program_name.as_ref()).unwrap_or_else(|err| {
        eprintln!("{}: cannot load {}: {}", interp_name, program_name, err);
        std::process::exit(1);
    });

    // We accept two types of input. The file can either be a user-space ELF file,
    // or it can be a config file.
    if loader.is_elf() {
        if let Err(msg) = loader.validate_elf() {
            eprintln!("{}: {}", interp_name, msg);
            std::process::exit(1);
        }
        unsafe { FLAGS.user_only = true }
    } else {
        // Full-system emulation is needed. Originally we uses kernel path as "program name"
        // directly, but as full-system emulation requires many peripheral devices as well,
        // we decided to only accept config files.
        let config: config::Config = toml::from_slice(loader.as_slice()).unwrap_or_else(|err| {
            eprintln!("{}: invalid config file: {}", interp_name, err);
            std::process::exit(1);
        });
        unsafe { RoCell::init(&CONFIG, config) };

        // Currently due to our icache implementation, we cannot efficiently support >32 cores
        if CONFIG.core > 32 {
            eprintln!("{}: at most 32 cores allowed", interp_name);
            std::process::exit(1);
        }

        loader = emu::loader::Loader::new(&CONFIG.kernel).unwrap_or_else(|err| {
            eprintln!("{}: cannot load {}: {}", interp_name, CONFIG.kernel.to_string_lossy(), err);
            std::process::exit(1);
        });
    }

    // Create fibers for all threads
    let mut fibers = Vec::new();
    let mut contexts = Vec::new();

    let num_cores = if get_flags().user_only { 1 } else { CONFIG.core };

    // Create a fiber for event-driven simulation, e.g. timer, I/O
    let event_fiber = fiber::Fiber::new();
    unsafe { std::ptr::write(event_fiber.data_pointer(), emu::EventLoop::new()) };
    unsafe { RoCell::init(&EVENT_LOOP, &*event_fiber.data_pointer()) }
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

    unsafe {
        CONTEXTS = Box::leak(contexts.into_boxed_slice());
        RoCell::init(&SHARED_CONTEXTS, CONTEXTS.iter().map(|x| &(**x).shared).collect());
    }

    // These should only be initialised for full-system emulation
    if !get_flags().user_only {
        io::console::console_init();
        emu::init();
    }

    // Load the program
    unsafe { emu::loader::load(&loader, &mut std::iter::once(program_name).chain(args)) };
    std::mem::drop(loader);

    loop {
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
            fibers = group.run();
        } else {
            // Run one fiber per thread.
            let handles: Vec<_> = fibers.into_iter().map(|fiber| {
                std::thread::spawn(move || {
                    let mut group = fiber::FiberGroup::new();
                    group.add(fiber);
                    group.run().pop().unwrap()
                })
            }).collect();
            fibers = handles.into_iter().map(|handle| handle.join().unwrap()).collect();
        }

        // Remove old translation cache
        emu::interp::icache_reset();

        // Alert all contexts in case they having interrupts yet to process
        for i in 0..core_count() {
            shared_context(i).alert();
        }
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
