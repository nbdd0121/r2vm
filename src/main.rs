#[macro_use]
extern crate log;

pub mod io;
#[macro_use]
pub mod util;
pub mod config;
pub mod emu;
pub mod fiber;
pub mod sim;

use std::ffi::CString;
use util::RoCell;

macro_rules! usage_string {
    () => {
        "Usage: {} [options] program [arguments...]
Options:
  --no-direct-memory    Disable generation of memory access instruction, use
                        call to helper function instead.
  --strace              Log system calls.
  --disassemble         Log decoded instructions.
  --perf                Generate /tmp/perf-<PID>.map for perf tool.
  --lockstep            Use lockstep non-threaded mode for execution.
  --sysroot             Change the sysroot to a non-default value.
  --dump-fdt            Save FDT to the specified path.
  --help                Display this help message.
"
    };
}

pub struct Flags {
    // Whether direct memory access or call to helper should be generated for guest memory access.
    no_direct_memory_access: bool,

    // A flag to determine whether to print instruction out when it is decoded.
    disassemble: bool,

    // The highest privilege mode emulated
    prv: u8,

    /// If perf map should be generated
    perf: bool,

    // Whether threaded mode should be used
    thread: bool,

    /// The active model ID used
    model_id: usize,

    /// Dump FDT option
    dump_fdt: Option<String>,
}

static mut FLAGS: Flags = Flags {
    no_direct_memory_access: true,
    disassemble: false,
    prv: 1,
    perf: false,
    thread: true,
    model_id: 0,
    dump_fdt: None,
};

pub fn get_flags() -> &'static Flags {
    unsafe { &FLAGS }
}

static SHARED_CONTEXTS: RoCell<Vec<&'static emu::interp::SharedContext>> =
    unsafe { RoCell::new_uninit() };

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
    get_flags().thread
}

use lazy_static::lazy_static;
lazy_static! {
    static ref EXIT_REASON: parking_lot::Mutex<Option<ExitReason>> = Default::default();
}

/// Reason for exiting executors
enum ExitReason {
    SwitchModel(usize),
    Exit(i32),
    ClearStats,
    PrintStats,
}

fn shutdown(reason: ExitReason) {
    // Shutdown event loop as soon as possible
    event_loop().shutdown();

    *EXIT_REASON.lock() = Some(reason);

    // Shutdown all execution threads
    for i in 0..core_count() {
        shared_context(i).shutdown();
    }
}

static CONFIG: RoCell<config::Config> = unsafe { RoCell::new_uninit() };

extern "C" {
    fn fiber_interp_run();
}

pub fn main() {
    // Allow any one to ptrace us, mainly for debugging purpose
    unsafe { libc::prctl(libc::PR_SET_PTRACER, (-1) as libc::c_long) };

    // Top priority: set up page fault handlers so safe_memory features will work.
    emu::signal::init();
    emu::interp::init_fp();
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
            },
            "--strace" => unsafe {
                RoCell::replace(&emu::syscall::STRACE, true);
            },
            "--disassemble" => unsafe {
                FLAGS.disassemble = true;
            },
            "--perf" => unsafe { FLAGS.perf = true },
            "--lockstep" => unsafe { FLAGS.model_id = 1 },
            "--help" => {
                eprintln!(usage_string!(), interp_name);
                std::process::exit(0);
            }
            _ => {
                if arg.starts_with("--sysroot=") {
                    let path_slice = &arg["--sysroot=".len()..];
                    sysroot = path_slice.to_owned();
                } else if arg.starts_with("--dump-fdt=") {
                    let path_slice = &arg["--dump-fdt=".len()..];
                    unsafe {
                        FLAGS.dump_fdt = Some(path_slice.to_owned());
                    }
                } else {
                    eprintln!("{}: unrecognized option '{}'", interp_name, arg);
                    std::process::exit(1);
                }
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
        unsafe { FLAGS.prv = 0 }
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

        if CONFIG.firmware.is_some() {
            unsafe { FLAGS.prv = 3 }
        }

        loader = emu::loader::Loader::new(&CONFIG.kernel).unwrap_or_else(|err| {
            eprintln!("{}: cannot load {}: {}", interp_name, CONFIG.kernel.to_string_lossy(), err);
            std::process::exit(1);
        });
    }

    // Create fibers for all threads
    let mut fibers = Vec::new();
    let mut contexts = Vec::new();
    let mut shared_contexts = Vec::new();

    let num_cores = if get_flags().prv == 0 { 1 } else { CONFIG.core };

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
            frm: 0,
            instret: 0,
            lr_addr: 0,
            lr_value: 0,
            cause: 0,
            tval: 0,
            // FPU turned on by default
            mstatus: 0x6000,
            scause: 0,
            sepc: 0,
            stval: 0,
            satp: 0,
            sscratch: 0,
            stvec: 0,
            scounteren: 0,
            mideleg: 0,
            medeleg: 0,
            mcause: 0,
            mepc: 0,
            mtval: 0,
            mie: 0,
            mscratch: 0,
            mtvec: 0,
            mcounteren: 0,
            // These are set by setup_mem, so we don't really care now.
            pc: 0,
            prv: 0,
            hartid: i as u64,
            minstret: 0,
        };
        // x0 must always be 0
        newctx.registers[0] = 0;

        if CONFIG.firmware.is_none() {
            newctx.mideleg = 0x222;
            newctx.medeleg = 0xB35D;
            newctx.mcounteren = 0b111;
        }

        let fiber = fiber::Fiber::new();
        let ptr = fiber.data_pointer();
        unsafe {
            std::ptr::write(ptr, newctx);
        }
        contexts.push(unsafe { &mut *ptr });
        shared_contexts.push(unsafe { &(*ptr).shared });
        fibers.push(fiber);
    }

    unsafe { RoCell::init(&SHARED_CONTEXTS, shared_contexts) };

    // These should only be initialised for full-system emulation
    if get_flags().prv != 0 {
        io::console::console_init();
        emu::init();
    }

    // Load the program
    unsafe {
        emu::loader::load(&loader, &mut std::iter::once(program_name).chain(args), &mut contexts)
    };
    std::mem::drop(loader);

    // Load firmware if present
    if let Some(ref firmware) = CONFIG.firmware {
        let loader = emu::loader::Loader::new(firmware).unwrap_or_else(|err| {
            eprintln!("{}: cannot load {}: {}", interp_name, firmware.to_string_lossy(), err);
            std::process::exit(1);
        });
        // Load this past memory location
        let location = 0x40000000 + ((CONFIG.memory * 0x100000 + 0x1fffff) & !0x1fffff);
        unsafe { loader.load_kernel(location as u64) };

        for ctx in contexts.iter_mut() {
            ctx.registers[12] = ctx.pc;
            ctx.pc = location as u64;
            ctx.prv = 3;
        }
    }

    unsafe {
        crate::sim::switch_model(FLAGS.model_id);
    }

    loop {
        fibers[0].set_fn(|| {
            let this: &emu::EventLoop = unsafe { &*fiber::Fiber::scratchpad() };
            this.event_loop()
        });
        for fiber in &mut fibers[1..] {
            fiber.set_fn(|| unsafe { fiber_interp_run() });
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
            let handles: Vec<_> = fibers
                .into_iter()
                .enumerate()
                .map(|(idx, fiber)| {
                    let name = if idx == 0 {
                        "event-loop".to_owned()
                    } else {
                        if crate::get_flags().perf {
                            "hart".to_owned()
                        } else {
                            format!("hart {}", idx - 1)
                        }
                    };

                    std::thread::Builder::new()
                        .name(name)
                        .spawn(move || {
                            let mut group = fiber::FiberGroup::new();
                            group.add(fiber);
                            group.run().pop().unwrap()
                        })
                        .unwrap()
                })
                .collect();
            fibers = handles.into_iter().map(|handle| handle.join().unwrap()).collect();
        }

        match EXIT_REASON.lock().as_ref().unwrap() {
            &ExitReason::SwitchModel(id) => {
                unsafe {
                    crate::sim::switch_model(id);
                    FLAGS.model_id = id;
                    let threaded = !crate::sim::get_memory_model().require_lockstep();
                    FLAGS.thread = threaded;
                    info!("switching to model={} threaded={}", id, threaded);
                }

                // Remove translation cache and L0 I$ and D$
                emu::interp::icache_reset();
                for ctx in contexts.iter_mut() {
                    ctx.shared.clear_local_cache();
                    ctx.shared.clear_local_icache();
                }
            }
            &ExitReason::Exit(code) => {
                print_stats(&mut contexts).unwrap();
                std::process::exit(code);
            }
            ExitReason::ClearStats => {
                unsafe {
                    crate::TIME_BASE = crate::util::cpu_time();
                    crate::CYCLE_BASE = crate::event_loop().cycle();
                }
                for ctx in contexts.iter_mut() {
                    ctx.instret = 0;
                    ctx.minstret = 0;
                }
                crate::sim::get_memory_model().reset_stats();
            }
            ExitReason::PrintStats => {
                print_stats(&mut contexts).unwrap();
            }
        }

        // Alert all contexts in case they having interrupts yet to process
        for i in 0..core_count() {
            shared_context(i).alert();
        }
    }
}

pub static mut TIME_BASE: std::time::Duration = std::time::Duration::from_secs(0);
pub static mut CYCLE_BASE: u64 = 0;

fn print_stats(ctxs: &[&mut emu::interp::Context]) -> std::io::Result<()> {
    use std::io::Write;
    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    let _stdout = stdout.lock();
    let mut stderr = stderr.lock();
    let time = unsafe { util::cpu_time() - TIME_BASE };
    let cycle = unsafe { event_loop().cycle() - CYCLE_BASE } as i64;
    writeln!(stderr, "TIME = {:?}", time)?;
    writeln!(stderr, "CYCLE = {}", cycle)?;
    let mut instret = 0;
    let mut minstret = 0;
    for ctx in ctxs {
        instret += ctx.instret;
        minstret += ctx.minstret;
        writeln!(
            stderr,
            "Hart {}: INSTRET = {}, MINSTRET = {}",
            ctx.hartid, ctx.instret, ctx.minstret
        )?;
    }
    writeln!(stderr, "Total: INSTRET = {}, MINSTRET = {}", instret, minstret)?;
    writeln!(stderr)?;
    crate::sim::get_memory_model().print_stats(&mut stderr)?;
    Ok(())
}
