#[macro_use]
extern crate log;
#[macro_use]
extern crate memoffset;

pub mod config;
pub mod emu;
pub mod sim;
pub mod util;

use ro_cell::RoCell;
use std::cell::UnsafeCell;
use std::ffi::CString;
use std::path::PathBuf;

macro_rules! usage_string {
    () => {
        "Usage: {} [options] program [arguments...]
Options:
  --strace              Log system calls.
  --disassemble         Log decoded instructions.
  --perf                Generate /tmp/perf-<PID>.map for perf tool.
  --lockstep            Use lockstep non-threaded mode for execution.
  --wfi-nop             Treat WFI as nops in lock-step mode.
  --sysroot             Change the sysroot to a non-default value.
  --dump-fdt            Save FDT to the specified path.
  --help                Display this help message.
"
    };
}

pub struct Flags {
    // A flag to determine whether to print instruction out when it is decoded.
    disassemble: bool,

    // The highest privilege mode emulated
    prv: u8,

    /// If perf map should be generated
    perf: bool,

    // Whether threaded mode should be used
    thread: bool,

    // Whether blocking IO should be offloaded to a separate thread or block on the event loop.
    blocking_io: bool,

    /// The active model ID used
    model_id: usize,

    /// Whether WFI should be treated as NOP in lock-step mode
    wfi_nop: bool,

    /// Dump FDT option
    dump_fdt: Option<String>,

    /// A flag to determine whether to trace all system calls. If true then all guest system calls will be logged.
    strace: bool,

    /// The actual path of the executable. Needed by src/emu/syscall.rs to redirect /proc/self/*
    exec_path: CString,

    /// Path of sysroot. When the guest application tries to open a file, and the corresponding file exists in sysroot,
    /// it will be redirected.
    sysroot: PathBuf,
}

static FLAGS: RoCell<Flags> = unsafe { RoCell::new_uninit() };

pub fn get_flags() -> &'static Flags {
    &FLAGS
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

static EXIT_REASON: parking_lot::Mutex<Option<ExitReason>> =
    parking_lot::Mutex::const_new(<parking_lot::RawMutex as lock_api::RawMutex>::INIT, None);

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
    pretty_env_logger::init();

    let mut args = std::env::args();

    // Ignore interpreter name
    let mut item = args.next();
    let interp_name = item.expect("program name should not be absent");

    let mut flags = Flags {
        disassemble: false,
        prv: 1,
        perf: false,
        thread: true,
        blocking_io: false,
        model_id: 0,
        wfi_nop: false,
        dump_fdt: None,
        strace: false,
        exec_path: CString::default(),
        sysroot: "/opt/riscv/sysroot".into(),
    };

    item = args.next();
    while let Some(ref arg) = item {
        // We've parsed all arguments. This indicates the name of the executable.
        if !arg.starts_with('-') {
            break;
        }

        match arg.as_str() {
            "--strace" => flags.strace = true,
            "--disassemble" => flags.disassemble = true,
            "--perf" => flags.perf = true,
            "--lockstep" => {
                flags.model_id = 1;
                flags.blocking_io = true;
            }
            "--wfi-nop" => flags.wfi_nop = true,
            "--help" => {
                eprintln!(usage_string!(), interp_name);
                std::process::exit(0);
            }
            _ => {
                if arg.starts_with("--sysroot=") {
                    flags.sysroot = arg["--sysroot=".len()..].into();
                } else if arg.starts_with("--dump-fdt=") {
                    let path_slice = &arg["--dump-fdt=".len()..];
                    flags.dump_fdt = Some(path_slice.to_owned());
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

    flags.exec_path = CString::new(program_name.as_str()).unwrap();

    unsafe { RoCell::init(&FLAGS, flags) };

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
        unsafe { RoCell::as_mut(&FLAGS).prv = 0 }
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
            unsafe { RoCell::as_mut(&FLAGS).prv = 3 }
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
    let event_fiber = fiber::FiberContext::new(emu::EventLoop::new());
    unsafe { RoCell::init(&EVENT_LOOP, std::mem::transmute(event_fiber.data::<emu::EventLoop>())) }
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
            cycle_offset: 0,
        };
        // x0 must always be 0
        newctx.registers[0] = 0;

        if CONFIG.firmware.is_none() {
            newctx.mideleg = 0x222;
            newctx.medeleg = 0xB35D;
            newctx.mcounteren = 0b111;
        }

        let fiber = fiber::FiberContext::new(UnsafeCell::new(newctx));
        let ptr = fiber.data::<UnsafeCell<emu::interp::Context>>().get();
        contexts.push(unsafe { &mut *ptr });
        shared_contexts.push(unsafe { &(*ptr).shared });
        fibers.push(fiber);
    }

    unsafe { RoCell::init(&SHARED_CONTEXTS, shared_contexts) };

    // These should only be initialised for full-system emulation
    if get_flags().prv != 0 {
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
        unsafe {
            if loader.is_elf() {
                loader.load_kernel(location as u64);
            } else {
                loader.load_bin(location as u64);
            }
        }

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
        let fn_of_idx = |idx| -> fn() {
            if idx == 0 {
                || {
                    fiber::with_context(|data: &emu::EventLoop| data.event_loop());
                }
            } else {
                || unsafe { fiber_interp_run() }
            }
        };

        if !crate::threaded() {
            // Run multiple fibers in the same group.
            fiber::FiberGroup::with(|group| {
                for (idx, fiber) in fibers.iter_mut().enumerate() {
                    group.spawn(fiber, fn_of_idx(idx));
                }
            });
        } else {
            // Run one fiber per thread.
            let handles: Vec<_> = fibers
                .into_iter()
                .enumerate()
                .map(|(idx, mut fiber)| {
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
                            fiber::FiberGroup::with(|group| {
                                group.spawn(&mut fiber, fn_of_idx(idx));
                            });
                            fiber
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
                    RoCell::as_mut(&FLAGS).model_id = id;
                    info!("switching to model={} threaded={}", id, FLAGS.thread);
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
                    crate::CPU_TIME_BASE = crate::util::cpu_time();
                    crate::CYCLE_TIME_BASE = crate::event_loop().cycle();
                    crate::CYCLE_BASE = crate::event_loop().get_lockstep_cycles();
                }
                for ctx in contexts.iter_mut() {
                    ctx.instret = 0;
                    ctx.minstret = 0;
                    ctx.cycle_offset = 0;
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

pub static mut CPU_TIME_BASE: std::time::Duration = std::time::Duration::from_secs(0);
pub static mut CYCLE_TIME_BASE: u64 = 0;
pub static mut CYCLE_BASE: u64 = 0;

fn print_stats(ctxs: &[&mut emu::interp::Context]) -> std::io::Result<()> {
    use std::io::Write;
    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    let _stdout = stdout.lock();
    let mut stderr = stderr.lock();
    let cpu_time = unsafe { util::cpu_time() - CPU_TIME_BASE };
    let cycle_time = unsafe { event_loop().cycle() - CYCLE_TIME_BASE };
    writeln!(stderr, "CPU TIME = {:?}", cpu_time)?;
    writeln!(stderr, "CYCLE TIME = {}", cycle_time)?;
    let mut instret = 0;
    let mut minstret = 0;
    let mut cycle = 0;
    for ctx in ctxs {
        instret += ctx.instret;
        minstret += ctx.minstret;
        let mcycle = unsafe { ctx.get_mcycle() - CYCLE_BASE };
        cycle += mcycle;
        writeln!(
            stderr,
            "Hart {}: CYCLE = {}, INSTRET = {}, MINSTRET = {}",
            ctx.hartid, mcycle, ctx.instret, ctx.minstret
        )?;
    }
    writeln!(stderr, "Total: CYCLE = {}, INSTRET = {}, MINSTRET = {}", cycle, instret, minstret)?;
    writeln!(stderr)?;
    crate::sim::get_memory_model().print_stats(&mut stderr)?;
    Ok(())
}
