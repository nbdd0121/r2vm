#[macro_use]
extern crate log;
#[macro_use]
extern crate memoffset;

pub mod config;
pub mod emu;
pub mod sim;
pub mod util;

use clap::{CommandFactory, FromArgMatches, Parser};
use ro_cell::RoCell;
use std::cell::UnsafeCell;
use std::ffi::{CString, OsString};
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;

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
    dump_fdt: Option<OsString>,

    /// A flag to determine whether to trace all system calls. If true then all guest system calls will be logged.
    strace: bool,

    /// The actual path of the executable. Needed by src/emu/syscall.rs to redirect /proc/self/*
    exec_path: CString,

    /// Path of sysroot. When the guest application tries to open a file, and the corresponding file exists in sysroot,
    /// it will be redirected.
    sysroot: PathBuf,
}

#[derive(Parser)]
pub struct Args {
    /// Log system calls.
    #[arg(long)]
    strace: bool,
    /// Log decoded instructions.
    #[arg(long)]
    disassemble: bool,
    /// Generate /tmp/perf-<PID>.map for perf tool.
    #[arg(long)]
    perf: bool,
    /// Use lockstep non-threaded mode for execution.
    #[arg(long)]
    lockstep: bool,
    /// Turn WFI instructions to NOPs instead of sleeping.
    #[arg(long)]
    wfi_nop: bool,
    /// Change the sysroot to a non-default value.
    #[arg(long)]
    sysroot: Option<OsString>,
    /// Save FDT to the specified path.
    #[arg(long)]
    dump_fdt: Option<OsString>,
    #[arg(value_name = "PROGRAM")]
    exec_path: OsString,
    arguments: Vec<OsString>,
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

    let mut command = Args::command();
    let args = Args::from_arg_matches_mut(&mut command.get_matches_mut()).unwrap();

    let mut flags = Flags {
        disassemble: args.disassemble,
        prv: 1,
        perf: args.perf,
        thread: true,
        blocking_io: !args.lockstep,
        model_id: if args.lockstep { 1 } else { 0 },
        wfi_nop: args.wfi_nop,
        dump_fdt: args.dump_fdt,
        strace: args.strace,
        exec_path: CString::default(),
        sysroot: args.sysroot.unwrap_or_else(|| "/opt/riscv/sysroot".into()).into(),
    };

    let program_name = args.exec_path;
    flags.exec_path = CString::new(program_name.as_bytes()).unwrap();

    unsafe { RoCell::init(&FLAGS, flags) };

    let mut loader = emu::loader::Loader::new(program_name.as_ref()).unwrap_or_else(|err| {
        command
            .error(
                clap::error::ErrorKind::ValueValidation,
                format_args!("cannot load {}: {}", program_name.to_string_lossy(), err),
            )
            .exit();
    });

    // We accept two types of input. The file can either be a user-space ELF file,
    // or it can be a config file.
    if loader.is_elf() {
        if let Err(msg) = loader.validate_elf() {
            command.error(clap::error::ErrorKind::ValueValidation, msg).exit();
        }
        unsafe { RoCell::as_mut(&FLAGS).prv = 0 }
    } else {
        // Full-system emulation is needed. Originally we uses kernel path as "program name"
        // directly, but as full-system emulation requires many peripheral devices as well,
        // we decided to only accept config files.
        let Ok(toml_str) = std::str::from_utf8(loader.as_slice()) else {
            command
                .error(clap::error::ErrorKind::InvalidUtf8, "invalid config file: not utf8")
                .exit();
        };
        let config: config::Config = toml::from_str(toml_str).unwrap_or_else(|err| {
            command
                .error(
                    clap::error::ErrorKind::ValueValidation,
                    format_args!("invalid config file: {}", err),
                )
                .exit();
        });
        unsafe { RoCell::init(&CONFIG, config) };

        // Currently due to our icache implementation, we cannot efficiently support >32 cores
        if CONFIG.core > 32 {
            command
                .error(clap::error::ErrorKind::ValueValidation, "at most 32 cores allowed")
                .exit();
        }

        if CONFIG.firmware.is_some() {
            unsafe { RoCell::as_mut(&FLAGS).prv = 3 }
        }

        loader = emu::loader::Loader::new(&CONFIG.kernel).unwrap_or_else(|err| {
            command
                .error(
                    clap::error::ErrorKind::ValueValidation,
                    format_args!("cannot load {}: {}", CONFIG.kernel.to_string_lossy(), err),
                )
                .exit();
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
            newctx.scounteren = 0b111;
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
        emu::loader::load(
            &loader,
            &mut std::iter::once(program_name).chain(args.arguments),
            &mut contexts,
        )
    };
    std::mem::drop(loader);

    // Load firmware if present
    if let Some(ref firmware) = CONFIG.firmware {
        let loader = emu::loader::Loader::new(firmware).unwrap_or_else(|err| {
            command
                .error(
                    clap::error::ErrorKind::ValueValidation,
                    format_args!("cannot load {}: {}", firmware.to_string_lossy(), err),
                )
                .exit();
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
