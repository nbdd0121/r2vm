#[macro_use]
extern crate log;
extern crate pretty_env_logger;
extern crate rand;
extern crate fnv;

pub mod riscv;
pub mod io;
pub mod util;
pub mod emu;

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
  --strict-exception    Enable strict enforcement of excecution correctness in
                        case of segmentation fault.
  --enable-phi          Allow load elimination to emit PHI nodes.
  --region-limit=<n>    Number of basic blocks that can be included in a single
                        compilation region by the IR-based binary translator.
  --compile-threshold=<n> Number of execution required for a block to be
                        considered by the IR-based binary translator.
  --monitor-performance Display metrics about performance in compilation phase.
  --sysroot             Change the sysroot to a non-default value.
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
    no_instret: bool,

    // A flag to determine whether correctness in case of segmentation fault should be dealt strictly.
    strict_exception: bool,

    // A flag to determine whether PHI nodes should be introduced to the graph by load elimination.
    enable_phi: bool,

    // Whether compilation performance counters should be enabled.
    monitor_performance: bool,

    // The actual path of the executable. Needed to redirect /proc/self/*
    exec_path: *const i8,

    // Path of sysroot. When the guest application tries to open a file, and the corresponding file exists in sysroot,
    // it will be redirected.
    sysroot: Option<String>,

    // Upper limit of number of blocks that can be placed in a region.
    region_limit: u32,

    // Threshold beyond which the IR DBT will start working
    compile_threshold: u32,

    // If we are only emulating userspace code
    user_only: bool,
}

pub static mut FLAGS: Flags = Flags {
    no_direct_memory_access: true,
    strace: false,
    disassemble: false,
    no_instret: true,
    strict_exception: false,
    enable_phi: false,
    monitor_performance: false,
    exec_path: ptr::null(),
    sysroot: None,
    region_limit: 16,
    compile_threshold: 0,
    user_only: true,
};

#[no_mangle]
pub extern fn get_flags() -> &'static Flags {
    unsafe {
        return &FLAGS;
    }
}

static mut CTX: riscv::interp::Context = riscv::interp::Context {
    registers: [0xCCCCCCCCCCCCCCCC; 32],
    fp_registers: [0xFFFFFFFFFFFFFFFF; 32],
    fcsr: 0,
    instret: 0,
    lr: 0,
    // UXL = 0b10, indicating 64-bit
    sstatus: 0x200000000,
    scause: 0,
    sepc: 0,
    stval: 0,
    satp: 0,
    sip: 0,
    sie: 0,
    sscratch: 0,
    stvec: 0,
    pending: 0,
    timecmp: u64::max_value(),
    // These are set by setup_mem, so we don't really care now.
    pc: 0,
    prv: 0,
    line: [riscv::interp::CacheLine {
        tag: i64::max_value() as u64,
        paddr: 0
    }; 1024]
};

#[no_mangle]
extern "C" fn interrupt() {
    unsafe {
        CTX.sip |= 512;
        CTX.pending = if (CTX.sstatus & 0x2) != 0 { CTX.sip & CTX.sie } else { 0 };
    }
}

pub fn main() {
    // Top priority: set up page fault handlers so safe_memory features will work.
    emu::safe_memory::init();
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
            "--strict-exception" => unsafe {
                FLAGS.strict_exception = true;
            }
            "--enable-phi" => unsafe {
                FLAGS.enable_phi = true;
            }
            "--monitor-performance" => unsafe {
                FLAGS.monitor_performance = true;
            }
            "--help" => {
                eprintln!(usage_string!(), interp_name);
                std::process::exit(0);
            }
            _ => if arg.starts_with("--region-limit=") {
                let num_slice = &arg["--region-limit=".len()..];
                let num = num_slice.parse::<u32>().unwrap_or(0);
                if num > 0 {
                    unsafe {
                        FLAGS.region_limit = num;
                    }
                } else {
                    eprintln!("{}: '{}' is not a valid positive integer", interp_name, num_slice);
                    std::process::exit(1);
                }
            } else if arg.starts_with("--compile-threshold=") {
                let num_slice = &arg["--compile-threshold=".len()..];
                if let Ok(num) = num_slice.parse::<u32>() {
                    unsafe {
                        FLAGS.compile_threshold = num;
                    }
                } else {
                    eprintln!("{}: '{}' is not a valid non-negative integer", interp_name, num_slice);
                    std::process::exit(1);
                }
            } else if arg.starts_with("--sysroot=") {
                let path_slice = &arg["--sysroot=".len()..];
                sysroot = path_slice.to_owned();
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
    // Simple guess: If not elf, then we assume it is vmlinux
    match loader.validate_elf() {
        Ok(_) => (),
        Err(_) => unsafe { FLAGS.user_only = false },
    }

    // These should only be initialised for full-system emulation
    if !get_flags().user_only {
        io::console::console_init();
        emu::init();
    }

    // x0 must always be 0
    unsafe { CTX.registers[0] = 0 };
    unsafe { emu::loader::load(&loader, &mut CTX, &mut std::iter::once(program_name).chain(args)) };

    loop {
        unsafe {
            riscv::interp::run_block_ex(&mut CTX)
        }
    }
}
