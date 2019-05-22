#[macro_use]
extern crate log;
extern crate pretty_env_logger;
extern crate rand;
extern crate fnv;

pub mod riscv;
pub mod io;
pub mod util;
pub mod emu;

#[no_mangle]
pub extern "C" fn rust_init() {
    pretty_env_logger::init();
    io::console::console_init();
    emu::init();
}
