#[macro_use]
extern crate log;
extern crate pretty_env_logger;
extern crate rand;

pub mod io;
pub mod util;
pub mod emu;
#[no_mangle]
pub extern "C" fn rust_init() {
    pretty_env_logger::init();
    io::console::console_init();
    emu::init();
}
