use io::serial::{Serial, Console};
use once_cell::sync::Lazy;

pub static CONSOLE: Lazy<Console> = Lazy::new(|| {
    let mut console = Console::new().unwrap();
    let mut escape_hit = false;
    console.set_processor(move |x| {
        if !escape_hit {
            if x == 1 {
                // Ctrl + A hit, wait for another byte to arrive
                escape_hit = true;
                return None;
            }
            return Some(x);
        }
        
        // Byte after Ctrl + A hit, do corresponding action
        match x {
            b't' => {
                let model_id = if crate::get_flags().model_id == 0 { 1 } else { 0 };
                crate::shutdown(crate::ExitReason::SwitchModel(model_id));
            }
            b'x' => {
                println!("Terminated");
                crate::shutdown(crate::ExitReason::Exit(0));
            }
            b'p' => {
                crate::shutdown(crate::ExitReason::PrintStats);
            }
            b'c' => unsafe {
                libc::raise(libc::SIGTRAP);
            },
            // Hit Ctrl + A twice, send Ctrl + A to guest
            1 => return Some(x),
            // Ignore all other characters
            _ => (),
        }
        None
    });
    console
});

pub fn console_init() {
    Lazy::force(&CONSOLE);
}

pub fn console_putchar(char: u8) {
    (&*CONSOLE as &dyn Serial).try_write(std::slice::from_ref(&char)).unwrap();
}

pub fn console_getchar() -> i64 {
    let mut ret = 0;
    match (&*CONSOLE as &dyn Serial).try_read(std::slice::from_mut(&mut ret)) {
        Err(_) => -1,
        _ => ret as i64,
    }
}
