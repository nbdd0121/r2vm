use std::io::{Read, Write};
use std::sync::mpsc::{Receiver, TryRecvError};

/// Stores the tty config before the program is launched, so we can store it properly.
static mut OLD_TTY: Option<libc::termios> = None;

extern "C" fn console_exit() {
    unsafe { libc::tcsetattr(0, libc::TCSANOW, OLD_TTY.as_ref().unwrap()) };
}

static mut RX: Option<Receiver<u8>> = None;

pub fn console_init() {
    // Make tty as raw terminal
    unsafe {
        let mut tty: libc::termios = std::mem::uninitialized();
        libc::tcgetattr(0, &mut tty);
        OLD_TTY = Some(tty);
        libc::cfmakeraw(&mut tty);
        // Still treat \n as \r\n, for convience of logging
        tty.c_oflag |= libc::OPOST;
        tty.c_cc[libc::VMIN] = 1;
        tty.c_cc[libc::VTIME] = 0;
        libc::tcsetattr(0, libc::TCSANOW, &tty);
        libc::atexit(console_exit);
    }

    // Spawn a thread to handle keyboard inputs.
    // In the future this thread may also use epolls etc to handle other IOs.
    // We spawn a new thread instead of using non-blocking and let guest OS to pull us so we can
    // terminate the process using Ctrl+A X whenever we like.
    let (tx, rx) = std::sync::mpsc::channel::<u8>();
    std::thread::spawn(move || {
        let mut buffer = 0;
        loop {
            // Just read a single character
            std::io::stdin().read_exact(std::slice::from_mut(&mut buffer)).unwrap();

            // Ctrl + A hit, read another and do corresponding action
            if buffer == 1 {
                std::io::stdin().read_exact(std::slice::from_mut(&mut buffer)).unwrap();
                match buffer {
                    b'x' => {
                        println!("Terminated");
                        crate::print_stats_and_exit(0);
                    }
                    b'c' => {
                        unsafe { libc::raise(libc::SIGTRAP); }
                    }
                    // Hit Ctrl + A twice, send Ctrl + A to guest
                    1 => (),
                    // Ignore all other characters
                    _ => continue,
                }
            }
            tx.send(buffer).unwrap();
        }
    });

    unsafe { RX = Some(rx) };
}

pub fn console_putchar(char: u8) {
    let mut out = std::io::stdout();
    out.write_all(std::slice::from_ref(&char)).unwrap();
    out.flush().unwrap();
}

pub fn console_getchar() -> i64 {
    // TODO: Remove unsafe
    match unsafe { RX.as_mut() }.unwrap().try_recv() {
        Ok(key) => key as i64,
        Err(TryRecvError::Empty) => -1,
        Err(TryRecvError::Disconnected) => unreachable!(),
    }
}
