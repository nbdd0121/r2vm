use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::future::Future;
use std::io::{Read, Write};
use std::sync::Arc;
use std::task::Waker;

lazy_static! {
    /// Stores the tty config before the program is launched, so we can store it properly.
    static ref OLD_TTY: Mutex<Option<libc::termios>> = {
        unsafe {
            libc::atexit(console_exit);
        }
        Mutex::new(None)
    };
}

// Regardless the destructor of Console is executed or not, we always want the tty to be restored
// when exiting. Therefore, use atexit to guard this.
extern "C" fn console_exit() {
    let mut guard = OLD_TTY.lock();
    if let Some(ref tty) = guard.take() {
        unsafe { libc::tcsetattr(0, libc::TCSANOW, tty) };
    }
}

struct Inner {
    buffer: VecDeque<u8>,
    waker: Option<Waker>,
}

pub struct Console {
    rx: Arc<Mutex<Inner>>,
    size_callback: Mutex<Option<Box<dyn FnMut() + Send>>>,
}

impl Drop for Console {
    fn drop(&mut self) {
        console_exit();
    }
}

impl Console {
    fn new() -> Console {
        let mut guard = OLD_TTY.lock();
        // It's an error to create a new console while previous one isn't cleaned up.
        if guard.is_some() {
            panic!("Console can only be initialized once")
        }

        // Make tty as raw terminal
        unsafe {
            let mut tty = std::mem::MaybeUninit::uninit();
            libc::tcgetattr(0, tty.as_mut_ptr());
            let mut tty = tty.assume_init();
            *guard = Some(tty);
            libc::cfmakeraw(&mut tty);
            // Still treat \n as \r\n, for convience of logging
            tty.c_oflag |= libc::OPOST;
            tty.c_cc[libc::VMIN] = 1;
            tty.c_cc[libc::VTIME] = 0;
            libc::tcsetattr(0, libc::TCSANOW, &tty);
        }

        let inner = Arc::new(Mutex::new(Inner { buffer: VecDeque::new(), waker: None }));
        let inner_clone = inner.clone();

        // Spawn a thread to handle keyboard inputs.
        // In the future this thread may also use epolls etc to handle other IOs.
        // We spawn a new thread instead of using non-blocking and let guest OS to pull us so we can
        // terminate the process using Ctrl+A X whenever we like.
        std::thread::Builder::new()
            .name("console".to_owned())
            .spawn(move || {
                let mut buffer = 0;
                loop {
                    // Just read a single character
                    if std::io::stdin().read(std::slice::from_mut(&mut buffer)).unwrap() == 0 {
                        // EOF. Very unlikely. In this case we will just exit
                        return;
                    }

                    // Ctrl + A hit, read another and do corresponding action
                    if buffer == 1 {
                        std::io::stdin().read_exact(std::slice::from_mut(&mut buffer)).unwrap();
                        match buffer {
                            b't' => {
                                crate::shutdown(crate::ExitReason::SetThreaded(!crate::threaded()));
                                continue;
                            }
                            b'x' => {
                                println!("Terminated");
                                crate::shutdown(crate::ExitReason::Exit(0));
                            }
                            b'c' => unsafe {
                                libc::raise(libc::SIGTRAP);
                            },
                            // Hit Ctrl + A twice, send Ctrl + A to guest
                            1 => (),
                            // Ignore all other characters
                            _ => continue,
                        }
                    }
                    let mut inner = inner_clone.lock();
                    inner.buffer.push_back(buffer);
                    inner.waker.take().map(|x| x.wake());
                }
            })
            .unwrap();

        Console { rx: inner, size_callback: Mutex::new(None) }
    }

    pub fn send(&self, data: &[u8]) -> std::io::Result<usize> {
        let mut out = std::io::stdout();
        out.write_all(data)?;
        out.flush()?;
        Ok(data.len())
    }

    pub fn try_recv(&self, data: &mut [u8]) -> std::io::Result<usize> {
        let mut rx = match self.rx.try_lock() {
            Some(v) => v,
            None => return Ok(0),
        };
        let mut len = 0;
        while len < data.len() {
            match rx.buffer.pop_front() {
                Some(key) => {
                    data[len] = key;
                    len += 1;
                }
                None => break,
            }
        }
        Ok(len)
    }

    pub async fn recv(&self, data: &mut [u8]) -> std::io::Result<usize> {
        use std::pin::Pin;
        use std::task::{Context, Poll};

        struct RecvFuture<'a>(&'a Console, &'a mut [u8]);

        impl<'a> Future for RecvFuture<'a> {
            type Output = std::io::Result<usize>;

            fn poll(mut self: Pin<&mut Self>, ctx: &mut Context) -> Poll<std::io::Result<usize>> {
                let mut inner = self.0.rx.lock();
                if inner.buffer.is_empty() {
                    inner.waker = Some(ctx.waker().clone());
                    return Poll::Pending;
                }

                let mut len = 0;
                while len < self.1.len() {
                    match inner.buffer.pop_front() {
                        Some(key) => {
                            self.1[len] = key;
                            len += 1;
                        }
                        None => break,
                    }
                }
                Poll::Ready(Ok(len))
            }
        }

        RecvFuture(self, data).await
    }

    pub fn get_size(&self) -> std::io::Result<(u16, u16)> {
        unsafe {
            let mut size = std::mem::MaybeUninit::<libc::winsize>::uninit();
            if libc::ioctl(0, libc::TIOCGWINSZ, &mut size) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            let size = size.assume_init();
            Ok((size.ws_col, size.ws_row))
        }
    }

    pub fn on_size_change(&self, callback: Option<Box<dyn FnMut() + Send>>) {
        if callback.is_some() {
            WINCH.call_once(|| unsafe {
                let mut act: libc::sigaction = std::mem::zeroed();
                act.sa_sigaction = handle_winch as usize;
                act.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
                libc::sigaction(libc::SIGWINCH, &act, std::ptr::null_mut());
            });
        }
        *self.size_callback.lock() = callback;
    }
}

/// Handle SIGWINCH for tty size change notification
unsafe extern "C" fn handle_winch(
    _: libc::c_int,
    _: &mut libc::siginfo_t,
    _: &mut libc::ucontext_t,
) {
    CONSOLE.size_callback.lock().as_mut().map(|x| x());
}

static WINCH: std::sync::Once = std::sync::Once::new();

lazy_static! {
    pub static ref CONSOLE: Console =  Console::new() ;
}

pub fn console_init() {
    lazy_static::initialize(&CONSOLE);
}

pub fn console_putchar(char: u8) {
    CONSOLE.send(std::slice::from_ref(&char)).unwrap();
}

pub fn console_getchar() -> i64 {
    let mut ret = 0;
    match CONSOLE.try_recv(std::slice::from_mut(&mut ret)).unwrap() {
        0 => -1,
        _ => ret as i64,
    }
}
