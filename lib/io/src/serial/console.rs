use super::Serial;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

static ACTIVE_CONSOLE: Lazy<Mutex<Option<Arc<Inner>>>> = Lazy::new(|| Mutex::new(None));
static AT_EXIT: std::sync::Once = std::sync::Once::new();
static WINCH: std::sync::Once = std::sync::Once::new();

struct State {
    rx_buffer: VecDeque<u8>,
    rx_waker: Vec<Waker>,
    size_changed: bool,
    size_changed_wakers: Vec<Waker>,
    processor: Box<dyn FnMut(u8) -> Option<u8> + Send>,
}

struct Inner {
    state: Mutex<State>,
    old_tty: libc::termios,
}

/// A [`Serial`] implementation that uses stdin/stdout TTY.
///
/// This device allow custom processing of the TTY input, to support escape keys, see `set_processor`.
pub struct Console(Arc<Inner>);

// Regardless the destructor of Console is executed or not, we always want the tty to be restored
// when exiting. Therefore, use atexit to guard this.
extern "C" fn console_exit() {
    if let Some(inner) = ACTIVE_CONSOLE.lock().as_ref() {
        unsafe { libc::tcsetattr(0, libc::TCSANOW, &inner.old_tty) };
    }
}

impl Drop for Console {
    fn drop(&mut self) {
        // Remove ACTIVE_CONSOLE. Without this the inner struct won't actually be dropped.
        *ACTIVE_CONSOLE.lock() = None;

        // Restore old TTY config
        unsafe { libc::tcsetattr(0, libc::TCSANOW, &self.0.old_tty) };
    }
}

impl Console {
    /// Construct a `Console`. As `Console` takes control of the stdin/stdout TTY, only 1 `Console`
    /// can be constructed at the same time. If there is already one alive, [`None`] is returned.
    pub fn new() -> Option<Self> {
        // Lock here first to avoid reentrnace.
        let mut active_console = ACTIVE_CONSOLE.lock();
        if active_console.is_some() {
            return None;
        }

        // Make tty as raw terminal, and save old config
        let old_tty = unsafe {
            let mut tty = std::mem::MaybeUninit::uninit();
            libc::tcgetattr(0, tty.as_mut_ptr());
            let mut tty = tty.assume_init();
            let old_tty = tty;
            libc::cfmakeraw(&mut tty);
            // Still treat \n as \r\n, for convience of logging
            tty.c_oflag |= libc::OPOST;
            tty.c_cc[libc::VMIN] = 1;
            tty.c_cc[libc::VTIME] = 0;
            libc::tcsetattr(0, libc::TCSANOW, &tty);
            old_tty
        };

        let inner = Arc::new(Inner {
            old_tty,
            state: Mutex::new(State {
                rx_buffer: VecDeque::new(),
                rx_waker: Vec::new(),
                size_changed: false,
                size_changed_wakers: Vec::new(),
                processor: Box::new(|x| Some(x)),
            }),
        });

        // Register the exit hook if not already done.
        AT_EXIT.call_once(|| unsafe {
            libc::atexit(console_exit);
        });
        *active_console = Some(inner.clone());

        let weak = Arc::downgrade(&inner);
        // Construct early to ensure destructor will run even when thread creation fails.
        let ret = Console(inner);

        // Spawn a thread to handle keyboard inputs.
        // We spawn a new thread instead of using non-blocking and let guest OS to pull us so we can
        // terminate the process using Ctrl+A X whenever we like.
        std::thread::Builder::new()
            .name("console".to_owned())
            .spawn(move || {
                let mut buffer = [0; 64];
                loop {
                    // Just read a single character
                    let size = std::io::stdin().read(&mut buffer).unwrap();
                    if size == 0 {
                        // EOF. Very unlikely. In this case we will just exit
                        return;
                    }

                    let inner = match weak.upgrade() {
                        Some(v) => v,
                        None => return,
                    };
                    let mut guard = inner.state.lock();

                    for &byte in &buffer[0..size] {
                        if let Some(x) = (&mut guard.processor)(byte) {
                            guard.rx_buffer.push_back(x);
                            guard.rx_waker.drain(..).for_each(|w| w.wake());
                        }
                    }
                }
            })
            .unwrap();

        Some(ret)
    }

    /// Set the function that is to be used for processing TTY inputs. The value returned will be
    /// read by the user of this device. If `None` is received, the input is discarded.
    pub fn set_processor(&mut self, processor: impl FnMut(u8) -> Option<u8> + Send + 'static) {
        self.0.state.lock().processor = Box::new(processor);
    }
}

/// Handle SIGWINCH for tty size change notification
unsafe extern "C" fn handle_winch(
    _: libc::c_int,
    _: &mut libc::siginfo_t,
    _: &mut libc::ucontext_t,
) {
    if let Some(inner) = ACTIVE_CONSOLE.lock().as_ref() {
        let mut guard = inner.state.lock();
        guard.size_changed = true;
        guard.size_changed_wakers.drain(..).for_each(|w| w.wake());
    }
}

impl Serial for Console {
    fn poll_write(&self, _cx: &mut Context, buf: &[u8]) -> Poll<std::io::Result<usize>> {
        let mut out = std::io::stdout();
        out.write_all(buf)?;
        out.flush()?;
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_read(&self, cx: &mut Context, buf: &mut [u8]) -> Poll<std::io::Result<usize>> {
        let mut inner = self.0.state.lock();
        if inner.rx_buffer.is_empty() {
            if !inner.rx_waker.iter().any(|w| w.will_wake(cx.waker())) {
                inner.rx_waker.push(cx.waker().clone());
            }
            return Poll::Pending;
        }

        let mut len = 0;
        while len < buf.len() {
            match inner.rx_buffer.pop_front() {
                Some(key) => {
                    buf[len] = key;
                    len += 1;
                }
                None => break,
            }
        }
        Poll::Ready(Ok(len))
    }

    fn get_window_size(&self) -> std::io::Result<(u16, u16)> {
        unsafe {
            let mut size = std::mem::MaybeUninit::<libc::winsize>::uninit();
            if libc::ioctl(0, libc::TIOCGWINSZ, &mut size) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            let size = size.assume_init();
            Ok((size.ws_col, size.ws_row))
        }
    }

    fn poll_window_size_changed(&self, cx: &mut Context) -> Poll<std::io::Result<()>> {
        let mut guard = self.0.state.lock();
        if guard.size_changed {
            guard.size_changed = false;
            Poll::Ready(Ok(()))
        } else {
            if !guard.size_changed_wakers.iter().any(|w| w.will_wake(cx.waker())) {
                guard.size_changed_wakers.push(cx.waker().clone());

                WINCH.call_once(|| unsafe {
                    let mut act: libc::sigaction = std::mem::zeroed();
                    act.sa_sigaction = handle_winch as usize;
                    act.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
                    libc::sigaction(libc::SIGWINCH, &act, std::ptr::null_mut());
                });
            }
            Poll::Pending
        }
    }
}
