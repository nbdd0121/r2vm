use std::sync::Arc;
use std::sync::Mutex;
use libslirp as slirp;
use std::os::unix::io::RawFd;

use crossbeam_channel::{bounded, Sender, Receiver};

pub struct Slirp {
    inner: Arc<Mutex<Inner>>,
    rx: Receiver<Vec<u8>>,
}

struct Inner {
    slirp: Option<slirp::context::Context<Handler>>,
    stop: bool,
}

struct Handler(Sender<Vec<u8>>, Arc<Mutex<Inner>>);

struct Timer {
    inner: Arc<Mutex<Inner>>,
    func: Box<dyn FnMut()>,
    time: u64,
}

unsafe impl Send for Timer {}

impl Timer {
    fn check(&mut self) {
        if crate::EVENT_LOOP.time() >= self.time {
            let _guard = self.inner.lock().unwrap();
            (self.func)();
            self.time = u64::max_value()
        }
    }
}

#[allow(unused_variables)]
impl slirp::context::Handler for Handler {
    type Timer = Arc<Mutex<Timer>>;

    fn clock_get_ns(&mut self) -> i64 {
        crate::EVENT_LOOP.time() as i64 * 1000
    }

    fn send_packet(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // If the other end is disconnected, ignore error because we are exiting as well.
        let _ = self.0.send(buf.to_owned());
        Ok(buf.len())
    }

    fn register_poll_fd(&mut self, _fd: RawFd) {}

    fn unregister_poll_fd(&mut self, _fd: RawFd) {}

    fn guest_error(&mut self, msg: &str) {
        error!(target: "slirp", "{}", msg);
    }

    fn notify(&mut self) {}

    fn timer_new(&mut self, func: Box<dyn FnMut()>) -> Box<Self::Timer> {
        Box::new(Arc::new(Mutex::new(Timer {
            inner: self.1.clone(),
            func,
            time: u64::max_value(),
        })))
    }

    fn timer_mod(&mut self, timer: &mut Box<Self::Timer>, expire_time: i64) {
        // Compute the new expiration time
        let time = crate::EVENT_LOOP.time() + (expire_time * 1000) as u64;
        let mut inner = timer.lock().unwrap();
        inner.time = time;
        let timer_clone = timer.clone();
        crate::EVENT_LOOP.queue_time(time, Box::new(move || timer_clone.lock().unwrap().check()));
    }

    fn timer_free(&mut self, timer: Box<Self::Timer>) {
        timer.lock().unwrap().time = u64::max_value();
    }
}

// The main poll loop
fn poll_loop(slirp: Arc<Mutex<Inner>>) {
    let mut guard = slirp.lock().unwrap();
    loop {
        if guard.stop { return }

        let mut vec = Vec::new();
        let mut timeout = i32::max_value() as u32;
        guard.slirp.as_mut().unwrap().pollfds_fill(&mut timeout, |fd, events| {
            let mut poll = 0;
            if events.has_in() { poll |= libc::POLLIN; }
            if events.has_out() { poll |= libc::POLLOUT; }
            if events.has_err() { poll |= libc::POLLERR; }
            if events.has_hup() { poll |= libc::POLLHUP; }
            if events.has_pri() { poll |= libc::POLLPRI; }
            vec.push(libc::pollfd {
                fd,
                events: poll,
                revents: 0,
            });
            return (vec.len() - 1) as i32
        });

        std::mem::drop(guard);
        let err = unsafe { libc::poll(vec.as_mut_ptr(), vec.len() as _, timeout as _) };
        guard = slirp.lock().unwrap();

        guard.slirp.as_mut().unwrap().pollfds_poll(err == -1, |idx| {
            let revents = vec[idx as usize].revents;
            use slirp::context::PollEvents;
            let mut ret = PollEvents::empty();
            if revents & libc::POLLIN != 0 { ret |= PollEvents::poll_in() }
            if revents & libc::POLLOUT != 0 { ret |= PollEvents::poll_out() }
            if revents & libc::POLLERR != 0 { ret |= PollEvents::poll_err() }
            if revents & libc::POLLHUP != 0 { ret |= PollEvents::poll_hup() }
            if revents & libc::POLLPRI != 0 { ret |= PollEvents::poll_pri() }
            return ret
        });
    }
}

impl Slirp {
    pub fn new() -> Self {
        let slirp_opt = slirp::config::Config {
            restricted: false,
            ipv4: Some(Default::default()),
            ipv6: None,
            hostname: None,
            tftp: None,
            dns_suffixes: Vec::new(),
            domainname: None,
        };
        let (tx, rx) = bounded(2);
        let inner = Arc::new(Mutex::new(Inner { slirp: None, stop: false }));
        let slirp = slirp::context::Context::new(&slirp_opt, Handler(tx, inner.clone()));
        inner.lock().unwrap().slirp = Some(slirp);
        let clone = inner.clone();
        std::thread::spawn(move || poll_loop(clone));
        Self {
            inner,
            rx,
        }
    }
}

impl Drop for Slirp {
    fn drop(&mut self) {
        self.inner.lock().unwrap().stop = true;
    }
}

impl super::Network for Slirp {
    fn send(&self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.lock().unwrap().slirp.as_mut().unwrap().input(buf);
        Ok(buf.len())
    }

    fn recv(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        let buffer = self.rx.recv().unwrap();
        let len = buffer.len().min(buf.len());
        buf[..len].copy_from_slice(&buffer[..len]);
        Ok(len)
    }
}