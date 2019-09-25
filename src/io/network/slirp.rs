use std::sync::{Arc, Weak};
use parking_lot::Mutex;
use libslirp as slirp;
use std::os::unix::io::RawFd;

use std::sync::mpsc::{sync_channel, SyncSender, Receiver};

pub struct Slirp {
    inner: Arc<Inner>,
    rx: Mutex<Receiver<Vec<u8>>>,
}

struct Inner {
    slirp: slirp::context::Context<Handler>,
    stop: Mutex<bool>,
}

struct Handler(SyncSender<Vec<u8>>, Mutex<Weak<Inner>>);

struct Timer {
    inner: Weak<Inner>,
    func: Box<dyn FnMut()>,
    time: u64,
}

unsafe impl Send for Timer {}

impl Timer {
    fn check(&mut self) {
        if crate::EVENT_LOOP.time() >= self.time {
            let inner = match self.inner.upgrade() {
                Some(v) => v,
                None => return,
            };
            inner.slirp.fire_timer(&mut self.func);
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

    fn timer_new(&mut self, func: Box<dyn FnMut()>) -> Self::Timer {
        Arc::new(Mutex::new(Timer {
            inner: self.1.lock().clone(),
            func,
            time: u64::max_value(),
        }))
    }

    fn timer_mod(&mut self, timer: &mut Self::Timer, expire_time: i64) {
        // Compute the new expiration time
        let time = crate::EVENT_LOOP.time() + (expire_time * 1000) as u64;
        let mut inner = timer.lock();
        inner.time = time;
        let timer_clone = Arc::downgrade(&timer);
        crate::EVENT_LOOP.queue_time(time, Box::new(move || {
            if let Some(timer) = timer_clone.upgrade() {
                timer.lock().check()
            }
        }));
    }

    fn timer_free(&mut self, timer: Self::Timer) {
        timer.lock().time = u64::max_value();
    }
}

// The main poll loop
fn poll_loop(inner: Arc<Inner>) {
    loop {
        let mut vec = Vec::new();
        let mut timeout = i32::max_value() as u32;
        inner.slirp.pollfds_fill(&mut timeout, |fd, events| {
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

        let err = unsafe { libc::poll(vec.as_mut_ptr(), vec.len() as _, timeout as _) };

        if *inner.stop.lock() { return }
        inner.slirp.pollfds_poll(err == -1, |idx| {
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
            ipv6: Some(Default::default()),
            hostname: None,
            tftp: None,
            dns_suffixes: Vec::new(),
            domainname: None,
        };
        let (tx, rx) = sync_channel(1024);
        let slirp = slirp::context::Context::new(&slirp_opt, Handler(tx, Mutex::new(Weak::new())));
        let inner = Arc::new(Inner { slirp, stop: Mutex::new(false) });
        *inner.slirp.get_handler().1.lock() = Arc::downgrade(&inner);
        let clone = inner.clone();
        std::thread::Builder::new()
            .name("slirp".to_owned())
            .spawn(move || poll_loop(clone))
            .unwrap();
        Self {
            inner,
            rx: Mutex::new(rx),
        }
    }
}

impl Drop for Slirp {
    fn drop(&mut self) {
        *self.inner.stop.lock() = true;
    }
}

impl super::Network for Slirp {
    fn send(&self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.slirp.input(buf);
        Ok(buf.len())
    }

    fn recv(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        let buffer = self.rx.lock().recv().unwrap();
        let len = buffer.len().min(buf.len());
        buf[..len].copy_from_slice(&buffer[..len]);
        Ok(len)
    }
}
