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
    slirp: slirp::context::Context<Handler>,
    stop: bool,
}

struct Handler(Sender<Vec<u8>>);

#[allow(unused_variables)]
impl slirp::context::Handler for Handler {
    type Timer = ();

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

    fn notify(&mut self) {
        unimplemented!()
    }

    fn timer_new(&mut self, func: Box<dyn FnMut()>) -> Box<Self::Timer> {
        unimplemented!()
    }

    fn timer_mod(&mut self, timer: &mut Box<Self::Timer>, expire_time: i64) {
        unimplemented!()
    }

    fn timer_free(&mut self, timer: Box<Self::Timer>) {
        unimplemented!()
    }
}

// The main poll loop
fn poll_loop(slirp: Arc<Mutex<Inner>>) {
    let mut guard = slirp.lock().unwrap();
    loop {
        if guard.stop { return }

        let mut vec = Vec::new();
        let mut timeout = i32::max_value() as u32;
        guard.slirp.pollfds_fill(&mut timeout, |fd, events| {
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

        guard.slirp.pollfds_poll(err == -1, |idx| {
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
        let slirp = slirp::context::Context::new(&slirp_opt, Handler(tx));
        let inner = Arc::new(Mutex::new(Inner {
            slirp,
            stop: false,
        }));
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
        self.inner.lock().unwrap().slirp.input(buf);
        Ok(buf.len())
    }

    fn recv(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        let buffer = self.rx.recv().unwrap();
        let len = buffer.len().min(buf.len());
        buf[..len].copy_from_slice(&buffer[..len]);
        Ok(len)
    }
}
