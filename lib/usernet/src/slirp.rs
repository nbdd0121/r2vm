use crate::Context;
use libslirp_sys::*;
use log::{error, warn};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, Ipv6Addr};
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Arc, Weak};
use std::task::{Poll, Waker};
use std::time::Duration;
use std::{fmt, slice, str};

extern "C" {
    fn g_free(ptr: *mut std::ffi::c_void);
}

/// Emulated network interface.
pub struct Network {
    context: Arc<Mutex<Inner>>,
    dhcp_start: Ipv4Addr,
}

impl Drop for Network {
    fn drop(&mut self) {
        self.context.lock().stop = true;
    }
}

/// The inner representation of the Network.
/// This type must be pinned on heap because we cannot move SlirpCb.
/// All operations on `Inner` need a `&mut self`, which guarantees that only one single
/// thread can operate on slirp at a given time. Callbacks are allowed to retrieve `&mut self` from
/// opaque pointers because they are called by slirp, and is known to be on the same thread.
struct Inner {
    callbacks: SlirpCb,
    context: Box<dyn Context + Send>,
    slirp: *mut Slirp,
    /// A pointer to Mutex<Inner> which allows reconstruction of `Arc<Mutex<Inner>>`
    /// by only having `&mut Inner`.
    weak: Weak<Mutex<Inner>>,
    /// Packets yet to be received,
    packet: VecDeque<Vec<u8>>,
    /// Waker to call when a new packet arrives.
    waker: Vec<Waker>,
    /// Indicate that the `Network` for this slirp is dropped, but by doing so it only signals to
    /// poll stop, and we perform cleanup only until all Arc references to Inner are dropped.
    stop: bool,
}

/// Inner can be Send between threads, but is not Sync.
unsafe impl Send for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            slirp_cleanup(self.slirp);
        }
    }
}

/// The timer instance to be used by slirp.
struct Timer {
    inner: Weak<Mutex<Inner>>,
    cb: SlirpTimerCb,
    cb_opaque: *mut c_void,
    task: Mutex<crate::util::ReplaceableTask>,
}

unsafe impl Send for Timer {}
unsafe impl Sync for Timer {}

pub trait Handler {}

extern "C" fn send_packet(buf: *const c_void, len: usize, opaque: *mut c_void) -> isize {
    let inner = unsafe { &mut *(opaque as *mut Inner) };
    let slice = unsafe { slice::from_raw_parts(buf as *const u8, len) };
    // Too many packets queued, discard
    if inner.packet.len() > 1024 {
        warn!(target: "slirp", "recv queue full, discarding packets");
        return -1;
    }
    inner.packet.push_back(slice.to_owned());
    inner.waker.drain(..).for_each(|x| x.wake());
    len as isize
}

extern "C" fn guest_error(msg: *const c_char, _opaque: *mut c_void) {
    let msg = str::from_utf8(unsafe { CStr::from_ptr(msg) }.to_bytes()).unwrap_or("");
    error!(target: "slirp", "{}", msg);
}

extern "C" fn clock_get_ns(opaque: *mut c_void) -> i64 {
    let inner = unsafe { &mut *(opaque as *mut Inner) };
    inner.context.now().as_nanos() as i64
}

extern "C" fn timer_new(
    cb: SlirpTimerCb,
    cb_opaque: *mut c_void,
    opaque: *mut c_void,
) -> *mut c_void {
    let inner = unsafe { &mut *(opaque as *mut Inner) };

    let (task, to_spawn) = crate::util::ReplaceableTask::new();
    inner.context.spawn(Box::pin(to_spawn));
    let timer =
        Arc::new(Timer { inner: inner.weak.clone(), cb, cb_opaque, task: Mutex::new(task) });

    Arc::into_raw(timer) as *mut c_void
}

extern "C" fn timer_free(timer: *mut c_void, _opaque: *mut c_void) {
    unsafe { Arc::from_raw(timer as *const Timer) };
}

extern "C" fn timer_mod(timer: *mut c_void, expire_time: i64, opaque: *mut c_void) {
    let inner = unsafe { &mut *(opaque as *mut Inner) };
    let timer = unsafe { Arc::from_raw(timer as *mut Timer) };

    // Calculate the deadline for scheduling a timer.
    let time = inner.context.now() + Duration::from_millis(expire_time.max(0) as u64);

    // Replace the task
    let future = inner.context.create_timer(time);
    let timer_clone = Arc::downgrade(&timer);
    timer.task.lock().replace(Box::pin(async move {
        future.await;
        let timer = match timer_clone.upgrade() {
            // Timer is dropped, abort
            None => return,
            Some(v) => v,
        };
        let inner = match timer.inner.upgrade() {
            // Slirp is dropped, abort
            None => return,
            Some(v) => v,
        };
        // We must execute callback while no other threads are performing slirp tasks.
        // So lock the inner.
        let _guard = inner.lock();
        // Execute callback
        if let Some(cb) = timer.cb {
            unsafe { cb(timer.cb_opaque) }
        }
    }));

    core::mem::forget(timer);
}

extern "C" fn register_poll_fd(_fd: c_int, _opaque: *mut c_void) {}

extern "C" fn unregister_poll_fd(_fd: c_int, _opaque: *mut c_void) {}

extern "C" fn notify(_opaque: *mut c_void) {}

impl Inner {
    fn pollfds_fill(&mut self, timeout: &mut u32, fds: &mut Vec<libc::pollfd>) {
        extern "C" fn callback(fd: c_int, events: c_int, opaque: *mut c_void) -> c_int {
            let fds = unsafe { &mut *(opaque as *mut Vec<libc::pollfd>) };

            let mut poll = 0;
            if events & SLIRP_POLL_IN != 0 {
                poll |= libc::POLLIN;
            }
            if events & SLIRP_POLL_OUT != 0 {
                poll |= libc::POLLOUT;
            }
            if events & SLIRP_POLL_ERR != 0 {
                poll |= libc::POLLERR;
            }
            if events & SLIRP_POLL_HUP != 0 {
                poll |= libc::POLLHUP;
            }
            if events & SLIRP_POLL_PRI != 0 {
                poll |= libc::POLLPRI;
            }
            fds.push(libc::pollfd { fd, events: poll, revents: 0 });
            (fds.len() - 1) as i32
        }

        unsafe {
            slirp_pollfds_fill(self.slirp, timeout, Some(callback), fds as *mut _ as *mut c_void);
        }
    }

    fn pollfds_poll(&mut self, error: bool, fds: &[libc::pollfd]) {
        extern "C" fn callback(idx: c_int, opaque: *mut c_void) -> c_int {
            let fds = unsafe { *(opaque as *const &[libc::pollfd]) };

            let revents = fds[idx as usize].revents;
            let mut ret = 0;
            if revents & libc::POLLIN != 0 {
                ret |= SLIRP_POLL_IN;
            }
            if revents & libc::POLLOUT != 0 {
                ret |= SLIRP_POLL_OUT;
            }
            if revents & libc::POLLERR != 0 {
                ret |= SLIRP_POLL_ERR;
            }
            if revents & libc::POLLHUP != 0 {
                ret |= SLIRP_POLL_HUP;
            }
            if revents & libc::POLLPRI != 0 {
                ret |= SLIRP_POLL_PRI;
            }
            ret
        }

        unsafe {
            slirp_pollfds_poll(
                self.slirp,
                error as i32,
                Some(callback),
                &fds as *const &[libc::pollfd] as *mut c_void,
            );
        }
    }
}

impl Network {
    /// Create a new instance of [`Network`], with supplied configuration and context.
    pub fn new(config: &crate::Config, context: impl Context + Send + 'static) -> Self {
        Self::new_internal(config, Box::new(context))
    }

    fn new_internal(config: &crate::Config, context: Box<dyn Context + Send>) -> Self {
        let (ipv4_enabled, vnetwork, vnetmask, vhost, vdhcp_start, vnameserver) = match config.ipv4
        {
            None => (
                false,
                Ipv4Addr::new(0, 0, 0, 0),
                Ipv4Addr::new(0, 0, 0, 0),
                Ipv4Addr::new(0, 0, 0, 0),
                Ipv4Addr::new(0, 0, 0, 0),
                Ipv4Addr::new(0, 0, 0, 0),
            ),
            Some(ref ipv4) => (true, ipv4.net, ipv4.mask, ipv4.host, ipv4.dhcp_start, ipv4.dns),
        };

        let (ipv6_enabled, vprefix_addr6, vprefix_len, vhost6, vnameserver6) = match config.ipv6 {
            None => (
                false,
                Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0),
                0,
                Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0),
                Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0),
            ),
            Some(ref ipv6) => (true, ipv6.prefix, ipv6.prefix_len, ipv6.host, ipv6.dns),
        };

        let (tftp_server_name, tftp_path, tftp_bootfile) = match config.tftp {
            None => (None, None, None),
            Some(ref tftp) => (tftp.name.as_ref(), Some(tftp.root.clone()), tftp.bootfile.as_ref()),
        };

        // Convert options to FFI types
        let cstr_vdns: Vec<_> =
            config.dns_suffixes.iter().map(|arg| CString::new(arg.as_bytes()).unwrap()).collect();
        let mut p_vdns: Vec<_> =
            cstr_vdns.iter().map(|arg| arg.as_ptr() as *const c_char).collect();
        p_vdns.push(std::ptr::null());

        let as_ptr = |p: &Option<CString>| p.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
        let tftp_path = tftp_path.and_then(|s| CString::new(s.to_string_lossy().into_owned()).ok());
        let vhostname = config.hostname.as_ref().and_then(|s| CString::new(s.as_bytes()).ok());
        let tftp_server_name = tftp_server_name.and_then(|s| CString::new(s.as_bytes()).ok());
        let tftp_bootfile = tftp_bootfile.and_then(|s| CString::new(s.as_bytes()).ok());
        let vdomainname = config.domainname.as_ref().and_then(|s| CString::new(s.as_bytes()).ok());

        // Create inner. `Arc<Mutex<Inner>>` is enough to pin Inner.
        let inner = Arc::new(Mutex::new(Inner {
            callbacks: SlirpCb {
                send_packet: Some(send_packet),
                guest_error: Some(guest_error),
                clock_get_ns: Some(clock_get_ns),
                timer_new: Some(timer_new),
                timer_free: Some(timer_free),
                timer_mod: Some(timer_mod),
                register_poll_fd: Some(register_poll_fd),
                unregister_poll_fd: Some(unregister_poll_fd),
                notify: Some(notify),
            },
            context,
            slirp: std::ptr::null_mut(),
            weak: Weak::new(),
            packet: VecDeque::new(),
            waker: Vec::new(),
            stop: false,
        }));

        let mut lock = inner.lock();
        // Assign weak pointer. This must be done before slirp_init as timer_create may be called
        // when initing.
        lock.weak = Arc::downgrade(&inner);
        let callback = &lock.callbacks as *const _;
        let ptr = &*lock as *const Inner as *mut c_void;
        drop(lock);

        let context = unsafe {
            slirp_init(
                config.restricted as i32,
                ipv4_enabled,
                vnetwork.into(),
                vnetmask.into(),
                vhost.into(),
                ipv6_enabled,
                vprefix_addr6.into(),
                vprefix_len,
                vhost6.into(),
                as_ptr(&vhostname),
                as_ptr(&tftp_server_name),
                as_ptr(&tftp_path),
                as_ptr(&tftp_bootfile),
                vdhcp_start.into(),
                vnameserver.into(),
                vnameserver6.into(),
                p_vdns.as_mut_ptr(),
                as_ptr(&vdomainname),
                callback,
                ptr,
            )
        };
        inner.lock().slirp = context;

        // Start the poll loop for this slirp
        let arc_clone = inner.clone();
        std::thread::Builder::new()
            .name("slirp".to_owned())
            .spawn(move || poll_loop(arc_clone))
            .unwrap();

        assert!(!context.is_null());
        Network { context: inner, dhcp_start: vdhcp_start }
    }

    /// Send a packet to the interface.
    ///
    /// # Result
    /// The number of bytes transmitted is returned.
    pub fn poll_send(
        &self,
        _cx: &mut std::task::Context,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        unsafe {
            slirp_input(self.context.lock().slirp, buf.as_ptr(), buf.len() as i32);
        }
        Poll::Ready(Ok(buf.len()))
    }

    /// Receive a packet from the interface. If the buffer is not large enough, the packet gets
    /// truncated.
    ///
    /// # Result
    /// The number of bytes copied into the buffer is returned.
    pub fn poll_recv(
        &self,
        cx: &mut std::task::Context,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        let mut guard = self.context.lock();
        match guard.packet.pop_front() {
            Some(recv_buf) => {
                drop(guard);
                let len = buf.len().min(recv_buf.len());
                buf[..len].copy_from_slice(&recv_buf[..len]);
                Poll::Ready(Ok(len))
            }
            None => {
                if !guard.waker.iter().any(|w| w.will_wake(cx.waker())) {
                    guard.waker.push(cx.waker().clone());
                }
                Poll::Pending
            }
        }
    }

    /// Forward a host port to a guest port.
    pub fn add_host_forward(
        &self,
        udp: bool,
        host_addr: Ipv4Addr,
        host_port: u16,
        guest_addr: Option<Ipv4Addr>,
        guest_port: u16,
    ) -> std::io::Result<()> {
        let lock = self.context.lock();
        if unsafe {
            slirp_add_hostfwd(
                lock.slirp,
                udp as _,
                host_addr.into(),
                host_port as _,
                guest_addr.unwrap_or(self.dhcp_start).into(),
                guest_port as _,
            )
        } < 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// Serialise the current state to a [`Write`] stream. The bytes written into `writer` are
    /// intended only for [`load_state`](#method.load_state). This method is provided for
    /// serialisation and migration, and the content should not be interpreted or modified
    /// otherwise.
    ///
    /// # Result
    /// Upon successful serialisation, the number of bytes written is returned.
    pub fn save_state(&self, writer: &mut dyn Write) -> std::io::Result<usize> {
        let version: i32 = unsafe { slirp_state_version() };
        writer.write_all(&version.to_be_bytes())?;

        let mut res = Ok(4);
        let mut pair = (&mut res, writer);

        extern "C" fn callback(buf: *const c_void, len: usize, opaque: *mut c_void) -> isize {
            let pair =
                unsafe { &mut *(opaque as *mut (&mut std::io::Result<usize>, &mut dyn Write)) };
            let slice = unsafe { slice::from_raw_parts(buf as *const u8, len) };

            match pair.1.write(slice) {
                Ok(n) => {
                    *pair.0 = Ok(*pair.0.as_ref().unwrap() + n);
                    n as isize
                }
                Err(e) => {
                    *pair.0 = Err(e);
                    -1
                }
            }
        }

        unsafe {
            slirp_state_save(
                self.context.lock().slirp,
                Some(callback),
                &mut pair as *mut _ as *mut c_void,
            );
        }

        res
    }

    /// Deserialise the state from a [`Read`] stream. The data should be previously serialised
    /// using [`save_state`](#method.save_state). It should saved from the same or an earlier,
    /// semver-compatible version of this library.
    ///
    /// # Result
    /// Upon successful deserialisation, the number of bytes read is returned.
    pub fn load_state(&self, reader: &mut dyn Read) -> std::io::Result<usize> {
        let mut version = [0; 4];
        reader.read_exact(&mut version)?;
        let version_id = i32::from_be_bytes(version);

        let mut res = Ok(0);
        let mut pair = (&mut res, reader);

        extern "C" fn callback(buf: *mut c_void, len: usize, opaque: *mut c_void) -> isize {
            let pair =
                unsafe { &mut *(opaque as *mut (&mut std::io::Result<usize>, &mut dyn Read)) };
            let slice = unsafe { slice::from_raw_parts_mut(buf as *mut u8, len) };

            match pair.1.read(slice) {
                Ok(n) => {
                    *pair.0 = Ok(*pair.0.as_ref().unwrap() + n);
                    n as isize
                }
                Err(e) => {
                    *pair.0 = Err(e);
                    -1
                }
            }
        }

        let ret = unsafe {
            slirp_state_load(
                self.context.lock().slirp,
                version_id,
                Some(callback),
                &mut pair as *mut _ as *mut c_void,
            )
        };

        if ret < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "cannot deserialise the state",
            ));
        }

        res
    }
}

/// The main poll loop for an Inner.
fn poll_loop(inner: Arc<Mutex<Inner>>) {
    let mut inner = inner.lock();
    loop {
        let mut vec = Vec::new();
        let mut timeout = i32::max_value() as u32;
        inner.pollfds_fill(&mut timeout, &mut vec);

        let err = parking_lot::MutexGuard::unlocked(&mut inner, || unsafe {
            libc::poll(vec.as_mut_ptr(), vec.len() as _, timeout as _)
        });

        if inner.stop {
            return;
        }
        inner.pollfds_poll(err == -1, &vec);
    }
}

impl fmt::Debug for Network {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let ptr = unsafe { slirp_connection_info(self.context.lock().slirp) };
        let ret = write!(fmt, "{}", unsafe { CStr::from_ptr(ptr) }.to_string_lossy());
        unsafe { g_free(ptr as _) }
        ret
    }
}
