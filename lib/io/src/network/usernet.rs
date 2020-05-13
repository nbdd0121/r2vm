use crate::RuntimeContext;
use std::future::Future;
use std::net::Ipv4Addr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

/// Network device simulated in userspace.
pub struct Usernet {
    inner: usernet::Network,
}

struct EventLoopContext(Arc<dyn RuntimeContext>);

impl usernet::Context for EventLoopContext {
    fn now(&mut self) -> Duration {
        self.0.now()
    }

    fn create_timer(&mut self, time: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        self.0.create_timer(time)
    }

    fn spawn(&mut self, future: Pin<Box<dyn Future<Output = ()> + Send>>) {
        self.0.spawn(future)
    }
}

impl Usernet {
    /// Create a new `Usernet` instance with the given runtime context.
    pub fn new(ctx: Arc<dyn RuntimeContext>) -> Self {
        let usernet_opt = usernet::Config {
            restricted: false,
            ipv4: Some(Default::default()),
            ipv6: Some(Default::default()),
            hostname: None,
            tftp: None,
            dns_suffixes: Vec::new(),
            domainname: None,
        };
        let usernet = usernet::Network::new(&usernet_opt, EventLoopContext(ctx));
        Self { inner: usernet }
    }

    /// Forward a host port to a guest port.
    pub fn add_host_forward(
        &self,
        udp: bool,
        host_addr: Ipv4Addr,
        host_port: u16,
        guest_port: u16,
    ) -> std::io::Result<()> {
        self.inner.add_host_forward(udp, host_addr, host_port, None, guest_port)
    }
}

impl super::Network for Usernet {
    fn poll_send(&self, cx: &mut Context, buf: &[u8]) -> Poll<std::io::Result<usize>> {
        self.inner.poll_send(cx, buf)
    }

    fn poll_recv(&self, cx: &mut Context, buf: &mut [u8]) -> Poll<std::io::Result<usize>> {
        self.inner.poll_recv(cx, buf)
    }
}
