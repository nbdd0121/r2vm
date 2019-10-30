use async_trait::async_trait;
use std::future::Future;
use std::net::Ipv4Addr;
use std::pin::Pin;

pub struct Usernet {
    inner: usernet::Network,
}

struct EventLoopContext;

impl usernet::Context for EventLoopContext {
    fn now(&mut self) -> u64 {
        crate::event_loop().time() * 1000
    }

    fn create_timer(&mut self, time: u64) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(crate::event_loop().on_time(time / 1000))
    }

    fn spawn(&mut self, future: Pin<Box<dyn Future<Output = ()> + Send>>) {
        crate::event_loop().spawn(future);
    }
}

impl Usernet {
    pub fn new() -> Self {
        let usernet_opt = usernet::Config {
            restricted: false,
            ipv4: Some(Default::default()),
            ipv6: Some(Default::default()),
            hostname: None,
            tftp: None,
            dns_suffixes: Vec::new(),
            domainname: None,
        };
        let usernet = usernet::Network::new(&usernet_opt, EventLoopContext);
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

#[async_trait]
impl super::Network for Usernet {
    async fn send(&self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.send(buf).await
    }

    async fn recv(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.recv(buf).await
    }
}
