use std::net::{Ipv4Addr, Ipv6Addr};
use std::path::PathBuf;

/// IPv4-related configuration.
#[derive(Debug, Clone)]
pub struct Ipv4Config {
    /// IP network. This field is used in conjuction with [`mask`](#structfield.mask) to
    /// denote the subnet.
    pub net: Ipv4Addr,
    /// IP network mask.
    pub mask: Ipv4Addr,
    /// Guest-visible address of the host.
    pub host: Ipv4Addr,
    /// The first of the 16 IPs the built-in DHCP server can assign.
    pub dhcp_start: Ipv4Addr,
    /// Guest-visible address of the virtual nameserver.
    pub dns: Ipv4Addr,
}

impl Default for Ipv4Config {
    fn default() -> Self {
        Self {
            net: Ipv4Addr::new(10, 0, 2, 0),
            mask: Ipv4Addr::new(255, 255, 255, 0),
            host: Ipv4Addr::new(10, 0, 2, 2),
            dhcp_start: Ipv4Addr::new(10, 0, 2, 15),
            dns: Ipv4Addr::new(10, 0, 2, 3),
        }
    }
}

/// IPv6-related configuration.
#[derive(Debug, Clone)]
pub struct Ipv6Config {
    /// IPv6 network prefix for the subnet.
    pub prefix: Ipv6Addr,
    /// IPv6 network prefix length.
    pub prefix_len: u8,
    /// Guest-visible IPv6 address of the host.
    pub host: Ipv6Addr,
    /// Guest-visible address of the virtual nameserver.
    pub dns: Ipv6Addr,
}

impl Default for Ipv6Config {
    fn default() -> Self {
        Self {
            prefix: Ipv6Addr::new(0xfec0, 0, 0, 0, 0, 0, 0, 0),
            prefix_len: 64,
            host: Ipv6Addr::new(0xfec0, 0, 0, 0, 0, 0, 0, 2),
            dns: Ipv6Addr::new(0xfec0, 0, 0, 0, 0, 0, 0, 3),
        }
    }
}

/// TFTP-related configuration.
#[derive(Debug)]
pub struct TftpConfig {
    /// RFC2132 "TFTP server name" string.
    pub name: Option<String>,
    /// root directory of the built-in TFTP server.
    pub root: PathBuf,
    /// BOOTP filename, for use with tftp.
    pub bootfile: Option<String>,
}

/// Configuration needed to create a [`Network`](super::Network).
#[derive(Debug)]
pub struct Config {
    /// Isolate guest from host.
    pub restricted: bool,
    /// Config related to IPv4. If set to [`None`], IPv4 support will be disabled.
    pub ipv4: Option<Ipv4Config>,
    /// Config related to IPv6. If set to [`None`], IPv6 support will be disabled.
    pub ipv6: Option<Ipv6Config>,
    /// Client hostname reported by the builtin DHCP server.
    pub hostname: Option<String>,
    /// Config related to TFTP. If set to [`None`], TFTP support will be disabled.
    pub tftp: Option<TftpConfig>,
    /// List of DNS suffixes to search, passed as DHCP option to the guest.
    pub dns_suffixes: Vec<String>,
    /// Guest-visible domain name of the virtual nameserver from DHCP server
    pub domainname: Option<String>,
}
