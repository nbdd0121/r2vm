use serde::{Deserialize, Serialize};
use std::net::Ipv4Addr;
use std::path::PathBuf;

fn return_true() -> bool {
    true
}

fn default_core() -> usize {
    4
}
fn default_memory() -> usize {
    1024
}
fn default_cmdline() -> String {
    "console=hvc0 rw root=/dev/vda".to_owned()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    /// Number of cores.
    #[serde(default = "default_core")]
    pub core: usize,

    /// Location of kernel.
    /// It should be of ELF format, not containing any firmware.
    pub kernel: PathBuf,

    /// Location of firmware.
    /// It should be of ELF format. If firmware is present, R2VM will start with machine mode.
    /// If firmware is not present, R2VM will start in supervisor mode and provide SBI interface.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub firmware: Option<PathBuf>,

    /// Memory size, in MiB.
    #[serde(default = "default_memory")]
    pub memory: usize,

    /// Linux boot command line
    #[serde(default = "default_cmdline")]
    pub cmdline: String,

    #[serde(default)]
    pub clint: Option<DeviceConfig<ClintConfig>>,

    #[serde(default)]
    pub console: Option<DeviceConfig<ConsoleConfig>>,

    /// Whether a RTC device should be instantiated.
    #[serde(default = "return_true")]
    pub rtc: bool,

    /// Block devices
    #[serde(default)]
    pub drive: Vec<DeviceConfig<DriveConfig>>,

    /// Random devices
    #[serde(default)]
    pub random: Vec<DeviceConfig<RandomConfig>>,

    /// 9p file sharing
    #[serde(default)]
    pub share: Vec<DeviceConfig<ShareConfig>>,

    /// Network adapters
    #[serde(default)]
    pub network: Vec<DeviceConfig<NetworkConfig>>,
}

/// Specifies which particular address is to be used for an IO device
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct DeviceConfig<T> {
    #[serde(default)]
    pub io_base: Option<usize>,

    #[serde(flatten)]
    pub config: T,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct ClintConfig {}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum ConsoleType {
    Virtio,
    NS16550,
}

impl Default for ConsoleType {
    fn default() -> Self {
        ConsoleType::Virtio
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ConsoleConfig {
    /// Whether a virtio console device should be exposed. Note that under current implementation,
    /// sbi_get_char will always produce -1 after virtio is initialised.
    #[serde(default)]
    pub r#type: ConsoleType,

    /// Whether resizing feature should be enabled. Only useful when virtio is enabled.
    #[serde(default = "return_true")]
    pub resize: bool,
}

impl Default for ConsoleConfig {
    fn default() -> Self {
        ConsoleConfig { r#type: ConsoleType::Virtio, resize: true }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DriveConfig {
    /// Whether changes should be written back to the file.
    #[serde(default)]
    pub shadow: bool,

    /// Path to backing file.
    pub path: PathBuf,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum RandomType {
    Pseudo,
    OS,
}

fn default_seed() -> u64 {
    0xcafebabedeadbeef
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RandomConfig {
    pub r#type: RandomType,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ShareConfig {
    /// 9p sharing tag
    pub tag: String,

    /// Path to the shared directory
    pub path: PathBuf,
}

fn default_host_addr() -> Ipv4Addr {
    Ipv4Addr::new(127, 0, 0, 1)
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ForwardProtocol {
    Tcp,
    Udp,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ForwardConfig {
    /// The protocol to forward.
    pub protocol: ForwardProtocol,

    /// The IP address that host listens to.
    #[serde(default = "default_host_addr")]
    pub host_addr: Ipv4Addr,

    /// The port number that host listens to.
    pub host_port: u16,

    /// The port number that guest listens on.
    pub guest_port: u16,
}

fn default_net_type() -> String {
    "virtio".to_owned()
}

fn default_mac() -> String {
    "02:00:00:00:00:01".to_owned()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NetworkConfig {
    /// Device type
    #[serde(default = "default_net_type")]
    pub r#type: String,

    /// MAC address. For convience, we first parse it as string.
    #[serde(default = "default_mac")]
    pub mac: String,

    /// Forward configurations.
    #[serde(default)]
    pub forward: Vec<ForwardConfig>,
}
