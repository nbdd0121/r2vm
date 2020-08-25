use io::system::config::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    pub plic: DeviceConfig<PlicConfig>,

    #[serde(default)]
    pub console: Option<DeviceConfig<ConsoleConfig>>,

    /// Whether a RTC device should be instantiated.
    #[serde(default)]
    pub rtc: Option<DeviceConfig<RTCConfig>>,

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

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct PlicConfig {}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct ClintConfig {}
