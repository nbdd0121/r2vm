# R2VM

R2VM is the **R**ust for **R**ISC-V **V**irtual **M**achine.

R2VM is a full-system, multi-core, cycle-level simulator, with binary translation to provide high performance. It can run RISC-V supervisor and userspace software on AMD64 Linux machines.

## Citation

Some approaches of this simulator are described in [our paper](https://carrv.github.io/2020/papers/CARRV2020_paper_6_Guo.pdf).

If you would like to cite this work, you may cite as
> Xuan Guo and Robert Mullins. 2020. Accelerate Cycle-Level Full-System Simulation of Multi-Core RISC-V Systems with Binary Translation. In *Fourth Workshop on Computer Architecture Research with RISC-V*

Or use the following bibtex snippet:
```bibtex
@inproceedings{guo2020r2vm,
  title={Accelerate Cycle-Level Full-System Simulation of Multi-Core RISC-V Systems with Binary Translation},
  author={Guo, Xuan and Mullins, Robert},
  year={2020},
  booktitle={Fourth Workshop on Computer Architecture Research with RISC-V}
}
```

## Installation

R2VM is written in Rust. To compile it, you will need to [install Rust](https://rustup.rs/) first, then:

```bash
git clone https://github.com/nbdd0121/r2vm.git
cd rv2m
cargo build --release
```

You can run R2VM with `cargo run --release` or locate the compiled binary at `target/release/r2vm`.

If you don't want to modify code, you can also use `cargo install --git https://github.com/nbdd0121/r2vm.git` to install it directly.

## Configuration

To run userspace programs, you can simply run `r2vm [name of binary]`. To run kernels such as Linux, you will need to use a configuration file. Configuration must be in toml format.

Here is an example configuration file:

```toml
core = 2
memory = 1024

kernel = "linux/vmlinux"
cmdline = "console=hvc0 rw root=/dev/vda"

# Turn on RTC. You'll need to enable Xilinx MPSoC RTC driver in Linux
[rtc]

[console]
type = "virtio"

# Create a block device
[[drive]]
path = "rootfs.img"

# Create a RNG device. Important to speed up Linux booting.
[[random]]
type = "os"

# Create a 9p sharing
[[share]]
tag = "share"
path = "share"

[[network]]
# Forward port 22222 to the guest's SSH port
[[network.forward]]
protocol = "tcp"
host_port = 22222
guest_port = 22

```

Use `r2vm config.toml` to run R2VM with supervisor software. For detailed possible configuration options, check `src/config.rs`.
