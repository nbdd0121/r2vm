RUST_FILES = $(shell find src lib -type f -name '*')

default: all

.PHONY: all clean register unregister

all: codegen

clean:

codegen: target/debug/rvm
	cp $< $@

release: target/release/rvm
	cp $< $@

target/debug/rvm: $(RUST_FILES)
	cargo build

target/release/rvm: $(RUST_FILES)
	cargo build --release

register: codegen
	sudo bash -c "echo ':riscv:M::\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\xf3\x00:\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff:$(shell realpath codegen):' > /proc/sys/fs/binfmt_misc/register"

unregister:
	sudo bash -c "echo -1 > /proc/sys/fs/binfmt_misc/riscv"

doc:
	cargo doc --no-deps --workspace --all-features

clippy:
	cargo clippy -- --allow clippy::cast_lossless --allow clippy::unreadable_literal --allow clippy::trivially_copy_pass_by_ref -A clippy::identity_op -A clippy::cognitive_complexity -A clippy::new_without_default -A clippy::len_without_is_empty -A clippy::verbose_bit_mask
