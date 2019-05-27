RUST_FILES = $(shell find dbt/src -type f -name '*')

default: all

.PHONY: all clean register unregister

all: codegen

clean:

codegen: dbt/target/debug/dbt
	cp $< $@

release: dbt/target/release/dbt
	cp $< $@

dbt/target/debug/dbt: $(RUST_FILES)
	cd dbt; cargo build

dbt/target/release/dbt: $(RUST_FILES)
	cd dbt; cargo build --release

register: codegen
	sudo bash -c "echo ':riscv:M::\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\xf3\x00:\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff:$(shell realpath codegen):' > /proc/sys/fs/binfmt_misc/register"

unregister:
	sudo bash -c "echo -1 > /proc/sys/fs/binfmt_misc/riscv"
