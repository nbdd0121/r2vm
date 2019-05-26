LD = g++-7
CXX = g++-7

LD_FLAGS = -g -pie -Wl,-Ttext-segment=0x7fff00000000
CXX_FLAGS = -g -fPIE -std=c++17 -fconcepts -Wall -Wextra -Iinclude/ -Og -fno-stack-protector

LD_RELEASE_FLAGS = -g -flto -march=native -O2 -pie -Wl,-Ttext-segment=0x7fff00000000
CXX_RELEASE_FLAGS = -g -fPIE -std=c++17 -fconcepts -Wall -Wextra -Iinclude/ -O2 -march=native -DRELEASE=1 -flto -fno-stack-protector

RUST_FILES = $(shell find dbt/src -type f -name '*')

OBJS = \
	emu/mmu.o \
	main/main.o \
	riscv/decoder.o \
	riscv/step.o \
	softfp/float.o

default: all

.PHONY: all clean register unregister

all: codegen

clean:
	rm $(patsubst %,bin/%,$(OBJS) $(OBJS:.o=.d))

codegen: $(patsubst %,bin/%,$(OBJS)) $(LIBS) dbt/target/debug/libdbt.a
	$(LD) $(LD_FLAGS) $^ -o $@ -lpthread -ldl

release: $(patsubst %,bin/release/%,$(OBJS)) $(LIBS) dbt/target/release/libdbt.a
	$(LD) $(LD_RELEASE_FLAGS) $^ -o $@ -lpthread -ldl

-include $(patsubst %,bin/%,$(OBJS:.o=.d))
-include $(patsubst %,bin/release/%,$(OBJS:.o=.d))

# Special rule for feature testing
bin/feature.o: src/feature.cc
	@mkdir -p $(dir $@)
	$(CXX) -c -MMD -MP $(CXX_FLAGS) $< -o $@

bin/%.o: src/%.cc bin/feature.o
	@mkdir -p $(dir $@)
	$(CXX) -c -MMD -MP $(CXX_FLAGS) $< -o $@

bin/%.o: src/%.s
	@mkdir -p $(dir $@)
	$(CXX) -c -MMD -MP $(CXX_FLAGS) $< -o $@

bin/release/%.o: src/%.cc bin/feature.o
	@mkdir -p $(dir $@)
	$(CXX) -c -MMD -MP $(CXX_RELEASE_FLAGS) $< -o $@

bin/release/%.o: src/%.s
	@mkdir -p $(dir $@)
	$(CXX) -c -MMD -MP $(CXX_RELEASE_FLAGS) $< -o $@

dbt/target/debug/libdbt.a: $(RUST_FILES)
	cd dbt; cargo build

dbt/target/release/libdbt.a: $(RUST_FILES)
	cd dbt; cargo build --release

register: codegen
	sudo bash -c "echo ':riscv:M::\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\xf3\x00:\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff:$(shell realpath codegen):' > /proc/sys/fs/binfmt_misc/register"

unregister:
	sudo bash -c "echo -1 > /proc/sys/fs/binfmt_misc/riscv"
