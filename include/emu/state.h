#ifndef EMU_STATE_H
#define EMU_STATE_H

#include <memory>
#include <stdexcept>

#include "emu/typedef.h"

namespace riscv {

struct Context;

};

namespace emu {

namespace state {

// The program/data break of the address space. original_brk represents the initial brk from information gathered
// in elf. Both values are set initially to original_brk by elf_loader, and original_brk should not be be changed.
// A constraint original_brk <= brk must be satisified.
extern reg_t original_brk;
extern reg_t brk;
extern reg_t heap_start;
extern reg_t heap_end;

struct flags_t {

    // Whether direct memory access or call to helper should be generated for guest memory access.
    bool no_direct_memory_access;

    // A flag to determine whether to trace all system calls. If true then all guest system calls will be logged.
    bool strace;

    // A flag to determine whether to print instruction out when it is decoded.
    bool disassemble;

    // A flag to determine whether instret should be updated precisely in binary translated code.
    bool no_instret;

    // A flag to determine whether correctness in case of segmentation fault should be dealt strictly.
    bool strict_exception;

    // A flag to determine whether PHI nodes should be introduced to the graph by load elimination.
    bool enable_phi;

    // Whether compilation performance counters should be enabled.
    bool monitor_performance;

    // The actual path of the executable. Needed to redirect /proc/self/*
    const char* exec_path;

    // Path of sysroot. When the guest application tries to open a file, and the corresponding file exists in sysroot,
    // it will be redirected.
    const char* sysroot;

    // Upper limit of number of blocks that can be placed in a region.
    uint32_t region_limit;

    // Threshold beyond which the IR DBT will start working
    uint32_t compile_threshold;

    bool user_only;
};

extern "C" flags_t& get_flags();

}

// Load elf, and setup auxillary vectors.
reg_t load_elf(const char *filename, reg_t& sp);

// Load a binary to specified location
reg_t load_bin(const char *filename, reg_t location);

}

#endif
