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

// All parts of the emulator will share a global state. Originally global variable is avoided, but by doing so many
// objects need to hold a reference to the state object, which incurs unnecessary overhead and complexity.

// The actual path of the executable. Needed to redirect /proc/self/*
extern std::string exec_path;

// Path of sysroot. When the guest application tries to open a file, and the corresponding file exists in sysroot,
// it will be redirected.
extern std::string sysroot;

// The program/data break of the address space. original_brk represents the initial brk from information gathered
// in elf. Both values are set initially to original_brk by elf_loader, and original_brk should not be be changed.
// A constraint original_brk <= brk must be satisified.
extern reg_t original_brk;
extern reg_t brk;
extern reg_t heap_start;
extern reg_t heap_end;

// A flag to determine whether to print instruction out when it is decoded.
extern bool disassemble;

// A flag to determine whether instret should be updated precisely in binary translated code.
extern bool no_instret;

// Upper limit of number of blocks that can be inlined by IR DBT.
extern int inline_limit;

// Threshold beyond which the IR DBT will start working
extern int compile_threshold;

// A flag to determine whether to trace all system calls. If true then all guest system calls will be logged.
extern bool strace;

// A flag to determine whether correctness in case of segmentation fault should be dealt strictly.
extern bool strict_exception;

// A flag to determine whether PHI nodes should be introduced to the graph by load elimination.
extern bool enable_phi;

// Whether compilation performance counters should be enabled.
extern bool monitor_performance;

// Whether direct memory access or call to helper should be generated for guest memory access.
extern bool no_direct_memory_access;

}

// This is not really an error. However it shares some properties with an exception, as it needs to break out from
// any nested controls and stop executing guest code.
struct Exit_control: std::runtime_error {
    uint8_t exit_code;
    Exit_control(uint8_t exit_code): std::runtime_error { "exit" }, exit_code {exit_code} {}
};

// Load elf, and setup auxillary vectors.
reg_t load_elf(const char *filename, reg_t& sp);

// Load a binary to specified location
void load_bin(const char *filename, reg_t location);

}

#endif
