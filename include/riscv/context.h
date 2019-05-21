#ifndef RISCV_STATE_H
#define RISCV_STATE_H

#include "main/executor.h"
#include "riscv/typedef.h"

namespace riscv {

struct cache_line {
    uint64_t tag;
    // Least significant bit indicates writable
    uint64_t paddr;
};

// This class represent the state of a single hart (i.e. hardware thread).
struct Context {
    // registers[0] is reserved and maybe used for other purpose later
    reg_t registers[32];
    freg_t fp_registers[32];
    reg_t pc;
    reg_t instret;

    reg_t fcsr;

    // For load-reserved
    reg_t lr;

    // S-mode CSRs
    reg_t sstatus;
    reg_t sie;
    reg_t stvec;
    reg_t sscratch;
    reg_t sepc;
    reg_t scause;
    reg_t stval;
    reg_t sip;
    reg_t satp;

    reg_t timecmp;

    // Current privilege level
    int prv;

    // Pending exceptions: sstatus.sie ? sie & sip : 0
    reg_t pending;

    cache_line line[1024];

    // The execution engine that is currently operating on this context.
    Executor *executor;
};

class Instruction;
reg_t translate(Context *context, reg_t pc, bool write);
void step(Context *context, Instruction inst);
reg_t sbi_call(Context *context,
    reg_t nr,
    reg_t arg0, reg_t arg1, reg_t arg2, [[maybe_unused]] reg_t arg3, [[maybe_unused]] reg_t arg4, [[maybe_unused]] reg_t arg5
);

} // riscv

#endif
