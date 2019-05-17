#ifndef RISCV_CSR_H
#define RISCV_CSR_H

namespace riscv {

enum class Csr {
    fflags = 0x001,
    frm = 0x002,
    fcsr = 0x003,

    cycle = 0xC00,
    time = 0xC01,
    instret = 0xC02,
    
    /* Rv32I only */
    cycleh = 0xC80,
    timeh = 0xC81,
    instreth = 0xC82,

    sstatus = 0x100,
    sie = 0x104,
    stvec = 0x105,
    scounteren = 0x106,

    sscratch = 0x140,
    sepc = 0x141,
    scause = 0x142,
    stval = 0x143,
    sip = 0x144,

    satp = 0x180,
};

} // riscv

#endif
