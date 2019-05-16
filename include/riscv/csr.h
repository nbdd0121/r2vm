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
};

} // riscv

#endif
