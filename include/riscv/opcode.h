#ifndef RISCV_OPCODE_H
#define RISCV_OPCODE_H

namespace riscv {

// This includes all supported RISC-V opcodes.
// Opcodes should be sorted in the following order
// * Canonical order of extension
// * Increasing base opcode number
// * Increasing funct3 and then funct7, or their ordering in RISC-V spec
enum class Opcode {
    illegal,

    /* F extension */
    /* Base Opcode = LOAD-FP */
    /* Base Opcode = STORE-FP */
    /* Base Opcode = OP-FP */
    fadd_s,
    fsub_s,
    fmul_s,
    fdiv_s,
    fsqrt_s,
    fsgnj_s,
    fsgnjn_s,
    fsgnjx_s,
    fmin_s,
    fmax_s,
    fcvt_w_s,
    fcvt_wu_s,
    fcvt_l_s,
    fcvt_lu_s,
    fmv_x_w,
    feq_s,
    flt_s,
    fle_s,
    fclass_s,
    fcvt_s_w,
    fcvt_s_wu,
    fcvt_s_l,
    fcvt_s_lu,
    fmv_w_x,
    /* Base Opcode = MADD */
    fmadd_s,
    /* Base Opcode = MSUB */
    fmsub_s,
    /* Base Opcode = NMSUB */
    fnmsub_s,
    /* Base Opcode = NMADD */
    fnmadd_s,

    /* D extension */
    /* Base Opcode = LOAD-FP */
    /* Base Opcode = STORE-FP */
    /* Base Opcode = OP-FP */
    fadd_d,
    fsub_d,
    fmul_d,
    fdiv_d,
    fsqrt_d,
    fsgnj_d,
    fsgnjn_d,
    fsgnjx_d,
    fmin_d,
    fmax_d,
    fcvt_s_d,
    fcvt_d_s,
    feq_d,
    flt_d,
    fle_d,
    fclass_d,
    fcvt_w_d,
    fcvt_wu_d,
    fcvt_l_d,
    fcvt_lu_d,
    fmv_x_d,
    fcvt_d_w,
    fcvt_d_wu,
    fcvt_d_l,
    fcvt_d_lu,
    fmv_d_x,
    /* Base Opcode = MADD */
    fmadd_d,
    /* Base Opcode = MSUB */
    fmsub_d,
    /* Base Opcode = NMSUB */
    fnmsub_d,
    /* Base Opcode = NMADD */
    fnmadd_d,
};

} // riscv

#endif
