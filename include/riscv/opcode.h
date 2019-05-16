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
    /* RV64I */
    /* Base Opcode = LOAD */
    lb,
    lh,
    lw,
    ld,
    lbu,
    lhu,
    lwu,
    /* Base Opcode = LOAD-FP */
    /* Base Opcode = MISC-MEM */
    fence,
    fence_i,
    /* Base Opcode = OP-IMM */
    addi,
    slli,
    slti,
    sltiu,
    xori,
    srli,
    srai,
    ori,
    andi,
    /* Base Opcode = AUIPC */
    auipc,
    /* Base Opcode = OP-IMM-32 */
    addiw,
    slliw,
    srliw,
    sraiw,
    /* Base Opcode = STORE */
    sb,
    sh,
    sw,
    sd,
    /* Base Opcode = STORE-FP */
    /* Base Opcode = AMO */
    /* Base Opcode = OP */
    add,
    sub,
    sll,
    slt,
    sltu,
    // xor, or, and are C++ keywords. We add prefix i_ to disambiguate.
    i_xor,
    srl,
    sra,
    i_or,
    i_and,
    /* Base Opcode = LUI */
    lui,
    /* Base Opcode = OP-32 */
    addw,
    subw,
    sllw,
    srlw,
    sraw,
    /* Base Opcode = MADD */
    /* Base Opcode = MSUB */
    /* Base Opcode = NMSUB */
    /* Base Opcode = NMADD */
    /* Base Opcode = OP-FP */
    /* Base Opcode = BRANCH */
    beq,
    bne,
    blt,
    bge,
    bltu,
    bgeu,
    /* Base Opcode = JALR */
    jalr,
    /* Base Opcode = JAL */
    jal,
    /* Base Opcode = SYSTEM */
    ecall,
    ebreak,
    csrrw,
    csrrs,
    csrrc,
    csrrwi,
    csrrsi,
    csrrci,

    /* M extension */
    /* Base Opcode = OP */
    mul,
    mulh,
    mulhsu,
    mulhu,
    div,
    divu,
    rem,
    remu,
    /* Base Opcode = OP-32 */
    mulw,
    divw,
    divuw,
    remw,
    remuw,

    /* A extension */
    /* Base Opcode = AMO */
    lr_w,
    lr_d,
    sc_w,
    sc_d,
    amoswap_w,
    amoswap_d,
    amoadd_w,
    amoadd_d,
    amoxor_w,
    amoxor_d,
    amoand_w,
    amoand_d,
    amoor_w,
    amoor_d,
    amomin_w,
    amomin_d,
    amomax_w,
    amomax_d,
    amominu_w,
    amominu_d,
    amomaxu_w,
    amomaxu_d,

    /* F extension */
    /* Base Opcode = LOAD-FP */
    flw,
    /* Base Opcode = STORE-FP */
    fsw,
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
    fld,
    /* Base Opcode = STORE-FP */
    fsd,
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
