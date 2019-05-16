#ifndef X86_OPCODE_H
#define X86_OPCODE_H

#include <cstdint>

namespace x86 {

enum class Condition_code: uint8_t {
    overflow = 0x0,
    not_overflow = 0x1,
    below = 0x2, carry = 0x2, not_above_equal = 0x2,
    above_equal = 0x3, not_below = 0x3, not_carry = 0x3,
    equal = 0x4, zero = 0x4,
    not_equal = 0x5, not_zero = 0x5,
    below_equal = 0x6, not_above = 0x6,
    above = 0x7, not_below_equal = 0x7,
    sign = 0x8,
    not_sign = 0x9,
    parity = 0xA, parity_even = 0xA,
    not_parity = 0xB, parity_odd = 0xB,
    less = 0xC, not_greater_equal = 0xC,
    greater_equal = 0xD, not_less = 0xD,
    less_equal = 0xE, not_greater = 0xE,
    greater = 0xF, not_less_equal = 0xF,
};

enum class Opcode: uint16_t {
    illegal,
    add,
    i_and,
    cdqe,
    call,
    cmovcc,
    cmp,
    cdq, cqo,
    div,
    idiv,
    imul,
    jcc,
    jmp,
    lea,
    mov,
    movabs,
    movsx,
    movzx,
    mul,
    neg,
    nop,
    i_not,
    i_or,
    push,
    pop,
    ret,
    sar,
    sbb,
    setcc,
    shl,
    shr,
    sub,
    test,
    xchg,
    i_xor,

    // aliases
    movsxd = movsx,
};

} // x86

#endif
