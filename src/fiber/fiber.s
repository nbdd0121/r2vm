.intel_syntax noprefix

.global fiber_yield_raw
.global fiber_sleep

.global fiber_start

# Yield to next fiber
# Before calling this, make sure:
# - rbp already points to the desired FiberStack.
# - all registers other than rsp and rbp are saved properly, as they will be clobbered upon return
fiber_yield_raw:
    # Save current stack pointer
    mov [rbp - 32], rsp
    # Move to next fiber
    mov rbp, [rbp - 16]
    # Restore stack pointer
    mov rsp, [rbp - 32]
    ret

# Yield the fiber "rdi" many times.
fiber_sleep:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    stmxcsr [rsp]
    fnstcw  [rsp + 4]

    # Retrieve the fiber base pointer
    mov rbp, rsp
    and rbp, -0x200000
    add rbp, 32

    push rdi
.L1:
    call fiber_yield_raw
    sub qword ptr [rsp], 1
    jne .L1

    fldcw [rsp + 12]
    ldmxcsr [rsp + 8]
    add rsp, 16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

fiber_start:
    lea rbp, [rdi + 32]
    mov rsp, [rdi]
    ret

.extern find_block
.extern trap
.global fiber_interp_run
fiber_interp_run:
    mov rdi, rbp
    call find_block
    call rax
    # Read the pending register
    cmp qword ptr [rbp + 32 * 8 + 16], 0
    jnz 1f
2:
    call fiber_yield_raw
    jmp fiber_interp_run
1:
    mov rdi, rbp
    call trap
    jmp 2b

.extern interp_block
.global fiber_interp_block
fiber_interp_block:
    mov rdi, rbp
    jmp interp_block
