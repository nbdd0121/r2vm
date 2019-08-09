.intel_syntax noprefix

.global fiber_yield_raw
.global fiber_sleep

.global fiber_start

# Save all non-volatile registers.
# will destroy RAX and set RBP to the fiber base pointer
.global fiber_save_raw
fiber_save_raw:
    pop rax
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
    jmp rax

# Restore all non-volatile registers and return
.global fiber_restore_ret_raw
fiber_restore_ret_raw:
    fldcw [rsp + 4]
    ldmxcsr [rsp]
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

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
    # Get the return address, and save all volatile registers (will be destroyed when entering
    # fiber).
    call fiber_save_raw

    # To ensure that we are capable of returning from any exiting fiber, we need to fill
    # top of each fiber stack with this format:
    # +-----------+
    # | 0         | // This is necessary as AMD64 requires 16-byte stack alignment
    # +-----------+
    # | Saved RSP |
    # +-----------+
    # | Exit RIP  |
    # + ----------+
    mov rbp, rdi
    movabs rax, offset fiber_exit
1:
    mov [rbp - 32 + 0x200000 - 16], rsp
    mov [rbp - 32 + 0x200000 - 24], rax
    mov rbp, [rbp - 16]
    cmp rbp, rdi
    jne 1b

    mov rsp, [rbp - 32]
    ret

fiber_exit:
    # Retrieve the original RSP and restore
    mov rsp, [rsp]
    mov rax, rbp
    and rax, -0x200000
    add rax, 32

    jmp fiber_restore_ret_raw

.global fiber_current
fiber_current:
    mov rax, rsp
    and rax, -0x200000
    add rax, 32
    ret
