.intel_syntax noprefix

.global fiber_yield_raw
.global fiber_sleep_raw
.global fiber_sleep

.global fiber_start

OFFSET_DATA = -64
OFFSET_NEXT = 16 + OFFSET_DATA
OFFSET_CYCLES_TO_SLEEP = 40 + OFFSET_DATA
OFFSET_NEXT_AVAIL = 48 + OFFSET_DATA
OFFSET_STACK_POINTER = 56 + OFFSET_DATA

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
    add rbp, -OFFSET_DATA
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

# Yield to next fiber for "rdi + 1" times.
# Before calling this, make sure:
# - rbp already points to the desired FiberStack.
# - all registers other than rsp and rbp are saved properly, as they will be clobbered upon return
fiber_sleep_raw:
    # Save number of cycles to sleep
    mov [rbp + OFFSET_CYCLES_TO_SLEEP], rdi
fiber_yield_raw:
    # Save current stack pointer
    mov [rbp + OFFSET_STACK_POINTER], rsp
1:
    # Move to next fiber
    mov rbp, [rbp + OFFSET_NEXT_AVAIL]
    # If this field is non-zero, it means that we have more cycles to sleep.
    cmp qword ptr [rbp + OFFSET_CYCLES_TO_SLEEP], 0
    jnz 2f
    # Restore stack pointer
    mov rsp, [rbp + OFFSET_STACK_POINTER]
    ret
2:
    sub qword ptr [rbp + OFFSET_CYCLES_TO_SLEEP], 1
    jmp 1b

# Yield the fiber "rdi + 1" many times.
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
    add rbp, -OFFSET_DATA

    call fiber_sleep_raw

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

fiber_start:
    # Get the return address, and save all volatile registers (will be destroyed when entering
    # fiber).
    call fiber_save_raw

    # To ensure that we are capable of returning from any exiting fiber, we need to fill
    # top of each fiber stack with this format:
    # +-----------+
    # | Pad       | // This is necessary as AMD64 requires 16-byte stack alignment
    # +-----------+
    # | Saved RSP |
    # +-----------+
    # | Exit RIP  |
    # + ----------+
    mov rbp, rdi
    movabs rax, offset fiber_exit
1:
    mov [rbp + OFFSET_DATA + 0x200000 - 16], rsp
    mov [rbp + OFFSET_DATA + 0x200000 - 24], rax
    mov rdx, rbp
    mov rbp, [rbp + OFFSET_NEXT]
    cmp rbp, rdi
    jne 1b

    mov rbp, [rdx + OFFSET_NEXT_AVAIL]
    mov rsp, [rbp + OFFSET_STACK_POINTER]
    ret

fiber_exit:
    # Retrieve the original RSP and restore
    mov rsp, [rsp]
    mov rax, rbp
    and rax, -0x200000
    add rax, -OFFSET_DATA

    jmp fiber_restore_ret_raw

.global fiber_current
fiber_current:
    mov rax, rsp
    and rax, -0x200000
    add rax, -OFFSET_DATA
    ret
