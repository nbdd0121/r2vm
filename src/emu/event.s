.intel_syntax noprefix

.global event_loop_wait
.extern fiber_save_raw
.extern fiber_yield_raw
.extern fiber_restore_ret_raw
event_loop_wait:
    call fiber_save_raw
1:
    call fiber_yield_raw
    # Increase the global cycle counter. No need for atomics, as we are the only writer.
    mov rax, [rbp]
    add rax, 1
    mov [rbp], rax
    # If the cycle counter exceeds the `next_event` value, we can return
    cmp rax, [rbp + 8]
    jae fiber_restore_ret_raw
    jmp 1b
