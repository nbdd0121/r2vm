.intel_syntax noprefix

.extern fiber_yield_raw
.extern event_loop_handle
.global fiber_event_loop
fiber_event_loop:
    mov rdi, rbp
    call event_loop_handle
3:
    # Increase the global cycle counter
    mov rax, [rbp]
    add rax, 1
    mov [rbp], rax
    # If the cycle counter exceeds the `next_event` value, we need to do things.
    cmp rax, [rbp + 8]
    jae 1f
2:
    call fiber_yield_raw
    jmp 3b
1:
    mov rdi, rbp
    call event_loop_handle
    jmp 2b
