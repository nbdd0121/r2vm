.intel_syntax noprefix

# All helper functions are called with RSP misaligned to 16-byte boundary
# So when return address is pushed, they are properly aligned

.global helper_trap
.extern trap
helper_trap:
    # RDI -> context
    mov rdi, rbp
    # We use EBX to store the instruction offset within the current basic block
    # Lower 16 bits are PC offset and upper 16 bits are INSTRET offset.
    movsx rax, bx
    add [rbp + 0x100], rax
    shr ebx, 16
    movsx rax, bx
    add [rbp + 0x108], rax

    call trap
    ret

# RSI -> vaddr
.global helper_misalign
helper_misalign:
# RDI -> context
    mov rdi, rbp
    # We use EBX to store the instruction offset within the current basic block
    # Lower 16 bits are PC offset and upper 16 bits are INSTRET offset.
    movsx rax, bx
    add [rbp + 0x100], rax
    shr ebx, 16
    movsx rax, bx
    add [rbp + 0x108], rax

    call handle_misalign
    test al, al
    jnz 1f
    ret
1:
    mov rdi, rbp
    call trap
    ret

.global helper_icache_cross_miss
.extern icache_cross_miss
helper_icache_cross_miss:
    mov rdi, rbp
    # The instruction to translate
    movzx eax, word ptr [rsi]
    shl eax, 16
    or ecx, eax
    # Load the return address
    mov rdx, [rsp]
    movsx rax, dword ptr [rdx + 1]
    lea rdx, [rdx+rax+5]
    jmp icache_cross_miss

.global helper_patch_direct_jump
.extern find_block
helper_patch_direct_jump:
    pop rbx
    mov rdi, rbp
    call find_block
    # If rax is zero, it means we have a rollover, so don't patch
    test rax, rax
    jz 1f
    # RBX contains the RIP past the call instruction
    # So we RDX - RBX will be the offset needed to patch the offset.
    # Note we use RDX as it is the nonspeculative entry point
    mov rcx, rdx
    sub rcx, rbx
    mov dword ptr [rbx - 5], 0xE9
    mov [rbx - 4], ecx
1:
    jmp rdx

.global helper_check_interrupt
.extern check_interrupt
helper_check_interrupt:
    mov rdi, rbp
    call check_interrupt
    test al, al
    jnz 1f
    ret
1:
    # If check_interrupt returns Err(()), it indicates that we would like to shutdown.
    # We therefore need to return from fiber_interp_run.
    add rsp, 8
    ret

.global helper_san_fail
helper_san_fail:
    ud2

.global helper_pred_miss
helper_pred_miss:
    mov rdi, rbp
    call find_block
    # If rax is zero, it means we have a rollover, so don't patch
    test rax, rax
    jz 1f
    # RBX contains the RIP past the call instruction
    # So we RAX - RBX will be the offset needed to patch the offset.
    sub rax, rbx
    mov [rbx - 4], eax
1:
    jmp rdx

.global fiber_interp_run
fiber_interp_run:
    call fiber_save_raw
    call 1f
    jmp fiber_restore_ret_raw

1:
    mov rdi, rbp
    sub rsp, 8
    call find_block
    add rsp, 8
    call rdx
    jmp 1b
