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
.global helper_read_misalign
helper_read_misalign:
    # Set scause and stval
    mov qword ptr [rbp + 32 * 8 + 16], 4
    mov [rbp + 32 * 8 + 24], rsi
    jmp helper_trap

# RSI -> vaddr
.global helper_write_misalign
helper_write_misalign:
    # Set scause and stval
    mov qword ptr [rbp + 32 * 8 + 16], 5
    mov [rbp + 32 * 8 + 24], rsi
    jmp helper_trap

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

.global helper_icache_wrong
.extern find_block_and_patch
helper_icache_wrong:
    mov rdi, rbp
    # Load return address into RSI
    mov rsi, [rsp]
    jmp find_block_and_patch


.global helper_icache_patch2
.extern find_block_and_patch2
helper_icache_patch2:
    mov rdi, rbp
    # Load return address into RSI
    mov rsi, [rsp]
    sub qword ptr [rsp], 5
    jmp find_block_and_patch2

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

.extern find_block
.global fiber_interp_run
fiber_interp_run:
    mov rdi, rbp
    sub rsp, 8
    call find_block
    add rsp, 8
    call rax
    jmp fiber_interp_run
