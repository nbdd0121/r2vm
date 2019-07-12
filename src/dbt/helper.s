.intel_syntax noprefix

.global helper_trap
.extern handle_trap

# All helper functions are called with RSP misaligned to 16-byte boundary
# So when return address is pushed, they are properly aligned

helper_trap:
    # RDI -> context
    mov rdi, rbp
    # RDI -> trapping PC. Do not pop here as it will cause misalignment
    mov rsi, [rsp]
    call handle_trap
    # Pop out trapping PC
    add rsp, 8
    ret

.global helper_step
.extern riscv_step

helper_step:
    mov rdi, rbp
    call riscv_step
    test al, al
    jnz 1f
    ret
1:
    jmp helper_trap

.global helper_step_tail
helper_step_tail:
    mov rdi, rbp
    call riscv_step
    test al, al
    jnz 1f
    add rsp, 8
    ret
1:
    jmp helper_trap

.global helper_translate_cache_miss
.extern translate_cache_miss
helper_translate_cache_miss:
    mov rdi, rbp
    call translate_cache_miss
    test al, al
    jnz 1f
    mov rsi, rdx
    ret
1:
    jmp helper_trap

.global helper_icache_miss
.extern insn_translate_cache_miss
helper_icache_miss:
    mov rdi, rbp
    call insn_translate_cache_miss
    test al, al
    jnz 1f
    ret
1:
    # we're not yet prepared for this
    ud2
