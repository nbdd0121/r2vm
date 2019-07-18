.intel_syntax noprefix

.global helper_trap
.extern handle_trap
.extern trap

# All helper functions are called with RSP misaligned to 16-byte boundary
# So when return address is pushed, they are properly aligned

# RSI -> vaddr
.global helper_misalign
helper_misalign:
    add edx, 4
    # Set scause and stval
    mov [rbp + 32 * 8 + 32], rdx
    mov [rbp + 32 * 8 + 40], rsi
    # Fall-through to helper_trap

helper_trap:
    # RDI -> context
    mov rdi, rbp
    # RDI -> trapping PC. Do not pop here as it will cause misalignment
    mov rsi, [rsp]
    call handle_trap
    # For synchronous traps, take them now instead of falling back to the main loop.
    mov rdi, rbp
    call trap
    # Pop out trapping PC
    add rsp, 8
    ret

.global helper_step
.extern riscv_step

helper_step:
    mov rdi, rbp
    call riscv_step
    test al, al
    jnz helper_trap
    ret

.global helper_translate_cache_miss
.extern translate_cache_miss
helper_translate_cache_miss:
    mov rdi, rbp
    call translate_cache_miss
    test al, al
    jnz helper_trap
    mov rsi, rdx
    ret

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
