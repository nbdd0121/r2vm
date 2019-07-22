.intel_syntax noprefix

.global helper_trap
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
    # We use EBX to store the instruction offset within the current basic block
    # Lower 16 bits are PC offset and upper 16 bits are INSTRET offset.
    movzx eax, bx
    sub [rbp + 0x100], rax
    shr ebx, 16
    sub [rbp + 0x108], rbx
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
    mov rsi, rdx
    ret
1:
    # we're not yet prepared for this
    # ud2

    # this makes sense in the begin block only!!!!
    mov rdi, rbp
    call trap
    add rsp, 8
    ret

.global helper_icache_wrong
.extern find_block_and_patch
helper_icache_wrong:
    mov rdi, rbp
    # Load return address into RSI
    mov rsi, [rsp]
    call find_block_and_patch
    ret

.global helper_check_interrupt
.extern check_interrupt
helper_check_interrupt:
    mov rdi, rbp
    jmp check_interrupt
