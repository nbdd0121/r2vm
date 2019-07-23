.intel_syntax noprefix

# All helper functions are called with RSP misaligned to 16-byte boundary
# So when return address is pushed, they are properly aligned
.global helper_region_start
helper_region_start:

# This is a duplicate of fiber_yield_raw. It exists here so it can be copied
# to be within +-4G of the DBT-ed code.
.global helper_yield
helper_yield:
    mov [rbp - 32], rsp
    mov rbp, [rbp - 16]
    mov rsp, [rbp - 32]
    ret

# RSI -> vaddr
.global helper_misalign
helper_misalign:
    add edx, 4
    # Set scause and stval
    mov [rbp + 32 * 8 + 32], rdx
    mov [rbp + 32 * 8 + 40], rsi
    # Fall-through to helper_trap

.global helper_trap
.extern trap
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

.extern riscv_step
.global helper_step
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

.global helper_region_end
helper_region_end:
    nop
