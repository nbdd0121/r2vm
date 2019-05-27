.intel_syntax noprefix

.global memory_probe_start
.global memory_probe_read
.global memory_probe_write
.global memory_probe_end

memory_probe_start:
memory_probe_read:
    mov eax, [rdi]
    xor eax, eax
    ret

memory_probe_write:
    lock add dword ptr [rdi], 0
    xor eax, eax
    ret
memory_probe_end:
