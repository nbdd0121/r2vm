IO Memory
=========

Instead of checking the address for each access and determine whether a memory location is I/O
memory, a different approach is used. Under this approach, we use `mmap` to allocate the I/O
memory region with protection `PROT_NONE`, and thus all memory access to the region will trap.

The interpreter does not treat I/O specially. When it accesses the I/O region as normal memory,
a page fault will trigger and the OS will send a signal. The signal handler will be able to catch
this trap, decode the faulting x86 instruction, and simulate the memory access.

As I/O access is uncommon, it reduces the length of hot path, and does not require special cases
when dealing with memory accesses. This also makes the handling more similar to that of hypervisor.
