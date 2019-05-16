#include <sys/ucontext.h>
#include <csignal>
#include <cstring>
#include <limits>

#include "main/signal.h"
#include "util/memory.h"
#include "x86/decoder.h"
#include "x86/opcode.h"

namespace {

void handle_fault(int sig) {
    ASSERT(sig == SIGSEGV || sig == SIGBUS);

    sigset_t x;
    sigemptyset(&x);
    sigaddset(&x, sig);
    sigprocmask(SIG_UNBLOCK, &x, nullptr);
    throw Segv_exception {sig};
}

void handle_fpe(int sig, siginfo_t*, void* context) {
    ASSERT(sig == SIGFPE);

    auto ucontext = reinterpret_cast<ucontext_t*>(context);

    constexpr int reg_list[] = {
        REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
        REG_R8, REG_R9, REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15
    };

    // Decode the current instruction.
    uint64_t current_ip = ucontext->uc_mcontext.gregs[REG_RIP];
    x86::Decoder decoder {current_ip};
    x86::Instruction inst = decoder.decode_instruction();
    uint64_t next_ip = decoder.pc();

    ASSERT(inst.opcode == x86::Opcode::div || inst.opcode == x86::Opcode::idiv);
    uint64_t divisor;
    int opsize;

    // Retrieve the value of the divisor.
    if (inst.operands[0].is_register()) {
        x86::Register reg = inst.operands[0].as_register();
        opsize = static_cast<int>(reg) & x86::reg_gpq ? 8 : 4;
        divisor = ucontext->uc_mcontext.gregs[reg_list[static_cast<int>(reg) & 15]];
        if (opsize == 4) divisor = static_cast<uint32_t>(divisor);

    } else {
        const auto& memory = inst.operands[0].as_memory();

        // Sign-extend displacement to 64-bit.
        uint64_t address = static_cast<int64_t>(static_cast<int32_t>(memory.displacement));
        if (memory.base != x86::Register::none) {
            address += ucontext->uc_mcontext.gregs[reg_list[static_cast<int>(memory.base) & 15]];
        }

        if (memory.index != x86::Register::none) {
            address += ucontext->uc_mcontext.gregs[reg_list[static_cast<int>(memory.index) & 15]] * memory.scale;
        }

        opsize = memory.size;
        divisor = memory.size == 8 ?
                util::read_as<uint64_t>(reinterpret_cast<void*>(address)) :
                util::read_as<uint32_t>(reinterpret_cast<void*>(address));
    }

    // Retrive dividend. Note that technically RDX is also dividend, but we assume it is also sign/zero-extended.
    uint64_t dividend = ucontext->uc_mcontext.gregs[REG_RAX];
    if (opsize == 4) dividend = static_cast<uint32_t>(dividend);

    ASSERT(opsize == 4 || opsize == 8);

    if (divisor == 0) {
        // For divide by zero, per RISC-V we set quotient to -1 and remainder to dividend.
        ucontext->uc_mcontext.gregs[REG_RAX] = opsize == 8 ? static_cast<uint64_t>(-1) : static_cast<uint32_t>(-1);
        ucontext->uc_mcontext.gregs[REG_RDX] = dividend;

    } else {
        // Integer division overflow. Per RISC-V we set quotient to dividend and remainder to 0.
        ASSERT(inst.opcode == x86::Opcode::idiv);
        if (opsize == 8) {
            ASSERT(static_cast<int64_t>(dividend) == std::numeric_limits<int64_t>::min() &&
                   static_cast<int64_t>(divisor) == -1);

        } else {
            ASSERT(static_cast<int32_t>(dividend) == std::numeric_limits<int32_t>::min() &&
                   static_cast<int32_t>(divisor) == -1);
        }

        ucontext->uc_mcontext.gregs[REG_RAX] = dividend;
        ucontext->uc_mcontext.gregs[REG_RDX] = 0;
    }

    // Advance to next ip.
    ucontext->uc_mcontext.gregs[REG_RIP] = next_ip;
}

}

void setup_fault_handler() {
    struct sigaction act;

    memset (&act, 0, sizeof(act));
    act.sa_handler = handle_fault;
    sigaction(SIGSEGV, &act, NULL);
    sigaction(SIGBUS, &act, NULL);

    memset (&act, 0, sizeof(act));
    act.sa_sigaction = handle_fpe;
    act.sa_flags = SA_SIGINFO;
    sigaction(SIGFPE, &act, NULL);
}
