#include "emu/state.h"
#include "emu/mmu.h"
#include "emu/unwind.h"
#include "main/dbt.h"
#include "main/signal.h"
#include "riscv/basic_block.h"
#include "riscv/context.h"
#include "riscv/decoder.h"
#include "riscv/instruction.h"
#include "riscv/opcode.h"
#include "util/assert.h"
#include "util/code_buffer.h"
#include "util/format.h"
#include "util/functional.h"
#include "util/memory.h"
#include "x86/builder.h"
#include "x86/disassembler.h"
#include "x86/encoder.h"
#include "x86/instruction.h"
#include "x86/opcode.h"

// Shorthand for instruction coding.
using namespace x86::builder;

// Declare the exception handling registration functions.
extern "C" void __register_frame(void*);
extern "C" void __deregister_frame(void*);

// Denotes a translated block.
struct Dbt_block {

    // Translated code.
    util::Code_buffer code;

    // Decoded instructions.
    riscv::Basic_block block;

    // Specify the mapping between RISC-V instruction and x86 instruction.
    std::vector<uint8_t> pc_map;

    // Exception handling frame
    std::unique_ptr<uint8_t[]> cie;

    ~Dbt_block() {
        if (cie) {
            __deregister_frame(cie.get());
        }
    }
};

// A separate class is used instead of generating code directly in Dbt_runtime, so it is easier to define and use
// helper functions that are shared by many instructions.
class Dbt_compiler {
private:
    Dbt_runtime& runtime_;
    Dbt_block& block_;
    x86::Encoder encoder_;

    Dbt_compiler& operator <<(const x86::Instruction& inst);

    /* Helper functions */
    void emit_move(int rd, int rs);
    void emit_move32(int rd, int rs);
    void emit_load_immediate(int rd, riscv::reg_t imm);
    void emit_branch(riscv::Instruction inst, riscv::reg_t pc_diff, x86::Condition_code cc);

    /* Translated instructions */
    void emit_jalr(riscv::Instruction inst, riscv::reg_t pc_diff);
    void emit_jal(riscv::Instruction inst, riscv::reg_t pc_diff);

    void emit_lb(riscv::Instruction inst, bool u);
    void emit_lh(riscv::Instruction inst, bool u);
    void emit_lw(riscv::Instruction inst, bool u);
    void emit_ld(riscv::Instruction inst);
    void emit_sb(riscv::Instruction inst);
    void emit_sh(riscv::Instruction inst);
    void emit_sw(riscv::Instruction inst);
    void emit_sd(riscv::Instruction inst);

    void emit_addi(riscv::Instruction inst);
    void emit_shifti(riscv::Instruction inst, x86::Opcode opcode);
    void emit_slti(riscv::Instruction inst);
    void emit_sltiu(riscv::Instruction inst);
    void emit_xori(riscv::Instruction inst);
    void emit_ori(riscv::Instruction inst);
    void emit_andi(riscv::Instruction inst);

    void emit_addiw(riscv::Instruction inst);
    void emit_shiftiw(riscv::Instruction inst, x86::Opcode opcode);

    void emit_add(riscv::Instruction inst);
    void emit_sub(riscv::Instruction inst);
    void emit_shift(riscv::Instruction inst, x86::Opcode opcode);
    void emit_slt(riscv::Instruction inst);
    void emit_sltu(riscv::Instruction inst);
    void emit_xor(riscv::Instruction inst);
    void emit_or(riscv::Instruction inst);
    void emit_and(riscv::Instruction inst);

    void emit_addw(riscv::Instruction inst);
    void emit_subw(riscv::Instruction inst);
    void emit_shiftw(riscv::Instruction inst, x86::Opcode opcode);

    void emit_mul(riscv::Instruction inst);
    void emit_mulh(riscv::Instruction inst, bool u);
    void emit_mulhsu(riscv::Instruction inst);
    void emit_mulw(riscv::Instruction inst);
    void emit_div(riscv::Instruction inst, bool u, bool rem);
    void emit_divw(riscv::Instruction inst, bool u, bool rem);

public:
    Dbt_compiler(Dbt_runtime& runtime, Dbt_block& block): runtime_{runtime}, block_{block}, encoder_{block.code} {}
    void compile(emu::reg_t pc);
    void generate_eh_frame();
};

_Unwind_Reason_Code dbt_personality(
    [[maybe_unused]] int version,
    _Unwind_Action actions,
    [[maybe_unused]] uint64_t exception_class,
    [[maybe_unused]] struct _Unwind_Exception *exception_object,
    [[maybe_unused]] struct _Unwind_Context *context
) {
    if (actions & _UA_SEARCH_PHASE) {

        // We don't catch anything, just continue the unwind.
        return _URC_CONTINUE_UNWIND;

    } else {

        // Cleanup phase.

        // First retrieve the associated Dbt_block by reading from LSDA.
        Dbt_block& block = *reinterpret_cast<Dbt_block*>(_Unwind_GetLanguageSpecificData(context));

        // Retrive the runtime context by reading register RBP, which has id 5.
        riscv::Context* ctx = reinterpret_cast<riscv::Context*>(_Unwind_GetGR(context, 5));

        // Calculate the index and offset of the trapping instruction.
        uint64_t current_ip = _Unwind_GetIP(context);
        uint64_t host_offset = current_ip - reinterpret_cast<uint64_t>(block.code.data());
        size_t guest_offset = 0, i;
        for (i = 0; i < block.pc_map.size(); i++) {
            if (host_offset < block.pc_map[i]) {
                break;
            }
            host_offset -= block.pc_map[i];
            guest_offset += block.block.instructions[i].length();
        }
        ASSERT(i < block.pc_map.size());

        // Make sure emulated CPU state is consistency.
        ctx->instret += i;
        ctx->pc += guest_offset;
        return _URC_CONTINUE_UNWIND;
    }
}

Dbt_runtime::Dbt_runtime() {
    icache_tag_ = std::unique_ptr<emu::reg_t[]> { new emu::reg_t[4096] };
    icache_ = std::unique_ptr<std::byte*[]> { new std::byte*[4096] };
    for (size_t i = 0; i < 4096; i++) {
        icache_tag_[i] = 0;
    }
}

// Necessary as Dbt_block is incomplete in header.
Dbt_runtime::~Dbt_runtime() {}

void Dbt_runtime::step(riscv::Context& context) {
    const emu::reg_t pc = context.pc;
    const ptrdiff_t tag = (pc >> 1) & 4095;

    // If the cache misses, compile the current block.
    if (UNLIKELY(icache_tag_[tag] != pc)) {
        compile(pc);
    }

    auto func = reinterpret_cast<void(*)(riscv::Context&)>(icache_[tag]);
    ASSERT(func);
    func(context);
    return;
}

void Dbt_runtime::compile(emu::reg_t pc) {
    const ptrdiff_t tag = (pc >> 1) & 4095;
    auto& block_ptr = inst_cache_[pc];

    // Reserve a page in case that the buffer is empty, it saves the code buffer from reallocating (which is expensive
    // as code buffer is backed up by mmap and munmap at the moment.
    // If buffer.size() is not zero, it means that we have compiled the code previously but it is not in the hot cache.
    if (!block_ptr) {
        block_ptr = std::make_unique<Dbt_block>();
        block_ptr->code.reserve(4096);
        Dbt_compiler compiler { *this, *block_ptr };
        compiler.compile(pc);
    }

    // Update tag to reflect newly compiled code.
    icache_[tag] = block_ptr->code.data();
    icache_tag_[tag] = pc;
}

Dbt_compiler& Dbt_compiler::operator <<(const x86::Instruction& inst) {
    bool disassemble = emu::state::disassemble;
    std::byte *pc;
    if (disassemble) {
        pc = encoder_.buffer().data() + encoder_.buffer().size();
    }
    encoder_.encode(inst);
    if (disassemble) {
        std::byte *new_pc = encoder_.buffer().data() + encoder_.buffer().size();
        x86::disassembler::print_instruction(
            reinterpret_cast<uintptr_t>(pc), reinterpret_cast<const char*>(pc), new_pc - pc, inst);
    }
    return *this;
}

#define memory_of_register(reg) (x86::Register::rbp + (offsetof(riscv::Context, registers) + sizeof(emu::reg_t) * reg - 0x80))
#define memory_of(name) (x86::Register::rbp + (offsetof(riscv::Context, name) - 0x80))

void Dbt_compiler::compile(emu::reg_t pc) {
    riscv::Decoder decoder { pc };
    block_.block = decoder.decode_basic_block();
    riscv::Basic_block& block = block_.block;

    if (emu::state::disassemble) {
        util::log("Translating {:x} to {:x}\n", pc, reinterpret_cast<uintptr_t>(encoder_.buffer().data()));
    }

    // Prolog. We place context + 0x80 to rbp instead of context directly, as it allows all registers to be located
    // within int8 offset from rbp, so the assembly representation will uses a shorter encoding.
    *this << push(x86::Register::rbp);
    *this << lea(x86::Register::rbp, qword(x86::Register::rdi + 0x80));

    int pc_diff = 0;
    int instret_diff = 0;

    // We treat the last instruction differently.
    for (size_t i = 0; i < block.instructions.size() - 1; i++) {

        riscv::Instruction inst = block.instructions[i];
        riscv::Opcode opcode = inst.opcode();

        // We treat the prologue as part of the first instruction.
        size_t host_pc_start = i == 0 ? 0 : block_.code.size();

        switch (opcode) {
            case riscv::Opcode::lb: emit_lb(inst, false); break;
            case riscv::Opcode::lh: emit_lh(inst, false); break;
            case riscv::Opcode::lw: emit_lw(inst, false); break;
            case riscv::Opcode::ld: emit_ld(inst); break;
            case riscv::Opcode::lbu: emit_lb(inst, true); break;
            case riscv::Opcode::lhu: emit_lh(inst, true); break;
            case riscv::Opcode::lwu: emit_lw(inst, true); break;
            case riscv::Opcode::sb: emit_sb(inst); break;
            case riscv::Opcode::sh: emit_sh(inst); break;
            case riscv::Opcode::sw: emit_sw(inst); break;
            case riscv::Opcode::sd: emit_sd(inst); break;
            case riscv::Opcode::fence: break;

            case riscv::Opcode::addi: emit_addi(inst); break;
            case riscv::Opcode::slli: emit_shifti(inst, x86::Opcode::shl); break;
            case riscv::Opcode::slti: emit_slti(inst); break;
            case riscv::Opcode::sltiu: emit_sltiu(inst); break;
            case riscv::Opcode::xori: emit_xori(inst); break;
            case riscv::Opcode::srli: emit_shifti(inst, x86::Opcode::shr); break;
            case riscv::Opcode::srai: emit_shifti(inst, x86::Opcode::sar); break;
            case riscv::Opcode::ori: emit_ori(inst); break;
            case riscv::Opcode::andi: emit_andi(inst); break;

            case riscv::Opcode::addiw: emit_addiw(inst); break;
            case riscv::Opcode::slliw: emit_shiftiw(inst, x86::Opcode::shl); break;
            case riscv::Opcode::srliw: emit_shiftiw(inst, x86::Opcode::shr); break;
            case riscv::Opcode::sraiw: emit_shiftiw(inst, x86::Opcode::sar); break;

            case riscv::Opcode::add: emit_add(inst); break;
            case riscv::Opcode::sub: emit_sub(inst); break;
            case riscv::Opcode::sll: emit_shift(inst, x86::Opcode::shl); break;
            case riscv::Opcode::slt: emit_slt(inst); break;
            case riscv::Opcode::sltu: emit_sltu(inst); break;
            case riscv::Opcode::i_xor: emit_xor(inst); break;
            case riscv::Opcode::srl: emit_shift(inst, x86::Opcode::shr); break;
            case riscv::Opcode::sra: emit_shift(inst, x86::Opcode::sar); break;
            case riscv::Opcode::i_or: emit_or(inst); break;
            case riscv::Opcode::i_and: emit_and(inst); break;

            case riscv::Opcode::addw: emit_addw(inst); break;
            case riscv::Opcode::subw: emit_subw(inst); break;
            case riscv::Opcode::sllw: emit_shiftw(inst, x86::Opcode::shl); break;
            case riscv::Opcode::srlw: emit_shiftw(inst, x86::Opcode::shr); break;
            case riscv::Opcode::sraw: emit_shiftw(inst, x86::Opcode::sar); break;

            case riscv::Opcode::mul: emit_mul(inst); break;
            case riscv::Opcode::mulh: emit_mulh(inst, false); break;
            case riscv::Opcode::mulhsu: emit_mulhsu(inst); break;
            case riscv::Opcode::mulhu: emit_mulh(inst, true); break;
            case riscv::Opcode::mulw: emit_mulw(inst); break;
            case riscv::Opcode::div: emit_div(inst, false, false); break;
            case riscv::Opcode::divu: emit_div(inst, true, false); break;
            case riscv::Opcode::rem: emit_div(inst, false, true); break;
            case riscv::Opcode::remu: emit_div(inst, true, true); break;
            case riscv::Opcode::divw: emit_divw(inst, false, false); break;
            case riscv::Opcode::divuw: emit_divw(inst, true, false); break;
            case riscv::Opcode::remw: emit_divw(inst, false, true); break;
            case riscv::Opcode::remuw: emit_divw(inst, true, true); break;

            case riscv::Opcode::lui:
                emit_load_immediate(inst.rd(), inst.imm());
                break;
            case riscv::Opcode::auipc: {
                // AUIPC is special: it needs pc_diff, so do not move it to a separate function.
                const int rd = inst.rd();
                if (rd == 0) break;
                *this << mov(x86::Register::rax, qword(memory_of(pc)));
                *this << add(x86::Register::rax, pc_diff + inst.imm());
                *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
                break;
            }
            default:
                *this << mov(x86::Register::rsi, util::read_as<uint64_t>(&inst));
                *this << lea(x86::Register::rdi, qword(x86::Register::rbp - 0x80));
                *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(riscv::step));
                *this << call(x86::Register::rax);
                break;
        }

        pc_diff += inst.length();
        instret_diff++;

        // Keep track of the translation relationship.
        size_t host_pc_end = block_.code.size();
        block_.pc_map.push_back(host_pc_end - host_pc_start);
    }

    riscv::Instruction inst = block.instructions.back();
    pc_diff += inst.length();
    instret_diff += 1;

    if (!emu::state::no_instret) {
        *this << add(qword(memory_of(instret)), instret_diff);
    }

    switch (inst.opcode()) {
        case riscv::Opcode::jalr: emit_jalr(inst, pc_diff); break;
        case riscv::Opcode::jal: emit_jal(inst, pc_diff); break;
        case riscv::Opcode::beq: emit_branch(inst, pc_diff, x86::Condition_code::equal); break;
        case riscv::Opcode::bne: emit_branch(inst, pc_diff, x86::Condition_code::not_equal); break;
        case riscv::Opcode::blt: emit_branch(inst, pc_diff, x86::Condition_code::less); break;
        case riscv::Opcode::bge: emit_branch(inst, pc_diff, x86::Condition_code::greater_equal); break;
        case riscv::Opcode::bltu: emit_branch(inst, pc_diff, x86::Condition_code::below); break;
        case riscv::Opcode::bgeu: emit_branch(inst, pc_diff, x86::Condition_code::above_equal); break;
        case riscv::Opcode::fence_i: {
            *this << add(qword(memory_of(pc)), pc_diff);
            *this << mov(x86::Register::rdi, reinterpret_cast<uintptr_t>(&runtime_));
            *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(AS_FUNCTION_POINTER(&Dbt_runtime::flush_cache)));
            *this << pop(x86::Register::rbp);
            *this << jmp(x86::Register::rax);
            break;
        }
        default:
            *this << add(qword(memory_of(pc)), pc_diff);
            *this << mov(x86::Register::rsi, util::read_as<uint64_t>(&inst));
            *this << lea(x86::Register::rdi, qword(x86::Register::rbp - 0x80));
            *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(riscv::step));
            *this << pop(x86::Register::rbp);
            *this << jmp(x86::Register::rax);
            break;
    }

    generate_eh_frame();
}

void Dbt_compiler::generate_eh_frame() {
    // TODO: Create an dwarf generation to replace this hard-coded template.
    static const unsigned char cie_template[] = {
        // CIE
        // Length
        0x1C, 0x00, 0x00, 0x00,
        // CIE
        0x00, 0x00, 0x00, 0x00,
        // Version
        0x01,
        // Augmentation string
        'z', 'P', 'L', 0,
        // Instruction alignment factor = 1
        0x01,
        // Data alignment factor = -8
        0x78,
        // Return register number
        0x10,
        // Augmentation data
        0x0A, // Data for z
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // abs format, personality routine
        0x00, // abs format for LSDA
        // Instructions
        // def_cfa(rsp, 8)
        0x0c, 0x07, 0x08,
        // offset(rsp, cfa-8)
        0x90, 0x01,
        // Padding

        // FDE
        // Length
        0x24, 0x00, 0x00, 0x00,
        // CIE Pointer
        0x24, 0x00, 0x00, 0x00,
        // Initial location
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // Augumentation data
        0x8,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // LSDA
        // advance_loc(1)
        0x41,
        // def_cfa_offset(16)
        0x0E, 0x10,
        // offset(rbp, cfa-16)
        0x86, 0x02,
        // Padding
        0x00, 0x00,

        0x00, 0x00, 0x00, 0x00
    };

    block_.cie = std::make_unique<uint8_t[]>(sizeof(cie_template));
    uint8_t *cie = block_.cie.get();

    memcpy(cie, cie_template, sizeof(cie_template));
    util::write_as<uint64_t>(cie + 0x12, reinterpret_cast<uint64_t>(dbt_personality));
    util::write_as<uint64_t>(cie + 0x28, reinterpret_cast<uint64_t>(block_.code.data()));
    util::write_as<uint64_t>(cie + 0x30, block_.code.size());
    util::write_as<uint64_t>(cie + 0x39, reinterpret_cast<uint64_t>(&block_));

    __register_frame(cie);
}

void Dbt_runtime::flush_cache() {
    for (int i = 0; i < 4096; i++)
        icache_tag_[i] = 0;
    inst_cache_.clear();
}

void Dbt_compiler::emit_move(int rd, int rs) {
    if (rd == 0 || rd == rs) return;

    if (rs == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs)));
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_move32(int rd, int rs) {
    if (rd == 0) return;

    if (rs == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    *this << movsx(x86::Register::rax, dword(memory_of_register(rs)));
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_load_immediate(int rd, riscv::reg_t imm) {
    if (rd == 0) return;

    *this << mov(qword(memory_of_register(rd)), imm);
}

void Dbt_compiler::emit_branch(riscv::Instruction inst, riscv::reg_t pc_diff, x86::Condition_code cc) {
    const int rs1 = inst.rs1();
    const int rs2 = inst.rs2();

    if (rs1 == rs2) {
        bool result = cc == x86::Condition_code::equal ||
                      cc == x86::Condition_code::greater_equal ||
                      cc == x86::Condition_code::above_equal;

        if (result) {
            *this << add(qword(memory_of(pc)), pc_diff - inst.length() + inst.imm());
        } else {
            *this << add(qword(memory_of(pc)), pc_diff);
        }

        *this << pop(x86::Register::rbp);
        *this << ret();
        return;
    }

    // Compare and set flags.
    // If either operand is 0, it should be treated specially.
    if (rs2 == 0) {
        *this << cmp(qword(memory_of_register(rs1)), 0);
    } else if (rs1 == 0) {

        // Switch around condition code in this case.
        switch (cc) {
            case x86::Condition_code::less: cc = x86::Condition_code::greater; break;
            case x86::Condition_code::greater_equal: cc = x86::Condition_code::less_equal; break;
            case x86::Condition_code::below: cc = x86::Condition_code::above; break;
            case x86::Condition_code::above_equal: cc = x86::Condition_code::below_equal; break;
            default: break;
        }

        *this << cmp(qword(memory_of_register(rs2)), 0);
    } else {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
        *this << cmp(x86::Register::rax, qword(memory_of_register(rs2)));
    }

    // If flag set, then change rax to offset of new target
    *this << mov(x86::Register::rdx, pc_diff - inst.length() + inst.imm());
    *this << mov(x86::Register::rax, pc_diff);
    *this << cmovcc(cc, x86::Register::rax, x86::Register::rdx);

    // Update pc
    *this << add(qword(memory_of(pc)), x86::Register::rax);

    *this << pop(x86::Register::rbp);
    *this << ret();
}

void Dbt_compiler::emit_jalr(riscv::Instruction inst, riscv::reg_t pc_diff) {
    const int rd = inst.rd();
    const int rs1 = inst.rs1();
    riscv::reg_t imm = inst.imm();

    if (rd != 0) {
        *this << mov(x86::Register::rdx, qword(memory_of(pc)));
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));

    if (imm != 0) {
        *this << add(x86::Register::rax, imm);
    }

    *this << i_and(x86::Register::rax, ~1);
    *this << mov(qword(memory_of(pc)), x86::Register::rax);

    if (rd != 0) {
        *this << add(x86::Register::rdx, pc_diff);
        *this << mov(qword(memory_of_register(rd)), x86::Register::rdx);
    }

    *this << pop(x86::Register::rbp);
    *this << ret();
}

void Dbt_compiler::emit_jal(riscv::Instruction inst, riscv::reg_t pc_diff) {
    const int rd = inst.rd();

    if (rd != 0) {
        *this << mov(x86::Register::rax, qword(memory_of(pc)));
    }

    *this << add(qword(memory_of(pc)), pc_diff - inst.length() + inst.imm());

    if (rd != 0) {
        *this << add(x86::Register::rax, pc_diff);
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
    }

    *this << pop(x86::Register::rbp);
    *this << ret();
}

void Dbt_compiler::emit_lb(riscv::Instruction inst, bool u) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    riscv::reg_t imm = inst.imm();

    // We can generate better code if the MMU is flat.
    if (!emu::state::no_direct_memory_access) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));

        if (u) {
            *this << movzx(x86::Register::eax, byte(x86::Register::rax + imm));
        } else {
            *this << movsx(x86::Register::rax, byte(x86::Register::rax + imm));
        }

    } else {
        *this << mov(x86::Register::rdi, qword(memory_of_register(rs1)));
        if (imm != 0) {
            *this << add(x86::Register::rdi, imm);
        }

        *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(&emu::load_memory<uint8_t>));
        *this << call(x86::Register::rax);
        if (rd != 0) {
            if (u) {
                // High 32 bits in rax may contain garbage, so do a mov to zero higher bits.
                *this << movzx(x86::Register::eax, x86::Register::al);
            } else {
                *this << movsx(x86::Register::rax, x86::Register::al);
            }
        }
    }

    if (rd != 0) {
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
    }
}

void Dbt_compiler::emit_lh(riscv::Instruction inst, bool u) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    riscv::reg_t imm = inst.imm();

    // We can generate better code if the MMU is flat.
    if (!emu::state::no_direct_memory_access) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));

        if (u) {
            *this << movzx(x86::Register::eax, word(x86::Register::rax + imm));
        } else {
            *this << movsx(x86::Register::rax, word(x86::Register::rax + imm));
        }

    } else {
        *this << mov(x86::Register::rdi, qword(memory_of_register(rs1)));
        if (imm != 0) {
            *this << add(x86::Register::rdi, imm);
        }

        *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(&emu::load_memory<uint16_t>));
        *this << call(x86::Register::rax);
        if (rd != 0) {
            if (u) {
                // High 32 bits in rax may contain garbage, so do a mov to zero higher bits.
                *this << movzx(x86::Register::eax, x86::Register::ax);
            } else {
                *this << movsx(x86::Register::rax, x86::Register::ax);
            }
        }
    }

    if (rd != 0) {
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
    }
}

void Dbt_compiler::emit_lw(riscv::Instruction inst, bool u) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    riscv::reg_t imm = inst.imm();

    // We can generate better code if the MMU is flat.
    if (!emu::state::no_direct_memory_access) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));

        if (u) {
            *this << mov(x86::Register::eax, dword(x86::Register::rax + imm));
        } else {
            *this << movsx(x86::Register::rax, dword(x86::Register::rax + imm));
        }

    } else {
        *this << mov(x86::Register::rdi, qword(memory_of_register(rs1)));
        if (imm != 0) {
            *this << add(x86::Register::rdi, imm);
        }

        *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(&emu::load_memory<uint32_t>));
        *this << call(x86::Register::rax);
        if (rd != 0) {
            if (u) {
                // High 32 bits in rax may contain garbage, so do a mov to zero higher bits.
                *this << mov(x86::Register::eax, x86::Register::eax);
            } else {
                *this << cdqe();
            }
        }
    }

    if (rd != 0) {
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
    }
}

void Dbt_compiler::emit_ld(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    riscv::reg_t imm = inst.imm();

    // We can generate better code if the MMU is flat.
    if (!emu::state::no_direct_memory_access) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
        *this << mov(x86::Register::rax, qword(x86::Register::rax + imm));

    } else {
        *this << mov(x86::Register::rdi, qword(memory_of_register(rs1)));
        if (imm != 0) {
            *this << add(x86::Register::rdi, imm);
        }

        *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(&emu::load_memory<uint64_t>));
        *this << call(x86::Register::rax);
    }

    if (rd != 0) {
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
    }
}

void Dbt_compiler::emit_sb(riscv::Instruction inst) {
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();
    riscv::reg_t imm = inst.imm();

    // We can generate better code if the MMU is id.
    if (!emu::state::no_direct_memory_access) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));

        if (rs2 == 0) {
            *this << mov(byte(x86::Register::rax + imm), 0);
        } else {
            *this << mov(x86::Register::dl, byte(memory_of_register(rs2)));
            *this << mov(byte(x86::Register::rax + imm), x86::Register::dl);
        }

    } else {
        *this << mov(x86::Register::rdi, qword(memory_of_register(rs1)));
        if (imm != 0) {
            *this << add(x86::Register::rdi, imm);
        }

        if (rs2 == 0) {
            *this << mov(x86::Register::rsi, 0);
        } else {
            *this << mov(x86::Register::sil, byte(memory_of_register(rs2)));
        }

        *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(&emu::store_memory<uint8_t>));
        *this << call(x86::Register::rax);
    }
}

void Dbt_compiler::emit_sh(riscv::Instruction inst) {
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();
    riscv::reg_t imm = inst.imm();

    // We can generate better code if the MMU is flat.
    if (!emu::state::no_direct_memory_access) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));

        if (rs2 == 0) {
            *this << mov(word(x86::Register::rax + imm), 0);
        } else {
            *this << mov(x86::Register::dx, word(memory_of_register(rs2)));
            *this << mov(word(x86::Register::rax + imm), x86::Register::dx);
        }
        // TODO: Add bounds checking
    } else {
        *this << mov(x86::Register::rdi, qword(memory_of_register(rs1)));
        if (imm != 0) {
            *this << add(x86::Register::rdi, imm);
        }

        if (rs2 == 0) {
            *this << mov(x86::Register::rsi, 0);
        } else {
            *this << mov(x86::Register::si, word(memory_of_register(rs2)));
        }

        *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(&emu::store_memory<uint16_t>));
        *this << call(x86::Register::rax);
    }
}

void Dbt_compiler::emit_sw(riscv::Instruction inst) {
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();
    riscv::reg_t imm = inst.imm();

    // We can generate better code if the MMU is flat.
    if (!emu::state::no_direct_memory_access) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));

        if (rs2 == 0) {
            *this << mov(dword(x86::Register::rax + imm), 0);
        } else {
            *this << mov(x86::Register::edx, dword(memory_of_register(rs2)));
            *this << mov(dword(x86::Register::rax + imm), x86::Register::edx);
        }

    } else {
        *this << mov(x86::Register::rdi, qword(memory_of_register(rs1)));
        if (imm != 0) {
            *this << add(x86::Register::rdi, imm);
        }

        if (rs2 == 0) {
            *this << mov(x86::Register::rsi, 0);
        } else {
            *this << mov(x86::Register::esi, dword(memory_of_register(rs2)));
        }

        *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(&emu::store_memory<uint32_t>));
        *this << call(x86::Register::rax);
    }
}

void Dbt_compiler::emit_sd(riscv::Instruction inst) {
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();
    riscv::reg_t imm = inst.imm();

    // We can generate better code if the MMU is flat.
    if (!emu::state::no_direct_memory_access) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));

        if (rs2 == 0) {
            *this << mov(qword(x86::Register::rax + imm), 0);
        } else {
            *this << mov(x86::Register::rdx, qword(memory_of_register(rs2)));
            *this << mov(qword(x86::Register::rax + imm), x86::Register::rdx);
        }

    } else {
        *this << mov(x86::Register::rdi, qword(memory_of_register(rs1)));
        if (imm != 0) {
            *this << add(x86::Register::rdi, imm);
        }

        if (rs2 == 0) {
            *this << mov(x86::Register::rsi, 0);
        } else {
            *this << mov(x86::Register::rsi, qword(memory_of_register(rs2)));
        }

        *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(&emu::store_memory<uint64_t>));
        *this << call(x86::Register::rax);
    }
}

void Dbt_compiler::emit_addi(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    riscv::reg_t imm = inst.imm();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_load_immediate(rd, imm);
        return;
    }

    if (imm == 0) {
        emit_move(rd, rs1);
        return;
    }

    if (rd == rs1) {
        *this << add(qword(memory_of_register(rd)), imm);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
    *this << add(x86::Register::rax, imm);
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_shifti(riscv::Instruction inst, x86::Opcode opcode) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    riscv::reg_t imm = inst.imm();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    if (imm == 0) {
        emit_move(rd, rs1);
        return;
    }

    if (rd == rs1) {
        *this << binary(opcode, qword(memory_of_register(rd)), imm);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));

    // For left shift by 1, we can use add instead.
    if (opcode == x86::Opcode::shl && imm == 1) {
        *this << add(x86::Register::rax, x86::Register::rax);
    } else {
        *this << binary(opcode, x86::Register::rax, imm);
    }

    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_slti(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    riscv::sreg_t imm = inst.imm();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_load_immediate(rd, imm > 0);
        return;
    }

    // When immediate is zero, this instruction basically determines the sign of the value in rs1. We can logical right
    // shift the value by 63 bits to achieve the same result.
    if (imm == 0) {
        if (rd == rs1) {
            *this << shr(qword(memory_of_register(rd)), 63);
            return;
        }

        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
        *this << shr(x86::Register::rax, 63);
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    // For positive numbers we decrease the value by one and the compare less equal. This can allow 1 more possible
    // immediate value to use shorter encoding.
    x86::Condition_code cc = x86::Condition_code::less;
    if (imm > 0) {
        imm--;
        cc = x86::Condition_code::less_equal;
    }

    *this << i_xor(x86::Register::eax, x86::Register::eax);
    *this << cmp(qword(memory_of_register(rs1)), imm);
    *this << setcc(cc, x86::Register::al);
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_sltiu(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    // Even though the instruction is sltiu, we still convert it to signed integer to ease code generation.
    riscv::sreg_t imm = inst.imm();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_load_immediate(rd, imm != 0);
        return;
    }

    if (imm == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    x86::Condition_code cc;
    if (imm > 0) {
        imm--;
        cc = imm == 0 ? x86::Condition_code::equal : x86::Condition_code::below_equal;
    } else {
        cc = imm == -1 ? x86::Condition_code::not_equal : x86::Condition_code::below;
    }

    *this << i_xor(x86::Register::eax, x86::Register::eax);
    *this << cmp(qword(memory_of_register(rs1)), imm);
    *this << setcc(cc, x86::Register::al);
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_xori(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    riscv::sreg_t imm = inst.imm();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_load_immediate(rd, imm);
        return;
    }

    if (imm == 0) {
        emit_move(rd, rs1);
        return;
    }

    if (rd == rs1) {
        if (imm == -1) {
            *this << i_not(qword(memory_of_register(rd)));
            return;
        }

        *this << i_xor(qword(memory_of_register(rd)), imm);
        return;
    }

    if (imm == -1) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
        *this << i_not(x86::Register::rax);
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
    *this << i_xor(x86::Register::rax, imm);
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_ori(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    riscv::sreg_t imm = inst.imm();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_load_immediate(rd, imm);
        return;
    }

    if (imm == 0) {
        emit_move(rd, rs1);
        return;
    }

    if (imm == -1) {
        emit_load_immediate(rd, -1);
        return;
    }

    if (rd == rs1) {
        *this << i_or(qword(memory_of_register(rd)), imm);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
    *this << i_or(x86::Register::rax, imm);
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_andi(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    riscv::reg_t imm = inst.imm();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    if (imm == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    if (imm == static_cast<riscv::reg_t>(-1)) {
        emit_move(rd, rs1);
        return;
    }

    if (rd == rs1) {
        *this << i_and(qword(memory_of_register(rd)), imm);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
    *this << i_and(x86::Register::rax, imm);
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_addiw(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    riscv::reg_t imm = inst.imm();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_load_immediate(rd, imm);
        return;
    }

    if (imm == 0) {
        emit_move32(rd, rs1);
        return;
    }

    *this << mov(x86::Register::eax, dword(memory_of_register(rs1)));
    *this << add(x86::Register::eax, imm);
    *this << cdqe();
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_shiftiw(riscv::Instruction inst, x86::Opcode opcode) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    riscv::reg_t imm = inst.imm();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    if (imm == 0) {
        emit_move32(rd, rs1);
        return;
    }

    *this << mov(x86::Register::eax, dword(memory_of_register(rs1)));

    if (opcode == x86::Opcode::shl && imm == 1) {
        *this << add(x86::Register::eax, x86::Register::eax);
    } else {
        *this << binary(opcode, x86::Register::eax, imm);
    }

    *this << cdqe();
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_add(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_move(rd, rs2);
        return;
    }

    if (rs2 == 0) {
        emit_move(rd, rs1);
        return;
    }

    // Add one variable to itself can be efficiently implemented as an in-place shift.
    if (rd == rs1 && rd == rs2) {
        *this << shl(qword(memory_of_register(rd)), 1);
        return;
    }

    if (rd == rs1) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs2)));
        *this << add(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    if (rd == rs2) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
        *this << add(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    if (rs1 == rs2) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
        *this << add(x86::Register::rax, x86::Register::rax);
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
    *this << add(x86::Register::rax, qword(memory_of_register(rs2)));
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_sub(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    // rd = rs1 - 0
    if (rs2 == 0) {
        emit_move(rd, rs1);
        return;
    }

    // rd = rs1 - rs1 = 0
    if (rs1 == rs2) {
        emit_load_immediate(rd, 0);
        return;
    }

    // rd -= rs2
    if (rd == rs1) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs2)));
        *this << sub(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    // rd = -rd
    if (rd == rs2 && rs1 == 0) {
        *this << neg(qword(memory_of_register(rd)));
        return;
    }

    // rd = -rs2
    if (rs1 == 0) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs2)));
        *this << neg(x86::Register::rax);
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
    *this << sub(x86::Register::rax, qword(memory_of_register(rs2)));
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_shift(riscv::Instruction inst, x86::Opcode opcode) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    if (rs2 == 0) {
        emit_move(rd, rs1);
        return;
    }

    if (rd == rs1) {
        *this << mov(x86::Register::cl, byte(memory_of_register(rs2)));
        *this << binary(opcode, qword(memory_of_register(rd)), x86::Register::cl);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
    *this << mov(x86::Register::cl, byte(memory_of_register(rs2)));
    *this << binary(opcode, x86::Register::rax, x86::Register::cl);
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_slt(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs1 == rs2) {
        emit_load_immediate(rd, 0);
        return;
    }

    if (rs1 == 0) {
        *this << i_xor(x86::Register::eax, x86::Register::eax);
        *this << cmp(qword(memory_of_register(rs2)), 0);
        *this << setcc(x86::Condition_code::greater, x86::Register::al);
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    // Similar to slti, shift by 63 bits yield the sign.
    if (rs2 == 0) {
        if (rd == rs1) {
            *this << shr(qword(memory_of_register(rd)), 63);
            return;
        }

        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
        *this << shr(x86::Register::rax, 63);
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
    *this << cmp(x86::Register::rax, qword(memory_of_register(rs2)));
    *this << setcc(x86::Condition_code::less, x86::Register::al);
    *this << movzx(x86::Register::eax, x86::Register::al);
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_sltu(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs2 == 0 || rs1 == rs2) {
        emit_load_immediate(rd, 0);
        return;
    }

    // snez
    if (rs1 == 0) {
        *this << i_xor(x86::Register::eax, x86::Register::eax);
        *this << cmp(qword(memory_of_register(rs2)), 0);
        *this << setcc(x86::Condition_code::not_equal, x86::Register::al);
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
    *this << cmp(x86::Register::rax, qword(memory_of_register(rs2)));
    *this << setcc(x86::Condition_code::below, x86::Register::al);
    *this << movzx(x86::Register::eax, x86::Register::al);
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_and(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs1 == 0 || rs2 == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    if (rs1 == rs2) {
        emit_move(rd, rs1);
        return;
    }

    if (rd == rs1) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs2)));
        *this << i_and(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    if (rd == rs2) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
        *this << i_and(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
    *this << i_and(x86::Register::rax, qword(memory_of_register(rs2)));
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_xor(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_move(rd, rs2);
        return;
    }

    if (rs2 == 0) {
        emit_move(rd, rs1);
        return;
    }

    if (rs1 == rs2) {
        emit_load_immediate(rd, 0);
        return;
    }

    if (rd == rs1) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs2)));
        *this << i_xor(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    if (rd == rs2) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
        *this << i_xor(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
    *this << i_xor(x86::Register::rax, qword(memory_of_register(rs2)));
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_or(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs1 == 0 || rs1 == rs2) {
        emit_move(rd, rs2);
        return;
    }

    if (rs2 == 0) {
        emit_move(rd, rs1);
        return;
    }

    if (rd == rs1) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs2)));
        *this << i_or(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    if (rd == rs2) {
        *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
        *this << i_or(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));
    *this << i_or(x86::Register::rax, qword(memory_of_register(rs2)));
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_addw(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_move32(rd, rs2);
        return;
    }

    if (rs2 == 0) {
        emit_move32(rd, rs1);
        return;
    }

    if (rs1 == rs2) {
        // ADDW rd, rs1, rs1
        *this << mov(x86::Register::eax, dword(memory_of_register(rs1)));
        *this << add(x86::Register::eax, x86::Register::eax);
    } else {
        *this << mov(x86::Register::eax, dword(memory_of_register(rs1)));
        *this << add(x86::Register::eax, dword(memory_of_register(rs2)));
    }

    *this << cdqe();
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_subw(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs2 == 0) {
        emit_move32(rd, rs1);
        return;
    }

    if (rs1 == rs2) {
        emit_load_immediate(rd, 0);
        return;
    }

    if (rs1 == 0) {
        *this << mov(x86::Register::eax, dword(memory_of_register(rs2)));
        *this << neg(x86::Register::eax);
        *this << cdqe();
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    *this << mov(x86::Register::eax, dword(memory_of_register(rs1)));
    *this << sub(x86::Register::eax, dword(memory_of_register(rs2)));
    *this << cdqe();
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_shiftw(riscv::Instruction inst, x86::Opcode opcode) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs1 == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    if (rs2 == 0) {
        emit_move32(rd, rs1);
        return;
    }

    *this << mov(x86::Register::eax, dword(memory_of_register(rs1)));
    *this << mov(x86::Register::cl, byte(memory_of_register(rs2)));
    *this << binary(opcode, x86::Register::eax, x86::Register::cl);
    *this << cdqe();
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_mul(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs1 == 0 || rs2 == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));

    if (rs1 == rs2) {
        *this << imul(x86::Register::rax, x86::Register::rax);
    } else {
        *this << imul(x86::Register::rax, qword(memory_of_register(rs2)));
    }

    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_mulh(riscv::Instruction inst, bool u) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs1 == 0 || rs2 == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));

    if (rs1 == rs2) {
        *this << unary(u ? x86::Opcode::mul : x86::Opcode::imul, x86::Register::rax);
    } else {
        *this << unary(u ? x86::Opcode::mul : x86::Opcode::imul, qword(memory_of_register(rs2)));
    }

    *this << mov(qword(memory_of_register(rd)), x86::Register::rdx);
}

void Dbt_compiler::emit_mulhsu(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs1 == 0 || rs2 == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    // Load value to register and multiply.
    *this << mov(x86::Register::rcx, qword(memory_of_register(rs1)));
    *this << mov(x86::Register::rsi, qword(memory_of_register(rs2)));
    *this << mov(x86::Register::rax, x86::Register::rcx);
    *this << mul(x86::Register::rsi);

    // Fix up negative by: if (rs1 < 0) rd = rd - rs2
    // Note that this is identical to rd = rd -(rs1 >>> 63) & rs2
    *this << sar(x86::Register::rcx, 63);
    *this << i_and(x86::Register::rcx, x86::Register::rsi);
    *this << sub(x86::Register::rdx, x86::Register::rcx);
    *this << mov(qword(memory_of_register(rd)), x86::Register::rdx);
}

void Dbt_compiler::emit_mulw(riscv::Instruction inst) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    if (rs1 == 0 || rs2 == 0) {
        emit_load_immediate(rd, 0);
        return;
    }

    *this << mov(x86::Register::eax, dword(memory_of_register(rs1)));

    if (rs1 == rs2) {
        *this << imul(x86::Register::eax, x86::Register::eax);
    } else {
        *this << imul(x86::Register::eax, dword(memory_of_register(rs2)));
    }

    *this << cdqe();
    *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
}

void Dbt_compiler::emit_div(riscv::Instruction inst, bool u, bool rem) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    // x / 0 = -1, x % 0 = x
    if (rs2 == 0) {
        if (rem) {
            emit_move(rd, rs1);
        } else {
            emit_load_immediate(rd, -1);
        }
        return;
    }

    if (rs1 == 0) {
        // 0 % x = 0, and 0 % 0 = 0.
        if (rem) {
            emit_load_immediate(rd, 0);
            return;
        }

        // 0 / x = 0, but 0 / 0 = -1.
        *this << cmp(qword(memory_of_register(rs2)), 1);
        *this << sbb(x86::Register::rax, x86::Register::rax);
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    *this << mov(x86::Register::rax, qword(memory_of_register(rs1)));

    if (u) {
        *this << i_xor(x86::Register::edx, x86::Register::edx);
    } else {
        *this << cqo();
    }

    *this << unary(u ? x86::Opcode::div : x86::Opcode::idiv, qword(memory_of_register(rs2)));
    *this << mov(qword(memory_of_register(rd)), rem ? x86::Register::rdx : x86::Register::rax);
}

void Dbt_compiler::emit_divw(riscv::Instruction inst, bool u, bool rem) {
    int rd = inst.rd();
    int rs1 = inst.rs1();
    int rs2 = inst.rs2();

    if (rd == 0) return;

    // x / 0 = -1, x % 0 = x
    if (rs2 == 0) {
        if (rem) {
            emit_move32(rd, rs1);
        } else {
            emit_load_immediate(rd, -1);
        }
        return;
    }

    if (rs1 == 0) {
        // 0 % x = 0, and 0 % 0 = 0.
        if (rem) {
            emit_load_immediate(rd, 0);
            return;
        }

        // 0 / x = 0, but 0 / 0 = -1.
        *this << cmp(dword(memory_of_register(rs2)), 1);
        *this << sbb(x86::Register::rax, x86::Register::rax);
        *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
        return;
    }

    *this << mov(x86::Register::eax, dword(memory_of_register(rs1)));

    if (u) {
        *this << i_xor(x86::Register::edx, x86::Register::edx);
    } else {
        *this << cdq();
    }

    *this << unary(u ? x86::Opcode::div : x86::Opcode::idiv, dword(memory_of_register(rs2)));

    if (rem) {
        *this << movsx(x86::Register::rdx, x86::Register::edx);
    } else {
        *this << cdqe();
    }

    *this << mov(qword(memory_of_register(rd)), rem ? x86::Register::rdx : x86::Register::rax);
}
