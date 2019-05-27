#include "emu/state.h"
#include "ir/builder.h"
#include "ir/node.h"
#include "riscv/basic_block.h"
#include "riscv/context.h"
#include "riscv/frontend.h"
#include "riscv/instruction.h"
#include "riscv/opcode.h"
#include "util/memory.h"

namespace riscv {

struct Frontend {
    ir::Graph graph;
    ir::Builder builder {graph};
    const Basic_block* block;

    ir::Node* block_node;

    // The latest memory value.
    ir::Value last_memory;

    // Current pc (before the processing instruction).
    emu::reg_t pc;

    // Difference between stored instret and true instret (excluding the processing instruction).
    emu::reg_t instret;

    ir::Value emit_load_register(ir::Type type, uint16_t reg);
    void emit_store_register(uint16_t reg, ir::Value value, bool sext = true);

    // If some instruction has the possibility to throw, for correctness we need to update pc and instret to
    // correct value before the instruction.
    void update_pc();
    void update_instret();

    void emit_load(Instruction inst, ir::Type type, bool sext);
    void emit_store(Instruction inst, ir::Type type);
    void emit_alui(Instruction inst, uint16_t opcode, bool w);
    void emit_shifti(Instruction inst, uint16_t opcode, bool w);
    void emit_slti(Instruction inst, uint16_t opcode);
    void emit_alu(Instruction inst, uint16_t opcode, bool w);
    void emit_shift(Instruction inst, uint16_t opcode, bool w);
    void emit_slt(Instruction inst, uint16_t opcode);
    void emit_mul(Instruction inst);
    void emit_div(Instruction inst, uint16_t opcode, bool rem, bool w);
    void emit_branch(Instruction instead, uint16_t opcode, emu::reg_t pc);

    void compile(const Basic_block& block);
};

ir::Value Frontend::emit_load_register(ir::Type type, uint16_t reg) {
    ir::Value ret;
    if (reg == 0) {
        ret = builder.constant(type, 0);
    } else {
        std::tie(last_memory, ret) = builder.load_register(last_memory, reg);
        if (type != ir::Type::i64) ret = builder.cast(type, false, ret);
    }
    return ret;
}

void Frontend::emit_store_register(uint16_t reg, ir::Value value, bool sext) {
    ASSERT(reg != 0);
    if (value.type() != ir::Type::i64) value = builder.cast(ir::Type::i64, sext, value);
    last_memory = builder.store_register(last_memory, reg, value);
}

void Frontend::update_pc() {
    // Update pc
    auto pc_value = builder.constant(ir::Type::i64, pc);
    last_memory = builder.store_register(last_memory, 64, pc_value);
}

void Frontend::update_instret() {
    // Update instret
    if (!emu::state::no_instret) {
        ir::Value instret_value;
        std::tie(last_memory, instret_value) = builder.load_register(last_memory, 65);
        auto instret_offset_value = builder.constant(ir::Type::i64, instret);
        auto new_instret_value = builder.arithmetic(ir::Opcode::add, instret_value, instret_offset_value);
        last_memory = builder.store_register(last_memory, 65, new_instret_value);
        instret = 0;
    }
}

void Frontend::emit_load(Instruction inst, ir::Type type, bool sext) {
    update_pc();
    update_instret();

    auto rs1_value = emit_load_register(ir::Type::i64, inst.rs1());
    auto imm_value = builder.constant(ir::Type::i64, inst.imm());
    auto address = builder.arithmetic(ir::Opcode::add, rs1_value, imm_value);
    ir::Value rd_value;
    std::tie(last_memory, rd_value) = builder.load_memory(last_memory, type, address);
    emit_store_register(inst.rd(), rd_value, sext);
}

void Frontend::emit_store(Instruction inst, ir::Type type) {
    update_pc();
    update_instret();

    auto rs2_value = emit_load_register(type, inst.rs2());
    auto rs1_value = emit_load_register(ir::Type::i64, inst.rs1());
    auto imm_value = builder.constant(ir::Type::i64, inst.imm());
    auto address = builder.arithmetic(ir::Opcode::add, rs1_value, imm_value);
    last_memory = builder.store_memory(last_memory, address, rs2_value);
}

void Frontend::emit_alui(Instruction inst, uint16_t opcode, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_value = emit_load_register(type, inst.rs1());
    auto imm_value = builder.constant(type, inst.imm());
    auto rd_value = builder.arithmetic(opcode, rs1_value, imm_value);
    emit_store_register(inst.rd(), rd_value);
}

void Frontend::emit_shifti(Instruction inst, uint16_t opcode, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_value = emit_load_register(type, inst.rs1());
    auto imm_value = builder.constant(ir::Type::i8, inst.imm());
    auto rd_value = builder.shift(opcode, rs1_value, imm_value);
    emit_store_register(inst.rd(), rd_value);
}

void Frontend::emit_slti(Instruction inst, uint16_t opcode) {
    if (inst.rd() == 0) return;
    auto rs1_value = emit_load_register(ir::Type::i64, inst.rs1());
    auto imm_value = builder.constant(ir::Type::i64, inst.imm());
    auto rd_value = builder.compare(opcode, rs1_value, imm_value);
    emit_store_register(inst.rd(), rd_value);
}

void Frontend::emit_alu(Instruction inst, uint16_t opcode, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_value = emit_load_register(type, inst.rs1());
    auto rs2_value = emit_load_register(type, inst.rs2());
    auto rd_value = builder.arithmetic(opcode, rs1_value, rs2_value);
    emit_store_register(inst.rd(), rd_value);
}

void Frontend::emit_shift(Instruction inst, uint16_t opcode, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_value = emit_load_register(type, inst.rs1());
    auto rs2_value = emit_load_register(ir::Type::i8, inst.rs2());
    auto rd_value = builder.shift(opcode, rs1_value, rs2_value);
    emit_store_register(inst.rd(), rd_value);
}

void Frontend::emit_slt(Instruction inst, uint16_t opcode) {
    if (inst.rd() == 0) return;
    auto rs1_value = emit_load_register(ir::Type::i64, inst.rs1());
    auto rs2_value = emit_load_register(ir::Type::i64, inst.rs2());
    auto rd_value = builder.compare(opcode, rs1_value, rs2_value);
    emit_store_register(inst.rd(), rd_value);
}

void Frontend::emit_mul(Instruction inst) {
    if (inst.rd() == 0) return;

    // Detemrine the type and opcode for IR node first.
    ir::Type type = inst.opcode() == Opcode::mulw ? ir::Type::i32 : ir::Type::i64;
    uint16_t opcode =
        inst.opcode() == Opcode::mulhu || inst.opcode() == Opcode::mulhsu ? ir::Opcode::mulu : ir::Opcode::mul;

    auto rs1_value = emit_load_register(type, inst.rs1());
    auto rs2_value = emit_load_register(type, inst.rs2());
    auto mul_node = builder.create(opcode, {type, type}, {rs1_value, rs2_value});

    if (inst.opcode() == Opcode::mul || inst.opcode() == Opcode::mulw) {
        emit_store_register(inst.rd(), mul_node->value(0));
    } else if (inst.opcode() == Opcode::mulh || inst.opcode() == Opcode::mulhu) {
        emit_store_register(inst.rd(), mul_node->value(1));
    } else {
        // For mulhsu, we translate to the following:
        // First do unsigned multiplication first, then apply fix up. How to fix up is described in dbt.cc and step.cc.
        auto rs1_shift = builder.shift(ir::Opcode::sar, rs1_value, builder.constant(ir::Type::i8, 63));
        auto rs2_masked = builder.arithmetic(ir::Opcode::i_and, rs1_shift, rs2_value);
        auto result = builder.arithmetic(ir::Opcode::sub, mul_node->value(1), rs2_masked);
        emit_store_register(inst.rd(), result);
    }
}

void Frontend::emit_div(Instruction inst, uint16_t opcode, bool rem, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_value = emit_load_register(type, inst.rs1());
    auto rs2_value = emit_load_register(type, inst.rs2());
    auto div_node = builder.create(opcode, {type, type}, {rs1_value, rs2_value});
    emit_store_register(inst.rd(), div_node->value(rem ? 1 : 0));
}

void Frontend::emit_branch(Instruction inst, uint16_t opcode, emu::reg_t pc) {
    auto rs1_value = emit_load_register(ir::Type::i64, inst.rs1());
    auto rs2_value = emit_load_register(ir::Type::i64, inst.rs2());
    auto cmp_value = builder.compare(opcode, rs1_value, rs2_value);

    auto true_pc_value = builder.constant(ir::Type::i64, pc + inst.imm());
    auto false_pc_value = builder.constant(ir::Type::i64, block->end_pc);

    bool use_mux = false;
    if (pc + inst.imm() == block->start_pc) use_mux = false;

    if (use_mux) {
        auto mux_value = builder.mux(cmp_value, true_pc_value, false_pc_value);
        auto store_pc_value = builder.store_register(last_memory, 64, mux_value);
        auto jmp_value = builder.jmp(store_pc_value);

        // Pair block and jmp node up.
        static_cast<ir::Paired*>(jmp_value.node())->mate(block_node);
        static_cast<ir::Paired*>(block_node)->mate(jmp_value.node());

        graph.exit()->operands({jmp_value});

    } else {
        auto if_node = builder.i_if(last_memory, cmp_value);

        // Pair block and if node up.
        static_cast<ir::Paired*>(if_node)->mate(block_node);
        static_cast<ir::Paired*>(block_node)->mate(if_node);

        // Build the false branch.
        auto false_block_value = builder.block({if_node->value(1)});
        auto false_pc_store = builder.store_register(false_block_value, 64, false_pc_value);
        auto false_jmp_value = builder.jmp(false_pc_store);

        // Pair block and jmp node up
        static_cast<ir::Paired*>(false_jmp_value.node())->mate(false_block_value.node());
        static_cast<ir::Paired*>(false_block_value.node())->mate(false_jmp_value.node());

        // If the jump target happens to be the basic block itself, create a loop.
        if (pc + inst.imm() == block->start_pc) {
            (*graph.entry()->value(0).references().begin())->operand_add(if_node->value(0));
            graph.exit()->operands({false_jmp_value});
            return;
        }

        // Building the true branch.
        auto true_block_value = builder.block({if_node->value(0)});
        auto true_pc_store = builder.store_register(true_block_value, 64, true_pc_value);
        auto true_jmp_value = builder.jmp(true_pc_store);

        // Pair block and jmp node up
        static_cast<ir::Paired*>(true_jmp_value.node())->mate(true_block_value.node());
        static_cast<ir::Paired*>(true_block_value.node())->mate(true_jmp_value.node());

        graph.exit()->operands({true_jmp_value, false_jmp_value});
    }
}

void Frontend::compile(const Basic_block& block) {
    this->block = &block;

    auto entry_value = graph.entry()->value(0);
    auto block_value = builder.block({entry_value});
    block_node = block_value.node();
    last_memory = block_value;

    pc = block.start_pc;
    instret = 0;

    for (size_t i = 0; i < block.instructions.size() - 1; i++) {
        auto inst = block.instructions[i];

        switch (inst.opcode()) {
            case Opcode::auipc: {
                if (inst.rd() == 0) break;
                auto rd_value = builder.constant(ir::Type::i64, pc + inst.imm());
                last_memory = builder.store_register(last_memory, inst.rd(), rd_value);
                break;
            }
            case Opcode::lui: {
                if (inst.rd() == 0) break;
                auto imm_value = builder.constant(ir::Type::i64, inst.imm());
                last_memory = builder.store_register(last_memory, inst.rd(), imm_value);
                break;
            }
            case Opcode::fence: break;
            case Opcode::lb: emit_load(inst, ir::Type::i8, true); break;
            case Opcode::lh: emit_load(inst, ir::Type::i16, true); break;
            case Opcode::lw: emit_load(inst, ir::Type::i32, true); break;
            case Opcode::ld: emit_load(inst, ir::Type::i64, false); break;
            case Opcode::lbu: emit_load(inst, ir::Type::i8, false); break;
            case Opcode::lhu: emit_load(inst, ir::Type::i16, false); break;
            case Opcode::lwu: emit_load(inst, ir::Type::i32, false); break;
            case Opcode::sb: emit_store(inst, ir::Type::i8); break;
            case Opcode::sh: emit_store(inst, ir::Type::i16); break;
            case Opcode::sw: emit_store(inst, ir::Type::i32); break;
            case Opcode::sd: emit_store(inst, ir::Type::i64); break;
            case Opcode::addi: emit_alui(inst, ir::Opcode::add, false); break;
            case Opcode::slli: emit_shifti(inst, ir::Opcode::shl, false); break;
            case Opcode::slti: emit_slti(inst, ir::Opcode::lt); break;
            case Opcode::sltiu: emit_slti(inst, ir::Opcode::ltu); break;
            case Opcode::xori: emit_alui(inst, ir::Opcode::i_xor, false); break;
            case Opcode::srli: emit_shifti(inst, ir::Opcode::shr, false); break;
            case Opcode::srai: emit_shifti(inst, ir::Opcode::sar, false); break;
            case Opcode::ori: emit_alui(inst, ir::Opcode::i_or, false); break;
            case Opcode::andi: emit_alui(inst, ir::Opcode::i_and, false); break;
            case Opcode::addiw: emit_alui(inst, ir::Opcode::add, true); break;
            case Opcode::slliw: emit_shifti(inst, ir::Opcode::shl, true); break;
            case Opcode::srliw: emit_shifti(inst, ir::Opcode::shr, true); break;
            case Opcode::sraiw: emit_shifti(inst, ir::Opcode::sar, true); break;
            case Opcode::add: emit_alu(inst, ir::Opcode::add, false); break;
            case Opcode::sub: emit_alu(inst, ir::Opcode::sub, false); break;
            case Opcode::sll: emit_shift(inst, ir::Opcode::shl, false); break;
            case Opcode::slt: emit_slt(inst, ir::Opcode::lt); break;
            case Opcode::sltu: emit_slt(inst, ir::Opcode::ltu); break;
            case Opcode::i_xor: emit_alu(inst, ir::Opcode::i_xor, false); break;
            case Opcode::srl: emit_shift(inst, ir::Opcode::shr, false); break;
            case Opcode::sra: emit_shift(inst, ir::Opcode::sar, false); break;
            case Opcode::i_or: emit_alu(inst, ir::Opcode::i_or, false); break;
            case Opcode::i_and: emit_alu(inst, ir::Opcode::i_and, false); break;
            case Opcode::addw: emit_alu(inst, ir::Opcode::add, true); break;
            case Opcode::subw: emit_alu(inst, ir::Opcode::sub, true); break;
            case Opcode::sllw: emit_shift(inst, ir::Opcode::shl, true); break;
            case Opcode::srlw: emit_shift(inst, ir::Opcode::shr, true); break;
            case Opcode::sraw: emit_shift(inst, ir::Opcode::sar, true); break;
            /* M extension */
            case Opcode::mul:
            case Opcode::mulh:
            case Opcode::mulhsu:
            case Opcode::mulhu:
            case Opcode::mulw: emit_mul(inst); break;
            case Opcode::div: emit_div(inst, ir::Opcode::div, false, false); break;
            case Opcode::divu: emit_div(inst, ir::Opcode::divu, false, false); break;
            case Opcode::rem: emit_div(inst, ir::Opcode::div, true, false); break;
            case Opcode::remu: emit_div(inst, ir::Opcode::divu, true, false); break;
            case Opcode::divw: emit_div(inst, ir::Opcode::div, false, true); break;
            case Opcode::divuw: emit_div(inst, ir::Opcode::divu, false, true); break;
            case Opcode::remw: emit_div(inst, ir::Opcode::div, true, true); break;
            case Opcode::remuw: emit_div(inst, ir::Opcode::divu, true, true); break;
            default: {
                auto serialized_inst = builder.constant(ir::Type::i64, util::read_as<uint64_t>(&inst));
                last_memory = graph.manage(new ir::Call(
                    reinterpret_cast<uintptr_t>(step), true, {ir::Type::memory}, {last_memory, serialized_inst}
                ))->value(0);
                break;
            }
        }

        pc += inst.length();
        instret++;
    }

    // For last instruction, update instret beforehand.
    instret++;
    update_instret();

    auto inst = block.instructions.back();
    switch (inst.opcode()) {
        case Opcode::jal: {
            if (inst.rd()) {
                auto end_pc_value = builder.constant(ir::Type::i64, pc + inst.length());
                last_memory = builder.store_register(last_memory, inst.rd(), end_pc_value);
            }
            ASSERT(pc + inst.length() == block.end_pc);
            auto new_pc_value = builder.constant(ir::Type::i64, pc + inst.imm());
            last_memory = builder.store_register(last_memory, 64, new_pc_value);
            break;
        }
        case Opcode::jalr: {
            auto rs_value = emit_load_register(ir::Type::i64, inst.rs1());
            auto imm_value = builder.constant(ir::Type::i64, inst.imm());
            auto new_pc_value = builder.arithmetic(
                ir::Opcode::i_and,
                builder.arithmetic(ir::Opcode::add, rs_value, imm_value),
                builder.constant(ir::Type::i64, ~1)
            );
            if (inst.rd()) {
                auto end_pc_value = builder.constant(ir::Type::i64, pc + inst.length());
                last_memory = builder.store_register(last_memory, inst.rd(), end_pc_value);
            }
            last_memory = builder.store_register(last_memory, 64, new_pc_value);
            break;
        }
        case Opcode::beq: emit_branch(inst, ir::Opcode::eq, pc); return;
        case Opcode::bne: emit_branch(inst, ir::Opcode::ne, pc); return;
        case Opcode::blt: emit_branch(inst, ir::Opcode::lt, pc); return;
        case Opcode::bge: emit_branch(inst, ir::Opcode::ge, pc); return;
        case Opcode::bltu: emit_branch(inst, ir::Opcode::ltu, pc); return;
        case Opcode::bgeu: emit_branch(inst, ir::Opcode::geu, pc); return;
        default: {
            pc += inst.length();
            update_pc();

            auto serialized_inst = builder.constant(ir::Type::i64, util::read_as<uint64_t>(&inst));
            last_memory = graph.manage(new ir::Call(
                reinterpret_cast<uintptr_t>(step), true, {ir::Type::memory}, {last_memory, serialized_inst}
            ))->value(0);
            break;
        }
    }

    auto jmp_value = builder.jmp(last_memory);

    // Pair block and jmp node up.
    static_cast<ir::Paired*>(jmp_value.node())->mate(block_node);
    static_cast<ir::Paired*>(block_node)->mate(jmp_value.node());

    graph.exit()->operands({jmp_value});
}

ir::Graph compile(const Basic_block& block) {
    Frontend compiler;
    compiler.compile(block);
    return std::move(compiler.graph);
}

}
