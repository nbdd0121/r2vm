#include <list>

#include "emu/state.h"
#include "util/memory.h"
#include "x86/backend.h"
#include "x86/builder.h"
#include "x86/disassembler.h"

using namespace x86::builder;

/* Helper functions to assist code generation. */

// Get the index of the register. AH/CH/DH/BH are not considered here as they will never appear in our generated code.
static int register_id(x86::Register reg) {
    ASSERT(reg != x86::Register::none);
    return static_cast<uint8_t>(reg) & 0xF;
}

// Get the register name with required index and width.
static x86::Register register_of_id(ir::Type type, int reg) {
    using namespace x86;
    switch (type) {
        case ir::Type::i1:
        case ir::Type::i8: return static_cast<Register>(reg | (reg >= 4 && reg <= 7 ? reg_gpb2 : reg_gpb));
        case ir::Type::i16: return static_cast<Register>(reg | reg_gpw);
        case ir::Type::i32: return static_cast<Register>(reg | reg_gpd);
        case ir::Type::i64: return static_cast<Register>(reg | reg_gpq);
        default: ASSERT(0);
    }
}

// Tell if two operands refers to the same memory location. Sizes are not considered here.
static bool same_location(const x86::Operand& a, const x86::Operand& b) {

    // Always return false if either operand is an immediate.
    if (a.is_immediate() || b.is_immediate()) return false;

    // Check whether the register is the same
    if (a.is_register()) {
        if (!b.is_register()) return false;
        return register_id(a.as_register()) == register_id(b.as_register());
    }

    ASSERT(a.is_memory());
    if (!b.is_memory()) return false;
    const x86::Memory& am = a.as_memory();
    const x86::Memory& bm = b.as_memory();
    return am.base == bm.base && am.index == bm.index && am.scale == bm.scale && am.displacement == bm.displacement;
}

// Get the corresponding register name with specified width.
static x86::Register modify_size(ir::Type type, x86::Register loc) {
    return register_of_id(type, register_id(loc));
}

static x86::Operand modify_size(ir::Type type, const x86::Operand& loc) {
    using namespace x86;
    if (loc.is_register()) {
        return modify_size(type, loc.as_register());
    } else if (loc.is_immediate()) {
        return loc;
    } else {
        Memory mem = loc.as_memory();
        mem.size = ir::get_type_size(type) / 8;
        return mem;
    }
}

namespace x86::backend {

void Code_generator::emit(const Instruction& inst) {
    bool disassemble = emu::state::disassemble;
    size_t size_before_emit;
    if (disassemble) {
        size_before_emit = _encoder.buffer().size();
    }
    try {
        _encoder.encode(inst);
    } catch (...) {
        if (disassemble) {
            x86::disassembler::print_instruction(
                reinterpret_cast<uintptr_t>(_encoder.buffer().data() + size_before_emit), nullptr, 0, inst
            );
        }
        throw;
    }
    if (disassemble) {
        std::byte *pc = _encoder.buffer().data() + size_before_emit;
        std::byte *new_pc = _encoder.buffer().data() + _encoder.buffer().size();
        x86::disassembler::print_instruction(
            reinterpret_cast<uintptr_t>(pc), reinterpret_cast<const char*>(pc), new_pc - pc, inst);
    }
}

void Code_generator::emit_move(ir::Type type, const Operand& dst, const Operand& src) {

    // Ignore move to self.
    if (!src.is_immediate() && same_location(dst, src)) return;

    if (dst.is_memory() || src.is_memory() || type == ir::Type::i64 || type == ir::Type::i32) {
        emit(mov(dst, src));

    } else {

        // 32-bit move is shorter than 16 or 8-bit move.
        emit(mov(modify_size(ir::Type::i32, dst), modify_size(ir::Type::i32, src)));
    }
}

Operand Code_generator::get_allocation(ir::Value value) {
    if (value.is_const()) {
        return value.const_value();
    }

    return _regalloc.get_allocation(value);
}

void Code_generator::emit_binary(ir::Node* node, Opcode opcode) {
    Register reg = get_allocation(node->value(0)).as_register();
    ASSERT(same_location(get_allocation(node->operand(0)), reg));

    emit(binary(opcode, reg, get_allocation(node->operand(1))));
}

void Code_generator::emit_unary(ir::Node* node, Opcode opcode) {
    Register reg = get_allocation(node->value(0)).as_register();
    ASSERT(same_location(get_allocation(node->operand(0)), reg));

    emit(unary(opcode, reg));
}

void Code_generator::emit_mul(ir::Node* node, Opcode opcode) {
    // Make sure all source/dest registers are legal.
    ASSERT(same_location(get_allocation(node->operand(0)), Register::rax));
    ASSERT(same_location(get_allocation(node->value(0)), Register::rax));
    ASSERT(same_location(get_allocation(node->value(1)), Register::rdx));

    emit(unary(opcode, get_allocation(node->operand(1))));
}

void Code_generator::emit_div(ir::Node* node, Opcode opcode) {
    auto quo = node->value(0);

    // Make sure all source/dest registers are legal.
    ASSERT(same_location(get_allocation(node->operand(0)), Register::rax));
    ASSERT(same_location(get_allocation(quo), Register::rax));
    ASSERT(same_location(get_allocation(node->value(1)), Register::rdx));

    // Setup edx/rdx to be zero/sign-extension of dividend.
    if (opcode == Opcode::div) {
        emit(i_xor(x86::Register::edx, x86::Register::edx));
    } else {
        if (quo.type() == ir::Type::i64) {
            emit(cqo());
        } else {
            emit(cdq());
        }
    }

    emit(unary(opcode, get_allocation(node->operand(1))));
}

Condition_code Code_generator::emit_compare(ir::Value value) {
    auto node = value.node();
    Condition_code cc;
    switch (node->opcode()) {
        case ir::Opcode::eq: cc = Condition_code::equal; break;
        case ir::Opcode::ne: cc = Condition_code::not_equal; break;
        case ir::Opcode::lt: cc = Condition_code::less; break;
        case ir::Opcode::ge: cc = Condition_code::greater_equal; break;
        case ir::Opcode::ltu: cc = Condition_code::below; break;
        case ir::Opcode::geu: cc = Condition_code::above_equal; break;
        default: ASSERT(0);
    }

    Operand loc0 = get_allocation(node->operand(0));
    Operand loc1 = get_allocation(node->operand(1));
    if (loc0.is_immediate()) {
        std::swap(loc0, loc1);
        switch (cc) {
            case x86::Condition_code::less: cc = x86::Condition_code::greater; break;
            case x86::Condition_code::greater_equal: cc = x86::Condition_code::less_equal; break;
            case x86::Condition_code::below: cc = x86::Condition_code::above; break;
            case x86::Condition_code::above_equal: cc = x86::Condition_code::below_equal; break;
            default: break;
        }
    }

    emit(cmp(loc0, loc1));
    return cc;
}

Memory Code_generator::emit_address(ir::Type type, ir::Value value) {
    ASSERT(value.opcode() == Target_opcode::address);

    auto node = value.node();
    auto base = node->operand(0);
    auto index = node->operand(1);
    auto scale = node->operand(2);
    auto disp = node->operand(3);

    ASSERT(scale.is_const() && disp.is_const());

    Register base_reg = base.is_const() && base.const_value() == 0
        ? Register::none
        : modify_size(ir::Type::i64, get_allocation(base).as_register());

    if (scale.const_value() == 0) {
        Memory ret = qword(base_reg + disp.const_value());
        ret.size = get_type_size(type) / 8;
        return ret;
    }

    Register index_reg = modify_size(ir::Type::i64, get_allocation(index).as_register());
    Memory ret = qword(base_reg + index_reg * scale.const_value() + disp.const_value());
    ret.size = get_type_size(type) / 8;
    return ret;
}

void Code_generator::visit(ir::Node* node) {
    switch (node->opcode()) {
        case ir::Opcode::load_register: {
            uint16_t regnum = static_cast<ir::Register_access*>(node)->regnum();

            // Move the emulated register to 64-bit version of allocated machine register.
            emit(mov(get_allocation(node->value(1)), qword(Register::rbp + regnum * 8)));
            break;
        }
        case ir::Opcode::store_register: {
            uint16_t regnum = static_cast<ir::Register_access*>(node)->regnum();

            // Move the allocated machine register back to the emulated register.
            emit(mov(qword(Register::rbp + regnum * 8), get_allocation(node->operand(1))));
            break;
        }
        case ir::Opcode::load_memory: {
            auto output = node->value(1);
            emit(mov(get_allocation(output), emit_address(output.type(), node->operand(1))));
            break;
        }
        case ir::Opcode::store_memory: {
            auto value = node->operand(2);
            emit(mov(emit_address(value.type(), node->operand(1)), get_allocation(value)));
            break;
        }
        case Target_opcode::lea: {
            emit(lea(
                modify_size(ir::Type::i64, get_allocation(node->value(0)).as_register()),
                emit_address(ir::Type::i64, node->operand(0))
            ));
            break;
        }
        case ir::Opcode::call: {
            auto call_node = static_cast<ir::Call*>(node);
            if (call_node->need_context()) {
                emit(mov(Register::rdi, Register::rbp));
            }

            // All other arguments should have been placed into the correct register already by register allocator.
            emit(mov(Register::rax, call_node->target()));
            emit(call(Register::rax));
            break;
        }
        case ir::Opcode::copy: {
            auto output = node->value(0);
            emit_move(output.type(), get_allocation(output), get_allocation(node->operand(0)));
            break;
        }
        case ir::Opcode::cast: {
            auto output = node->value(0);
            auto op = node->operand(0);

            Register reg = get_allocation(output).as_register();

            // Special handling for i1 upcast
            if (op.type() == ir::Type::i1) {
                Condition_code cc = emit_compare(op);
                Register reg8 = modify_size(ir::Type::i8, reg);
                emit(setcc(cc, reg8));
                if (output.type() != ir::Type::i8) {
                    emit(movzx(modify_size(ir::Type::i32, reg), reg8));
                }
                break;
            }

            Operand loc0 = get_allocation(op);

            // Get size before and after the cast.
            auto op_type = op.type();
            int old_size = ir::get_type_size(op_type);
            int new_size = ir::get_type_size(output.type());

            if (old_size > new_size) {

                // Down-cast can be treated as simple move. If the size is less than 32-bit, we use 32-bit move.
                emit_move(output.type(), reg, modify_size(output.type(), loc0));

            } else {

                // Up-cast needs actual work.
                if (static_cast<ir::Cast*>(node)->sign_extend()) {
                    if (loc0.is_register()) {
                        if (loc0.as_register() == Register::eax && reg == Register::rax) {
                            emit(cdqe());
                            break;
                        }
                    }
                    emit(movsx(reg, loc0));
                } else {

                    // 32-bit to 64-bit cast is a move.
                    if (op_type == ir::Type::i32) {
                        emit(mov(modify_size(ir::Type::i32, reg), loc0));
                    } else {
                        emit(movzx(modify_size(ir::Type::i32, reg), loc0));
                    }
                }
            }

            break;
        }
        case ir::Opcode::mux: {
            Condition_code cc = emit_compare(node->operand(0));

            Register reg = get_allocation(node->value(0)).as_register();
            ASSERT(same_location(get_allocation(node->operand(2)), reg));

            emit(cmovcc(cc, reg, get_allocation(node->operand(1))));
            break;
        }
        case ir::Opcode::add: emit_binary(node, Opcode::add); break;
        case ir::Opcode::sub: emit_binary(node, Opcode::sub); break;
        case ir::Opcode::i_xor: emit_binary(node, Opcode::i_xor); break;
        case ir::Opcode::i_and: emit_binary(node, Opcode::i_and); break;
        case ir::Opcode::i_or: emit_binary(node, Opcode::i_or); break;
        case ir::Opcode::shl: emit_binary(node, Opcode::shl); break;
        case ir::Opcode::shr: emit_binary(node, Opcode::shr); break;
        case ir::Opcode::sar: emit_binary(node, Opcode::sar); break;
        case ir::Opcode::i_not: emit_unary(node, Opcode::i_not); break;
        case ir::Opcode::neg: emit_unary(node, Opcode::neg); break;
        case ir::Opcode::mul: emit_mul(node, Opcode::imul); break;
        case ir::Opcode::mulu: emit_mul(node, Opcode::mul); break;
        case ir::Opcode::div: emit_div(node, Opcode::idiv); break;
        case ir::Opcode::divu: emit_div(node, Opcode::div); break;
        default: ASSERT(0);
    }
}

bool Code_generator::need_phi(ir::Value control) {
    auto target = ir::analysis::Block::get_target(control);

    // Find out the operand id, which will tells us the correct operand of PHI node to use.
    size_t id = target->operand_find(control);
    for (auto ref: target->value(0).references()) {
        if (ref->opcode() != ir::Opcode::phi) continue;

        // We do not need emit any instructions for the PHI node if everything stays.
        if (!same_location(get_allocation(ref->operand(id + 1)), get_allocation(ref->value(0)))) return true;
    }

    return false;
}

void Code_generator::emit_phi(ir::Value control) {
    auto target = ir::analysis::Block::get_target(control);

    // List tracking all moves necessary for the control edge. List is used to ease reordering.
    std::list<std::pair<Operand, Operand>> phi_nodes;

    // Find out the operand id, which will tells us the correct operand of PHI node to use.
    size_t id = target->operand_find(control);
    for (auto ref: target->value(0).references()) {
        if (ref->opcode() != ir::Opcode::phi) continue;

        auto src = get_allocation(ref->operand(id + 1));
        auto dst = get_allocation(ref->value(0));
        if (!same_location(dst, src)) phi_nodes.push_back({dst, src});
    }

    if (phi_nodes.empty()) return;

    while (true) {
        bool changed = true;
        while (changed) {
            changed = false;

            // Loop through all PHI nodes, and pick one whose destination is not used
            auto iter = phi_nodes.begin();
            while (iter != phi_nodes.end()) {
                const auto& [dst, src] = *iter;

                // Check if other PHI nodes may need to use old dst.
                bool has_conflict = false;
                auto iter2 = phi_nodes.begin();
                while (iter2 != phi_nodes.end()) {
                    if (iter2 != iter) {
                        const auto& src2 = iter2->second;
                        if (same_location(dst, src2)) {
                            has_conflict = true;
                            break;
                        }
                    }
                    ++iter2;
                }

                if (!has_conflict) {
                    // Move to the designated PHI value storage area.
                    if (src.is_register() || dst.is_register()) {
                        emit(mov(dst, src));
                    } else {
                        emit(mov(Register::r11, src));
                        emit(mov(dst, Register::r11));
                    }

                    changed = true;
                    iter = phi_nodes.erase(iter);
                } else {
                    ++iter;
                }
            }
        }

        // Break if we successfully emitted all moves.
        if (phi_nodes.empty()) break;

        // Now we encounter cyclical dependency. To break the cyclical dependency, we can find a node whose src is also
        // going to modified, then generate `xchg dst, src`, and update dependencies accordingly.
        auto iter = phi_nodes.begin();
        while (iter != phi_nodes.end()) {
            const auto& [dst, src] = *iter;

            // Check if other PHI nodes will write to src.
            bool will_write = false;
            auto iter2 = phi_nodes.begin();
            while (iter2 != phi_nodes.end()) {
                if (iter2 != iter) {
                    const auto& src2 = iter2->first;
                    if (same_location(src, src2)) {
                        will_write = true;
                        break;
                    }
                }
                ++iter2;
            }

            if (will_write) {
                if (src.is_register() || dst.is_register()) {
                    emit(xchg(dst, src));
                } else {
                    emit(mov(Register::r11, src));
                    emit(xchg(Register::r11, dst));
                    emit(mov(src, Register::r11));
                }

                auto iter2 = phi_nodes.begin();
                while (iter2 != phi_nodes.end()) {
                    if (iter2 == iter) {
                        ++iter2;
                        continue;
                    }
                    auto& [dst2, src2] = *iter2;
                    // XXX: This is okay as currently all PHI nodes are i64.
                    if (same_location(src2, src)) {
                        src2 = dst;
                    } else if (same_location(src2, dst)) {
                        src2 = src;
                    } else {
                        ++iter2;
                        continue;
                    }

                    // dst2 and src2 may end up in the same location if modify them. Elide if possible.
                    if (same_location(dst2, src2)) {
                        iter2 = phi_nodes.erase(iter2);
                    } else {
                        ++iter2;
                    }
                }

                changed = true;
                phi_nodes.erase(iter);
                break;
            } else {
                ++iter;
            }
        }

        ASSERT(changed);
    }
}

void Code_generator::run() {

    // Generate epilogue.
    int stack_size = _regalloc.get_stack_size();
    emit(push(Register::rbp));
    emit(mov(Register::rbp, Register::rdi));
    if (stack_size) emit(sub(Register::rsp, stack_size));

    // Get a linear list of blocks.
    std::vector<ir::Node*> blocks = _block_analysis.blocks();

    // Push exit to the block list to ease processing.
    blocks.push_back(_graph.exit());

    // These are used for relocation
    std::unordered_map<ir::Node*, size_t> label_def;
    std::unordered_map<ir::Node*, std::vector<size_t>> label_use;
    std::vector<size_t> trampoline_loc;

    size_t exit_refcount = _graph.exit()->operand_count();

    emit_phi(_graph.entry()->value(0));

    for (size_t i = 0; i < blocks.size() - 1; i++) {
        auto block = blocks[i];
        auto next_block = blocks[i + 1];
        auto end = static_cast<ir::Paired*>(block)->mate();

        // Store the label for relocation purpose.
        label_def[block] = _encoder.buffer().size();

        // Generate code for the block.
        for (auto node: _scheduler.get_node_list(block)) {
            visit(node);
        }

        // This records the fallthrough target.
        ir::Value target_control;
        ir::Node* target = nullptr;

        if (end->opcode() == ir::Opcode::i_if) {

            // Extract targets
            ir::Value target_control_true = end->value(0);
            auto true_target = ir::analysis::Block::get_target(end->value(0));
            target_control = end->value(1);
            target = ir::analysis::Block::get_target(target_control);

            auto op = end->operand(1);
            if (op.is_const()) {
                if (op.const_value()) {
                    target = true_target;
                    target_control = target_control_true;
                }
            } else {
                Condition_code cc = emit_compare(op);
                bool true_need_phi = need_phi(target_control_true);
                bool false_need_phi = need_phi(target_control);

                // It's always better if we don't need PHIs on the true path.
                if (true_need_phi && !false_need_phi) {
                    // Invert condition code.
                    cc = static_cast<Condition_code>(static_cast<uint8_t>(cc) ^ 1);
                    std::swap(true_target, target);
                    std::swap(target_control_true, target_control);
                    std::swap(true_need_phi, false_need_phi);
                }

                // If we can swap targets to get a fallthrough.
                if (true_target == next_block && true_need_phi == false_need_phi) {
                    // Invert condition code.
                    cc = static_cast<Condition_code>(static_cast<uint8_t>(cc) ^ 1);
                    std::swap(true_target, target);
                    std::swap(target_control_true, target_control);
                }

                if (true_need_phi) {
                    emit(jcc(static_cast<Condition_code>(static_cast<uint8_t>(cc) ^ 1), 0xAAAA));
                    size_t s = _encoder.buffer().size();
                    emit_phi(target_control_true);
                    emit(jmp(0xCAFE));

                    label_use[true_target].push_back(_encoder.buffer().size());

                    size_t s2 = _encoder.buffer().size();
                    util::write_as<uint32_t>(_encoder.buffer().data() + s - 4, s2 - s);

                } else {
                    emit(jcc(cc, 0xAAAA));
                    label_use[true_target].push_back(_encoder.buffer().size());
                }
            }
        } else {
            ASSERT(end->opcode() == ir::Opcode::jmp);
            target_control = end->value(0);
            target = ir::analysis::Block::get_target(target_control);
            ir::Value target_pc = ir::analysis::Block::get_tail_jmp_pc(target_control, 64);
            if (target_pc && target_pc.is_const()) {
                // If the pc is set to a constant before the jump, we would like to emit
                //     jmp translated_address
                // But since it is possible that the target is not yet translated, we generate
                // .trampoline:
                //     mov rax, .trampoline
                //     ret
                // And then when the target address is known, the trampoline will be replaced with the jump.

                if (stack_size) emit(add(Register::rsp, stack_size));

                trampoline_loc.push_back(_encoder.buffer().size());

                // Trampoline. It will be patched later.
                emit(pop(Register::rbp));
                emit(mov(Register::rax, 0xCCCCCCCCC));
                emit(ret());

                exit_refcount--;
                continue;
            }
        }

        emit_phi(target_control);

        if (target != next_block) {
            emit(jmp(0xBEEF));
            label_use[target].push_back(_encoder.buffer().size());
        }
    }

    label_def[_graph.exit()] = _encoder.buffer().size();

    // Patching labels
    for (const auto& pair: label_def) {
        auto& uses = label_use[pair.first];
        for (auto use: uses) {
            util::write_as<uint32_t>(_encoder.buffer().data() + use - 4, pair.second - use);
        }
    }

    // Patching trampolines.
    for (auto loc: trampoline_loc) {
        uintptr_t rip = reinterpret_cast<uintptr_t>(_encoder.buffer().data()) + loc;
        util::write_as<uint64_t>(reinterpret_cast<void*>(rip + 3), rip);
    }

    if (exit_refcount) {
        if (stack_size) emit(add(Register::rsp, stack_size));
        emit(pop(Register::rbp));

        // Return 0, meaning that nothing needs to be patched.
        emit(i_xor(Register::eax, Register::eax));
        emit(ret());
    }
}

}
