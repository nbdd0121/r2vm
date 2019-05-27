#include "util/assert.h"
#include "util/int_size.h"
#include "x86/backend.h"
#include "x86/builder.h"

using namespace x86::builder;

static constexpr int volatile_register[] = {0, 1, 2, 6, 7, 8, 9, 10, 11};

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

namespace x86::backend {

Register Register_allocator::alloc_register_no_spill(ir::Type type, Register hint) {

    // If hint is given, try to use it first
    if (hint != Register::none) {
        int hint_id = register_id(hint);
        ASSERT(!_pinned[hint_id]);
        if (!_register_content[hint_id]) {
            return register_of_id(type, hint_id);
        }
    }

    // Scan through all usable registers for match.
    for (int reg: volatile_register) {
        if (!_pinned[reg] && !_register_content[reg]) {
            return register_of_id(type, reg);
        }
    }

    // Return Register::none means no such register exist.
    return Register::none;
}

Register Register_allocator::alloc_register(ir::Type type, Register hint) {

    // Try to allocate register with hint.
    Register reg = alloc_register_no_spill(type, hint);
    if (reg != Register::none) return reg;

    // Spill out the hint.
    if (hint != Register::none) {
        int hint_id = register_id(hint);
        ASSERT(!_pinned[hint_id]);
        reg = register_of_id(type, hint_id);
        spill_register(reg);
        return reg;
    }

    for (int loc : volatile_register) {
        if (!_pinned[loc]) {
            reg = register_of_id(type, loc);
            spill_register(reg);
            return reg;
        }
    }

    // Running out of registers. This should never happen.
    ASSERT(0);
}

Register Register_allocator::alloc_register(ir::Type type, const Operand& op) {
    if (op.is_register()) return alloc_register(type, op.as_register());
    return alloc_register(type);
}

void Register_allocator::spill_register(Register reg) {

    // Retrieve the value and invalidate.
    auto value = _register_content[register_id(reg)];
    ASSERT(value);
    _register_content[register_id(reg)] = {};

    auto& actual_value = _actual_node[value];

    auto ptr = _memory_node.find(value);
    if (ptr == _memory_node.end()) {

        // Create a copy of the node and assign a stack slot.
        auto copied_value = create_copy(actual_value);
        _allocation[copied_value] = alloc_stack_slot(value.type());

        // Associate the memory node with the original value.
        _memory_node[value] = copied_value;
        actual_value = copied_value;

    } else {

        // The value is spilled already, reuse the value.
        actual_value = ptr->second;
    }
}

Memory Register_allocator::alloc_stack_slot(ir::Type type) {
    Memory mem;
    if (_free_memory.empty()) {
        _stack_size += 8;
        mem = qword(Register::rsp + (_stack_size - 8));
    } else {
        mem = _free_memory.back();
        _free_memory.pop_back();
    }
    mem.size = get_type_size(type) / 8;
    return mem;
}

void Register_allocator::spill_all_registers() {
    for (int reg: volatile_register) {
        if (_register_content[reg]) {
            spill_register(static_cast<Register>(reg | reg_gpq));
        }
    }
}

void Register_allocator::pin_register(Register reg) {
    _pinned[register_id(reg)] = true;
}

void Register_allocator::unpin_register(Register reg) {
    _pinned[register_id(reg)] = false;
}

void Register_allocator::pin_value(ir::Value value) {
    const Operand& loc = _allocation[value];
    if (loc.is_register()) _pinned[register_id(loc.as_register())] = true;
}

void Register_allocator::unpin_value(ir::Value value) {
    const Operand& loc = _allocation[value];
    if (loc.is_register()) _pinned[register_id(loc.as_register())] = false;
}


void Register_allocator::bind_register(ir::Value value, Register loc) {
    _allocation[value] = loc;

    ASSERT(!_register_content[register_id(loc)]);
    if (value.references().empty()) {
        return;
    }

    _actual_node[value] = value;
    _register_content[register_id(loc)] = value;
    _reference_count[value] = value.references().size();
}

void Register_allocator::ensure_register(ir::Value value, Register reg) {
    ir::Value& actual_value = _actual_node[value];
    Operand loc = _allocation[actual_value];

    // If it is already in that register, then good.
    if (same_location(loc, reg)) return;

    // If the target register is already occupied, spill it.
    if (_register_content[register_id(reg)]) {
        spill_register(reg);
    }

    // Build a new node that represents the target location.
    auto copied_value = create_copy(actual_value);

    // Assign register.
    _allocation[copied_value] = reg;

    // Invalidate old content.
    if (loc.is_register()) {
        int reg = register_id(loc.as_register());
        ASSERT(_register_content[reg] == value);
        _register_content[reg] = {};
    }

    int reg_id = register_id(reg);
    ASSERT(!_register_content[reg_id]);
    _register_content[reg_id] = value;

    actual_value = copied_value;
}

void Register_allocator::decrease_reference(ir::Value value) {
    if (value.is_const()) {
        return;
    }

    auto refptr = _reference_count.find(value);
    ASSERT(refptr != _reference_count.end());

    // When reference count reaches zero the value could be wiped out.
    if (--refptr->second == 0) {

        // Remove from reference counting map.
        _reference_count.erase(refptr);

        // Get the actual value and remove from the map.
        auto nodeptr = _actual_node.find(value);
        ASSERT(nodeptr != _actual_node.end());
        auto actual_value = nodeptr->second;
        _actual_node.erase(nodeptr);

        const Operand& loc = _allocation[actual_value];

        if (loc.is_register()) {
            _register_content[register_id(loc.as_register())] = {};
        }

        auto memptr = _memory_node.find(value);
        if (memptr != _memory_node.end()) {
            auto mem_value = memptr->second;
            _memory_node.erase(memptr);
            _recent_freed_memory.push_back(_allocation[mem_value].as_memory());
        }

        return;
    }

    ASSERT(refptr->second > 0);
}

ir::Value Register_allocator::ensure_register_and_deref(ir::Node* node, size_t index, Register reg) {
    ir::Value value = node->operand(index);
    if (value.is_const()) {
        if (_register_content[register_id(reg)]) spill_register(reg);
        auto copied = create_copy(value);
        _allocation[copied] = reg;
        node->operand_set(index, copied);
        return copied;
    }

    ensure_register(value, reg);
    ir::Value actual_value = _actual_node[value];
    if (actual_value != value) node->operand_set(index, actual_value);
    decrease_reference(value);
    return actual_value;
}

ir::Value Register_allocator::get_actual_value(ir::Value value, bool allow_mem, bool allow_imm) {
    if (value.is_const()) {

        // If immediate is allowed and immediate fits, then no-op.
        if (allow_imm && util::is_int32(value.const_value())) {
            return value;
        }

        // Assign register.
        Register reg = alloc_register(value.type());
        auto copied_value = create_copy(value);
        _allocation[copied_value] = reg;
        return copied_value;
    }

    ir::Value& actual_value = _actual_node[value];
    const Operand& loc = _allocation[actual_value];
    if (allow_mem || loc.is_register()) return actual_value;

    // Assign register.
    Register reg = alloc_register(value.type());

    // Build a new node that represents the register. This must be placed *after* the alloc.
    auto copied_value = create_copy(actual_value);
    _allocation[copied_value] = reg;
    actual_value = copied_value;

    int reg_id = register_id(reg);
    ASSERT(!_register_content[reg_id]);
    _register_content[reg_id] = value;

    return copied_value;
}

ir::Value Register_allocator::get_actual_value_and_deref(
    ir::Node* node, size_t index, bool allow_mem, bool allow_imm
) {
    ir::Value value = node->operand(index);
    ir::Value actual_value = get_actual_value(value, allow_mem, allow_imm);
    if (actual_value != value) node->operand_set(index, actual_value);
    decrease_reference(value);
    return actual_value;
}

ir::Value Register_allocator::create_copy(ir::Value value) {
    // Build a new node that represents the register.
    auto copy_node = _graph.manage(new ir::Node(ir::Opcode::copy, {value.type()}, {value}));
    auto copied_value = copy_node->value(0);
    _nodelist->insert(_nodelist->begin() + _node_index, copy_node);
    _node_index++;
    return copied_value;
}

void Register_allocator::emit_compare(ir::Value value) {
    ASSERT(value.references().size() == 1);
    auto node = value.node();

    if (node->operand(0).is_const()) {
        auto op1 = get_actual_value_and_deref(node, 1, false, false);
        pin_value(op1);
        get_actual_value_and_deref(node, 0, true, true);
        unpin_value(op1);

    } else {
        auto op0 = get_actual_value_and_deref(node, 0, false, false);
        pin_value(op0);
        get_actual_value_and_deref(node, 1, true, true);
        unpin_value(op0);
    }
}

void Register_allocator::emit_address(ir::Value value) {
    ASSERT(value.references().size() == 1);
    auto node = value.node();

    // Assert that scale and displacement are both constant.
    auto base = node->operand(0);
    auto scale = node->operand(2);
    ASSERT(scale.is_const() && node->operand(3).is_const());

    // No base register.
    if (base.is_const() && base.const_value() == 0) {
        if (scale.const_value() != 0) {
            get_actual_value_and_deref(node, 1, false, false);
        }
        return;
    }

    base = get_actual_value_and_deref(node, 0, false, false);
    if (scale.const_value() != 0) {
        pin_value(base);
        get_actual_value_and_deref(node, 1, false, false);
        unpin_value(base);
    }
}

Operand Register_allocator::get_allocation(ir::Value value) {
#ifdef RELEASE
    return _allocation[value];
#else
    return _allocation.at(value);
#endif
}

void Register_allocator::allocate() {

    // Generate code for the block.
    for (auto block: _block_analysis.blocks()) {

        // Bind all PHI nodes first (except memory-allocated once, which will be done by copy).
        int phi_id = 0;
        for (auto ref: block->value(0).references()) {
            if (ref->opcode() == ir::Opcode::phi) {
                auto value = ref->value(0);

                // First try to fit into registers. The R11 is reserved for code generator to perform PHI reordering.
                constexpr int num_phi_can_fit = sizeof(volatile_register) / sizeof(volatile_register[0]) - 1;
                if (phi_id < num_phi_can_fit) {
                    auto reg = register_of_id(value.type(), volatile_register[phi_id]);
                    bind_register(value, reg);
                } else {

                    // If cannot fit into memory, put it into a stack slot.
                    auto mem = alloc_stack_slot(value.type());
                    _allocation[value] = mem;
                    ASSERT(!value.references().empty());
                    _actual_node[value] = value;
                    _memory_node[value] = value;
                    _reference_count[value] = value.references().size();
                }

                phi_id++;
            }
        }

        _nodelist = &_scheduler.get_mutable_node_list(block);
        for (_node_index = 0; _node_index < _nodelist->size(); _node_index++) {
            auto node = (*_nodelist)[_node_index];
            switch (node->opcode()) {
                // These nodes are handled by their users. Remove from the nodelist.
                case ir::Opcode::eq:
                case ir::Opcode::ne:
                case ir::Opcode::lt:
                case ir::Opcode::ge:
                case ir::Opcode::ltu:
                case ir::Opcode::geu:
                case Target_opcode::address:
                    _nodelist->erase(_nodelist->begin() + _node_index);
                    _node_index--;
                    break;
                case ir::Opcode::load_register: {
                    auto output = node->value(1);
                    Register reg = alloc_register(output.type());
                    bind_register(output, reg);
                    break;
                }
                case ir::Opcode::store_register: {
                    get_actual_value_and_deref(node, 1, false, true);
                    break;
                }
                case ir::Opcode::load_memory: {
                    auto output = node->value(1);
                    emit_address(node->operand(1));

                    Register reg = alloc_register(output.type());
                    bind_register(output, reg);
                    break;
                }
                case ir::Opcode::store_memory: {
                    auto value = get_actual_value_and_deref(node, 2, false, true);
                    pin_value(value);
                    emit_address(node->operand(1));
                    unpin_value(value);
                    break;
                }
                case Target_opcode::lea: {
                    auto output = node->value(0);
                    emit_address(node->operand(0));

                    Register reg = alloc_register(output.type());
                    bind_register(output, reg);
                    break;
                }
                case ir::Opcode::call: {
                    auto call_node = static_cast<ir::Call*>(node);

                    static constexpr Register reglist[] = {Register::rdi, Register::rsi, Register::rdx};

                    // Setup arguments
                    size_t op_index = call_node->need_context() ? 1 : 0;
                    for (size_t i = 1; i < node->operand_count(); i++) {
                        ASSERT(op_index < 3);
                        int reg_id = register_id(reglist[op_index++]);
                        ensure_register_and_deref(node, i, register_of_id(node->operand(i).type(), reg_id));
                    }

                    spill_all_registers();

                    // If the helper function returns a value, bind it.
                    if (node->value_count() == 2) {
                        auto output = node->value(1);
                        bind_register(output, register_of_id(output.type(), 0));
                    }
                    break;
                }
                case ir::Opcode::cast: {
                    auto output = node->value(0);
                    auto op = node->operand(0);

                    // Special handling for i1 upcast
                    if (op.type() == ir::Type::i1) {
                        emit_compare(op);

                        // Allocate and bind register.
                        Register reg = alloc_register(output.type());
                        bind_register(output, reg);
                        break;
                    }

                    // Decrease reference early, so if it is the last use we can eliminate one move.
                    ir::Value actual_op = get_actual_value_and_deref(node, 0, true, true);

                    // Allocate and bind register. Try to use loc0 if possible to eliminate move.
                    Register reg = alloc_register(output.type(), _allocation[actual_op]);
                    bind_register(output, reg);
                    break;
                }
                case ir::Opcode::mux: {
                    auto output = node->value(0);
                    auto op0 = node->operand(0);

                    auto op2 = get_actual_value_and_deref(node, 2, true, true);
                    pin_value(op2);
                    auto op1 = get_actual_value_and_deref(node, 1, true, false);
                    pin_value(op1);

                    unpin_value(op2);
                    Operand loc2 = op2.is_const() ? Register::none : _allocation[op2];
                    Register reg = alloc_register(output.type(), loc2);
                    bind_register(output, reg);
                    pin_register(reg);

                    emit_compare(op0);

                    unpin_register(reg);
                    unpin_value(op1);

                    // Note that this is actually placed after compare
                    if (op2.is_const() || !same_location(loc2, reg)) {
                        auto copied = create_copy(op2);
                        node->operand_set(2, copied);
                        _allocation[copied] = reg;
                    }

                    break;
                }
                case ir::Opcode::add:
                case ir::Opcode::sub:
                case ir::Opcode::i_xor:
                case ir::Opcode::i_and:
                case ir::Opcode::i_or: {
                    auto output = node->value(0);

                    auto op0 = get_actual_value_and_deref(node, 0, true, false);
                    pin_value(op0);
                    auto op1 = get_actual_value_and_deref(node, 1, true, true);
                    pin_value(op1);

                    unpin_value(op0);
                    Register reg = alloc_register(output.type(), _allocation[op0]);
                    bind_register(output, reg);
                    unpin_value(op1);

                    if (!same_location(reg, _allocation[op0])) {
                        auto copied = create_copy(op0);
                        node->operand_set(0, copied);
                        _allocation[copied] = reg;
                    }
                    break;
                }
                case ir::Opcode::shl:
                case ir::Opcode::shr:
                case ir::Opcode::sar: {
                    auto output = node->value(0);
                    auto const_shift = node->operand(1).is_const();

                    // For non-constant shifts the value must be in CL.
                    if (!const_shift) {
                        ensure_register_and_deref(node, 1, Register::cl);
                        pin_register(Register::cl);
                    }

                    auto op0 = get_actual_value_and_deref(node, 0, true, false);

                    Register reg = alloc_register(output.type(), _allocation[op0]);
                    bind_register(output, reg);

                    if (!same_location(reg, _allocation[op0])) {
                        auto copied = create_copy(op0);
                        node->operand_set(0, copied);
                        _allocation[copied] = reg;
                    }

                    if (!const_shift) unpin_register(Register::cl);
                    break;
                }
                case ir::Opcode::i_not:
                case ir::Opcode::neg: {
                    auto output = node->value(0);

                    auto op = get_actual_value_and_deref(node, 0, true, true);

                    Register reg = alloc_register(output.type(), _allocation[op]);
                    bind_register(output, reg);

                    if (!same_location(reg, _allocation[op])) {
                        auto copied = create_copy(op);
                        node->operand_set(0, copied);
                        _allocation[copied] = reg;
                    }
                    break;
                }

                case ir::Opcode::mul:
                case ir::Opcode::mulu: {
                    auto lo = node->value(0);
                    auto hi = node->value(1);

                    ASSERT(lo.type() == hi.type());
                    Register rax = register_of_id(lo.type(), 0);
                    Register rdx = register_of_id(hi.type(), 2);

                    // If one of the operand is already in rax, let it be the first operand.
                    if (_register_content[0] == node->operand(1)) {
                        node->operand_swap(0, 1);
                    }

                    ensure_register_and_deref(node, 0, rax);
                    pin_register(rax);
                    get_actual_value_and_deref(node, 1, true, false);
                    unpin_register(rax);

                    // Make sure useful values in rax and rdx are stored away safely.
                    if (_register_content[0]) spill_register(rax);
                    if (_register_content[2]) spill_register(rdx);

                    bind_register(lo, rax);
                    bind_register(hi, rdx);
                    break;
                }
                case ir::Opcode::div:
                case ir::Opcode::divu: {
                    auto quo = node->value(0);
                    auto rem = node->value(1);

                    ASSERT(quo.type() == rem.type());
                    Register rax = register_of_id(quo.type(), 0);
                    Register rdx = register_of_id(rem.type(), 2);

                    ensure_register_and_deref(node, 0, rax);
                    pin_register(rax);
                    pin_register(rdx);

                    // Make sure useful values in rax and rdx are stored away safely.
                    if (_register_content[0]) spill_register(rax);
                    if (_register_content[2]) spill_register(rdx);

                    get_actual_value_and_deref(node, 1, true, false);

                    unpin_register(rax);
                    unpin_register(rdx);

                    bind_register(quo, rax);
                    bind_register(rem, rdx);
                    break;
                }
                default: ASSERT(0);
            }

            // Add all elements in the recent free memory list to the free pool.
            if (!_recent_freed_memory.empty()) {
                _free_memory.insert(_free_memory.end(), _recent_freed_memory.begin(), _recent_freed_memory.end());
                _recent_freed_memory.clear();
            }
        }

        auto end = static_cast<ir::Paired*>(block)->mate();
        if (end->opcode() == ir::Opcode::i_if) {
            auto op = end->operand(1);
            if (!op.is_const()) {
                emit_compare(op);
            }
        }

        // Spill R11 if it is occupied, as it is reserved by code generator for reordering PHI nodes.
        if (_register_content[11]) spill_register(Register::r11);

        // Fill in actual nodes into PHI nodes.
        for (size_t i = 0; i < end->value_count(); i++) {
            auto target = ir::analysis::Block::get_target(end->value(i));
            size_t id = target->operand_find(end->value(i));

            for (auto ref: target->value(0).references()) {
                if (ref->opcode() != ir::Opcode::phi) continue;

                // Get the correct value to assign to the PHI node.
                auto value = ref->operand(id + 1);
                if (value.is_const()) continue;
                auto actual_value = _actual_node[value];
                if (actual_value != value) ref->operand_set(id + 1, actual_value);

                decrease_reference(value);
            }
        }

        // For anything cross basic block, life time management is harder. We cannot recycle its memory when we
        // encounter the last use for example, as it may still be used later. We workaround this by spilling everything
        // into memory, adding 1 to each variable live at the end of basic block to make it live across entire region.
        spill_all_registers();
        for (auto& pair: _reference_count) {
            pair.second++;
        }
    }

    // Finally align the stack to 16 bytes
    _stack_size = (_stack_size + 15) &~ 15;
}

}
