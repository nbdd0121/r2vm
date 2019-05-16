#ifndef X86_BACKEND_H
#define X86_BACKEND_H

#include "ir/analysis.h"
#include "ir/pass.h"
#include "x86/encoder.h"
#include "x86/instruction.h"

#include <unordered_map>

namespace x86::backend {

namespace Target_opcode {

enum: uint16_t {
    // (Value base, Value index, Value scale, Value displacement) -> Value
    address = ir::Opcode::target_start,
    lea,
};

}

class Lowering {
private:
    ir::Graph& _graph;
    ir::Value match_address(ir::Value value, bool required);
public:
    Lowering(ir::Graph& graph): _graph{graph} {};
    void run();
};

class Dot_printer: public ir::pass::Dot_printer {
protected:
    virtual void write_node_content(std::ostream& stream, ir::Node* node) override;
};

class Register_allocator {
public:
    ir::Graph& _graph;
    ir::analysis::Block& _block_analysis;
    ir::analysis::Scheduler& _scheduler;
    std::unordered_map<ir::Value, Operand> _allocation;

    std::vector<ir::Node*>* _nodelist;
    size_t _node_index;

    int _stack_size = 0;
    std::unordered_map<ir::Value, int> _reference_count;

    // The current node that shares the same value of the given node.
    std::unordered_map<ir::Value, ir::Value> _actual_node;

    // Spilled copy of the node. This is used to avoid re-storing into memory if spilled again.
    std::unordered_map<ir::Value, ir::Value> _memory_node;

    // Freed memory locations that hasn't been placed into the pool yet. We will only place them into the free memory
    // pool after a whole instruction to avoid having to deal with pinning memory locations.
    std::vector<x86::Memory> _recent_freed_memory;

    // Free stack slots pool.
    std::vector<x86::Memory> _free_memory;

    // Tracks what is stored in each register. Note that not all registers are used, but for easiness we still make its
    // size 16.
    std::array<ir::Value, 16> _register_content {};

    // Tracks whether a register can be spilled, i.e. not pinned.
    std::array<bool, 16> _pinned {};

public:
    Register_allocator(ir::Graph& graph, ir::analysis::Block& block_analysis, ir::analysis::Scheduler& scheduler):
        _graph{graph}, _block_analysis{block_analysis}, _scheduler{scheduler} {

    }

private:
    /* Internal methods handling register allocation and spilling. */

    // Allocate a register without spilling. This is the fundamental operation for register allocation.
    Register alloc_register_no_spill(ir::Type type, Register hint);

    // Allocate a register, possibly spill other registers to memory. The hint will be respected only if it is a
    // register. The size of the hint will be ignored.
    Register alloc_register(ir::Type type, Register hint = Register::none);
    Register alloc_register(ir::Type type, const Operand& hint);

    ir::Value create_copy(ir::Value value);

    // Spill a specified register to memory. Size of the register will be ignored.
    void spill_register(Register reg);
    Memory alloc_stack_slot(ir::Type type);

    // Spill all volatile registers to memory.
    void spill_all_registers();

    // Pin and unpin registers so they cannot be spilled. Size of the register will be ignored.
    void pin_register(Register reg);
    void unpin_register(Register reg);
    void pin_value(ir::Value value);
    void unpin_value(ir::Value value);

    // Bind a register to a node, and set up reference count.
    void bind_register(ir::Value value, Register reg);
    void ensure_register(ir::Value value, Register reg);
    void decrease_reference(ir::Value value);

    ir::Value ensure_register_and_deref(ir::Node* node, size_t operand, Register reg);

    ir::Value get_actual_value(ir::Value value, bool allow_mem, bool allow_imm);
    ir::Value get_actual_value_and_deref(ir::Node* node, size_t operand, bool allow_mem, bool allow_imm);

    void emit_compare(ir::Value value);
    void emit_address(ir::Value);

public:
    int get_stack_size() { return _stack_size; }
    Operand get_allocation(ir::Value value);
    void allocate();
};

class Code_generator {
private:
    ir::Graph& _graph;
    ir::analysis::Block& _block_analysis;
    ir::analysis::Scheduler& _scheduler;
    backend::Register_allocator& _regalloc;
    x86::Encoder _encoder;

public:
    Code_generator(
        util::Code_buffer& buffer,
        ir::Graph& graph,
        ir::analysis::Block& block_analysis,
        ir::analysis::Scheduler& scheduler,
        Register_allocator& regalloc
    ): _graph{graph}, _block_analysis{block_analysis}, _scheduler{scheduler}, _regalloc{regalloc}, _encoder{buffer} {}

    void emit(const Instruction& inst);
    void emit_move(ir::Type type, const Operand& dst, const Operand& src);

    Operand get_allocation(ir::Value value);

    bool need_phi(ir::Value control);
    void emit_phi(ir::Value control);

    void emit_binary(ir::Node* node, Opcode opcode);
    void emit_unary(ir::Node* node, Opcode opcode);
    void emit_mul(ir::Node* node, Opcode opcode);
    void emit_div(ir::Node* node, Opcode opcode);
    Condition_code emit_compare(ir::Value value);
    Memory emit_address(ir::Type type, ir::Value value);

    void visit(ir::Node* node);

public:
    void run();
};

}

#endif
