#include <list>

#include "ir/analysis.h"
#include "ir/visit.h"
#include "util/reverse_iterable.h"

namespace ir::analysis {

Node* Block::get_target(Value control) {
    size_t refcount = control.references().size();
    ASSERT(refcount == 1 || refcount == 2);
    bool skip_exit = refcount == 2;
    for (auto ref: control.references()) {
        if (skip_exit && ref->opcode() == Opcode::exit) continue;
        return ref;
    }
    ASSERT(0);
}

Value Block::get_tail_jmp_pc(Value control, uint16_t pc_regnum) {
    size_t refcount = control.references().size();
    if (refcount != 1) {
        // This jmp contains a keepalive edge, it therefore cannot be a tail jump.
        ASSERT(refcount == 2);
        return {};
    }

    auto target = *control.references().begin();

    // Not tail position
    if (target->opcode() != ir::Opcode::exit) return {};

    auto last_mem = control.node()->operand(0);
    if (last_mem.opcode() == ir::Opcode::store_register &&
        static_cast<ir::Register_access*>(last_mem.node())->regnum() == pc_regnum) {

        return last_mem.node()->operand(1);

    }

    return {};
}

void Block::enumerate_blocks() {
    std::vector<Node*> stack { *_graph.entry()->value(0).references().begin() };
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();

        // Already visited.
        if (std::find(_blocks.begin(), _blocks.end(), node) != _blocks.end()) continue;

        _blocks.push_back(node);
        auto end = static_cast<Paired*>(node)->mate();
        for (auto value: end->values()) {
            for (auto ref: value.references()) {
                if (ref->opcode() == Opcode::exit) continue;
                stack.push_back(ref);
            }
        }
    }
}

void Block::update_keepalive() {
    std::vector<Node*> stack;

    bool trim_existing_keepalive = false;
    for (auto operand: _graph.exit()->operands()) {

        // Skip keepalive edges.
        if (operand.references().size() == 2) {
            trim_existing_keepalive = true;
            continue;
        }

        ASSERT(operand.opcode() != Opcode::entry);
        stack.push_back(static_cast<Paired*>(operand.node())->mate());
    }

    // Remove existing keepalive edges if any.
    if (trim_existing_keepalive) {
        auto operands = _graph.exit()->operands();
        // Remove all keepalive edges.
        operands.erase(
            std::remove_if(operands.begin(), operands.end(), [](auto operand) {
                return operand.references().size() == 2;
            }),
            operands.end()
        );
        _graph.exit()->operands(std::move(operands));
    }

    // Create a clone of the list of all blocks. Use list here for better erase performance.
    std::list<Node*> unseen_blocks(_blocks.begin(), _blocks.end());

    while (true) {
        while (!stack.empty()) {
            auto node = stack.back();
            stack.pop_back();

            auto ptr = std::find(unseen_blocks.begin(), unseen_blocks.end(), node);

            // Already visited.
            if (ptr == unseen_blocks.end()) continue;

            // Remove from unseen blocks
            unseen_blocks.erase(ptr);

            for (auto operand: node->operands()) {
                if (operand.opcode() == Opcode::entry) continue;
                stack.push_back(static_cast<Paired*>(operand.node())->mate());
            }
        }

        // All nodes have been visited.
        if (unseen_blocks.empty()) {
            break;
        }

        // Keepalive edges need to be inserted. Note that as a heuristic, we prefer blocks later in unseen blocks.
        for (auto block: util::reverse_iterable(unseen_blocks)) {

            // Only insert keepalive edges with jmp node.
            auto end = static_cast<Paired*>(block)->mate();
            if (end->opcode() == Opcode::jmp) {
                _graph.exit()->operand_add(end->value(0));
                stack.push_back(block);
                break;
            }
        }

        ASSERT(!stack.empty());
    }
}

void Block::simplify_graph() {
    size_t block_count = _blocks.size();
    for (size_t i = 0; i < block_count; i++) {
        auto block = static_cast<ir::Paired*>(_blocks[i]);
        auto end = static_cast<ir::Paired*>(block->mate());

        // One predecessor and one successor. If the block is empty, then it can be folded away.
        if (block->operand_count() == 1 && end->opcode() == Opcode::jmp && end->value(0).references().size() == 1 &&
            end->operand(0) == block->value(0)) {

            // Link predecessor and successor together.
            replace_value(end->value(0), block->operand(0));

            // Remove current block as successor. This will maintain the constraint that control is used only once.
            block->operand_set(0, end->value(0));

            // Update constraints
            _blocks.erase(_blocks.begin() + i);
            i--;
            block_count--;
            continue;
        }

        // One predecessor, and current block is the only successor of previous block. Merge them in this case.
        if (block->operand_count() == 1 && block->operand(0).opcode() == Opcode::jmp &&
            block->operand(0).references().size() == 1) {

            auto prev_jmp = static_cast<ir::Paired*>(block->operand(0).node());
            auto prev_block = static_cast<ir::Paired*>(prev_jmp->mate());

            // Link two blocks together.
            replace_value(block->value(0), prev_jmp->operand(0));

            // Update mate information.
            end->mate(prev_block);
            prev_block->mate(end);

            // Update constraints
            _blocks.erase(_blocks.begin() + i);
            i--;
            block_count--;
            continue;
        }
    }
}

void Block::reorder(Dominance& dominance) {

    // A very simple algorithm that gives a heuristic penalty about a certain ordering of blocks.
    // We would like to reduce the number of jumps as much as possible. Therefore we assign a penalty of one if we need
    // to emit a jump. However if we use such heuristic, then there could be many plateaus, causing difficulties to
    // find minimum. Therefore we also add an additional penalty which measures the distance between two blocks.
    auto penalty = [](std::vector<ir::Node*>& blocks) {
        size_t penalty = 0;
        size_t block_count = blocks.size();
        for (size_t i = 0; i < block_count; i++) {
            auto block = blocks[i];
            auto end = static_cast<ir::Paired*>(block)->mate();
            for (auto value: end->values()) {
                auto target = get_target(value);

                // We do not consider jump to self.
                if (target == block) continue;

                size_t target_index;
                if (target->opcode() == ir::Opcode::exit) {
                    target_index = blocks.size();
                } else {
                    target_index = std::find(blocks.begin(), blocks.end(), target) - blocks.begin();
                }

                // Perfect position.
                if (target_index == i + 1) continue;

                // We add distance + 1 to the penalty.
                penalty += target_index > i ? target_index - i : i - target_index + 1;
            }
        }
        return penalty;
    };

    size_t current_penalty = penalty(_blocks);
    size_t block_count = _blocks.size();

    // For each iteration in the loop, we will look at i-th and (i+1)th block, so upper bound is block_count - 1.
    // The entry block must always be at 0, so the lower bound is 1.
    for (size_t i = 1; i < block_count - 1; i++) {

        // Do not move dominated blocks before their dominators. Doing so is harder for register allocation and code
        // generation. Note that doing this check alone is sufficient to maintain the desired constraint. As we
        // initially visit the blocks in DFS, the constraint is satisified by default, so the property will be kept as
        // long as our modifications here are not violating the constraint.
        if (dominance.immediate_dominator(_blocks[i + 1]) == _blocks[i]) continue;

        // Tentative change the order.
        std::swap(_blocks[i], _blocks[i + 1]);
        size_t new_penalty = penalty(_blocks);

        if (new_penalty < current_penalty) {

            // If the penalty improves, then acknowledge the swap.
            current_penalty = new_penalty;

            // After swapping the pair, we need to inspect (i-1)th and i-th (the (i+1)th block) block, as such a swap
            // may now be profitable. However if i is already 1, there are no earlier blocks so do not attempt to move
            // pointer back.
            if (i != 1) i -= 2;
        } else {

            // If the penalty does not improve, restore the old ordering and continue.
            std::swap(_blocks[i], _blocks[i + 1]);
        }
    }
}

}
