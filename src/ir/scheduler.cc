#include "ir/analysis.h"
#include "util/reverse_iterable.h"

namespace ir::analysis {

void Scheduler::schedule_node_early(Node* node) {

    // Provided node is already scheduled, we can mark all its dependencies as one step closer to ready.
    for (auto value: node->values()) {
        std::vector<Node*> ready;
        // TODO: By iterating through references and adding to the list, we are using the "random" order. This needs to be fixed.
        for (auto ref: value.references()) {
            switch (ref->opcode()) {
                // Impossible ones.
                case Opcode::entry:
                case Opcode::exit:
                case Opcode::block:
                case Opcode::constant:
                    ASSERT(0);
                // Ignore block-ending nodes.
                case Opcode::i_if:
                case Opcode::jmp:
                case Opcode::phi:
                    break;
                default: {
                    ssize_t remaining = --_unsatisified_input_count[ref];
                    ASSERT(remaining >= 0);

                    // For nodes ready for the first time, schedule it.
                    if (remaining == 0) {
                        _list->push_back(ref);
                        schedule_node_early(ref);
                    }
                    break;
                }
            }
        }
    }
}

void Scheduler::schedule_node_late(Node* node) {
    Node* block = nullptr;
    for (auto value: node->values()) {
        for (auto ref: value.references()) {
            if (ref->opcode() != Opcode::phi) {
                ASSERT(_late[ref]);
                block = _dominance.least_common_dominator(block, _late[ref]);
                continue;
            }

            for (size_t i = 1; i < ref->operand_count(); i++) {
                if (ref->operand(i) == value) {
                    block = _dominance.least_common_dominator(
                        block,
                        static_cast<Paired*>(ref->operand(0).node()->operand(i - 1).node())->mate()
                    );
                }
            }
        }
    }

    ASSERT(block);
    _late[node] = block;
    _nodelist[block].push_back(node);
}

void Scheduler::schedule_block(Node* block) {

    // Schedule all nodes that depends on control flow reaching the block.
    // This scheduling scheme will schedule nodes to the earliest possible location.
    std::vector<ir::Node*> nodes;
    _block = block;
    _list = &nodes;
    schedule_node_early(block);

    // Also schedule all PHI nodes attached to the block.
    for (auto ref: block->value(0).references()) {
        if (ref->opcode() == Opcode::phi) {
            schedule_node_early(ref);
        }
    }

    // Schedule rest of blocks in dominator tree order.
    for (auto next: _block_analysis.blocks()) {
        if (_dominance.immediate_dominator(next) == block) {
            schedule_block(next);
        }
    }

    // Now all nodes depending on nodes scheduled in previous schedule_node_early calls are scheduled. We can then try
    // to schedule the nodes to their latest possible location, based on the occurance of their references.

    // To ease scheduling we first schedule the last node in the block.
    _late[static_cast<Paired*>(block)->mate()] = block;
    _block = block;

    // Do it in reverse order to account for dependencies.
    for (auto node: util::reverse_iterable(nodes)) {
        schedule_node_late(node);
    }
}

void Scheduler::schedule() {

    // Initialize counter for every node to the number of operands.
    for (auto node: _graph.nodes()) {
        switch (node->opcode()) {
            // Skip special control flow nodes.
            case Opcode::entry:
            case Opcode::exit:
            case Opcode::block:
            case Opcode::i_if:
            case Opcode::jmp:
                break;
            // We need to handle constant specially as they are satisfied by default.
            case Opcode::constant:
            case Opcode::phi:
                break;
            default:
                ASSERT(node->operand_count());
                _unsatisified_input_count[node] = node->operand_count();
                break;
        }
    }

    auto first_block = Block::get_target(_graph.entry()->value(0));
    std::vector<Node*> entry_nodes;
    _block = first_block;
    _list = &entry_nodes;

    // Immediate mark constants as ready to schedule. Note that these constants are not associated with any blocks.
    // Ideally this is not needed, but if the code gets scheduled is not constant folded then it is pretty possible.
    // It is also possible that some target-specific nodes depend only on constants.
    // Note that _block and _list must be setup correctly before running this. We decide to associate these nodes with
    // the first block.
    for (auto node: _graph.nodes()) {
        if (node->opcode() == Opcode::constant) {
            schedule_node_early(node);
        }
    }

    // Kickstart the scheduling.
    schedule_block(first_block);

    for (auto node: util::reverse_iterable(entry_nodes)) {
        schedule_node_late(node);
    }

    // Clear up memory early.
    entry_nodes.clear();
    _unsatisified_input_count.clear();

#ifndef RELEASE

    // Verify all nodes are scheduled correctly.
    for (auto node: _graph.nodes()) {
        switch (node->opcode()) {
            // Skip special control flow nodes.
            case Opcode::entry:
            case Opcode::exit:
            case Opcode::block:
            case Opcode::constant:
            case Opcode::phi:
                ASSERT(_late.find(node) == _late.end());
                break;
            default:
                ASSERT(_late.find(node) != _late.end());
                break;
        }
    }

#endif

    // Nodes in nodelist are pushed in reverse order, so reverse them to normal order now.
    // TODO: After this we haven't reached the optimal local scheduling of instructions yet.
    for (auto& pair: _nodelist) {
        std::reverse(pair.second.begin(), pair.second.end());
    }
}

}
