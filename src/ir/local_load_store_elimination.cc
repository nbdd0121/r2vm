#include "emu/state.h"

#include "ir/analysis.h"
#include "ir/node.h"
#include "ir/visit.h"

namespace ir::analysis {

void Local_load_store_elimination::run() {

    // This is a simplified version of local store elimination which only remove redundancies within basic block.
    // This is already very useful in simple cases though.

    std::vector<std::vector<Value>> value_stack(_regcount, std::vector<Value>{{}});
    for (auto block: _block_analysis.blocks()) {
        auto end = static_cast<Paired*>(block)->mate();

        visit_local_memops_postorder(end, [&](Node* node) {
            if (node->opcode() == Opcode::load_register) {
                uint16_t regnum = static_cast<Register_access*>(node)->regnum();
                auto value = value_stack[regnum].back();
                if (value) {
                    replace_value(node->value(0), node->operand(0));
                    replace_value(node->value(1), value);
                    return;
                }

                value_stack[regnum].push_back(node->value(1));

            } else if (node->opcode() == Opcode::store_register) {
                uint16_t regnum = static_cast<Register_access*>(node)->regnum();
                value_stack[regnum].push_back(node->operand(1));

            } else if (node->opcode() == Opcode::call && static_cast<Call*>(node)->need_context()) {
                for (uint16_t regnum = 0; regnum < _regcount; regnum++) {
                    value_stack[regnum].push_back({});
                }
            }
        });

        // Reset value stacks to invalid.
        for (auto& stack: value_stack) stack.resize(1);

        visit_local_memops_preorder(end, [&](Node* node) {
            if (node->opcode() == Opcode::load_register) {
                uint16_t regnum = static_cast<Register_access*>(node)->regnum();
                value_stack[regnum].push_back({});

            } else if (node->opcode() == Opcode::store_register) {
                uint16_t regnum = static_cast<Register_access*>(node)->regnum();

                if (value_stack[regnum].back()) {
                    replace_value(node->value(0), node->operand(0));
                    return;
                }

                value_stack[regnum].push_back(node->value(0));

            } else if (emu::state::strict_exception ||
                        (node->opcode() == Opcode::call && static_cast<Call*>(node)->need_context())) {

                for (uint16_t regnum = 0; regnum < _regcount; regnum++) {
                    value_stack[regnum].push_back({});
                }
            }
        });

        // Reset value stacks to invalid.
        for (auto& stack: value_stack) stack.resize(1);
    }
}

}
