#ifndef IR_VISIT_H
#define IR_VISIT_H

#include "ir/node.h"

namespace ir {

namespace internal {

void clear_visited_flags(Graph& graph);

template<typename F>
void visit_postorder_actual(Node* node, F func) {
    if (node->_visited) return;
    node->_visited = 1;

    // Visit all dependencies
    for (auto operand: node->operands()) visit_postorder_actual(operand.node(), func);
    func(node);
}

}

void replace_value(Value oldvalue, Value newvalue);

// Visit the dependence graph in postorder.
template<typename F>
void visit_postorder(Graph& graph, Node* node, F func) {
    internal::clear_visited_flags(graph);
    internal::visit_postorder_actual(node, func);
}

template<typename F>
void visit_postorder(Graph& graph, F func) {
    visit_postorder(graph, graph.exit(), func);
}

// Visit the dependence graph in postorder, do not cross block boundary, and only care about memory nodes.
template<typename F>
void visit_local_memops_postorder(Node* node, F func) {

    // Memory nodes are chained as a list (not a DAG). Therefore we don't have to track whether a node is visited.
    for (auto op: node->operands()) {
        if (op.type() == Type::memory) {
            visit_local_memops_postorder(op.node(), func);
        }
    }

    switch (node->opcode()) {
        case Opcode::load_register:
        case Opcode::store_register:
        case Opcode::load_memory:
        case Opcode::store_memory:
        case Opcode::call:
            func(node);
            break;
    }
}

// Visit the dependence graph in preorder, do not cross block boundary, and only care about memory nodes.
template<typename F>
void visit_local_memops_preorder(Node* node, F func) {

    // Memory nodes are chained as a list (not a DAG). Therefore we don't have to track whether a node is visited.
    switch (node->opcode()) {
        case Opcode::load_register:
        case Opcode::store_register:
        case Opcode::load_memory:
        case Opcode::store_memory:
        case Opcode::call:
            func(node);
            break;
    }

    for (auto op: node->operands()) {
        if (op.type() == Type::memory) {
            visit_local_memops_preorder(op.node(), func);
        }
    }
}

}

#endif
