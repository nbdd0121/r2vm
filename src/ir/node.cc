#include <algorithm>

#include "ir/node.h"
#include "ir/visit.h"

namespace ir {

Node::Node(uint16_t opcode, Type_container&& type, Operand_container&& operands):
    _operands(std::move(operands)),  _type{std::move(type)}, _opcode{opcode}, _visited{0} {

    link();
    _references.resize(_type.size());
}

Node::~Node() {
    for (auto ref: _references) ASSERT(ref.size() == 0);
    unlink();
}

void Node::link() {
    for (auto operand: _operands) {
        operand.node()->_references[operand.index()].insert(this);
    }
}

void Node::unlink() {
    for (auto operand: _operands) {
        operand.node()->_references[operand.index()].remove(this);
    }
}

void Node::operands(Operand_container&& operands) {
    unlink();
    _operands = std::move(operands);
    link();
}

void Node::operand_set(size_t index, Value value) {
    ASSERT(index < _operands.size());

    auto& ptr = _operands[index];
    value.node()->_references[value.index()].insert(this);
    ptr.node()->_references[ptr.index()].remove(this);
    ptr = value;
}

size_t Node::operand_find(Value value) {
    auto ptr = std::find(_operands.begin(), _operands.end(), value);
    ASSERT(ptr != _operands.end());
    return ptr - _operands.begin();
}

void Node::operand_add(Value value) {
    _operands.push_back(value);
    value.node()->_references[value.index()].insert(this);
}

void Node::operand_delete(Value value) {
    size_t index = operand_find(value);
    _operands.erase(_operands.begin() + index);
    value.node()->_references[value.index()].remove(this);
}

Graph::Graph() {
    _entry = manage(new Node(Opcode::entry, {Type::control}, {}));
    _exit = manage(new Node(Opcode::exit, {Type::control}, {}));
}

Graph& Graph::operator=(Graph&& graph) {
    _heap.swap(graph._heap);
    _entry = graph._entry;
    _exit = graph._exit;
    return *this;
}

Graph::~Graph() {
    for (auto node: _heap) {
        node->_operands.clear();
        node->_references.clear();
        delete node;
    }
}

void Graph::garbage_collect() {

    // Mark all reachable nodes.
    visit_postorder(*this, [](Node*){});

    ASSERT(_entry->_visited);

    // Clear operands so that references are also cleared. This is necessary to maintain correctness of outgoing
    // references.
    size_t size = _heap.size();
    for (size_t i = 0; i < size; i++) {
        if (!_heap[i]->_visited) _heap[i]->operands({});
    }

    for (size_t i = 0; i < size; i++) {
        if (!_heap[i]->_visited) {

            // Reclaim memory.
            delete _heap[i];

            // Move last element to current.
            _heap[i--] = _heap[--size];
        }
    }

    _heap.resize(size);
}

Graph Graph::clone() const {
    Graph ret;
    std::unordered_map<Node*, Node*> mapping;

    // First create objects, but leave operands dummy.
    for (auto node: _heap) {
        Node* result;
        switch (node->opcode()) {
            case Opcode::entry:
                // This node is already managed.
                mapping[node] = ret._entry;
                continue;
            case Opcode::exit:
                // This node is already managed.
                mapping[node] = ret._exit;
                continue;
            case Opcode::constant:
                result = new Constant(node->_type[0], static_cast<Constant*>(node)->const_value());
                break;
            case Opcode::cast:
                result = new Cast(node->_type[0], static_cast<Cast*>(node)->sign_extend(), ret._entry->value(0));
                break;
            case Opcode::load_register:
            case Opcode::store_register:
                result = new Register_access(
                    static_cast<Register_access*>(node)->regnum(),
                    node->_opcode,
                    Node::Type_container(node->_type),
                    {}
                );
                break;
            case Opcode::block:
            case Opcode::jmp:
            case Opcode::i_if:
                result = new Paired(node->_opcode, Node::Type_container(node->_type), {});
                break;
            case Opcode::call:
                result = new Call(
                    static_cast<Call*>(node)->target(),
                    static_cast<Call*>(node)->need_context(),
                    Node::Type_container(node->_type),
                    {}
                );
                break;
            default:
                result = new Node(node->_opcode, Node::Type_container(node->_type), {});
                break;
        }
        mapping[node] = ret.manage(result);
    }

    // Fill objects
    for (auto node: _heap) {
        size_t op_count = node->_operands.size();

        Node::Operand_container operands(op_count);
        for (size_t i = 0; i < op_count; i++) {
            Value oldvalue = node->operand(i);
            operands[i] = { mapping[oldvalue.node()], oldvalue.index() };
        }
        mapping[node]->operands(std::move(operands));

        if (node->opcode() == Opcode::block || node->opcode() == Opcode::jmp || node->opcode() == Opcode::i_if) {
            static_cast<Paired*>(mapping[node])->mate(mapping[static_cast<Paired*>(node)->mate()]);
        }
    }

    return ret;
}

void Graph::inline_graph(Value control, Graph&& graph) {

    // We can only inline control to exit.
    ASSERT(control.references().size() == 1);
    ASSERT(*control.references().begin() == _exit);

    // Redirect control to the exit node in this graph.
    const auto& controls_to_exit = graph.exit()->operands();
    auto operands = _exit->operands();

    // We will erase the old control and insert new ones at the back. By doing so inlining will be breadth-first
    // instead of depth first.
    operands.erase(std::find(operands.begin(), operands.end(), control));
    operands.insert(operands.end(), controls_to_exit.begin(), controls_to_exit.end());
    graph.exit()->operands({});
    _exit->operands(std::move(operands));

    // Redirect the entry node.
    replace_value(graph.entry()->value(0), control);

    // Take control of everything except entry and exit.
    _heap.insert(_heap.end(), graph._heap.begin() + 2, graph._heap.end());
    graph._heap.resize(2);
}

}
