#ifndef IR_BUILDER_H
#define IR_BUILDER_H

#include <tuple>

#include "ir/node.h"

namespace ir {

class Builder {
private:
    Graph& _graph;
public:
    Builder(Graph& graph): _graph{graph} {}

    Node* create(uint16_t opcode, Node::Type_container&& type, Node::Operand_container&& opr) {
        return _graph.manage(new Node(opcode, std::move(type), std::move(opr)));
    }

    Value control(uint16_t opcode, Node::Operand_container&& opr) {
        return create(opcode, {Type::control}, std::move(opr))->value(0);
    }

    Value block(Node::Operand_container&& operands) {
        return _graph.manage(new Paired(Opcode::block, {Type::memory}, std::move(operands)))->value(0);
    }

    Value jmp(Value operand) {
        return _graph.manage(new Paired(Opcode::jmp, {Type::control}, {operand}))->value(0);
    }

    Paired* i_if(Value memory, Value cond) {
        auto node = new Paired(Opcode::i_if, {Type::control, Type::control}, {memory, cond});
        _graph.manage(node);
        return node;
    }

    Value constant(Type type, uint64_t value) {
        return _graph.manage(new Constant(type, value))->value(0);
    }

    Value cast(Type type, bool sext, Value operand) {
        return _graph.manage(new Cast(type, sext, operand))->value(0);
    }

    std::tuple<Value, Value> load_register(Value dep, uint16_t regnum) {
        auto inst = _graph.manage(new Register_access(regnum, Opcode::load_register, {Type::memory, Type::i64}, {dep}));
        return {inst->value(0), inst->value(1)};
    }

    Value store_register(Value dep, uint16_t regnum, Value operand) {
        auto inst = _graph.manage(new Register_access(regnum, Opcode::store_register, {Type::memory}, {dep, operand}));
        return inst->value(0);
    }

    std::tuple<Value, Value> load_memory(Value dep, Type type, Value address) {
        auto inst = create(Opcode::load_memory, {Type::memory, type}, {dep, address});
        return {inst->value(0), inst->value(1)};
    }

    Value store_memory(Value dep, Value address, Value value) {
        return create(Opcode::store_memory, {Type::memory}, {dep, address, value})->value(0);
    }

    Value arithmetic(uint16_t opcode, Value left, Value right) {
        ASSERT(left.type() == right.type());
        return create(opcode, {left.type()}, {left, right})->value(0);
    }

    Value shift(uint16_t opcode, Value left, Value right) {
        ASSERT(right.type() == Type::i8);
        return create(opcode, {left.type()}, {left, right})->value(0);
    }

    Value compare(uint16_t opcode, Value left, Value right) {
        ASSERT(left.type() == right.type());
        return create(opcode, {Type::i1}, {left, right})->value(0);
    }

    Value mux(Value cond, Value left, Value right) {
        ASSERT(cond.type() == Type::i1 && left.type() == right.type());
        return create(Opcode::mux, {left.type()}, {cond, left, right})->value(0);
    }
};

}

#endif
