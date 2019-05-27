#include "ir/builder.h"
#include "ir/visit.h"
#include "util/int_size.h"
#include "x86/backend.h"

namespace x86::backend {

ir::Value Lowering::match_address(ir::Value value, bool required) {
    if (value.opcode() == Target_opcode::lea) {
        if (value.references().size() == 1) {
            return value.node()->operand(0);
        }

        // If the LEA node has more than one user, then we need to clone the address node, as address node can only
        // have single user.
        return _graph.manage(new ir::Node(
            Target_opcode::address, {ir::Type::i64},
            ir::Node::Operand_container(value.node()->operand(0).node()->operands())
        ))->value(0);
    }

    ir::Value base = value;
    ir::Value index;
    ir::Value scale;
    ir::Value displacement;

    // Handle cases in form (a+b), (a+const), ((a+b)+const)
    if (base.opcode() == ir::Opcode::add) {
        index = base.node()->operand(1);
        base = base.node()->operand(0);

        // If it turns out the right operand fits in immediate, do that.
        if (index.is_const() && util::is_int32(index.const_value())) {
            displacement = index;
            index = {};

            // Now index is gone. If base is still an addition, we can continue.
            if (base.opcode() == ir::Opcode::add) {
                index = base.node()->operand(1);
                base = base.node()->operand(0);
            }
        }
    }

    // Handle cases in form ((a+const)+b), (a+(b+const)).
    if (index && !displacement) {
        if (base.opcode() == ir::Opcode::add &&
            base.node()->operand(1).is_const() &&
            util::is_int32(base.node()->operand(1).const_value())) {

            displacement = base.node()->operand(1);
            base = base.node()->operand(0);

        } else if (index.opcode() == ir::Opcode::add &&
                   index.node()->operand(1).is_const() &&
                   util::is_int32(index.node()->operand(1).const_value())) {

            displacement = index.node()->operand(1);
            index = index.node()->operand(0);
        }
    }

    // Now we have completed displacement. We first normalize (a*x+b) to (b+a*x)
    if (base.opcode() == ir::Opcode::shl) {
        std::swap(base, index);
    }

    // Pattern detection on index * sscale.
    if (index && index.opcode() == ir::Opcode::shl &&
        index.node()->operand(1).is_const() &&
        index.node()->operand(1).const_value() <= 3) {

        scale = _graph.manage(new ir::Constant(ir::Type::i8, 1 << index.node()->operand(1).const_value()))->value(0);
        index = index.node()->operand(0);
    }

    if (!required) {
        // In optional case, we will not return an address node if it is not worth doing it.
        // A simple heuristic is used here: if more than three fields are filled, then we consider it worthwhile.
        int filled = 0;
        if (base) filled++;
        if (scale) filled++;
        if (index) filled++;
        if (displacement) filled++;
        if (filled < 3) return {};
    }

    if (!base) base = _graph.manage(new ir::Constant(ir::Type::i64, 0))->value(0);
    if (!scale) scale = _graph.manage(new ir::Constant(ir::Type::i8, index ? 1 : 0))->value(0);
    if (!index) index = _graph.manage(new ir::Constant(ir::Type::i64, 0))->value(0);
    if (!displacement) displacement = _graph.manage(new ir::Constant(ir::Type::i64, 0))->value(0);

    return _graph.manage(new ir::Node(
        Target_opcode::address, {ir::Type::i64}, {base, index, scale, displacement}
    ))->value(0);
}

void Lowering::run() {
    visit_postorder(_graph, [this](ir::Node* node) {
        switch (node->opcode()) {
            case ir::Opcode::load_memory: {
                auto addr = match_address(node->operand(1), true);
                node->operand_set(1, addr);
                break;
            }
            case ir::Opcode::store_memory: {
                auto addr = match_address(node->operand(1), true);
                if (addr) node->operand_set(1, addr);
                break;
            }
            case ir::Opcode::add: {
                auto output = node->value(0);
                auto addr = match_address(output, false);
                if (addr) {
                    replace_value(output, _graph.manage(new ir::Node(Target_opcode::lea, {output.type()}, {addr}))->value(0));
                }
                break;
            }
            case ir::Opcode::cast:
            case ir::Opcode::mux:
            case ir::Opcode::i_if: {
                size_t index = node->opcode() != ir::Opcode::i_if ? 0 : 1;
                auto op = node->operand(index);

                // It will be easier for the backend if non-constant node producing i1 has only single user.
                if (op.type() == ir::Type::i1 && op.opcode() != ir::Opcode::constant && op.references().size() != 1) {
                    node->operand_set(index, _graph.manage(new ir::Node(
                        op.opcode(), {ir::Type::i1},
                        ir::Node::Operand_container(op.node()->operands())
                    ))->value(0));
                }
                break;
            }
            default: break;
        }
    });
}

void Dot_printer::write_node_content(std::ostream& stream, ir::Node* node) {
    if (!ir::is_target_specific(node->opcode())) {
        return ir::pass::Dot_printer::write_node_content(stream, node);
    }

    switch (node->opcode()) {
        case Target_opcode::address: stream << "x86::address"; break;
        case Target_opcode::lea: stream << "x86::lea"; break;
        default: ASSERT(0);
    }
}

}
