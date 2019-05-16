#include "ir/builder.h"
#include "ir/node.h"
#include "ir/pass.h"
#include "ir/visit.h"
#include "util/bit_op.h"
#include "util/int128.h"

namespace ir::pass {

size_t Local_value_numbering::Hash::operator ()(Node* node) const noexcept {
    size_t hash = static_cast<uint8_t>(node->opcode());

    ASSERT(is_pure_opcode(node->opcode()));

    for (auto value: node->values()) {
        hash ^= static_cast<uint8_t>(value.type());
    }

    for (auto operand: node->operands()) {
        hash ^= reinterpret_cast<uintptr_t>(operand.node()) ^ operand.index();
    }

    switch (node->opcode()) {
        case Opcode::constant:
            hash ^= static_cast<Constant*>(node)->const_value();
            break;
        case Opcode::cast:
            hash ^= static_cast<Cast*>(node)->sign_extend();
            break;
        default: break;
    }

    return hash;
}

bool Local_value_numbering::Equal_to::operator ()(Node* a, Node* b) const noexcept {
    if (a->opcode() != b->opcode()) return false;

    ASSERT(is_pure_opcode(a->opcode()));

    if (a->value_count() != b->value_count()) return false;

    for (size_t i = 0; i < a->value_count(); i++) {
        if (a->value(i).type() != b->value(i).type()) return false;
    }

    size_t operand_count = a->operand_count();
    if (operand_count != b->operand_count()) return false;

    for (size_t i = 0; i < operand_count; i++) {
        if (a->operand(i) != b->operand(i)) return false;
    }

    switch (a->opcode()) {
        case Opcode::constant:
            if (static_cast<Constant*>(a)->const_value() != static_cast<Constant*>(b)->const_value())
                return false;
            break;
        case Opcode::cast:
            if (static_cast<Cast*>(a)->sign_extend() != static_cast<Cast*>(b)->sign_extend()) return false;
            break;
        default: break;
    }

    return true;
}

// Sign-extend value of type to i64
uint64_t Local_value_numbering::sign_extend(Type type, uint64_t value) {
    switch (type) {
        case Type::i1: return value ? 1 : 0;
        case Type::i8: return static_cast<int64_t>(static_cast<int8_t>(value));
        case Type::i16: return static_cast<int64_t>(static_cast<int16_t>(value));
        case Type::i32: return static_cast<int64_t>(static_cast<int32_t>(value));
        case Type::i64: return value;
        default: ASSERT(0);
    }
}

// Zero-extend value of type to i64
uint64_t Local_value_numbering::zero_extend(Type type, uint64_t value) {
    switch (type) {
        case Type::i1: return value ? 1 : 0;
        case Type::i8: return static_cast<uint8_t>(value);
        case Type::i16: return static_cast<uint16_t>(value);
        case Type::i32: return static_cast<uint32_t>(value);
        case Type::i64: return value;
        default: ASSERT(0);
    }
}

// Evaluate cast.
uint64_t Local_value_numbering::cast(Type type, Type oldtype, bool sext, uint64_t value) {
    // For signed upcast, it can be represented as sign-extend to 64-bit and downcast.
    // For unsigned upcast, it can be represented as zero-extend to 64-bit and downcast.
    // For downcast, sign-extending or zero-extending makes no difference.
    // We choose to express all values using 64-bit number, sign-extended, as this representation allows comparision
    // without knowing the type of the value.
    if (sext) {
        return sign_extend(type, value);
    } else {
        return sign_extend(type, zero_extend(oldtype, value));
    }
}

// Evaluate binary operations.
uint64_t Local_value_numbering::binary(Type type, uint16_t opcode, uint64_t l, uint64_t r) {
    switch (opcode) {
        case Opcode::add: return sign_extend(type, l + r);
        case Opcode::sub: return sign_extend(type, l - r);
        // Bitwise operations will preserve the sign-extension.
        case Opcode::i_xor: return l ^ r;
        case Opcode::i_or: return l | r;
        case Opcode::i_and: return l & r;
        case Opcode::shl: return sign_extend(type, l << (r & (get_type_size(type) - 1)));
        // To maintain correctness, convert to zero-extension, perform operation, then convert back.
        case Opcode::shr: return sign_extend(type, zero_extend(type, l) >> (r & (get_type_size(type) - 1)));
        case Opcode::sar: return static_cast<int64_t>(l) >> (r & (get_type_size(type) - 1));
        case Opcode::eq: return l == r;
        case Opcode::ne: return l != r;
        // All comparisions will work with sign-extension (which is the reason sign-extension is chosen).
        case Opcode::lt: return static_cast<int64_t>(l) < static_cast<int64_t>(r);
        case Opcode::ge: return static_cast<int64_t>(l) >= static_cast<int64_t>(r);
        case Opcode::ltu: return l < r;
        case Opcode::geu: return l >= r;
        default: ASSERT(0);
    }
}

Value Local_value_numbering::new_constant(Type type, uint64_t const_value) {

    // Create a new constant node.
    Node* new_node = _graph.manage(new Constant(type, const_value));

    auto pair = _set.insert(new_node);
    return pair.second ? new_node->value(0) : (*pair.first)->value(0);
}

// Helper function that replaces current value with a constant value. It will keep type intact.
void Local_value_numbering::replace_with_constant(Value value, uint64_t const_value) {
    if (value.references().empty()) return;
    replace_value(value, new_constant(value.type(), const_value));
}

void Local_value_numbering::lvn(Node* node) {
    // perform the actual local value numbering.
    // Try insert into the set. If insertion succeeded, then this is a new node, so return.
    auto pair = _set.insert(node);
    if (pair.second) return;

    // Otherwise replace with the existing one.
    if (node != *pair.first) {
        auto target = *pair.first;
        for (auto v: node->values()) {
            replace_value(v, target->value(v.index()));
        }
    }
}

void Local_value_numbering::process(Node* node) {
    auto opcode = node->opcode();

    if (!is_pure_opcode(opcode)) return;

    if (opcode == Opcode::constant) {
        return lvn(node);
    }

    if (opcode == Opcode::cast) {
        // Folding cast node.
        auto output = node->value(0);
        auto x = node->operand(0);
        bool sext = static_cast<Cast*>(node)->sign_extend();

        // If the operand is constant, then perform constant folding.
        if (x.is_const()) {
            return replace_with_constant(output, cast(output.type(), x.type(), sext, x.const_value()));
        }

        // Two casts can be possibly folded.
        if (x.opcode() == Opcode::cast) {
            auto y = x.node()->operand(0);
            bool x_sext = static_cast<Cast*>(x.node())->sign_extend();

            size_t ysize = get_type_size(y.type());
            size_t size = get_type_size(output.type());

            // A down-cast followed by an up-cast cannot be folded.
            size_t xsize = get_type_size(x.type());
            if (ysize > xsize && xsize < size) return lvn(node);

            // An up-cast followed by an up-cast cannot be folded if sext does not match.
            if (ysize < xsize && xsize < size && x_sext != sext) return lvn(node);

            // If the size is same, then eliminate.
            if (ysize == size) {
                return replace_value(output, y);
            }

            // This can either be up-cast followed by up-cast, up-cast followed by down-cast.
            // As the result is an up-cast, we need to select the correct sext.
            if (ysize < size) {
                static_cast<Cast*>(node)->sign_extend(x_sext);
            }

            node->operand_set(0, y);
        }

        return lvn(node);
    }

    if (is_binary_opcode(opcode)) {
        // Folding binary operation node.
        auto output = node->value(0);
        auto x = node->operand(0);
        auto y = node->operand(1);

        // If both operands are constant, then perform constant folding.
        if (x.is_const() && y.is_const()) {
            return replace_with_constant(output, binary(x.type(), node->opcode(), x.const_value(), y.const_value()));
        }

        // Canonicalization, for commutative opcodes move constant to the right.
        // TODO: For non-abelian comparisions, we can also move constant to the right by performing transformations on
        // immediate.
        if (x.is_const()) {
            if (is_commutative_opcode(opcode)) {
                node->operand_swap(0, 1);
                std::swap(x, y);
            } else {
                if (x.const_value() == 0) {
                    // Arithmetic identity folding for non-abelian operations.
                    switch (opcode) {
                        case Opcode::sub:
                            node->opcode(Opcode::neg);
                            node->operands({y});
                            return lvn(node);
                        case Opcode::shl:
                        case Opcode::shr:
                        case Opcode::sar:
                            return replace_with_constant(output, 0);
                        // 0 < unsigned is identical to unsigned != 0
                        case Opcode::ltu:
                            node->opcode(Opcode::ne);
                            node->operand_swap(0, 1);
                            return lvn(node);
                        // 0 >= unsigned is identical to unsigned == 0
                        case Opcode::geu:
                            node->opcode(Opcode::eq);
                            node->operand_swap(0, 1);
                            return lvn(node);
                        default: break;
                    }
                }
            }
        }

        // Arithmetic identity folding.
        // TODO: Other arithmetic identity worth considering:
        // x + x == x << 1
        // x >> 63 == x < 0
        if (y.is_const()) {
            if (y.const_value() == 0) {
                switch (opcode) {
                    // For these node x @ 0 == x
                    case Opcode::add:
                    case Opcode::sub:
                    case Opcode::i_xor:
                    case Opcode::i_or:
                    case Opcode::shl:
                    case Opcode::shr:
                    case Opcode::sar:
                        return replace_value(output, x);
                    // For these node x @ 0 == 0
                    case Opcode::i_and:
                    case Opcode::ltu:
                        return replace_with_constant(output, 0);
                    // unsigned >= 0 is tautology
                    case Opcode::geu:
                        return replace_with_constant(output, 1);
                    default: break;
                }
            } else if (y.const_value() == static_cast<uint64_t>(-1)) {
                switch (opcode) {
                    case Opcode::i_xor:
                        node->opcode(Opcode::i_not);
                        node->operands({x});
                        return lvn(node);
                    case Opcode::i_and:
                        return replace_value(output, x);
                    case Opcode::i_or:
                        return replace_with_constant(output, -1);
                    default: break;
                }
            }
        }

        if (x == y) {
            switch (opcode) {
                case Opcode::sub:
                case Opcode::i_xor:
                case Opcode::ne:
                case Opcode::lt:
                case Opcode::ltu:
                    return replace_with_constant(output, 0);
                case Opcode::i_or:
                case Opcode::i_and:
                    return replace_value(output, x);
                case Opcode::eq:
                case Opcode::ge:
                case Opcode::geu:
                    return replace_with_constant(output, 1);
                default: break;
            }
        }

        // More folding for add
        if (opcode == Opcode::add && y.is_const() && x.opcode() == Opcode::add && x.node()->operand(1).is_const()) {
            y = new_constant(
                output.type(), sign_extend(output.type(), y.const_value() + x.node()->operand(1).const_value())
            );
            x = x.node()->operand(0);
            node->operand_set(0, x);
            node->operand_set(1, y);
        }

        // Translate ands to casts
        if (opcode == Opcode::i_and && y.is_const()) {

            if (y.const_value() == 0xFF) {
                // For i8, 0xFF is sign-extended to -1.
                ASSERT(output.type() != Type::i8);
                auto downcast = _graph.manage(new Cast(Type::i8, false, x));
                auto upcast = _graph.manage(new Cast(output.type(), false, downcast->value(0)));
                replace_value(output, upcast->value(0));
                process(downcast);
                return process(upcast);
            } else if (y.const_value() == 0xFFFF) {
                ASSERT(output.type() != Type::i8 && output.type() != Type::i16);
                auto downcast = _graph.manage(new Cast(Type::i16, false, x));
                auto upcast = _graph.manage(new Cast(output.type(), false, downcast->value(0)));
                replace_value(output, upcast->value(0));
                process(downcast);
                return process(upcast);
            } else if (y.const_value() == 0xFFFFFFFF) {
                ASSERT(output.type() == Type::i64);
                auto downcast = _graph.manage(new Cast(Type::i32, false, x));
                auto upcast = _graph.manage(new Cast(output.type(), false, downcast->value(0)));
                replace_value(output, upcast->value(0));
                process(downcast);
                return process(upcast);
            }
        }

        // Translate shls followed by shrs to casts
        if ((opcode == Opcode::shr || opcode == Opcode::sar) && y.is_const() && x.opcode() == Opcode::shl &&
            x.node()->operand(1).is_const() && y.const_value() == x.node()->operand(1).const_value()) {

            auto op = x.node()->operand(0);
            bool sext = opcode == Opcode::sar;
            auto width = get_type_size(output.type()) - y.const_value();
            if (width == 8) {
                auto downcast = _graph.manage(new Cast(Type::i8, false, op));
                auto upcast = _graph.manage(new Cast(output.type(), sext, downcast->value(0)));
                replace_value(output, upcast->value(0));
                process(downcast);
                return process(upcast);
            } else if (width == 16) {
                auto downcast = _graph.manage(new Cast(Type::i16, false, op));
                auto upcast = _graph.manage(new Cast(output.type(), sext, downcast->value(0)));
                replace_value(output, upcast->value(0));
                process(downcast);
                return process(upcast);
            } else if (width == 32) {
                auto downcast = _graph.manage(new Cast(Type::i32, false, op));
                auto upcast = _graph.manage(new Cast(output.type(), sext, downcast->value(0)));
                replace_value(output, upcast->value(0));
                process(downcast);
                return process(upcast);
            }
        }

        return lvn(node);
    }

    if (opcode == Opcode::mux) {
        auto output = node->value(0);
        auto x = node->operand(0);
        auto y = node->operand(1);
        auto z = node->operand(2);

        if (x.is_const()) {
            return replace_value(output, x.const_value() ? y : z);
        }

        return lvn(node);
    }

    if (opcode == Opcode::i_not) {
        auto output = node->value(0);
        auto input = node->operand(0);

        if (input.is_const()) {
            return replace_with_constant(output, sign_extend(output.type(), ~input.const_value()));
        }

        return lvn(node);
    }

    if (opcode == Opcode::neg) {
        auto output = node->value(0);
        auto input = node->operand(0);

        if (input.is_const()) {
            return replace_with_constant(output, sign_extend(output.type(), -input.const_value()));
        }

        return lvn(node);
    }

    if (opcode == Opcode::mul || opcode == Opcode::mulu) {
        auto lo = node->value(0);
        auto hi = node->value(1);
        auto op0 = node->operand(0);
        auto op1 = node->operand(1);

        // Fold if both operands are constant.
        if (op0.is_const() && op1.is_const()) {
            if (lo.type() == Type::i64) {
                if (opcode == Opcode::mul) {
                    util::int128_t a = static_cast<int64_t>(op0.const_value());
                    util::int128_t b = static_cast<int64_t>(op1.const_value());
                    replace_with_constant(hi, (a * b) >> 64);
                } else {
                    util::uint128_t a = op0.const_value();
                    util::uint128_t b = op1.const_value();
                    replace_with_constant(hi, (a * b) >> 64);
                }
                replace_with_constant(lo, op0.const_value() * op1.const_value());
            } else {
                uint64_t a = opcode == Opcode::mul ? op0.const_value() : zero_extend(op0.type(), op0.const_value());
                uint64_t b = opcode == Opcode::mul ? op1.const_value() : zero_extend(op1.type(), op1.const_value());
                replace_with_constant(lo, sign_extend(lo.type(), a * b));
                replace_with_constant(hi, sign_extend(lo.type(), (a * b) >> get_type_size(lo.type())));
            }
            return;
        }

        // Normalize constant to the right.
        if (op0.is_const()) {
            std::swap(op0, op1);
            node->operand_swap(0, 1);
        }

        if (op1.is_const()) {
            uint64_t v = op1.const_value();
            if (v == 0) {
                replace_with_constant(lo, 0);
                replace_with_constant(hi, 0);
                return;
            } else if (v == 1) {
                replace_value(lo, op0);
                replace_with_constant(hi, 0);
                return;
            }

            // Reduce multiplication to shifts for power of two.
            int logv = util::log2_floor(v);
            if ((1ULL << logv) == v) {
                Builder builder { _graph };
                replace_value(lo, builder.shift(Opcode::shl, op0, builder.constant(Type::i8, logv)));
                if (!hi.references().empty()) {
                    replace_value(hi, builder.shift(
                        opcode == Opcode::mul ? Opcode::sar : Opcode::shr,
                        op0,
                        builder.constant(Type::i8, get_type_size(hi.type()) - logv)
                    ));
                }
                return;
            }
        }

        // If the same node exists in lvn table already, use it.
        auto iter = _set.find(node);
        if (iter != _set.end()) {
            replace_value(lo, (*iter)->value(0));
            replace_value(hi, (*iter)->value(1));
            return;
        }

        // Multiplication node is a little bit more complex than other nodes: if signedness differ, lo parts are still
        // the same, but the hi parts will be different. Therefore we need to also query the node with different
        // signedness. To do so, we pretend to be of opposite sign to lookup in lvn table, the restore the signedness.
        node->opcode(opcode == Opcode::mul ? Opcode::mulu : Opcode::mul);
        iter = _set.find(node);
        node->opcode(opcode);
        if (iter != _set.end()) {
            auto other_node = *iter;

            // Only lo output of the other node is used, then redirect to this node.
            if (other_node->value(1).references().empty()) {
                _set.erase(iter);
                replace_value(other_node->value(0), lo);
                _set.insert(node);
                return;
            }

            // Only lo output of this node is used, then redirect to the other node.
            if (hi.references().empty()) {
                replace_value(lo, other_node->value(0));
                return;
            }

            // Hi parts of both nodes are used. We will consider the node as fresh in this case.
        }

        _set.insert(node);
        return;
    }

    if (opcode == Opcode::div || opcode == Opcode::divu) {
        auto quo = node->value(0);
        auto rem = node->value(1);
        auto op0 = node->operand(0);
        auto op1 = node->operand(1);

        // Fold if both operands are constant.
        if (op0.is_const() && op1.is_const()) {
            uint64_t v0 = op0.const_value();
            uint64_t v1 = op1.const_value();

            // RISC-V specific exceptional case handling.
            if (v1 == 0) {
                replace_with_constant(quo, -1);
                replace_with_constant(rem, v0);
                return;
            }

            if (opcode == Opcode::div) {
                int64_t type_min = quo.type() == Type::i64 ?
                        std::numeric_limits<int64_t>::min() :
                        std::numeric_limits<int32_t>::min();

                if (v0 == static_cast<uint64_t>(type_min) && v1 == static_cast<uint64_t>(-1)) {
                    replace_with_constant(quo, type_min);
                    replace_with_constant(rem, 0);
                    return;
                }
            }

            if (opcode == Opcode::divu) {
                if (quo.type() == Type::i64) {
                    replace_with_constant(quo, v0 / v1);
                    replace_with_constant(rem, v0 % v1);
                } else {
                    replace_with_constant(quo, sign_extend(
                        ir::Type::i32, static_cast<uint32_t>(v0) / static_cast<uint32_t>(v1)
                    ));
                    replace_with_constant(rem, sign_extend(
                        ir::Type::i32, static_cast<uint32_t>(v0) % static_cast<uint32_t>(v1)
                    ));
                }
            } else {
                if (quo.type() == Type::i64) {
                    replace_with_constant(quo, static_cast<int64_t>(v0) / static_cast<int64_t>(v1));
                    replace_with_constant(rem, static_cast<int64_t>(v0) % static_cast<int64_t>(v1));
                } else {
                    replace_with_constant(quo, static_cast<int32_t>(v0) / static_cast<int32_t>(v1));
                    replace_with_constant(rem, static_cast<int32_t>(v0) % static_cast<int32_t>(v1));
                }
            }

            return;
        }

        lvn(node);
        return;
    }

    ASSERT(0);
}

void Local_value_numbering::run() {
    visit_postorder(_graph, [this](Node* node) { process(node); });
}

}
