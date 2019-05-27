#include "ir/node.h"
#include "ir/pass.h"
#include "ir/visit.h"

#include "util/assert.h"
#include "util/format.h"

namespace ir::pass {

const char* Dot_printer::opcode_name(uint16_t opcode) {
    switch (opcode) {
#define CASE(name) case Opcode::name: return #name;
        CASE(entry)
        CASE(exit)
        CASE(block)
        case Opcode::i_if: return "if";
        CASE(jmp)
        CASE(phi)
        CASE(constant)
        CASE(copy)
        CASE(cast)
        CASE(load_register)
        CASE(store_register)
        CASE(load_memory)
        CASE(store_memory)
        CASE(call)
        CASE(neg)
        case Opcode::i_not: return "not";
        CASE(add)
        CASE(sub)
        case Opcode::i_xor: return "xor";
        case Opcode::i_or: return "or";
        case Opcode::i_and: return "and";
        CASE(shl)
        CASE(shr)
        CASE(sar)
        CASE(eq)
        CASE(ne)
        CASE(lt)
        CASE(ge)
        CASE(ltu)
        CASE(geu)
        CASE(mux)
        CASE(mul)
        CASE(mulu)
        CASE(div)
        CASE(divu)
#undef CASE
        default: return "(unknown)";
    }
}

const char* Dot_printer::type_name(Type type) {
    switch (type) {
#define CASE(name) case Type::name: return #name;
        CASE(control)
        CASE(memory)
        CASE(none)
        CASE(i1)
        CASE(i8)
        CASE(i16)
        CASE(i32)
        CASE(i64)
#undef CASE
        default: return "(unknown)";
    }
}

void Dot_printer::write_node_content(std::ostream& stream, Node* node) {
    // Print opcode and opcode relevant information.
    stream << opcode_name(node->opcode());

    switch (node->opcode()) {
        case Opcode::constant: {
            uint64_t value = static_cast<Constant*>(node)->const_value();
            int64_t svalue = value;

            if (static_cast<int8_t>(value) == svalue) {

                // For small enough number we print decimal.
                stream << ' ' << svalue;

            } else {
                if (svalue < 0)
                    util::log(" -{:#x}", -value);
                else
                    util::log(" {:#x}", value);
            }
            break;
        }
        case Opcode::cast:
            if (static_cast<Cast*>(node)->sign_extend()) stream << " sext";
            break;
        case Opcode::load_register:
        case Opcode::store_register:
            stream << " r" << static_cast<Register_access*>(node)->regnum();
            break;
        case Opcode::call: {
            auto call_node = static_cast<Call*>(node);
            util::log(" {:#x}{}", call_node->target(), call_node->need_context() ? " with context" : "");
            break;
        }
        default: break;
    }
}

void Dot_printer::run(Graph& graph) {

    // Print preamble
    std::clog << "digraph G {\n\trankdir = BT;\n\tnode [shape=record];\n";
    
    visit_postorder(graph, [this](Node* node) {
        uint16_t opcode = node->opcode();

        // Constant nodes are treated specially, we print out one node per each usage.
        if (opcode == Opcode::constant) return;

        // Draw the node with type, opcode
        util::log("\t\"{:x}\" [label=\"{{", reinterpret_cast<uintptr_t>(node));

        // First compute whether input label is needed.
        bool need_label;
        if (node->operand_count() <= 1) {
            // No ambiguities in this case.
            need_label = false;

        } else if (is_target_specific(opcode)) {
            need_label = true;

        } else if (opcode == Opcode::call || opcode == Opcode::phi) {
            // If call has more than 1 arguments, we needs to distinguish between them.
            need_label = node->operand_count() > 2;

        } else if (opcode == Opcode::block || opcode == Opcode::store_memory) {
            // Block is ordered, and store_memory's address/value operands need distinction.
            need_label = true;

        } else {
            need_label = is_pure_opcode(opcode) && !is_commutative_opcode(opcode);
        }

        if (need_label) {
            size_t operand_count = node->operand_count();
            if (operand_count != 0) {
                std::clog << '{';
                for (size_t i = 0; i < operand_count; i++) {
                    if (i != 0) std::clog << '|';
                    util::log("<i{}>", i);
                }
                std::clog << "}|";
            }
        }

        // If node only produces one value, we will inline the type with the instruction body.
        if (node->value_count() == 1) {
            Type type = node->value(0).type();
            if (type != Type::control && type != Type::memory) {
                std::clog << type_name(type) << ' ';
            }
        }

        // Print the node content
        write_node_content(std::clog, node);

        // If node produces multiple value, print out fields.
        if (node->value_count() != 1) {
            std::clog << "|{";
            for (auto value: node->values()) {
                if (value.index() != 0) std::clog << '|';
                util::log("<o{}>{}", value.index(), type_name(value.type()));
            }
            std::clog << '}';
        }

        std::clog << "}\"]\n";

        auto operands = node->operands();
        for (size_t i = 0; i < operands.size(); i++) {
            auto operand = operands[i];

            const char* color;
            switch (operand.type()) {
                case Type::control: color = "red"; break;
                case Type::memory: color = "blue"; break;
                default: color = nullptr; break;
            }

            // Color keepalive edges differently.
            if (node->opcode() == Opcode::exit && operand.references().size() == 2) color = "pink";

            // Create a new node for a constant operand. This makes the graph more visible.
            if (operand.is_const()) {
                util::log(
                    "\t\"{:x}_{}\" [label=\"{} ", reinterpret_cast<uintptr_t>(node), i, type_name(operand.type())
                );

                write_node_content(std::clog, operand.node());
                std::clog << "\"]\n";
            }

            util::log(need_label ? "\t\"{:x}\":i{} -> " : "\t\"{:x}\" -> ", reinterpret_cast<uintptr_t>(node), i);

            if (operand.is_const()) {
                util::log("\"{:x}_{}\"", reinterpret_cast<uintptr_t>(node), i);
            } else {
                util::log(
                    operand.node()->value_count() == 1 ? "\"{:x}\"" : "\"{:x}\":o{}",
                    reinterpret_cast<uintptr_t>(operand.node()), operand.index()
                );
            }

            util::log(color ? " [color={}];\n" : ";\n", color);
        }
    });

    // Print epilogue for exit node.
    std::clog << "}" << std::endl;
}

}
