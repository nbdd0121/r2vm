#include "emu/mmu.h"
#include "emu/state.h"
#include "ir/builder.h"
#include "ir/pass.h"
#include "ir/visit.h"
#include "util/functional.h"

namespace ir::pass {

void Lowering::run(Graph& graph) {

    // We perform target-independent lowering here. After lowering, load/store_memory represents loading and storing
    // represents host address space instead of guest's. For paging MMU, memory operations are translated to helper
    // function calls.

    Builder builder { graph };

    visit_postorder(graph, [&](Node* node) {
        switch (node->opcode()) {
            case Opcode::load_memory: {

                // In this case lowering is not needed.
                if (!emu::state::no_direct_memory_access) break;

                auto output = node->value(1);

                uintptr_t func;
                switch (output.type()) {
                    case Type::i8: func = reinterpret_cast<uintptr_t>(&emu::load_memory<uint8_t>); break;
                    case Type::i16: func = reinterpret_cast<uintptr_t>(&emu::load_memory<uint16_t>); break;
                    case Type::i32: func = reinterpret_cast<uintptr_t>(&emu::load_memory<uint32_t>); break;
                    case Type::i64: func = reinterpret_cast<uintptr_t>(&emu::load_memory<uint64_t>); break;
                    default: ASSERT(0);
                }

                auto call_node = graph.manage(new Call(
                    func, false, {Type::memory, output.type()}, {node->operand(0), node->operand(1)}
                ));

                replace_value(node->value(0), call_node->value(0));
                replace_value(output, call_node->value(1));
                break;
            }
            case Opcode::store_memory: {

                // In this case lowering is not needed.
                if (!emu::state::no_direct_memory_access) break;

                auto value = node->operand(2);

                uintptr_t func;
                switch (value.type()) {
                    case Type::i8: func = reinterpret_cast<uintptr_t>(&emu::store_memory<uint8_t>); break;
                    case Type::i16: func = reinterpret_cast<uintptr_t>(&emu::store_memory<uint16_t>); break;
                    case Type::i32: func = reinterpret_cast<uintptr_t>(&emu::store_memory<uint32_t>); break;
                    case Type::i64: func = reinterpret_cast<uintptr_t>(&emu::store_memory<uint64_t>); break;
                    default: ASSERT(0);
                }

                auto call_node = graph.manage(new Call(
                    func, false, {Type::memory}, {node->operand(0), node->operand(1), value}
                ));

                replace_value(node->value(0), call_node->value(0));
                break;
            }
            default:
                break;
        }
    });
}

}
