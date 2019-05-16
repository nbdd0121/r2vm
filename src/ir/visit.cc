#include "ir/node.h"
#include "ir/pass.h"
#include "ir/visit.h"

namespace ir {

namespace internal {

void clear_visited_flags(Graph& graph) {
    for (auto node: graph._heap) {
        node->_visited = false;
    }
}

}

void replace_value(Value oldvalue, Value newvalue) {
    while (!oldvalue.references().empty()) {
        (*oldvalue.references().rbegin())->operand_update(oldvalue, newvalue);
    }
}

}
