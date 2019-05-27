#ifndef IR_PASS_H
#define IR_PASS_H

#include <ostream>
#include <unordered_set>

#include "ir/node.h"

namespace ir::pass {

class Dot_printer {
public:
    static const char* opcode_name(uint16_t opcode);
    static const char* type_name(Type type);

protected:
    virtual void write_node_content(std::ostream& stream, Node* node);

public:
    void run(Graph& graph);
};

class Local_value_numbering {
private:
    struct Hash {
        size_t operator ()(Node* node) const noexcept;
    };

    struct Equal_to {
        bool operator ()(Node* a, Node* b) const noexcept;
    };

private:
    Graph& _graph;
    std::unordered_set<Node*, Hash, Equal_to> _set;

    static uint64_t sign_extend(Type type, uint64_t value);
    static uint64_t zero_extend(Type type, uint64_t value);
    static uint64_t cast(Type type, Type oldtype, bool sext, uint64_t value);
    static uint64_t binary(Type type, uint16_t opcode, uint64_t l, uint64_t r);

    Value new_constant(Type type, uint64_t const_value);
    void replace_with_constant(Value value, uint64_t const_value);
    void lvn(Node* node);
    void process(Node* node);

public:
    Local_value_numbering(Graph& graph): _graph{graph} {};
    void run();
};

// Target-independent lowering pass.
class Lowering {
public:
    void run(Graph& graph);
};

}

#endif
