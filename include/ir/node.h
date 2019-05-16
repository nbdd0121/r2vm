#ifndef IR_NODE_H
#define IR_NODE_H

#include <cstdint>
#include <utility>
#include <vector>

#include "util/assert.h"
#include "util/multiset.h"
#include "util/small_vector.h"

namespace ir {

class Node;
class Value;
class Graph;

namespace pass {
class Pass;
}

namespace internal {
void clear_visited_flags(Graph&);
template<typename F>
void visit_postorder_actual(Node* node, F func);
}

enum class Type: uint8_t {
    none = 0,
    i1 = 1,
    i8 = 8,
    i16 = 16,
    i32 = 32,
    i64 = 64,
    memory = 0xFE,
    control = 0xFF,
};

static inline size_t get_type_size(Type type) {
    return static_cast<uint8_t>(type);
}

namespace Opcode {

enum: uint16_t {
    /** Control flow opcodes **/
    // Input: None. Output: Memory.
    entry,

    // Input: (Control|Memory)[]. Output: None.
    // Memory edges are keepalive edges to keep endless loop alive.
    exit,

    // Input: Control[]. Output: Memory.
    block,

    // Input: Memory, Value. Output: Control, Control.
    i_if,

    // Input: Memory. Output: Control.
    jmp,

    // Input: Memory, Value[]. Output: Value.
    phi,

    /** Opcodes with side-effects **/

    /* Machine register load/store */
    // Input: Memory. Output: Memory, Value.
    load_register,

    // Input: Memory, Value. Output: Memory.
    store_register,

    /* Memory load/store */
    // Input: Memory, Value. Output: Memory, Value.
    load_memory,

    // Input: Memory, Value, Value. Output: Memory.
    store_memory,

    // Call a helper function.
    // Input: Memory, Value[]. Output: Memory, Value(opt)
    call,

    /** Pure opcodes **/

    // Input: None. Output: Value.
    constant,

    // Used for assisting register allocation.
    copy,

    // Input: Value. Output: Value.
    cast,

    /*
     * Unary ops
     * Input: Value. Output: Value.
     */
    neg,
    i_not,

    /*
     * Binary ops
     * Input: Value, Value. Output: Value.
     */
    /* Arithmetic operations */
    add,
    sub,
    i_xor,
    i_or,
    i_and,

    /* Shift operations */
    shl,
    shr,
    sar,

    /* Compare */
    eq,
    ne,
    lt,
    ge,
    ltu,
    geu,

    /*
     * Ternary op
     * Input: Value, Value, Value. Output: Value.
     */
    mux,

    /* Other arithmetic ops */
    // Multiplication. It returns both lower and higher bits.
    // Input: Value, Value. Output: Value, Value.
    mul,
    mulu,

    // Division. It returns both quotient and remainder.
    // Input: Value, Value. Output: Value, Value.
    div,
    divu,

    /* Opcodes after target_start are target-specific opcodes */
    target_start,
};

}

static inline bool is_target_specific(uint16_t opcode) {
    return opcode >= Opcode::target_start;
}

static inline bool is_pure_opcode(uint16_t opcode) {
    return !is_target_specific(opcode) && opcode >= Opcode::constant;
}

static inline bool is_binary_opcode(uint16_t opcode) {
    // Due to the specialness of multiplication, they are not included here.
    return opcode >= Opcode::add && opcode <= Opcode::geu;
}

static inline bool is_commutative_opcode(uint16_t opcode) {
    switch(opcode) {
        case Opcode::add:
        case Opcode::i_xor:
        case Opcode::i_or:
        case Opcode::i_and:
        case Opcode::eq:
        case Opcode::ne:
        case Opcode::mul:
        case Opcode::mulu:
            return true;
        default:
            return false;
    }
}

// Represents a value defined by a node. Note that the node may be null.
class Value {
private:
    Node* _node;
    size_t _index;
public:
    Value(): _node{nullptr}, _index{0} {}
    Value(Node* node, size_t index): _node{node}, _index{index} {}

    Node* node() const { return _node; }
    size_t index() const { return _index; }

    inline Type type() const;
    inline const util::Multiset<Node*>& references() const;

    explicit operator bool() { return _node != nullptr; }

    // Some frequently used utility function.
    inline uint16_t opcode() const;
    inline bool is_const() const;
    inline uint64_t const_value() const;
};

static inline bool operator ==(Value a, Value b) {
    return a.node() == b.node() && a.index() == b.index();
}

static inline bool operator !=(Value a, Value b) { return !(a == b); }

class Node {
public:
    // Helper classes for iterating output values.
    class Value_iterator {
        Node* _node;
        size_t _index;
    public:
        Value_iterator(Node* node, size_t index) noexcept: _node{node}, _index{index} {};
        Value operator *() const { return _node->value(_index); }
        bool operator !=(const Value_iterator& iter) const noexcept {
            ASSERT(_node == iter._node);
            return _index != iter._index;
        }
        Value_iterator& operator ++() noexcept { _index++; return *this; }
    };

    class Value_iterable {
        Node* _node;
    public:
        Value_iterable(Node* node) noexcept: _node{node} {};
        Value_iterator begin() const noexcept { return {_node, 0}; }
        Value_iterator end() const noexcept { return {_node, _node->value_count()}; }
    };

    using Operand_container = util::Small_vector<Value, 2>;
    using Type_container = util::Small_vector<Type, 2>;
private:

    // Values that this node references.
    Operand_container _operands;

    // Nodes that references the value of this node.
    util::Small_vector<util::Multiset<Node*>, 2> _references;

    // The output type of this node.
    Type_container _type;

    // Opcode of the node.
    uint16_t _opcode;

    // Whether the node is visited. For graph walking only.
    // 0 - not visited, 1 - visited, 2 - visiting.
    uint8_t _visited;

public:
    Node(uint16_t opcode, Type_container&& type, Operand_container&& operands);
    virtual ~Node();

    // Disable copy construction and assignment. Node should live on heap.
    Node(const Node& node) = delete;
    Node(Node&& node) = delete;
    void operator =(const Node& node) = delete;
    void operator =(Node&& node) = delete;

private:
    void link();
    void unlink();

public:
    // Field accessors and mutators
    // A node can produce one or more values. The following functions allow access to these values.
    size_t value_count() const { return _type.size(); }
    Value value(size_t index) { return {this, index}; }
    Value_iterable values() { return {this}; }

    uint16_t opcode() const { return _opcode; }
    void opcode(uint16_t opcode) { _opcode = opcode; }

    // Operand accessors and mutators
    const Operand_container& operands() const { return _operands; }
    void operands(Operand_container&& operands);
    size_t operand_count() const { return _operands.size(); }

    Value operand(size_t index) const {
        ASSERT(index < _operands.size());
        return _operands[index];
    }

    void operand_set(size_t index, Value value);
    size_t operand_find(Value value);
    void operand_add(Value value);
    void operand_delete(Value value);
    void operand_swap(size_t first, size_t second) { std::swap(_operands[first], _operands[second]); }
    void operand_update(Value oldvalue, Value newvalue) { operand_set(operand_find(oldvalue), newvalue); }

    friend Value;
    friend Graph;
    friend void internal::clear_visited_flags(Graph&);
    template<typename F>
    friend void internal::visit_postorder_actual(Node* node, F func);
    friend pass::Pass;
};

class Constant: public Node {
private:
    uint64_t _const_value;

public:
    Constant(Type type, uint64_t value): Node(Opcode::constant, {type}, {}), _const_value{value} {}

    uint64_t const_value() const { return _const_value; }
    void const_value(uint64_t value) { _const_value = value; }
};

class Cast: public Node {
private:
    bool _sext;

public:
    Cast(Type type, bool sext, Value value): Node(Opcode::cast, {type}, {value}), _sext{sext} {}

    bool sign_extend() const { return _sext; }
    void sign_extend(bool sext) { _sext = sext; }
};

class Register_access: public Node {
private:
    uint16_t _regnum;

public:
    Register_access(uint16_t regnum, uint16_t opcode, Type_container&& type, Operand_container&& operands):
        Node(opcode, std::move(type), std::move(operands)), _regnum{regnum} {}

    uint16_t regnum() const { return _regnum; }
};

// For all nodes that is paired with another node. This include block/jmp/if.
class Paired: public Node {
private:
    Node* _mate;

public:
    // Inherit constructor.
    using Node::Node;

    Node* mate() const { return _mate; }
    void mate(Node* mate) { _mate = mate; }
};

class Call: public Node {
private:
    // The helper function to call.
    uintptr_t _target;
    // Whether the evaluation context is needed to call such a function.
    bool _need_context;

public:
    Call(uintptr_t target, bool need_context, Type_container&& types, Operand_container&& operands):
        Node(Opcode::call, std::move(types), std::move(operands)), _target{target}, _need_context{need_context} {}

    uintptr_t target() const { return _target; }
    bool need_context() const { return _need_context; }
};

class Graph {
private:
    std::vector<Node*> _heap;
    Node* _entry;
    Node* _exit;

public:
    Graph();
    Graph(const Graph&) = delete;
    Graph(Graph&&) = default;
    ~Graph();

    Graph& operator =(const Graph&) = delete;
    Graph& operator =(Graph&&);

    Node* manage(Node* node) {
        _heap.push_back(node);
        return node;
    }

    const std::vector<Node*>& nodes() { return _heap; }

    // Free up dead nodes. Not necessary during compilation, but useful for reducing footprint when graph needs to be
    // cached.
    void garbage_collect();

    Node* entry() const { return _entry; }
    Node* exit() const { return _exit; }

    Graph clone() const;
    void inline_graph(Value control, Graph&& graph);

    friend void internal::clear_visited_flags(Graph&);
};

Type Value::type() const { return _node->_type[_index]; }
const util::Multiset<Node*>& Value::references() const { return _node->_references[_index]; }

uint16_t Value::opcode() const { return _node->_opcode; }
bool Value::is_const() const { return _node->_opcode == Opcode::constant; }
uint64_t Value::const_value() const { return static_cast<Constant*>(_node)->const_value(); }

} // ir

namespace std {

template<typename Key> struct hash;

template<> struct hash<::ir::Value> {
    size_t operator()(ir::Value val) const {
        return reinterpret_cast<uintptr_t>(val.node()) ^ val.index();
    }
};

} // std

#endif
