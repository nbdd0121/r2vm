#ifndef IR_ANALYSIS_H
#define IR_ANALYSIS_H

#include <unordered_map>
#include <unordered_set>

#include "ir/node.h"

namespace ir::analysis {

class Dominance;

// Helper function for control flow related analysis.
class Block {
public:
    // Get the real target of a control. Ignore keepalive edges.
    static Node* get_target(Value control);

    // Given a control, verify if it is a tail jump (jump to exit), and whether the pc of the next block is a known
    // value. The value of pc of next block will be returned, or null will be returned if it is not a tail jump, or
    // the value of pc is unknown.
    static Value get_tail_jmp_pc(Value control, uint16_t pc_regnum);

private:
    Graph& _graph;
    std::vector<Node*> _blocks;

public:
    Block(Graph& graph): _graph{graph} {
        enumerate_blocks();
    }

private:
    void enumerate_blocks();

public:
    const std::vector<Node*>& blocks() { return _blocks; }

    void update_keepalive();
    void simplify_graph();

    // Reorder basic blocks so that number of jumps emitted by backend is reduced. It relies on dominance calculation
    // to avoid keeping dominator before dominated blocks (which is simpler for code generator).
    void reorder(Dominance& dominance);
};

class Dominance {
    Graph& _graph;
    Block& _block_analysis;

    // Immediate dominators of nodes and their height.
    std::unordered_map<Node*, std::pair<Node*, size_t>> _idom;

    // Immediate post-dominators of nodes.
    std::unordered_map<Node*, Node*> _ipdom;

    // Dominance frontier of nodes.
    std::unordered_map<Node*, std::unordered_set<Node*>> _df;

    // Post-dominance frontier of nodes.
    std::unordered_map<Node*, std::unordered_set<Node*>> _pdf;

public:
    Dominance(Graph& graph, Block& block_analysis): _graph{graph}, _block_analysis{block_analysis} {
        compute_idom();
        compute_ipdom();
        compute_df();
        compute_pdf();
    }

    Node* immediate_dominator(Node* block) { return _idom[block].first; }
    Node* immediate_postdominator(Node* block) { return _ipdom[block]; }
    const std::unordered_set<Node*>& dominance_frontier(Node* block) { return _df[block]; }
    const std::unordered_set<Node*>& postdominance_frontier(Node* block) { return _pdf[block]; }
    Node* least_common_dominator(Node* a, Node* b);

private:
    void compute_idom();
    void compute_ipdom();
    void compute_df();
    void compute_pdf();

};

class Scheduler {
private:
    Graph& _graph;
    Block& _block_analysis;
    Dominance& _dominance;

    // Record how many inputs are yet to be satisified in schedule_node_early.
    std::unordered_map<Node*, ssize_t> _unsatisified_input_count;
    std::unordered_map<Node*, Node*> _late;
    std::unordered_map<Node*, std::vector<Node*>> _nodelist;

    // Record the all nodes scheduled to the current block in schedule_block.
    std::vector<Node*>* _list;

    Node* _block;

public:
    Scheduler(Graph& graph, Block& block_analysis, Dominance& dominance):
        _graph{graph}, _block_analysis{block_analysis}, _dominance{dominance} {

    }

private:
    void schedule_node_early(Node* node);
    void schedule_node_late(Node* node);

    void schedule_block(Node* block);

public:
    void schedule();
    const std::vector<Node*>& get_node_list(Node* block) { return _nodelist[block]; }
    std::vector<Node*>& get_mutable_node_list(Node* block) { return _nodelist[block]; }
};

class Local_load_store_elimination {
private:
    Graph& _graph;
    Block& _block_analysis;
    size_t _regcount;

public:
    Local_load_store_elimination(Graph& graph, Block& block_analysis, size_t regcount):
        _graph{graph}, _block_analysis{block_analysis}, _regcount{regcount} {
    }

public:
    void run();
};

class Load_store_elimination {
private:
    Graph& _graph;
    Block& _block_analysis;
    Dominance& _dom;

    // Tracks all memory operations with containing basic blocks.
    std::unordered_map<Node*, std::vector<Node*>> _memops;

    // Tracks all PHI nodes created.
    std::vector<std::unordered_map<Node*, Node*>> _phis;

    // The value stack used in the standard renaming algorithm. nullptr here indicates that the value is unavailable.
    std::vector<std::vector<Value>> _value_stack;

    // Used for walking through each block to get all memory related nodes.
    std::vector<Node*>* _oplist;

public:
    Load_store_elimination(Graph& graph, Block& block_analysis, Dominance& dom, size_t regcount):
        _graph{graph}, _block_analysis{block_analysis}, _dom{dom},
        _memops(regcount), _value_stack(regcount, std::vector<Value>{{}}) {

        populate_memops();
    }

private:
    void populate_memops();

    void fill_load_phi(Node* block);
    void rename_load(Node* block);

    void fill_store_phi(Node* block);
    void rename_store(Node* block);

public:
    void eliminate_load();
    void eliminate_store();
};

}

#endif
