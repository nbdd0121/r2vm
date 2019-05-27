#include <deque>
#include <functional>

#include "ir/analysis.h"

namespace ir::analysis {

void Dominance::compute_idom() {

    // Mapping between dfn and vertex. 0 represents the entry node.
    std::unordered_map<Node*, size_t> dfn;
    std::vector<Node*> vertices;

    // Parent in the DFS tree.
    std::vector<ssize_t> parents;

    // Do a depth-first search to assign these nodes a DFN and determine their parents in DFS tree.
    {
        std::deque<std::pair<size_t, Node*>> stack { {-1, _graph.entry() }};
        while (!stack.empty()) {
            size_t parent;
            Node* node;
            std::tie(parent, node) = stack.front();
            stack.pop_front();

            auto& id = dfn[node];

            // If id == 0, then it is either the entry node, or this is a freshly encountered node.
            // As the entry node will only be visited once, id != 0 means the node is already visited.
            if (id != 0) continue;

            id = vertices.size();
            vertices.push_back(node);
            parents.push_back(parent);

            if (node->opcode() == Opcode::exit) continue;

            auto end = node->opcode() == Opcode::entry ? node : static_cast<Paired*>(node)->mate();
            for (auto value: end->values()) {

                // Skip keepalive edges.
                bool skip_exit = value.references().size() == 2;

                for (auto ref: value.references()) {
                    if (skip_exit && ref->opcode() == Opcode::exit) continue;
                    stack.push_front({id, ref});
                }
            }
        }
    }

    // Initialize variables.
    size_t count = vertices.size();
    std::vector<size_t> sdoms(count);
    std::vector<ssize_t> idoms(count, -1);
    std::vector<ssize_t> ancestors(count, -1);
    std::vector<size_t> bests(count);
    std::vector<std::vector<size_t>> buckets(count);
    for (size_t i = 0; i < count; i++) {
        sdoms[i] = i;
        bests[i] = i;
    }

    // Lengauer-Tarjan algorithm with simple eval and link.
    std::function<size_t(size_t)> eval = [&](size_t node) {
        auto ancestor = ancestors[node];
        if (ancestor == -1) return node;
        if (ancestors[ancestor] != -1) {
            auto u = eval(ancestor);
            ASSERT(u == bests[ancestor]);
            if (sdoms[bests[node]] > sdoms[u]) bests[node] = u;
            ancestors[node] = ancestors[ancestor];
        }
        return bests[node];
    };

    auto link = [&](size_t parent, size_t node) {
        ASSERT(ancestors[node] == -1);
        ancestors[node] = parent;
    };

    for (size_t i = count - 1; i > 0; i--) {
        auto node = vertices[i];
        auto parent = parents[i];
        for (auto operand: node->operands()) {

            // Skip keepalive edges.
            if (node->opcode() == Opcode::exit && operand.references().size() == 2) continue;

            // Retrieve the starting node.
            auto block = operand.node();
            if (block->opcode() != Opcode::entry) block = static_cast<Paired*>(block)->mate();

            size_t pred = dfn[block];
            size_t u = eval(pred);
            if (sdoms[i] > sdoms[u]) {
                sdoms[i] = sdoms[u];
            }
        }
        buckets[sdoms[i]].push_back(i);
        link(parent, i);

        for (auto v: buckets[parent]) {
            auto u = eval(v);
            idoms[v] = sdoms[u] < sdoms[v] ? u : parent;
        }
        buckets[parent].clear();
    }

    for (size_t i = 1; i < count; i++) {
        ASSERT(idoms[i] != -1);
        if (static_cast<size_t>(idoms[i]) != sdoms[i]) {
            idoms[i] = idoms[idoms[i]];
        }

        // Turn DFN relation into relations between actual ir::Node's.
        auto idom_node = vertices[idoms[i]];
        _idom[vertices[i]] = {idom_node, idoms[i] == 0 ? 1 : _idom[idom_node].second + 1};
    }
}

void Dominance::compute_ipdom() {

    // Mapping between dfn and vertex. 0 represents the exit node.
    std::unordered_map<Node*, size_t> dfn;
    std::vector<Node*> vertices;

    // Parent in the DFS tree.
    std::vector<ssize_t> parents;

    // Do a depth-first search to assign these nodes a DFN and determine their parents in DFS tree.
    {
        std::deque<std::pair<size_t, Node*>> stack { {-1, _graph.exit() }};
        while (!stack.empty()) {
            size_t parent;
            Node* node;
            std::tie(parent, node) = stack.front();
            stack.pop_front();

            auto& id = dfn[node];

            // If id == 0, then it is either the exit node, or this is a freshly encountered node.
            // As the exit node will only be visited once, id != 0 means the node is already visited.
            if (id != 0) continue;

            id = vertices.size();
            vertices.push_back(node);
            parents.push_back(parent);

            for (auto operand: node->operands()) {

                // Note that we will treat keepalive edges as real edges. Otherwise, post-dominator tree may not exist
                // if there are infinite loops. Note that this is only safe when blocks referenced by keepalive edges
                // have otherwise no ways to reach the exit node. The keepalive edges inserted by ir::analysis::Block
                // always satisfy the constraint.

                // Retrive the starting node.
                auto block = operand.node();
                if (block->opcode() != Opcode::entry) block = static_cast<Paired*>(block)->mate();

                stack.push_front({id, block});
            }
        }
    }

    // Initialize variables.
    size_t count = vertices.size();
    std::vector<size_t> sdoms(count);
    std::vector<ssize_t> idoms(count, -1);
    std::vector<ssize_t> ancestors(count, -1);
    std::vector<size_t> bests(count);
    std::vector<std::vector<size_t>> buckets(count);
    for (size_t i = 0; i < count; i++) {
        sdoms[i] = i;
        bests[i] = i;
    }

    // Lengauer-Tarjan algorithm with simple eval and link.
    std::function<size_t(size_t)> eval = [&](size_t node) {
        auto ancestor = ancestors[node];
        if (ancestor == -1) return node;
        if (ancestors[ancestor] != -1) {
            eval(ancestor);
            if (sdoms[bests[node]] > sdoms[bests[ancestor]]) bests[node] = bests[ancestor];
            ancestors[node] = ancestors[ancestor];
        }
        return bests[node];
    };

    auto link = [&](size_t parent, size_t node) {
        ancestors[node] = parent;
    };

    for (size_t i = count - 1; i > 0; i--) {
        auto node = vertices[i];
        auto parent = parents[i];

        auto end = node->opcode() == Opcode::entry ? node : static_cast<Paired*>(node)->mate();
        for (auto value: end->values()) {

            for (auto ref: value.references()) {
                size_t pred = dfn[ref];

                // Unencountered node in DFS. This should not happen if keepalive edges are correctly inserted.
                ASSERT(pred != 0 || ref->opcode() == Opcode::exit);

                size_t u = eval(pred);
                if (sdoms[i] > sdoms[u]) {
                    sdoms[i] = sdoms[u];
                }
            }
        }
        buckets[sdoms[i]].push_back(i);
        link(parent, i);

        for (auto v: buckets[parent]) {
            auto u = eval(v);
            idoms[v] = sdoms[u] < sdoms[v] ? u : parent;
        }
        buckets[parent].clear();
    }

    for (size_t i = 1; i < count; i++) {
        ASSERT(idoms[i] != -1);
        if (static_cast<size_t>(idoms[i]) != sdoms[i]) {
            idoms[i] = idoms[idoms[i]];
        }

        // Turn DFN relation into relations between actual ir::Node's.
        _ipdom[vertices[i]] = vertices[idoms[i]];
    }
}

void Dominance::compute_df() {
    for (auto node: _block_analysis.blocks()) {

        // Nodes in dominance frontier must have multiple predecessor.
        if (node->operand_count() == 1) continue;

        auto idom = _idom[node].first;
        for (auto operand: node->operands()) {
            auto runner = operand.node();
            if (runner->opcode() != Opcode::entry) runner = static_cast<Paired*>(runner)->mate();

            // Walk up the DOM tree until the idom is met.
            while (runner != idom) {
                ASSERT(runner);
                _df[runner].insert(node);
                runner = _idom[runner].first;
            }
        }
    }
}

void Dominance::compute_pdf() {
    for (auto node: _block_analysis.blocks()) {

        // Nodes in post-dominance frontier must have multiple successor.
        auto end = static_cast<Paired*>(node)->mate();
        if (end->value_count() == 1) continue;

        auto ipdom = _ipdom[node];
        for (auto value: end->values()) {
            for (auto runner: value.references()) {
                while (runner != ipdom) {
                    ASSERT(runner);
                    _pdf[runner].insert(node);
                    runner = _ipdom[runner];
                }
            }
        }
    }
}

Node* Dominance::least_common_dominator(Node* a, Node* b) {

    // Special cases.
    if (a == nullptr) return b;
    if (b == nullptr) return a;
    if (a == b) return a;

    std::pair<Node*, size_t> adom = _idom[a];
    std::pair<Node*, size_t> bdom = _idom[b];

    // Walking up the tree until a and b are on the same height.
    while (adom.second > bdom.second) {
        a = adom.first;
        adom = _idom[a];
    }

    while (bdom.second > adom.second) {
        b = bdom.first;
        bdom = _idom[b];
    }

    // Further walking up util the lowest common dominator is found.
    while (a != b) {
        a = adom.first;
        b = bdom.first;
        adom = _idom[a];
        bdom = _idom[b];
    }

    return a;
}

}
