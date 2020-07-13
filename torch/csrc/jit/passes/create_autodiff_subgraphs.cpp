#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <ATen/core/functional.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>


namespace torch {
namespace jit {

namespace {

// Which index in b's owning Node is b
size_t blockIndex(const Block* b) {
  auto n = b->owningNode();
  AT_ASSERT(n);
  for (size_t i = 0; i < n->blocks().size(); ++i) {
    if (n->blocks()[i] == b) {
      return i;
    }
  }
  AT_ASSERT(false);
}

bool isAfter(Node* n1, Node* n2) {
  // Invalid to call with the same node as both args
  AT_ASSERT(n1 != n2);

  // Set n1 and n2 to be the number of blocks from the Graph block
  size_t d_1 = n1->blocksFromGraphBlock();
  size_t d_2 = n2->blocksFromGraphBlock();

  for (; d_1 > d_2; --d_1) {
    n1 = n1->owningBlock()->owningNode();
    // n2 contains n1
    if (n1 == n2) {
      return true;
    }
  }

  for (; d_2 > d_1; --d_2) {
    n2 = n2->owningBlock()->owningNode();
    // n1 contains n2
    if (n2 == n1) {
      return false;
    }
  }

  // Now they are the same numer of blocks from the graph block,
  // recurse upwards, checking if they are on the same block
  while (true) {
    if (n1->owningBlock() == n2->owningBlock()) {
      return n1->isAfter(n2);
    }

    auto new_n1 = n1->owningBlock()->owningNode();
    auto new_n2 = n2->owningBlock()->owningNode();

    AT_ASSERT(new_n1 != nullptr);
    AT_ASSERT(new_n2 != nullptr);

    if (new_n1 == new_n2) {
      // take whichever node is in the later block
      auto index_1 = blockIndex(n1->owningBlock());
      auto index_2 = blockIndex(n2->owningBlock());
      return index_1 > index_2;
    }

    n1 = new_n1;
    n2 = new_n2;
  }
}

bool isAfter(const Use& a, const Use& b) {
  // If two uses are the same node, we order on offset
  if (a.user == b.user) {
    return a.offset > b.offset;
  }

  return isAfter(a.user, b.user);
}

c10::optional<const Use> lastUse(Value * v) {
  if (v->uses().size() == 0) {
    return c10::nullopt;
  }
  Use last_use = v->uses()[0];
  for (size_t i = 1; i < v->uses().size(); ++i) {
    auto n_use = v->uses()[i];
    if (!isAfter(last_use, n_use)) {
      last_use = n_use;
    }
  }

  return last_use;
}

std::vector<c10::optional<const Use>> gatherLastUses(at::ArrayRef<Value*> values) {
  return fmap(values, lastUse);
}

struct ValueMapper {
    ValueMapper(Node * n, AliasDb& db, size_t subgraph_num_outputs) {
      last_uses_ = gatherLastUses(n->outputs());
      subgraph_num_outputs_ = subgraph_num_outputs;
      WithInsertPoint guard(n);
      auto g = n->owningGraph();
      placeholder_node_ = g->insertNode(g->create(prim::Uninitialized, 0));
      for (size_t i = 0; i < n->outputs().size(); ++i) {
        Value * existing = n->outputs().at(i);
        Value * new_value = placeholder_node_->insertOutput(i)->copyMetadata(n->outputs().at(i));
        db.replaceWithNewValue(existing, new_value);
      }
    }

    bool usesEqual(const Use& a, const Use& b) {
      return a.user == b.user && a.offset == b.offset;
    }

    void copyAliasing(Node * merged_node, AliasDb& db) {
      auto num_outputs = merged_node->outputs().size();
      auto new_outputs = merged_node->outputs().slice(subgraph_num_outputs_, num_outputs - subgraph_num_outputs_);
      for (Value * v: new_outputs) {
        auto maybe_last_use = lastUse(v);
        if (!maybe_last_use) {
          continue;
        }
        const Use last_use = *maybe_last_use;

        size_t i = 0;
        while (i < last_uses_.size() && last_uses_.at(i).has_value() && !usesEqual(*last_uses_.at(i), last_use)) {
          ++i;
        }
        TORCH_INTERNAL_ASSERT(i != last_uses_.size());
        db.replaceWithNewValue(placeholder_node_->outputs().at(i), v);
      }
      placeholder_node_->destroy();
    }

    std::vector<c10::optional<const Use>> last_uses_;
    size_t subgraph_num_outputs_;
    Node * placeholder_node_;
};

struct WorkPair : public std::pair<Node*, Node*> {
  using pair::pair;

  Node* start() {
    return this->first;
  }
  Node* end() {
    return this->second;
  }
};

class SubgraphSlicer {
 public:
  SubgraphSlicer(
      Block* block,
      std::shared_ptr<Graph> graph,
      size_t minSubgraphSize,
      AliasDb& aliasDb)
      : block_(block),
        graph_(std::move(graph)),
        minSubgraphSize_(minSubgraphSize),
        aliasDb_(aliasDb) {}

  void run(std::vector<Node*>& diffGraphs) {
    // buildWorkSets();

    // We need to run the slicer multiple times in order to get all merge
    // opportunities. This is because moveBeforeTopologicalValid may reorder
    // nodes to be AFTER the current iteration point. In order to properly
    // consider those nodes for merging, we need run the pass until no changes
    // have been made.
    //
    // Example:
    //   c = f(a, b)
    //   d = f(c)
    //   e = f(d)  <- iter is here, moving upward
    // After c.moveBeforeTopologicallyValid(e), we have:
    //   c = f(a, b)
    //   e = f(d)  <- iter still here
    //   d = f(c)  <- this was node moved on the other side.
    bool any_changed = true;

    auto worksets = buildWorkSets();
    for (auto& workset : worksets) {
      auto curr_work_group = buildWorkGroup(workset);
      while (any_changed) {
        any_changed = false;
        for (auto it = workset.end()->reverseIterator(); it != workset.start()->reverseIterator();) {
          bool changed;
          std::tie(it, changed) = scanNode(*it, aliasDb_);
          any_changed |= changed;
        }
      }
    }

    // while (any_changed) {
    //   any_changed = false;
    //   for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
    //     bool changed;
    //     std::tie(it, changed) = scanNode(*it, aliasDb_);
    //     any_changed |= changed;
    //   }
    // }

    // Done constructing subgraphs. Do some post-processing cleanup:
    // 1. Run CSE to delete redundant constant nodes.
    // 2. We may need to re-inline ones that are too small.
    for (auto node : block_->nodes()) {
      for (auto subBlock : node->blocks()) {
        SubgraphSlicer(subBlock, graph_, minSubgraphSize_, aliasDb_).run(diffGraphs);
      }
    }

    auto curNode = *block_->nodes().rbegin();
    while (curNode != *block_->nodes().rend()) {
      // Save the previous node, since we might delete `curNode` in next block
      auto prevNode = curNode->prev();
      if (curNode->kind() == prim::DifferentiableGraph) {
        // Inlining nodes may cause some subexpression to come back in the
        // subgraphs (for example, copying constants in repeatedly will generate
        // redundant prim::Constants). Run CSE to clean them up.
        EliminateCommonSubexpression(curNode->g(attr::Subgraph));

        if (!inlineIfTooSmall(curNode)) {
          diffGraphs.push_back(curNode);
        }
      }
      curNode = prevNode;
    }
    // Run CSE one more time to eliminate duplicates that may have occurred
    // while re-inlining subgraphs.
    EliminateCommonSubexpression(graph_);
  }

 private:

  std::unordered_set<Node *> buildWorkGroup(WorkPair& pair) {
    Node * curr = pair.start()->next();
    std::unordered_set<Node *> nodes;
    while (curr != pair.end()) {
      nodes.insert(curr);
      curr = curr->next();
    }
    return nodes;
  }

  std::vector<WorkPair> buildWorkSets() {

    // work sets are delineated by the nodes that cannot be moved,
    // so they are exclusive and represent [bound_node, bound_node]
    Node * end_bound_node = block_->return_node();
    Node * curr = end_bound_node->prev();

    std::vector<WorkPair> worklist;

    while (curr != block_->param_node()) {
      // constants are allowed in all sets, so we ignore them
      if (curr->kind() == prim::Constant) {
        curr = curr->prev();
        continue;
      }

      if (curr->hasSideEffects()) {
        worklist.emplace_back(curr, end_bound_node);
        end_bound_node = curr;
      }
      curr = curr->prev();
    }
    worklist.emplace_back(curr, end_bound_node);
    return worklist;
  }


  // Inline this node's group subgraph into the outer graph if it's smaller
  // than the specified minimum size.
  //
  // Returns true if an inlining has occurred, false otherwise.
  bool inlineIfTooSmall(Node* n) {
    AT_ASSERT(n->kind() == prim::DifferentiableGraph);
    auto subgraph = SubgraphUtils::getSubgraph(n);
    size_t i = 0;
    for (auto it = subgraph->nodes().begin(); it != subgraph->nodes().end();
         ++it) {
      if (++i >= minSubgraphSize_) {
        return false;
      }
    }

    SubgraphUtils::unmergeSubgraph(n);
    return true;
  }

  value_list sortReverseTopological(ArrayRef<Value*> inputs) {
    value_list result;
    for (auto i : inputs) {
      if (i->node()->owningBlock() == block_) {
        result.push_back(i);
      }
    }
    // Sort in reverse topological order
    std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
      return a->node()->isAfter(b->node());
    });
    return result;
  }

  bool shouldConsiderForMerge(Node* node) {
    // if we're already in the process of merging
    if (node->kind() == prim::DifferentiableGraph) {
      return true;
    }
    if (node->kind() == prim::Constant) {
      return false;
    }
    return isDifferentiable(node);
  }



  std::pair<graph_node_list::iterator, bool> scanNode(
      Node* consumer,
      AliasDb& aliasDb) {
    if (shouldConsiderForMerge(consumer)) {
      if (consumer->kind() != prim::DifferentiableGraph) {
        // we need a way to map the node's outputs to the new Singleton subgraphs outputs

        ValueMapper vm(consumer, aliasDb, 0);
        consumer = SubgraphUtils::createSingletonSubgraph(consumer, prim::DifferentiableGraph);
        vm.copyAliasing(consumer, aliasDb);
      }
      auto inputs = sortReverseTopological(consumer->inputs());
      for (auto input : inputs) {
        if (auto group = tryMerge(consumer, input->node(), aliasDb)) {
          // we successfully merged, so the new group's `inputs` may have
          // changed. So rescan the new group for more merging opportunities.
          return std::make_pair(group.value()->reverseIterator(), true);
        }
      }
    }

    return std::make_pair(++consumer->reverseIterator(), false);
  }

  // Try to merge `producer` into `consumer`. If successful, this destroys
  // `producer` and returns the `consumer` group.
  c10::optional<Node*> tryMerge(
      Node* consumer,
      Node* producer,
      AliasDb& aliasDb) {
    AT_ASSERT(consumer->kind() == prim::DifferentiableGraph);
    bool canMerge = shouldConsiderForMerge(producer) &&
        aliasDb.moveBeforeTopologicallyValid(producer, consumer);

    if (!canMerge) {
      return c10::nullopt;
    }

    ValueMapper vm(producer, aliasDb, consumer->outputs().size());
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
    vm.copyAliasing(consumer, aliasDb);
    return consumer;
  }

  Block* block_;
  std::shared_ptr<Graph> graph_;
  size_t minSubgraphSize_;
  AliasDb& aliasDb_;
  // std::vector<WorkPair> workset_;
  // std::unordered_set<Node*> curr_work_group_;

};
} // anonymous namespace

std::vector<Node*> CreateAutodiffSubgraphs(
    const std::shared_ptr<Graph>& graph,
    size_t threshold) {
  std::vector<Node*> diff_nodes;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  AliasDb db(graph);
  SubgraphSlicer(graph->block(), graph, threshold, db).run(diff_nodes);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count() << "[s]" << std::endl;

  return diff_nodes;
}
} // namespace jit
} // namespace torch
