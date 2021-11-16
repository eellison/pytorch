#include <torch/csrc/jit/passes/canonicalize_for_shape_analysis.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>

namespace torch {
namespace jit {

static std::atomic<int64_t> idx(0);

bool CanonicalizeForShapeAnalysis(std::shared_ptr<Graph> graph) {

  DepthFirstGraphNodeIterator graph_it(graph);
  bool changed = false;

  for (auto next_node = graph_it.next(); next_node != nullptr;) {
    Node* node = next_node;
    next_node = graph_it.next();

    if (node->kind() == aten::slice && node->inputs().at(0)->node()->kind() == aten::size) {
      auto size_node = node->inputs().at(0)->node();
      auto tt = size_node->inputs().at(0)->type()->cast<TensorType>();
      if (!tt || !tt->symbolic_sizes().rank()) {
        continue;
      }
      changed = true;
      auto ten_inp = size_node->inputs().at(0);
      auto ss = tt->symbolic_sizes();
      auto rank = *ss.rank();
      std::vector<Value*> li_inps;
      WithInsertPoint g(size_node);
      for (size_t i = 0; i < rank; ++i) {
        auto inp = graph->insert(aten::size, {ten_inp, graph->insertConstant(static_cast<int64_t>(i))});
        li_inps.push_back(inp);
      }
      auto li_node = graph->insertNode(graph->createList(IntType::get(), li_inps));
      size_node->replaceAllUsesWith(li_node);
      size_node->destroy();
    } else if (node->kind() == prim::ListUnpack && node->inputs().at(0)->node()->kind() == aten::size) {
            auto size_node = node->inputs().at(0)->node();
      auto tt = size_node->inputs().at(0)->type()->cast<TensorType>();
      if (!tt || !tt->symbolic_sizes().rank()) {
        continue;
      }
      changed = true;
      auto ten_inp = size_node->inputs().at(0);
      auto ss = tt->symbolic_sizes();
      auto rank = *ss.rank();
      std::vector<Value*> li_inps;
      WithInsertPoint g(node);
      for (size_t i = 0; i < rank; ++i) {
        auto inp = graph->insert(aten::size, {ten_inp, graph->insertConstant(static_cast<int64_t>(i))});
        node->outputs().at(i)->replaceAllUsesWith(inp);
      }
      node->destroy();

    }
  }

  if (changed) {
    PeepholeOptimizeListIdioms(graph);
    EliminateDeadCode(graph);
  }

  return changed;
}


void MarkUniqueBlocksWithCounters(Block * b) {
  if (b->owningNode()) {
    WithInsertPoint block(*b->nodes().begin());
    auto g = b->owningGraph();
    auto node = g->insertNode(g->create(Symbol::prim("RunCounter"), {}, 0));
    node->i_(attr::value, {0});
    node->i_(attr::warn_id, idx);
    idx++;
  }
  for (Node * n: b->nodes()) {
    for (Block * block: n->blocks()) {
      MarkUniqueBlocksWithCounters(block);
    }
  }
}


void MarkUniqueBlocksWithCounters(std::shared_ptr<Graph> graph) {
  MarkUniqueBlocksWithCounters(graph->block());
}

} // namespace jit
} // namespace torch
