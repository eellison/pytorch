#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/autodiff.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/utils/memory.h>
#include <torch/csrc/jit/passes/constant_pooling.h>

// namespace torch {
// namespace jit {
//
// struct Functionalizor {
//
//   void run() {
//     Functionalize(graph_->block());
//     CreateFunctionalBlocks(graph_->block());
//   }
//
//   Functionalizor(std::shared_ptr<Graph> graph)
//       : graph_(std::move(graph)) {
//     aliasDb_ = torch::make_unique<AliasDb>(graph_);
//   }
//
// private:
//   void Functionalize(at::ArrayRef<Block*> blocks) {
//     for (Block* block : blocks) {
//       Functionalize(block);
//     }
//   }
//
//   void mergeNodeIntoFunctionalBlock(Node * functional_node, Node * merge_node) {
//     merge_node->moveAfter(functional_node->blocks().at(0)->param_node());
//   }
//
//   void CreateFunctionalBlocks(Block * block) {
//     WithInsertPoint insert(block->return_node());
//     auto reverse_iter = block->nodes().reverse();
//
//     Node * last_functional_node = graph_->insertNode(graph_->createWithSubgraph(prim::FunctionalBlock));
//     std::vector<std::pair<Node *, std::vector<Node *>>> functional_block_sets;
//     std::vector<Node *> contained_nodes;
//     Node * moveBeforePoint = nullptr;
//
//     for (auto it = reverse_iter.begin(); it != reverse_iter.end();) {
//       Node* n = *it++;
//       // close over constants
//       if (n->kind() == prim::FunctionalBlock || n->kind() == prim::Constant) {
//         continue;
//       }
//
//       if (functional_nodes_.count(n)) {
//         if (aliasDb_->couldMoveBeforeTopologically(n, moveBeforePoint)) {
//           moveBeforePoint = n;
//           contained_nodes.push_back(n);
//         } else {
//           functional_block_sets.emplace_back(last_functional_node, contained_nodes);
//           contained_nodes = {};
//           last_functional_node = graph_->createWithSubgraph(prim::FunctionalBlock)->insertAfter(n);
//           moveBeforePoint = n;
//         }
//       }
//       // TODO: If and loop blocks
//       //  else {
//       //   CreateFunctionalBlocks(block);
//       // }
//     }
//
//     functional_block_sets.emplace_back(last_functional_node, contained_nodes);
//     for (size_t i = 0; i < functional_block_sets.size(); ++i) {
//       Node * functional_node;
//       std::vector<Node *> contained_nodes;
//       std::tie(functional_node, contained_nodes) = functional_block_sets[i];
//       for (Node * n: contained_nodes) {
//         SubgraphUtils::mergeNodeIntoSubgraph(n, functional_node);
//       }
//     }
//   }
//
//   bool Functionalize(Node* n) {
//     bool functional_outputs = true;
//     for (Value * v: n->outputs()) {
//       if (!aliasDb_->hasWriters(v) && !aliasDb_->escapesScope({v})) {
//         functional_values_.insert(v);
//       } else {
//         functional_outputs = false;
//       }
//     }
//     bool functional_blocks = true;
//     for (Block * block: n->blocks()) {
//       functional_blocks = functional_blocks && Functionalize(block);
//     }
//     auto inputs = n->inputs();
//     bool functional_inputs = std::all_of(inputs.begin(), inputs.end(), [&](Value* v) {
//       return functional_values_.count(v);
//     });
//     if (functional_outputs && functional_blocks && functional_inputs) {
//       functional_nodes_.insert(n);
//       return true;
//     } else if (functional_outputs) {
//       functional_producer_.insert(n);
//     } else if (functional_inputs) {
//       functional_consumer_.insert(n);
//     }
//     return false;
//   }
//
//   bool Functionalize(Block* block) {
//     bool is_functional_block = true;
//     // block inputs will not yet have been iterated through
//     for (Value* v : block->inputs()) {
//       if (!aliasDb_->hasWriters(v) && !aliasDb_->escapesScope({v})) {
//         functional_values_.insert(v);
//       } else {
//         is_functional_block = false;
//       }
//     }
//     for (Node* n : block->nodes()) {
//       is_functional_block = is_functional_block && Functionalize(n);
//     }
//     is_functional_block = is_functional_block &&
//         std::all_of(block->outputs().begin(),
//                     block->outputs().end(),
//                     [&](Value* v) { return functional_values_.count(v); });
//     return is_functional_block;
//   }
//
//   std::unordered_set<Node *> functional_consumer_;
//   std::unordered_set<Node *> functional_producer_;
//   std::unordered_set<Node *> functional_nodes_;
//   std::unordered_set<Value *> functional_values_;
//   std::shared_ptr<Graph> graph_;
//   std::unique_ptr<AliasDb> aliasDb_;
// };
//
//
// void CreateFunctionalBlocks(
//     const std::shared_ptr<Graph>& graph) {
//   ConstantPooling(graph);
//   Functionalizor func(graph);
//   func.run();
// }
//
// } // namespace jit
// } // namespace torch
