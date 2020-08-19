// #include <torch/csrc/jit/passes/update_differentiable_graph_grad_property.h>
// #include <torch/csrc/jit/ir/ir.h>
// #include <torch/csrc/jit/passes/utils/subgraph_utils.h>

// namespace torch {
// namespace jit {

// namespace {

// void InlineAutodiffSubgraphs(Block* block, size_t threshold);

// void UpdateDifferentiableGraph(Node * node) {
//     auto subgraph = node->g(attr::Subgraph);
//     for (Node * n: subgraph->block()->nodes()) {
//         if (n->kind() == prim::profile) {
//             // n->ty_(attr::profiled_type, n->ty(attr::profiled_type)->expect<TensorType>->
//         }
//     }
// }


// void findDifferentiableNodes(Block * block, std::vector<Node*>& diff_nodes) {
//     for (Node * n: block->nodes()) {
//         for (Block * b: n->blocks()) {
//             findDifferentiableNodes(b, diff_nodes);
//         }

//         if (n->kind() == prim::DifferentiableGraph) {
//             diff_nodes.push_back(n);
//         }
//     }
// }

// } // anonymous namespace

// void UpdateDiffGraphRequiresGrad(std::shared_ptr<Graph>& graph) {
//     std::vector<Node*> diff_nodes;
//     findDifferentiableNodes(graph->block(), diff_nodes);
//     for (Node * n: diff_nodes) {
//         UpdateDifferentiableGraph(n);
//     }
// }

// } // namespace jit
// } // namespace torch
