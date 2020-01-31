#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

// // insert GraphExecutor nodes that group together
// // subgraphs that are differentiable by the jit's autodiff passes
// // threshold - minimum number of nodes that will appear in a block
// // returns all differentiable blocks that have been found
TORCH_API void CreateFunctionalBlocks(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
