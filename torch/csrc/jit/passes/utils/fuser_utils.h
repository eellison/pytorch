#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Utilities for passes fusing a subset of nodes into a common subgraph.
namespace FuserUtils {

TORCH_API Node* replaceBlockWithFallbackGraph(
    Block* b,
    ArrayRef<Value*> inputs);

TORCH_API void RemoveTensorTypeSpecializations(std::shared_ptr<Graph>& graph);

TORCH_API void RemoveProfileNodesAndSpecializeTypes(
    std::shared_ptr<Graph>& graph);

TORCH_API void guardFusionGroup(Node* fusion_group);

} // namespace FuserUtils
} // namespace jit
} // namespace torch
