#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>
#include <memory>

namespace torch {
namespace jit {

TORCH_API void PropagateShapesWithShapeFunction(Node *n, const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
