#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// return true if graph is modified
// Rewriting x.size()[x:] and a, b, c, d = x.size()
// into canonicalized form sym shape analysis can understand
// TODO: expand sym shape analysis more generally
TORCH_API bool CanonicalizeForShapeAnalysis(std::shared_ptr<Graph> graph);

} // namespace jit
} // namespace torch
