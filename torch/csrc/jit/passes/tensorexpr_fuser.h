#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <memory>

namespace torch {
namespace jit {

struct Graph;

// Run TensorExpressions-based fuser.
//
// If shape checks are disabled it is the responsibilty of
// the caller to ensure that the resultant subgraph is correctly
// annotated with shapes by the time "getOperation" is called
// on the node.
TORCH_API void FuseTensorExprs(
    std::shared_ptr<Graph>& graph,
    size_t min_group_size = 2,
    bool disable_shape_checks = false);

TORCH_API void setTensorExprFuserEnabled(bool val);
TORCH_API bool tensorExprFuserEnabled();
TORCH_API bool setTexprReductionsEnabled(bool value);
TORCH_API bool texprReductionsEnabled();

namespace tensorexpr {
TORCH_API bool isSupported(Node* node);
}
} // namespace jit
} // namespace torch
