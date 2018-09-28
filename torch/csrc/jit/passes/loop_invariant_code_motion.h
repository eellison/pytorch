#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

std::unordered_map<Value*, size_t> loopHoistCode(Node* loop_node, std::unordered_map<TypePtr, Value*>& map);
std::unordered_map<Value*, size_t> nodeHoistCode(Node* loop_node, std::unordered_map<TypePtr, Value*>& map);
TORCH_API void LoopInvariantCodeMotion(std::shared_ptr<Graph>& graph);

}}
