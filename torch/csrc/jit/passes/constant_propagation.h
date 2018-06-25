#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

void ConstantPropagation(std::shared_ptr<Graph>& graph);
void ConstantPropagation(Block* block, bool recurse);
void ConstantPropagation(Node* n, bool recurse);

}}