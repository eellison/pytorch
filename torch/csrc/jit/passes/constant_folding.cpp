#include "torch/csrc/jit/passes/constant_folding.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/jit/interpreter.h"

namespace torch { namespace jit {

std::vector<at::Tensor> runNode(const Node *n) {
  auto new_g = std::make_shared<Graph>();
  auto new_block = new_g->block();
  std::unordered_map<Value*, Value*> local_map;
  auto env = [&](Value * v) {
    auto it = local_map.find(v);
    if(it != local_map.end())
      return it->second;
    barf("Constant Folding encountered a use of a value not in scope");
  };

  for(auto input : n->inputs()) {
    local_map[input] = new_block->addInput()->copyMetadata(input)->setStage(input->stage());
    new_g->setStage(std::max(new_g->stage(), input->stage()));
  }

  auto new_node = new_block->appendNode(new_g->createClone(n, env, /*non-recursive */ false));
  new_node->setStage(n->stage());
  new_g->setStage(std::max(graph->stage(), n->stage()));

  for(size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    local_map[oo] = no;
    no->copyMetadata(oo);
    no->setStage(oo->stage());
  }
  for(auto output : n->outputs()) {
    new_block->registerOutput(env(output));
  }
  std::vector<at::Tensor> outputs;
  runOneStage(InterpreterState(InterpreterState(Code(graph)), new_block->inputs(), outputs));
  return outputs;
}

void runOneStage(InterpreterState & interp, const std::vector<at::Tensor> & inputs, std::vector<at::Tensor> & outputs) {
  outputs = inputs;
  interp.runOneStage(outputs);
}

void propagateNode(Node *n) {
  auto outputs = runNode(n);
  for(size_t i = 0; i < n->outputs().size(); ++i) {
    n->outputs()[i]->replaceAllUsesWith(outputs[i]);
  }
}

void ConstantFolding(Node* n, bool recurse) {
  auto & graph = *n->owningGraph();
  auto all_constant_inputs = std::all_of(n->inputs().begin(), node->inputs().end(),
                [&](Value *v) {
                  v->node()->kind() == prim::Constant
                });
  if (all_constant_inputs && node->kind() != prim::Print) {
    propagateNode(n);
    for (Block * block : node->blocks())
      ConstantFolding(block, recurse, memo);
    // n->destroy();
  }
}

void ConstantFolding(Block* block, bool recurse) {
  for(auto it = block->nodes().begin(), end = block->nodes().end(); it != end;) {
    auto n = *it++;
    ConstantFolding(n, recurse);
  }
}

void ConstantFolding(std::shared_ptr<Graph>& graph) {
  ConstantFolding(graph->block(), true);
  // EliminateDeadCode(graph);
}

}}
