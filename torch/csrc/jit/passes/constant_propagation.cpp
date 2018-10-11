#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/ivalue.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/utils/functional.h"

namespace torch { namespace jit {

namespace {

//non deterministic ops handled in a separate condition
std::unordered_set<Symbol> skip_list = {
  prim::If,
  prim::Loop, //TODO: handle Loop
  prim::Print,
  prim::PythonOp, //may have side effects
  prim::LoadWorld,
  prim::StoreWorld,
  prim::Constant,
  prim::Undefined,
  prim::NoneGenerator,
  // TODO (zach): we should consider skipping tensor factories in the cases
  // where the constant tensor would be large but cheap to create.
 };

std::vector<IValue> runNode(Node* n) {
  auto op = getOperation(n);
  Stack stack;
  for (auto input : n->inputs()) {
    stack.push_back(*(toIValue(input)));
  }
  op(stack);
  auto var_outputs = fmap(stack, [&](IValue v) -> IValue {
    if (v.isTensor()) {
      auto t = std::move(v).toTensor();
      if(t.defined()) {
        return IValue(autograd::as_variable_ref(t).data());
      } else {
        return t;
      }
    } else {
      return v;
    }
  });
  return var_outputs;
}

void propagateNode(Node* n) {
  auto outputs = runNode(n);
  auto graph = n->owningGraph();
  WithInsertPoint guard(n);
  for (size_t i = 0; i < outputs.size(); ++i) {
    try {
      auto new_output = graph->insertConstant(outputs[i]);
      n->outputs()[i]->replaceAllUsesWith(new_output);
    } catch(constant_not_supported_error& err) {
      // we cannot actually represent the IValue as a constant node,
      // so we give up replacing it
    }
    // let dce elimination remove n
  }
}

void removeLoopNode(Node * n) {
  for (size_t i = 0; i < n->outputs().size(); ++i) {
    n->outputs().at(i)->replaceAllUsesWith(n->inputs().at(i - 2));
  }
  n->destroy();
}

void inlineIf(Block *body, Node * n) {
  for(auto it = body->nodes().begin(); it != body->nodes().end();) {
    Node *body_node = *it;
    //advance iterator because after body_node is moved its next pointer will be
    //to n
    it++;
    body_node->moveBefore(n);
  }
  for (size_t i = 0; i < n->outputs().size(); ++i) {
    n->outputs().at(i)->replaceAllUsesWith(body->outputs().at(i));
  }
  // NB: destroy the node here, because it might contain side effects, like print
  n->destroy();
}

bool isTrueConstant(Value *val) {
  at::optional<bool> maybe_value = constant_as<bool>(val);
  JIT_ASSERT(maybe_value);
  return *maybe_value;
}

bool loopWillNotRun(Node* node) {
  Value* trip_count = node->inputs().at(0);
  int64_t iter_len = constant_as<int64_t>(trip_count).value_or(1);

  Value* start_cond = node->inputs().at(1);
  int64_t cond_val = constant_as<int64_t>(start_cond).value_or(1);

  return (cond_val == 0 && iter_len <= 0);
}


void inlineIf(Node *n) {
  if (isTrueConstant(n->input())) {
    inlineIf(n->blocks()[0], n);
  } else {
    inlineIf(n->blocks()[1], n);
  }
}

//remove extra outputs from the node
bool removeExtraLoopOutputs(Node *n) {
  JIT_ASSERTM(n->kind() == prim::Loop, "Only supported for If nodes");
  auto block = n->blocks()[0];
  for (size_t i = 0; i < n->outputs().size(); ) {
    if (block->outputs().at(i + 1) == n->inputs().at(i - 1)) {
      block->inputs().at(i + 1)->replaceAllUsesWith(n->inputs().at(i - 1));

      n->eraseOutput(i);
      n->removeInput(i + 2);

      block->eraseInput(i + 1);
      block->eraseOutput(i + 1);
    } else {
      ++i;
    }
  }


  auto true_block = n->blocks()[0];
  auto false_block = n->blocks()[1];
  auto initial_outputs = true_block->outputs().size();
  for (size_t i = 0; i < true_block->outputs().size(); ) {
    //neither block changes the output value
    if (true_block->outputs()[i] == false_block->outputs()[i]) {
      n->outputs().at(i)->replaceAllUsesWith(true_block->outputs()[i]);
      n->eraseOutput(i);
      true_block->eraseOutput(i);
      false_block->eraseOutput(i);
    } else {
      i++; //increment bc we didn't remove current index
    }
  }
  //an output was removed
  return initial_outputs != true_block->outputs().size();
}

//remove extra outputs from the node
bool removeExtraNodeOutputs(Node *n) {
  JIT_ASSERTM(n->kind() == prim::If, "Only supported for If nodes");
  auto true_block = n->blocks()[0];
  auto false_block = n->blocks()[1];
  auto initial_outputs = true_block->outputs().size();
  for (size_t i = 0; i < true_block->outputs().size(); ) {
    //neither block changes the output value
    if (true_block->outputs()[i] == false_block->outputs()[i]) {
      n->outputs().at(i)->replaceAllUsesWith(true_block->outputs()[i]);
      n->eraseOutput(i);
      true_block->eraseOutput(i);
      false_block->eraseOutput(i);
    } else {
      i++; //increment bc we didn't remove current index
    }
  }
  //an output was removed
  return initial_outputs != true_block->outputs().size();
}

} // anonymous namespace

void ConstantPropagation(Node* n, bool recurse) {
  bool constant_inputs =
      std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
        return v->node()->kind() == prim::Constant;
      });
  bool supported_node = !n->kind().is_onnx() && !n->isNondeterministic() &&
    skip_list.count(n->kind()) == 0;
  auto run_blocks = [&]() {
    if (recurse) {
      for (Block * block : n->blocks()) {
        ConstantPropagation(block, recurse);
      }
    }
  };
  if (n->kind() == prim::If) {
    run_blocks();
    //inline node if we can, otherwise check for simplified outputs
    if (constant_inputs) {
      inlineIf(n);
    } else {
      removeExtraNodeOutputs(n);
    }
    //don't rerun run_blocks
    return;
  } else if (n->kind() == prim::Loop) {
    if (loopWillNotRun(n)) {
      removeLoopNode(n);
    } else {
      run_blocks();
      removeExtraLoopOutputs(n);
    }
    //don't rerun run_blocks
    return;
  } else if (constant_inputs && supported_node) {
    propagateNode(n);
  }
  run_blocks();
}

void ConstantPropagation(Block* block, bool recurse) {
  for(auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node *n = *it;
    it++; //advance iterator bc the current node may be destroyed
    ConstantPropagation(n, recurse);
  }
}

void ConstantPropagation(std::shared_ptr<Graph>& graph) {
  ConstantPropagation(graph->block(), true);
  EliminateDeadCode(graph);
}

}}
