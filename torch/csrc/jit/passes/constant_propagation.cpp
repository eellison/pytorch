#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/ivalue.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/variable_tensor_functions.h"
#include <ios>
#include <fstream>

namespace torch { namespace jit {

namespace {

std::unordered_set<Symbol> skip_list = {
  //FIXME If & Loop require special casing because they cannot be run as a
  //single node.
  prim::If,
  prim::Loop,
  //FIXME Same problem as in DCE - cpp & python PythonOp and CppOp should be
  //FIXME treated as having side effects but ONNX depends on them being removed
  prim::Print,
  //all the rand functions from native_functions.yaml
  aten::permute,
  aten::rand,
  aten::rand_out,
  aten::rand_like,
  aten::randint,
  aten::randint_out,
  aten::randint_like,
  aten::randn,
  aten::randn_out,
  aten::randn_like,
  aten::randperm,
  aten::randperm_out,
 };

std::unordered_set<Symbol> size_ops = {
  //not including random size ops
  aten::size,
  aten::zeros_like,
  aten::ones_like,
  aten::empty_like,
  aten::expand_as,
  aten::reshape_as,
  aten::reshape,
  aten::view_as,
  aten::empty_like,
  aten::full_like,
};

std::vector<IValue> runNode(Node* n) {
  auto op = getOperation(n);
  Stack stack;
  for (auto input : n->inputs()) {
    auto iv = *(toIValue(input));
    stack.push_back(iv);
  }
  op(stack);
  auto var_outputs = fmap(stack, [&](IValue v) {
    if (v.isTensor()) {
      return IValue(autograd::as_variable_ref(v.toTensor()).data());
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
    auto new_output = insertConstant(*graph, outputs[i]);
    n->outputs()[i]->replaceAllUsesWith(new_output);
    // let dce elimination remove n
  }
}

void handleSizeOps(Node *n) {
  WithInsertPoint guard(n);
  auto graph = n->owningGraph();
  for (size_t i = 0; i < n->inputs().size(); i++) {
    //don't need to
    if (n->inputs()[i]->node()->kind() == prim::Constant) {
      continue;
    }

    //already tested that all inputs are tensor types
    auto type = n->inputs()[i]->type()->cast<TensorType>();

    auto backend = type->device() == -1 ? at::kCPU : at::kCUDA;
    at::DeviceGuard device_guard(type->device());
    auto& attype = at::getType(backend, type->scalarType());
    auto zero = attype.tensor(type->sizes(), type->strides()).zero_();

    //dce will remove constant after use
    auto* constant = insertConstant(*graph, zero);
    n->replaceInput(i, constant);
  }
  propagateNode(n);
}


} // anonymous namespace

void ConstantPropagation(Node* n, bool recurse) {
  bool constant_inputs = (n->inputs().size() > 0) &&
    std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
      return v->node()->kind() == prim::Constant;
    });
  bool supported_node = skip_list.count(n->kind()) == 0;
  if (constant_inputs && supported_node) {
    propagateNode(n);
  } else if (size_ops.count(n->kind()) &&
    std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
      return v->node()->kind() == prim::Constant || v->type()->cast<TensorType>();
  })) {
    handleSizeOps(n);
  };
  if (recurse) {
    for (Block * block : n->blocks())
      ConstantPropagation(block, recurse);
  }
}

void ConstantPropagation(Block* block, bool recurse) {
  ConstantPropagation(block->param_node(), recurse);
  for (auto n: block->nodes()) {
    ConstantPropagation(n, recurse);
  }
}

void ConstantPropagation(std::shared_ptr<Graph>& graph) {
  ConstantPropagation(graph->block(), true);
  EliminateDeadCode(graph);
}

}}
