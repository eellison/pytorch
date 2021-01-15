#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/exception_message.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/utils/memory.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/peephole.h>

#include <exception>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

c10::optional<size_t> normIndex(int64_t index, size_t len) {
  if (index < 0) {
    index = index + len;
  }
  if (index >= 0 && index < static_cast<int64_t>(len)) {
    return index;
  } else {
    return c10::nullopt;
  }
}

void PeepholeOptimizeExceptionBlocks(Block * b) {
    for (Node * n: b->nodes()) {
        for (Block * block: n->blocks()) {
            PeepholeOptimizeExceptionBlocks(block);
        }
        if (n->kind() == prim::If) {
            auto true_block = n->blocks().at(0);
            auto false_block = n->blocks().at(1);
            for (size_t i = 0; i < n->outputs().size(); ++i) {


            }
        }



    }



}


struct SymbolicShapeAnalyzer {
  void replaceWithIValue(Value* v, IValue val) {
    WithInsertPoint guard(*v->node()->owningBlock()->nodes().begin());
    v->replaceAllUsesWith(v->owningGraph()->insertConstant(val));
  }

  SymbolicShapeAnalyzer(std::shared_ptr<Graph> graph, Node * n)
      : graph_(graph->copy()) {
    for (size_t i = 0; i < n->inputs().size(); i++) {
        auto type = n->input(i)->type();
        if (auto tt = type->castRaw<TensorType>()) {
            c10::SymbolicShape symbolic_shapes = tt->symbolic_sizes();
            if (symbolic_shapes.isComplete()) {
                replaceWithIValue(graph_->inputs().at(i), *tt->sizes().concrete_sizes());
                continue;
            }
            tensor_indices.push_back(i);
            tensor_sizes_.push_back(symbolic_shapes);
        } else if (type->cast<ListType>() &&  type->cast<ListType>()->getElementType()->cast<TensorType>()) {
            TORCH_INTERNAL_ASSERT(false); // not handled yet
        } else {
            if (auto ival = toIValue(n->input(i))) {
                replaceWithIValue(n->input(i), *ival);
            }
        }
    }
  }

  c10::SymbolicShape run() {
      // TODO: if all inputs dont have uses (have been replaced with constant values)
      // just run graph
      // TODO Inliner
      for (size_t i = 0; i < 6; i++) {
        inputInputTensorProperties();
        LowerSimpleTuples(graph_);
        RemoveListMutation(graph_);
        UnrollConstantLoops(graph_);
        ConstantPropagation(graph_);
        PeepholeOptimize(graph_);
        ConstantPropagation(graph_);
      }
      graph_->dump();
      return extractOutputShape();
  }



 private:
  void inputTensorProperties(const c10::SymbolicShape& shape, const Use& use) {
      if (!shape.rank()) {
          return;
      }

      switch (use.user->kind()) {
          case aten::len: {
            replaceWithIValue(use.user->output(), static_cast<int64_t>(*shape.rank()));
          } break;
          case aten::__getitem__: {
              auto index = constant_as<int64_t>(use.user->inputs().at(1));
              if (index) {
                auto norm_index = normIndex(*index, *shape.rank());
                  // TODO: HANDLE non-static present value
                if (norm_index && shape[*norm_index].is_static()) {
                    replaceWithIValue(use.user->output(), shape[*norm_index].static_size());
                }
              }
          }
      }
  }

  c10::SymbolicShape extractOutputShape() {
      TORCH_INTERNAL_ASSERT(graph_->outputs().size() == 1);
      auto output = graph_->outputs().at(0);
      TORCH_INTERNAL_ASSERT(output->type()->cast<ListType>() && output->type()->cast<ListType>()->getElementType()->cast<IntType>());
      if (output->uses().size() != 1 && output->node()->kind() != prim::ListConstruct) {
          return c10::SymbolicShape();
      }
      Node * list_construct = output->node();
      std::vector<c10::optional<int64_t>> output_shape;
      for (Value *input: list_construct->inputs()) {
         output_shape.push_back(constant_as<int64_t>(input));
      }
      return c10::SymbolicShape(output_shape);
  }

  void inputInputTensorProperties() {
      for (auto index: tensor_indices) {
          auto value = graph_->inputs().at(index);
          for (const auto&  use: value->uses()) {
              inputTensorProperties(tensor_sizes_[index], use);
          }
      }
  }

  std::vector<int64_t> tensor_indices;
  std::vector<c10::SymbolicShape> tensor_sizes_;
  std::shared_ptr<Graph> graph_;
};

void PropagateShapesWithShapeFunction(Node *n, const std::shared_ptr<Graph>& graph) {
    c10::SymbolicShape out = SymbolicShapeAnalyzer(graph, n).run();
    n->output()->setType(n->output()->type()->expect<TensorType>()->withSymbolicShapes(out));
}


} // namespace jit
} // namespace torch
