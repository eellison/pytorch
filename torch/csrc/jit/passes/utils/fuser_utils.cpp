#include <torch/csrc/jit/passes/utils/fuser_utils.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

namespace torch {
namespace jit {
namespace FuserUtils {
namespace {

void updateFusionProfiledTypes(Node* fusion_group) {
  for (Value* v : fusion_group->outputs()) {
    v->setType(v->type()->expect<TensorType>()->withProfiledType(false));
  }

  auto update_value_list = [](at::ArrayRef<Value*> values) {
    for (Value* v : values) {
      if (auto tensor = v->type()->cast<TensorType>()) {
        v->setType(tensor->withProfiledType(false));
      }
    }
  };

  auto subgraph = SubgraphUtils::getSubgraph(fusion_group);
  update_value_list(subgraph->inputs());
  for (Node* n : subgraph->nodes()) {
    update_value_list(n->outputs());
  }
}

} // namespace

Node* replaceBlockWithFallbackGraph(Block* b, ArrayRef<Value*> inputs) {
  auto graph = std::make_shared<Graph>();

  // we are copying the block inside If or prim::Loop otherwise we are copying
  // the whole graph we need to differentiate the two cases  because cloneFrom
  // automatically adds inputs if we are copying graph's block and we will
  //  need the inputs from a user otherwise
  if (b->owningNode() != nullptr) {
    std::unordered_map<Value*, Value*> input_mapping;
    auto value_map = [&input_mapping](Value* v) { return input_mapping[v]; };
    for (auto inp : inputs) {
      input_mapping[inp] = graph->block()->addInput();
    }
    graph->block()->cloneFrom(b, value_map);
  } else {
    auto value_map = [](Value* v) { return v; };
    graph->block()->cloneFrom(b, value_map);
  }

  auto fallback = b->owningGraph()->create(
      prim::FallbackGraph, inputs, b->outputs().size());
  fallback->g_(attr::Subgraph, graph);
  b->prependNode(fallback);

  for (size_t i = 0; i < inputs.size(); i++) {
    graph->inputs()[i]->setType(inputs[i]->type());
    graph->inputs()[i]->copyMetadata(inputs[i]);
  }

  for (size_t i = 0; i < b->outputs().size(); i++) {
    fallback->output(i)->setType(b->outputs()[i]->type());
    fallback->output(i)->copyMetadata(b->outputs()[i]);
    b->replaceOutput(i, fallback->output(i));
  }

  ProfilingRecord::removeProfilingNodes(graph->block());

  for (auto it = b->nodes().rbegin(); it != fallback->iterator(); it++) {
    it.destroyCurrent();
  }

  RemoveTensorTypeSpecializations(graph);

  return fallback;
}

void removeTensorTypeSpecialization(Value* v) {
  if (!v->type()->cast<TensorType>()) {
    return;
  }
  // Constants & TensorExprGroup will always produce specialized tensor type,
  // TypeCheck are inserted by this pass and only used by fusion groups that
  // insert proper guards
  if (v->node()->kind() == prim::Constant ||
      v->node()->kind() == prim::TypeCheck ||
      v->node()->kind() == prim::TensorExprGroup) {
    return;
  }
  v->setType(TensorType::get());
}

void removeTensorTypeSpecializations(Block* block) {
  for (Value* v : block->inputs()) {
    removeTensorTypeSpecialization(v);
  }
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      removeTensorTypeSpecializations(b);
    }
    for (Value* v : n->outputs()) {
      removeTensorTypeSpecialization(v);
    }
  }
}

void RemoveTensorTypeSpecializations(std::shared_ptr<Graph>& graph) {
  removeTensorTypeSpecializations(graph->block());
}

void eraseProfiledTypes(Value* v) {
  if (auto ten_type = v->type()->cast<TensorType>()) {
    if (ten_type->isProfiled()) {
      v->setType(TensorType::get());
    }
  }
}

void eraseProfiledTypes(Block* block) {
  for (Value* v : block->inputs()) {
    eraseProfiledTypes(v);
  }
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      eraseProfiledTypes(b);
    }
    for (Value* v : n->outputs()) {
      removeTensorTypeSpecialization(v);
    }
  }
}

void EraseProfiledTypes(std::shared_ptr<Graph>& graph) {
  eraseProfiledTypes(graph->block());
}

// TODO: if a value has differently typed uses, temporarrily insert a node
// specializing the type for each use and later remove, instead of bailing
bool profiledWithDifferentTypes(Value* v) {
  std::vector<TypePtr> types;
  for (const auto& use : v->uses()) {
    if (use.user->kind() == prim::profile) {
      types.push_back(use.user->ty(attr::profiled_type));
    }
  }
  for (size_t i = 1; i < types.size(); ++i) {
    if (types.at(i - 1) != types.at(i)) {
      return true;
    }
  }
  return false;
}

void removeProfileNodesAndSpecializeTypes(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::profile) {
      GRAPH_DEBUG("Removing prim::profile: %", it->output()->debugName());
      it->output()->replaceAllUsesWith(it->input());
      if (!profiledWithDifferentTypes(it->input())) {
        it->input()->setType(it->ty(attr::profiled_type)
                                 ->expect<TensorType>()
                                 ->withProfiledType(true));
      } else {
        GRAPH_DEBUG(
            "Ignoring value with differently typed profiles :%",
            it->output()->debugName());
      }
      it.destroyCurrent();
    } else {
      for (Block* ib : it->blocks()) {
        removeProfileNodesAndSpecializeTypes(ib);
      }
    }
  }
}

void RemoveProfileNodesAndSpecializeTypes(std::shared_ptr<Graph>& graph) {
  removeProfileNodesAndSpecializeTypes(graph->block());
}

void guardFusionGroup(Node* fusion_group) {
  GRAPH_DEBUG("Inserting a typecheck guard for a node", *fusion_group);
  auto subgraph = SubgraphUtils::getSubgraph(fusion_group);

  // Fixup types of the subgraph inputs
  std::vector<Value*> inputs_to_check;
  for (Value* input : fusion_group->inputs()) {
    // We only check inputs of the fusion group and expect NNC to infer
    // intermediates and outputs shapes
    if (!input->type()->cast<TensorType>()) {
      continue;
    }

    // we only need to guard it if it's a profiled type
    if (input->type()->expect<TensorType>()->isProfiled()) {
      inputs_to_check.push_back(input);
    }
  }

  updateFusionProfiledTypes(fusion_group);

  if (!inputs_to_check.size()) {
    return;
  }

  // Add prim::TypeCheck node
  //
  // TypeCheck nodes  look like the following:
  //   %out1 : Float(2, 3), %out2 : Int(10, 30), %types_match : bool =
  //   prim::TypeCheck(%inp1 : Tensor, %inp2 : Tensor)
  //
  // They have N inputs whose types we are going to check and N+1 outputs. The
  // first N outputs specify expected types and N+1-th output holds the result
  // of the check (bool).
  Node* typecheck_node =
      fusion_group->owningGraph()
          ->create(prim::TypeCheck, inputs_to_check, inputs_to_check.size() + 1)
          ->insertBefore(fusion_group);
  Value* typecheck_result = typecheck_node->output(inputs_to_check.size());

  std::unordered_map<Value*, Value*> typechecked_inputs;
  for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
    typechecked_inputs[typecheck_node->input(i)] = typecheck_node->output(i);
  }

  // Fixup types of the typecheck node outputs, which are used by the op in
  // execution.
  typecheck_node->output(inputs_to_check.size())->setType(BoolType::get());
  for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
    typecheck_node->output(i)->setType(typecheck_node->input(i)
                                           ->type()
                                           ->expect<TensorType>()
                                           ->withProfiledType(false));
  }

  // Insert if
  auto versioning_if =
      fusion_group->owningGraph()
          ->create(prim::If, {typecheck_result}, fusion_group->outputs().size())
          ->insertAfter(typecheck_node);
  for (size_t idx = 0; idx < fusion_group->outputs().size(); ++idx) {
    versioning_if->output(idx)->setType(
        unshapedType(fusion_group->output(idx)->type()));
    fusion_group->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
  }
  auto true_block = versioning_if->addBlock();
  auto false_block = versioning_if->addBlock();

  // Fill in the false block. It should contain the unoptimized
  // copy of the fused subgraph.
  WithInsertPoint guard(false_block->return_node());
  const auto subgraph_outputs = insertGraph(
      *fusion_group->owningGraph(), *subgraph, fusion_group->inputs());
  for (Value* output : subgraph_outputs) {
    false_block->registerOutput(output);
  }

  FuserUtils::replaceBlockWithFallbackGraph(
      false_block, fusion_group->inputs());

  // Fill in the true block. It has all inputs type-checked and its
  // body should be the fusion group node.
  fusion_group->moveBefore(true_block->return_node());
  for (size_t idx = 0; idx < fusion_group->inputs().size(); ++idx) {
    if (typechecked_inputs.count(fusion_group->input(idx))) {
      fusion_group->replaceInput(
          idx, typechecked_inputs.at(fusion_group->input(idx)));
    }
  }
  for (Value* output : fusion_group->outputs()) {
    true_block->registerOutput(output);
  }
}

} // namespace FuserUtils
} // namespace jit
} // namespace torch
