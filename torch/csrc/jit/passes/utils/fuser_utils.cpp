#include <torch/csrc/jit/passes/utils/fuser_utils.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

namespace torch {
namespace jit {
namespace FuserUtils {
namespace {} // namespace

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

} // namespace FuserUtils
} // namespace jit
} // namespace torch
