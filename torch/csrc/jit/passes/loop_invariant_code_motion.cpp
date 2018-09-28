#include "torch/csrc/jit/passes/loop_invariant_code_motion.h"
#include <set>
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/ivalue.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/utils/functional.h"

namespace torch {
namespace jit {

namespace {

std::unordered_set<Symbol> side_effect = {
    prim::Print,
    prim::PythonOp, // may have side effects
    // if & loop may may contain ops with side effects
    prim::If,
    prim::Loop,
};

bool isConstantPositiveForLoop(Node* node) {
  Value* trip_count = node->inputs().at(0);
  int64_t iter_len = constant_as<int64_t>(trip_count).value_or(0);

  Value* start_cond = node->inputs().at(1);
  int64_t cond_val = constant_as<int64_t>(start_cond).value_or(0);

  return (cond_val != 0 && iter_len > 0);
}

Value* tripsCountComparison(
    Graph* g,
    Value* cur_trip_count,
    Value* max_trip_count) {
  return g->insertNode(g->create(aten::lt, {cur_trip_count, max_trip_count}, 1))
      ->output()
      ->setType(IntType::get());
}

// new_cond = (i < max_trip_count) && cond
Value* createTripCountConjunctiveCondition(
    Graph* g,
    Value* cur_trip_count,
    Value* max_trip_count,
    Value* cond) {
  // Emit initial comparison -- initial_trip_count < max_trip_count
  Value* initial_comparison_value =
      tripsCountComparison(g, cur_trip_count, max_trip_count);

  // Replace initial condition with logical `and` of trip count and
  // initial condition
  Value* new_cond =
      g->insertNode(
           g->create(aten::__and__, {initial_comparison_value, cond}, 1))
          ->output()
          ->setType(IntType::get());
  return new_cond;
}

void eraseLoopInputOutput(Node* loop_node, size_t loop_body_index) {
  loop_node->eraseOutput(loop_body_index - 1);
  loop_node->removeInput(loop_body_index + 1);

  loop_node->blocks().at(0)->eraseInput(loop_body_index);
  loop_node->blocks().at(0)->eraseOutput(loop_body_index);
}

void initialLoopCondition(Node* loop_node) {
  WithInsertPoint insert(loop_node);
  auto graph = loop_node->owningGraph();

  Value* trip_count = loop_node->inputs().at(0);
  int64_t iter_len = constant_as<int64_t>(trip_count).value_or(0);

  Value* start_cond = loop_node->inputs().at(1);
  int64_t cond_val = constant_as<int64_t>(start_cond).value_or(0);

  Value* condition;

  if (cond_val != 0 && iter_len > 0) {
    return;
  } else if (iter_len > 0) {
    condition = start_cond;
  } else if (cond_val != 0) {
    Value* zero_val = graph->insertConstant(0);
    condition = tripsCountComparison(graph, zero_val, trip_count);
  } else {
    // currently cannot happen, including for future compatibility
    Value* zero_val = graph->insertConstant(0);
    condition = createTripCountConjunctiveCondition(
        graph, zero_val, trip_count, start_cond);
  }

  Node* if_node = graph->insertNode(graph->create(prim::If, 0));
  auto* true_block = if_node->addBlock();
  auto* false_block = if_node->addBlock();

  if_node->addInput(condition);

  //replace all loop_carried_outputs with the outputs of the new if node
  //set the false block outputs as the loop_carried_inputs
  for (size_t i = 0; i < loop_node->outputs().size(); i++) {
    auto loop_output = loop_node->outputs().at(i);
    auto if_output = if_node->addOutput()->setType(loop_output->type());
    if (loop_output->hasUniqueName())
      if_output->setUniqueName(loop_output->uniqueName());
    loop_output->replaceAllUsesWith(if_output);
    true_block->registerOutput(loop_output);
    false_block->registerOutput(loop_node->inputs().at(i + 2));
  }

  loop_node->moveAfter(true_block->nodes().front());
}

// Nodes are loop-invariant if all of their inputs are loop_invariants
std::vector<Node*> calculateLoopInvariants(
    Block* loop_body,
    std::unordered_set<Value*>& loop_written_values,
    std::unordered_map<Value*, size_t>& loop_usages) {
  std::vector<Node*> loop_invariants;
  for (auto n : loop_body->nodes()) {
    // non-idempotent ops, ops with side effects
    if (n->isNondeterministic() || side_effect.count(n->kind()))
      continue;

    // all inputs of the node do not change on each iteration
    bool loop_invariant_inputs =
        std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
          return loop_written_values.count(v) == 0;
        });

    if (loop_invariant_inputs) {
      loop_invariants.push_back(n);
      for (auto input : n->inputs())
        loop_usages[input]--;
      for (auto output : n->outputs())
        loop_written_values.erase(output);
    }
  }

  return loop_invariants;
}

// Calculate which nodes can be sunk after the loop node
// A node may be moved if all of its outputs are not used within the loop, and
// it does not have side effects.
// A value is not used within the loop if it is not an input to any node within
// the loop. If the value and if it is a loop carried dependency, the carried
// dependency is not an input to any node. If an input to a sink node is
// loop-scoped it is later added as a loop carried depdency.
std::vector<Node*> calculateSinkNodes(
    Block* loop_body,
    const std::set<Node*>& loop_invariants,
    std::unordered_set<Value*>& loop_written_values,
    std::unordered_map<Value*, size_t>& loop_usages) {
  std::unordered_set<Value*> loop_carried_output_usages;

  //Note: a value may be used in multiple indices of loop_body outputs
  for (size_t i = 1; i < loop_body->outputs().size(); i++)
    if (loop_usages[loop_body->inputs().at(i)] != 0)
      loop_carried_output_usages.insert(loop_body->outputs().at(i));

  std::vector<Node*> sink_nodes;
  for (auto n : loop_body->nodes().reverse()) {
    // note: non-deterministic ops are not moved for model reproducibility
    if (loop_invariants.count(n) != 0 || side_effect.count(n->kind()) != 0 ||
        n->isNondeterministic())
      continue;

    bool loop_outputs_not_used =
        std::all_of(n->outputs().begin(), n->outputs().end(), [&](Value* v) {
          if (loop_usages[v] != 0)
            return false;
          return loop_carried_output_usages.count(v) == 0;
        });

    if (loop_outputs_not_used) {
      sink_nodes.push_back(n);
      for (auto input : n->inputs())
        loop_usages[input]--;
      for (auto output : n->outputs())
        loop_written_values.erase(output);
    }
  }
  return sink_nodes;
}

Value* materializeUndefinedType(
    std::unordered_map<TypePtr, Value*>& map,
    Value* v) {
  auto type = unshapedType(v->type());
  auto existing_type = map.find(type);
  if (existing_type != map.end()) {
    return existing_type->second;
  }

  auto graph = v->owningGraph();
  WithInsertPoint guard(graph->block()->nodes().front());

  Node* placeholder = graph->insertNode(graph->createUndefined());
  auto val = placeholder->output()->setType(type);

  map[type] = val;
  return val;
}

// Make a value that is emitted within a loop loop_carried outputs
Value* addLoopScopedValueToLoopOutputs(
    Node* loop_node,
    Value* loop_value,
    std::unordered_map<TypePtr, Value*>& map) {
  auto loop_body = loop_node->blocks().at(0);

  auto typ = loop_value->type();
  auto typed_undefined_val = materializeUndefinedType(map, loop_value);

  loop_node->addInput(typed_undefined_val);
  auto loop_output = loop_node->addOutput()->setType(loop_value->type());
  if (loop_value->hasUniqueName())
    loop_output->setUniqueName(loop_value->uniqueName());

  loop_body->addInput()->setType(loop_value->type());
  loop_body->registerOutput(loop_value);

  return loop_output;
}

void sinkNodesAfterLoop(
    Node* loop_node,
    const std::vector<Node*> sink_nodes,
    const std::unordered_set<Value*>& loop_written_values,
    const std::unordered_map<Value*, size_t>& loop_usages,
    std::unordered_map<TypePtr, Value*>& map) {
  Block* loop_body = loop_node->blocks().at(0);

  std::unordered_map<Value*, size_t> loop_carried_outputs;
  for (size_t i = 1; i < loop_body->outputs().size(); i++)
    loop_carried_outputs[loop_body->outputs()[i]] = i;
  std::unordered_map<Value*, size_t> loop_carried_inputs;
  for (size_t i = 1; i < loop_body->inputs().size(); i++)
    loop_carried_inputs[loop_body->inputs()[i]] = i;

  for (auto it = sink_nodes.begin(); it != sink_nodes.end(); ++it) {
    Node* n = *it;
    n->moveAfter(loop_node);

    for (size_t i = 0; i < n->inputs().size(); i++) {
      auto input = n->inputs()[i];
      auto index = -1;
      if (loop_carried_outputs.count(input)) {
        index = loop_carried_outputs[input];
      } else if (loop_carried_inputs.count(input)) {
        index = loop_carried_inputs[input];
      }
      if (index != -1) {
        // input is a loop-carried value, replace with the output
        n->replaceInput(i, loop_node->outputs().at(index - 1));
      } else if (loop_written_values.count(n->inputs()[i])) {
        // XXX: node has an input a loop-scoped value. make the loop-scoped
        // value an output of the loop, and replace the  input with the output
        // of the loop. could alternatively not sink the node
        // This also handles the case where the iter value is an input
        auto new_val =
            addLoopScopedValueToLoopOutputs(loop_node, n->inputs()[i], map);

        // the new output is appended to the end, so it is the last index
        auto index = loop_body->inputs().size() - 1;
        loop_carried_inputs[loop_body->inputs().at(index)] = index;
        loop_carried_outputs[loop_body->outputs().at(index)] = index;

        n->replaceInput(i, new_val);
      }
    }
    // outputs of v that are loop_carried_block_outputs are handled in
    // updateLoopCarriedDep
  }
}

// Determine whether the loop should be hoisted
// TODO: more complicated heuristic, and incorporate the net difference of added
// loop-carried dependencies that result from a sunk node having a loop-scoped
// input, or a loop-invariant node no longer being a loop-carried dependency
bool shouldHoistLoop(
    std::vector<Node*> loop_invariants,
    std::vector<Node*> sink_nodes) {
  // TODO: remove filter, improve constant pooling so that constants are not
  // emmitted within loops
  return filter(
             loop_invariants,
             [](Node* n) {
               return n->outputs().size() != 1 ||
                   toIValue(n->output()) == at::nullopt;
             })
             .size() != 0 ||
      sink_nodes.size() != 0;
}

/*
Scans through all inputs & outputs to a loop node and removes an input/output
if the value is not written to in the loop, and the loop-carried value is not
used. The second condition can occur in the following example:

for i in range(x):
    print(b)
    b = 2

%b.3 : int = prim::Loop(%x, %3, %b.1)
  block0(%i : int, %5 : int) {
     = prim::Print(%5)
    -> (%7, %b.2)
  }
-> (%b.3)

b.2 = prim::Constant[value=2]() is a loop-invariant instruction, so it gets
hoisted above the loop. The value no longer written-to in the loop, but it
needs to be remain an input-output because the loop-carried value is used.

XXX: can only be run on a loop that is guaranteed to execute. For example:
for i in range(x):
    b = 2
    print(b)
b = 2 is loop-invariant, and the loop-carried value of b is not used.
Replacing all outputs of the loop node with b = 2 is only valid if the loop
executes.

Also handles nodes which are sunk after the loop
block0(%i : int, %9 : int, %10 : int) {
  %b.2 : int = aten::mul(%i, %7)
  -> (%11, %b.2, %b.2)
}
b.2 will be sunk after the loop because its output is not used. After it is sunk,
the loop outputs will be replaced and then removed
*/
void updateLoopCarriedDep(
    Node* loop_node,
    const std::unordered_set<Value*>& loop_written_values,
    const std::unordered_map<Value*, size_t>& loop_usages) {
  Block* loop_body = loop_node->blocks().at(0);
  std::vector<size_t> removed_indices;

  for (size_t i = 1; i < loop_body->inputs().size();) {
    auto input = loop_body->inputs()[i];
    auto output = loop_body->outputs().at(i);

    // loop_carried value not written to during loop
    // the value must be scoped outside of the loop
    bool no_dep = loop_written_values.count(output) == 0;

    // the input cannot be used because that means the loop input value was
    // used on first iteration and then written to
    // constants may be emitted outside of the loop so they are not written to
    //
    // replace with == uses = 0 - number of uses in return statement
    // : only use is in the return node
    auto element = loop_usages.find(input);
    bool no_uses = element == loop_usages.end() || element->second == 0;

    if (no_uses && no_dep) {
      loop_node->outputs().at(i - 1)->replaceAllUsesWith(output);
      eraseLoopInputOutput(loop_node, i);
    } else {
      i++;
    }
  }
}

} // namespace

std::unordered_map<Value*, size_t> loopHoistCode(
    Block* loop_body,
    std::unordered_map<TypePtr, Value*>& map) {
  std::unordered_map<Value*, size_t> loop_usages;
  for (auto it = loop_body->nodes().begin(); it != loop_body->nodes().end();) {
    Node* n = *it;
    it++; // advance iterator bc block may be mutated
    for (auto input : n->inputs())
      loop_usages[input]++;
    if (n->kind() == prim::Loop) {
      auto node_uses = nodeHoistCode(n, map);
      for (auto kv : node_uses) {
        loop_usages[kv.first] += kv.second;
      }
    } else if (n->kind() == prim::If) {
      for (Block* b : n->blocks()) {
        auto block_uses = loopHoistCode(b, map);
        for (auto kv : block_uses) {
          loop_usages[kv.first] += kv.second;
        }
      }
    }
  }
  return loop_usages;
}

std::unordered_map<Value*, size_t>& prepareUsesForReturn(
    std::unordered_map<Value*, size_t>& uses,
    Block* loop_body,
    std::vector<Node*> loop_invariants,
    std::vector<Node*> sink_nodes,
    bool was_lifted_to_if) {

  // uses need to be copied at each level of recursion, so filter 0s
  for (auto it = uses.begin(); it != uses.end();)
    if (it->second == 0)
      it = uses.erase(it);
    else
      ++it;

  // add block carried dependencies
  for (auto output : loop_body->outputs())
    uses[output]++;

  // loop invariants & sink nodes uses were removed because their inputs were no
  // longer used within the loop block, add them back in here
  for (Node* n : loop_invariants)
    for (Value* input : n->inputs())
      uses[input]++;

  for (Node* n : sink_nodes)
    for (Value* input : n->inputs())
      uses[input]++;

  if (was_lifted_to_if) {
    auto while_node = loop_body->owningNode();
    auto if_node = while_node->owningBlock()->owningNode();
    JIT_ASSERT(if_node->kind() == prim::If);
    for (Value* input : if_node->inputs())
      uses[input]++;
    for (Block* b : if_node->blocks())
      for (auto output : b->outputs())
        uses[output]++;
  }
  return uses;
}

std::unordered_map<Value*, size_t> nodeHoistCode(
    Node* loop_node,
    std::unordered_map<TypePtr, Value*>& map) {
  JIT_ASSERT(loop_node->kind() == prim::Loop);
  Block* loop_body = loop_node->blocks().at(0);

  // count of value uses in loop
  std::unordered_map<Value*, size_t> loop_usages =
      loopHoistCode(loop_body, map);

  // the first output is used as the loop continuation condition
  // prevents output from being sunk out of the loop
  loop_usages[loop_body->outputs()[0]]++;

  std::unordered_set<Value*> loop_written_values; // values written to in loop
  for (auto n : loop_body->nodes()) {
    for (auto output : n->outputs())
      loop_written_values.insert(output);
  }
  for (auto v : loop_body->inputs())
    loop_written_values.insert(v);

  std::vector<Node*> loop_invariants =
      calculateLoopInvariants(loop_body, loop_written_values, loop_usages);

  std::set<Node*> loop_invariants_set(
      loop_invariants.begin(), loop_invariants.end());

  std::vector<Node*> sink_nodes = calculateSinkNodes(
      loop_body, loop_invariants_set, loop_written_values, loop_usages);

  // for loops with a constant trip count are guaranteed to execute
  // so we always attempt to optimize them
  bool constant_for_loop = isConstantPositiveForLoop(loop_node);
  if (!constant_for_loop) {
    if (!shouldHoistLoop(loop_invariants, sink_nodes)) {
      return prepareUsesForReturn(
          loop_usages, loop_body, loop_invariants, sink_nodes, false);
    }
    initialLoopCondition(loop_node);
  }

  sinkNodesAfterLoop(
      loop_node, sink_nodes, loop_written_values, loop_usages, map);

  for (Node* invariant : loop_invariants) {
    invariant->moveBefore(loop_node);
  }

  updateLoopCarriedDep(loop_node, loop_written_values, loop_usages);

  return prepareUsesForReturn(
      loop_usages,
      loop_body,
      loop_invariants,
      sink_nodes,
      /*hoisted to if node*/!constant_for_loop);
}

// namespace
void LoopInvariantCodeMotion(std::shared_ptr<Graph>& graph) {
  std::unordered_map<TypePtr, Value*> placeholder_map;
  loopHoistCode(graph->block(), placeholder_map);
  EliminateDeadCode(graph);
}

} // namespace jit
} // namespace torch
