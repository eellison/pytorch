#include <torch/csrc/jit/script/exit_transforms.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/script/error_report.h>

namespace torch {
namespace jit {

namespace {

void registerBlockOutputs(Block* b, at::ArrayRef<Value*> outs) {
  for (Value* out : outs) {
    b->registerOutput(out);
  }
}

Symbol owningNodeKind(Block* block) {
  if (block->owningNode()) {
    return block->owningNode()->kind();
  }
  return Symbol();
}

} // namespace

enum class ExitStatus { WILL, MIGHT, WONT };

enum class Transform { Returns, LoopContinuations };

// hasExited() indicates whether or not an exit has been hit.
// The ExitTransform pass maintains a false boolean false_val_ && a true boolean
// true_val_.
// if hasExited() == true_val_ then we have exited, if == false_val_ we have
// not. Otherwise, we might have exited.
// exitValues() are the values that we are propagating to a destination block.
// this is used for block outputs of loops and outputs of graphs & closures
struct ExitPair : public std::pair<Value*, std::vector<Value*>> {
  using pair::pair;

  ExitPair(Value* exit_v, at::ArrayRef<Value*> exit_val_ref) {
    std::vector<Value*> exit_vals;
    for (Value* v : exit_val_ref) {
      exit_vals.push_back(v);
    }
    AT_ASSERT(exit_v->type() == BoolType::get());
    this->first = exit_v;
    this->second = std::move(exit_vals);
  }

  Value* hasExited() const {
    return this->first;
  }

  std::vector<Value*> exitValues() const {
    return this->second;
  }
};

/**
 * This pass currently transforms the Graph so that all exit nodes targeting
 * a block location are removed from the graph and unified.
 * The exit node for breaks/continues is LoopContinuation, and the exit for
 * Graphs & Closures is ReturnStmt.
 *
 * Once we hit an Exit Node, we do not execute any further instructions
 * until the exit target has been reached.
 *
 * For blocks and control flow nodes that have an exit statement that may
 * have been hit, we conditionalize all execution on a boolean value that
 * indicates whether we have hit the exit, hasExited().
 */
struct ExitTransformer {
  ExitTransformer(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
    WithInsertPoint guard(graph_->block()->nodes().front());
    true_val_ = graph_->insertConstant(true);
    false_val_ = graph_->insertConstant(false);
  };

  void run(Transform transform_kind) {
    if (transform_kind == Transform::Returns) {
      current_exit_kind_ = prim::ReturnStmt;
      convertReturnOutputsToReturnStmts(graph_->block());
    } else {
      current_exit_kind_ = prim::LoopContinuation;
      convertLoopOutputsToContinuations(graph_->block());
    }
    transformExits(graph_->block());
  }

 private:
  // The Logic for the exiy transform simplifies if the block outputs
  // are converted to Exits before running, because you do not
  // have to handle an exit that could have maybe exited, could have not exited,
  // or must have exited. Now, it must have to have exited.
  static void convertBlockOutputsToNode(Block* block, Symbol kind) {
    auto ret_node = block->return_node();
    auto node = block->owningGraph()->create(kind, 0)->insertBefore(ret_node);
    for (auto inp : ret_node->inputs()) {
      node->addInput(inp);
    }
    removeOutputs(block);
  }

  static void convertLoopOutputsToContinuations(Block* block) {
    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        convertLoopOutputsToContinuations(b);
      }
    }
    if (owningNodeKind(block) == prim::Loop) {
      convertBlockOutputsToNode(block, prim::LoopContinuation);
    }
  }

  static bool isGraphOrClosureBlock(Block* block) {
    return owningNodeKind(block) == prim::Function ||
        block->owningNode() == nullptr;
  }

  static void convertReturnOutputsToReturnStmts(Block* block) {
    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        convertReturnOutputsToReturnStmts(b);
      }
    }
    if (isGraphOrClosureBlock(block)) {
      convertBlockOutputsToNode(block, prim::ReturnStmt);
    }
  }

  static void removeOutputs(Block* b) {
    while (b->outputs().size() > 0) {
      b->eraseOutput(0);
    }
  }

  static void addIfOutputs(
      Node* n,
      at::ArrayRef<Value*> true_outs,
      at::ArrayRef<Value*> false_outs) {
    IfView if_view(n);
    registerBlockOutputs(if_view.thenBlock(), true_outs);
    registerBlockOutputs(if_view.elseBlock(), false_outs);
    for (size_t i = 0; i < true_outs.size(); ++i) {
      auto out_type =
          unifyTypes(true_outs.at(i)->type(), false_outs.at(i)->type());
      n->addOutput()->setType(*out_type);
    }
  }

  // creates a vector of uninitialized values of the same type as the
  // values_to_match
  std::vector<Value*> matchValuesWithUnitialized(
      at::ArrayRef<Value*> values_to_match) {
    std::vector<Value*> match_values;
    for (Value* val : values_to_match) {
      match_values.push_back(getUnitValue(val->type()));
    }
    return match_values;
  }

  ExitPair transformLoop(Node* node) {
    LoopView loop(node);
    Block* body = loop.bodyBlock();
    auto exit_pair = transformExits(body);
    // if we're not exiting to outside the loop we don't need to do any work.
    if (getExitStatus(exit_pair) == ExitStatus::WONT) {
      return exit_pair;
    }

    // if we are, we need to update the loop continue condition so that
    // we exit the loop if we've hit an exit
    WithInsertPoint insert(body);
    auto new_if = graph_->insertNode(graph_->create(prim::If, 0));
    new_if->addInput(exit_pair.hasExited());
    new_if->addBlock()->registerOutput(false_val_);
    new_if->addBlock()->registerOutput(loop.nextCond());
    auto new_condition = new_if->addOutput()->setType(BoolType::get());
    loop.bodyBlock()->eraseOutput(0);
    loop.bodyBlock()->insertOutput(0, new_condition);

    // we also need to propagate hasExited() and exitValues() outside the loop

    // we didn't exit if we didn't enter the loop
    node->addInput(false_val_);
    body->addInput()->setType(BoolType::get());
    body->registerOutput(exit_pair.hasExited());
    Value* new_has_exited = node->addOutput()->setType(BoolType::get());

    for (Value* exit_value : exit_pair.exitValues()) {
      auto typ = exit_value->type();
      node->addInput(getUnitValue(typ));
      node->addOutput()->setType(typ);
      body->addInput()->setType(typ);
      body->registerOutput(exit_value);
    }

    auto exit_vals = node->outputs().slice(
        node->outputs().size() - exit_pair.exitValues().size());

    return ExitPair(new_has_exited, exit_vals);
  }

  // Recursively transforms the if node
  ExitPair transformIf(Node* node) {
    auto then_block = node->blocks().at(0);
    auto else_block = node->blocks().at(1);

    auto then_pair = transformExits(then_block);
    auto else_pair = transformExits(else_block);
    auto then_status = getExitStatus(then_pair);
    auto else_status = getExitStatus(else_pair);

    if (then_status == ExitStatus::WONT && else_status == ExitStatus::WONT) {
      return ExitPair(false_val_, std::vector<Value*>({}));
    }

    // for the block that is not exitting, its' exit values will not get
    // used so we create uninitialized values of the same type as the other
    // block
    if (then_status == ExitStatus::WONT) {
      std::vector<Value*> exit_vals =
          matchValuesWithUnitialized(else_pair.exitValues());
      then_pair = ExitPair(false_val_, exit_vals);
    } else if (else_status == ExitStatus::WONT) {
      std::vector<Value*> exit_vals =
          matchValuesWithUnitialized(then_pair.exitValues());
      else_pair = ExitPair(false_val_, exit_vals);
    }

    Value* has_exited;
    if (then_status == ExitStatus::WILL && else_status == ExitStatus::WILL) {
      // Need to maintain the invariant that if hasExited() == true_val_
      // then we have exited.
      has_exited = true_val_;
    } else {
      addIfOutputs(node, {then_pair.hasExited()}, {else_pair.hasExited()});
      has_exited = node->outputs().at(node->outputs().size() - 1);
    }
    addIfOutputs(node, then_pair.exitValues(), else_pair.exitValues());
    size_t num_exit_vals = then_pair.exitValues().size();
    auto exit_vals =
        node->outputs().slice(node->outputs().size() - num_exit_vals);
    return ExitPair(has_exited, exit_vals);
  }

  ExitStatus getExitStatus(ExitPair& exit_pair) {
    Value* exit_v = exit_pair.hasExited();
    if (exit_v == true_val_) {
      return ExitStatus::WILL;
    } else if (exit_v == false_val_) {
      return ExitStatus::WONT;
    } else {
      return ExitStatus::MIGHT;
    }
  }

  // Guards the remaining nodes in the block with an if node that takes
  // the has exited value as its conditional
  ExitPair guardBlockNodes(
      Block* block,
      const ExitPair& exit_pair,
      graph_node_list_iterator& iter) {
    auto new_if = graph_->create(prim::If, 0)->insertBefore(*iter);
    new_if->addInput(exit_pair.hasExited());

    auto exit_block = new_if->addBlock();
    auto guard_block = new_if->addBlock();

    // Move all remaining nodes into the guard block
    while (iter != block->nodes().end()) {
      auto node = *iter++;
      node->moveBefore(guard_block->return_node());
    }

    std::vector<Value*> exit_block_vals;
    // after an exit, the only values that will get used
    // are the hasExited() and exitValues(), so we match the existing
    // block outputs with unitialized
    exit_block_vals = matchValuesWithUnitialized(block->outputs());

    // Set the new if to have the same outputs of the original block,
    // then replace the original block outputs with new if's outputs
    for (size_t i = 0; i < block->outputs().size(); ++i) {
      exit_block->registerOutput(exit_block_vals.at(i));
      guard_block->registerOutput(block->outputs().at(i));
      new_if->addOutput()->setType(block->outputs().at(i)->type());
    }

    while (block->outputs().size() > 0) {
      block->eraseOutput(0);
    }
    for (auto out : new_if->outputs()) {
      block->registerOutput(out);
    }

    graph_->create(current_exit_kind_, {exit_pair.exitValues()}, 0)
        ->insertBefore(exit_block->return_node());
    return transformIf(new_if);
  }

  // these nodes my have uses,
  // such as in the case:
  // if i == 1:
  //    break
  //    j = j + 1
  // where the j + 1 value will be a block output, but since they will
  // never be used, it is safe to replace them with unitialized value
  void destroyNodeAfterExit(Node* n) {
    for (auto output : n->outputs()) {
      if (output->uses().size() > 0) {
        output->replaceAllUsesWith(getUnitValue(output->type()));
      }
    }
    n->destroy();
  }

  void deleteAfterExitNodes(Block* block, graph_node_list_iterator& iter) {
    if (iter == block->nodes().end()) {
      return;
    }
    WithInsertPoint insert(*block->nodes().begin());
    // need to destroy in reverse order so nodes have no uses when destroyed
    for (auto it = block->nodes().reverse().begin(); it != iter;) {
      Node* n = *it++;
      if (*it != block->return_node()) {
        destroyNodeAfterExit(n);
      }
    }
    destroyNodeAfterExit(*iter);
  }

  // if we are transforming LoopContinuations, then when we enter a loop
  // the exits target that loop. if transforming ReturnStmts, exit target
  // the most recent Closure/Graph
  void setTargetBlock(Block* block) {
    if (current_exit_kind_ == prim::LoopContinuation) {
      if (owningNodeKind(block) == prim::Loop) {
        target_block_ = block;
      }
    } else if (current_exit_kind_ == prim::ReturnStmt) {
      if (isGraphOrClosureBlock(block)) {
        target_block_ = block;
      }
    }
  }

  ExitPair transformExits(Block* block) {
    Block* prev_target_block = target_block_;
    setTargetBlock(block);
    ExitPair exit_pair = ExitPair(false_val_, std::vector<Value*>({}));
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* node = *it;
      it++;
      switch (node->kind()) {
        case prim::ReturnStmt:
        case prim::LoopContinuation: {
          if (node->kind() == current_exit_kind_) {
            exit_pair = ExitPair(true_val_, node->inputs());
            node->destroy();
          }
        } break;
        case prim::If: {
          exit_pair = transformIf(node);
        } break;
        case prim::Function: {
          exit_pair = transformExits(node->blocks().at(0));
        } break;
        case prim::Loop: {
          exit_pair = transformLoop(node);
        } break;
      }

      // if we have hit a node that might exit, we need to conditionally execute
      // all subsequent nodes in the block. if we've hit a node that will exit
      // we can remove all subsequent nodes.
      ExitStatus status = getExitStatus(exit_pair);
      if (status == ExitStatus::WILL) {
        deleteAfterExitNodes(block, it);
        break;
      }
      if (status == ExitStatus::MIGHT) {
        if (it != block->nodes().end()) {
          exit_pair = guardBlockNodes(block, exit_pair, it);
        }
        break;
      }
    }

    // if we are targeting this block, update the output values to the
    // exit values. since the exit does not extend outside this block,
    // update returned exit to false. then, reset the target_block to whatever
    // it was previously
    if (target_block_ == block) {
      TORCH_INTERNAL_ASSERT(getExitStatus(exit_pair) == ExitStatus::WILL);
      registerBlockOutputs(block, exit_pair.exitValues());
      exit_pair = ExitPair(false_val_, std::vector<Value*>({}));
    }
    target_block_ = prev_target_block;
    return exit_pair;
  }

  Value* getUnitValue(const TypePtr& type) {
    auto maybe_val = unit_values_.find(type);
    if (maybe_val != unit_values_.end()) {
      return maybe_val->second;
    }
    auto unit = graph_->createUninitialized(type)
                    ->insertAfter(graph_->param_node())
                    ->output();
    unit_values_[type] = unit;
    return unit;
  }

  // we create one uninitialized value per type, cache it here and reuse it
  std::unordered_map<TypePtr, Value*> unit_values_;

  // can either be LoopContinuation/ReturnStmt
  Symbol current_exit_kind_;
  Value* true_val_;
  Value* false_val_;

  // the block that we are currently targeting in the transform.
  // e.g. when we see a ReturnStmt
  Block* target_block_ = nullptr;
  std::shared_ptr<Graph> graph_;
};

// This pass takes in a graph where LoopContinuation & ReturnStmts exist in the
// graph and erases them in the graph, correctly setting block outputs.
// prim::LoopContinuation(*vals) means that the values are targeting the most
// recent loop block. prim::ReturnStmt(*vals) means that the values are
// targeting the most recent Closure or Graph Block. Once we hit an exit node,
// we do not execute any further instructions until the block exit reaches its
// destination. If we encounter a node that contains nested blocks that may
// have hit an exit node, such as an if statement that exits in one block
// and does not exit in the other, we use a boolean value to indicate if the
// exit has been hit or not. Then, we conditionalize further execution.
//
// The logic for the pass simplifies removing Loop Block Outputs and replacing
// them with LoopContinuations. We run that pass first, then we remove
// LoopContinuations
// Python example:
// while i < 5:
//   if i == 3:
//     i += 1
//     continue
//   i += 2
//
// -> transforms to:
//
// continue_loop = i < 5
// while continue_loop:
//   if i == 3:
//     i = i + 1
//     continue_loop = i < 5
//     did_exit = True
//   if did_exit:
//     pass
//   else:
//     i = i + 2
//     continue_loop = i < 5
// IR as it enters pass:
// %36 : bool = aten::lt(%i.1, %3)
// %i : int = prim::Loop(%1, %36, %i.1)
//   block0(%5 : int, %i.17 : int):
//     %8 : bool = aten::eq(%i.17, %7)
//     %i.16 : int = prim::If(%8)
//       block0():
//         %i.6 : int = aten::add(%i.17, %11)
//         %33 : bool = aten::lt(%i.6, %3)
//          = prim::LoopContinuation(%33, %i.6)
//         -> (%i.6)
//       block1():
//         -> (%i.17)
//     %i.13 : int = aten::add(%i.16, %19)
//     %4 : bool = aten::lt(%i.13, %3)
//     -> (%4, %i.13)
// return (%i)
//
//   -> transforms to
//
// %false_val : bool = prim::Constant[value=0]()
// %true_val : bool = prim::Constant[value=1]()
// %40 : int = prim::Uninitialized()
// %39 : bool = prim::Uninitialized()
// %36 : bool = aten::lt(%i.1, %3)
// %i : int = prim::Loop(%1, %36, %i.1)
//   block0(%5 : int, %i.17 : int):
//     %8 : bool = aten::eq(%i.17, %7)
//     %did_exit : bool, %continue_loop : bool, %43 : int, %i.16 : int =
//     prim::If(%8)
//       block0():
//         %i.6 : int = aten::add(%i.17, %11)
//         %33 : bool = aten::lt(%i.6, %3)
//         -> (%true_val, %33, %i.6, %i.6)
//       block1():
//         -> (%false_val, %39, %40, %i.17)
//     %44 : bool, %i : int = prim::If(%did_exit)
//       block0():
//         -> (%continue_loop, %43)
//       block1():
//         %i.13 : int = aten::add(%i.16, %19)
//         %4 : bool = aten::lt(%i.13, %3)
//         -> (%4, %i.13)
//     -> (%44, %i)

void TransformExits(std::shared_ptr<Graph>& graph) {
  ExitTransformer e_loop(graph);
  e_loop.run(Transform::LoopContinuations);
  ExitTransformer e_ret(graph);
  e_ret.run(Transform::Returns);
}

} // namespace jit
} // namespace torch
