#include <torch/csrc/jit/script/break_transform.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/final_returns.h>

namespace torch {
namespace jit {
namespace script {

/**
 * This pass transforms so that break & continue statements are removed.
 */

// Will a block or node continue or breaks
enum LoopStatus { WONT, MIGHT, WILL };

// Are we transforming breaks or continues
enum Transform { BREAKS, CONTINUES };

struct LoopTransformer {
  LoopTransformer(std::shared_ptr<Graph> graph_, Transform transform_) : graph(std::move(graph_)) {
    WithInsertPoint guard(graph->block()->nodes().front());
    true_val = graph->insertConstant(true);
    false_val = graph->insertConstant(false);
    transform = transform_;
  };

  const std::string& getVarname() {
    static const std::string& break_name = "$did_break";
    static const std::string& continue_name = "$did_continue";
    return transform == BREAKS ? break_name : continue_name;
  }

  Symbol transformKind() {
    return transform == BREAKS ? prim::BreakStmt : prim::ContinueStmt;
  }

  // Recurses on the if node and returns its return status
  // If status != WONT, sets the block_return_val and sentinel val
  // of its parent block before exit
  LoopStatus handleIf(Node* node) {
    auto true_block = node->blocks().at(0);
    auto false_block = node->blocks().at(1);

    auto true_status = handleBreaks(true_block);
    auto false_status = handleBreaks(false_block);

    if (true_status == WONT && false_status == WONT) {
      return WONT;
    } else if (true_status == WILL && false_status == WILL) {
      return WILL;
    } else {
      return MIGHT;
    }
  }

  LoopStatus guardBlockNodes(
      Block* block,
      generic_graph_node_list_iterator<Node>& iter) {

    // if we hit an if node and it might hit a break or continue,
    // we guard all subsequent nodes in the block, and do not execute
    // them if we did continue.

    auto new_if = graph->create(prim::If, 0)->insertBefore(*iter);
    auto sentinel = graph->createLoad(getVarname(), BoolType::get())->insertBefore(new_if);
    new_if->addInput(sentinel->output());

    auto hit_control_flow_block = new_if->addBlock();
    auto guard_block = new_if->addBlock();

    while (iter != block->nodes().end()) {
      auto node = *iter++;
      node->moveBefore(guard_block->return_node());
    }

    {
      WithInsertPoint insert(hit_control_flow_block);
      // NB: insert var scape before transform kind so it is not removed
      // See note in convert_to_ssa for why we need to insert VarEscape
      graph->insertNode(graph->create(prim::VarEscape, 0));
      graph->insertNode(graph->create(transformKind(), 0));
    }
    return handleIf(new_if);
  }

  void deleteAfterBreakNodes(Block* block, graph_node_list_iterator& iter) {
    if (iter == block->nodes().end()) {
      return;
    }
    // need to destroy in reverse order so nodes have no uses when destroyed
    for (auto it = block->nodes().reverse().begin(); it != iter;) {
      if (*it == block->return_node()) {
        it++;
      } else {
        it.destroyCurrent();
      }
    }
    iter->destroy();
  }

  void inlineLoopConditionIntoLoopBody(Node * n) {
    auto body_block = n->blocks().at(0);
    auto pre_header = n->blocks().at(1);
    for (auto it = pre_header->nodes().begin();
         it != pre_header->nodes().end();) {
      auto block_node = *it++;
      block_node->moveBefore(body_block->return_node());
    }
    body_block->insertOutput(0, pre_header->outputs().at(0));
    n->eraseBlock(1);
  }

  void handleLoop(Node* n) {
    Block * body_block = n->blocks().at(0);
    auto ret_status = handleBreaks(body_block);

    // When we're transforming breaks:
    // the body condition has not yet been inlined. If we we are not breaking
    // we need to inline the condition block into the end of the loop.
    // if we might break, we create an if statement and only execute the loop
    // header if we did not break.
    // Since we run the continue pass before the break pass,
    // we do not need to do any additional work in continues; guardBlock nodes
    // ensures that we do not execute any ops present in the block after a continue,
    // and loop condition is inlined after.

    if (transform == CONTINUES) {
      return;
    }

    if (ret_status == WONT) {
      inlineLoopConditionIntoLoopBody(n);
      return;
    }

    WithInsertPoint insert(body_block);
    auto did_break = graph->insertNode(graph->createLoad(getVarname(), BoolType::get()))->output();

    auto new_loop_condition = graph->insertNode(graph->create(prim::If));
    new_loop_condition->addInput(did_break);
    new_loop_condition->output()->setType(BoolType::get());

    // if we did break, we do not continue
    new_loop_condition->addBlock()->registerOutput(false_val);
    auto original_condition = new_loop_condition->addBlock();
    auto pre_header = n->blocks().at(1);
    for (auto it = pre_header->nodes().begin();
         it != pre_header->nodes().end();) {
      auto block_node = *it++;
      block_node->moveBefore(original_condition->return_node());
    }
    original_condition->insertOutput(0, pre_header->outputs().at(0));
    n->eraseBlock(1);
    body_block->registerOutput(new_loop_condition->output());
  };

  LoopStatus handleBreaks(Block* block) {
    auto ret_status = WONT;
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* node = *it;
      it++;
      switch (node->kind()) {
        case prim::Function: {
          handleBreaks(node->blocks().at(0));
        } break;
        case prim::ContinueStmt:
        case prim::BreakStmt: {
          if (node->kind() != transformKind()) {
            continue;
          }
          WithInsertPoint b(block);
          node->destroy();
          ret_status = WILL;
        } break;
        case prim::If: {
          ret_status = handleIf(node);
        } break;
        case prim::Loop: {
          handleLoop(node);
          // break statement can only effect the loop node
          ret_status = WONT;
        } break;
      }
      if (ret_status == WILL) {
        deleteAfterBreakNodes(block, it);
        break;
      } else if (ret_status == MIGHT) {
        if (it != block->nodes().end()) {
          ret_status = guardBlockNodes(block, it);
        }
        break;
      }
    }

    {
      // MIGHT value must be an output of an if, so we do not need to set it
      WithInsertPoint insert(block);
      if (ret_status == WILL) {
        graph->insertNode(graph->createStore(getVarname(), true_val));
      } else if (ret_status == WONT) {
        graph->insertNode(graph->createStore(getVarname(), false_val));
      }
    }

    return ret_status;
  }

  void run() {
    handleBreaks(graph->block());
  }

  Transform transform;
  Value* true_val = nullptr;
  Value* false_val = nullptr;

  std::shared_ptr<Graph> graph;
};

void TransformBreaks(std::shared_ptr<Graph>& graph) {
  // We transform the continues first, so the loop body condition is not yet
  // inlined, and the loop condition still executes even if a continue is hit.
  LoopTransformer continues(graph, CONTINUES);
  continues.run();
  LoopTransformer breaks(graph, BREAKS);
  breaks.run();
}

} // namespace script
} // namespace jit
} // namespace torch
