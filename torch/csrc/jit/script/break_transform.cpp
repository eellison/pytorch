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

using ValueTable = std::unordered_map<std::string, Value*>;
struct MiniEnvironment {
  MiniEnvironment(Block* b, std::shared_ptr<MiniEnvironment> next = nullptr)
      : b(b), next(std::move(next)) {}

  Block* b;
  std::shared_ptr<MiniEnvironment> next;

  Value* findInThisFrame(const std::string& name) {
    auto it = value_table.find(name);
    if (it != value_table.end()) {
      return it->second;
    }
    return nullptr;
  }

  Value* findInAnyFrame(const std::string& name) {
    for (auto runner = this; runner; runner = runner->next.get()) {
      if (auto r = runner->findInThisFrame(name)) {
        return r;
      }
    }
    AT_ASSERT(false); // should never happen in break transform
  }

  void setVar(const std::string& name, Value* value) {
    value_table[name] = std::move(value);
  }

 private:
  ValueTable value_table;
};

/**
 * This pass transforms the Graph so that all ReturnStmts are merged into a
 * single value at the end of the graph.
 *
 * For blocks and control flow nodes that have a return statement that may have
 * been hit, we add an extra output for the return value, and an extra output
 * indicating whether or not the return has been hit (a sentinel value).
 *
 * When we encounter a node that might return, we guard all subsequent nodes
 * in the block with the sentinel value of that node.
 */

// Will a block or node return
enum BreakStatus { WONT_BREAK, MIGHT_BREAK, WILL_BREAK };

// The return output of Control Flow Nodes is the second to last output,
// sentinel output is the output
constexpr size_t HAS_BROKE_OFFSET = 1;

struct BreakTransformer {
  BreakTransformer(std::shared_ptr<Graph> graph_) : graph(std::move(graph_)) {
    WithInsertPoint guard(graph->block()->nodes().front());
    true_val = graph->insertConstant(true);
    false_val = graph->insertConstant(false);
  };

  void registerHasBroke(Block* block, Value* sent) {
    AT_ASSERT(sent->type() == BoolType::get());
    block->registerOutput(sent);
  }

  BreakStatus getBlockStatus(Block* block) {
    auto v = block_sentinel_val[block];
    if (v == false_val) {
      return WONT_BREAK;
    } else if (v == true_val) {
      return WILL_BREAK;
    } else {
      return MIGHT_BREAK;
    }
  }

  std::string uniqueName(Value * v) {
    return v->hasUniqueName() ? v->uniqueName() : "";
  }

  void addSentinel(Block* block) {
    auto b_status = getBlockStatus(block);
    if (b_status == WONT_BREAK) {
      registerHasBroke(block, false_val);
    } else if (b_status == WILL_BREAK) {
      registerHasBroke(block, true_val);
    } else if (b_status == MIGHT_BREAK) {
      registerHasBroke(block, block_sentinel_val[block]);
    } else {
      AT_ASSERT(false);
    }
  }

  // hi

  // The break status of a Loop is always WONT_BREAK,
  // because a break statement only applies to the innermost loop
  BreakStatus handleLoop(Node* node) {
    auto loop_block = node->blocks().at(0);
    handleBreaks(loop_block);

    if (getBlockStatus(loop_block) == WONT_BREAK) {
      return WONT_BREAK;
    }

    // TODO - more complicated logic and or peephole eliminations to simplify
    // BOOLEAN ORS

    auto break_if = loop_block->appendNode(graph->create(prim::If, 0));
    break_if->addInput(block_sentinel_val[loop_block]);
    break_if->addBlock()->registerOutput(false_val);
    break_if->addBlock()->registerOutput(loop_block->outputs().at(0));
    auto new_continue_condition =
        break_if->addOutput()->setType(BoolType::get());
    loop_block->eraseOutput(0);
    loop_block->insertOutput(0, new_continue_condition);

    auto out_names = node->ss(attr::value);
    // skip block continue val
    for (size_t i = 1; i < out_names.size(); ++i) {
      environment_stack->setVar(out_names[i], node->outputs().at(i - 1));
    }
    return WONT_BREAK;
  }

  // Recurses on the if node and returns its return status
  // If status != WONT_BREAK, sets the block_return_val and sentinel val
  // of its parent block before exit
  BreakStatus handleIf(Node* node) {
    auto true_block = node->blocks().at(0);
    auto false_block = node->blocks().at(1);

    // recurse
    auto true_status = handleBreaks(true_block);
    auto false_status = handleBreaks(false_block);

    size_t i = 0;
    for (const auto& name : node->ss(attr::value)) {
      environment_stack->setVar(name, node->outputs().at(i++));
    }

    if (true_status == WONT_BREAK && false_status == WONT_BREAK) {
      return WONT_BREAK;
    }

    addSentinel(true_block);
    addSentinel(false_block);
    auto sent = node->addOutput()
                    ->setType(BoolType::get())
                    ->setUniqueName("__did_break");

    block_sentinel_val[node->owningBlock()] = sent;
    if (true_status == WILL_BREAK && false_status == WILL_BREAK) {
      block_sentinel_val[node->owningBlock()] = true_val;
      return WILL_BREAK;
    } else {
      block_sentinel_val[node->owningBlock()] = sent;
    }

    return MIGHT_BREAK;
  }

  // Guards the remaining nodes in the block with an if node that takes
  // sentinel as its conditional
  BreakStatus guardBlockNodes(
      Block* block,
      generic_graph_node_list_iterator<Node>& iter) {

    AT_ASSERT(getBlockStatus(block) == MIGHT_BREAK);
    auto sentinel = block_sentinel_val[block];
    auto new_if = graph->create(prim::If, 0)->insertAfter(sentinel->node());
    new_if->addInput(sentinel);

    auto break_block = new_if->addBlock();
    auto guard_block = new_if->addBlock();

    // Move all remaining nodes into the guard block
    while (iter != block->nodes().end()) {
      auto node = *iter++;
      node->moveBefore(guard_block->return_node());
    }
    std::vector<std::string> block_output_names =
        block->owningNode()->ss(attr::value);
    for (auto name : block_output_names) {
      break_block->registerOutput(environment_stack->findInAnyFrame(name));
    }
    for (size_t i = 0; i < block->outputs().size(); ++i) {
      guard_block->registerOutput(block->outputs().at(i));
    }
    new_if->ss_(attr::value, block_output_names);

    for (size_t i = 0; i < block->outputs().size(); ++i) {
      auto orig_output = block->outputs().at(i);
      new_if->addOutput()->setType(orig_output->type())
                         ->setUniqueName(uniqueName(orig_output));
    }
    while (block->outputs().size() > 0) {
      block->eraseOutput(0);
    }
    for (auto out : new_if->outputs()) {
      block->registerOutput(out);
    }
    block_sentinel_val[break_block] = true_val;
    return handleIf(new_if);
  }

  void deleteAfterBreakNodes(Block* block, graph_node_list_iterator& iter) {
    auto names = block->owningNode()->ss(attr::value);
    for (size_t i = 0; i < block->outputs().size(); ++i) {
      block->eraseOutput(i);
      block->insertOutput(i, environment_stack->findInAnyFrame(names[i]));
    }
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

  void handleBreakStmt(Node* node) {
    const auto& names = node->ss(attr::value);
    for (size_t i = 0; i < node->inputs().size(); ++i) {
      environment_stack->setVar(names[i], node->inputs()[i]);
    }
    node->destroy();
  }

  BreakStatus handleBreaks(Block* block) {
    if (block_sentinel_val.count(block) != 0) {
      // Guarded return blocks have their status set prior
      AT_ASSERT(
          getBlockStatus(block) == WILL_BREAK &&
          block->nodes().begin() == block->nodes().end());
      return WILL_BREAK;
    }

    auto ret_status = WONT_BREAK;
    pushFrame(block);
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* node = *it;
      it++;
      switch (node->kind()) {
        case prim::BreakStmt: {
          handleBreakStmt(node);
          block_sentinel_val[block] = true_val;
          ret_status = WILL_BREAK;
        } break;
        case prim::If: {
          ret_status = handleIf(node);
        } break;
        case prim::Loop:
          ret_status = handleLoop(node);
        default:
          break;
      }
      if (ret_status == WILL_BREAK) {
        deleteAfterBreakNodes(block, it);
        break;
      } else if (ret_status == MIGHT_BREAK) {
        ret_status = guardBlockNodes(block, it);
        break;
      }
    }
    popFrame();
    if (ret_status == WONT_BREAK) {
      block_sentinel_val[block] = false_val;
    }
    return ret_status;
  }

  void eraseControlFlowAttr(Block* block) {
    for (auto n : block->nodes()) {
      if (n->kind() == prim::If || n->kind() == prim::Loop) {
        if (n->hasAttribute(attr::value)) {
          // some of new ifs won't have value set
          n->removeAttribute(attr::value);
        }
      }
      for (Block* b : n->blocks()) {
        eraseControlFlowAttr(b);
      }
    }
  }

  void run() {
    handleBreaks(graph->block());
    eraseControlFlowAttr(graph->block());
  }

  void setLoopCarriedVars(Block* b) {
    auto names = b->owningNode()->ss(attr::value);
    // we set the continue loop value to be true, since
    // we only add values from the enclosing scope for break guarded loops,
    // which will not use the continue value $continue
    environment_stack->setVar("$continue_loop", true_val);
    for (size_t i = 1; i < names.size(); ++i) {
      environment_stack->setVar(names[i], b->inputs().at(i));
    }
  }

  void pushFrame(Block* b) {
    environment_stack = std::make_shared<MiniEnvironment>(b, environment_stack);
    if (b->owningNode() && b->owningNode()->kind() == prim::Loop) {
      setLoopCarriedVars(b);
    }
  }

  std::shared_ptr<MiniEnvironment> popFrame() {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    return old_frame;
  }

  std::shared_ptr<MiniEnvironment> environment_stack = nullptr;
  std::unordered_map<Block*, Value*> block_sentinel_val;

  Value* bottom_val = nullptr;
  Value* true_val = nullptr;
  Value* false_val = nullptr;

  std::shared_ptr<Graph> graph;
};

void transformBreaks(std::shared_ptr<Graph>& graph) {
  ConstantPooling(graph);
  BreakTransformer e(graph);
  e.run();
}

} // namespace script
} // namespace jit
} // namespace torch
