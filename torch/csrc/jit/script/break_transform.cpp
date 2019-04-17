#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/final_returns.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/break_transform.h>
#include <ATen/core/jit_type.h>

namespace torch {
namespace jit {
namespace script {

using NameToValue = std::unordered_map<std::string, Value *>;
using ValueToName = std::unordered_map<Value *, std::string>;

/**
 * This pass transforms the Graph so that all ReturnStmts are merged into a single
 * value at the end of the graph.
 *
 * For blocks and control flow nodes that have a return statement that may have
 * been hit, we add an extra output for the return value, and an extra output
 * indicating whether or not the return has been hit (a sentinel value).
 *
 * When we encounter a node that might return, we guard all subsequent nodes
 * in the block with the sentinel value of that node.
 */

// Will a block or node return
enum BreakStatus {WONT_BREAK, MIGHT_BREAK, WILL_BREAK};

// The return output of Control Flow Nodes is the second to last output,
// sentinel output is the output
constexpr size_t HAS_BROKE_OFFSET = 1;

struct BreakTransformer {
  BreakTransformer(std::shared_ptr<Graph> graph_): graph(std::move(graph_)) {};

  Value * getSentinelVal(Node * n) {
    AT_ASSERT(n->kind() == prim::If || n->kind() == prim::Loop);
    return n->outputs().at(n->outputs().size() - HAS_BROKE_OFFSET);
  }

  Value * getSentinelVal(Block * b) {
    return b->outputs().at(b->outputs().size() - HAS_BROKE_OFFSET);
  }

  void registerHasBroke(Block * block, Value * sent) {
    AT_ASSERT(sent->type() == BoolType::get());
    block->registerOutput(sent);
  }

  void addSentinel(Block * block) {
    auto b_status = block_status[block];
    if (b_status == WONT_BREAK) {
      registerHasBroke(block, getBoolVal(false));
    } else if (b_status == WILL_BREAK) {
      registerHasBroke(block, getBoolVal(true));
    } else if (b_status == MIGHT_BREAK) {
      registerHasBroke(block, block_sentinel_val[block]);
    } else {
      AT_ASSERT(false);
    }
  }

  NameToValue convertVarNameToValueNode(Node * node) {
    AT_ASSERT(node->kind() == prim::VarCapture || node->kind() == prim::BreakStmt);
    NameToValue c;
    const auto& names = node->ss(attr::value);
    const auto& inputs = node->inputs();
    size_t i = 0;
    for (const auto& name: names) {
      c[name] = inputs[i++];
    }
    return c;
  }


  std::map<std::string, size_t> getNameBlockIndices(Block * block, NameToValue c) {
    std::map<std::string, size_t> mappings;
    for (const auto& capture: c) {
      bool set = false;
      for (size_t i = 0; i < block->outputs().size(); ++i) {
        if (block->outputs()[i] == capture.second) {
          mappings[capture.first] = i;
          set = true;
          break;
        }
      }
      AT_ASSERT(set);
    }
    return mappings;
  }

  ValueToName switch_map(NameToValue map) {
    ValueToName map_2;
    for (const auto& pair: map) {
      map_2[pair.second] = pair.first;
    }
    return map_2;
  }

  // void stitchCorrectBlockReturns(Block * b, NameToValue new_outputs) {
  //   Node * var_capture = nullptr;
  //   for (Node * n: b->nodes()) {
  //     if (n->kind() == prim::VarCapture) {
  //       var_capture = n;
  //       break;
  //     }
  //   }
  //   AT_ASSERT(var_capture != nullptr);
  //   auto mappings = getNameBlockIndices(b, convertVarNameToValueNode(var_capture));
  //   for (const auto& pair: mappings) {
  //     auto new_output = new_outputs[pair.first];
  //     b->eraseOutput(pair.second);
  //     b->insertOutput(pair.second, new_output);
  //   }
  //   deleteAfterBreakNodes(b, var_capture);
  //   var_capture->destroy();
  //
  // }


  // The break status of a Loop is always WONT_BREAK,
  // because a break statement only applies to the innermost loop
  BreakStatus handleLoop(Node * node) {
    auto loop_block = node->blocks().at(0);
    handleBreaks(loop_block);

    if (block_status[loop_block] == WONT_BREAK) {
      return WONT_BREAK;
    }

    // did break
    // TODO - more complicated logic and or peephole eliminations to simplify
    // BOOLEAN ORS

    auto break_if = loop_block->appendNode(graph->create(prim::If, 0));
    break_if->addInput(block_sentinel_val[loop_block]);
    break_if->addBlock()->registerOutput(false_val);
    break_if->addBlock()->registerOutput(loop_block->outputs().at(0));
    auto new_continue_condition = break_if->addOutput()->setType(BoolType::get());
    loop_block->eraseOutput(0);
    loop_block->insertOutput(0, new_continue_condition);

    return WONT_BREAK;
  }

  // Recurses on the if node and returns its return status
  // If status != WONT_BREAK, sets the block_return_val and sentinel val
  // of its parent block before exit
  BreakStatus handleIf(Node * node) {
    auto true_block = node->blocks().at(0);
    auto false_block = node->blocks().at(1);

    // recurse
    handleBreaks(true_block);
    handleBreaks(false_block);

    auto true_status = block_status[true_block];
    auto false_status = block_status[false_block];

    if (true_status == WONT_BREAK && false_status == WONT_BREAK) {
      return WONT_BREAK;
    }

    addSentinel(true_block);
    addSentinel(false_block);
    auto sent = node->addOutput()->setType(BoolType::get())->setUniqueName("$did_break");

    block_sentinel_val[node->owningBlock()] = sent;
    NameToValue empty;
    NameToValue true_capture = block_capture.count(true_block) ?
      block_capture[true_block] : empty;
    NameToValue false_capture = block_capture.count(false_block) ?
      block_capture[false_block] : empty;

    // mappings of all Names -> Index
    auto true_name_indices = getNameBlockIndices(true_block, true_capture);
    auto false_name_indices = getNameBlockIndices(false_block, false_capture);

    NameToValue ret_capture;
    for (const auto& pair: true_name_indices) {
      ret_capture[pair.first] = node->outputs().at(pair.second);
    }
    for (const auto& pair: false_name_indices) {
      ret_capture[pair.first] = node->outputs().at(pair.second);
    }
    // we should now have all of the output name of the output variables
    std::cout << "In handle if\n";
    for (const auto& pair: ret_capture) {
      std::cout << pair.first << " : " << '%' << pair.second->uniqueName() << "\n";
    }

    // TODO AST Transform that removes all statements after breaks in a block
    node_capture[node] = ret_capture;

    if (true_status == WILL_BREAK && false_status == WILL_BREAK) {
      block_sentinel_val[node->owningBlock()] = getBoolVal(true);
      return WILL_BREAK;
      //TODO - ensure that there no assignments after will break
      // stitchCorrectBlockReturns(node->owningBlock(), ret_capture);
    }

    // AT_ASSERTM(false, "not handled yet");
    return MIGHT_BREAK;
  }

  BreakStatus guardLoopBlockNodes(Block * block, Value * sentinel, generic_graph_node_list_iterator<Node>& iter) {
    auto new_if = graph->create(prim::If, 0)->insertAfter(sentinel->node());
    new_if->addInput(sentinel);

    auto break_block = new_if->addBlock();
    auto guard_block = new_if->addBlock();

    // NB: need to set return_block status and before recursing
    // or an empty block will appear to be a WONT_BREAK block
    block_status[break_block] = WILL_BREAK;

    NameToValue break_captures = node_capture[sentinel->node()];
    std::cout << " BREAK CAPTURES \n";
    for (auto& pair: break_captures) {
      std::cout << pair.first << ": " << '%' << pair.second->uniqueName() << "\n";
    }
    NameToValue block_orig_captures = block_capture[block];
    std::cout << " IN GUARD BLOCK\n";
    for (auto& pair: block_orig_captures) {
      std::cout << pair.first << ": " << '%' << pair.second->uniqueName() << "\n";
    }
    // Move all remaining nodes into the guard block
    while (iter != block->nodes().end()) {
      auto node = *iter++;
      node->moveBefore(guard_block->return_node());
    }

    auto name_to_index = getNameBlockIndices(block, block_orig_captures);
    ValueToName value_to_name = switch_map(block_orig_captures);

    std::cout << " Value to name\n";
    for (const auto& pair : value_to_name) {
      std::cout << "Value: " << '%' << pair.first->uniqueName() << ": " << pair.second << "\n";
    }


    NameToValue new_break_captures;

    // TODO
    // // DID BREAK
    // TODO - you need to speical case this for loop node
    // because there is no special var emitted from it
    // loop continue val
    break_block->registerOutput(false_val);
    new_break_captures["$continue_loop"] = break_block->outputs().at(0);

    //
    // bool guard_block_has_loop_continue_condition = true;
    // if (guard_block_has_loop_continue_condition) {
    guard_block->registerOutput(block->outputs().at(0));
    block_orig_captures["$continue_loop"] = guard_block->outputs().at(0);

    // }
    // new_if->addOutput()->setType(BoolType::get());

    // set up block output
    // skip the loop continue condition
    for (size_t i = 1; i < block->outputs().size(); ++i) {
      Value * block_output = block->outputs().at(i);
      std::string output_name = value_to_name[block_output];
      std::cout << "OUTPUT NAME: " << output_name << " contained:" << break_captures.count(output_name) << "\n";
      // if there is a value for the output, take that value, otherwise
      // copy the loop carried condition
      if (break_captures.count(output_name)) {
        block_output = break_captures[output_name];
      } else {
        block_output = block->inputs().at(i);
      }
      new_break_captures[output_name] = block_output;
      break_block->registerOutput(block_output);
    }
    block_capture[break_block] = new_break_captures;


    for (size_t i = 1; i < block->outputs().size(); ++i) {
      guard_block->registerOutput(block->outputs().at(i));
    }
    block_capture[guard_block] = block_orig_captures;

    for (size_t i = 0; i < block->outputs().size(); ++i) {
       new_if->addOutput()->setType(block->outputs().at(i)->type());
    }

    while (block->outputs().size() > 0) {
      block->eraseOutput(0);
    }

    //you need to know if a block output will be the loop continue
    //condition, because if it is, you need to unify it with the
    //break condition

    NameToValue new_block_captures;
    for (auto out: new_if->outputs()) {
      block->registerOutput(out);
    }
    block_status[break_block] = WILL_BREAK;
    return handleIf(new_if);
  }



  // Guards the remaining nodes in the block with an if node that takes
  // sentinel as its conditional
  BreakStatus guardBlockNodes(Block * block, Value * sentinel, generic_graph_node_list_iterator<Node>& iter) {
    auto new_if = graph->create(prim::If, 0)->insertAfter(sentinel->node());
    new_if->addInput(sentinel);

    auto break_block = new_if->addBlock();
    auto guard_block = new_if->addBlock();

    // NB: need to set return_block status and before recursing
    // or an empty block will appear to be a WONT_BREAK block
    block_status[break_block] = WILL_BREAK;

    NameToValue break_captures = node_capture[sentinel->node()];
    std::cout << " BREAK CAPTURES \n";
    for (auto& pair: break_captures) {
      std::cout << pair.first << ": " << '%' << pair.second->uniqueName() << "\n";
    }
    NameToValue block_orig_captures = block_capture[block];
    std::cout << " IN GUARD BLOCK\n";
    for (auto& pair: block_orig_captures) {
      std::cout << pair.first << ": " << '%' << pair.second->uniqueName() << "\n";
    }
    // Move all remaining nodes into the guard block
    while (iter != block->nodes().end()) {
      auto node = *iter++;
      node->moveBefore(guard_block->return_node());
    }

    auto name_to_index = getNameBlockIndices(block, block_orig_captures);
    ValueToName value_to_name = switch_map(block_orig_captures);

    std::cout << " Value to name\n";
    for (const auto& pair : value_to_name) {
      std::cout << "Value: " << '%' << pair.first->uniqueName() << ": " << pair.second << "\n";
    }

    NameToValue new_break_captures;

    // set up block output
    // skip the loop continue condition
    for (size_t i = 0; i < block->outputs().size(); ++i) {
      Value * block_output = block->outputs().at(i);
      std::string output_name = value_to_name[block_output];
      std::cout << "OUTPUT NAME: " << output_name << " contained:" << break_captures.count(output_name) << "\n";
      // if there is a value for the output, take that value, otherwise
      // copy the loop carried condition
      if (break_captures.count(output_name)) {
        block_output = break_captures[output_name];
      }
      // else {
      //   block_ouptut
      //   block_output = block->inputs().at(i);
      // }
      new_break_captures[output_name] = block_output;
      break_block->registerOutput(block_output);
    }
    block_capture[break_block] = new_break_captures;

    for (size_t i = 0; i < block->outputs().size(); ++i) {
      guard_block->registerOutput(block->outputs().at(i));
    }
    block_capture[guard_block] = block_orig_captures;

    for (size_t i = 0; i < block->outputs().size(); ++i) {
       new_if->addOutput()->setType(block->outputs().at(i)->type());
    }

    while (block->outputs().size() > 0) {
      block->eraseOutput(0);
    }

    //you need to know if a block output will be the loop continue
    //condition, because if it is, you need to unify it with the
    //break condition

    NameToValue new_block_captures;
    for (auto out: new_if->outputs()) {
      block->registerOutput(out);
    }
    block_status[break_block] = WILL_BREAK;
    return handleIf(new_if);
  }

  void deleteAfterBreakNodes(Block * block, Node * return_node) {
    auto nodes = block->nodes().reverse();
    for (auto it = nodes.begin(); it != nodes.end() && *it != return_node;) {
      auto node = it;
      it++;
      if (*node != block->return_node()) {
        node->destroy();
      }
    }
    if (return_node->kind() == prim::BreakStmt) {
      return_node->destroy();
    }
  }

  void handleBreaks(Block * block) {
    auto ret_status = WONT_BREAK;
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* node = *it;
      it++;
      switch (node->kind()) {
        case prim::BreakStmt: {
          block_capture[block] = convertVarNameToValueNode(node);
          ret_status = WILL_BREAK;
        } break;
        case prim::If: {
          ret_status = handleIf(node);
        } break;
        case prim::Loop:
          handleLoop(node);
          // break statement can only effect the loop node
          ret_status = WONT_BREAK;
        default:
          break;
      }
      if (ret_status == WILL_BREAK) {
        deleteAfterBreakNodes(block, node);
        break;
      } else if (ret_status == MIGHT_BREAK) {
        if (it != block->nodes().end()) {
          if (block->owningNode()->kind() == prim::Loop) {
            ret_status = guardLoopBlockNodes(block, getSentinelVal(node), it);
          } else {
            ret_status = guardBlockNodes(block, getSentinelVal(node), it);
          }
        }
        break;
      }
    }
    if (block_status.count(block) == 0) {
      block_status[block] = ret_status;
    } else {
      // Guarded return blocks have their status set prior
      AT_ASSERT(block_status[block] == WILL_BREAK
        && block->nodes().begin() == block->nodes().end());
    }
  }

  Value * getBottomVal() {
    if (bottom_val != nullptr) {
      return bottom_val;
    }
    WithInsertPoint guard(graph->block()->nodes().front());
    bottom_val = graph->insertNode(graph->create(prim::Bottom, {}, 1))->output()
      ->setType(BottomType::get());
    return bottom_val;
  }

  Value * getBoolVal(bool val) {
    WithInsertPoint guard(graph->block()->nodes().front());
    if (val) {
      if (true_val != nullptr) {
        return true_val;
      }
      true_val = graph->insertConstant(true);
      return true_val;
    } else {
      if (false_val != nullptr) {
        return false_val;
      }
      false_val = graph->insertConstant(false);
      return false_val;
    }
  }
  // for (Node * n: block->nodes()) {
  void associateVarCaptures(Block * block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node * n = *it;
      it++;
      if (n->kind() == prim::VarCapture) {
        block_capture[block] = convertVarNameToValueNode(n);
        n->destroy();
      }
      for (Block * b: n->blocks()) {
        associateVarCaptures(b);
      }
    }
  }

  void run() {
    associateVarCaptures(graph->block());
    handleBreaks(graph->block());
  }


  // a block may have a value that is used in the loop continue condition
  std::unordered_map<Block *, Value *> loop_continue_condition;

  // After a call to handleBreaks, a block will have set its break status
  std::unordered_map<Block *, BreakStatus> block_status;

  // Blocks that might break need a sentinel value to indicate if they
  // broke or not
  std::unordered_map<Block *, Value *> block_sentinel_val;

  std::unordered_map<Block *, NameToValue> block_capture;
  // std::unordered_map<Block *, NameToValue> var_capture_blocks;

  std::unordered_map<Node *, NameToValue> node_capture;

  Value * bottom_val = nullptr;
  Value * true_val = nullptr;
  Value * false_val = nullptr;

  std::shared_ptr<Graph> graph;
};


void transformBreaks(std::shared_ptr<Graph>& graph) {
  ConstantPooling(graph);
  BreakTransformer e(graph);
  e.run();
  // maybe dce ?
}


} // namespace script
} // namespace jit
} // namespace torch
