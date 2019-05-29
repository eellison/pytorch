#include <torch/csrc/jit/script/convert_to_ssa.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/mini_environment.h>

namespace torch {
namespace jit {
namespace script {

// At this point the graph has already undergone type checking and the correct
// outputs and inputs have already been set for Control Flow Nodes. prim::Store
// sets the value of a variable in the current scope, and we replace all
// uses of prim::Load with the currently scoped value. Since all error
// checking has already has already occurred & control outputs set
// we can use a simple environment to handle variable scoping.

using ValueEnvironment = MiniEnvironment<Value*>;

struct SSATransformer {
  void convertBlockToSSA(Block* block) {
    pushFrame(block);
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      auto n = *it;
      it++;
      switch (n->kind()) {
        case prim::If:
        case prim::Loop: {
          for (auto b : n->blocks()) {
            convertBlockToSSA(b);
          }
        } break;
        case prim::Store: {
          environment_stack->setVar(n->s(attr::name), n->input());
          n->destroy();
        } break;
        case prim::Load: {
          auto name = n->s(attr::name);
          auto var = environment_stack->findInAnyFrame(name);
          AT_ASSERT(var);
          n->output()->replaceAllUsesWith(var);
          n->destroy();
        } break;
      }
    }
    popFrame();
  }

  void pushFrame(Block* b) {
    environment_stack =
        std::make_shared<ValueEnvironment>(b, environment_stack);
  }

  std::shared_ptr<ValueEnvironment> popFrame() {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    return old_frame;
  }

  void run(std::shared_ptr<Graph>& graph) {
    convertBlockToSSA(graph->block());
  }

  std::shared_ptr<ValueEnvironment> environment_stack = nullptr;
};

void ConvertToSSA(std::shared_ptr<Graph>& graph) {
  SSATransformer e;
  e.run(graph);
}

} // namespace script
} // namespace jit
} // namespace torch
