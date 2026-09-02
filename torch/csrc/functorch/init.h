#include <Python.h>

namespace torch::functorch::impl {

void initFuncTorchBindings(PyObject* module);
PyObject* unwrap_dead_wrappers(PyObject* args);

// Pops the functorch dynamic layer stack back to `depth`, undoing the
// transforms above it. Used by
// torch._C._dynamo.eval_frame.exit_compiled_region.
void dynamo_pop_dynamic_layer_stack_to_depth(size_t depth);

} // namespace torch::functorch::impl
