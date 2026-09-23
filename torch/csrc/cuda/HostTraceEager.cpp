#include <torch/csrc/python_headers.h>

#include <torch/csrc/utils/pybind.h>

#include <ATen/cuda/host_trace/ti/EagerOps.h>
#include <ATen/cuda/host_trace/ti/TensorIteratorSym.h>

#include <vector>

// Python entry points for the eager CUDA hosts with a traced sibling beside
// the real host (index_select in Indexing.cu, cat in Shape.cu, triu / tril in
// TriangularOps.cu) and for the index_copy_ / index_put_ siblings. The trace
// mode in torch/cuda/_host_trace_ti.py calls them in place of the op; outside a
// trace they run the same launches in ordinary mode. Forward declared in
// torch/csrc/Module.cpp next to THCPHostTrace_init.

// NOLINTNEXTLINE(misc-use-internal-linkage)
void THCPHostTraceEager_init(PyObject* module) {
  namespace ti = at::cuda::host_trace::ti;
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_host_trace_ti_index_select",
      [](const at::Tensor& self, int64_t dim, const at::Tensor& index) {
        return ti::index_select_traced(self, dim, index);
      });
  m.def(
      "_host_trace_ti_cat",
      [](const std::vector<at::Tensor>& tensors, int64_t dim) {
        return ti::cat_traced(at::ITensorListRef(tensors), dim);
      });
  m.def(
      "_host_trace_ti_index_copy_",
      [](at::Tensor result,
         int64_t dim,
         const at::Tensor& index,
         const at::Tensor& source) {
        return ti::index_copy_traced(result, dim, index, source);
      });
  m.def(
      "_host_trace_ti_index_put_",
      [](at::Tensor src,
         const at::Tensor& value,
         const std::vector<at::Tensor>& indices,
         const std::vector<c10::SymInt>& indexed_sizes,
         const std::vector<c10::SymInt>& indexed_strides) {
        return ti::index_put_traced(
            src, value, indices, indexed_sizes, indexed_strides);
      });
  m.def(
      "_host_trace_ti_index",
      [](const at::Tensor& src,
         const std::vector<at::Tensor>& indices,
         const std::vector<c10::SymInt>& indexed_sizes,
         const std::vector<c10::SymInt>& indexed_strides) {
        return ti::index_traced(src, indices, indexed_sizes, indexed_strides);
      });
  // _index_put_impl_'s at::assert_no_overlap on the original operands, which
  // the Python entry restrides before the sibling sees them
  m.def(
      "_host_trace_ti_assert_no_overlap",
      [](const at::Tensor& a, const at::Tensor& b) {
        ti::assert_no_overlap_sym(a, b);
      });
  m.def(
      "_host_trace_ti_triu_tril",
      [](const at::Tensor& self,
         const c10::SymInt& k,
         bool upper,
         at::Tensor result) {
        return ti::triu_tril_traced(self, k, upper, result);
      });
  m.def(
      "_host_trace_ti_embedding_dense_backward",
      [](const at::Tensor& grad,
         const at::Tensor& indices,
         const c10::SymInt& num_weights,
         const c10::SymInt& padding_idx,
         bool scale_grad_by_freq) {
        return ti::embedding_dense_backward_traced(
            grad, indices, num_weights, padding_idx, scale_grad_by_freq);
      });
}
