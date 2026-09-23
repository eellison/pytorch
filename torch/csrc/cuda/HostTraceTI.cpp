#include <torch/csrc/python_headers.h>

#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_symnode.h>

#include <ATen/ScalarOps.h>

#include <ATen/HostTraceSiblingBindings.h>
#include <ATen/cuda/host_trace/NllLossHost.h>
#include <ATen/cuda/host_trace/SoftmaxHost.h>
#include <ATen/cuda/host_trace/ti/ForeachOps.h>
#include <ATen/cuda/host_trace/ti/Ops.h>
#include <ATen/cuda/host_trace/ti/ReduceOps.h>
#if defined(USE_DISTRIBUTED) && defined(USE_C10D)
#include <torch/csrc/distributed/c10d/symm_mem/HostTraceSymm.hpp>
#endif

#include <optional>
#include <string>

// Python entry points for the elementwise ops and reductions that opt into
// the traced TensorIterator sibling (aten/src/ATen/cuda/host_trace/ti). The
// trace mode in torch/cuda/_host_trace.py calls them in place of the real op;
// outside a trace they run the same launches in ordinary mode. Forward declared
// in torch/csrc/Module.cpp next to THCPHostTrace_init.

namespace {
// A binary operand: a tensor, or the Python number the dispatcher unwrapped
// from its wrapped 0-dim tensor, wrapped again as the argument parser does
// (a wrapped number, so it promotes like one).
at::Tensor operand(const py::handle& obj) {
  PyObject* p = obj.ptr();
  if (THPVariable_Check(p)) {
    return THPVariable_Unpack(p);
  }
  at::Scalar s;
  if (PyBool_Check(p)) {
    s = at::Scalar(p == Py_True);
  } else if (THPUtils_checkLong(p)) {
    s = at::Scalar(THPUtils_unpackLong(p));
  } else if (PyComplex_Check(p)) {
    s = at::Scalar(c10::complex<double>(
        PyComplex_RealAsDouble(p), PyComplex_ImagAsDouble(p)));
  } else if (THPUtils_checkDouble(p)) {
    s = at::Scalar(THPUtils_unpackDouble(p));
  } else {
    TORCH_CHECK_TYPE(
        false, "host_trace: a binary operand must be a tensor or a number");
  }
  return at::native::wrapped_scalar_tensor(s);
}

// a Scalar argument (pow's exponent) from the Python number the schema carries
at::Scalar scalar_arg(const py::handle& obj) {
  PyObject* p = obj.ptr();
  if (PyBool_Check(p)) {
    return at::Scalar(p == Py_True);
  } else if (THPUtils_checkLong(p)) {
    return at::Scalar(THPUtils_unpackLong(p));
  } else if (PyComplex_Check(p)) {
    return at::Scalar(c10::complex<double>(
        PyComplex_RealAsDouble(p), PyComplex_ImagAsDouble(p)));
  } else if (THPUtils_checkDouble(p)) {
    return at::Scalar(THPUtils_unpackDouble(p));
  }
  TORCH_CHECK_TYPE(false, "host_trace: a Scalar argument must be a number");
}

// a bound of arange: a Python number, or a torch.SymInt / torch.SymFloat
// kept symbolic (the Scalar caster does not load; the SymInt ones do)
at::Scalar bound_arg(const py::handle& obj) {
  if (torch::is_symint(obj)) {
    return at::Scalar(py::cast<c10::SymInt>(obj));
  }
  if (torch::is_symfloat(obj)) {
    return at::Scalar(py::cast<c10::SymFloat>(obj));
  }
  return scalar_arg(obj);
}
} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
void THCPHostTraceTI_init(PyObject* module) {
  namespace ti = at::cuda::host_trace::ti;
  auto m = py::handle(module).cast<py::module>();
  // `out`: the structured kernel's out= / in-place destination (self for an
  // in-place op), or None for the allocating form
  m.def(
      "_host_trace_ti_add",
      [](const py::handle& self,
         const py::handle& other,
         const py::handle& alpha,
         const std::optional<at::Tensor>& out) {
        // alpha as the number it is (an int an int64 Scalar, converted to
        // opmath once by the kernel host, as add_kernel does), a symbolic
        // number pinned by that conversion; the entry is the generated one
        // in UfuncCUDA_add.cu (E36), this binding keeps the argument rules
        return ti::gen::add_traced(
            operand(self),
            operand(other),
            bound_arg(alpha),
            out.value_or(at::Tensor()));
      },
      py::arg("self"),
      py::arg("other"),
      py::arg("alpha"),
      py::arg("out") = std::nullopt);
  m.def(
      "_host_trace_ti_mul",
      [](const py::handle& self,
         const py::handle& other,
         const std::optional<at::Tensor>& out) {
        return ti::mul_traced(
            operand(self), operand(other), out.value_or(at::Tensor()));
      },
      py::arg("self"),
      py::arg("other"),
      py::arg("out") = std::nullopt);
  m.def(
      "_host_trace_ti_div",
      [](const py::handle& self,
         const py::handle& other,
         const std::optional<at::Tensor>& out) {
        return ti::div_traced(
            operand(self), operand(other), out.value_or(at::Tensor()));
      },
      py::arg("self"),
      py::arg("other"),
      py::arg("out") = std::nullopt);
  m.def("_host_trace_ti_fill_", [](at::Tensor self, const py::handle& value) {
    return ti::fill_traced(self, scalar_arg(value));
  });
  m.def("_host_trace_ti_zero_", [](at::Tensor self) {
    return ti::zero_traced(self);
  });
  m.def(
      "_host_trace_ti_arange",
      [](const py::handle& start, const py::handle& step, at::Tensor out) {
        return ti::arange_traced(bound_arg(start), bound_arg(step), out);
      });
  m.def(
      "_host_trace_ti_compare",
      [](const py::handle& self,
         const py::handle& other,
         const std::string& op) {
        return ti::compare_traced(operand(self), operand(other), op);
      });
  m.def(
      "_host_trace_ti_masked_fill_",
      [](at::Tensor self, const at::Tensor& mask, const py::handle& value) {
        return ti::masked_fill_traced(self, mask, scalar_arg(value));
      });
  m.def(
      "_host_trace_ti_clamp",
      [](const at::Tensor& self,
         const py::handle& min,
         const py::handle& max,
         const std::optional<at::Tensor>& out) {
        return ti::clamp_scalar_traced(
            self,
            min.is_none() ? std::nullopt : std::optional(scalar_arg(min)),
            max.is_none() ? std::nullopt : std::optional(scalar_arg(max)),
            out.value_or(at::Tensor()));
      },
      py::arg("self"),
      py::arg("min"),
      py::arg("max"),
      py::arg("out") = std::nullopt);
  m.def("_host_trace_ti_silu", [](const at::Tensor& self) {
    return ti::silu_traced(self);
  });
  m.def(
      "_host_trace_ti_gelu",
      [](const at::Tensor& self, const std::string& approximate) {
        return ti::gelu_traced(self, approximate);
      });
  m.def("_host_trace_ti_copy_", [](at::Tensor dst, const at::Tensor& src) {
    return ti::copy_traced(dst, src);
  });
  m.def("_host_trace_ti_reciprocal", [](const at::Tensor& self) {
    return ti::reciprocal_traced(self);
  });
  m.def("_host_trace_ti_tanh", [](const at::Tensor& self) {
    return ti::tanh_traced(self);
  });
  m.def("_host_trace_ti_sqrt", [](const at::Tensor& self) {
    return ti::sqrt_traced(self);
  });
  m.def(
      "_host_trace_ti_pow_tensor_scalar",
      [](const at::Tensor& self, const py::handle& exponent) {
        return ti::pow_tensor_scalar_traced(self, scalar_arg(exponent));
      });
  m.def(
      "_host_trace_ti_where",
      [](const at::Tensor& condition,
         const at::Tensor& self,
         const at::Tensor& other) {
        return ti::where_traced(condition, self, other);
      });
  m.def("_host_trace_ti_sin", [](const at::Tensor& self) {
    return ti::sin_traced(self);
  });
  m.def("_host_trace_ti_cos", [](const at::Tensor& self) {
    return ti::cos_traced(self);
  });
  m.def("_host_trace_ti_exp", [](const at::Tensor& self) {
    return ti::exp_traced(self);
  });
  m.def("_host_trace_ti_rsqrt", [](const at::Tensor& self) {
    return ti::rsqrt_traced(self);
  });
  m.def("_host_trace_ti_neg", [](const at::Tensor& self) {
    return ti::neg_traced(self);
  });
  // the converted one-shot symmetric-memory all-reduce host
  // (CUDASymmetricMemoryOps.cu): `real` is the buffer the handle is looked up
  // by, the traced input's real tensor (torch/cuda/_host_trace_symm.py)
  m.def(
      "_host_trace_symm_one_shot_all_reduce_out",
      [](const at::Tensor& input,
         const at::Tensor& real,
         const std::optional<at::Tensor>& local_input,
         const std::string& reduce_op,
         const std::string& group_name,
         at::Tensor out) {
#if defined(USE_DISTRIBUTED) && defined(USE_C10D)
        return c10d::symmetric_memory::host_trace_one_shot_all_reduce_out(
            input, real, local_input, reduce_op, group_name, std::move(out));
#else
        TORCH_CHECK(false, "host_trace: built without distributed support");
        return out;
#endif
      });
  // the converted softmax host (SoftMax.cu): the entry allocates the output
  // through the trace mode, so it is a traced root, and calls the host
  m.def(
      "_host_trace_softmax_out",
      [](const at::Tensor& self,
         int64_t dim,
         bool half_to_float,
         bool log_softmax,
         at::Tensor out) {
        return at::native::host_trace_softmax_out(
            self, dim, half_to_float, log_softmax, out);
      });
  m.def(
      "_host_trace_softmax_backward_out",
      [](const at::Tensor& grad,
         const at::Tensor& output,
         int64_t dim,
         at::ScalarType input_dtype,
         bool log_softmax,
         at::Tensor grad_input) {
        return at::native::host_trace_softmax_backward_out(
            grad, output, dim, input_dtype, log_softmax, grad_input);
      });
  // the converted nll_loss hosts (Loss.cu): the entries allocate the outputs
  // through the trace mode, so they are traced roots, and call the hosts
  m.def(
      "_host_trace_nll_loss_forward_out",
      [](const at::Tensor& self,
         const at::Tensor& target,
         const std::optional<at::Tensor>& weight,
         int64_t reduction,
         int64_t ignore_index,
         at::Tensor output,
         at::Tensor total_weight) {
        return at::native::host_trace_nll_loss_forward_out(
            self,
            target,
            weight,
            reduction,
            ignore_index,
            output,
            total_weight);
      });
  m.def(
      "_host_trace_nll_loss_backward_out",
      [](const at::Tensor& grad_output,
         const at::Tensor& self,
         const at::Tensor& target,
         const std::optional<at::Tensor>& weight,
         int64_t reduction,
         int64_t ignore_index,
         const at::Tensor& total_weight,
         at::Tensor grad_input) {
        return at::native::host_trace_nll_loss_backward_out(
            grad_output,
            self,
            target,
            weight,
            reduction,
            ignore_index,
            total_weight,
            grad_input);
      });
  m.def(
      "_host_trace_ti_native_dropout",
      [](const at::Tensor& self, double p, std::optional<bool> train) {
        return ti::native_dropout_traced(self, p, train);
      });
  // foreach / fused optimizers over the traced multi_tensor_apply host
  // (ti/ForeachOps.h): in place on the lists, the scalars Python numbers
  m.def(
      "_host_trace_foreach_add_scalar_",
      [](const std::vector<at::Tensor>& tensors, const py::handle& scalar) {
        ti::foreach_add_scalar_traced_(tensors, scalar_arg(scalar));
      });
  m.def(
      "_host_trace_foreach_add_list_",
      [](const std::vector<at::Tensor>& tensors1,
         const std::vector<at::Tensor>& tensors2,
         const py::handle& alpha) {
        ti::foreach_add_list_traced_(tensors1, tensors2, scalar_arg(alpha));
      });
  m.def(
      "_host_trace_fused_adamw_",
      [](const std::vector<at::Tensor>& params,
         const std::vector<at::Tensor>& grads,
         const std::vector<at::Tensor>& exp_avgs,
         const std::vector<at::Tensor>& exp_avg_sqs,
         const std::vector<at::Tensor>& max_exp_avg_sqs,
         const std::vector<at::Tensor>& state_steps,
         const std::optional<at::Tensor>& lr_tensor,
         double lr,
         double beta1,
         double beta2,
         double weight_decay,
         double eps,
         bool amsgrad,
         bool maximize,
         const std::optional<at::Tensor>& grad_scale,
         const std::optional<at::Tensor>& found_inf) {
        ti::fused_adamw_traced_(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            lr_tensor,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            amsgrad,
            maximize,
            grad_scale,
            found_inf);
      });
  // the in-place distributions: the generator argument is the op's (None or
  // a torch.Generator); the bounds are the op's numbers
  m.def(
      "_host_trace_ti_random_from_to",
      [](at::Tensor self,
         int64_t from,
         std::optional<int64_t> to,
         const std::optional<at::Generator>& generator) {
        return ti::random_from_to_traced(self, from, to, generator);
      });
  m.def(
      "_host_trace_ti_random",
      [](at::Tensor self, const std::optional<at::Generator>& generator) {
        return ti::random_traced(self, generator);
      });
  m.def(
      "_host_trace_ti_uniform",
      [](at::Tensor self,
         double from,
         double to,
         const std::optional<at::Generator>& generator) {
        return ti::uniform_traced(self, from, to, generator);
      });
  m.def(
      "_host_trace_ti_normal",
      [](at::Tensor self,
         double mean,
         double std,
         const std::optional<at::Generator>& generator) {
        return ti::normal_traced(self, mean, std, generator);
      });
  m.def(
      "_host_trace_ti_bernoulli_scalar",
      [](at::Tensor self,
         double p,
         const std::optional<at::Generator>& generator) {
        return ti::bernoulli_scalar_traced(self, p, generator);
      });
  m.def(
      "_host_trace_ti_bernoulli_tensor",
      [](at::Tensor self,
         const at::Tensor& p,
         const std::optional<at::Generator>& generator) {
        return ti::bernoulli_tensor_traced(self, p, generator);
      });
  m.def(
      "_host_trace_ti_exponential",
      [](at::Tensor self,
         double lambda,
         const std::optional<at::Generator>& generator) {
        return ti::exponential_traced(self, lambda, generator);
      });
  m.def(
      "_host_trace_ti_geometric",
      [](at::Tensor self,
         double p,
         const std::optional<at::Generator>& generator) {
        return ti::geometric_traced(self, p, generator);
      });
  m.def(
      "_host_trace_ti_cauchy",
      [](at::Tensor self,
         double median,
         double sigma,
         const std::optional<at::Generator>& generator) {
        return ti::cauchy_traced(self, median, sigma, generator);
      });
  m.def(
      "_host_trace_ti_log_normal",
      [](at::Tensor self,
         double mean,
         double std,
         const std::optional<at::Generator>& generator) {
        return ti::log_normal_traced(self, mean, std, generator);
      });
  // reductions: dims=[] reduces every dim; the output has the input's dtype
  m.def(
      "_host_trace_ti_sum",
      [](const at::Tensor& self,
         const std::vector<int64_t>& dims,
         bool keepdim,
         const std::optional<at::Tensor>& out) {
        return ti::sum_traced(self, dims, keepdim, out);
      },
      py::arg("self"),
      py::arg("dims"),
      py::arg("keepdim"),
      py::arg("out") = std::nullopt);
  m.def(
      "_host_trace_ti_mean",
      [](const at::Tensor& self,
         const std::vector<int64_t>& dims,
         bool keepdim) { return ti::mean_traced(self, dims, keepdim); });
  m.def(
      "_host_trace_ti_amax",
      [](const at::Tensor& self,
         const std::vector<int64_t>& dims,
         bool keepdim) { return ti::amax_traced(self, dims, keepdim); });
  m.def(
      "_host_trace_ti_allany",
      [](const at::Tensor& self,
         const std::vector<int64_t>& dims,
         bool keepdim,
         bool all_of) {
        return ti::allany_traced(self, dims, keepdim, all_of);
      });
  // the generated siblings (torchgen over ti/siblings.yaml and add's
  // ufunc_inner_loop): _host_trace_ti_gen_<op> and their table
  host_trace_sibling_bindings(m, operand, scalar_arg);
}
