// The elementwise ops that opt into the traced sibling iterator. Each entry
// does what the op's CUDA kernel host does around gpu_kernel, on the sibling
// iterator, and is called by the trace mode in place of the real op; outside a
// trace it runs the same launches in ordinary mode, which is how the parity
// test compares it with the real op. An entry is defined in eager's own .cu
// beside the host it mirrors, so both instantiate one kernel over eager's
// functor (DECISIONS E36); add's is generated into UfuncCUDA_add.cu with
// eager's host (HostTraceSiblingOps.h declares it).
#pragma once
#include <optional>
#include <tuple>
#include <ATen/core/Generator.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>

#include <string_view>

namespace at::cuda::host_trace::ti {

// mul / div take one CPU scalar operand (a wrapped Python number) on either
// side as the real kernel hosts do (add likewise, through its generated
// entry; sub and rsub are add with -alpha, BinaryOps.cpp sub_out / rsub,
// composed in the registry). A defined `out` is the structured kernel's
// out= / in-place form: the launch writes it (an in-place op passes self), an
// undefined one is allocated by the iterator
TORCH_CUDA_CU_API Tensor mul_traced(const Tensor& self, const Tensor& other, const Tensor& out = {});
TORCH_CUDA_CU_API Tensor div_traced(const Tensor& self, const Tensor& other, const Tensor& out = {});
TORCH_CUDA_CU_API Tensor silu_traced(const Tensor& self);
TORCH_CUDA_CU_API Tensor gelu_traced(const Tensor& self, std::string_view approximate);
// native_dropout (train mode): (output, bool mask); the philox increment is
// declared on the tape as an expression of the element count
TORCH_CUDA_CU_API std::tuple<Tensor, Tensor> native_dropout_traced(const Tensor& self, double p, std::optional<bool> train);
TORCH_CUDA_CU_API Tensor& copy_traced(Tensor& dst, const Tensor& src);
// the in-place distributions (DistributionTemplates.h's entries in each
// Distribution*.cu): the generator is the op's argument, the default CUDA
// generator or none (another declines by name); the philox increment is
// declared on the tape as an expression of the element count
TORCH_CUDA_CU_API Tensor& random_from_to_traced(Tensor& self, int64_t from, std::optional<int64_t> to, const std::optional<at::Generator>& gen);
TORCH_CUDA_CU_API Tensor& random_traced(Tensor& self, const std::optional<at::Generator>& gen);
TORCH_CUDA_CU_API Tensor& uniform_traced(Tensor& self, double from, double to, const std::optional<at::Generator>& gen);
TORCH_CUDA_CU_API Tensor& normal_traced(Tensor& self, double mean, double std, const std::optional<at::Generator>& gen);
TORCH_CUDA_CU_API Tensor& bernoulli_scalar_traced(Tensor& self, double p, const std::optional<at::Generator>& gen);
TORCH_CUDA_CU_API Tensor& bernoulli_tensor_traced(Tensor& self, const Tensor& p, const std::optional<at::Generator>& gen);
TORCH_CUDA_CU_API Tensor& exponential_traced(Tensor& self, double lambda, const std::optional<at::Generator>& gen);
TORCH_CUDA_CU_API Tensor& geometric_traced(Tensor& self, double p, const std::optional<at::Generator>& gen);
TORCH_CUDA_CU_API Tensor& cauchy_traced(Tensor& self, double median, double sigma, const std::optional<at::Generator>& gen);
TORCH_CUDA_CU_API Tensor& log_normal_traced(Tensor& self, double mean, double std, const std::optional<at::Generator>& gen);
TORCH_CUDA_CU_API Tensor reciprocal_traced(const Tensor& self);
TORCH_CUDA_CU_API Tensor tanh_traced(const Tensor& self);
TORCH_CUDA_CU_API Tensor sqrt_traced(const Tensor& self);
// pow.Tensor_Scalar's kernel host (PowKernel.cu pow_tensor_scalar_kernel) for
// a floating base: 0.5 / -0.5 / -1 route to sqrt / rsqrt / reciprocal, 2, 3
// and -2 are closed forms, any other exponent is captured in the base's type
TORCH_CUDA_CU_API Tensor pow_tensor_scalar_traced(const Tensor& self, const Scalar& exponent);
// fill_ (FillKernel.cu fill_kernel_cuda): the value baked into the functor
// as a constant of the variant; the factories (full, zeros, ones, *_like,
// new_*) are the allocation plus this in the registry, as in eager
TORCH_CUDA_CU_API Tensor& fill_traced(Tensor& self, const Scalar& value);
// zero_ (TensorFactories.cu zero_cuda_): a memset record over a dense tensor
// (the memset node eager's capture holds), fill_(0) over a strided one; the
// zero factories (zeros, zeros_like, new_zeros) end in it as in eager
TORCH_CUDA_CU_API Tensor& zero_traced(Tensor& self);
// arange_out's launch (RangeFactories.cu arange_cuda_out) into the
// contiguous 1-D `out` the registry sized and allocated: start and step are
// fields of the functor and may be symbolic (a SymInt for an integral out, a
// SymFloat for a floating one)
TORCH_CUDA_CU_API Tensor& arange_traced(const Scalar& start, const Scalar& step, Tensor& out);
// eq / ne / lt / le / gt / ge (CompareEQKernel.cu, CompareKernels.cu): a bool
// output over one dtype, one operand may be a CPU scalar
TORCH_CUDA_CU_API Tensor compare_traced(const Tensor& self, const Tensor& other, std::string_view op);
TORCH_CUDA_CU_API Tensor compare_eq_ne_traced(const Tensor& self, const Tensor& other, bool equal);
// masked_fill_ (Indexing.cu masked_fill__cuda): the mask expanded to self
TORCH_CUDA_CU_API Tensor& masked_fill_traced(Tensor& self, const Tensor& mask, const Scalar& value);
// clamp / clamp_min / clamp_max with scalar bounds (TensorCompare.cpp
// clamp_out and TensorCompare.cu launch_clamp_scalar); `out` as for add
TORCH_CUDA_CU_API Tensor clamp_scalar_traced(const Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max, const Tensor& out = {});
// where.self (TensorCompare.cpp where_self_out, TensorCompare.cu
// where_kernel_impl): a bool condition selecting between two operands of one
// dtype, dispatched on the element size as the real kernel is
TORCH_CUDA_CU_API Tensor where_traced(const Tensor& condition, const Tensor& self, const Tensor& other);

TORCH_CUDA_CU_API Tensor sin_traced(const Tensor& self);
TORCH_CUDA_CU_API Tensor cos_traced(const Tensor& self);
TORCH_CUDA_CU_API Tensor exp_traced(const Tensor& self);
TORCH_CUDA_CU_API Tensor rsqrt_traced(const Tensor& self);
TORCH_CUDA_CU_API Tensor neg_traced(const Tensor& self);

} // namespace at::cuda::host_trace::ti
