#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native { namespace {

enum class OpType {GE, GT, LE, LT};

template<typename scalar_t>
struct CompareFunctor{
  constexpr CompareFunctor(OpType op): op_(op) {};
  OpType op_;
  __device__ __forceinline__ bool operator() (scalar_t a, scalar_t b) const {
    if (op_ == OpType::GE) {
      return a >= b;
    } else if (op_ == OpType::GT) {
      return a > b;
    } else if (op_ == OpType::LE) {
      return a <= b;
    } else { //LT
      return a < b;
    }
  }
};

// compare_scalar_kernel's device function: the comparison and the CPU scalar, as its lambda
// captured them
template <typename scalar_t>
struct CompareScalarFunctor {
  CompareFunctor<scalar_t> f;
  scalar_t rhs;
  __device__ bool operator()(scalar_t lhs) const {
    return f(lhs, rhs);
  }
};

// Reflects the comparison operator, so reflect(op)(a, b) == op(b, a)
OpType reflect(OpType x) {
  switch (x) {
    case OpType::GE: return OpType::LE;
    case OpType::GT: return OpType::LT;
    case OpType::LE: return OpType::GE;
    case OpType::LT: return OpType::GT;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid OpType");
}

}  // namespace (anonymous)

template <typename scalar_t>
void compare_scalar_kernel(TensorIteratorBase &iter, OpType op, scalar_t rhs) {
  CompareFunctor<scalar_t> f(op);
  gpu_kernel(iter, CompareScalarFunctor<scalar_t>{f, rhs});
}

template <typename scalar_t>
void compare_kernel_impl(TensorIteratorBase &iter, OpType op) {
  // If either input is a cpu scalar, perform the equivalent comparison
  // where the scalar is on the right hand side. This saves us from
  // generating two otherwise identical kernels with mirrored
  // arguments.
  if (iter.is_cpu_scalar(1)) {
    const scalar_t lhs = iter.scalar_value<scalar_t>(1);
    iter.remove_operand(1);
    const DeviceGuard device_guard(iter.device(1));
    compare_scalar_kernel(iter, reflect(op), lhs);
  } else if (iter.is_cpu_scalar(2)) {
    const scalar_t rhs = iter.scalar_value<scalar_t>(2);
    iter.remove_operand(2);
    compare_scalar_kernel(iter, op, rhs);
  } else {
    CompareFunctor<scalar_t> f(op);
    gpu_kernel(iter, f);
  }
}

C10_NOINLINE void compare_kernel_with_scalars(TensorIteratorBase &iter, OpType op) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "compare_cuda", [&]() {
    compare_kernel_impl<scalar_t>(iter, op);
  });
}


void ge_kernel_cuda(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::GE);
}

void gt_kernel_cuda(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::GT);
}

void le_kernel_cuda(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::LE);
}

void lt_kernel_cuda(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::LT);
}

REGISTER_DISPATCH(ge_stub, &ge_kernel_cuda)
REGISTER_DISPATCH(gt_stub, &gt_kernel_cuda)
REGISTER_DISPATCH(le_stub, &le_kernel_cuda)
REGISTER_DISPATCH(lt_stub, &lt_kernel_cuda)

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of compare_kernel_impl, compiled here
// so the sibling and the real host above instantiate the one kernel over CompareFunctor / CompareScalarFunctor
// (DECISIONS E36): the tape's launch is eager's function object, not a twin. Outside a trace the
// entry runs the same launches in ordinary mode, which is how the parity test compares it with
// the real op.
#include <ATen/cuda/host_trace/ti/LoopsSym.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>

namespace at::cuda::host_trace::ti {

// eq / ne / lt / le / gt / ge for two CUDA tensors of one dtype, or one CUDA tensor and a CPU
// scalar (a scalar on the left of an ordered comparison is reflected to the right, as
// compare_kernel_impl does). The output is bool. Complex, float8 and bits dtypes decline by name.
Tensor compare_traced(const Tensor& self, const Tensor& other, std::string_view op) {
  if (op == "eq" || op == "ne") {
    return compare_eq_ne_traced(self, other, op == "eq");
  }
  TensorIteratorSym iter = TensorIteratorSym::comparison_op(Tensor(), self, other);
  const ScalarType dtype = iter.common_dtype();
  if (isComplexType(dtype) || isFloat8Type(dtype) || isBitsType(dtype) || dtype == ScalarType::Float4_e2m1fn_x2) {
    decline(c10::str("host_trace: ", op, " on ", dtype, " is not traced (declined)"));
  }
  using at::native::OpType;
  OpType kind = OpType::LT;
  if (op == "ge") {
    kind = OpType::GE;
  } else if (op == "gt") {
    kind = OpType::GT;
  } else if (op == "le") {
    kind = OpType::LE;
  } else {
    TORCH_CHECK(op == "lt", "host_trace: unknown comparison ", op);
  }
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, dtype, "compare_traced", [&] {
    using F = at::native::CompareFunctor<scalar_t>;
    if (iter.is_cpu_scalar(1)) {
      Zeroed<at::native::CompareScalarFunctor<scalar_t>> f(F(at::native::reflect(kind)), iter.scalar_value<scalar_t>(1));
      iter.remove_operand(1);
      gpu_kernel(iter, f.get());
    } else if (iter.is_cpu_scalar(2)) {
      Zeroed<at::native::CompareScalarFunctor<scalar_t>> f(F(kind), iter.scalar_value<scalar_t>(2));
      iter.remove_operand(2);
      gpu_kernel(iter, f.get());
    } else {
      gpu_kernel(iter, F(kind));
    }
  });
  return iter.output();
}

} // namespace at::cuda::host_trace::ti
