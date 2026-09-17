#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native { namespace {

enum class EqOpType {EQ, NE};

template<typename scalar_t>
struct CompareEqFunctor{
  CompareEqFunctor(EqOpType op): op_(op) {}
  const EqOpType op_;
  __device__ __forceinline__ bool operator() (scalar_t a, scalar_t b) const {
    if (op_ == EqOpType::EQ) {
      return a == b;
    } else { //NE
      return a != b;
    }

  }
 };
}

C10_NOINLINE void compare_eq_ne_kernel(TensorIteratorBase &iter, EqOpType op) {
  AT_DISPATCH_V2(iter.common_dtype(), "compare_eq_ne_cuda", AT_WRAP([&]() {
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
        iter, CompareEqFunctor<scalar_t>(op));
  }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kBComplex32, kHalf, kBFloat16, kBool, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kFloat4_e2m1fn_x2);
}

void eq_kernel_cuda(TensorIteratorBase& iter) {
  compare_eq_ne_kernel(iter, EqOpType::EQ);
}

void ne_kernel_cuda(TensorIteratorBase& iter) {
  compare_eq_ne_kernel(iter, EqOpType::NE);
}

REGISTER_DISPATCH(eq_stub, &eq_kernel_cuda)
REGISTER_DISPATCH(ne_stub, &ne_kernel_cuda)

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of compare_eq_ne_kernel, compiled here
// so the sibling and the real host above instantiate the one kernel over CompareEqFunctor
// (DECISIONS E36): the tape's launch is eager's function object, not a twin. Outside a trace the
// entry runs the same launches in ordinary mode, which is how the parity test compares it with
// the real op.
#include <ATen/cuda/host_trace/ti/LoopsSym.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>

namespace at::cuda::host_trace::ti {

// eq / ne for two CUDA tensors of one dtype or one CUDA tensor and a CPU scalar, launched as
// compare_eq_ne_kernel launches them (BinaryFunctor / AUnaryFunctor over CompareEqFunctor through
// the symmetric scalar helper). The output is bool; complex, float8 and bits dtypes decline by
// name. compare_traced (CompareKernels.cu) routes here.
Tensor compare_eq_ne_traced(const Tensor& self, const Tensor& other, bool equal) {
  TensorIteratorSym iter = TensorIteratorSym::comparison_op(Tensor(), self, other);
  const ScalarType dtype = iter.common_dtype();
  if (isComplexType(dtype) || isFloat8Type(dtype) || isBitsType(dtype) || dtype == ScalarType::Float4_e2m1fn_x2) {
    decline(c10::str("host_trace: ", equal ? "eq" : "ne", " on ", dtype, " is not traced (declined)"));
  }
  const at::native::EqOpType op = equal ? at::native::EqOpType::EQ : at::native::EqOpType::NE;
  AT_DISPATCH_V2(dtype, "compare_eq_ne_traced", AT_WRAP([&]() {
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(iter, at::native::CompareEqFunctor<scalar_t>(op));
  }), AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16, kBool, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  return iter.output();
}

} // namespace at::cuda::host_trace::ti
