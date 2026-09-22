#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/cuda/PeerToPeerAccess.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/host_trace/HostTable.h>
#include <ATen/cuda/host_trace/ti/LoopsSym.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
#include <cuda_fp8.h>
#endif

namespace at::native {

namespace {

// Initial pool size for CUDA events per device.
constexpr size_t kInitialEventPoolSize = 8;

at::cuda::CUDAEventPool::Event getEventFromPool(const at::DeviceIndex device_idx) {
  // Pre-populate the pool with events to avoid stalls in creating events
  static auto* event_pool = new at::cuda::CUDAEventPool(kInitialEventPoolSize);
  return event_pool->get(device_idx);
}

} // namespace

void neg_kernel_cuda(TensorIteratorBase &iter);
void conj_kernel_cuda(TensorIteratorBase &iter);

// the copy lambdas as named functors: the traced sibling at the end of this
// file instantiates the kernels through them, so both hosts launch the one
// instantiation of this translation unit
template <typename scalar_t>
struct CopyFunctor {
  __device__ scalar_t operator()(scalar_t x) const {
    return x;
  }
};

template <typename from_t, typename to_t>
struct CastCopyFunctor {
  __device__ to_t operator()(from_t value) const {
    return static_cast<to_t>(value);
  }
};

void float16_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, CastCopyFunctor<float, at::Half>{});
}

void bfloat16_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, CastCopyFunctor<float, at::BFloat16>{});
}

void bfloat16tofloat32_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, CastCopyFunctor<at::BFloat16, float>{});
}
void float16tofloat32_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, CastCopyFunctor<at::Half, float>{});
}

template <typename SrcT>
struct ConvertToFloat8E4M3fnOp {
  __device__ __forceinline__ Float8_e4m3fn operator()(SrcT value) const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    __nv_fp8_storage_t x;
    if constexpr (std::is_same_v<SrcT, float>) {
      x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    } else if constexpr (std::is_same_v<SrcT, Half>) {
      x = __nv_cvt_halfraw_to_fp8(static_cast<__half>(value), __NV_SATFINITE, __NV_E4M3);
    } else if constexpr (std::is_same_v<SrcT, BFloat16>) {
      x = __nv_cvt_bfloat16raw_to_fp8(static_cast<__nv_bfloat16>(value), __NV_SATFINITE, __NV_E4M3);
    } else {
      x = __nv_cvt_float_to_fp8(static_cast<float>(value), __NV_SATFINITE, __NV_E4M3);
    }
    return Float8_e4m3fn(x, Float8_e4m3fn::from_bits());
#else
    return Float8_e4m3fn(value);
#endif
  }
};

// e5m2 intrinsics are correct but slower; only used for float on Blackwell
// to work around the ptxas subnormal codegen bug.
struct ConvertFloatToFloat8E5M2Op {
  __device__ __forceinline__ Float8_e5m2 operator()(float value) const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13020 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    auto x = __nv_cvt_float_to_fp8(value, __NV_NOSAT, __NV_E5M2);
    return Float8_e5m2(x, Float8_e5m2::from_bits());
#else
    return Float8_e5m2(value);
#endif
  }
};

void float8_copy_kernel_cuda(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  ScalarType other_dtype = iter.dtype(1);
  if (dtype == kFloat8_e4m3fn) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, ConvertToFloat8E4M3fnOp<float>{});
         break;
      case kHalf:
         gpu_kernel_nocast(iter, ConvertToFloat8E4M3fnOp<Half>{});
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, ConvertToFloat8E4M3fnOp<BFloat16>{});
         break;
      default:
        gpu_kernel(iter, [] GPU_LAMBDA(Float8_e4m3fn x) { return x; });
        break;
    }
  } else if (dtype == kFloat8_e5m2) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, ConvertFloatToFloat8E5M2Op{});
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e5m2(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e5m2(value);
         });
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e5m2 x) { return x; });
         break;
    }
  } else if (dtype == kFloat8_e4m3fnuz) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e4m3fnuz(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e4m3fnuz(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e4m3fnuz(value);
         });
         break;
      default:
        gpu_kernel(iter, [] GPU_LAMBDA(Float8_e4m3fnuz x) { return x; });
        break;
    }
  } else if (dtype == kFloat8_e5m2fnuz) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e5m2fnuz(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e5m2fnuz(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e5m2fnuz(value);
         });
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e5m2fnuz x) { return x; });
         break;
    }
  } else if (dtype == kFloat8_e8m0fnu) {
    // TODO(#146647): clean this up, too much copy-pasta
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e8m0fnu(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e8m0fnu(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e8m0fnu(value);
         });
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e8m0fnu x) { return x; });
         break;
    }
  } else {
    TORCH_CHECK(false, "This supposed to be called only for Float8 types");
  }
}

// TODO: We probably can use the opaque type trick to avoid creating duplicate
// kernels for equivalent bit lengths
void direct_copy_kernel_cuda(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  if (isQIntType(dtype)) {
    AT_DISPATCH_QINT_TYPES(dtype, "copy_", [&] {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else if (isFloat8Type(dtype)) {
     float8_copy_kernel_cuda(iter);
  } else if (iter.dtype(1) == kFloat && (dtype == kBFloat16 || dtype == kHalf)) {
     if (dtype == kBFloat16) {
       bfloat16_copy_kernel_cuda(iter);
     } else {
       float16_copy_kernel_cuda(iter);
     }
  }
  else if ((iter.dtype(1) == kBFloat16 || iter.dtype(1) == kHalf) && dtype == kFloat) {
    if (iter.dtype(1) == kBFloat16) {
      bfloat16tofloat32_copy_kernel_cuda(iter);
    } else {
      float16tofloat32_copy_kernel_cuda(iter);
    }
  }
  else if (isBitsType(dtype)) {
    TORCH_CHECK(dtype == iter.dtype(1), "copy_() does not support casting "
      "bits types to different bits types. Source dtype is ", iter.dtype(1), "target dtype is ", dtype);
    AT_DISPATCH_BIT_TYPES(dtype, "copy_", [&] {
      gpu_kernel_nocast(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else if (dtype == ScalarType::Float4_e2m1fn_x2) {
    TORCH_CHECK(dtype == iter.dtype(1), "copy_() does not support casting "
      "Float4_e2m1fn_x2 to different types. Source dtype is ", iter.dtype(1), "target dtype is ", dtype);
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(Float4_e2m1fn_x2 x) { return x; });
  } else {
    AT_DISPATCH_V2(
        dtype, "copy_", AT_WRAP([&] {
          gpu_kernel(iter, CopyFunctor<scalar_t>{});
    }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kHalf, kBool, kBFloat16, kComplexHalf, kBComplex32, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
}

void neg_conj_kernel_cuda(TensorIteratorBase &iter) {
  AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_cuda", [&] {
    gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return -std::conj(x); });
  });
}

using namespace at::cuda;

// device-to-device copy, does type conversion
void copy_device_to_device(TensorIterator& iter,
                           bool non_blocking,
                           bool p2p_enabled) {
  int64_t numel = iter.numel();

  // We can memcpy the memory if both tensors have the same type AND both
  // tensors are contiguous after dimension coalescing and reordering.
  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool same_conj = iter.tensor(0).is_conj() == iter.tensor(1).is_conj();
  bool same_neg = iter.tensor(0).is_neg() == iter.tensor(1).is_neg();
  bool memcpy_eligible = same_type && same_conj && same_neg && iter.is_contiguous();

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  CUDAGuard device_guard(src_device);

  // We always perform the copy on the source device, using the current stream
  // on the source device, and we fully synchronize on both src and dst's
  // current streams for completion of the copy. We have to explicitly do this
  // for non-contig copies. This mimics the behavior of cross-device
  // cudaMemcpyAsync on the default stream.
  CUDAStream copy_stream = getCurrentCUDAStream(src_device.index());
  if (src_device != dst_device) {
    // This is a cross-device copy on the src current stream and dst current
    // stream. We perform a two-way barrier between both devices' streams
    // before the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are handled, so
    // that no one is operating on the dst memory when we perform the copy.
    // src waits on dst barrier (src already waits on src)

    // Use event pool for better performance instead of creating new events
    auto dst_ready = getEventFromPool(dst_device.index());
    device_guard.set_device(dst_device);
    dst_ready->record(getCurrentCUDAStream(dst_device.index()));

    device_guard.set_device(src_device);
    dst_ready->block(copy_stream);
  }

  if (memcpy_eligible) {
    void *dst = iter.data_ptr(0);
    void *src = iter.data_ptr(1);
    size_t size = numel * iter.element_size(0);
    if (src != dst || src_device != dst_device) {
      // Due to bizarre cuda driver intricacies, copies of
      // cudaMallocAsynced memory between devices that aren't
      // peer-to-peer-capable need "cudaMemcpyPeerAsync".
      // So we let the allocator implement the correct call
      // (either cudaMemcpyAsync or cudaMemcpyPeerAsync)
      AT_CUDA_CHECK(CUDACachingAllocator::memcpyAsync(
        dst, dst_device.index(),
        src, src_device.index(),
        size, copy_stream, p2p_enabled));
    }
  } else {
    if (same_neg) {
      if (!same_conj) {
        conj_kernel_cuda(iter);
      } else {
        direct_copy_kernel_cuda(iter);
      }
    } else {
      if (!same_conj) {
        neg_conj_kernel_cuda(iter);
      } else {
        neg_kernel_cuda(iter);
      }
    }
  }

  if (src_device != dst_device) {
    // dst waits on src barrier (dst already waits on dst). We cannot
    // operate on dst's copy until the copy is complete.

    // Still on src_device, record stream event
    auto src_ready = getEventFromPool(src_device.index());
    src_ready->record(copy_stream);

    device_guard.set_device(dst_device);
    src_ready->block(getCurrentCUDAStream(dst_device.index()));
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

static bool copy_requires_temporaries(TensorIterator& iter, bool p2p_enabled) {
  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  if (dst_device == src_device) {
    // We never require temporaries for copies on the same GPU.
    TORCH_INTERNAL_ASSERT(dst_device.is_cuda() && src_device.is_cuda());
    return false;
  }

  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    // Contiguous same-dtype copies can always use cudaMemcpyAsync
    return false;
  } else if (dst_device.is_cuda() && src_device.is_cuda()) {
    // Copies between GPUs can use the copy kernel if P2P is supported
    return !p2p_enabled;
  } else {
    // The remaining cases require temporaries. For example, this includes
    // non-contiguous copies between CPU and GPU.
    return true;
  }
}

static bool maybe_enable_p2p_access(Device dst_device, Device src_device) {
  if (dst_device.is_cpu() || src_device.is_cpu()) {
    return false;
  }
  return at::cuda::get_p2p_access(src_device.index(), dst_device.index());
}

static void copy_kernel_cuda(TensorIterator& iter, bool non_blocking) {
  TORCH_CHECK(iter.ntensors() == 2);

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  // Enable p2p access between devices. (No-op if it involves the CPU)
  bool p2p_enabled = maybe_enable_p2p_access(dst_device, src_device);

  if (copy_requires_temporaries(iter, p2p_enabled)) {
    // NB: this involves recursive calls to copy. Be careful that those copies
    // don't require temporaries or you will cause an infinite recursion!
    auto& dst = iter.tensor(0);
    Tensor dst_contig;
    Tensor src_contig;

    // If non_blocking is true - type conversions are performed on the GPU
    // For blocking transfers conversions are performed on CPU to avoid allocating
    // extra GPU memory
    // for GPU-GPU transfers conversions are performed on the source device
    auto conversion_device = non_blocking ? kCUDA : kCPU;
    if (iter.device_type(1) == conversion_device) {
      dst_contig = dst.is_contiguous() ? dst : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).to(iter.dtype(0)).expand_as(dst).contiguous();
    } else {
      bool same_type = iter.dtype(0) == iter.dtype(1);
      dst_contig = (dst.is_contiguous() && same_type) ? dst : at::empty_like(dst, iter.dtype(1), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).expand_as(dst).contiguous();
    }

    // propagate the correct conjugate bit
    dst_contig._set_conj(dst.is_conj());
    src_contig._set_conj(iter.tensor(1).is_conj());

    dst_contig._set_neg(dst.is_neg());
    src_contig._set_neg(iter.tensor(1).is_neg());

    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    dst_contig.copy_(src_contig, non_blocking);

    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
      TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
      dst.copy_(dst_contig, non_blocking);
    }
    return;
  }

  // Copy on GPU (or between GPUs)
  if (dst_device.is_cuda() && src_device.is_cuda()) {
    copy_device_to_device(iter, non_blocking, p2p_enabled);
    return;
  }

  // Copy between CPU and GPU
  cuda::OptionalCUDAGuard device_guard;
  cudaMemcpyKind kind;
  const Tensor* host_tensor = nullptr;
  if (dst_device.is_cuda() && src_device.is_cpu()) {
    device_guard.set_device(dst_device);
    kind = cudaMemcpyHostToDevice;
    host_tensor = &iter.tensor(1);
  } else if (dst_device.is_cpu() && src_device.is_cuda()) {
    device_guard.set_device(src_device);
    kind = cudaMemcpyDeviceToHost;
    host_tensor = &iter.tensor(0);
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in GPU copy_()");
  }

  // Check for unpinned CPU memory during CUDA graph capture
  if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
    TORCH_CHECK(
        host_tensor->is_pinned(),
        "Cannot copy between CPU and CUDA tensors during CUDA graph capture ",
        "unless the CPU tensor is pinned. Please use tensor.pin_memory() or ",
        "allocate the tensor with pin_memory=True.");
  }

  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  int64_t nbytes = iter.numel() * iter.element_size(0);
  CUDAStream stream = getCurrentCUDAStream();

  if (non_blocking) {
    AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));
    // we use both the storage context and the tensor data pointer as the key
    // for the caching host allocator. This allows us to better attribute the
    // events to the original tensor allocation correctly. The cases we seek to
    // handle are:

    // 1: a user can pass a pinned memory tensor with an alternative
    // context, for example if allocating memory directly from the pinned memory
    // allocator and constructing a tensor with torch::from_blob.

    // 2: a user can pass a tensor with a different base pointer to the original
    // allocation (via slicing).
    const auto& dst_tensor = iter.tensor(0);
    const auto& src_tensor = iter.tensor(1);
    const auto& host_tensor = (dst_device == kCPU ? dst_tensor : src_tensor);
    auto* ptr = (dst_device == kCPU ? dst : src);
    auto* ctx = host_tensor.storage().data_ptr().get_context();
    // TODO: warn on the return value.
    at::getHostAllocator(at::kCUDA)->record_event(ptr, ctx, stream.unwrap());
  } else {
    at::cuda::memcpy_and_sync(dst, src, nbytes, kind, stream);
  }

  if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
     iter.tensor(0).conj_physical_();
  }
  if (iter.tensor(0).is_neg() != iter.tensor(1).is_neg()) {
     iter.tensor(0).neg_();
  }
}

REGISTER_DISPATCH(copy_stub, &copy_kernel_cuda)

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of copy_device_to_device's
// kernel branch (direct_copy_kernel_cuda for two CUDA tensors). It lives in this translation
// unit so that its launches are the instantiations the host above makes: the replay's kernel
// nodes hold the function handle eager's do, not only its name. A pair of one dtype runs
// CopyFunctor; the real op copies a contiguous pair of one dtype with a cudaMemcpyAsync
// rather than a kernel, and so does the sibling, through copy_d2d (HostTable.h): a memcpy
// record on the tape, a memcpy node in the replay's capture. A pair of two dtypes is the real
// op's cast: float <-> Half / BFloat16 through the four CastCopyFunctor no-cast launches,
// every other pair through the dynamic-cast path of gpu_kernel dispatched on the destination
// dtype. Quantized, float8, bits and complex destinations decline by name.
namespace at::cuda::host_trace::ti {

Tensor& copy_traced(Tensor& dst, const Tensor& src) {
  if (dst.is_conj() || src.is_conj() || dst.is_neg() || src.is_neg()) {
    decline("host_trace: copy_ with a conjugate or negative bit is not traced (declined)");
  }
  const ScalarType dtype = dst.scalar_type();
  const ScalarType other = src.scalar_type();
  if (isQIntType(dtype) || isFloat8Type(dtype) || isBitsType(dtype) || dtype == ScalarType::Float4_e2m1fn_x2 ||
      isComplexType(dtype) || isComplexType(other)) {
    decline(c10::str("host_trace: copy_ from ", other, " to ", dtype, " is not traced (declined)"));
  }
  TensorIteratorSymConfig config;
  config.resize_outputs_ = false;
  config.check_all_same_dtype_ = false;
  TensorIteratorSym iter;
  iter.add_output(dst);
  iter.add_input(src);
  iter.build(config);
  // the route (one memcpy, the copy kernel, a cast kernel) and its tests
  KernelChoice choice;
  if (dtype == other) {
    if (iter.is_contiguous()) {
      // copy_device_to_device's memcpy_eligible branch: one cudaMemcpyAsync of
      // the whole extent, nothing when the two addresses are one (each
      // comparison a guard of the trace)
      if (iter.numel() != 0 && iter.data_ptr(0) != iter.data_ptr(1)) {
        copy_d2d(
            iter.data_ptr(0),
            iter.data_ptr(1),
            iter.numel() * iter.element_size(0),
            at::cuda::getCurrentCUDAStream());
      }
      return dst;
    }
    AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, dtype, "copy_traced", [&] {
      gpu_kernel(iter, at::native::CopyFunctor<scalar_t>{});
    });
    return dst;
  }
  if (other == kFloat && dtype == kBFloat16) {
    gpu_kernel(iter, at::native::CastCopyFunctor<float, at::BFloat16>{});
  } else if (other == kFloat && dtype == kHalf) {
    gpu_kernel(iter, at::native::CastCopyFunctor<float, at::Half>{});
  } else if (other == kBFloat16 && dtype == kFloat) {
    gpu_kernel(iter, at::native::CastCopyFunctor<at::BFloat16, float>{});
  } else if (other == kHalf && dtype == kFloat) {
    gpu_kernel(iter, at::native::CastCopyFunctor<at::Half, float>{});
  } else {
    AT_DISPATCH_V2(dtype, "copy_traced", AT_WRAP([&] {
      gpu_kernel(iter, at::native::CopyFunctor<scalar_t>{});
    }), AT_EXPAND(AT_ALL_TYPES), kHalf, kBool, kBFloat16, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
  return dst;
}

} // namespace at::cuda::host_trace::ti
