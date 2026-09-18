# Owner(s): ["module: cuda"]
"""The host-to-device test hosts: kernels that read their operands from a
device table the host filled and copied through HostTable / copy_h2d, and a
gather whose ids arrive from pinned memory; the negatives the h2d suite
asserts on. Built as a test extension against the installed headers
(host_trace_testing.load_test_extension), once per name and source under
TORCH_EXTENSIONS_DIR; nothing of it is in libtorch. Used by
test_cuda_host_trace_h2d.py, and by the decode and embed_cat suites."""

from host_trace_testing import load_test_extension


SOURCE = r"""
#include <torch/extension.h>

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/host_trace/HostTable.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/irange.h>

#include <vector>

// Group g multiplies a[g] and b[g] elementwise into the g-th slice of one
// output; the kernel reads the three addresses and the length of every group
// from an int64 / int32 table built on the host and copied to the device.
// out[i] = table[ids[i]]: `ids` is a pinned CPU int64 tensor copied to the
// device inside the host (the copy's source is the input itself).
// Negatives: a pointer table element set from a raw address (declines under
// a trace), and a raw cudaMemcpyAsync between two module-owned buffers (a
// memcpy node the tape has no record for). copy_count copies size(0) * 8
// bytes from a four-element table and returns the last element copied (in
// contract while size(0) <= 4, the byte count is a guard); copy_zero copies
// no bytes (declines: no node); copy_into is copy_h2d of src's bytes into
// dst, the pinned-tensor overload as a host would call it; copy_into_beside
// is the same copy inside a call that also has a CUDA argument, so it can be
// traced (a host destination declines at the trace); copy_twice copies one
// four-element table unchanged into two device buffers and returns both
// buffers' elements (n * 100 + i twice: one image per copy on the tape);
// copy_rewrite_copy writes slot 0, copies it, rewrites slot 0 and writes
// slot 1, copies both, and returns {first copy, second copy} (n * 100,
// n * 200, n * 300; the eager path waits for the first copy before the
// rewrite, as a pinned source requires; under a capture nothing has run and
// the images keep each copy's bytes); copy_on_stream issues the table copy on
// a pool stream, forked from the current stream and joined back or never
// forked (the kernel still waits for it), both declined under a trace.

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/host_trace/HostTable.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/irange.h>

namespace at::cuda::host_trace::h2d {

namespace {

__global__ void grouped_mul_kernel(
    const int64_t* ptrs,
    const int32_t* sizes,
    int groups) {
  const int g = blockIdx.x;
  if (g >= groups) {
    return;
  }
  const float* a = reinterpret_cast<const float*>(ptrs[3 * g]);
  const float* b = reinterpret_cast<const float*>(ptrs[3 * g + 1]);
  float* out = reinterpret_cast<float*>(ptrs[3 * g + 2]);
  const int n = sizes[g];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = a[i] * b[i];
  }
}

template <typename T>
__global__ void gather_kernel(
    const T* table,
    const int64_t* ids,
    T* out,
    int rows,
    int cols) {
  const int r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const int64_t id = ids[r];
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    out[static_cast<int64_t>(r) * cols + c] = table[id * cols + c];
  }
}

__global__ void read_last_kernel(const int64_t* table, int64_t* out, int n) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = table[n - 1];
  }
}

// out = a[0..na) ++ b[0..nb)
__global__ void read_two_kernel(
    const int64_t* a,
    int na,
    const int64_t* b,
    int nb,
    int64_t* out) {
  const int i = threadIdx.x;
  if (i < na) {
    out[i] = a[i];
  } else if (i < na + nb) {
    out[i] = b[i - na];
  }
}

} // namespace

at::Tensor grouped_mul(
    const std::vector<at::Tensor>& a,
    const std::vector<at::Tensor>& b) {
  const int64_t groups = static_cast<int64_t>(a.size());
  TORCH_CHECK(groups > 0 && b.size() == a.size(), "grouped_mul: a and b pair up");
  c10::SymInt total = 0;
  for (const auto g : c10::irange(groups)) {
    TORCH_CHECK(
        a[g].dim() == 1 && b[g].dim() == 1 && a[g].is_cuda() && b[g].is_cuda(),
        "grouped_mul: 1-D CUDA tensors");
    TORCH_CHECK(
        a[g].scalar_type() == at::kFloat && b[g].scalar_type() == at::kFloat,
        "grouped_mul: float32");
    TORCH_SYM_CHECK(
        a[g].sym_size(0).sym_eq(b[g].sym_size(0)), "grouped_mul: a[g] and b[g] have one length");
    total = total + a[g].sym_size(0);
  }
  at::Tensor out = at::empty_symint({total}, a[0].options());
  // the table: 3 addresses per group, then the lengths; one device buffer
  HostTable<const void*> ptrs(3 * groups, "grouped_ptrs");
  HostTable<int32_t> sizes(groups, "grouped_sizes");
  c10::SymInt off = 0;
  const c10::SymInt out_ptr = sym_mutable_data_ptr<float>(out);
  for (const auto g : c10::irange(groups)) {
    ptrs[3 * g] = sym_const_data_ptr<float>(a[g]);
    ptrs[3 * g + 1] = sym_const_data_ptr<float>(b[g]);
    ptrs[3 * g + 2] = out_ptr + off * 4;
    sizes[g] = a[g].sym_size(0);
    off = off + a[g].sym_size(0);
  }
  const int64_t ptr_bytes = 3 * groups * 8;
  at::Tensor dev = at::empty({ptr_bytes + groups * 4}, out.options().dtype(at::kByte));
  auto stream = at::cuda::getCurrentCUDAStream();
  const c10::SymInt dev_ptr = sym_mutable_data_ptr(dev);
  copy_h2d(dev_ptr, ptrs, ptrs.nbytes(), stream);
  copy_h2d(dev_ptr + ptr_bytes, sizes, sizes.nbytes(), stream);
  launch(
      grouped_mul_kernel,
      Grid(groups),
      256,
      c10::SymInt(0),
      stream.stream(),
      dev_ptr,
      dev_ptr + ptr_bytes,
      static_cast<int>(groups));
  return out;
}

template <typename T>
static void gather_launch(
    const at::Tensor& table,
    const at::Tensor& ids_dev,
    const at::Tensor& out,
    const c10::SymInt& rows,
    const c10::SymInt& cols,
    cudaStream_t stream) {
  launch(
      gather_kernel<T>,
      Grid(rows),
      128,
      c10::SymInt(0),
      stream,
      sym_const_data_ptr<T>(table),
      sym_const_data_ptr<int64_t>(ids_dev),
      sym_mutable_data_ptr<T>(out),
      rows,
      cols);
}

at::Tensor gather(const at::Tensor& table, const at::Tensor& ids) {
  // float32, float16 or bfloat16 table (a decode's embedding rows in the
  // attention dtype)
  const auto st = table.scalar_type();
  TORCH_CHECK(
      table.dim() == 2 && table.is_cuda() &&
          (st == at::kFloat || st == at::kHalf || st == at::kBFloat16),
      "gather: a 2-D float32 / float16 / bfloat16 CUDA table");
  TORCH_CHECK(
      ids.dim() == 1 && ids.device().is_cpu() && ids.scalar_type() == at::kLong,
      "gather: 1-D int64 ids on the CPU");
  const c10::SymInt rows = ids.sym_size(0);
  const c10::SymInt cols = table.sym_size(1);
  at::Tensor ids_dev = at::empty_symint({rows}, table.options().dtype(at::kLong));
  at::Tensor out = at::empty_symint({rows, cols}, table.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  copy_h2d(sym_mutable_data_ptr(ids_dev), ids, rows * 8, stream);
  if (st == at::kFloat) {
    gather_launch<float>(table, ids_dev, out, rows, cols, stream.stream());
  } else if (st == at::kHalf) {
    gather_launch<at::Half>(table, ids_dev, out, rows, cols, stream.stream());
  } else {
    gather_launch<at::BFloat16>(table, ids_dev, out, rows, cols, stream.stream());
  }
  return out;
}

at::Tensor raw_table_address(const at::Tensor& x) {
  HostTable<const void*> ptrs(1, "raw_ptrs");
  // an address the host obtained some other way than sym_*_data_ptr
  ptrs[0] = reinterpret_cast<const void*>(static_cast<uintptr_t>(0x1000));
  return x;
}

at::Tensor copy_count(const at::Tensor& y) {
  TORCH_CHECK(y.dim() >= 1 && y.is_cuda(), "copy_count: a CUDA tensor");
  HostTable<int64_t> table(4, "count_table");
  const c10::SymInt n = y.sym_size(0);
  for (const auto i : c10::irange(4)) {
    table[i] = n * 100 + i;
  }
  at::Tensor dev = at::empty({32}, y.options().dtype(at::kByte));
  at::Tensor out = at::empty({1}, y.options().dtype(at::kLong));
  auto stream = at::cuda::getCurrentCUDAStream();
  copy_h2d(sym_mutable_data_ptr(dev), table, n * 8, stream);
  launch(
      read_last_kernel,
      Grid(1),
      32,
      c10::SymInt(0),
      stream.stream(),
      sym_const_data_ptr(dev), // a byte buffer the kernel reads as int64
      sym_mutable_data_ptr<int64_t>(out),
      n);
  return out;
}

at::Tensor copy_zero(const at::Tensor& x) {
  HostTable<int64_t> table(2, "zero_table");
  table[0] = x.sym_size(0);
  at::Tensor dev = at::empty({16}, x.options().dtype(at::kByte));
  copy_h2d(
      sym_mutable_data_ptr(dev), table, c10::SymInt(0), at::cuda::getCurrentCUDAStream());
  return x;
}

at::Tensor copy_into(const at::Tensor& dst, const at::Tensor& src) {
  copy_h2d(
      sym_mutable_data_ptr(dst),
      src,
      src.sym_numel() * static_cast<int64_t>(src.itemsize()),
      at::cuda::getCurrentCUDAStream());
  return dst;
}

at::Tensor copy_into_beside(
    const at::Tensor& x,
    const at::Tensor& dst,
    const at::Tensor& src) {
  copy_into(dst, src);
  return x;
}

at::Tensor copy_twice(const at::Tensor& y) {
  TORCH_CHECK(y.dim() >= 1 && y.is_cuda(), "copy_twice: a CUDA tensor");
  const c10::SymInt n = y.sym_size(0);
  HostTable<int64_t> table(4, "twice_table");
  for (const auto i : c10::irange(4)) {
    table[i] = n * 100 + i;
  }
  at::Tensor first = at::empty({32}, y.options().dtype(at::kByte));
  at::Tensor second = at::empty({32}, y.options().dtype(at::kByte));
  at::Tensor out = at::empty({8}, y.options().dtype(at::kLong));
  auto stream = at::cuda::getCurrentCUDAStream();
  copy_h2d(sym_mutable_data_ptr(first), table, table.nbytes(), stream);
  copy_h2d(sym_mutable_data_ptr(second), table, table.nbytes(), stream);
  launch(
      read_two_kernel,
      Grid(1),
      32,
      c10::SymInt(0),
      stream.stream(),
      sym_const_data_ptr(first),
      4,
      sym_const_data_ptr(second),
      4,
      sym_mutable_data_ptr<int64_t>(out));
  return out;
}

at::Tensor copy_rewrite_copy(const at::Tensor& y) {
  TORCH_CHECK(y.dim() >= 1 && y.is_cuda(), "copy_rewrite_copy: a CUDA tensor");
  const c10::SymInt n = y.sym_size(0);
  HostTable<int64_t> table(2, "rewrite_table");
  at::Tensor first = at::empty({8}, y.options().dtype(at::kByte));
  at::Tensor second = at::empty({16}, y.options().dtype(at::kByte));
  at::Tensor out = at::empty({3}, y.options().dtype(at::kLong));
  auto stream = at::cuda::getCurrentCUDAStream();
  table[0] = n * 100;
  copy_h2d(sym_mutable_data_ptr(first), table, c10::SymInt(8), stream);
  // a pinned source is rewritten only after the copy has read it: the eager
  // path waits for the stream; under a capture nothing has executed and the
  // wait is illegal, so the host does not wait. A plain graph capture of this
  // host would replay both copies from the final bytes; the tape records the
  // table as each copy read it.
  if (at::cuda::currentStreamCaptureStatusMayInitCtx() ==
      at::cuda::CaptureStatus::None) {
    stream.synchronize();
  }
  table[0] = n * 200;
  table[1] = n * 300;
  copy_h2d(sym_mutable_data_ptr(second), table, c10::SymInt(16), stream);
  launch(
      read_two_kernel,
      Grid(1),
      32,
      c10::SymInt(0),
      stream.stream(),
      sym_const_data_ptr(first),
      1,
      sym_const_data_ptr(second),
      2,
      sym_mutable_data_ptr<int64_t>(out));
  return out;
}

at::Tensor copy_on_stream(const at::Tensor& y, bool forked) {
  TORCH_CHECK(y.dim() >= 1 && y.is_cuda(), "copy_on_stream: a CUDA tensor");
  const c10::SymInt n = y.sym_size(0);
  HostTable<int64_t> table(2, "stream_table");
  table[0] = n * 100;
  table[1] = n * 100 + 1;
  at::Tensor dev = at::empty({16}, y.options().dtype(at::kByte));
  at::Tensor out = at::empty({2}, y.options().dtype(at::kLong));
  auto current = at::cuda::getCurrentCUDAStream();
  auto side = at::cuda::getStreamFromPool();
  at::cuda::CUDAEvent fork;
  at::cuda::CUDAEvent join;
  if (forked) {
    fork.record(current);
    fork.block(side);
  }
  copy_h2d(sym_mutable_data_ptr(dev), table, table.nbytes(), side);
  join.record(side);
  join.block(current);
  launch(
      read_two_kernel,
      Grid(1),
      32,
      c10::SymInt(0),
      current.stream(),
      sym_const_data_ptr(dev),
      2,
      sym_const_data_ptr(dev),
      0,
      sym_mutable_data_ptr<int64_t>(out));
  return out;
}

at::Tensor raw_memcpy(const at::Tensor& x) {
  // a copy the recorder does not see: two module-owned buffers, a raw
  // cudaMemcpyAsync on the current stream
  static at::Tensor src = at::zeros(
      {16}, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
  static at::Tensor dst = at::zeros({16}, x.options().dtype(at::kFloat));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      dst.data_ptr(),
      src.data_ptr(),
      64,
      cudaMemcpyHostToDevice,
      at::cuda::getCurrentCUDAStream().stream()));
  return x;
}

} // namespace at::cuda::host_trace::h2d

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace h2d = at::cuda::host_trace::h2d;
  // grouped_mul(a0, b0, a1, b1, ...): the group count is the arity
  m.def("grouped_mul", [](const py::args& args) {
    std::vector<at::Tensor> a;
    std::vector<at::Tensor> b;
    TORCH_CHECK(
        args.size() % 2 == 0 && !args.empty(),
        "grouped_mul takes pairs of tensors");
    for (size_t i = 0; i < args.size(); i += 2) {
      a.push_back(args[i].cast<at::Tensor>());
      b.push_back(args[i + 1].cast<at::Tensor>());
    }
    return h2d::grouped_mul(a, b);
  });
  m.def("gather", &h2d::gather);
  m.def("raw_table_address", &h2d::raw_table_address);
  m.def("raw_memcpy", &h2d::raw_memcpy);
  m.def("copy_count", &h2d::copy_count);
  m.def("copy_zero", &h2d::copy_zero);
  m.def("copy_into", &h2d::copy_into);
  m.def("copy_into_beside", &h2d::copy_into_beside);
  m.def("copy_twice", &h2d::copy_twice);
  m.def("copy_rewrite_copy", &h2d::copy_rewrite_copy);
  m.def("copy_on_stream", &h2d::copy_on_stream);
}
"""

_ext = None


def probe():
    """The extension module (built on first use in a process)."""
    global _ext
    if _ext is None:
        _ext = load_test_extension("hosttrace_h2d_probe", SOURCE)
    return _ext
