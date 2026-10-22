#include <ATen/cuda/host_trace/LaunchLayout.h>

#include <c10/core/DynamicCast.h>

#include <utility>

namespace at::cuda::host_trace {

namespace {
thread_local std::vector<LaunchLayout>* sink = nullptr;

template <class T>
bool write(const T& v, void* out) {
  std::memcpy(out, &v, sizeof(T));
  return true;
}
} // namespace

std::vector<LaunchLayout>* record_launch_layouts(std::vector<LaunchLayout>* into) {
  return std::exchange(sink, into);
}

void report_launch_layout(LaunchLayout layout) {
  if (sink) {
    sink->push_back(std::move(layout));
  }
}

bool cpu_scalar_bytes(const at::TensorBase& src, char cls, void* out) {
  const auto from = src.scalar_type();
  const void* p = src.const_data_ptr();
  if (cls >= '0' && cls <= '9') {
    // div_true_kernel_cuda: static_cast<opmath_t>(1.0 / iter.scalar_value<double>(2))
    const double r = 1.0 / c10::fetch_and_cast<double>(from, p);
    switch (static_cast<c10::ScalarType>(cls - '0')) {
      case c10::ScalarType::Float:
        return write(static_cast<float>(r), out);
      case c10::ScalarType::Double:
        return write(r, out);
      default:
        return false;
    }
  }
  if (cls < 'A' || cls > 'Z') {
    return false;
  }
  switch (static_cast<c10::ScalarType>(cls - 'A')) {
#define CPU_SCALAR_CASE(T, name) \
  case c10::ScalarType::name:    \
    return write(c10::fetch_and_cast<T>(from, p), out);
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(CPU_SCALAR_CASE)
#undef CPU_SCALAR_CASE
    default:
      return false;
  }
}

} // namespace at::cuda::host_trace
