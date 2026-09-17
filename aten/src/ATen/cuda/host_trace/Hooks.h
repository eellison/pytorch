// Internal to the recorder: the hooks torch/csrc/cuda/HostTrace.cpp installs
// so the recorder can read a Python SymNode's hint and mint an opaque symbol.
// Included by Recorder.cpp and the bindings only; not part of the host-facing
// surface (Recorder.h, Field.h, Launch.h).
#pragma once
#include <c10/core/SymBool.h>
#include <c10/core/SymFloat.h>
#include <c10/core/SymInt.h>
#include <c10/macros/Export.h>

#include <cstdint>
#include <string>

namespace at::cuda::host_trace::hooks {

struct Hooks {
  int64_t (*int_hint)(const c10::SymInt&) = nullptr;
  double (*float_hint)(const c10::SymFloat&) = nullptr;
  bool (*bool_hint)(const c10::SymBool&) = nullptr;
  c10::SymInt (*new_int_sym)(int64_t hint, const std::string& name) = nullptr;
};
TORCH_CUDA_CPP_API void set_hooks(const Hooks& h);

} // namespace at::cuda::host_trace::hooks
