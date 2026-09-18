// Internal to the recorder: the hooks torch/csrc/cuda/HostTrace.cpp installs
// so the recorder can read a Python SymNode's hint, mint an opaque symbol and
// narrow a symbolic float.
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
  c10::SymFloat (*round_float32)(const c10::SymFloat&) = nullptr;
  bool (*bool_hint)(const c10::SymBool&) = nullptr;
  c10::SymInt (*new_int_sym)(
      int64_t hint,
      const std::string& name,
      bool positive) = nullptr;
};
TORCH_CUDA_CPP_API void set_hooks(const Hooks& h);
// A fresh integer symbol with this hint (an opaque result, a host table's
// address), minted by the Python side of the trace in progress; `positive`
// is the declared domain (>= 1) of an opaque result.
TORCH_CUDA_CPP_API c10::SymInt mint_int_symbol(
    int64_t hint,
    const std::string& name,
    bool positive = false);

} // namespace at::cuda::host_trace::hooks
