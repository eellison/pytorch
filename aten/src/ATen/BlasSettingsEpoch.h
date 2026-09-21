#pragma once

#include <c10/macros/Export.h>

#include <cstdint>

namespace at {

// A counter of changes to the process-global settings that pick the kernels a
// BLAS library runs for a shape: the float32 matmul precision (allow_tf32), the
// fp16 / bf16 reduced-precision reduction options, fp16 accumulation, the
// deterministic-algorithms flag, the preferred BLAS backend, TunableOp's enable
// and tuning flags, and the cuBLAS / cuBLASLt workspace sizes. Every setter of
// one of them bumps it after its write. A cache keyed on those settings (the
// kernel template registry's per-site last key) compares the epoch instead of
// re-reading them: an equal epoch means no setting changed since.
TORCH_API uint64_t blasSettingsEpoch();
TORCH_API void bumpBlasSettingsEpoch();

} // namespace at
