// The philox state as a kernel argument: generator state of the capture,
// recorded as an `rng` field (never rewritten, never compared) like flash's
// Flash_fwd_params::philox_args. A host declares the increment it hands the
// generator with rng_increment before the launch that carries the state
// (Recorder.h), and the recorder pairs the two (Tape.h RngSlotRec).
#pragma once
#include <ATen/cuda/PhiloxCudaState.h>
#include <ATen/cuda/host_trace/Field.h>

#include <new>

namespace at::cuda::host_trace {

struct TracedPhilox : TracedBase {
  PhiloxCudaState pod;
  BytesField<0, sizeof(PhiloxCudaState)> bytes{this, "philox_args"};
  explicit TracedPhilox(const PhiloxCudaState& st)
      : TracedBase(&pod, sizeof(PhiloxCudaState)) {
    new (static_cast<void*>(bytes)) PhiloxCudaState(st);
  }
};

} // namespace at::cuda::host_trace
