#pragma once

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/ArrayRef.h>

#include <memory>
#include <optional>

namespace c10::cuda::CUDACachingAllocator {

class CUDAAllocator;

struct PendingGraphInput {
  const DataPtr* storage;
  size_t bytes;
  size_t alignment;
};

// The caller serializes operations and retains input owners throughout arm.
// Normal final deleters and recordStream calls may run concurrently.
class C10_CUDA_API PendingGraphInputs {
 public:
  ~PendingGraphInputs() noexcept;
  PendingGraphInputs(const PendingGraphInputs&) = delete;
  PendingGraphInputs& operator=(const PendingGraphInputs&) = delete;

  // Unsupported native backing/policy returns nullptr before any input drops.
  static std::unique_ptr<PendingGraphInputs> arm(
      ArrayRef<PendingGraphInput> inputs,
      CUDAAllocator* expected_allocator,
      CUDAStream replay_stream);

  // The caller proves every old use precedes every new use in the captured graph.
  // All inputs sharing backing must occur in the sorted eligible index list.
  std::optional<DataPtr> tryClaim(
      size_t bytes, size_t alignment, ArrayRef<size_t> eligible_inputs);
  void beginSubmission();
  void finishSubmitted();
  void abortUnsubmitted();
  bool submission_started() const noexcept;
  bool finished() const noexcept;

 private:
  struct Impl;
  explicit PendingGraphInputs(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace c10::cuda::CUDACachingAllocator
