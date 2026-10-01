#pragma once
// Host policies for op bodies shared by eager, fake and host trace.
//
// An op body is: checks, output metadata (graph-level), then
//
//   {
//     auto k = h.kernel_context();  // EagerHost: empty, compiles away
//     if (!k) return out;           // SymHost without a recorder: fake
//     ... kernel choice (coalescing, 32-bit split) and launches ...
//   }
//
// Output metadata is final before the block. With a recorder, guards raised
// inside the block select the kernel for this one op; they are not graph-level
// guards. The block exits at its closing brace, so work after it is graph-level
// again. A body entered inside another body's block (depth > 1) belongs to the
// outer op's selector.
#include <c10/core/SymInt.h>
#include <c10/macros/Export.h>

#include <cstdint>

namespace at::ht {

// Receives the host work of ops run on SymHost. Kernel launches would be
// recorded through ht::launch<&kernel>(h, grid, block, smem, args...), which is
// not part of this change.
struct TORCH_API Recorder {
  virtual ~Recorder() = default;
  static Recorder* current();
};

// Returns the previous recorder.
TORCH_API Recorder* set_recorder(Recorder* recorder);

struct RecorderGuard {
  explicit RecorderGuard(Recorder* recorder) : prev_(set_recorder(recorder)) {}
  ~RecorderGuard() {
    set_recorder(prev_);
  }
  RecorderGuard(const RecorderGuard&) = delete;
  RecorderGuard& operator=(const RecorderGuard&) = delete;

 private:
  Recorder* prev_;
};

// Number of kernel contexts entered on this thread.
TORCH_API int64_t kernel_context_depth();

class TORCH_API KernelContext {
 public:
  explicit KernelContext(bool entered);
  ~KernelContext();
  KernelContext(const KernelContext&) = delete;
  KernelContext& operator=(const KernelContext&) = delete;
  explicit operator bool() const {
    return entered_;
  }

 private:
  bool entered_;
};

struct EagerHost {
  using Int = int64_t;
  struct Context {
    constexpr explicit operator bool() const {
      return true;
    }
  };
  constexpr Context kernel_context() const {
    return {};
  }
};

struct SymHost {
  using Int = c10::SymInt;
  KernelContext kernel_context() const {
    return KernelContext(Recorder::current() != nullptr);
  }
};

} // namespace at::ht
