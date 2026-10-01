#include <ATen/native/HostPolicy.h>

#include <utility>

namespace at::ht {

namespace {
thread_local Recorder* current_recorder = nullptr;
thread_local int64_t depth = 0;
} // namespace

Recorder* Recorder::current() {
  return current_recorder;
}

Recorder* set_recorder(Recorder* recorder) {
  return std::exchange(current_recorder, recorder);
}

int64_t kernel_context_depth() {
  return depth;
}

KernelContext::KernelContext(bool entered) : entered_(entered) {
  if (entered_) {
    depth++;
  }
}

KernelContext::~KernelContext() {
  if (entered_) {
    depth--;
  }
}

} // namespace at::ht
