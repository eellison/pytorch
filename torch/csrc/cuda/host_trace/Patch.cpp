#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/LaunchLayout.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <c10/util/safe_numerics.h>
#include <torch/csrc/utils/object_ptr.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace torch::cuda::host_trace {

int64_t fail_after_setter = -1;
int64_t memory_node_sets = 0;
int64_t tma_encodes = 0;
int64_t tma_replaces = 0;

int64_t HostTraceVariant::field_value(
    const Field& f,
    const int64_t* v,
    const int64_t* bases) const {
  int64_t x = v[f.row];
  TORCH_CHECK_VALUE(
      !c10::add_overflows(x, f.delta, &x) &&
          (f.base < 0 || !c10::add_overflows(bases[f.base], x, &x)),
      "host_trace: a value outside int64");
  return x;
}

void HostTraceVariant::encode(
    const KernelRow& k,
    Descriptor& d,
    const int64_t* v,
    const int64_t* bases,
    uint8_t* image) const {
  const size_t rank = d.box.size();
  c10::SmallVector<int64_t, 10> x(2 * rank);
  for (size_t j = 0; j < x.size(); ++j) {
    x[j] = field_value(k.fields[d.first + j], v, bases);
  }
  // the driver rejects a placeholder's non-canonical top
  // (_host_trace_tape._PLACEHOLDER_LOW); a real address has none
  x[0] &= (int64_t{1} << 52) - 1;
  auto* driver = c10::cuda::DriverAPI::get();
  auto* address = reinterpret_cast<void*>(x[0]);
  if (d.last.empty() ||
      !std::equal(x.begin() + 1, x.end(), d.last.begin() + 1)) {
    std::array<cuuint64_t, 5> dims{};
    std::array<cuuint64_t, 5> strides{};
    std::array<cuuint32_t, 5> ones{1, 1, 1, 1, 1};
    for (size_t j = 0; j < rank; ++j) {
      dims[j] = static_cast<cuuint64_t>(x[1 + j]);
    }
    for (size_t j = 0; j + 1 < rank; ++j) {
      strides[j] = static_cast<cuuint64_t>(x[1 + rank + j]);
    }
    // the CuTe DSL's constants (_host_trace_triton_launch._encode_tma)
    C10_CUDA_DRIVER_CHECK(driver->cuTensorMapEncodeTiled_(
        &d.map,
        d.dtype,
        static_cast<cuuint32_t>(rank),
        address,
        dims.data(),
        strides.data(),
        d.box.data(),
        ones.data(),
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    ++tma_encodes;
  } else if (d.last[0] != x[0]) {
    C10_CUDA_DRIVER_CHECK(driver->cuTensorMapReplaceAddress_(&d.map, address));
    ++tma_replaces;
  }
  d.last.assign(x.begin(), x.end());
  uint8_t* at = image + k.param_offsets[d.param];
  std::memcpy(at, &d.map, sizeof(d.map));
  // a bit the DSL's inline encode sets and no encode option does
  at[8] |= 0x02;
}

void HostTraceVariant::pack(
    const KernelRow& k,
    const int64_t* v,
    const int64_t* bases,
    uint8_t* image) const {
  for (size_t i = 0; i < k.field_count; ++i) {
    const Field& f = k.fields[i];
    if (f.width == 0) {
      continue;
    }
    const int64_t x = field_value(f, v, bases);
    uint8_t* at = image + k.param_offsets[f.param] + f.offset;
    if (f.pointer) {
      TORCH_CHECK_VALUE(x >= 0, "host_trace: a negative pointer ", x);
      if (f.width == 4) {
        const auto y = static_cast<uint32_t>(x);
        std::memcpy(at, &y, sizeof(y));
      } else {
        std::memcpy(at, &x, sizeof(x));
      }
    } else if (f.width == 1) {
      TORCH_CHECK_VALUE(
          x >= std::numeric_limits<int8_t>::min() &&
              x <= std::numeric_limits<int8_t>::max(),
          "host_trace: ",
          x,
          " is not a 1-byte field's");
      const auto y = static_cast<int8_t>(x);
      std::memcpy(at, &y, sizeof(y));
    } else if (f.width == 2) {
      TORCH_CHECK_VALUE(
          x >= std::numeric_limits<int16_t>::min() &&
              x <= std::numeric_limits<int16_t>::max(),
          "host_trace: ",
          x,
          " is not a 2-byte field's");
      const auto y = static_cast<int16_t>(x);
      std::memcpy(at, &y, sizeof(y));
    } else if (f.width == 4) {
      TORCH_CHECK_VALUE(
          x >= std::numeric_limits<int32_t>::min() &&
              x <= std::numeric_limits<int32_t>::max(),
          "host_trace: ",
          x,
          " is not a 4-byte field's");
      const auto y = static_cast<int32_t>(x);
      std::memcpy(at, &y, sizeof(y));
    } else {
      std::memcpy(at, &x, sizeof(x));
    }
  }
  for (size_t i = 0; i < k.descriptor_count; ++i) {
    encode(k, k.descriptors[i], v, bases, image);
  }
}

std::array<int64_t, 4> HostTraceVariant::memset_shape(
    const MemsetRow& m,
    const int64_t* v,
    const int64_t* bases) const {
  const int64_t address = field_value(m.dst, v, bases);
  if (m.constant_shape) {
    return {address, m.shape[0], m.shape[1], m.shape[2]};
  }
  return {
      address, v[m.shape_rows[0]], v[m.shape_rows[1]], v[m.shape_rows[2]]};
}

std::array<int64_t, 3> HostTraceVariant::memcpy_shape(
    const MemcpyRow& m,
    const int64_t* v,
    const int64_t* bases) const {
  return {field_value(m.dst, v, bases), field_value(m.src, v, bases), v[m.bytes_row]};
}

void HostTraceVariant::patch_and_replay(Segment& run, Frame& frame) {
  const int64_t* v = frame.values.data();
  const int64_t* bases = frame.bases.data();
  auto* driver = c10::cuda::DriverAPI::get();
  // the segment's graph is the variant's; a reset or re-instantiation of it
  // (nothing does one) would free the exec the records' nodes are in
  TORCH_CHECK(
      run.native->has_graph_exec() &&
          reinterpret_cast<CUgraphExec>(run.native->raw_cuda_graph_exec()) ==
              run.execs[0],
      "host_trace: a replay's graph was reset or re-instantiated");
  const size_t form =
      run.forms.size() > 1 ? frame.forms[&run - segments_.data()] : 0;
  Segment::Clone* clone = form == 0 ? nullptr : &run.clones[form - 1];
  if (form != run.form) {
    // the form's exec holds its instantiation's parameters
    run.form = form;
    for (size_t i = run.first; i < run.stop; ++i) {
      records_[i].held = false;
    }
    if (clone) {
      for (Record& r : clone->pieces) {
        r.held = false;
      }
    }
  }
  const CUgraphExec exec = run.execs[form];
  // the philox offsets the segment's RNG calls take before the next, and
  // the intragraph offset of the current one's kernels
  int64_t running = 0;
  int64_t philox = 0;
  auto patch = [&](Record& r, CUgraphNode node) {
    const int64_t row = r.site >= 0 ? frame.site_entries[r.site] : 0;
    if (r.kind == Kind::Memset) {
      const MemsetRow& m = memset_row(r, row);
      // set every replay: the node holds the allocation its address was in
      // when set, and a free and reallocation at that address leaves it stale
      const std::array<int64_t, 4> held = memset_shape(m, v, bases);
      r.held = false;
      for (size_t k = 1; k < 4; ++k) {
        TORCH_CHECK_VALUE(held[k] >= 0, "host_trace: a memset of ", held[k]);
      }
      cudaMemsetParams params{};
      params.dst = reinterpret_cast<void*>(held[0]);
      params.value = m.value;
      params.elementSize = m.element_size;
      params.width = static_cast<size_t>(held[1]);
      params.height = static_cast<size_t>(held[2]);
      params.pitch = static_cast<size_t>(held[3]);
      C10_CUDA_CHECK(cudaGraphExecMemsetNodeSetParams(
          reinterpret_cast<cudaGraphExec_t>(exec),
          reinterpret_cast<cudaGraphNode_t>(node),
          &params));
      ++memory_node_sets;
      TORCH_CHECK(
          fail_after_setter < 0 || fail_after_setter-- != 0,
          "host_trace: a failure injected after a setter");
      r.held_memset = held;
      r.held_row = row;
      r.held = true;
      return;
    }
    if (r.kind == Kind::Memcpy) {
      // set every replay, as a memset is
      const std::array<int64_t, 3> held = memcpy_shape(r.copy, v, bases);
      r.held = false;
      TORCH_CHECK_VALUE(held[2] > 0, "host_trace: a memcpy of ", held[2], " bytes");
      C10_CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(
          reinterpret_cast<cudaGraphExec_t>(exec),
          reinterpret_cast<cudaGraphNode_t>(node),
          reinterpret_cast<void*>(held[0]),
          reinterpret_cast<const void*>(held[1]),
          static_cast<size_t>(held[2]),
          cudaMemcpyDeviceToDevice));
      ++memory_node_sets;
      TORCH_CHECK(
          fail_after_setter < 0 || fail_after_setter-- != 0,
          "host_trace: a failure injected after a setter");
      r.held_copy = held;
      r.held_row = row;
      r.held = true;
      return;
    }
    KernelRow& k = kernel_row(r, row);
    pack(k, v, bases, k.image);
    if (k.rng_increment != 0) {
      philox = running;
      running += k.rng_increment;
    }
    for (const RngField& f : k.rng) {
      const int64_t x = f.delta +
          (f.kind == 0       ? run.philox_seed
               : f.kind == 1 ? run.philox_offset
                             : philox);
      std::memcpy(k.image + k.param_offsets[f.param] + f.offset, &x, sizeof(x));
    }
    for (const CpuScalar& c : k.cpu_scalars) {
      TORCH_CHECK(
          c.source.device().is_cpu() && c.source.dim() == 0,
          "host_trace: a CPU scalar no longer a 0-dim CPU tensor");
      at::cuda::host_trace::cpu_scalar_bytes(
          c.source, c.cls, k.image + k.param_offsets[c.param] + c.offset);
    }
    std::array<int64_t, 7> dims = k.dims;
    if (!k.constant_dims) {
      for (size_t i = 0; i < dims.size(); ++i) {
        dims[i] = v[k.dim_rows[i]];
      }
    }
    if (r.held && r.held_row == row && dims == r.held_dims &&
        std::memcmp(k.image, k.held, k.nbytes) == 0) {
      return;
    }
    // a patch cut short leaves the node unknown, never stale
    r.held = false;
    for (int64_t d : dims) {
      TORCH_CHECK_VALUE(
          d >= 0 && d <= std::numeric_limits<unsigned int>::max(),
          "host_trace: a launch dimension of ",
          d);
    }
    k.params.gridDimX = static_cast<unsigned int>(dims[0]);
    k.params.gridDimY = static_cast<unsigned int>(dims[1]);
    k.params.gridDimZ = static_cast<unsigned int>(dims[2]);
    k.params.blockDimX = static_cast<unsigned int>(dims[3]);
    k.params.blockDimY = static_cast<unsigned int>(dims[4]);
    k.params.blockDimZ = static_cast<unsigned int>(dims[5]);
    k.params.sharedMemBytes = static_cast<unsigned int>(dims[6]);
    C10_CUDA_DRIVER_CHECK(
        driver->cuGraphExecKernelNodeSetParams_(exec, node, &k.params));
    TORCH_CHECK(
        fail_after_setter < 0 || fail_after_setter-- != 0,
        "host_trace: a failure injected after a setter");
    std::memcpy(k.held, k.image, k.nbytes);
    r.held_dims = dims;
    r.held_row = row;
    r.held = true;
  };
  for (size_t i = run.first; i < run.stop; ++i) {
    CUgraphNode node = clone ? clone->nodes[i - run.first] : records_[i].node;
    if (node) {
      patch(records_[i], node);
    }
  }
  if (clone) {
    for (Record& r : clone->pieces) {
      patch(r, r.node);
    }
  }
  // a replay advances the generator by what the rows' calls take, as eager
  if (run.generator && running != run.rng_increment) {
    run.native->set_generator_increment(
        *run.generator, static_cast<uint64_t>(running));
    run.rng_increment = running;
  }
  if (observed(run)) {
    TORCH_CHECK(
        form == 0,
        "host_trace: replay hooks on a segment replayed in another form");
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
    // the Python wrapper runs the hooks, the liveness check and the replay
    // stream's record for synchronize_before_release
    THPObjectPtr r(PyObject_CallMethod(run.graph.ptr(), "replay", nullptr));
    if (!r) {
      throw py::error_already_set();
    }
  } else {
    run.native->replay_exec(reinterpret_cast<cudaGraphExec_t>(exec));
  }
}

} // namespace torch::cuda::host_trace
#endif
