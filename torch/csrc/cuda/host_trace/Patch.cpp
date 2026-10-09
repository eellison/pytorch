#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/LaunchLayout.h>
#include <ATen/core/CachingHostAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/autograd/python_variable.h>
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
int64_t kernel_packs = 0;
int64_t kernel_node_sets = 0;
bool phase_timing = false;
std::array<int64_t, 16> phase_ns{};
int64_t tma_encodes = 0;
int64_t tma_replaces = 0;

static int driver_version() {
  static const int version = [] {
    int v = 0;
    C10_CUDA_CHECK(cudaDriverGetVersion(&v));
    return v;
  }();
  return version;
}

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
    // the CuTe DSL's constants but the fill (_host_trace_launch._encode_tma)
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
        d.fill));
    ++tma_encodes;
    if (d.edits && d.triton_elem != 0) {
      // Triton's launcher, as CUTLASS, on drivers <= 13010: clear where its
      // C int max byte index + 1 is under 128 KiB
      uint64_t index = 0;
      for (size_t j = 0; j < rank; ++j) {
        index += (dims[j] - 1) * (j == 0 ? d.triton_elem : strides[j - 1]);
      }
      d.clear_bit21 = driver_version() <= 13010 &&
          static_cast<int32_t>(static_cast<uint32_t>(index)) < 128 * 1024 - 1;
    }
  } else if (d.last[0] != x[0]) {
    C10_CUDA_DRIVER_CHECK(driver->cuTensorMapReplaceAddress_(&d.map, address));
    ++tma_replaces;
  }
  d.last.assign(x.begin(), x.end());
  uint8_t* at = image + k.param_offsets[d.param];
  std::memcpy(at, &d.map, sizeof(d.map));
  if (!d.edits) {
    return;
  }
  if (d.triton_elem == 0) {
    // a bit the DSL's inline encode sets and no encode option does
    at[8] |= 0x02;
  } else if (d.clear_bit21) {
    at[10] &= ~0x20; // bit 21 of the second word
  }
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

namespace {

// (key, record) pairs as CSR over n keys
void to_csr(
    std::vector<std::pair<uint32_t, uint32_t>>& pairs,
    size_t n,
    std::vector<uint32_t>& first,
    std::vector<uint32_t>& records) {
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  first.assign(n + 1, 0);
  records.resize(pairs.size());
  for (size_t i = 0; i < pairs.size(); ++i) {
    ++first[pairs[i].first + 1];
    records[i] = pairs[i].second;
  }
  for (size_t i = 0; i < n; ++i) {
    first[i + 1] += first[i];
  }
}

} // namespace

void HostTraceVariant::index_users() {
  // with users_kept_ (only rows were added since the last index), the rows and
  // bases some record read before, to keep their seen values
  std::vector<bool> tracked_rows, tracked_bases;
  if (users_kept_) {
    tracked_rows.assign(program_.num_rows(), false);
    for (uint32_t row : users_.rows) {
      tracked_rows[row] = true;
    }
    tracked_bases.assign(base_count_, false);
    for (uint32_t b : users_.segment_bases) {
      tracked_bases[b] = true;
    }
  }
  std::vector<std::pair<uint32_t, uint32_t>> rows;
  std::vector<std::pair<uint32_t, uint32_t>> bases;
  std::vector<std::pair<uint32_t, uint32_t>> segment_bases;
  always_.assign(records_.size(), 0);
  auto read = [&](int64_t row, int64_t base, uint32_t i, uint32_t segment) {
    rows.emplace_back(row, i);
    if (base >= 0) {
      bases.emplace_back(base, i);
      segment_bases.emplace_back(segment, base);
    }
  };
  for (uint32_t g = 0; g < segments_.size(); ++g) {
    for (size_t i = segments_[g].first; i < segments_[g].stop; ++i) {
      const Record& r = records_[i];
      const auto record = static_cast<uint32_t>(i);
      if (r.kind != Kind::Kernel) {
        // a keyed site's memset takes its row's fields: patched every replay
        if (r.site >= 0) {
          always_[i] = 1;
          continue;
        }
        if (r.kind == Kind::Memset) {
          read(r.memset.dst.row, r.memset.dst.base, record, g);
          if (!r.memset.constant_shape) {
            for (int64_t row : r.memset.shape_rows) {
              rows.emplace_back(row, record);
            }
          }
        } else {
          read(r.copy.dst.row, r.copy.dst.base, record, g);
          read(r.copy.src.row, r.copy.src.base, record, g);
          rows.emplace_back(r.copy.bytes_row, record);
        }
        continue;
      }
      const KernelRow& k = r.kernel;
      if (k.rng_increment != 0 || !k.rng.empty() || !k.cpu_scalars.empty()) {
        always_[i] = 1;
      }
      for (size_t j = 0; j < k.field_count; ++j) {
        read(k.fields[j].row, k.fields[j].base, record, g);
      }
      if (!k.constant_dims) {
        for (int64_t row : k.dim_rows) {
          rows.emplace_back(row, record);
        }
      }
      if (r.site >= 0 && r.site_pos < sites_[r.site].reads.size()) {
        const Site::Reads& e = sites_[r.site].reads[r.site_pos];
        always_[i] |= e.always;
        for (const auto& [row, base] : e.fields) {
          read(row, base, record, g);
        }
        for (int64_t row : e.dim_rows) {
          rows.emplace_back(row, record);
        }
      }
    }
  }
  to_csr(rows, program_.num_rows(), users_.row_first, users_.row_records);
  to_csr(bases, base_count_, users_.base_first, users_.base_records);
  to_csr(
      segment_bases,
      segments_.size(),
      users_.segment_first,
      users_.segment_bases);
  std::vector<bool> dynamic(program_.num_rows(), false);
  for (uint32_t row : dynamic_rows_) {
    dynamic[row] = true;
  }
  users_.rows.clear();
  users_.dynamic_rows.clear();
  for (uint32_t row = 0; row < program_.num_rows(); ++row) {
    if (users_.row_first[row] != users_.row_first[row + 1]) {
      users_.rows.push_back(row);
      if (dynamic[row]) {
        users_.dynamic_rows.push_back(row);
      }
    }
  }
  if (users_kept_) {
    // a record of a site with a new row is marked when the call selects the
    // row (seen_sites_); a row or base no record read before is marked at its
    // next value
    for (uint32_t row : users_.rows) {
      if (!tracked_rows[row]) {
        seen_values_[row] = std::numeric_limits<int64_t>::min();
      }
    }
    for (uint32_t b : users_.segment_bases) {
      if (!tracked_bases[b]) {
        seen_bases_[b] = std::numeric_limits<int64_t>::min();
      }
    }
  } else {
    dirty_.assign(records_.size(), 1);
    seen_values_.assign(program_.num_rows(), 0);
    seen_bases_.assign(base_count_, std::numeric_limits<int64_t>::min());
    seen_sites_.assign(sites_.size(), -1);
    seen_valid_ = false;
  }
  users_stale_ = users_kept_ = false;
}

void HostTraceVariant::mark(
    const std::vector<uint32_t>& first,
    const std::vector<uint32_t>& records,
    size_t i) {
  for (uint32_t k = first[i]; k < first[i + 1]; ++k) {
    dirty_[records[k]] = 1;
  }
}

void HostTraceVariant::mark_dirty(const Frame& frame) {
  if (users_stale_) {
    index_users();
  }
  const int64_t* v = frame.values.data();
  const uint64_t generation = frame.cached == this ? cached_generation_ : 0;
  if (!seen_valid_) {
    std::fill(dirty_.begin(), dirty_.end(), 1);
    for (uint32_t row : users_.rows) {
      seen_values_[row] = v[row];
    }
  } else {
    // the static rows are the cache's, unchanged since the generation's last
    // commit
    const bool statics = generation != 0 && generation == seen_generation_;
    // with delta_marked_, the cache's rows are those last seen
    if (!statics || !delta_marked_) {
      for (uint32_t row : statics ? users_.dynamic_rows : users_.rows) {
        if (v[row] != seen_values_[row]) {
          seen_values_[row] = v[row];
          mark(users_.row_first, users_.row_records, row);
        }
      }
    }
  }
  for (size_t s = 0; s < sites_.size(); ++s) {
    if (frame.site_entries[s] != seen_sites_[s]) {
      seen_sites_[s] = frame.site_entries[s];
      for (size_t i : sites_[s].records) {
        dirty_[i] = 1;
      }
    }
  }
  seen_generation_ = generation;
  seen_valid_ = true;
  delta_marked_ = generation != 0 && delta_valid_;
}

void HostTraceVariant::switch_form_held(Segment& run, size_t form) {
  // A keyed row's held image is shared by its site node's record and the
  // pieces at that node, live in different forms: each form keeps its own copy
  const size_t n = run.stop - run.first;
  auto slot = [&](size_t f, size_t i) -> Record* {
    if (i >= n) {
      return &run.clones[f - 1].pieces[i - n];
    }
    const bool live = f == 0 ? records_[run.first + i].node != nullptr
                             : run.clones[f - 1].nodes[i] != nullptr;
    return live ? &records_[run.first + i] : nullptr;
  };
  run.form_held.resize(run.execs.size());
  auto& saved = run.form_held[run.form];
  saved.resize(n + (run.form == 0 ? 0 : run.clones[run.form - 1].pieces.size()));
  for (size_t i = 0; i < saved.size(); ++i) {
    Record* r = slot(run.form, i);
    Segment::Held& h = saved[i];
    h.held = r && r->kind == Kind::Kernel && r->held;
    if (h.held) {
      const KernelRow& k = kernel_row(*r, r->held_row);
      h.row = r->held_row;
      h.dims = r->held_dims;
      h.image.assign(k.held, k.held + k.nbytes);
    }
  }
  auto& next = run.form_held[form];
  next.resize(n + (form == 0 ? 0 : run.clones[form - 1].pieces.size()));
  for (size_t i = 0; i < next.size(); ++i) {
    Record* r = slot(form, i);
    if (!r) {
      continue;
    }
    const Segment::Held& h = next[i];
    r->held = h.held;
    if (h.held) {
      std::memcpy(kernel_row(*r, h.row).held, h.image.data(), h.image.size());
      r->held_row = h.row;
      r->held_dims = h.dims;
    }
  }
  run.form = form;
}

void HostTraceVariant::patch_and_replay(Segment& run, Frame& frame) {
  int64_t pt = phase_now();
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
  if (form != run.form && form_held_enabled) {
    switch_form_held(run, form);
    for (size_t i = run.first; i < run.stop; ++i) {
      dirty_[i] = 1;
    }
  } else if (form != run.form) {
    // the form's exec holds its instantiation's parameters
    run.form = form;
    for (size_t i = run.first; i < run.stop; ++i) {
      records_[i].held = false;
      dirty_[i] = 1;
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
      const std::array<int64_t, 4> held = memset_shape(m, v, bases);
      // a node holds its destination's address, not the allocation there
      // (probe_memset_alloc.py: after a free and reallocation, or a VMM remap,
      // at the address, the unchanged node writes the new memory)
      if (r.held && r.held_row == row && held == r.held_memset) {
        return;
      }
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
      const std::array<int64_t, 3> held = memcpy_shape(r.copy, v, bases);
      if (r.held && r.held_row == row && held == r.held_copy) {
        return;
      }
      r.held = false;
      TORCH_CHECK_VALUE(held[2] > 0, "host_trace: a memcpy of ", held[2], " bytes");
      C10_CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(
          reinterpret_cast<cudaGraphExec_t>(exec),
          reinterpret_cast<cudaGraphNode_t>(node),
          reinterpret_cast<void*>(held[0]),
          reinterpret_cast<const void*>(held[1]),
          static_cast<size_t>(held[2]),
          r.copy.kind));
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
    ++kernel_packs;
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
    ++kernel_node_sets;
    TORCH_CHECK(
        fail_after_setter < 0 || fail_after_setter-- != 0,
        "host_trace: a failure injected after a setter");
    std::memcpy(k.held, k.image, k.nbytes);
    r.held_dims = dims;
    r.held_row = row;
    r.held = true;
  };
  if (users_stale_) {
    // a row added by a call from an eager step
    index_users();
  }
  const size_t g = &run - segments_.data();
  for (uint32_t k = users_.segment_first[g]; k < users_.segment_first[g + 1];
       ++k) {
    const uint32_t b = users_.segment_bases[k];
    if (bases[b] != seen_bases_[b]) {
      seen_bases_[b] = bases[b];
      mark(users_.base_first, users_.base_records, b);
    }
  }
  for (size_t i = run.first; i < run.stop; ++i) {
    CUgraphNode node = clone ? clone->nodes[i - run.first] : records_[i].node;
    if (node && dirty_[i]) {
      patch(records_[i], node);
      dirty_[i] = always_[i];
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
  // as eager's non_blocking copy_, at every replay whether or not its node
  // was set: the pinned block a host memcpy reads or writes is not reused
  // before the work queued on this stream (the replay) completes
  for (size_t i : run.host_copies) {
    const Record& r = records_[i];
    const int64_t a = r.copy.host_arg;
    TORCH_CHECK(
        a >= 0 && static_cast<size_t>(a) < frame.count && THPVariable_Check(frame.args[a]),
        "host_trace: a memcpy's host argument is not a tensor");
    const std::array<int64_t, 3> held = memcpy_shape(r.copy, v, bases);
    void* ptr = reinterpret_cast<void*>(held[r.copy.kind == cudaMemcpyHostToDevice ? 1 : 0]);
    void* ctx = THPVariable_Unpack(frame.args[a]).storage().data_ptr().get_context();
    at::getHostAllocator(at::kCUDA)->record_event(ptr, ctx, at::cuda::getCurrentCUDAStream(device_));
  }
  phase_add(5, pt);
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
  phase_add(6, pt);
}

} // namespace torch::cuda::host_trace
#endif
