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
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>

namespace torch::cuda::host_trace {

int64_t fail_after_setter = -1;
int64_t memory_node_sets = 0;
int64_t kernel_packs = 0;
int64_t kernel_node_sets = 0;
int64_t kernel_row_switches = 0;
int64_t piece_skips = 0;
int64_t index_builds = 0;
int64_t rows_unindexed = 0;
// on unless TORCH_HOST_TRACE_SPLIT=0: with split_dirty_min a call splits only where its patch would be exposed
// (dev-box A/B, 8B and 27B churn: neutral to 1% faster, never slower)
bool split_enabled = [] {
  const char* v = std::getenv("TORCH_HOST_TRACE_SPLIT");
  return v == nullptr || std::string(v) != "0";
}();
int64_t split_sections = 4;
double split_ratio = 4.0;
int64_t split_first_nodes = 0;
int64_t split_max_nodes = 0;
int64_t split_calls = 0;
int64_t split_dirty_min = 256;
int64_t split_dirty_wholes = 0;
int64_t exec_uploads = 0;
int64_t graph_launches = 0;
int64_t split_builds = 0;
bool phase_timing = false;
std::array<int64_t, 16> phase_ns{};
int64_t tma_encodes = 0;
int64_t tma_replaces = 0;
int64_t tma_cache_hits = 0;

void raise_dynamic_smem_limit(CUfunction f, unsigned int bytes) {
  if (bytes == 0) {
    return;
  }
  auto* driver = c10::cuda::DriverAPI::get();
  int limit = 0;
  C10_CUDA_DRIVER_CHECK(driver->cuFuncGetAttribute_(
      &limit, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, f));
  if (static_cast<int64_t>(bytes) > limit) {
    C10_CUDA_DRIVER_CHECK(driver->cuFuncSetAttribute_(
        f,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        static_cast<int>(bytes)));
  }
}

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
  auto seen = std::find_if(d.encoded.begin(), d.encoded.end(), [&](const Descriptor::Encoded& e) {
    return std::equal(x.begin() + 1, x.end(), e.values.begin() + 1, e.values.end());
  });
  const bool moved = d.last.empty() ||
      !std::equal(x.begin() + 1, x.end(), d.last.begin() + 1);
  if (moved && seen != d.encoded.end()) {
    d.map = seen->map;
    d.clear_bit21 = seen->clear_bit21;
    if (seen->values[0] != x[0]) {
      C10_CUDA_DRIVER_CHECK(driver->cuTensorMapReplaceAddress_(&d.map, address));
      ++tma_replaces;
    }
    ++tma_cache_hits;
    std::rotate(d.encoded.begin(), seen, seen + 1);
    d.encoded.front().values.assign(x.begin(), x.end());
    d.encoded.front().map = d.map;
  } else if (moved) {
    std::array<cuuint64_t, 5> dims{};
    std::array<cuuint64_t, 5> strides{};
    std::array<cuuint32_t, 5> ones{1, 1, 1, 1, 1};
    for (size_t j = 0; j < rank; ++j) {
      dims[j] = static_cast<cuuint64_t>(x[1 + j]);
    }
    for (size_t j = 0; j + 1 < rank; ++j) {
      strides[j] = static_cast<cuuint64_t>(x[1 + rank + j]);
    }
    // the CuTe DSL's constants but the fill and the L2 promotion
    // (_host_trace_launch._encode_tma)
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
        d.l2,
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
    d.encoded.insert(d.encoded.begin(), {std::vector<int64_t>(x.begin(), x.end()), d.map, d.clear_bit21});
    if (d.encoded.size() > kEncodedPerDescriptor) {
      d.encoded.pop_back();
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
  ++index_builds;
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
    const size_t first_pair = rows.size();
    for (size_t i = segments_[g].first; i < segments_[g].stop; ++i) {
      const Record& r = records_[i];
      const auto record = static_cast<uint32_t>(i);
      if (r.kind != Kind::Kernel) {
        // a keyed site's memset takes its row's fields, and a host memcpy's
        // pinned block may be another allocation at the same address (freed
        // and reallocated): patched every replay
        if (r.site >= 0 || (r.kind == Kind::Memcpy && r.copy.kind != cudaMemcpyDeviceToDevice)) {
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
    std::vector<int64_t>& read = segments_[g].read_rows;
    read.clear();
    for (size_t j = first_pair; j < rows.size(); ++j) {
      read.push_back(rows[j].first);
    }
    std::sort(read.begin(), read.end());
    read.erase(std::unique(read.begin(), read.end()), read.end());
  }
  ++users_version_;
  for (Site& s : sites_) {
    // every row's nodes, the records' own and each entry's
    std::vector<int64_t> piece_rows, piece_bases;
    s.piece_always = false;
    auto field = [&](const Field& f) {
      piece_rows.push_back(f.row);
      if (f.base >= 0) {
        piece_bases.push_back(f.base);
      }
    };
    auto kernel = [&](const KernelRow& k) {
      s.piece_always |= k.rng_increment != 0 || !k.rng.empty() || !k.cpu_scalars.empty();
      for (size_t j = 0; j < k.field_count; ++j) {
        field(k.fields[j]);
      }
      if (!k.constant_dims) {
        piece_rows.insert(piece_rows.end(), k.dim_rows.begin(), k.dim_rows.end());
      }
    };
    auto on_memset = [&](const MemsetRow& m) {
      field(m.dst);
      if (!m.constant_shape) {
        piece_rows.insert(piece_rows.end(), m.shape_rows.begin(), m.shape_rows.end());
      }
    };
    auto on_memcpy = [&](const MemcpyRow& m) {
      field(m.dst);
      field(m.src);
      piece_rows.push_back(m.bytes_row);
    };
    for (size_t i : s.records) {
      const Record& r = records_[i];
      if (r.kind == Kind::Kernel) {
        kernel(r.kernel);
      } else if (r.kind == Kind::Memset) {
        on_memset(r.memset);
      } else {
        on_memcpy(r.copy);
      }
    }
    for (const auto& e : s.entries) {
      std::for_each(e->kernels.begin(), e->kernels.end(), kernel);
      std::for_each(e->memsets.begin(), e->memsets.end(), on_memset);
      std::for_each(e->memcpys.begin(), e->memcpys.end(), on_memcpy);
    }
    for (std::vector<int64_t>* v : {&piece_rows, &piece_bases}) {
      std::sort(v->begin(), v->end());
      v->erase(std::unique(v->begin(), v->end()), v->end());
    }
    s.piece_rows = std::move(piece_rows);
    s.piece_bases = std::move(piece_bases);
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

std::vector<HostTraceVariant::Segment::Held>& HostTraceVariant::form_held(
    Segment& run,
    size_t form,
    bool split) {
  if (run.held.size() < 2 * run.execs.size()) {
    run.held.resize(2 * run.execs.size());
  }
  std::vector<Segment::Held>& held = run.held[2 * form + split];
  if (held.empty()) {
    const size_t n = run.stop - run.first;
    held.resize(n + (form == 0 ? 0 : run.clones[form - 1].pieces.size()));
    const std::vector<Segment::Held>& now = run.held[2 * run.form + run.split];
    for (size_t i = 0; i < n && i < now.size(); ++i) {
      held[i].copy = now[i].copy;
    }
  }
  return held;
}

bool HostTraceVariant::changed_since(
    const Segment& run,
    const Segment::Seen& s,
    const Frame& frame,
    uint8_t* out) const {
  const size_t n = run.stop - run.first;
  if (!s.saved || s.users_version != users_version_) {
    std::fill(out, out + n, 1);
    return false;
  }
  const int64_t* v = frame.values.data();
  const int64_t* bases = frame.bases.data();
  const size_t g = &run - segments_.data();
  const uint32_t* segment_bases = users_.segment_bases.data() + users_.segment_first[g];
  const size_t base_count = users_.segment_first[g + 1] - users_.segment_first[g];
  std::copy(s.dirty.begin(), s.dirty.end(), out);
  auto mark_in_run = [&](const std::vector<uint32_t>& first, const std::vector<uint32_t>& records, size_t i) {
    for (uint32_t k = first[i]; k < first[i + 1]; ++k) {
      if (records[k] >= run.first && records[k] < run.stop) {
        out[records[k] - run.first] = 1;
      }
    }
  };
  for (size_t j = 0; j < run.read_rows.size(); ++j) {
    if (v[run.read_rows[j]] != s.rows[j]) {
      mark_in_run(users_.row_first, users_.row_records, run.read_rows[j]);
    }
  }
  for (size_t j = 0; j < base_count; ++j) {
    if (bases[segment_bases[j]] != s.bases[j]) {
      mark_in_run(users_.base_first, users_.base_records, segment_bases[j]);
    }
  }
  for (size_t j = 0; j < run.sites.size(); ++j) {
    if (frame.site_entries[run.sites[j]] != s.sites[j]) {
      for (size_t i : sites_[run.sites[j]].records) {
        out[i - run.first] = 1;
      }
    }
  }
  return true;
}

void HostTraceVariant::switch_form(
    Segment& run,
    size_t form,
    bool split,
    const Frame& frame) {
  std::vector<Segment::Held>& next = form_held(run, form, split);
  uint8_t* dirty = dirty_.data() + run.first;
  const size_t n = run.stop - run.first;
  if (!form_held_enabled) {
    // the state's exec as unknown
    for (Segment::Held& h : next) {
      h.held = false;
    }
    std::fill(dirty, dirty + n, 1);
    run.form = form;
    run.split = split;
    return;
  }
  const int64_t* v = frame.values.data();
  const int64_t* bases = frame.bases.data();
  const size_t g = &run - segments_.data();
  const uint32_t* segment_bases = users_.segment_bases.data() + users_.segment_first[g];
  const size_t base_count = users_.segment_first[g + 1] - users_.segment_first[g];
  if (run.seen.size() < 2 * run.execs.size()) {
    run.seen.resize(2 * run.execs.size());
  }
  // the current state's records are clean at the call's rows, bases and site
  // rows but those dirty_ marks
  Segment::Seen& a = run.seen[2 * run.form + run.split];
  a.rows.resize(run.read_rows.size());
  for (size_t j = 0; j < run.read_rows.size(); ++j) {
    a.rows[j] = v[run.read_rows[j]];
  }
  a.bases.resize(base_count);
  for (size_t j = 0; j < base_count; ++j) {
    a.bases[j] = bases[segment_bases[j]];
  }
  a.sites.resize(run.sites.size());
  for (size_t j = 0; j < run.sites.size(); ++j) {
    a.sites[j] = frame.site_entries[run.sites[j]];
  }
  a.dirty.assign(dirty, dirty + n);
  a.saved = true;
  a.users_version = users_version_;
  run.form = form;
  run.split = split;
  changed_since(run, run.seen[2 * form + split], frame, dirty);
}

void HostTraceVariant::set_split_cuts(size_t segment, const std::vector<size_t>& cuts) {
  TORCH_CHECK_VALUE(segment < segments_.size(), "no segment ", segment);
  Segment& run = segments_[segment];
  const size_t n = run.stop - run.first;
  for (size_t j = 0; j < cuts.size(); ++j) {
    TORCH_CHECK_VALUE(cuts[j] > 0 && cuts[j] < n && (j == 0 || cuts[j] > cuts[j - 1]), "a segment's cut points");
  }
  run.cuts = cuts;
}

std::vector<size_t> HostTraceVariant::split_bounds(Segment& run) {
  if (users_stale_) {
    index_users();
  }
  const size_t n = run.stop - run.first;
  // the records a call can re-set: those reading a dynamic row or an address,
  // and those set every call
  std::vector<int64_t> dyn(n + 1, 0);
  auto take = [&](const std::vector<uint32_t>& first, const std::vector<uint32_t>& records, size_t i) {
    for (uint32_t k = first[i]; k < first[i + 1]; ++k) {
      if (records[k] >= run.first && records[k] < run.stop) {
        dyn[records[k] - run.first + 1] = 1;
      }
    }
  };
  for (uint32_t row : users_.dynamic_rows) {
    take(users_.row_first, users_.row_records, row);
  }
  for (size_t b = 0; b + 1 < users_.base_first.size(); ++b) {
    take(users_.base_first, users_.base_records, b);
  }
  for (size_t i = 0; i < n; ++i) {
    dyn[i + 1] = (dyn[i + 1] || always_[run.first + i]) + dyn[i];
  }
  // a section's patch is hidden behind the GPU time of the one before it
  // while its re-set records are at most split_ratio times that one's records:
  // sections grow by split_ratio from a first of about dyn / (1 + R + ... +
  // R^(K-1)) re-set records (at least 16 records), at most split_sections of
  // them; split_first_nodes / split_max_nodes set the first section's records
  // and the most a section takes instead
  std::vector<size_t> bounds{0};
  const std::vector<size_t>& cuts = run.cuts;
  if (split_max_nodes > 0) {
    // each section the most records up to split_max_nodes (a longer one where
    // no cut is nearer)
    const auto most = static_cast<size_t>(split_max_nodes);
    while (n - bounds.back() > most) {
      const size_t lo = bounds.back();
      size_t take = 0;
      for (size_t c : cuts) {
        if (c <= lo) {
          continue;
        }
        if (c - lo <= most || take == 0) {
          take = c;
        }
        if (c - lo >= most) {
          break;
        }
      }
      if (take == 0) {
        break;
      }
      bounds.push_back(take);
    }
    bounds.push_back(n);
    return bounds;
  }
  double sum = 0;
  for (int64_t j = 0; j < split_sections; ++j) {
    sum += std::pow(split_ratio, static_cast<double>(j));
  }
  const auto target = static_cast<int64_t>(std::ceil(static_cast<double>(dyn[n]) / std::max(sum, 1.0)));
  auto first = std::find_if(cuts.begin(), cuts.end(), [&](size_t c) {
    return split_first_nodes > 0 ? c >= static_cast<size_t>(split_first_nodes) : c >= 16 && dyn[c] >= target;
  });
  if (first == cuts.end()) {
    return {0, n};
  }
  bounds.push_back(*first);
  while (static_cast<int64_t>(bounds.size()) < split_sections && bounds.back() < n) {
    const size_t lo = bounds.back();
    const double room = split_ratio * static_cast<double>(lo - bounds[bounds.size() - 2]);
    if (static_cast<double>(dyn[n] - dyn[lo]) <= room) {
      break;
    }
    size_t best = 0;
    for (size_t c : cuts) {
      if (c > lo && static_cast<double>(dyn[c] - dyn[lo]) <= room) {
        best = c;
      }
    }
    if (best == 0) {
      break;
    }
    bounds.push_back(best);
  }
  bounds.push_back(n);
  return bounds;
}

void HostTraceVariant::upload_once(Segment& run, size_t form) {
  if (run.uploaded.size() < run.execs.size()) {
    run.uploaded.resize(run.execs.size(), 0);
  }
  if (!run.uploaded[form]) {
    C10_CUDA_CHECK(cudaGraphUpload(reinterpret_cast<cudaGraphExec_t>(run.execs[form]), at::cuda::getCurrentCUDAStream(device_)));
    run.uploaded[form] = 1;
    ++exec_uploads;
  }
}

bool HostTraceVariant::split_ready(Segment& run, size_t form, const Frame& frame) {
  if (run.splits.size() < run.execs.size()) {
    run.splits.resize(run.execs.size());
  }
  Segment::Split& sp = run.splits[form];
  if (sp.tried) {
    return sp.ok;
  }
  sp.tried = true;
  const size_t n = run.stop - run.first;
  sp.bounds = split_bounds(run);
  if (sp.bounds.size() < 3) {
    return false;
  }
  Segment::Clone* clone = form == 0 ? nullptr : &run.clones[form - 1];
  auto node_of = [&](size_t i) {
    return clone ? clone->nodes[i] : records_[run.first + i].node;
  };
  auto range_of = [&](size_t i) {
    return static_cast<size_t>(std::upper_bound(sp.bounds.begin(), sp.bounds.end(), i) - sp.bounds.begin() - 1);
  };
  const size_t pieces = clone ? clone->pieces.size() : 0;
  sp.nodes.assign(n, nullptr);
  sp.piece_nodes.assign(pieces, nullptr);
  sp.piece_range.assign(pieces, 0);
  for (size_t j = 0; j < pieces; ++j) {
    // a site's pieces replace its records, all in one range
    sp.piece_range[j] = range_of(sites_[clone->pieces[j].site].records.front() - run.first);
  }
  const int64_t* v = frame.values.data();
  const int64_t* bases = frame.bases.data();
  const auto graph = form == 0 ? run.native->raw_cuda_graph() : reinterpret_cast<cudaGraph_t>(clone->graph);
  auto fail = [&]() {
    (void)cudaGetLastError();
    for (CUgraphExec e : sp.execs) {
      C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(e)));
    }
    for (CUgraph h : sp.graphs) {
      C10_CUDA_CHECK_WARN(cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(h)));
    }
    sp.execs.clear();
    sp.graphs.clear();
    return false;
  };
  for (size_t k = 0; k + 1 < sp.bounds.size(); ++k) {
    cudaGraph_t sub = nullptr;
    if (cudaGraphClone(&sub, graph) != cudaSuccess) {
      return fail();
    }
    sp.graphs.push_back(reinterpret_cast<CUgraph>(sub));
    auto take = [&](CUgraphNode original, bool inside, CUgraphNode& out) {
      cudaGraphNode_t node = nullptr;
      if (cudaGraphNodeFindInClone(&node, reinterpret_cast<cudaGraphNode_t>(original), sub) != cudaSuccess) {
        return false;
      }
      if (inside) {
        out = reinterpret_cast<CUgraphNode>(node);
        return true;
      }
      return cudaGraphDestroyNode(node) == cudaSuccess;
    };
    for (size_t i = 0; i < n; ++i) {
      CUgraphNode unused = nullptr;
      if (node_of(i) && !take(node_of(i), range_of(i) == k, range_of(i) == k ? sp.nodes[i] : unused)) {
        return fail();
      }
    }
    for (size_t j = 0; j < pieces; ++j) {
      CUgraphNode unused = nullptr;
      if (!take(clone->pieces[j].node, sp.piece_range[j] == k, sp.piece_range[j] == k ? sp.piece_nodes[j] : unused)) {
        return fail();
      }
    }
    // the driver reads a memcpy node's operands at instantiation: the call's
    auto live = [&](const Record& r, CUgraphNode node) {
      if (!node || r.kind != Kind::Memcpy) {
        return true;
      }
      const int64_t row = r.site >= 0 ? frame.site_entries[r.site] : 0;
      const MemcpyRow& m = memcpy_row(r, row);
      const std::array<int64_t, 3> at = memcpy_shape(m, v, bases);
      return at[2] > 0 &&
          cudaGraphMemcpyNodeSetParams1D(
              reinterpret_cast<cudaGraphNode_t>(node),
              reinterpret_cast<void*>(at[0]),
              reinterpret_cast<const void*>(at[1]),
              static_cast<size_t>(at[2]),
              m.kind) == cudaSuccess;
    };
    for (size_t i = 0; i < n; ++i) {
      if (range_of(i) == k && !live(records_[run.first + i], sp.nodes[i])) {
        return fail();
      }
    }
    for (size_t j = 0; j < pieces; ++j) {
      if (sp.piece_range[j] == k && !live(clone->pieces[j], sp.piece_nodes[j])) {
        return fail();
      }
    }
    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiateWithFlags(&exec, sub, cudaGraphInstantiateFlagAutoFreeOnLaunch | cudaGraphInstantiateFlagUseNodePriority) != cudaSuccess) {
      return fail();
    }
    sp.execs.push_back(reinterpret_cast<CUgraphExec>(exec));
    // before any node is set (an update of a never-uploaded exec can leave the
    // node on a slow path for good)
    C10_CUDA_CHECK(cudaGraphUpload(exec, at::cuda::getCurrentCUDAStream(device_)));
    ++exec_uploads;
  }
  ++split_builds;
  sp.calls_set.assign(sp.execs.size(), 0);
  sp.nodes_set.assign(sp.execs.size(), 0);
  sp.launches.assign(sp.execs.size(), 0);
  sp.ok = true;
  return true;
}

bool HostTraceVariant::set_argument_copies(Frame& frame) {
  const int64_t* v = frame.values.data();
  const size_t first =
      !steps_.empty() && steps_[0].segment >= 0 ? steps_[0].segment : SIZE_MAX;
  for (size_t g = 0; g < segments_.size(); ++g) {
    if (g == first) {
      continue;
    }
    Segment& run = segments_[g];
    const size_t form =
        run.forms.size() > 1 ? frame.forms[g] : 0;
    std::vector<Segment::Held>& form_state = form_held(run, form);
    for (size_t i = run.first; i < run.stop; ++i) {
      Record& r = records_[i];
      Segment::Held& h = form_state[i - run.first];
      // an entry's memcpy (a section's, in its piece) is set by its row in patch
      if (r.kind != Kind::Memcpy || (r.site >= 0 && frame.site_entries[r.site] != 0)) {
        continue;
      }
      // an argument's address is its row's value; another operand keeps the
      // node's (an allocation of the caching allocator's)
      std::array<int64_t, 3> held = h.copy;
      bool moved = false;
      for (const auto& [f, k] : {std::pair{&r.copy.dst, 0}, std::pair{&r.copy.src, 1}}) {
        if (f->base < 0) {
          const int64_t x = field_value(*f, v, nullptr);
          moved |= x != held[k];
          held[k] = x;
        }
      }
      if (!moved) {
        continue;
      }
      CUgraphNode node =
          form == 0 ? r.node : run.clones[form - 1].nodes[i - run.first];
      const cudaError_t set = cudaGraphExecMemcpyNodeSetParams1D(
          reinterpret_cast<cudaGraphExec_t>(run.execs[form]),
          reinterpret_cast<cudaGraphNode_t>(node),
          reinterpret_cast<void*>(held[0]),
          reinterpret_cast<const void*>(held[1]),
          static_cast<size_t>(held[2]),
          r.copy.kind);
      if (set == cudaErrorInvalidValue) {
        (void)cudaGetLastError();
        return false;
      }
      C10_CUDA_CHECK(set);
      ++memory_node_sets;
      h.copy = held;
    }
  }
  return true;
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
  CUgraphExec exec = run.execs[form];
  // the hooks' Python replay runs the graph's own exec, whole
  const bool hooks = observed(run);
  // the philox offsets the segment's RNG calls take before the next, and
  // the intragraph offset of the current one's kernels
  int64_t running = 0;
  int64_t philox = 0;
  auto patch = [&](Record& r, Segment::Held& h, CUgraphNode node) {
    const int64_t row = r.site >= 0 ? frame.site_entries[r.site] : 0;
    if (r.kind == Kind::Memset) {
      const MemsetRow& m = memset_row(r, row);
      const std::array<int64_t, 4> held = memset_shape(m, v, bases);
      // a node holds its destination's address, not the allocation there
      // (probe_memset_alloc.py: after a free and reallocation, or a VMM remap,
      // at the address, the unchanged node writes the new memory)
      if (h.held && h.row == row && held == h.memset) {
        return;
      }
      if (h.held && h.row != row && held == h.memset) {
        const MemsetRow& before = memset_row(r, h.row);
        if (before.value == m.value && before.element_size == m.element_size) {
          h.row = row;
          return;
        }
      }
      h.held = false;
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
      h.memset = held;
      h.row = row;
      h.held = true;
      return;
    }
    if (r.kind == Kind::Memcpy) {
      const MemcpyRow& m = memcpy_row(r, row);
      const std::array<int64_t, 3> held = memcpy_shape(m, v, bases);
      // a device node holds its operands' addresses; a host one is set every
      // replay (index_users)
      if (h.held && h.row == row && held == h.copy && m.kind == cudaMemcpyDeviceToDevice) {
        return;
      }
      h.held = false;
      TORCH_CHECK_VALUE(held[2] > 0, "host_trace: a memcpy of ", held[2], " bytes");
      const cudaError_t set = cudaGraphExecMemcpyNodeSetParams1D(
          reinterpret_cast<cudaGraphExec_t>(exec),
          reinterpret_cast<cudaGraphNode_t>(node),
          reinterpret_cast<void*>(held[0]),
          reinterpret_cast<const void*>(held[1]),
          static_cast<size_t>(held[2]),
          m.kind);
      if (set == cudaErrorInvalidValue) {
        // an operand of the other kind of memory: the node is unchanged
        (void)cudaGetLastError();
        throw MemoryKindMismatch{};
      }
      C10_CUDA_CHECK(set);
      ++memory_node_sets;
      TORCH_CHECK(
          fail_after_setter < 0 || fail_after_setter-- != 0,
          "host_trace: a failure injected after a setter");
      h.copy = held;
      h.row = row;
      h.held = true;
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
    if (h.held && dims == h.dims && h.image.size() == k.nbytes &&
        std::memcmp(k.image, h.image.data(), k.nbytes) == 0) {
      if (h.row == row) {
        return;
      }
      // another key's row whose launch is the node's as it is (a cuBLAS key
      // of another M running the same kernel at the same grid and arguments)
      const KernelRow& before = kernel_row(r, h.row);
      if (before.params.func == k.params.func &&
          before.param_sizes == k.param_sizes) {
        h.row = row;
        ++kernel_row_switches;
        return;
      }
    }
    // a patch cut short leaves the node unknown, never stale
    h.held = false;
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
    raise_dynamic_smem_limit(k.params.func, k.params.sharedMemBytes);
    C10_CUDA_DRIVER_CHECK(
        driver->cuGraphExecKernelNodeSetParams_(exec, node, &k.params));
    ++kernel_node_sets;
    TORCH_CHECK(
        fail_after_setter < 0 || fail_after_setter-- != 0,
        "host_trace: a failure injected after a setter");
    h.image.assign(k.image, k.image + k.nbytes);
    h.dims = dims;
    h.row = row;
    h.held = true;
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
  const size_t n = run.stop - run.first;
  // a split run launches its form's sections every call, each patched while
  // the one before it runs; the whole exec only without sections
  bool split = split_enabled && !hooks && !run.cuts.empty() && !run.generator;
  if (split && split_dirty_min > 0) {
    // the records this call patches in the current state's exec (a switch to the other state's patches those its
    // exec last held otherwise: changed_since). Decided before the sections are built: split_ready uploads them,
    // and an uploaded exec this call does not launch slows every graph kernel of the context until it launches
    const uint8_t* d = dirty_.data() + run.first;
    const auto dirty = static_cast<int64_t>(std::count(d, d + n, uint8_t{1}));
    if (dirty < split_dirty_min) {
      split = false;
      ++split_dirty_wholes;
    }
  }
  split = split && split_ready(run, form, frame);
  if (!split) {
    // the exec this call launches, uploaded here on its first launch (not at capture or instantiation): an uploaded
    // exec not yet launched slows every graph kernel of the context until its first launch; uploaded before any of
    // its nodes is set (a set before an exec's first upload leaves the node on a slow path for good)
    upload_once(run, form);
  }
  if (form != run.form || split != run.split) {
    switch_form(run, form, split, frame);
  }
  std::vector<Segment::Held>& held = form_held(run, form, split);
  Segment::Split* sp = split ? &run.splits[form] : nullptr;
  // range k's records (all of them whole: k -1) and its sites' pieces
  auto patch_range = [&](int64_t k) {
    const size_t lo = k < 0 ? 0 : sp->bounds[k];
    const size_t hi = k < 0 ? n : sp->bounds[k + 1];
    for (size_t i = run.first + lo; i < run.first + hi; ++i) {
      CUgraphNode node = sp ? sp->nodes[i - run.first] : clone ? clone->nodes[i - run.first] : records_[i].node;
      if (node && dirty_[i]) {
        patch(records_[i], held[i - run.first], node);
        dirty_[i] = always_[i];
      }
    }
    if (!clone) {
      return;
    }
    thread_local std::vector<int64_t> reads;
    for (Segment::Clone::Group& group : clone->groups) {
      if (sp && (group.first == group.stop || sp->piece_range[group.first] != static_cast<size_t>(k))) {
        continue;
      }
      const Site& s = sites_[group.site];
      reads.clear();
      reads.push_back(frame.site_entries[group.site]);
      for (int64_t row : s.piece_rows) {
        reads.push_back(v[row]);
      }
      for (int64_t b : s.piece_bases) {
        reads.push_back(bases[b]);
      }
      bool same = !s.piece_always && group.users_version[split] == users_version_ && reads == group.seen[split];
      for (size_t j = group.first; same && j < group.stop; ++j) {
        same = held[n + j].held;
      }
      if (same) {
        ++piece_skips;
        continue;
      }
      for (size_t j = group.first; j < group.stop; ++j) {
        Record& r = clone->pieces[j];
        patch(r, held[n + j], sp ? sp->piece_nodes[j] : r.node);
      }
      group.seen[split] = reads;
      group.users_version[split] = users_version_;
    }
  };
  if (sp) {
    ++split_calls;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_);
    for (size_t k = 0; k + 1 < sp->bounds.size(); ++k) {
      exec = sp->execs[k];
      const int64_t before = kernel_node_sets + memory_node_sets;
      patch_range(static_cast<int64_t>(k));
      const int64_t set = kernel_node_sets + memory_node_sets - before;
      sp->calls_set[k] += set > 0;
      sp->nodes_set[k] += set;
      ++sp->launches[k];
      ++graph_launches;
      // the first launch runs the graph's replay prologue, once per call
      if (k == 0) {
        run.native->replay_exec(reinterpret_cast<cudaGraphExec_t>(exec));
      } else {
        C10_CUDA_CHECK(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(exec), stream));
      }
    }
  } else {
    patch_range(-1);
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
  if (sp) {
    // launched range by range above
  } else if (hooks) {
    ++graph_launches;
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
    ++graph_launches;
    run.native->replay_exec(reinterpret_cast<cudaGraphExec_t>(exec));
  }
  phase_add(6, pt);
}

} // namespace torch::cuda::host_trace
#endif
