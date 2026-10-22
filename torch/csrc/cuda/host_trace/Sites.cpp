#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM)
#include <torch/csrc/autograd/python_variable.h>

#include <c10/util/ScopeExit.h>
#include <c10/util/SmallVector.h>

#include <algorithm>
#include <numeric>
#include <utility>

namespace torch::cuda::host_trace {

int64_t delta_skips = 0;
int64_t delta_partials = 0;
int64_t delta_fulls = 0;
int64_t delta_rows_evaluated = 0;
bool delta_enabled = true;
int64_t delta_slots = 3;
int64_t delta_exchanges = 0;
int64_t valid_first_misses = 0;
bool variant_order_enabled = true;
bool placement_reuse_enabled = true;
bool form_held_enabled = true;
bool entry_descriptors_enabled = true;

namespace {

uint64_t key_hash(const int64_t* key, size_t n) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (size_t i = 0; i < n; ++i) {
    h = (h ^ static_cast<uint64_t>(key[i])) * 0x100000001b3ULL;
    h ^= h >> 29;
  }
  return h;
}

HostTraceBindingTable::Source parse_table_source(py::handle h) {
  auto [scratch, index, delta] = h.cast<std::tuple<bool, uint32_t, int64_t>>();
  return {scratch, index, delta};
}

} // namespace

bool native_fill_enabled = true;
int64_t native_fills = 0;

void HostTraceBindingTable::add(
    const std::vector<int64_t>& key,
    py::handle nodes,
    const std::vector<int64_t>& scratch,
    int64_t topology) {
  TORCH_CHECK_VALUE(bindings_.empty() || key.size() == width_, "a table key of ", key.size(), " values");
  width_ = key.size();
  if (find(key.data(), key.size()) != nullptr) {
    return;
  }
  Binding b;
  b.scratch = scratch;
  b.topology = topology;
  for (py::handle h : nodes.cast<py::tuple>()) {
    auto t = h.cast<py::tuple>();
    Node n{};
    n.memset = t[0].cast<uint8_t>() == 1;
    if (n.memset) {
      n.dst = parse_table_source(t[1]);
      n.value = t[2].cast<unsigned int>();
      n.element_size = t[3].cast<unsigned int>();
      TORCH_CHECK_VALUE(n.element_size == 1 || n.element_size == 2 || n.element_size == 4, "a memset element of ", n.element_size, " bytes");
      n.shape = {t[4].cast<int64_t>(), t[5].cast<int64_t>(), t[6].cast<int64_t>()};
    } else {
      n.function = t[1].cast<uintptr_t>();
      const auto block = t[2].cast<std::array<int64_t, 3>>();
      const auto grid = t[4].cast<std::array<int64_t, 3>>();
      n.dims = {grid[0], grid[1], grid[2], block[0], block[1], block[2], t[3].cast<int64_t>()};
      for (py::handle image : t[5].cast<py::tuple>()) {
        n.images.push_back(image.cast<std::string>());
      }
      for (py::handle f : t[6].cast<py::tuple>()) {
        auto ft = f.cast<py::tuple>();
        Field field{ft[0].cast<uint32_t>(), ft[1].cast<uint32_t>(), parse_table_source(py::make_tuple(ft[2], ft[3], ft[4]))};
        TORCH_CHECK_VALUE(field.param < n.images.size() && field.offset + 8 <= n.images[field.param].size(), "a field outside its parameter");
        n.fields.push_back(field);
      }
    }
    b.nodes.push_back(std::move(n));
  }
  index_[key_hash(key.data(), width_)].push_back(static_cast<uint32_t>(bindings_.size()));
  keys_.insert(keys_.end(), key.begin(), key.end());
  bindings_.push_back(std::move(b));
}

const HostTraceBindingTable::Binding* HostTraceBindingTable::find(const int64_t* key, size_t n) const {
  if (n != width_ || bindings_.empty()) {
    return nullptr;
  }
  auto it = index_.find(key_hash(key, n));
  if (it == index_.end()) {
    return nullptr;
  }
  for (uint32_t i : it->second) {
    if (std::equal(key, key + n, keys_.begin() + static_cast<ptrdiff_t>(i) * n)) {
      return &bindings_[i];
    }
  }
  return nullptr;
}

void HostTraceVariant::set_site_table(
    size_t site,
    std::shared_ptr<HostTraceBindingTable> table,
    const std::vector<int8_t>& codes,
    const std::vector<int32_t>& companions,
    const std::vector<std::pair<int64_t, int64_t>>& operands,
    const std::vector<std::array<int64_t, 4>>& scratch,
    const std::vector<int64_t>& native_scratch,
    int64_t topology) {
  TORCH_CHECK_VALUE(site < sites_.size() && sites_[site].predicates.empty(), "no keyed site ", site);
  Site& s = sites_[site];
  TORCH_CHECK_VALUE(codes.size() == s.key_rows.size() && companions.size() == codes.size(), "a table link of ", codes.size(), " codes");
  for (size_t j = 0; j < codes.size(); ++j) {
    TORCH_CHECK_VALUE(
        codes[j] >= 0 && codes[j] <= 2 && (codes[j] != 2 || (companions[j] >= 0 && static_cast<size_t>(companions[j]) < codes.size())),
        "a table link's code ",
        static_cast<int>(codes[j]));
  }
  for (const auto& [row, base] : operands) {
    check_row(row);
    TORCH_CHECK_VALUE(base == -1 || check_base(base) >= 0, "no base ", base);
  }
  for (const auto& x : scratch) {
    check_row(x[1]);
    TORCH_CHECK_VALUE(x[2] == -1 || check_base(x[2]) >= 0, "no base ", x[2]);
  }
  if (native_scratch.size() != s.scratch.size()) {
    return; // add_row would decline its rows' scratch bytes: _fill's path
  }
  s.link = std::make_unique<Site::TableLink>(Site::TableLink{std::move(table), codes, companions, operands, scratch, native_scratch, topology});
}

int64_t HostTraceVariant::fill_from_table(size_t i, const int64_t* key) {
  Site& s = sites_[i];
  if (!native_fill_enabled || !s.link) {
    return -1;
  }
  const Site::TableLink& l = *s.link;
  const size_t n = s.key_rows.size();
  c10::SmallVector<int64_t, 32> normal(key, key + n);
  for (size_t j = 0; j < n; ++j) {
    if (l.codes[j] == 1) {
      // _host_trace_harvest._class: the largest power of two dividing the address, up to 256
      normal[j] = key[j] ? (key[j] & -key[j]) : 256;
    } else if (l.codes[j] == 2 && key[l.companions[j]] == 1) {
      normal[j] = 0;
    }
  }
  const HostTraceBindingTable::Binding* b = l.table->find(normal.data(), n);
  if (b == nullptr || b->topology != l.topology || b->nodes.size() != s.records.size()) {
    return -1;
  }
  // site.fits: every scratch buffer the binding takes is one of the site's, within its bytes
  auto scratch_of = [&](uint32_t index) -> const std::array<int64_t, 4>* {
    for (const auto& x : l.scratch) {
      if (x[0] == static_cast<int64_t>(index)) {
        return &x;
      }
    }
    return nullptr;
  };
  for (size_t j = 0; j < b->scratch.size(); ++j) {
    const auto* x = scratch_of(static_cast<uint32_t>(j));
    if (x == nullptr || ((*x)[3] >= 0 && b->scratch[j] > (*x)[3])) {
      return -1;
    }
  }
  auto source = [&](const HostTraceBindingTable::Source& src, Field& f) {
    if (src.scratch) {
      const auto* x = scratch_of(src.index);
      if (x == nullptr) {
        return false;
      }
      f.row = (*x)[1];
      f.base = (*x)[2];
    } else {
      if (src.index >= l.operands.size()) {
        return false;
      }
      f.row = l.operands[src.index].first;
      f.base = l.operands[src.index].second;
    }
    f.delta = src.delta;
    return true;
  };
  auto e = std::make_unique<Entry>();
  e->arm = 0;
  for (int64_t j : l.native_scratch) {
    e->scratch.push_back(j >= 0 && static_cast<size_t>(j) < b->scratch.size() ? b->scratch[j] : 0);
  }
  e->kernels.resize(b->nodes.size());
  e->memsets.resize(b->nodes.size());
  for (size_t k = 0; k < b->nodes.size(); ++k) {
    const HostTraceBindingTable::Node& node = b->nodes[k];
    if (records_[s.records[k]].kind != (node.memset ? Kind::Memset : Kind::Kernel)) {
      return -1;
    }
    if (node.memset) {
      MemsetRow& m = e->memsets[k];
      m.dst = Field{0, 0, 8, true, 0, -1, 0};
      if (!source(node.dst, m.dst)) {
        return -1;
      }
      m.value = node.value;
      m.element_size = node.element_size;
      m.constant_shape = true;
      m.shape = node.shape;
      continue;
    }
    KernelRow& kr = e->kernels[k];
    kr.params.func = reinterpret_cast<CUfunction>(node.function);
    kr.dims = node.dims;
    kr.constant_dims = true;
    kr.offset = e->image.size();
    for (const std::string& image : node.images) {
      e->image.resize((e->image.size() + 7) / 8 * 8);
      kr.param_offsets.push_back(e->image.size() - kr.offset);
      kr.param_sizes.push_back(static_cast<uint32_t>(image.size()));
      e->image.insert(e->image.end(), image.begin(), image.end());
    }
    kr.nbytes = e->image.size() - kr.offset;
    kr.first_field = e->fields.size();
    for (const HostTraceBindingTable::Field& tf : node.fields) {
      Field f{tf.param, tf.offset, 8, true, 0, -1, 0};
      if (!source(tf.source, f)) {
        return -1;
      }
      e->fields.push_back(f);
    }
    kr.field_count = e->fields.size() - kr.first_field;
  }
  e->held = e->image;
  for (size_t k = 0; k < b->nodes.size(); ++k) {
    if (!b->nodes[k].memset) {
      bind_row(e->kernels[k], e->image.data(), e->held.data(), e->fields.data());
    }
  }
  row_added(s, *e);
  s.entries.push_back(std::move(e));
  insert(s, key, static_cast<int64_t>(s.entries.size()));
  ++native_fills;
  return static_cast<int64_t>(s.rows.size()) - 1;
}

int64_t HostTraceVariant::find(const Site& s, const int64_t* key) const {
  const size_t n = s.key_rows.size();
  auto it = s.index.find(key_hash(key, n));
  if (it == s.index.end()) {
    return -1;
  }
  for (uint32_t i : it->second) {
    if (std::equal(key, key + n, s.keys.begin() + i * n)) {
      return i;
    }
  }
  return -1;
}

int64_t HostTraceVariant::select(const Site& s, const int64_t* values) {
  for (size_t k = 0; k < s.predicates.size(); ++k) {
    if (values[s.predicates[k]] == 1) {
      return static_cast<int64_t>(k);
    }
  }
  return -1;
}

void HostTraceVariant::insert(Site& s, const int64_t* key, int64_t row) {
  const size_t n = s.key_rows.size();
  s.index[key_hash(key, n)].push_back(static_cast<uint32_t>(s.rows.size()));
  s.keys.insert(s.keys.end(), key, key + n);
  s.rows.push_back(row);
}

HostTraceVariant::KernelRow& HostTraceVariant::kernel_row(
    Record& r,
    int64_t row) const {
  return row == 0 ? r.kernel
                  : sites_[r.site].entries[row - 1]->kernels[r.site_pos];
}

const HostTraceVariant::KernelRow& HostTraceVariant::kernel_row(
    const Record& r,
    int64_t row) const {
  return row == 0 ? r.kernel
                  : sites_[r.site].entries[row - 1]->kernels[r.site_pos];
}

const HostTraceVariant::MemcpyRow& HostTraceVariant::memcpy_row(
    const Record& r,
    int64_t row) const {
  return row == 0 ? r.copy
                  : sites_[r.site].entries[row - 1]->memcpys[r.site_pos];
}

const HostTraceVariant::MemsetRow& HostTraceVariant::memset_row(
    const Record& r,
    int64_t row) const {
  return row == 0 ? r.memset
                  : sites_[r.site].entries[row - 1]->memsets[r.site_pos];
}

bool HostTraceVariant::evaluate_rows(
    PyObject* const* args,
    size_t count,
    Frame& frame) const {
  int64_t pt = phase_now();
  c10::SmallVector<int64_t, 64> leaves(program_.num_leaves());
  if (!program_.bind(args, count, leaves.data())) {
    return false;
  }
  phase_add(2, pt);
  frame.args = args;
  frame.count = count;
  frame.values.resize_for_overwrite(program_.num_rows());
  if (valid_split_rows_ != program_.num_rows()) {
    // the rows the valid row reads first: a call another variant serves stops there
    std::vector<uint32_t> all(program_.num_rows());
    std::iota(all.begin(), all.end(), 0);
    program_.partition(static_cast<int64_t>(valid_), all, valid_first_rows_, valid_other_rows_);
    valid_split_rows_ = program_.num_rows();
  }
  frame.evaluated = false;
  if (program_.evaluate(valid_first_rows_, leaves.data(), frame.values.data()) != HostTraceProgram::Status::Success ||
      frame.values[valid_] != 1) {
    ++valid_first_misses;
    return false;
  }
  frame.evaluated = program_.evaluate(valid_other_rows_, leaves.data(), frame.values.data()) ==
      HostTraceProgram::Status::Success;
  if (!frame.evaluated) {
    return false;
  }
  frame.overlapped = !disjoint(args, count);
  phase_add(3, pt);
  return !frame.overlapped;
}

// after the rows, which hold only for nonempty tensor arguments
bool HostTraceVariant::disjoint(PyObject* const* args, size_t count) const {
  return disjoint(args, count, argument_pairs_, nullptr);
}

bool HostTraceVariant::disjoint(
    PyObject* const* args,
    size_t count,
    const std::vector<ArgumentPair>& pairs,
    const std::vector<size_t>* used) const {
  if (pairs.empty()) {
    return true;
  }
  // each argument's first and last byte
  c10::SmallVector<std::pair<uintptr_t, uintptr_t>, 8> extents(
      pair_arguments_.size());
  auto extent = [&](size_t k) {
    const size_t i = pair_arguments_[k];
    if (i >= count || !THPVariable_CheckExact(args[i])) {
      return false;
    }
    const at::Tensor& t = THPVariable_Unpack(args[i]);
    if (t.layout() != at::kStrided) {
      return false;
    }
    const auto sizes = t.sizes();
    const auto strides = t.strides();
    int64_t span = 0;
    for (size_t d = 0; d < sizes.size(); ++d) {
      span += (sizes[d] - 1) * strides[d];
    }
    const auto first = reinterpret_cast<uintptr_t>(t.const_data_ptr());
    extents[k] = {first, first + (span + 1) * t.itemsize() - 1};
    return true;
  };
  if (used) {
    for (size_t k : *used) {
      if (!extent(k)) {
        return false;
      }
    }
  } else {
    for (size_t k = 0; k < pair_arguments_.size(); ++k) {
      if (!extent(k)) {
        return false;
      }
    }
  }
  for (const ArgumentPair& p : pairs) {
    const auto [a_first, a_last] = extents[p.a];
    const auto [b_first, b_last] = extents[p.b];
    if (a_first <= b_last && b_first <= a_last) {
      return false;
    }
  }
  return true;
}

bool HostTraceVariant::evaluate(
    PyObject* const* args,
    size_t count,
    Frame& frame) const {
  if (!evaluate_rows(args, count, frame)) {
    return false;
  }
  int64_t pt = phase_now();
  const bool resolved = evaluate_sites(frame);
  phase_add(3, pt);
  return resolved;
}

bool HostTraceVariant::evaluate_static(
    PyObject* const* args,
    size_t count,
    Frame& frame,
    uint64_t generation) {
  if (generation == 0 || generation != cached_generation_ || cache_busy_) {
    delta_valid_ = delta_marked_ = false;
    frame.evaluated = false;
    const bool hit = evaluate(args, count, frame);
    if (!hit && frame.evaluated && generation != 0 && !cache_busy_ &&
        static_prefix_ != 0) {
      // a miss's static rows are the generation's too; the cache is not the
      // patched values' (seen_generation_), which only a commit moves
      std::swap(frame.values, cached_values_);
      cached_generation_ = generation;
      site_memo_cache_ = false;
    }
    return hit;
  }
  // moved, not swapped: a fresh frame's values are inline, and a swap with
  // them would copy every row (and allocate their buffer) at each call
  frame.values = std::move(cached_values_);
  cache_busy_ = true;
  frame.cached = this;
  // whether the memo's reads are the cache's as this call found it; not
  // after the call changes the cache, until its sites resolve
  bool memo_fresh = site_memo_cache_;
  site_memo_cache_ = false;
  // the rows this call changes: none (no dynamic leaf changed) or delta's
  // (restricted), else every dynamic row
  bool restricted = false;
  DeltaRows* delta = nullptr;
  int64_t pt = phase_now();
  c10::SmallVector<int64_t, 64> leaves;
  leaves.resize_for_overwrite(program_.num_leaves());
  if (!program_.bind(args, count, dynamic_leaves_, leaves.data())) {
    return false;
  }
  phase_add(2, pt);
  frame.args = args;
  frame.count = count;
  int64_t* v = frame.values.data();
  const size_t n = dynamic_leaves_.size();
  const std::vector<uint32_t>* valid_rows = &dynamic_valid_rows_;
  const std::vector<uint32_t>* other_rows = &dynamic_other_rows_;
  if (delta_enabled && delta_valid_) {
    auto differing = [&](const int64_t* other) {
      size_t c = 0;
      for (size_t j = 0; j < n; ++j) {
        c += leaves[dynamic_leaves_[j]] != other[j];
      }
      return c;
    };
    bool exchanged = false;
    if (delta_slots > 0 && !delta_slots_.empty()) {
      size_t fewest = differing(delta_leaves_.data());
      DeltaSlot* best = nullptr;
      for (DeltaSlot& d : delta_slots_) {
        size_t c = 0;
        if (fewest != 0 && d.generation == cached_generation_ &&
            (c = differing(d.leaves.data())) < fewest) {
          fewest = c;
          best = &d;
        }
      }
      if (best) {
        exchange(*best, v, memo_fresh);
        best->used = ++delta_tick_;
        ++delta_exchanges;
        exchanged = true;
      }
    }
    c10::SmallVector<uint64_t, 4> changed((n + 63) / 64, 0);
    bool any = false;
    for (size_t j = 0; j < n; ++j) {
      if (leaves[dynamic_leaves_[j]] != delta_leaves_[j]) {
        changed[j / 64] |= uint64_t{1} << (j % 64);
        any = true;
      }
    }
    restricted = true;
    if (!any) {
      valid_rows = nullptr;
      ++delta_skips;
    } else {
      delta = &delta_rows(changed);
      valid_rows = &delta->valid_rows;
      other_rows = &delta->other_rows;
      // where the call re-evaluates many rows (another size), the cache's are
      // kept first for a later call at its leaves; a few (a length) are not
      const size_t m = static_cast<size_t>(std::max<int64_t>(delta_slots, 0));
      if (!exchanged && m > 0 &&
          (valid_rows->size() + other_rows->size()) * 8 >= dynamic_rows_.size()) {
        DeltaSlot* d = nullptr;
        if (delta_slots_.size() < m) {
          d = &delta_slots_.emplace_back();
        } else {
          d = &*std::min_element(delta_slots_.begin(), delta_slots_.end(), [](const DeltaSlot& x, const DeltaSlot& y) { return x.used < y.used; });
        }
        d->generation = cached_generation_;
        d->used = ++delta_tick_;
        d->leaves = delta_leaves_;
        d->rows.resize(dynamic_rows_.size());
        for (size_t j = 0; j < dynamic_rows_.size(); ++j) {
          d->rows[j] = v[dynamic_rows_[j]];
        }
        d->memo = memo_fresh;
        d->memo_armed = site_memo_armed_;
        d->read_values = site_read_values_;
        d->entries = site_memo_entries_;
        d->forms = site_memo_forms_;
      }
      delta_last_changed_.assign(changed.begin(), changed.end());
      ++delta_partials;
    }
  } else {
    ++delta_fulls;
  }
  if (valid_rows) {
    // a call cut short leaves the rows partly evaluated
    delta_valid_ = delta_marked_ = false;
    if (program_.evaluate(*valid_rows, leaves.data(), v) !=
            HostTraceProgram::Status::Success ||
        v[valid_] != 1 ||
        program_.evaluate(*other_rows, leaves.data(), v) !=
            HostTraceProgram::Status::Success) {
      return false;
    }
    delta_rows_evaluated += static_cast<int64_t>(valid_rows->size() + other_rows->size());
    delta_leaves_.resize(n);
    for (size_t j = 0; j < n; ++j) {
      delta_leaves_[j] = leaves[dynamic_leaves_[j]];
    }
    delta_valid_ = true;
  }
  frame.overlapped =
      !disjoint(args, count, dynamic_pairs_, &dynamic_pair_arguments_);
  if (frame.overlapped) {
    phase_add(3, pt);
    return false;
  }
  // with a fresh memo, a site read can differ from it only among the rows
  // this call evaluated: none does, the memo's rows are the call's
  bool memo_holds = memo_fresh && restricted &&
      site_memo_armed_ == static_cast<int64_t>(armed_.size());
  if (memo_holds && delta) {
    if (delta->site_reads_version != site_reads_version_) {
      delta->site_reads.clear();
      for (const auto* rows : {&delta->valid_rows, &delta->other_rows}) {
        for (uint32_t r : *rows) {
          auto it = std::lower_bound(site_reads_.begin(), site_reads_.end(), r);
          if (it != site_reads_.end() && *it == r) {
            delta->site_reads.push_back(static_cast<uint32_t>(it - site_reads_.begin()));
          }
        }
      }
      delta->site_reads_version = site_reads_version_;
    }
    for (uint32_t j : delta->site_reads) {
      memo_holds = memo_holds && v[site_reads_[j]] == site_read_values_[j];
    }
  }
  bool resolved = true;
  if (memo_holds) {
    frame.site_entries.assign(site_memo_entries_.begin(), site_memo_entries_.end());
    if (!armed_.empty()) {
      frame.forms.assign(site_memo_forms_.begin(), site_memo_forms_.end());
    }
  } else {
    resolved = evaluate_sites(frame);
  }
  site_memo_cache_ = resolved;
  phase_add(3, pt);
  return resolved;
}

void HostTraceVariant::exchange(DeltaSlot& d, int64_t* v, bool& memo_fresh) {
  std::swap(delta_leaves_, d.leaves);
  for (size_t j = 0; j < dynamic_rows_.size(); ++j) {
    std::swap(v[dynamic_rows_[j]], d.rows[j]);
  }
  // the rows changed without a delta: mark_dirty compares them
  delta_marked_ = false;
  std::swap(site_read_values_, d.read_values);
  std::swap(site_memo_armed_, d.memo_armed);
  std::swap(site_memo_entries_, d.entries);
  std::swap(site_memo_forms_, d.forms);
  const bool kept = d.memo && site_read_values_.size() == site_reads_.size();
  d.memo = memo_fresh;
  memo_fresh = kept;
}

void HostTraceVariant::release(Frame& frame) {
  if (frame.cached == this) {
    cached_values_ = std::move(frame.values);
    cache_busy_ = false;
    frame.cached = nullptr;
  }
}

void HostTraceVariant::keep(Frame& frame, uint64_t generation) {
  if (static_prefix_ == 0 || cache_busy_ ||
      frame.values.size() != program_.num_rows()) {
    return;
  }
  std::swap(frame.values, cached_values_);
  cached_generation_ = generation;
  seen_generation_ = generation;
  site_memo_cache_ = false;
  delta_valid_ = delta_marked_ = false;
}

HostTraceVariant::DeltaRows& HostTraceVariant::delta_rows(
    c10::ArrayRef<uint64_t> changed) {
  for (DeltaRows& d : delta_memo_) {
    if (changed.equals(d.changed)) {
      return d;
    }
  }
  if (delta_memo_.size() == 4) {
    delta_memo_.erase(delta_memo_.begin());
  }
  DeltaRows d;
  d.changed.assign(changed.begin(), changed.end());
  std::vector<uint32_t> leaves, rows;
  for (size_t j = 0; j < dynamic_leaves_.size(); ++j) {
    if ((changed[j / 64] >> (j % 64)) & 1) {
      leaves.push_back(dynamic_leaves_[j]);
    }
  }
  program_.reading(leaves, dynamic_rows_, rows);
  for (uint32_t r : rows) {
    const bool valid = std::binary_search(
        dynamic_valid_rows_.begin(), dynamic_valid_rows_.end(), r);
    (valid ? d.valid_rows : d.other_rows).push_back(r);
  }
  delta_memo_.push_back(std::move(d));
  return delta_memo_.back();
}

py::list HostTraceVariant::delta_changed() const {
  py::list out;
  for (size_t j = 0; j < dynamic_leaves_.size(); ++j) {
    if (j / 64 < delta_last_changed_.size() &&
        ((delta_last_changed_[j / 64] >> (j % 64)) & 1)) {
      auto [input, kind, dim] = program_.leaf(dynamic_leaves_[j]);
      out.append(py::make_tuple(input, kind, dim));
    }
  }
  return out;
}

void HostTraceVariant::set_static_prefix(size_t n) {
  users_stale_ = true;
  users_kept_ = false;
  static_prefix_ = n;
  cached_generation_ = 0;
  site_memo_cache_ = false;
  delta_valid_ = delta_marked_ = false;
  delta_memo_.clear();
  delta_slots_.clear();
  delta_last_changed_.clear();
  dynamic_pairs_.clear();
  dynamic_pair_arguments_.clear();
  if (n == 0) {
    dynamic_rows_.clear();
    dynamic_valid_rows_.clear();
    dynamic_other_rows_.clear();
    dynamic_leaves_.clear();
    return;
  }
  program_.split(static_cast<int64_t>(n), dynamic_rows_, dynamic_leaves_);
  program_.partition(
      static_cast<int64_t>(valid_),
      dynamic_rows_,
      dynamic_valid_rows_,
      dynamic_other_rows_);
  std::vector<bool> used(pair_arguments_.size(), false);
  for (const ArgumentPair& p : argument_pairs_) {
    if (pair_arguments_[p.a] < n && pair_arguments_[p.b] < n) {
      continue;
    }
    dynamic_pairs_.push_back(p);
    used[p.a] = used[p.b] = true;
  }
  for (size_t k = 0; k < used.size(); ++k) {
    if (used[k]) {
      dynamic_pair_arguments_.push_back(k);
    }
  }
}

std::pair<size_t, size_t> HostTraceVariant::static_split() const {
  return {dynamic_rows_.size(), program_.num_rows()};
}

bool HostTraceVariant::evaluate_sites(Frame& frame) const {
  const int64_t* v = frame.values.data();
  if (site_memo_armed_ == static_cast<int64_t>(armed_.size())) {
    size_t j = 0;
    while (j < site_reads_.size() && v[site_reads_[j]] == site_read_values_[j]) {
      ++j;
    }
    if (j == site_reads_.size()) {
      frame.site_entries.assign(site_memo_entries_.begin(), site_memo_entries_.end());
      if (!armed_.empty()) {
        frame.forms.assign(site_memo_forms_.begin(), site_memo_forms_.end());
      }
      return true;
    }
  }
  if (!evaluate_sites_slow(frame)) {
    return false;
  }
  site_reads_.clear();
  for (const Site& s : sites_) {
    site_reads_.insert(site_reads_.end(), s.key_rows.begin(), s.key_rows.end());
    site_reads_.insert(site_reads_.end(), s.predicates.begin(), s.predicates.end());
  }
  std::sort(site_reads_.begin(), site_reads_.end());
  site_reads_.erase(std::unique(site_reads_.begin(), site_reads_.end()), site_reads_.end());
  site_read_values_.resize(site_reads_.size());
  for (size_t j = 0; j < site_reads_.size(); ++j) {
    site_read_values_[j] = v[site_reads_[j]];
  }
  site_memo_entries_.assign(frame.site_entries.begin(), frame.site_entries.end());
  site_memo_forms_.assign(frame.forms.begin(), frame.forms.end());
  site_memo_armed_ = static_cast<int64_t>(armed_.size());
  // the memo is this frame's reads (the caller says whether those are the
  // cache's)
  ++site_reads_version_;
  site_memo_cache_ = false;
  return true;
}

bool HostTraceVariant::evaluate_sites_slow(Frame& frame) const {
  frame.site_entries.resize_for_overwrite(sites_.size());
  c10::SmallVector<int64_t, 32> key;
  for (size_t i = 0; i < sites_.size(); ++i) {
    const Site& s = sites_[i];
    if (!s.predicates.empty()) {
      const int64_t row = select(s, frame.values.data());
      if (row < 0) {
        frame.keyed_miss = true;
        return false;
      }
      frame.site_entries[i] = row;
      continue;
    }
    key.resize_for_overwrite(s.key_rows.size());
    for (size_t j = 0; j < key.size(); ++j) {
      key[j] = frame.values[s.key_rows[j]];
    }
    int64_t found = find(s, key.data());
    if (found < 0) {
      // the tables grow at a miss (frame.site_entries holds row indices, not entries)
      found = const_cast<HostTraceVariant*>(this)->fill_from_table(i, key.data());
    }
    if (found < 0) {
      frame.keyed_miss = true;
      return false;
    }
    if (s.rows[found] < 0) {
      return false;
    }
    frame.site_entries[i] = s.rows[found];
  }
  if (!armed_.empty()) {
    frame.forms.resize_for_overwrite(segments_.size());
    c10::SmallVector<int32_t, 8> arms;
    for (size_t g : armed_) {
      const int64_t f = form_of(segments_[g], frame.site_entries.data(), arms);
      if (f < 0) {
        frame.keyed_miss = true;
        return false;
      }
      frame.forms[g] = static_cast<int32_t>(f);
    }
  }
  return true;
}

int64_t HostTraceVariant::form_of(
    const Segment& g,
    const int64_t* site_entries,
    c10::SmallVector<int32_t, 8>& arms) const {
  arms.clear();
  for (size_t i : g.sites) {
    const int64_t row = site_entries[i];
    arms.push_back(row == 0 ? 0 : sites_[i].entries[row - 1]->arm);
  }
  for (size_t f = 0; f < g.forms.size(); ++f) {
    if (std::equal(arms.begin(), arms.end(), g.forms[f].begin(), g.forms[f].end())) {
      return static_cast<int64_t>(f);
    }
  }
  return -1;
}

bool HostTraceVariant::overlaps_py(py::handle args) const {
  TORCH_CHECK_TYPE(PyTuple_Check(args.ptr()), "args must be a tuple");
  Frame frame;
  evaluate_rows(
      &PyTuple_GET_ITEM(args.ptr(), 0),
      static_cast<size_t>(PyTuple_GET_SIZE(args.ptr())),
      frame);
  return frame.overlapped;
}

py::object HostTraceVariant::evaluate_py(py::handle args) const {
  TORCH_CHECK_TYPE(PyTuple_Check(args.ptr()), "args must be a tuple");
  Frame frame;
  if (!evaluate_rows(
          &PyTuple_GET_ITEM(args.ptr(), 0),
          static_cast<size_t>(PyTuple_GET_SIZE(args.ptr())),
          frame)) {
    return py::none();
  }
  py::list missing;
  py::list unformed;
  py::list unselected;
  std::vector<int64_t> rows(sites_.size());
  for (size_t i = 0; i < sites_.size(); ++i) {
    const Site& s = sites_[i];
    if (!s.predicates.empty()) {
      rows[i] = select(s, frame.values.data());
      if (rows[i] < 0) {
        unselected.append(i);
      }
      continue;
    }
    std::vector<int64_t> key;
    for (int64_t row : s.key_rows) {
      key.push_back(frame.values[row]);
    }
    int64_t found = find(s, key.data());
    if (found < 0) {
      found = const_cast<HostTraceVariant*>(this)->fill_from_table(i, key.data());
    }
    if (found < 0) {
      missing.append(py::make_tuple(i, key));
    } else if (s.rows[found] < 0) {
      return py::none();
    } else {
      rows[i] = s.rows[found];
    }
  }
  if (missing.empty() && unselected.empty()) {
    c10::SmallVector<int32_t, 8> arms;
    for (size_t g : armed_) {
      if (form_of(segments_[g], rows.data(), arms) < 0) {
        unformed.append(py::make_tuple(g, std::vector<int32_t>(arms.begin(), arms.end())));
      }
    }
  }
  std::vector<int64_t> values(frame.values.begin(), frame.values.end());
  return py::make_tuple(values, missing, unformed, unselected);
}

void HostTraceVariant::add_entry(
    size_t site,
    int64_t predicate,
    py::handle nodes,
    int32_t arm,
    const std::vector<int64_t>& temporaries,
    const std::vector<int64_t>& drops,
    int64_t alloc_at,
    int64_t free_at,
    int64_t made_seq,
    int64_t last_seq) {
  TORCH_CHECK_VALUE(
      site < sites_.size() && !sites_[site].predicates.empty(),
      "no selector ",
      site);
  Site& s = sites_[site];
  auto t = nodes.cast<py::tuple>();
  const bool piece = arm != 0;
  TORCH_CHECK_VALUE(
      arm >= 0 &&
          (piece ? t.size() > 0 && !s.records.empty()
                 : t.size() == s.records.size() && temporaries.empty() &&
                 drops.empty()),
      "an entry of ",
      t.size(),
      " nodes");
  auto e = std::make_unique<Entry>();
  e->arm = arm;
  for (int64_t row : temporaries) {
    e->temporaries.push_back(check_row(row));
  }
  if (piece) {
    // its places in its run's order: of its events, or of its tensors
    const auto step = std::find_if(steps_.begin(), steps_.end(), [&](const Step& x) {
      return x.segment == static_cast<int64_t>(s.segment);
    });
    TORCH_CHECK_VALUE(step != steps_.end(), "a section outside the runs");
    const StepMemory& m = memory_[step - steps_.begin()];
    const size_t n = m.order.empty() ? m.tensors.size() : m.order.size();
    TORCH_CHECK_VALUE(
        alloc_at >= 0 && alloc_at <= free_at && static_cast<size_t>(free_at) <= n,
        "a section's places ",
        alloc_at,
        ", ",
        free_at);
    e->alloc_at = static_cast<size_t>(alloc_at);
    e->free_at = static_cast<size_t>(free_at);
    TORCH_CHECK_VALUE(
        last_seq == -1 || (m.order.empty() && made_seq < last_seq),
        "a section's seqs ",
        made_seq,
        ", ",
        last_seq);
    e->made_seq = made_seq;
    e->last_seq = last_seq;
  }
  for (int64_t k : drops) {
    TORCH_CHECK_VALUE(
        k >= 0 && static_cast<size_t>(k) < n_alloc_ &&
            (allocations_[k].owner < 0 ||
             allocations_[k].owner == static_cast<int64_t>(site)) &&
            allocations_[k].site < 0 && !allocations_[k].arena,
        "a section's dropped allocation ",
        k);
    e->drops.push_back(k);
  }
  // the temporaries' bases, which its nodes' fields read
  e->first_base = base_count_;
  base_count_ += temporaries.size();
  auto restore = c10::make_scope_exit([&] {
    if (e) {
      base_count_ = e->first_base;
    }
  });
  std::vector<Kind> kinds(t.size());
  e->kernels.resize(t.size());
  e->memsets.resize(t.size());
  e->memcpys.resize(t.size());
  for (size_t i = 0; i < t.size(); ++i) {
    auto n = t[i].cast<py::tuple>();
    const auto kind = kinds[i] = static_cast<Kind>(n[0].cast<uint8_t>());
    TORCH_CHECK_VALUE(piece || kind != Kind::Memcpy, "an entry's memcpy");
    TORCH_CHECK_VALUE(
        piece || kind == records_[s.records[i]].kind,
        "an entry's node of another kind");
    if (kind == Kind::Memcpy) {
      // a section's memcpy: device to device
      const auto copy = static_cast<cudaMemcpyKind>(n[4].cast<int>());
      TORCH_CHECK_VALUE(copy == cudaMemcpyDeviceToDevice, "a section's host memcpy");
      e->memcpys[i] = {parse_source(n[1]), parse_source(n[2]), check_row(n[3].cast<int64_t>()), copy, -1};
      continue;
    }
    if (kind == Kind::Memset) {
      MemsetRow& m = e->memsets[i];
      m.dst = parse_source(n[1]);
      m.value = n[2].cast<unsigned int>();
      m.element_size = check_element_size(n[3]);
      for (size_t k = 0; k < 3; ++k) {
        m.shape_rows[k] = check_row(n[4 + k].cast<int64_t>());
      }
      continue;
    }
    KernelRow& k = e->kernels[i];
    k.dim_rows = set_launch(k, n[1], n[2], n[3], n[4]);
    for (int64_t row : k.dim_rows) {
      check_row(row);
    }
    append_images(k, n[5], e->image);
    k.first_field = e->fields.size();
    for (py::handle x : n[6].cast<py::tuple>()) {
      e->fields.push_back(parse_field(x, k));
    }
    k.field_count = e->fields.size() - k.first_field;
    TORCH_CHECK_VALUE(
        entry_descriptors_enabled || n[7].cast<py::tuple>().empty(),
        "an entry's TMA descriptor");
    parse_descriptors(k, n[7], e->descriptors);
    parse_rng(k, segments_[s.segment], n[8], n[9]);
    TORCH_CHECK_VALUE(
        n[10].cast<py::tuple>().empty(), "an entry's CPU scalar");
  }
  e->held = e->image;
  for (size_t i = 0; i < t.size(); ++i) {
    if (kinds[i] == Kind::Kernel) {
      bind_row(
          e->kernels[i], e->image.data(), e->held.data(), e->fields.data());
      e->kernels[i].descriptors =
          e->descriptors.data() + e->kernels[i].first_descriptor;
    }
  }
  if (piece) {
    if (std::find(armed_.begin(), armed_.end(), s.segment) == armed_.end()) {
      armed_.push_back(s.segment);
    }
    auto& sections = segments_[s.segment].sections;
    if (!temporaries.empty() &&
        std::find(sections.begin(), sections.end(), site) == sections.end()) {
      sections.push_back(site);
    }
  }
  for (int64_t k : drops) {
    allocations_[k].owner = static_cast<int64_t>(site);
  }
  s.predicates.push_back(check_row(predicate));
  note_reads(s, *e);
  s.entries.push_back(std::move(e));
  rows_added();
}

void HostTraceVariant::rows_added() {
  users_kept_ = users_kept_ || !users_stale_;
  users_stale_ = true;
}

bool HostTraceVariant::note_reads(Site& s, const Entry& e) {
  bool added = false;
  if (s.reads.size() < e.kernels.size()) {
    s.reads.resize(e.kernels.size());
    added = true;
  }
  for (size_t i = 0; i < e.kernels.size(); ++i) {
    const KernelRow& k = e.kernels[i];
    Site::Reads& r = s.reads[i];
    const bool always = k.rng_increment != 0 || !k.rng.empty() || !k.cpu_scalars.empty();
    added |= always && !r.always;
    r.always |= always;
    for (size_t j = 0; j < k.field_count; ++j) {
      const std::pair<int64_t, int64_t> f{k.fields[j].row, k.fields[j].base};
      if (std::find(r.fields.begin(), r.fields.end(), f) == r.fields.end()) {
        r.fields.push_back(f);
        added = true;
      }
    }
    for (int64_t row : k.dim_rows) {
      if (!k.constant_dims &&
          std::find(r.dim_rows.begin(), r.dim_rows.end(), row) ==
              r.dim_rows.end()) {
        r.dim_rows.push_back(row);
        added = true;
      }
    }
  }
  return added;
}

bool HostTraceVariant::pieces_cover(const Site& s, const Entry& e) {
  auto has = [](const std::vector<int64_t>& v, int64_t x) {
    return std::binary_search(v.begin(), v.end(), x);
  };
  auto field = [&](const Field& f) {
    return has(s.piece_rows, f.row) && (f.base < 0 || has(s.piece_bases, f.base));
  };
  for (const KernelRow& k : e.kernels) {
    if (!s.piece_always &&
        (k.rng_increment != 0 || !k.rng.empty() || !k.cpu_scalars.empty())) {
      return false;
    }
    for (size_t j = 0; j < k.field_count; ++j) {
      if (!field(k.fields[j])) {
        return false;
      }
    }
    if (!k.constant_dims &&
        !std::all_of(k.dim_rows.begin(), k.dim_rows.end(), [&](int64_t r) { return has(s.piece_rows, r); })) {
      return false;
    }
  }
  for (const MemsetRow& m : e.memsets) {
    if (!field(m.dst) ||
        (!m.constant_shape &&
         !std::all_of(m.shape_rows.begin(), m.shape_rows.end(), [&](int64_t r) { return has(s.piece_rows, r); }))) {
      return false;
    }
  }
  for (const MemcpyRow& m : e.memcpys) {
    if (!field(m.dst) || !field(m.src) || !has(s.piece_rows, m.bytes_row)) {
      return false;
    }
  }
  return true;
}

void HostTraceVariant::row_added(Site& s, const Entry& e) {
  // a row that reads only what the site's rows read already leaves the users
  // index as it is (index_users): the call that selects it marks its records
  // (seen_sites_); else the index is rebuilt, keeping what the records hold
  const bool read = note_reads(s, e);
  if (read || users_stale_ || !pieces_cover(s, e)) {
    rows_added();
  } else {
    ++rows_unindexed;
  }
}

void HostTraceVariant::set_program(const HostTraceProgram& program) {
  TORCH_CHECK_VALUE(
      program.num_rows() >= program_.num_rows(),
      "a program of fewer rows than the variant's");
  program_ = program;
  set_static_prefix(static_prefix_);
}

void HostTraceVariant::add_row(
    size_t site,
    const std::vector<int64_t>& key,
    py::handle nodes,
    int32_t arm,
    bool piece,
    const std::vector<int64_t>& scratch) {
  TORCH_CHECK_VALUE(site < sites_.size(), "no site ", site);
  Site& s = sites_[site];
  TORCH_CHECK_VALUE(
      key.size() == s.key_rows.size(), "a key of ", key.size(), " values");
  if (find(s, key.data()) >= 0) {
    return;
  }
  if (nodes.is_none()) {
    insert(s, key.data(), -1);
    return;
  }
  TORCH_CHECK_VALUE(arm >= 0, "no arm ", arm);
  TORCH_CHECK_VALUE(
      scratch.size() == s.scratch.size() &&
          std::all_of(
              scratch.begin(), scratch.end(), [](int64_t n) { return n >= 0; }),
      "a row's scratch bytes");
  auto t = nodes.cast<py::tuple>();
  TORCH_CHECK_VALUE(
      piece ? arm != 0 && t.size() > 0 : t.size() == s.records.size(),
      "a row of ",
      t.size(),
      " nodes");
  std::vector<Kind> kinds(t.size());
  auto e = std::make_unique<Entry>();
  e->arm = arm;
  e->scratch = scratch;
  e->kernels.resize(t.size());
  e->memsets.resize(t.size());
  for (size_t i = 0; i < t.size(); ++i) {
    auto n = t[i].cast<py::tuple>();
    const auto kind = kinds[i] = static_cast<Kind>(n[0].cast<uint8_t>());
    TORCH_CHECK_VALUE(kind != Kind::Memcpy, "a keyed row's memcpy");
    TORCH_CHECK_VALUE(
        piece || kind == records_[s.records[i]].kind,
        "a row's node of another kind");
    if (kind == Kind::Memset) {
      MemsetRow& m = e->memsets[i];
      m.dst = parse_source(n[1]);
      m.value = n[2].cast<unsigned int>();
      m.element_size = check_element_size(n[3]);
      m.constant_shape = true;
      for (size_t k = 0; k < 3; ++k) {
        m.shape[k] = n[4 + k].cast<int64_t>();
      }
      continue;
    }
    KernelRow& k = e->kernels[i];
    k.dims = set_launch(k, n[1], n[2], n[3], n[4]);
    k.constant_dims = true;
    append_images(k, n[5], e->image);
    parse_rng(k, segments_[s.segment], n[7], n[8]);
    k.first_field = e->fields.size();
    for (py::handle x : n[6].cast<py::tuple>()) {
      Field f = parse_field(x, k);
      TORCH_CHECK_VALUE(f.width != 0, "a keyed row's field of 0 bytes");
      e->fields.push_back(f);
    }
    k.field_count = e->fields.size() - k.first_field;
  }
  e->held = e->image;
  for (size_t i = 0; i < t.size(); ++i) {
    if (kinds[i] == Kind::Kernel) {
      bind_row(
          e->kernels[i], e->image.data(), e->held.data(), e->fields.data());
    }
  }
  row_added(s, *e);
  s.entries.push_back(std::move(e));
  insert(s, key.data(), static_cast<int64_t>(s.entries.size()));
  if (arm != 0 &&
      std::find(armed_.begin(), armed_.end(), s.segment) == armed_.end()) {
    armed_.push_back(s.segment);
  }
}

void HostTraceVariant::add_form(
    size_t segment,
    const std::vector<int32_t>& arms,
    uintptr_t exec,
    uintptr_t graph,
    py::handle pieces) {
  TORCH_CHECK_VALUE(segment < segments_.size(), "no segment ", segment);
  Segment& g = segments_[segment];
  TORCH_CHECK_VALUE(
      arms.size() == g.sites.size() &&
          std::find(g.forms.begin(), g.forms.end(), arms) == g.forms.end(),
      "a segment's form");
  Segment::Clone c{reinterpret_cast<CUgraph>(graph), {}, {}};
  std::vector<bool> replaced(sites_.size());
  for (py::handle p : pieces) {
    auto t = p.cast<py::tuple>();
    const auto site = t[0].cast<size_t>();
    TORCH_CHECK_VALUE(
        site < sites_.size() && sites_[site].segment == segment,
        "no site ",
        site,
        " in the segment");
    replaced[site] = true;
    c.groups.push_back({site, c.pieces.size(), c.pieces.size(), {}, {}});
    size_t pos = 0;
    for (py::handle h : t[1].cast<py::tuple>()) {
      auto n = h.cast<py::tuple>();
      Record r{};
      r.kind = static_cast<Kind>(n[0].cast<uint8_t>());
      TORCH_CHECK_VALUE(
          r.kind != Kind::Memcpy || !sites_[site].predicates.empty(),
          "a keyed row's memcpy");
      r.node = reinterpret_cast<CUgraphNode>(n[1].cast<uintptr_t>());
      r.site = static_cast<int64_t>(site);
      r.site_pos = pos++;
      c.pieces.push_back(std::move(r));
    }
    c.groups.back().stop = c.pieces.size();
  }
  for (size_t i = g.first; i < g.stop; ++i) {
    const Record& r = records_[i];
    cudaGraphNode_t node = nullptr;
    if (r.site < 0 || !replaced[r.site]) {
      C10_CUDA_CHECK(cudaGraphNodeFindInClone(
          &node,
          reinterpret_cast<cudaGraphNode_t>(r.node),
          reinterpret_cast<cudaGraph_t>(c.graph)));
    }
    c.nodes.push_back(reinterpret_cast<CUgraphNode>(node));
  }
  g.forms.push_back(arms);
  g.execs.push_back(reinterpret_cast<CUgraphExec>(exec));
  g.clones.push_back(std::move(c));
}

HostTraceVariant::~HostTraceVariant() {
  for (Segment& g : segments_) {
    for (const Segment::Split& sp : g.splits) {
      for (CUgraphExec e : sp.execs) {
        C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(e)));
      }
      for (CUgraph h : sp.graphs) {
        C10_CUDA_CHECK_WARN(cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(h)));
      }
    }
    for (size_t f = 1; f < g.execs.size(); ++f) {
      C10_CUDA_CHECK_WARN(
          cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(g.execs[f])));
    }
    for (const Segment::Clone& c : g.clones) {
      C10_CUDA_CHECK_WARN(
          cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(c.graph)));
    }
  }
}

} // namespace torch::cuda::host_trace
#endif
