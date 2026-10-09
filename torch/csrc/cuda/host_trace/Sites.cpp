#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM)
#include <torch/csrc/autograd/python_variable.h>

#include <c10/util/SmallVector.h>

#include <algorithm>
#include <utility>

namespace torch::cuda::host_trace {

int64_t delta_skips = 0;
int64_t delta_partials = 0;
int64_t delta_fulls = 0;
int64_t delta_rows_evaluated = 0;
bool delta_enabled = true;
bool variant_order_enabled = true;
bool placement_reuse_enabled = true;
bool form_held_enabled = false;
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

} // namespace

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
  frame.evaluated = program_.evaluate(leaves.data(), frame.values.data()) ==
      HostTraceProgram::Status::Success;
  if (!frame.evaluated || frame.values[valid_] != 1) {
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
    }
    return hit;
  }
  std::swap(frame.values, cached_values_);
  cache_busy_ = true;
  frame.cached = this;
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
    c10::SmallVector<uint64_t, 4> changed((n + 63) / 64, 0);
    bool any = false;
    for (size_t j = 0; j < n; ++j) {
      if (leaves[dynamic_leaves_[j]] != delta_leaves_[j]) {
        changed[j / 64] |= uint64_t{1} << (j % 64);
        any = true;
      }
    }
    if (!any) {
      valid_rows = nullptr;
      ++delta_skips;
    } else {
      const DeltaRows& d = delta_rows(changed);
      valid_rows = &d.valid_rows;
      other_rows = &d.other_rows;
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
  const bool resolved = !frame.overlapped && evaluate_sites(frame);
  phase_add(3, pt);
  return resolved;
}

void HostTraceVariant::release(Frame& frame) {
  if (frame.cached == this) {
    std::swap(frame.values, cached_values_);
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
  delta_valid_ = delta_marked_ = false;
}

const HostTraceVariant::DeltaRows& HostTraceVariant::delta_rows(
    c10::ArrayRef<uint64_t> changed) {
  for (const DeltaRows& d : delta_memo_) {
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
  delta_valid_ = delta_marked_ = false;
  delta_memo_.clear();
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
    const int64_t found = find(s, key.data());
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
    const int64_t found = find(s, key.data());
    if (found < 0) {
      missing.append(py::make_tuple(i, key));
    } else if (s.rows[found] < 0) {
      return py::none();
    } else {
      rows[i] = s.rows[found];
    }
  }
  if (missing.empty()) {
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
    py::handle nodes) {
  TORCH_CHECK_VALUE(
      site < sites_.size() && !sites_[site].predicates.empty(),
      "no selector ",
      site);
  Site& s = sites_[site];
  auto t = nodes.cast<py::tuple>();
  TORCH_CHECK_VALUE(
      t.size() == s.records.size(), "an entry of ", t.size(), " nodes");
  auto e = std::make_unique<Entry>();
  e->kernels.resize(t.size());
  e->memsets.resize(t.size());
  for (size_t i = 0; i < t.size(); ++i) {
    auto n = t[i].cast<py::tuple>();
    const auto kind = static_cast<Kind>(n[0].cast<uint8_t>());
    TORCH_CHECK_VALUE(kind != Kind::Memcpy, "an entry's memcpy");
    TORCH_CHECK_VALUE(
        kind == records_[s.records[i]].kind, "an entry's node of another kind");
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
    if (records_[s.records[i]].kind == Kind::Kernel) {
      bind_row(
          e->kernels[i], e->image.data(), e->held.data(), e->fields.data());
      e->kernels[i].descriptors =
          e->descriptors.data() + e->kernels[i].first_descriptor;
    }
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

void HostTraceVariant::note_reads(Site& s, const Entry& e) {
  if (s.reads.size() < e.kernels.size()) {
    s.reads.resize(e.kernels.size());
  }
  for (size_t i = 0; i < e.kernels.size(); ++i) {
    const KernelRow& k = e.kernels[i];
    Site::Reads& r = s.reads[i];
    r.always |= k.rng_increment != 0 || !k.rng.empty() || !k.cpu_scalars.empty();
    for (size_t j = 0; j < k.field_count; ++j) {
      const std::pair<int64_t, int64_t> f{k.fields[j].row, k.fields[j].base};
      if (std::find(r.fields.begin(), r.fields.end(), f) == r.fields.end()) {
        r.fields.push_back(f);
      }
    }
    for (int64_t row : k.dim_rows) {
      if (!k.constant_dims &&
          std::find(r.dim_rows.begin(), r.dim_rows.end(), row) ==
              r.dim_rows.end()) {
        r.dim_rows.push_back(row);
      }
    }
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
  note_reads(s, *e);
  s.entries.push_back(std::move(e));
  insert(s, key.data(), static_cast<int64_t>(s.entries.size()));
  rows_added();
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
    size_t pos = 0;
    for (py::handle h : t[1].cast<py::tuple>()) {
      auto n = h.cast<py::tuple>();
      Record r{};
      r.kind = static_cast<Kind>(n[0].cast<uint8_t>());
      TORCH_CHECK_VALUE(r.kind != Kind::Memcpy, "a keyed row's memcpy");
      r.node = reinterpret_cast<CUgraphNode>(n[1].cast<uintptr_t>());
      r.site = static_cast<int64_t>(site);
      r.site_pos = pos++;
      c.pieces.push_back(std::move(r));
    }
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
