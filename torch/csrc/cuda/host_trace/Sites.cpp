#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM)
#include <torch/csrc/autograd/python_variable.h>

#include <c10/util/SmallVector.h>

#include <algorithm>
#include <utility>

namespace torch::cuda::host_trace {

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
  c10::SmallVector<int64_t, 64> leaves(program_.num_leaves());
  if (!program_.bind(args, count, leaves.data())) {
    return false;
  }
  frame.args = args;
  frame.count = count;
  frame.values.resize_for_overwrite(program_.num_rows());
  if (program_.evaluate(leaves.data(), frame.values.data()) !=
          HostTraceProgram::Status::Success ||
      frame.values[valid_] != 1) {
    return false;
  }
  frame.overlapped = !disjoint(args, count);
  return !frame.overlapped;
}

// after the rows, which hold only for nonempty tensor arguments
bool HostTraceVariant::disjoint(PyObject* const* args, size_t count) const {
  if (argument_pairs_.empty()) {
    return true;
  }
  // each argument's first and last byte
  c10::SmallVector<std::pair<uintptr_t, uintptr_t>, 8> extents;
  for (size_t i : pair_arguments_) {
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
    extents.emplace_back(first, first + (span + 1) * t.itemsize() - 1);
  }
  for (const ArgumentPair& p : argument_pairs_) {
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
        n[7].cast<py::tuple>().empty(), "an entry's TMA descriptor");
    parse_rng(k, segments_[s.segment], n[8], n[9]);
    TORCH_CHECK_VALUE(
        n[10].cast<py::tuple>().empty(), "an entry's CPU scalar");
  }
  e->held = e->image;
  for (size_t i = 0; i < t.size(); ++i) {
    if (records_[s.records[i]].kind == Kind::Kernel) {
      bind_row(
          e->kernels[i], e->image.data(), e->held.data(), e->fields.data());
    }
  }
  s.predicates.push_back(check_row(predicate));
  s.entries.push_back(std::move(e));
}

void HostTraceVariant::set_program(const HostTraceProgram& program) {
  TORCH_CHECK_VALUE(
      program.num_rows() >= program_.num_rows(),
      "a program of fewer rows than the variant's");
  program_ = program;
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
