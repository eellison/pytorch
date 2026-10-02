#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM)
#include <c10/cuda/driver_api.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <algorithm>
#include <tuple>
#include <utility>

namespace torch::cuda::host_trace {

namespace {

template <typename T>
std::vector<T> ints(py::handle h) {
  return h.cast<std::vector<T>>();
}

} // namespace

HostTraceVariant::HostTraceVariant(py::handle spec)
    : program_(spec.attr("program").cast<const HostTraceProgram&>()),
      valid_(spec.attr("valid").cast<size_t>()),
      base_count_(spec.attr("base_count").cast<size_t>()) {
  const size_t rows = program_.num_rows();
  TORCH_CHECK_VALUE(valid_ < rows, "no row ", valid_);
  auto check_rows = [&](std::vector<int64_t> r) {
    for (int64_t row : r) {
      check_row(row);
    }
    return r;
  };
  for (py::handle a : spec.attr("allocations").cast<py::tuple>()) {
    auto t = a.cast<py::tuple>();
    TORCH_CHECK_TYPE(THPDtype_Check(t[2].ptr()), "an allocation's dtype");
    allocations_.push_back(
        {check_rows(ints<int64_t>(t[0])),
         check_rows(ints<int64_t>(t[1])),
         reinterpret_cast<THPDtype*>(t[2].ptr())->scalar_type,
         check_row(t[3].cast<int64_t>())});
  }
  n_alloc_ = allocations_.size();
  TORCH_CHECK_VALUE(n_alloc_ <= base_count_, "more allocations than bases");
  for (py::handle s : spec.attr("segments").cast<py::tuple>()) {
    auto t = s.cast<py::tuple>();
    py::object graph = py::reinterpret_borrow<py::object>(t[0]);
    auto native = graph.cast<std::shared_ptr<at::cuda::CUDAGraph>>();
    auto exec = reinterpret_cast<CUgraphExec>(native->raw_cuda_graph_exec());
    const size_t first = segments_.empty() ? 0 : segments_.back().stop;
    TORCH_CHECK_VALUE(t.size() == 5, "a segment of ", t.size(), " items");
    segments_.push_back(
        {graph,
         graph.attr("__dict__"),
         std::move(native),
         {exec},
         {{}},
         0,
         first,
         first + t[1].cast<size_t>(),
         {},
         {}});
    if (!t[2].is_none()) {
      segments_.back().generator = THPGenerator_Unwrap(t[2].ptr());
      segments_.back().philox_seed = t[3].cast<int64_t>();
      segments_.back().philox_offset = t[4].cast<int64_t>();
    }
  }
  for (py::handle l : spec.attr("launches").cast<py::tuple>()) {
    auto t = l.cast<py::tuple>();
    Record r{};
    r.kind = static_cast<Kind>(t[0].cast<uint8_t>());
    r.node = reinterpret_cast<CUgraphNode>(t[1].cast<uintptr_t>());
    r.held = true;
    r.site = -1;
    if (r.kind == Kind::Memset) {
      MemsetRow& m = r.memset;
      m.dst = parse_source(t[2]);
      m.value = t[3].cast<unsigned int>();
      m.element_size = check_element_size(t[4]);
      for (size_t k = 0; k < 3; ++k) {
        m.shape_rows[k] = check_row(t[5 + k].cast<int64_t>());
      }
      records_.push_back(std::move(r));
      continue;
    }
    if (r.kind == Kind::Memcpy) {
      r.copy = {parse_source(t[2]), parse_source(t[3]), check_row(t[4].cast<int64_t>())};
      records_.push_back(std::move(r));
      continue;
    }
    TORCH_CHECK_VALUE(r.kind == Kind::Kernel, "a record kind");
    TORCH_CHECK_NOT_IMPLEMENTED(
        c10::cuda::DriverAPI::get()->cuGraphExecKernelNodeSetParams_,
        "host_trace: the driver lacks cuGraphExecKernelNodeSetParams");
    KernelRow& k = r.kernel;
    k.dim_rows = set_launch(k, t[2], t[3], t[4], t[5]);
    for (int64_t row : k.dim_rows) {
      check_row(row);
    }
    append_images(k, t[6], held_bytes_);
    k.first_field = fields_.size();
    for (py::handle x : t[7].cast<py::tuple>()) {
      fields_.push_back(parse_field(x, k));
    }
    k.field_count = fields_.size() - k.first_field;
    k.first_descriptor = descriptors_.size();
    auto descriptors = t[8].cast<py::tuple>();
    if (!descriptors.empty()) {
      auto* driver = c10::cuda::DriverAPI::get();
      TORCH_CHECK_NOT_IMPLEMENTED(
          driver->cuTensorMapEncodeTiled_ &&
              driver->cuTensorMapReplaceAddress_,
          "host_trace: the driver lacks the TMA descriptor entries");
    }
    for (py::handle x : descriptors) {
      auto d = x.cast<py::tuple>();
      Descriptor desc{};
      desc.param = d[0].cast<uint32_t>();
      desc.first = d[1].cast<size_t>();
      desc.dtype = static_cast<CUtensorMapDataType>(d[2].cast<int>());
      desc.box = d[3].cast<std::vector<cuuint32_t>>();
      desc.swizzle = static_cast<CUtensorMapSwizzle>(d[4].cast<int>());
      const size_t rank = desc.box.size();
      TORCH_CHECK_VALUE(
          rank >= 1 && rank <= 5 && desc.first + 2 * rank <= k.field_count,
          "a descriptor's fields");
      TORCH_CHECK_VALUE(
          desc.param < k.param_sizes.size() &&
              k.param_sizes[desc.param] == sizeof(CUtensorMap),
          "a descriptor's parameter");
      descriptors_.push_back(std::move(desc));
    }
    k.descriptor_count = descriptors_.size() - k.first_descriptor;
    auto g = std::find_if(segments_.begin(), segments_.end(), [&](auto& x) {
      return records_.size() >= x.first && records_.size() < x.stop;
    });
    TORCH_CHECK_VALUE(g != segments_.end(), "a launch outside the segments");
    parse_rng(k, *g, t[9], t[10]);
    for (py::handle x : t[11].cast<py::tuple>()) {
      auto [param, offset, cls, source] =
          x.cast<std::tuple<uint32_t, uint32_t, std::string, at::Tensor>>();
      TORCH_CHECK_VALUE(
          cls.size() == 1 && source.device().is_cpu() && source.dim() == 0,
          "a CPU scalar field");
      const char c = cls[0];
      const bool reciprocal = c >= '0' && c <= '9';
      TORCH_CHECK_VALUE(reciprocal || (c >= 'A' && c <= 'Z'), "a CPU scalar class ", cls);
      const auto type = static_cast<c10::ScalarType>(reciprocal ? c - '0' : c - 'A');
      TORCH_CHECK_VALUE(
          param < k.param_sizes.size() &&
              offset + c10::elementSize(type) <= k.param_sizes[param],
          "a CPU scalar field outside its parameter");
      k.cpu_scalars.push_back({param, offset, c, std::move(source)});
    }
    records_.push_back(std::move(r));
  }
  TORCH_CHECK_VALUE(
      (segments_.empty() ? 0 : segments_.back().stop) == records_.size(),
      "the runs hold other launches");
  storage_ = held_bytes_;
  // pointers into the vectors only now that they no longer move
  for (Record& r : records_) {
    if (r.kind == Kind::Kernel) {
      KernelRow& k = r.kernel;
      bind_row(k, storage_.data(), held_bytes_.data(), fields_.data());
      k.descriptors = descriptors_.data() + k.first_descriptor;
    }
  }
  // what the exec nodes hold: the capture's parameters, at the traced call's
  // rows and bases
  auto [trace_rows, trace_bases] =
      spec.attr("traced")
          .cast<std::pair<std::vector<int64_t>, std::vector<int64_t>>>();
  TORCH_CHECK_VALUE(
      trace_rows.size() == rows && trace_bases.size() == base_count_,
      "the traced call's rows and bases");
  for (Record& r : records_) {
    if (r.kind == Kind::Memset) {
      r.held_memset =
          memset_shape(r.memset, trace_rows.data(), trace_bases.data());
      continue;
    }
    if (r.kind == Kind::Memcpy) {
      r.held_copy = memcpy_shape(r.copy, trace_rows.data(), trace_bases.data());
      continue;
    }
    const KernelRow& k = r.kernel;
    pack(k, trace_rows.data(), trace_bases.data(), k.held);
    for (size_t i = 0; i < k.dim_rows.size(); ++i) {
      r.held_dims[i] = trace_rows[k.dim_rows[i]];
    }
  }
  // each keyed site's table starts with the traced key, its records' own
  for (py::handle x : spec.attr("sites").cast<py::tuple>()) {
    auto t = x.cast<py::tuple>();
    Site site;
    site.key_rows = check_rows(ints<int64_t>(t[0]));
    site.records = ints<size_t>(t[1]);
    const bool selector = t.size() == 4;
    if (selector) {
      site.predicates.push_back(check_row(t[3].cast<int64_t>()));
    }
    TORCH_CHECK_VALUE(selector || !site.records.empty(), "a site of no nodes");
    site.segment = segments_.size();
    if (!site.records.empty()) {
      auto g = std::find_if(segments_.begin(), segments_.end(), [&](auto& x) {
        return site.records.front() >= x.first &&
            site.records.front() < x.stop;
      });
      TORCH_CHECK_VALUE(g != segments_.end(), "a site outside the segments");
      site.segment = g - segments_.begin();
      if (!selector) {
        g->sites.push_back(sites_.size());
      }
    }
    for (size_t pos = 0; pos < site.records.size(); ++pos) {
      const size_t i = site.records[pos];
      TORCH_CHECK_VALUE(
          i < records_.size() && records_[i].site < 0, "a site's record ", i);
      records_[i].site = static_cast<int64_t>(sites_.size());
      records_[i].site_pos = pos;
    }
    site.scratch = ints<int64_t>(t[2]);
    for (size_t j = 0; j < site.scratch.size(); ++j) {
      const int64_t k = site.scratch[j];
      TORCH_CHECK_VALUE(
          k >= 0 && static_cast<size_t>(k) < n_alloc_ &&
              allocations_[k].site < 0 && allocations_[k].sizes.size() == 1,
          "a site's scratch buffer ",
          k);
      allocations_[k].site = static_cast<int64_t>(sites_.size());
      allocations_[k].scratch = j;
    }
    if (!selector) {
      std::vector<int64_t> key;
      for (int64_t row : site.key_rows) {
        key.push_back(trace_rows[row]);
      }
      insert(site, key.data(), 0);
    }
    sites_.push_back(std::move(site));
  }
  for (Segment& g : segments_) {
    g.forms[0].assign(g.sites.size(), 0);
  }
  auto base_ref = [&](py::handle kind, py::handle index) -> BaseRef {
    const bool argument = kind.cast<bool>();
    const auto i = index.cast<int64_t>();
    TORCH_CHECK_VALUE(argument ? i >= 0 : check_base(i) >= 0, "no base ", i);
    return {argument, i};
  };
  for (py::handle v : spec.attr("views").cast<py::tuple>()) {
    auto t = v.cast<py::tuple>();
    std::optional<at::ScalarType> dtype;
    if (!t[5].is_none()) {
      TORCH_CHECK_TYPE(THPDtype_Check(t[5].ptr()), "a view's dtype");
      dtype = reinterpret_cast<THPDtype*>(t[5].ptr())->scalar_type;
    }
    views_.push_back(
        {base_ref(t[0], t[1]),
         check_rows(ints<int64_t>(t[2])),
         check_rows(ints<int64_t>(t[3])),
         check_row(t[4].cast<int64_t>()),
         dtype});
  }
  auto check_view = [this](int64_t i) {
    TORCH_CHECK_VALUE(
        i >= 0 && static_cast<size_t>(i) < views_.size(), "no view ", i);
    return i;
  };
  // A step's boxed form: false when the dispatcher cannot express it, which
  // leaves it to its Python run
  auto boxed_step = [&](EagerStep& step, const py::tuple& t) -> bool {
    auto op = c10::Dispatcher::singleton().findSchema(
        {t[0].cast<std::string>(), t[1].cast<std::string>()});
    if (!op) {
      return false;
    }
    const c10::FunctionSchema& schema = op->schema();
    auto unwrap = [](c10::TypePtr type) {
      if (auto optional = type->cast<c10::OptionalType>()) {
        return optional->getElementType();
      }
      return type;
    };
    for (const c10::Argument& r : schema.returns()) {
      c10::TypePtr type = r.type();
      if (auto list = type->cast<c10::ListType>()) {
        type = list->getElementType();
      }
      if (type->kind() != c10::TypeKind::TensorType) {
        return false;
      }
    }
    auto args = t[2].cast<py::tuple>();
    if (args.size() != schema.arguments().size()) {
      return false;
    }
    auto leaf_index = [&](py::handle x) {
      const auto i = x.cast<int64_t>();
      TORCH_CHECK_VALUE(
          i >= 0 && static_cast<size_t>(i) < step.leaves.size(), "no leaf ", i);
      return i;
    };
    auto is_leaf = [&](int64_t i, LeafKind kind) {
      return step.leaves[i].kind == kind;
    };
    for (size_t i = 0; i < args.size(); ++i) {
      const c10::Argument& a = schema.arguments()[i];
      auto x = args[i].cast<py::tuple>();
      const auto code = x[0].cast<int64_t>();
      const c10::TypePtr type = unwrap(a.type());
      BoxedArg arg{ArgKind::Constant, -1, {}, {}};
      if (code == 3) {
        if (!a.default_value()) {
          return false;
        }
        arg.constant = *a.default_value();
      } else if (code == 0) {
        try {
          arg.constant = torch::jit::toIValue(x[1], a.type(), a.N());
        } catch (const std::exception&) {
          return false;
        }
      } else if (code == 1) {
        arg.leaf = leaf_index(x[1]);
        const auto kind = type->kind();
        if (kind == c10::TypeKind::TensorType &&
            is_leaf(arg.leaf, LeafKind::View)) {
          arg.kind = ArgKind::Tensor;
        } else if (
            (kind == c10::TypeKind::IntType ||
             kind == c10::TypeKind::NumberType) &&
            is_leaf(arg.leaf, LeafKind::Scalar)) {
          arg.kind = ArgKind::Int;
        } else if (
            kind == c10::TypeKind::FloatType &&
            is_leaf(arg.leaf, LeafKind::Scalar)) {
          arg.kind = ArgKind::Double;
        } else {
          return false;
        }
      } else {
        TORCH_CHECK_VALUE(code == 2, "a boxed argument's kind ", code);
        auto list = type->cast<c10::ListType>();
        if (!list) {
          return false;
        }
        const auto element = list->getElementType()->kind();
        LeafKind leaf_kind = LeafKind::View;
        if (element == c10::TypeKind::TensorType) {
          arg.kind = ArgKind::TensorList;
        } else if (element == c10::TypeKind::IntType) {
          arg.kind = ArgKind::IntList;
          leaf_kind = LeafKind::Scalar;
        } else {
          return false;
        }
        for (py::handle item : x[1].cast<py::tuple>()) {
          auto it = item.cast<py::tuple>();
          if (it[0].cast<bool>()) {
            const int64_t leaf = leaf_index(it[1]);
            if (!is_leaf(leaf, leaf_kind)) {
              return false;
            }
            arg.items.push_back({leaf, {}});
          } else if (
              arg.kind == ArgKind::IntList && PyLong_Check(it[1].ptr())) {
            arg.items.push_back({-1, it[1].cast<int64_t>()});
          } else {
            return false;
          }
        }
      }
      step.args.push_back(std::move(arg));
    }
    for (py::handle o : t[3].cast<py::tuple>()) {
      auto p = o.cast<py::tuple>();
      if (!p[0].cast<bool>()) {
        step.outputs.push_back({-1, leaf_index(p[1]), {}, {}, -1, {}});
        continue;
      }
      const auto root = p[1].cast<int64_t>();
      TORCH_CHECK_VALUE(
          root >= 0 && n_alloc_ + root < base_count_, "no eager output ", root);
      TORCH_CHECK_TYPE(THPDtype_Check(p[5].ptr()), "an eager output's dtype");
      step.outputs.push_back(
          {root,
           -1,
           check_rows(ints<int64_t>(p[2])),
           check_rows(ints<int64_t>(p[3])),
           check_row(p[4].cast<int64_t>()),
           reinterpret_cast<THPDtype*>(p[5].ptr())->scalar_type});
    }
    step.target = py::reinterpret_borrow<py::object>(t[4]);
    step.name = py::str(step.target);
    if (!t[5].is_none()) {
      auto s = t[5].cast<py::tuple>();
      auto reduction = [](py::handle x) {
        const auto [reduced, splitk] = x.cast<std::pair<bool, bool>>();
        using Option = at::CuBLASReductionOption;
        return reduced ? Option::AllowReducedPrecisionWithSplitK
            : splitk   ? Option::DisallowReducedPrecisionAllowSplitK
                       : Option::DisallowReducedPrecisionDisallowSplitK;
      };
      step.blas = BlasState{
          at::str2precision(s[0].cast<std::string>()),
          reduction(s[1]),
          reduction(s[2]),
          s[3].cast<bool>(),
          s[4].cast<std::optional<int32_t>>(),
          s[5].cast<at::BlasBackend>()};
    }
    step.op = *op;
    return true;
  };
  for (py::handle s : spec.attr("steps").cast<py::tuple>()) {
    if (PyLong_Check(s.ptr())) {
      const auto segment = s.cast<int64_t>();
      TORCH_CHECK_VALUE(
          segment >= 0 && static_cast<size_t>(segment) < segments_.size(),
          "no run ",
          segment);
      steps_.push_back({segment, 0});
      continue;
    }
    auto t = s.cast<py::tuple>();
    EagerStep step{py::reinterpret_borrow<py::object>(t[0]), {}};
    for (py::handle x : t[1].cast<py::tuple>()) {
      auto leaf = x.cast<py::tuple>();
      const auto kind = static_cast<LeafKind>(leaf[0].cast<uint8_t>());
      if (kind == LeafKind::Constant) {
        step.leaves.push_back(
            {kind, -1, py::reinterpret_borrow<py::object>(leaf[1])});
      } else {
        TORCH_CHECK_VALUE(
            kind == LeafKind::View || kind == LeafKind::Scalar, "a leaf kind");
        const auto i = leaf[1].cast<int64_t>();
        step.leaves.push_back(
            {kind, kind == LeafKind::View ? check_view(i) : check_row(i), {}});
      }
    }
    if (!t[2].is_none() && !boxed_step(step, t[2].cast<py::tuple>())) {
      step.op.reset();
      step.args.clear();
      step.outputs.clear();
    }
    steps_.push_back({-1, eager_.size()});
    eager_.push_back(std::move(step));
  }
  for (py::handle m : spec.attr("memory").cast<py::tuple>()) {
    auto t = m.cast<py::tuple>();
    StepMemory memory{ints<int64_t>(t[0]), {}, ints<int64_t>(t[2])};
    memory.arguments = ints<int64_t>(t[4]);
    for (int64_t k : memory.tensors) {
      TORCH_CHECK_VALUE(
          k >= 0 && static_cast<size_t>(k) < n_alloc_, "no allocation ", k);
    }
    for (int64_t b : memory.drops) {
      check_base(b);
    }
    for (py::handle x : t[1].cast<py::tuple>()) {
      auto [k, seq, last] = x.cast<std::tuple<int64_t, int64_t, int64_t>>();
      TORCH_CHECK_VALUE(
          k >= 0 && static_cast<size_t>(k) < n_alloc_, "no allocation ", k);
      memory.temporaries.push_back({k, seq, last});
    }
    for (int64_t e : ints<int64_t>(t[3])) {
      const int64_t k = e < 0 ? -1 - e : e;
      const auto tmp = std::find_if(
          memory.temporaries.begin(),
          memory.temporaries.end(),
          [&](const Temporary& x) { return x.alloc == k; });
      const bool temporary = tmp != memory.temporaries.end();
      TORCH_CHECK_VALUE(
          k >= 0 && static_cast<size_t>(k) < n_alloc_ &&
              (e >= 0 || temporary),
          "no allocation or temporary ",
          k);
      memory.order.push_back(
          {e < 0 ? Op::Free : temporary ? Op::Temporary : Op::Tensor,
           k,
           static_cast<size_t>(tmp - memory.temporaries.begin())});
    }
    TORCH_CHECK_VALUE(
        memory.order.empty() ||
            (memory_.size() < steps_.size() &&
             steps_[memory_.size()].segment >= 0),
        "an allocation order for a step that is not a run");
    memory_.push_back(std::move(memory));
  }
  TORCH_CHECK_VALUE(
      memory_.size() == steps_.size() + 1, "a memory plan per step and one");
  for (py::handle o : spec.attr("outputs").cast<py::tuple>()) {
    auto t = o.cast<py::tuple>();
    const auto kind = static_cast<OutputKind>(t[0].cast<uint8_t>());
    switch (kind) {
      case OutputKind::Scalar:
        outputs_.push_back({kind, check_row(t[1].cast<int64_t>()), {}});
        break;
      case OutputKind::View:
        outputs_.push_back({kind, check_view(t[1].cast<int64_t>()), {}});
        break;
      case OutputKind::Base:
        outputs_.push_back({kind, -1, base_ref(t[1], t[2])});
        break;
      case OutputKind::Alias: {
        const auto i = t[1].cast<int64_t>();
        TORCH_CHECK_VALUE(
            i >= 0 && static_cast<size_t>(i) < outputs_.size(),
            "no output ",
            i);
        outputs_.push_back({kind, i, {}});
        break;
      }
      default:
        TORCH_CHECK_VALUE(false, "an output kind");
    }
  }
  auto slot = [this](int64_t arg) {
    const auto it =
        std::find(pair_arguments_.begin(), pair_arguments_.end(), arg);
    if (it != pair_arguments_.end()) {
      return static_cast<size_t>(it - pair_arguments_.begin());
    }
    pair_arguments_.push_back(arg);
    return pair_arguments_.size() - 1;
  };
  for (py::handle p : spec.attr("argument_pairs").cast<py::tuple>()) {
    auto [a, b] = p.cast<std::tuple<int64_t, int64_t>>();
    TORCH_CHECK_VALUE(0 <= a && a < b, "an argument pair (", a, ", ", b, ")");
    argument_pairs_.push_back({slot(a), slot(b)});
  }
  result_kind_ = spec.attr("result_kind").cast<int64_t>();
  TORCH_CHECK_VALUE(
      result_kind_ >= 0 && result_kind_ <= 3, "a result kind ", result_kind_);
  TORCH_CHECK_VALUE(
      result_kind_ != 1 || outputs_.size() == 1, "a tensor result's outputs");
  device_ = spec.attr("device").cast<c10::DeviceIndex>();
  disagreement_ = spec.attr("disagreement");
  auto hooks = spec.attr("replay_hooks").cast<py::tuple>();
  global_replay_start_hooks_ = hooks[0];
  global_replay_end_hooks_ = hooks[1];
  TORCH_CHECK_TYPE(
      PyDict_Check(global_replay_start_hooks_.ptr()) &&
          PyDict_Check(global_replay_end_hooks_.ptr()),
      "the global replay hooks");
  c10::cuda::DriverAPI::get();
}

int64_t HostTraceVariant::check_row(int64_t row) const {
  TORCH_CHECK_VALUE(
      row >= 0 && static_cast<size_t>(row) < program_.num_rows(),
      "no row ",
      row);
  return row;
}

int64_t HostTraceVariant::check_base(int64_t base) const {
  TORCH_CHECK_VALUE(
      base >= 0 && static_cast<size_t>(base) < base_count_, "no base ", base);
  return base;
}

HostTraceVariant::Field HostTraceVariant::parse_source(
    py::handle source) const {
  auto [row, base, delta] =
      source.cast<std::tuple<int64_t, int64_t, int64_t>>();
  TORCH_CHECK_VALUE(base == -1 || check_base(base) >= 0, "no base ", base);
  return {0, 0, 8, true, check_row(row), base, delta};
}

HostTraceVariant::Field HostTraceVariant::parse_field(
    py::handle field,
    const KernelRow& k) const {
  auto t = field.cast<py::tuple>();
  TORCH_CHECK_VALUE(t.size() == 7, "a field of ", t.size(), " items");
  Field f = parse_source(py::make_tuple(t[4], t[5], t[6]));
  f.param = t[0].cast<uint32_t>();
  f.offset = t[1].cast<uint32_t>();
  const auto width = t[2].cast<int64_t>();
  f.pointer = t[3].cast<bool>();
  TORCH_CHECK_VALUE(
      width == 0 || (width == 1 && !f.pointer) || width == 2 || width == 4 ||
          width == 8,
      "a field of ",
      width,
      " bytes");
  TORCH_CHECK_VALUE(
      width == 0 ||
          (f.param < k.param_sizes.size() &&
           f.offset + width <= k.param_sizes[f.param]),
      "a field outside its parameter");
  TORCH_CHECK_VALUE(f.pointer || f.base == -1, "a scalar field with a base");
  f.width = static_cast<uint8_t>(width);
  return f;
}

void HostTraceVariant::parse_rng(
    KernelRow& k,
    const Segment& g,
    py::handle rng,
    py::handle increment) const {
  for (py::handle x : rng.cast<py::tuple>()) {
    auto [param, offset, kind, delta] =
        x.cast<std::tuple<uint32_t, uint32_t, uint8_t, int64_t>>();
    TORCH_CHECK_VALUE(
        kind <= 2 && param < k.param_sizes.size() &&
            offset + 8 <= k.param_sizes[param],
        "a philox field outside its parameter");
    k.rng.push_back({param, offset, kind, delta});
  }
  k.rng_increment = increment.cast<int64_t>();
  TORCH_CHECK_VALUE(
      k.rng_increment >= 0 && k.rng_increment % 4 == 0,
      "an RNG increment of ",
      k.rng_increment);
  TORCH_CHECK_VALUE(
      g.generator || (k.rng.empty() && k.rng_increment == 0),
      "an RNG kernel in a segment of no generator");
}

unsigned int HostTraceVariant::check_element_size(py::handle size) {
  const auto n = size.cast<unsigned int>();
  TORCH_CHECK_VALUE(
      n == 1 || n == 2 || n == 4, "a memset element of ", n, " bytes");
  return n;
}

std::array<int64_t, 7> HostTraceVariant::set_launch(
    KernelRow& k,
    py::handle function,
    py::handle block,
    py::handle smem,
    py::handle grid) {
  k.params.func = reinterpret_cast<CUfunction>(function.cast<uintptr_t>());
  const auto g = grid.cast<std::array<int64_t, 3>>();
  const auto b = block.cast<std::array<int64_t, 3>>();
  return {g[0], g[1], g[2], b[0], b[1], b[2], smem.cast<int64_t>()};
}

void HostTraceVariant::append_images(
    KernelRow& k,
    py::handle images,
    std::vector<uint8_t>& bytes) {
  k.offset = bytes.size();
  for (py::handle image : images.cast<py::tuple>()) {
    auto b = image.cast<std::string>();
    // each parameter 8-byte aligned
    bytes.resize((bytes.size() + 7) / 8 * 8);
    k.param_offsets.push_back(bytes.size() - k.offset);
    k.param_sizes.push_back(static_cast<uint32_t>(b.size()));
    bytes.insert(bytes.end(), b.begin(), b.end());
  }
  k.nbytes = bytes.size() - k.offset;
}

void HostTraceVariant::bind_row(
    KernelRow& k,
    uint8_t* image,
    uint8_t* held,
    const Field* fields) {
  k.image = image + k.offset;
  k.held = held + k.offset;
  k.fields = fields + k.first_field;
  k.arguments.clear();
  for (size_t off : k.param_offsets) {
    k.arguments.push_back(k.image + off);
  }
  k.params.kernelParams = k.arguments.data();
}

} // namespace torch::cuda::host_trace
#endif
