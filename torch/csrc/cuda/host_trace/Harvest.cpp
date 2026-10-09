#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM) && defined(__linux__)
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/detail/TensorIteratorBuild.h>
#include <ATen/cuda/host_trace/LaunchLayout.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/sum.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAHostTraceOwners.h>
#include <c10/cuda/driver_api.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <pthread.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <tuple>
#include <unordered_map>

namespace torch::cuda::host_trace {

namespace {

std::pair<int64_t, int64_t> stack_bounds() {
  thread_local std::optional<std::pair<int64_t, int64_t>> bounds;
  if (bounds) {
    return *bounds;
  }
  pthread_attr_t attr;
  TORCH_CHECK(pthread_getattr_np(pthread_self(), &attr) == 0);
  void* lo = nullptr;
  size_t size = 0;
  const int rc = pthread_attr_getstack(&attr, &lo, &size);
  pthread_attr_destroy(&attr);
  TORCH_CHECK(rc == 0);
  const auto start = reinterpret_cast<int64_t>(lo);
  bounds.emplace(start, start + static_cast<int64_t>(size));
  return *bounds;
}

namespace alloc = c10::cuda::CUDACachingAllocator;
using Pool = std::tuple<int64_t, int64_t>;

bool same_pool(const c10::MempoolId_t& id, const Pool& pool) {
  return static_cast<int64_t>(id.first) == std::get<0>(pool) &&
      static_cast<int64_t>(id.second) == std::get<1>(pool);
}

// The pool's blocks on the device as (stream, address, size, requested
// size, active, in the small pool)
py::list pool_blocks(int64_t device, const Pool& pool) {
  const auto info =
      alloc::snapshot({std::get<0>(pool), std::get<1>(pool)}, false);
  py::list blocks;
  for (const auto& seg : info.segments) {
    if (seg.device != device || !same_pool(seg.owner_private_pool_id, pool)) {
      continue;
    }
    int64_t address = static_cast<int64_t>(seg.address);
    for (const auto& b : seg.blocks) {
      blocks.append(py::make_tuple(
          reinterpret_cast<int64_t>(seg.stream),
          address,
          b.size,
          b.requested_size,
          b.allocated || b.active,
          !seg.is_large));
      address += static_cast<int64_t>(b.size);
    }
  }
  return blocks;
}

// The (address, requested bytes) of each allocation into the pool since the
// last call, oldest first; the pool logs on while `on`
py::list pool_log(int64_t device, const Pool& pool, bool on) {
  py::list out;
  for (const auto& [ptr, bytes] : alloc::takePoolLog(
           static_cast<c10::DeviceIndex>(device),
           {std::get<0>(pool), std::get<1>(pool)},
           on)) {
    out.append(py::make_tuple(reinterpret_cast<int64_t>(ptr), bytes));
  }
  return out;
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12040
__attribute__((noinline)) void smear(int pattern) {
  unsigned char buf[1 << 16];
  std::memset(buf, pattern, sizeof(buf));
  asm volatile("" : : "r"(buf) : "memory");
}

// The op's CUDA kernel through the dispatcher, with the stack below it
// smeared first, so that uninitialized padding a library leaves in its
// kernels' parameters holds the pattern. Without the GIL, like an eager op
py::object smeared_call(
    const std::string& name,
    const std::string& overload,
    int64_t pattern,
    py::args args,
    const py::kwargs& kwargs) {
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      name.c_str(), overload.c_str());
  auto stack = torch::jit::createStackForSchema(
      op.schema(),
      torch::jit::tuple_slice(std::move(args)),
      kwargs,
      std::nullopt);
  {
    py::gil_scoped_release no_gil;
    smear(static_cast<int>(pattern));
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CUDA), &stack);
  }
  return torch::jit::createPyObjectForStack(std::move(stack));
}

bool harvest_driver() {
  auto* d = c10::cuda::DriverAPI::get();
  return d->cuGraphGetNodes_ && d->cuGraphGetEdges_ && d->cuGraphNodeGetType_ &&
      d->cuGraphKernelNodeGetParams_ && d->cuGraphKernelNodeGetAttribute_ &&
      d->cuGraphMemsetNodeGetParams_ && d->cuFuncGetName_ &&
      d->cuFuncGetParamInfo_ && d->cuLaunchKernelEx_ && d->cuMemsetD2D8Async_ &&
      d->cuMemsetD2D16Async_ && d->cuMemsetD2D32Async_ &&
      d->cuPointerGetAttribute_;
}

using Layout = std::vector<std::pair<size_t, size_t>>;

// each kernel's (offset, size) per parameter, by handle and name (a handle
// an unloaded module freed can come back as another kernel's)
const Layout& layout_of(CUfunction func, const char* name) {
  static std::map<std::pair<CUfunction, std::string>, Layout> layouts;
  auto [it, fresh] = layouts.try_emplace({func, name});
  if (fresh) {
    auto* driver = c10::cuda::DriverAPI::get();
    size_t offset = 0;
    size_t size = 0;
    while (driver->cuFuncGetParamInfo_(
               func, it->second.size(), &offset, &size) == CUDA_SUCCESS) {
      it->second.emplace_back(offset, size);
    }
  }
  return it->second;
}

using Attributes = std::map<int64_t, std::string>;

// the launch attributes of `ids` the driver reports for a kernel node, as
// raw CUlaunchAttributeValues; the programmatic edge is read from the graph
Attributes node_attributes(CUgraphNode node, const std::vector<int64_t>& ids) {
  auto* driver = c10::cuda::DriverAPI::get();
  Attributes out;
  for (const auto id : ids) {
    if (id == CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION) {
      continue;
    }
    CUlaunchAttributeValue v;
    std::memset(&v, 0, sizeof(v));
    if (driver->cuGraphKernelNodeGetAttribute_(
            node, static_cast<CUkernelNodeAttrID>(id), &v) == CUDA_SUCCESS) {
      out.emplace(
          id, std::string(reinterpret_cast<const char*>(&v), sizeof(v)));
    }
  }
  return out;
}

// The nodes of a captured graph in stream order, as
// _host_trace_capture.graph_nodes reads them, but the first (a plain
// launch, whose attributes the others' are read against): a kernel as
// (name, function, layout, grid, block, shared bytes, whether the edge into
// it is programmatic, the attributes of `ids` that differ from the first's
// as (id, raw value), parameter images), a memset as (dst, value, element
// size, width, height, pitch)
py::list captured_nodes(cudaGraph_t raw, const std::vector<int64_t>& ids) {
  auto* driver = c10::cuda::DriverAPI::get();
  auto graph = reinterpret_cast<CUgraph>(raw);
  size_t n = 0;
  C10_CUDA_DRIVER_CHECK(driver->cuGraphGetNodes_(graph, nullptr, &n));
  std::vector<CUgraphNode> handles(n);
  C10_CUDA_DRIVER_CHECK(driver->cuGraphGetNodes_(graph, handles.data(), &n));
  size_t m = 0;
  C10_CUDA_DRIVER_CHECK(
      driver->cuGraphGetEdges_(graph, nullptr, nullptr, nullptr, &m));
  std::vector<CUgraphNode> from(m);
  std::vector<CUgraphNode> to(m);
  std::vector<CUgraphEdgeData> data(m);
  C10_CUDA_DRIVER_CHECK(
      driver->cuGraphGetEdges_(graph, from.data(), to.data(), data.data(), &m));
  std::unordered_map<CUgraphNode, size_t> position;
  for (size_t i = 0; i < n; ++i) {
    position.emplace(handles[i], i);
  }
  std::vector<std::vector<size_t>> succ(n);
  std::vector<std::vector<CUgraphEdgeData>> preds(n);
  bool programmatic_edges = false;
  for (size_t e = 0; e < m; ++e) {
    succ[position.at(from[e])].push_back(position.at(to[e]));
    preds[position.at(to[e])].push_back(data[e]);
    programmatic_edges |= data[e].type == CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC;
  }
  const bool forks = std::any_of(
                         succ.begin(),
                         succ.end(),
                         [](const auto& v) { return v.size() > 1; }) ||
      std::any_of(preds.begin(), preds.end(), [](const auto& v) {
                       return v.size() > 1;
                     });
  TORCH_CHECK_VALUE(
      !(forks && programmatic_edges), "a programmatic edge in a fork or join");
  // a topological order, ties in node order
  std::vector<size_t> waiting(n);
  std::priority_queue<size_t, std::vector<size_t>, std::greater<>> ready;
  for (size_t i = 0; i < n; ++i) {
    waiting[i] = preds[i].size();
    if (!waiting[i]) {
      ready.push(i);
    }
  }
  std::vector<size_t> chain;
  while (!ready.empty()) {
    chain.push_back(ready.top());
    ready.pop();
    for (const auto b : succ[chain.back()]) {
      if (!--waiting[b]) {
        ready.push(b);
      }
    }
  }
  TORCH_CHECK_VALUE(!chain.empty(), "a capture without its plain launch");
  py::list out;
  Attributes plain;
  for (size_t c = 0; c < chain.size(); ++c) {
    const auto i = chain[c];
    CUgraphNode node = handles[i];
    CUgraphNodeType kind{};
    C10_CUDA_DRIVER_CHECK(driver->cuGraphNodeGetType_(node, &kind));
    const bool programmatic = !preds[i].empty() &&
        preds[i][0].type == CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC;
    TORCH_CHECK_VALUE(
        !programmatic ||
            (kind == CU_GRAPH_NODE_TYPE_KERNEL &&
             preds[i][0].from_port == CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC),
        "a programmatic edge from port ",
        static_cast<int>(preds[i][0].from_port),
        " into a node of type ",
        static_cast<int>(kind));
    if (kind == CU_GRAPH_NODE_TYPE_MEMSET && c) {
      CUDA_MEMSET_NODE_PARAMS p{};
      C10_CUDA_DRIVER_CHECK(driver->cuGraphMemsetNodeGetParams_(node, &p));
      out.append(py::make_tuple(
          static_cast<int64_t>(p.dst),
          p.value,
          p.elementSize,
          p.width,
          p.height,
          p.pitch));
      continue;
    }
    TORCH_CHECK_VALUE(
        kind == CU_GRAPH_NODE_TYPE_KERNEL,
        "the call adds a node of type ",
        static_cast<int>(kind));
    if (!c) {
      plain = node_attributes(node, ids);
      continue;
    }
    CUDA_KERNEL_NODE_PARAMS p{};
    C10_CUDA_DRIVER_CHECK(driver->cuGraphKernelNodeGetParams_(node, &p));
    const char* name = nullptr;
    C10_CUDA_DRIVER_CHECK(driver->cuFuncGetName_(&name, p.func));
    const auto& layout = layout_of(p.func, name);
    py::tuple images(layout.size());
    if (p.kernelParams) {
      for (size_t j = 0; j < layout.size(); ++j) {
        images[j] = py::bytes(
            static_cast<const char*>(p.kernelParams[j]), layout[j].second);
      }
    } else {
      const char* buffer = nullptr;
      const size_t* size = nullptr;
      for (void** e = p.extra; e && *e != CU_LAUNCH_PARAM_END; e += 2) {
        if (*e == CU_LAUNCH_PARAM_BUFFER_POINTER) {
          buffer = static_cast<const char*>(e[1]);
        } else if (*e == CU_LAUNCH_PARAM_BUFFER_SIZE) {
          size = static_cast<const size_t*>(e[1]);
        }
      }
      size_t end = 0;
      for (const auto& [o, k] : layout) {
        end = std::max(end, o + k);
      }
      TORCH_CHECK_VALUE(
          buffer && size && *size >= end,
          "a kernel node's extra parameters are not one buffer");
      for (size_t j = 0; j < layout.size(); ++j) {
        images[j] = py::bytes(buffer + layout[j].first, layout[j].second);
      }
    }
    py::list attributes;
    for (const auto& [id, v] : node_attributes(node, ids)) {
      auto it = plain.find(id);
      if (it == plain.end() || it->second != v) {
        attributes.append(py::make_tuple(id, py::bytes(v)));
      }
    }
    py::tuple params(layout.size());
    for (size_t j = 0; j < layout.size(); ++j) {
      params[j] = py::make_tuple(layout[j].first, layout[j].second);
    }
    out.append(py::make_tuple(
        name,
        reinterpret_cast<int64_t>(p.func),
        params,
        py::make_tuple(p.gridDimX, p.gridDimY, p.gridDimZ),
        py::make_tuple(p.blockDimX, p.blockDimY, p.blockDimZ),
        p.sharedMemBytes,
        programmatic,
        py::tuple(attributes),
        images));
  }
  return out;
}

using StreamParts = std::tuple<int64_t, int64_t, int64_t>;

// One capture of a harvest into `graph` from `pool` on the stream: the
// anchor's fill (a plain launch), then the op's CUDA kernel as
// smeared_call calls it, with its cuBLAS workspace at `workspace` (`size`
// bytes) and, with `zero_init`, the harvest flag set; the stream's cached
// cuBLAS workspace is released before and after. With `pretake`, the
// default generator hands the capture its philox state that many offsets
// on. Returns the call's result, its allocations as pool_log's, the
// workspace bytes it asked for, (seed, offset, offsets taken) or None, and
// what keeps its library kernels loaded (a capsule) or None
py::tuple harvest_capture(
    at::cuda::CUDAGraph& graph,
    const Pool& pool,
    const StreamParts& parts,
    const at::Tensor& anchor,
    int64_t workspace,
    size_t size,
    std::optional<int64_t> pretake,
    bool zero_init,
    const std::string& name,
    const std::string& overload,
    int64_t pattern,
    py::args args,
    const py::kwargs& kwargs) {
  const auto stream = c10::cuda::CUDAStream::unpack3(
      std::get<0>(parts),
      static_cast<c10::DeviceIndex>(std::get<1>(parts)),
      static_cast<c10::DeviceType>(std::get<2>(parts)));
  const auto device = stream.device_index();
  c10::cuda::CUDAStreamGuard guard(stream);
  at::cuda::clearCublasWorkspacesForStream(stream.stream());
  graph.capture_begin(
      {std::get<0>(pool), std::get<1>(pool)}, cudaStreamCaptureModeThreadLocal);
  py::object result;
  size_t requested = 0;
  py::object philox = py::none();
  c10::cuda::HostTraceOwners owners;
  try {
    anchor.fill_(1);
    at::Generator gen;
    std::tuple<at::Tensor, at::Tensor, at::Tensor> state;
    if (pretake) {
      gen = at::cuda::detail::getDefaultCUDAGenerator(device);
      std::scoped_lock lock(gen.mutex());
      state = gen.philox_state(*pretake);
    }
    pool_log(device, pool, true);
    const bool previous = c10::cuda::isHostTraceHarvesting();
    at::cuda::setCUDABlasWorkspaceAddressOverride(
        reinterpret_cast<void*>(workspace), size);
    c10::cuda::setHostTraceHarvesting(zero_init || previous);
    auto* sink = c10::cuda::setHostTraceOwnerSink(&owners);
    try {
      result = smeared_call(name, overload, pattern, std::move(args), kwargs);
    } catch (...) {
      c10::cuda::setHostTraceOwnerSink(sink);
      c10::cuda::setHostTraceHarvesting(previous);
      at::cuda::setCUDABlasWorkspaceAddressOverride(nullptr, 0);
      throw;
    }
    c10::cuda::setHostTraceOwnerSink(sink);
    c10::cuda::setHostTraceHarvesting(previous);
    requested = at::cuda::setCUDABlasWorkspaceAddressOverride(nullptr, 0);
    if (pretake) {
      std::scoped_lock lock(gen.mutex());
      const auto now = std::get<2>(gen.philox_state(0)).item<int64_t>();
      philox = py::make_tuple(
          std::get<0>(state), std::get<1>(state), now - *pretake);
    }
  } catch (...) {
    try {
      graph.capture_end();
    } catch (...) {
    }
    try {
      pool_log(device, pool, false);
    } catch (...) {
    }
    throw;
  }
  graph.capture_end();
  at::cuda::clearCublasWorkspacesForStream(stream.stream());
  auto allocations = pool_log(device, pool, false);
  py::object held = py::none();
  if (!owners.empty()) {
    held = py::capsule(
        new c10::cuda::HostTraceOwners(std::move(owners)), [](void* p) {
          delete static_cast<c10::cuda::HostTraceOwners*>(p);
        });
  }
  return py::make_tuple(result, allocations, requested, philox, held);
}

// Launches nodes in order on the stream: a kernel as (function, grid, block,
// shared bytes, launch attributes as (id, raw value), parameter images, and
// (parameter, offset, qword) of each slot), a memset as (dst, value,
// element size, width, height, pitch)
void launch(int64_t stream, const py::list& nodes) {
  auto* driver = c10::cuda::DriverAPI::get();
  auto s = reinterpret_cast<CUstream>(stream);
  for (const auto& handle : nodes) {
    const auto node = handle.cast<py::tuple>();
    if (node.size() == 6) {
      const auto dst = static_cast<CUdeviceptr>(node[0].cast<int64_t>());
      const auto value = node[1].cast<uint32_t>();
      const auto element = node[2].cast<size_t>();
      const auto width = node[3].cast<size_t>();
      const auto height = node[4].cast<size_t>();
      auto pitch = node[5].cast<size_t>();
      pitch = pitch ? pitch : width * element;
      if (element == 1) {
        C10_CUDA_DRIVER_CHECK(driver->cuMemsetD2D8Async_(
            dst, pitch, static_cast<unsigned char>(value), width, height, s));
      } else if (element == 2) {
        C10_CUDA_DRIVER_CHECK(driver->cuMemsetD2D16Async_(
            dst, pitch, static_cast<unsigned short>(value), width, height, s));
      } else {
        TORCH_CHECK_VALUE(element == 4, "a memset of ", element, " bytes");
        C10_CUDA_DRIVER_CHECK(
            driver->cuMemsetD2D32Async_(dst, pitch, value, width, height, s));
      }
      continue;
    }
    auto images = node[5].cast<std::vector<std::string>>();
    for (const auto& slot : node[6]) {
      const auto [param, offset, q] =
          slot.cast<std::tuple<size_t, size_t, int64_t>>();
      TORCH_CHECK_VALUE(
          param < images.size() && offset + 8 <= images[param].size(),
          "a slot past its parameter");
      std::memcpy(images[param].data() + offset, &q, 8);
    }
    std::vector<void*> params;
    params.reserve(images.size());
    for (auto& image : images) {
      params.push_back(image.data());
    }
    std::vector<CUlaunchAttribute> attrs;
    for (const auto& a : node[4]) {
      const auto [id, raw] = a.cast<std::pair<int64_t, std::string>>();
      CUlaunchAttribute attr;
      std::memset(&attr, 0, sizeof(attr));
      attr.id = static_cast<CUlaunchAttributeID>(id);
      std::memcpy(
          &attr.value, raw.data(), std::min(raw.size(), sizeof(attr.value)));
      attrs.push_back(attr);
    }
    const auto grid = node[1].cast<std::tuple<unsigned, unsigned, unsigned>>();
    const auto block = node[2].cast<std::tuple<unsigned, unsigned, unsigned>>();
    CUlaunchConfig cfg{};
    std::tie(cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ) = grid;
    std::tie(cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ) = block;
    cfg.sharedMemBytes = node[3].cast<unsigned>();
    cfg.hStream = s;
    cfg.attrs = attrs.data();
    cfg.numAttrs = static_cast<unsigned>(attrs.size());
    C10_CUDA_DRIVER_CHECK(driver->cuLaunchKernelEx_(
        &cfg,
        reinterpret_cast<CUfunction>(node[0].cast<int64_t>()),
        params.data(),
        nullptr));
  }
}

// The qwords at a node's (parameter, offset) `slots`, its images with those
// bytes zeroed, and the first qword, at every 4-byte offset more than 7
// bytes from each slot of its parameter, that is in a `live` range, or a
// CUDA address above 4 GiB outside `stack` that differs from the template's
// qword there: (parameter, offset, qword), or None
py::tuple slot_qwords(
    std::vector<std::string> images,
    const std::vector<std::string>& templates,
    const std::vector<std::pair<size_t, size_t>>& slots,
    const std::vector<std::pair<uint64_t, uint64_t>>& live,
    const std::pair<uint64_t, uint64_t>& stack) {
  std::vector<uint64_t> qwords;
  for (const auto& [param, off] : slots) {
    auto& image = images.at(param);
    TORCH_CHECK(off + 8 <= image.size(), "slot past its parameter's bytes");
    uint64_t q = 0;
    std::memcpy(&q, image.data() + off, 8);
    std::memset(image.data() + off, 0, 8);
    qwords.push_back(q);
  }
  py::list zeroed;
  for (const auto& image : images) {
    zeroed.append(py::bytes(image));
  }
  auto* driver = c10::cuda::DriverAPI::get();
  uint64_t low = std::numeric_limits<uint64_t>::max();
  uint64_t high = 0;
  for (const auto& [lo, hi] : live) {
    low = std::min(low, lo);
    high = std::max(high, hi);
  }
  // qwords the driver said are no CUDA address
  std::vector<uint64_t> plain;
  for (size_t param = 0; param < images.size(); ++param) {
    const auto& image = images[param];
    const auto& was_image = templates.at(param);
    const size_t n = std::min(image.size(), was_image.size());
    for (const size_t first : {size_t{0}, size_t{4}}) {
      for (size_t off = first; off + 8 <= n; off += 8) {
        uint64_t q = 0;
        uint64_t was = 0;
        std::memcpy(&q, image.data() + off, 8);
        std::memcpy(&was, was_image.data() + off, 8);
        if (std::any_of(slots.begin(), slots.end(), [&](const auto& s) {
              return s.first == param &&
                  (off > s.second ? off - s.second : s.second - off) <= 7;
            })) {
          continue;
        }
        bool stray =
            low <= q && q < high &&
            std::any_of(
                live.begin(),
                live.end(),
                [&](const auto& r) { return r.first <= q && q < r.second; });
        if (!stray && (q >> 32) && q != was &&
            !(stack.first <= q && q < stack.second) &&
            std::find(plain.begin(), plain.end(), q) == plain.end()) {
          unsigned int type = 0;
          stray = driver->cuPointerGetAttribute_(
                      &type,
                      CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                      static_cast<CUdeviceptr>(q)) == CUDA_SUCCESS;
          if (!stray) {
            plain.push_back(q);
          }
        }
        if (stray) {
          return py::make_tuple(zeroed, qwords, py::make_tuple(param, off, q));
        }
      }
    }
  }
  return py::make_tuple(zeroed, qwords, py::none());
}

std::vector<std::string> function_names(const std::vector<int64_t>& funcs) {
  auto* driver = c10::cuda::DriverAPI::get();
  std::vector<std::string> out;
  out.reserve(funcs.size());
  for (const auto f : funcs) {
    const char* name = nullptr;
    C10_CUDA_DRIVER_CHECK(
        driver->cuFuncGetName_(&name, reinterpret_cast<CUfunction>(f)));
    out.emplace_back(name);
  }
  return out;
}
#endif

// Each tensor's digest: the sums of its bits along its rows in memory and,
// with `columns[i]`, down its columns over at most 255 rows at a time (a
// single row's bits), wrapping at their width: a wider sum would read a
// converted copy, and a longer column sum takes a global buffer about twice
// the tensor (Reduce.cuh's split across blocks). All of them in one byte
// buffer, returned with each tensor's digest as a view of it, so one
// comparison covers them
std::tuple<at::Tensor, std::vector<at::Tensor>> digests(
    const std::vector<at::Tensor>& tensors,
    const std::vector<bool>& columns) {
  TORCH_CHECK(tensors.size() == columns.size());
  const size_t count = tensors.size();
  std::vector<at::Tensor> rows_of(count);
  std::vector<int64_t> lengths(count);
  for (size_t i = 0; i < count; ++i) {
    const auto& t = tensors[i];
    const auto bits = t.element_size() == 1 ? at::kByte
        : t.element_size() == 2             ? at::kShort
        : t.element_size() == 4             ? at::kInt
                                            : at::kLong;
    auto x = t.view(bits).squeeze();
    if (!x.is_contiguous()) {
      std::vector<int64_t> order(x.dim());
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return x.stride(a) > x.stride(b);
      });
      x = x.permute(order);
    }
    x = x.dim() < 2 ? x.reshape({1, -1}) : x.flatten(0, -2);
    const int64_t rows = x.size(0), n = x.size(1), g = (rows + 254) / 255;
    lengths[i] = rows < 2 ? n
        : columns[i]      ? rows + g * n + (rows / g * g < rows ? n : 0)
                          : rows;
    rows_of[i] = std::move(x);
  }
  // wider digests first, so each starts aligned to its element size
  std::vector<size_t> order(count);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return rows_of[a].element_size() > rows_of[b].element_size();
  });
  std::vector<int64_t> offsets(count);
  int64_t bytes = 0;
  for (const auto i : order) {
    offsets[i] = bytes;
    bytes += lengths[i] * rows_of[i].element_size();
  }
  auto flat = at::empty(
      {bytes},
      rows_of.empty() ? at::TensorOptions()
                      : rows_of[0].options().dtype(at::kByte));
  std::vector<at::Tensor> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const auto& x = rows_of[i];
    const auto bits = x.scalar_type();
    auto d =
        flat.narrow(0, offsets[i], lengths[i] * x.element_size()).view(bits);
    const int64_t rows = x.size(0), n = x.size(1);
    if (rows < 2) {
      d.copy_(x.reshape(-1));
      out.push_back(std::move(d));
      continue;
    }
    const int64_t g = (rows + 254) / 255, h = rows / g;
    auto row_sums = d.narrow(0, 0, rows);
    at::sum_out(row_sums, x, -1, false, bits);
    if (columns[i]) {
      auto groups = d.narrow(0, rows, g * n).view({g, n});
      at::sum_out(
          groups, x.narrow(0, 0, h * g).unflatten(0, {h, g}), 0, false, bits);
      if (h * g < rows) {
        auto last = d.narrow(0, rows + g * n, n);
        at::sum_out(last, x.narrow(0, h * g, rows - h * g), 0, false, bits);
      }
    }
    out.push_back(std::move(d));
  }
  return {flat, out};
}

// Tensors like `like` at these device addresses, not owning them: the
// harvest's operand sets in its arena, address space never mapped
std::vector<at::Tensor> arena_views(
    const std::vector<int64_t>& addresses,
    const std::vector<at::Tensor>& like) {
  std::vector<at::Tensor> out;
  out.reserve(like.size());
  for (size_t i = 0; i < like.size(); ++i) {
    out.push_back(
        at::for_blob(reinterpret_cast<void*>(addresses.at(i)), like[i].sizes())
            .strides(like[i].strides())
            .target_device(like[i].device())
            .options(like[i].options())
            .make_tensor());
  }
  return out;
}

} // namespace

void initHarvestBindings(py::module& m) {
  m.def("_cuda_hostTraceStackBounds", &stack_bounds);
  m.def("_cuda_hostTracePool", &pool_blocks);
  m.def("_cuda_hostTracePoolLog", &pool_log);
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12040
  m.def("_cuda_hostTraceHarvestDriver", &harvest_driver);
  m.def("_cuda_hostTraceHarvestCapture", &harvest_capture);
  m.def(
      "_cuda_hostTraceCapturedNodes",
      [](int64_t raw, const std::vector<int64_t>& ids) {
        return captured_nodes(reinterpret_cast<cudaGraph_t>(raw), ids);
      });
  m.def("_cuda_hostTraceLaunch", &launch);
  m.def("_cuda_hostTraceSlotQwords", &slot_qwords);
  m.def("_cuda_hostTraceFunctionNames", &function_names);
#endif
  m.def("_cuda_hostTraceDigests", &digests);
  m.def("_cuda_hostTraceArenaViews", &arena_views);
  m.def("_cuda_hostTraceSetHarvesting", [](bool enabled) {
    const bool previous = c10::cuda::isHostTraceHarvesting();
    c10::cuda::setHostTraceHarvesting(enabled);
    return previous;
  });
  // on: records the TensorIterators this thread builds; off: stops, returning
  // each one's (operands, noutputs, numel, common dtype, is_reduction)
  m.def("_cuda_hostTraceRecordIterators", [](bool on) {
    namespace ti_build = at::detail::ti_build;
    static thread_local std::vector<ti_build::BuiltIterator> built;
    ti_build::record_built_iterators(on ? &built : nullptr);
    py::list out;
    if (on) {
      built.clear();
      return out;
    }
    for (auto& it : built) {
      const std::vector<at::Tensor> operands(it.operands.begin(), it.operands.end());
      const auto common = it.common_dtype == at::ScalarType::Undefined ? py::none() : py::cast(it.common_dtype);
      out.append(py::make_tuple(operands, it.noutputs, it.numel, common, it.is_reduction));
    }
    built.clear();
    return out;
  });
  // on: records the launches this thread's pointwise launch sites report; off:
  // stops, returning each one's (function, grid, block, byte classes, bytes)
  // per parameter (LaunchLayout.h)
  m.def("_cuda_hostTraceRecordLaunches", [](bool on) {
    namespace ht = at::cuda::host_trace;
    static thread_local std::vector<ht::LaunchLayout> launched;
    ht::record_launch_layouts(on ? &launched : nullptr);
    py::list out;
    for (const auto& l : on ? std::vector<ht::LaunchLayout>() : std::exchange(launched, {})) {
      const std::vector<py::bytes> bytes(l.bytes.begin(), l.bytes.end());
      out.append(py::make_tuple(reinterpret_cast<uintptr_t>(l.function), py::make_tuple(l.grid.x, l.grid.y, l.grid.z), py::make_tuple(l.block.x, l.block.y, l.block.z), l.classes, bytes));
    }
    return out;
  });
  // the bytes eager's functor holds for a CPU scalar class (LaunchLayout.h) of
  // the 0-dim CPU tensor src
  m.def("_cuda_hostTraceCpuScalarBytes", [](const at::Tensor& src, const std::string& cls) {
    TORCH_CHECK(src.device().is_cpu() && src.dim() == 0 && cls.size() == 1, "a CPU scalar class of a 0-dim CPU tensor");
    char out[16];
    TORCH_CHECK(at::cuda::host_trace::cpu_scalar_bytes(src, cls[0], out), "not a CPU scalar class: ", cls);
    const auto type = static_cast<c10::ScalarType>(cls[0] <= '9' ? cls[0] - '0' : cls[0] - 'A');
    return py::bytes(out, c10::elementSize(type));
  });
}

} // namespace torch::cuda::host_trace
#endif
