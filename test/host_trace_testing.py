# Owner(s): ["module: cuda"]
"""Shared plumbing of the host-tracing suites (test_cuda_host_trace*.py): the
replay a suite drives (`build`), the capture that reads the nodes one call
launches, the bitwise comparison, the trace / build / replay driver and the
skips. A rule of one family (its kernel name pattern, its byte masks, the
reduce image check) stays in its suite.

The replay. A test builds a variant of a tape with `build(tape, fn, args)` and
replays it at other inputs; the tape's argument contract, input facts, opaque
calls and guards are evaluated in Python first (`TapeCheck`, through the
entry's own binder and evaluator), so a miss is named the same way on every
backend, and a served call runs one of two backends:

- the native replay (`torch._inductor.runtime._cudagraph.direct_hosttrace`,
  the runtime's parameterized graph prepared from the tape) where that
  runtime is built into torch: the integration line;
- eager itself where it is not (the stack's own installs): the tape decides
  the miss, eager executes the call. The replay property is then the tape's
  guards and eager's outputs; the graph's fidelity is the native backend's to
  prove.

`HOST_TRACE_REPLAY=eager` forces the eager form, `=native` forces the native
one and makes a tape the native line refuses a failure; by default (`auto`) a
refused tape falls back to the eager form with the refusal on
`variant.refused`, so a suite runs to the end and the refusals are counted
(the summary line at exit, `HOST_TRACE_REPLAY_LOG=<file>` for the per-test
rows). The native replay is bound to the stream it was prepared on (O29) and
declines a forked side stream (the preparation replays one stream): the two
cells the eager form serves and the native one refuses by name.
"""

import atexit
import inspect
import json
import os
import sys
import threading
import unittest
from typing import NamedTuple

import torch
from torch.testing._internal.common_utils import TEST_CUDA_PYTHON_BINDINGS, TestCase
from torch.utils._python_dispatch import TorchDispatchMode


if torch.cuda.is_available():
    from torch.cuda import _host_trace as ht

C = torch._C

# the source tree, for the tests that read the recorder's sources and run its
# lint (torch.__file__ is site-packages in CI); a wheel-only environment skips them
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

needs_two_gpus = unittest.skipIf(torch.cuda.device_count() < 2, "needs two GPUs")


def require_nvcc():
    """A SkipTest where nvcc is not available: a test host is built as an
    extension, in this process or a subprocess."""
    import shutil

    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = shutil.which("nvcc") or (
        CUDA_HOME and os.path.join(CUDA_HOME, "bin", "nvcc")
    )
    if not nvcc or not os.path.isfile(nvcc):
        raise unittest.SkipTest("requires nvcc for the test extension")


def load_test_extension(name, cuda_source):
    """A test host built as an extension against the installed headers
    (torch.utils.cpp_extension.load_inline: compiled once per name and source
    under TORCH_EXTENSIONS_DIR, every later process reuses the build), or a
    SkipTest where nvcc is not available."""
    from torch.utils.cpp_extension import load_inline

    require_nvcc()
    return load_inline(
        name,
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=None,
        with_cuda=True,
        extra_cflags=["-std=c++20"],
        extra_cuda_cflags=["-std=c++20"],
    )


def bits(t):
    # the bytes as integers: NaN payloads and signed zeros compare by value
    return t.contiguous().view(
        torch.int8
        if t.element_size() == 1
        else {2: torch.int16, 4: torch.int32, 8: torch.int64, 16: torch.int64}[
            t.element_size()
        ]
    )


class Case(NamedTuple):
    # one replay of a built variant: its outputs, or the miss that refused it
    out: list | None
    miss: str | None


# ---- the replay backend


def native_replay_module():
    """The adapter of the native replay when the runtime it drives is built into
    this torch (the integration line), else None."""
    if not hasattr(C, "_cuda_make_boxed_dispatch"):
        return None
    try:
        from torch._inductor.runtime._cudagraph import direct_hosttrace
    except ImportError:
        return None
    return direct_hosttrace


_BACKEND = os.environ.get("HOST_TRACE_REPLAY", "auto")
if _BACKEND not in ("auto", "native", "eager"):
    raise RuntimeError(f"HOST_TRACE_REPLAY={_BACKEND!r}: auto, native or eager")


def replay_backend():
    """ "native" or "eager": which backend `build` serves a tape with here."""
    if _BACKEND == "eager":
        return "eager"
    if native_replay_module() is not None:
        return "native"
    if _BACKEND == "native":
        raise RuntimeError("HOST_TRACE_REPLAY=native: this torch has no native replay")
    return "eager"


needs_native_replay = unittest.skipIf(
    not torch.cuda.is_available() or replay_backend() != "native",
    "a property of the native replay's graph: the eager form has none",
)


class TapeCheck:
    """The tape evaluated in Python at a call's inputs, in the recorder's order:
    the argument contract and every input's facts (`_bind_inputs`, the entry's
    own binder), the guards over the inputs alone, then the opaque calls in
    their order (the host's own function re-run; a result other than the
    traced one, or outside its declared domain, is a Miss) with the guards each
    result closes, through the entry's `_Evaluator`. A Miss names the failing
    guard as the tape's text. A guard over a symbol no input or opaque call
    binds (an allocation's address) cannot be decided without a replay and is
    an error here: no shipped host records one."""

    def __init__(self, tape, device):
        self.tape = tape
        self.device = device
        self.ev = ht._Evaluator()
        self.names = ht._input_names(tape.inputs)
        bound = set()
        for rec in tape.inputs:
            for v in (*rec.sizes, *rec.strides, rec.offset, rec.root.sym):
                name = ht._symbol_name(v)
                if name is not None:
                    bound.add(name)
        self.early = [g for g in tape.guards if ht._free_symbols(g) <= bound]
        pending = [g for g in tape.guards if g not in self.early]
        self.opaque: list = []
        for rec in sorted(tape.opaque, key=lambda o: o["seq"]):
            name = ht._symbol_name(rec["sym"])
            if name is not None:
                bound.add(name)
            closed = [g for g in pending if ht._free_symbols(g) <= bound]
            pending = [g for g in pending if g not in closed]
            self.opaque.append((rec, closed))
        self.tail = pending

    def _guards(self, guards, env):
        for g in guards:
            try:
                holds = self.ev.ev(g, env)
            except ZeroDivisionError:
                # a guard whose evaluation is undefined at these inputs (a
                # division by a size that is now zero) cannot hold
                raise ht.Miss(
                    f"guard failed: {self.ev.guard_text(g)} is undefined at these inputs"
                ) from None
            except NameError as e:
                raise AssertionError(
                    f"the guard {self.ev.guard_text(g)} reads a symbol no input or opaque call binds ({e}): undecidable without a replay"
                ) from None
            if not holds:
                note = self.tape.guard_notes.get(g)
                where = f" ({note})" if note else ""
                raise ht.Miss(
                    f"guard failed: {self.ev.guard_text(g)} is not true{where}"
                )

    def _opaque(self, rec, env):
        try:
            args = [int(self.ev.ev(a, env)) for a in rec["args"]]
        except NameError as e:
            raise AssertionError(
                f"opaque {rec['fn']} reads a symbol no input binds ({e}): undecidable without a replay"
            ) from None
        v = rec["call"](args)
        if rec["kind"] != "rebind" and v != rec["expected"]:
            raise ht.Miss(
                f"opaque {rec['fn']} is {v} at these inputs, the tape traced {rec['expected']}"
            )
        if rec["domain"] == "positive" and v < 1:
            raise ht.Miss(
                f"opaque {rec['fn']} is {v} at these inputs, below its declared domain"
            )
        name = ht._symbol_name(rec["sym"])
        if name is not None:
            env[name] = v

    def bind(self, args):
        """The input symbols' values at `args`, the argument contract and the
        input facts checked, the guards over the inputs alone held; a Miss."""
        env = ht._bind_inputs(self.tape, self.names, args, self.device)
        self._guards(self.early, env)
        return env

    def check(self, args):
        """`bind`, then the opaque calls and the guards they close: the whole
        tape at `args`; returns the env, or raises the Miss."""
        env = self.bind(args)
        for rec, closed in self.opaque:
            self._opaque(rec, env)
            self._guards(closed, env)
        self._guards(self.tail, env)
        return env


def _check_device_identity(tape, device):
    # the tape is bound to its device class (hosts fold the SM count into
    # values): a variant on a device of another class misses before any work
    for (k, traced), (_, here) in zip(
        tape.device_identity, ht._device_identity(device)
    ):
        if traced != here:
            raise ht.Miss(
                f"the tape was traced on a device with {k}={traced}; cuda:{device} has {k}={here}"
            )


class _Replay:
    """What a suite reads of a built variant, whichever backend serves it:
    `replay` (the outputs as a list, or a Miss by name), `try_replay`,
    `matches` (the inputs alone: the contract and the early guards), `calls`,
    `device`, `tape`, `wait_for_h2d`, `close`, `refused` (why the native line
    refused the tape, when the eager form serves it in its place)."""

    native = None  # the native entry, or None on the eager form
    refused = None

    def __init__(self, tape, fn, args, device):
        self.tape = tape
        self.fn = fn
        self.device = device if device is not None else tape.device.index
        _check_device_identity(tape, self.device)
        self.check = TapeCheck(tape, self.device)
        # a build at inputs that fail the tape's own guards is a miss before any
        # GPU work, on either backend
        self.check.check(args)
        self.calls = 0

    def matches(self, args):
        try:
            self.check.bind(tuple(args))
        except ht.Miss:
            return False
        return True

    def try_replay(self, args):
        """The replay, or None when this call cannot use the tape."""
        try:
            return self.replay(args)
        except ht.Miss:
            return None

    def __call__(self, args):
        return self.replay(args)

    def close(self):
        pass


class EagerReplay(_Replay):
    """The tape's guards decide, eager executes: the replay of the stack's own
    installs, where no graph runtime is built. Its copies from a pinned input
    are eager's, asynchronous on the current stream: `wait_for_h2d` waits for
    that stream (the contract a caller rewriting a pinned buffer in place has
    with every replay)."""

    graph = None
    _last_event = None

    def replay(self, args):
        args = tuple(args)
        self.check.check(args)
        out = self.fn(*args)
        self.calls += 1
        return [out] if isinstance(out, torch.Tensor) else list(out)

    def wait_for_h2d(self):
        torch.cuda.current_stream(self.device).synchronize()


class NativeReplay(_Replay):
    """One variant of the native replay prepared from the tape (no trace, no
    warm-up: nothing of `fn` runs; no planned arena), the tape's guards checked
    in Python beside the runtime's predicate: the two must agree on every call.
    A miss the tape does not decide (a closed region's node chain, E28, or a
    refused template) is the native entry's, with its text."""

    def __init__(self, tape, fn, args, device, staging_depth):
        super().__init__(tape, fn, args, device)
        module = native_replay_module()

        class OneVariant(module.HostTraceReplay):
            # a call the variant does not serve is a Miss by name (a
            # TopologyMiss carrying the tape when only the call's node chain is
            # not this variant's), never a trace
            def _miss(entry, call_args, tape=None):
                why = entry._pending_why or entry._why_or_contract()
                refused = [r for r in entry.refused.values() if r not in why]
                if refused:
                    why = "; ".join([why, *refused])
                if tape is not None:
                    raise ht.TopologyMiss(why, tape)
                raise ht.Miss(why)

        # no planned arena (a memory optimization with a suite of its own,
        # test_hosttrace_arena.py): the runtime owns the buffers, and a variant's
        # class is its tape's guards and node chains alone
        self.native = OneVariant(
            fn,
            args,
            tape=tape,
            staging_depth=staging_depth,
            device=self.device,
            arena=False,
        )

    @property
    def lowered(self):
        return self.native.lowered

    @property
    def graph(self):
        # the prepared capture's raw (graph, exec) handles
        return self.native.lowered.capture_handles

    @property
    def _last_event(self):
        return self.native._last_event

    def replay(self, args):
        args = tuple(args)
        try:
            self.check.check(args)
        except ht.Miss as miss:
            # the predicate must agree: a call the tape's guards refuse is a
            # native miss too, and its text names the same reason
            try:
                out = self.native(*args)
            except ht.Miss as native:
                raise ht.Miss(str(miss)) from native
            raise AssertionError(
                f"the native replay served a call the tape's guards refuse ({miss}); outputs {type(out).__name__}"
            ) from miss
        out = self.native(*args)
        self.calls += 1
        return [out] if isinstance(out, torch.Tensor) else list(out)

    def wait_for_h2d(self):
        self.native.wait_for_h2d()

    def close(self):
        self.native.close()


def wait_for_h2d(pinned):
    """Wait until every replay's copy from this pinned buffer, by any variant,
    has read it: a caller rewriting a buffer several variants read calls this
    first (the native replay's module-level wait; the eager form's copies are
    on the current stream)."""
    module = native_replay_module()
    if module is not None:
        module.wait_for_h2d(pinned)
    torch.cuda.current_stream().synchronize()


_STATS: dict = {}
_STATS_LOCK = threading.Lock()


def _test_id():
    for frame in inspect.stack():
        self = frame.frame.f_locals.get("self")
        if isinstance(self, unittest.TestCase):
            return f"{type(self).__name__}.{self._testMethodName}"
    return "<no test>"


def _note(key, value=1):
    with _STATS_LOCK:
        row = _STATS.setdefault(_test_id(), {"native": 0, "eager": 0, "refused": []})
        if key == "refused":
            row["refused"].append(value)
        else:
            row[key] += value


def build(tape, fn, args, device=None, *, staging_depth=2, backend=None):
    """The variant of `tape` a suite replays: prepared on the native replay
    where it is built (the integration line), else the tape evaluated in Python
    with eager as the executor; `backend` names one for a test that needs it.
    Nothing of `fn` runs here. A build at inputs outside the tape's class is a
    Miss; a tape the native line refuses by name falls back to the eager form
    (`variant.refused` holds the reason) unless HOST_TRACE_REPLAY=native."""
    args = tuple(args)
    backend = backend or replay_backend()
    if backend == "native":
        module = native_replay_module()
        from torch._inductor.runtime.cudagraph_launch_association import (
            UnsupportedCapture,
        )

        try:
            if torch.cuda.get_allocator_backend() == "cudaMallocAsync":
                # the native replay prepares under the caching allocator (the
                # graph's pool is its); the cudaMallocAsync backend records its
                # own allocation nodes into the preparation's capture, which the
                # preparation refuses as an internal error, not by name
                # (integration-09, reported to the runtime team)
                raise UnsupportedCapture(
                    "the native replay prepares under the caching allocator; the "
                    "cudaMallocAsync allocator backend records allocation nodes into "
                    "the preparation's capture (declined)"
                )
            variant = NativeReplay(tape, fn, args, device, staging_depth)
        except ht.Miss:
            raise
        except (UnsupportedCapture, ValueError) as e:
            # the lowering or the runtime declines by name (UnsupportedCapture),
            # or the runtime's registry refuses what it cannot hold (ValueError:
            # a region whose call launches nothing)
            if _BACKEND == "native":
                raise ht.Declined(f"the native replay refuses this tape: {e}") from e
            _note("refused", f"{type(e).__name__}: {e}")
            variant = EagerReplay(tape, fn, args, device)
            variant.refused = str(e)
            return variant
        _note("native")
        if TEST_CUDA_PYTHON_BINDINGS:
            # every graph a suite prepares is read back once: no memset node of
            # it is disabled (assert_no_disabled_memset)
            states = exec_node_states(variant.graph)
            off = [i for i, kind, on in states if kind == "memset" and on is False]
            if off:
                raise AssertionError(f"disabled memset node(s) {off}: {states}")
        del module
        return variant
    _note("eager")
    return EagerReplay(tape, fn, args, device)


def make_entry(fn, **kw):
    """`Entry` over the shared replay: its variants are `build`'s."""
    return ht.Entry(fn, build_variant=lambda tape, args: build(tape, fn, args), **kw)


def _report():
    if not _STATS:
        return
    native = sum(1 for row in _STATS.values() if row["native"])
    eager = sum(1 for row in _STATS.values() if row["eager"] and not row["native"])
    refused = sum(1 for row in _STATS.values() if row["refused"])
    print(
        f"host_trace replay: {len(_STATS)} tests built variants, {native} on the native replay, {eager} on the eager form only, {refused} with a tape the native line refused by name ({replay_backend()} backend)",
        file=sys.stderr,
        flush=True,
    )
    path = os.environ.get("HOST_TRACE_REPLAY_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"argv": sys.argv, "tests": _STATS}) + "\n")


atexit.register(_report)


# ---- the siblings in ordinary mode


class EntryMode(TorchDispatchMode):
    """Run a function through the traced entries in ordinary mode, as the
    trace mode routes it: an op with a traced sibling runs that sibling, every
    other op runs as usual, a composite decomposed, or its eager body run under
    the mode. The test seam for the siblings' entry contract outside a trace
    (their declines and eager's own checks); nothing in the stack uses it."""

    @classmethod
    def _should_skip_dynamo(cls):
        return False

    def __init__(self):
        super().__init__()
        self.entering: list = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        entry = ht._TRACED_ENTRIES.get(func)
        # the entry stands in for a CUDA op; a CPU scalar operand (a wrapped
        # Python number, a 0-dim CPU tensor) rides along as in the trace; a
        # closed op's entry serves only the case the trace routed to it (the
        # closed ops exist from the gemm commit on)
        closed = getattr(ht, "_CLOSED_OPS", frozenset())
        if (
            entry is not None
            and (func not in closed or ht._outer_product_bmm(func, args))
            and all(
                a.is_cuda or (a.is_cpu and a.dim() == 0)
                for a in args
                if isinstance(a, torch.Tensor)
            )
        ):
            if func in self.entering:
                raise ht.Declined(
                    f"host_trace: the traced entry for {func} dispatched {func} itself; "
                    "an entry runs its sibling host or declines (declined)"
                )
            self.entering.append(func)
            try:
                with self:
                    return entry(*args, **kwargs)
            finally:
                self.entering.pop()
        if func in ht._TRACEABLE:
            # a converted CUDA host: the mode stays on for the ops it calls
            # (a copy of a non-contiguous input), as in the trace
            with self:
                return func.redispatch(
                    C.DispatchKeySet(C.DispatchKey.CUDA), *args, **kwargs
                )
        # a composite the trace ran as eager's own body (E38) runs it here
        # too, under the mode, so its pieces take the entries the tape
        # describes; a view, an allocation or an entry's op on operands the
        # entry does not take runs as usual, as the trace never reached its
        # fallback for those
        body = None
        if entry is None and func not in ht._VIEW_OPS and func not in ht._ALLOC_OPS:
            body = ht._explicit_body_key(func, ht._key_below(args, kwargs))
        with self:
            if body is not None:
                return func.redispatch(body, *args, **kwargs)
            r = func.decompose(*args, **kwargs)
        if r is not NotImplemented:
            return r
        return func(*args, **kwargs)


# ---- reading a capture's nodes


def _raw_graph(graph):
    # a CUDAGraph, or the native replay's (graph, exec) handles
    return graph[0] if isinstance(graph, tuple) else graph.raw_cuda_graph()


def _raw_exec(graph):
    return graph[1] if isinstance(graph, tuple) else graph.raw_cuda_graph_exec()


def exec_node_states(graph):
    """(index, kind, enabled) over a graph in cudaGraphGetNodes order, the
    state read back from the exec with cudaGraphNodeGetEnabled; kernel, memset
    and memcpy nodes carry one, any other kind None."""
    from cuda.bindings import runtime as cudart

    from torch.cuda._utils import _check_cuda_bindings as check

    raw, exe = _raw_graph(graph), _raw_exec(graph)
    count = check(cudart.cudaGraphGetNodes(raw, 0))[1]
    nodes = check(cudart.cudaGraphGetNodes(raw, count))[0] if count else []
    kinds = {
        cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel: "kernel",
        cudart.cudaGraphNodeType.cudaGraphNodeTypeMemset: "memset",
        cudart.cudaGraphNodeType.cudaGraphNodeTypeMemcpy: "memcpy",
    }
    states = []
    for i, node in enumerate(nodes):
        kind = kinds.get(check(cudart.cudaGraphNodeGetType(node)))
        enabled = (
            bool(check(cudart.cudaGraphNodeGetEnabled(exe, node))) if kind else None
        )
        states.append((i, kind or "other", enabled))
    return states


def capture_graph(fn):
    """fn once eagerly, then under stream capture on a side stream that waits
    for the current one (where the caller made the inputs); the graph kept so
    its nodes can be read."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(g, stream=stream, capture_error_mode="relaxed"):
            fn()
    stream.synchronize()
    return g


def graph_nodes(graph):
    """The kernel and memset nodes of a graph in creation order, read through
    the driver (the closed regions' harvest reader): kernels as (name, grid,
    block, smem, image), memsets as (dst, bytes, value); a memcpy node is
    passed over (graph_functions reads it)."""
    raw = _raw_graph(graph)
    kernels, memsets = [], []
    for n in C._host_trace_harvest_nodes(raw):
        if n["kind"] == "memset":
            memsets.append((n["dst"], n["elem"] * n["width"], n["value"]))
        elif n["kind"] == "kernel":
            kernels.append(
                (n["name"], tuple(n["grid"]), tuple(n["block"]), n["smem"], n["image"])
            )
    return kernels, memsets


def graph_functions(graph):
    """The device function of every kernel node of a graph, the (CUfunction,
    CUkernel) handles of cuGraphKernelNodeGetParams, in cuGraphGetNodes order;
    a memcpy node as the string "memcpy", a memset node as "memset". The driver
    API: the runtime bindings are another cudart instance, where torch's
    kernels are not registered."""
    from cuda.bindings import driver as drv

    from torch.cuda._utils import _check_cuda_bindings as check

    raw = _raw_graph(graph)
    count = check(drv.cuGraphGetNodes(raw, 0))[1]
    nodes = check(drv.cuGraphGetNodes(raw, count))[0] if count else []
    kinds = {
        drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY: "memcpy",
        drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET: "memset",
    }
    out = []
    for node in nodes:
        kind = check(drv.cuGraphNodeGetType(node))
        if kind == drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            params = check(drv.cuGraphKernelNodeGetParams(node))
            out.append((int(params.func), int(params.kern)))
        else:
            out.append(kinds.get(kind, str(kind)))
    return out


def assert_eager_function_handles(test, real, args, entry=None, launches=None):
    """E36's gate for one call: the entry's own capture (when `entry` is given)
    and the tape of `real(*args)` hold at every node what eager's capture of
    the same call holds, so a replay launches eager's own kernel, not a twin:
    the entry's nodes by function handle; the tape's launches by kernel name,
    and the native replay's prepared graph by function handle where that
    replay is built. Returns eager's node list."""
    eager_graph = capture_graph(lambda: real(*args))
    eager = graph_functions(eager_graph)
    if entry is not None:
        test.assertEqual(graph_functions(capture_graph(lambda: entry(*args))), eager)
    tape = ht.trace(real, args)
    kernels, _ = graph_nodes(eager_graph)
    test.assertEqual([L["kernel"] for L in tape.launches], [k[0] for k in kernels])
    variant = build(tape, real, args)
    if variant.graph is not None:
        test.assertEqual(graph_functions(variant.graph), eager)
    if launches is not None:
        test.assertEqual(tape.num_launches, launches)
    return eager


def assert_no_disabled_memset(test, variant, what=""):
    """No memset node of a prepared graph is disabled: on driver 580.126.20 a
    kernel node behind a disabled memset node launches before the stream's
    prior work completes (a programmatic-dependent-launch pair in front makes
    it deterministic). Nothing in the stack disables a node (a graph holds
    exactly the capture's nodes); this reads the driver's view back. Returns
    the states, or None on the eager form (it prepares no graph)."""
    if variant.graph is None:
        return None
    states = exec_node_states(variant.graph)
    off = [i for i, kind, enabled in states if kind == "memset" and enabled is False]
    test.assertEqual(off, [], f"{what}: disabled memset node(s) {off}; nodes {states}")
    return states


class HostTraceTestCase(TestCase):
    def _capture_nodes(self, fn, memsets):
        # fn once eagerly, then under stream capture; the capture stream waits
        # for the current one, where the caller made the inputs
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(g, stream=stream, capture_error_mode="relaxed"):
                out = fn()
        stream.synchronize()
        kernels, memset_nodes = graph_nodes(g)
        return kernels, (memset_nodes if memsets else None), out

    def _capture(self, fn):
        # the kernel nodes one call produces: (name, grid, block, smem, image)
        kernels, _, out = self._capture_nodes(fn, memsets=False)
        return kernels, out

    def _capture_with_memsets(self, fn):
        # the kernel nodes and the memset nodes as (dst, bytes, value)
        return self._capture_nodes(fn, memsets=True)

    def _assert_bitwise(self, got, want, msg="outputs differ bitwise", *, stride=False):
        self.assertEqual(got.shape, want.shape)
        if stride:
            self.assertEqual(got.stride(), want.stride())
        self.assertEqual(got.dtype, want.dtype)
        self.assertTrue(torch.equal(bits(got), bits(want)), msg)

    def _replay_cases(
        self,
        fn,
        base_args,
        new_args_list,
        msg=None,
        atol=0,
        rtol=0,
        stride=False,
        **build_kw,
    ):
        # trace and build fn at base_args, then replay every case in turn: a
        # served replay's first output is compared with eager (bitwise unless a
        # tolerance is given; msg(args) names the case), a miss keeps its text
        tape = ht.trace(fn, base_args)
        variant = build(tape, fn, base_args, **build_kw)
        cases = []
        for args in new_args_list:
            try:
                out = variant.replay(args)
            except ht.Miss as e:
                cases.append(Case(None, str(e)))
                continue
            want = fn(*args)
            torch.cuda.synchronize()
            if atol or rtol:
                self.assertEqual(out[0].shape, want.shape)
                self.assertEqual(out[0], want, atol=atol, rtol=rtol)
            else:
                what = msg(args) if msg else "outputs differ bitwise"
                self._assert_bitwise(out[0], want, what, stride=stride)
            cases.append(Case(out, None))
        return tape, variant, cases
