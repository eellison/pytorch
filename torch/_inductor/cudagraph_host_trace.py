"""
Host-traced CUDA graphs for Inductor output (config.triton.cudagraph_host_trace).

Replaces cudagraph trees for a compiled graph when the flag is set; imported
only then.
"""

from __future__ import annotations

import types
from typing import Any, NoReturn, TYPE_CHECKING

import torch
from torch._dynamo.utils import counters
import torch._inductor.inductor_prims  # noqa: F401 (prims.inductor_seeds)
from torch._inductor import config
from torch._inductor.codegen.multi_kernel import (
    MultiKernelCall,
    SizeHintMultiKernelCall,
)
from torch._inductor.cudagraph_utils import (
    CUDAGraphPolicy,
    log_cudagraph_skip_and_bump_counter,
)
from torch._inductor.runtime.runtime_utils import assert_tensor_metadata
from torch._inductor.runtime.triton_compat import ASTSource
from torch._inductor.runtime.triton_heuristics import (
    CachingAutotuner,
    StaticTritonCompileResult,
    TritonCompileResult,
)
from torch._inductor.utils import ALIGNMENT, clone_preserve_strides
from torch._logging import trace_structured
from torch.cuda._host_trace import Declined
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.cuda._host_trace_launch import _probe_address
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.cuda._host_trace_tape import (
    _hint,
    _TracedTensor,
    current_trace,
    TrustedInputs,
)
from torch.cuda._host_trace_triton import triton_abi
from torch.cuda._host_trace_triton_launch import owned_module, record_launch
from torch.utils._debug_mode import get_active_debug_mode


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.cuda._host_trace_tape import _Trace


def _log_decline(why: str) -> None:
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "cudagraph_host_trace_decline",
            "encoding": "string",
        },
        payload_fn=lambda: why,
    )


class HostTracePolicy(CUDAGraphPolicy):
    def cudagraphify(
        self,
        model: Callable[..., Any],
        example_inputs: Sequence[Any],
        static_input_idxs: Sequence[int],
        *,
        device_index: int,
        is_backward: bool,
        is_inference: bool,
        graph_inputs: TrustedInputs | str | None,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        # the flag is not in the FX graph cache key
        if graph_inputs is None:
            why = "the graph is a cache entry compiled without triton.cudagraph_host_trace"
        elif isinstance(graph_inputs, str):
            why = graph_inputs
        else:
            try:
                # static_input_idxs promises stable addresses only where trees
                # checks them; the tape reads every input's address
                return _Installation(model, graph_inputs, static_input_idxs)
            except _Uncaptured:
                return model
            except _NotInstalled as e:
                why = str(e)
        counters["inductor"]["cudagraph_host_trace_declined"] += 1
        why = f"host_trace: {why}; the graph runs uncaptured"
        _log_decline(why)
        # a graph never captured is a skip, as for trees: counted, and an
        # error under cudagraph_or_error
        log_cudagraph_skip_and_bump_counter(why)
        return model


class _NotInstalled(Exception):
    pass


class _Uncaptured(Exception):
    """A graph with nothing for a trace to key on: it runs uncaptured, and
    that is not a decline."""


def _expr(v: Any) -> Any:
    return v.node.expr if isinstance(v, torch.SymInt) else int(v)


def graph_inputs(example_inputs: Sequence[Any]) -> TrustedInputs | str:
    """The compiled graph's TrustedInputs, or why it has none."""
    try:
        return _trusted_inputs(example_inputs)
    except _NotInstalled as e:
        return str(e)


def _trusted_inputs(example_inputs: Sequence[Any]) -> TrustedInputs:
    # what Dynamo guarded and Inductor compiled for: only its symbols are
    # symbols of the trace
    layouts: list[Any] = []
    symints: list[torch.SymInt] = []
    for i, e in enumerate(example_inputs):
        if isinstance(e, torch.Tensor):
            layouts.append((tuple(map(_expr, e.shape)), tuple(map(_expr, e.stride()))))
            symints += [v for v in (*e.shape, *e.stride()) if isinstance(v, torch.SymInt)]
        elif isinstance(e, (int, torch.SymInt)) and not isinstance(e, bool):
            layouts.append(_expr(e))
            symints += [e] if isinstance(e, torch.SymInt) else []
        elif isinstance(e, torch.Generator) and e.device.type == "cuda":
            # graphsafe RNG's state (checkpointed RNG): a constant of the trace
            layouts.append(None)
        elif isinstance(e, torch.Generator):
            raise _NotInstalled(f"input {i} is a Generator on {e.device}")
        else:
            raise _NotInstalled(f"input {i} is a {type(e).__name__}")
    ranges = {
        s: v.node.shape_env.var_to_range[s]
        for v in symints
        for s in v.node.expr.free_symbols
    }
    return TrustedInputs(tuple(layouts), ranges)


def _namespace(module: dict[str, Any]) -> dict[str, Any]:
    """A generated module's namespace with its functions cloned into it, each
    kernel a proxy and each C++ wrapper helper a seam."""
    scope = dict(module)
    for k, v in module.items():
        if isinstance(v, types.FunctionType) and v.__globals__ is module:
            scope[k] = types.FunctionType(
                v.__code__, scope, v.__name__, v.__defaults__, v.__closure__
            )
        elif isinstance(v, CachingAutotuner):
            scope[k] = _InductorKernel(v)
        elif isinstance(v, MultiKernelCall):
            scope[k] = _InductorMultiKernel(v)
        elif (seam := _seam(k, v)) is not None:
            scope[k] = seam
    return scope


def _harvest_budget() -> int:
    return config.triton.cudagraph_host_trace_harvest_budget


# one per family and process: its bindings serve every compiled graph
_PROVIDERS = {
    "blas": HarvestProvider(budget=_harvest_budget),
    "attention": HarvestProvider(("attention",), _harvest_budget),
    "conv": HarvestProvider(("conv",), _harvest_budget),
    "rng": HarvestProvider(("rng",), _harvest_budget, (torch.ops.prims.inductor_seeds.default,)),
    "reduce": HarvestProvider(("reduce",), _harvest_budget),
}


class _Installation:
    """A compiled graph's `call` served by host-trace replay: its generated
    code runs, over a clone of its module's namespace, under the trace. The
    module itself stays shared and unmodified.

    The module's CUDA tensors (Inductor's constants, frozen parameters) are
    arguments after the graph's, rebound in the namespace for the trace."""

    def __init__(
        self,
        model: Callable[..., Any],
        inputs: TrustedInputs,
        static_input_idxs: Sequence[int],
    ) -> None:
        module = getattr(getattr(model, "__func__", None), "__globals__", {})
        if module.get("call") != model:
            raise _NotInstalled(f"{model!r} is not a generated module's call")
        scope = _namespace(module)
        names = [
            k for k, v in module.items() if isinstance(v, torch.Tensor) and v.is_cuda
        ]
        self._constants = tuple(module[k] for k in names)
        if not names and not any(isinstance(v, tuple) for v in inputs.layouts):
            # e.g. a causal mask built from constants after a graph break
            raise _Uncaptured
        n = len(inputs.layouts)
        layouts = (*inputs.layouts, *_trusted_inputs(self._constants).layouts)
        trusted = TrustedInputs(layouts, inputs.ranges)
        partitions = []
        for p in model.__self__.partitions:
            name = getattr(p, "__name__", None)
            if module.get(name) is not p:
                raise _NotInstalled(f"partition {p!r} is not a function of the module")
            partitions.append(scope[name])
        call = types.FunctionType(model.__func__.__code__, scope)
        self_ = types.SimpleNamespace(partitions=partitions)
        # the positions of the graph's None outputs (a backward's gradients of
        # inputs that need none): fixed by the graph, and not the tape's
        self._nones: list[int] = []

        def tensors(*args: Any) -> list[Any]:
            scope.update(zip(names, args[n:]))
            try:
                out = call(self_, list(args[:n]))
            finally:
                scope.update(zip(names, self._constants))
            self._nones = [i for i, o in enumerate(out) if o is None]
            return [o for o in out if o is not None]

        knobs = config.triton
        opaque = tuple(_PROVIDERS[f] for f in knobs.cudagraph_host_trace_harvest)
        memory = knobs.cudagraph_host_trace_replay_memory
        # a boxed call hands over its list's references; a parameter's
        # (static_input_idxs) or constant's tensor stays held
        freed = set(range(n)) - set(static_input_idxs)
        self.replay = HostTraceReplay(
            tensors, trusted=trusted, opaque=opaque, memory=memory, freed_arguments=freed
        )
        self._variants = 0
        self._uncaptured = 0
        self._structural = 0
        self._reasons = 0
        self._guards = 0

    def __call__(self, new_inputs: list[Any]) -> Any:
        replay = self.replay
        traces = replay.traces
        # a replay drops a list's inputs at their last use, as the graph's own
        # call does; compile_fx_inner's boxed call also takes a tuple
        if isinstance(new_inputs, list):
            args = [*new_inputs, *self._constants]
            new_inputs.clear()
            out = list(replay.call_boxed(args))
        else:
            out = list(replay(*new_inputs, *self._constants))
        if replay.traces != traces:
            self._account()
        for i in self._nones:
            out.insert(i, None)
        return out

    def _account(self) -> None:
        replay = self.replay
        variants = len(replay.variants)
        if variants == self._variants and replay.uncaptured == self._uncaptured:
            counters["inductor"]["cudagraph_host_trace_declined"] += 1
        self._variants = variants
        self._uncaptured = replay.uncaptured
        skip = replay.structural != self._structural
        self._structural = replay.structural
        reasons = replay.declines[self._reasons :]
        for why in reasons:
            _log_decline(why)
        self._reasons = len(replay.declines)
        # the guards of real host decisions (an input's alignment, a
        # size-based kernel choice), which select the variant at a replay
        counters["inductor"]["cudagraph_host_trace_dispatch_guards"] += (
            replay.guards - self._guards
        )
        self._guards = replay.guards
        if skip:
            # under trust a structural decline is the graph's: it runs uncaptured
            log_cudagraph_skip_and_bump_counter("; ".join(reasons) or "host_trace: the trace declined")


def _traced_or(original: Callable[..., Any], traced: Callable[..., Any]) -> Any:
    def seam(*args: Any) -> Any:
        tr = current_trace()
        return original(*args) if tr is None else traced(tr, *args)

    return seam


def _skip(tr: _Trace, *args: Any) -> None:
    # a restatement of the inputs' validity, which the caller vouches for
    return None


def _empty_strided(tr: _Trace, size: Any, stride: Any, dtype: Any) -> Any:
    return torch.empty_strided(size, stride, dtype=dtype, device=tr.device)


def _empty_strided_cpu(tr: _Trace, size: Any, stride: Any, dtype: Any) -> Any:
    return tr.host_buffer(size, stride, dtype)


def _reinterpret_tensor(tr: _Trace, t: Any, size: Any, stride: Any, off: Any = 0) -> Any:
    if not isinstance(t, _TracedTensor):
        return torch._C._dynamo.guards._reinterpret_tensor(t, size, stride, off)
    return torch.as_strided(t, size, stride, t._sym_offset + off)


def _copy_if_misaligned(tr: _Trace, t: Any) -> Any:
    # the only runtime repair of the inputs Inductor emits: a dispatch guard
    if not isinstance(t, _TracedTensor):
        return torch._C._dynamo.guards.copy_if_misaligned(t)
    if bool(_probe_address(t) % ALIGNMENT == 0):
        return t
    return clone_preserve_strides(t)


def _refuse(name: str) -> Callable[..., Any]:
    def traced(tr: _Trace, *args: Any) -> NoReturn:
        raise tr.decline(f"{name} is not traced")

    return traced


# torch._C._dynamo.guards' helpers by their C++ names; the C++ aborts on a
# traced tensor, so any other of them refuses under a trace
_GUARDS_SEAMS = {
    "assert_size_stride": _skip,
    "assert_size_stride_grouped": _skip,
    "assert_alignment": _skip,
    "_empty_strided_cuda": _empty_strided,
    "_empty_strided_cpu": _empty_strided_cpu,
    "_reinterpret_tensor": _reinterpret_tensor,
    "copy_if_misaligned": _copy_if_misaligned,
}
# a host buffer or a pool alias the tape cannot bound
_REFUSED = ("alloc_from_pool", "empty_strided_p2p")


def _seam(name: str, v: Any) -> Any:
    if getattr(v, "__module__", None) == "torch._C._dynamo.guards" and callable(v):
        key = v.__name__
        return _traced_or(v, _GUARDS_SEAMS.get(key) or _refuse(key))
    if v is assert_tensor_metadata:
        # a fallback op's output check; it calls assert_size_stride directly
        return _traced_or(v, _skip)
    if name in _REFUSED:
        return _traced_or(v, _refuse(name))
    if name.startswith("cpp_fused") and callable(v):
        # a C++ kernel of host buffers: a host step
        return _traced_or(v, lambda tr, *args: tr.host_step(v, args, {}, name))
    return None


class _InductorKernel:
    """Stands in for a CachingAutotuner in an installed wrapper's namespace.
    Under a host trace, `run` records the tuned launcher's launch as a
    KernelLaunch; otherwise it is the autotuner's."""

    def __init__(self, autotuner: CachingAutotuner) -> None:
        self.autotuner = autotuner

    def run(self, *args: Any, stream: int, **kwargs: Any) -> Any:
        tr = current_trace()
        if tr is None:
            return self.autotuner.run(*args, stream=stream, **kwargs)
        try:
            self._record(tr, args, stream, kwargs)
        except Declined as e:
            if tr.declined is None:
                tr.declined = e
            raise
        except Exception as e:
            if torch.cuda._host_trace.raise_unexpected:
                raise
            name = self.autotuner.fn.__name__
            raise tr.decline(f"Inductor kernel {name} raised {e!r}") from e
        return None

    def _record(self, tr: _Trace, args: tuple, stream: int, kwargs: dict) -> None:
        autotuner = self.autotuner
        name = autotuner.fn.__name__
        meta = autotuner.inductor_meta

        def decline(why: str, retry: bool = False) -> NoReturn:
            e = tr.decline(f"Inductor kernel {name}: {why}")
            e.retry = retry
            raise e

        if kwargs:
            decline("keyword arguments are not traced")
        if type(autotuner) is not CachingAutotuner:
            decline(f"{type(autotuner).__name__} is not traced")
        if autotuner._plugins or not autotuner._cache_eligible:
            decline("autotuner plugins, interpret mode and launch dumps are not traced")
        if torch.autograd.profiler._is_profiler_enabled or get_active_debug_mode():
            decline("the profiler and debug mode run at the trace, never at a replay")
        if meta.get("host_tma_descriptor_args"):
            decline("host TMA descriptors are not traced")
        if tr.stream is None or stream != tr.stream.cuda_stream:
            decline("launched on a stream other than the trace's")
        # the first call tunes (untraced) and a later call traces
        launchers = autotuner.launchers
        if len(launchers) != 1:
            decline("not tuned yet", retry=True)
        config = launchers[0].config
        if meta.get("coordinate_descent_tuning") and not getattr(
            config, "found_by_coordesc", False
        ):
            decline("coordinate descent has not run yet", retry=True)
        if meta.get("combo_tuning_groups") and not getattr(
            config, "found_by_combo_autotune", False
        ):
            decline("combo-kernel tuning has not run yet", retry=True)

        result = getattr(launchers[0], "compile_result", None)
        if result is None:
            decline("the tuned launcher has no compile result")
        device = result.compile_meta.get("device")
        if device not in (None, tr.device.index):
            decline(f"compiled for cuda:{device}, not the trace's device")
        kernel: Any = result.kernel
        if type(result) is TritonCompileResult:
            src, metadata, function = kernel.src, kernel.metadata, kernel.function
        elif type(result) is StaticTritonCompileResult:
            # the compilation _precompile_config ran
            cm = result.compile_meta
            # pyrefly: ignore [not-callable]
            src = ASTSource(
                autotuner.fn, cm["signature"], cm["constants"], cm["configs"][0]
            )
            metadata = types.SimpleNamespace(
                num_warps=kernel.num_warps,
                shared=kernel.shared,
                num_ctas=1,
                launch_cooperative_grid=False,
                launch_pdl=False,
                global_scratch_size=kernel.global_scratch_size,
                profile_scratch_size=kernel.profile_scratch_size,
                tensordesc_meta=kernel.tensordesc_meta,
            )
            function = kernel.function
            if kernel.device_agnostic:
                function = kernel.functions.get(tr.device.index)
        else:
            decline(f"{type(result).__name__} is not traced")
        if function is None:
            decline("the kernel is not loaded on the trace's device")
        owner: Any = result
        if type(result) is TritonCompileResult:
            # the tape's own load (a static launcher keeps no cubin to load)
            owner = owned_module(kernel)
            function = owner.function
        abi = triton_abi(src, metadata)

        launcher = launchers[0]
        scope = launcher.__globals__
        hooks = [scope.get("launch_enter_hook"), scope.get("launch_exit_hook")]
        if any(getattr(h, "calls", h) for h in hooks):  # a HookChain or a callable
            decline("launch hooks run at the trace, never at a replay")
        code = launcher.__code__
        names = code.co_varnames[: code.co_argcount - 1]  # the def args, then stream
        if len(args) != len(names):
            decline(f"{len(args)} arguments; the launcher takes {len(names)}")
        values = dict(zip(names, args))
        # the specializations of the compilation: each a guard, or under
        # trusted inputs a fact Inductor proved, read at the hint
        for a in abi.args:
            v = values.get(a.name)
            if isinstance(v, _TracedTensor):
                v = _probe_address(v)
            if not isinstance(v, (int, torch.SymInt)):
                continue
            if tr.trusted is not None:
                v = _hint(v)
            if a.slot is None and not bool(v == a.constant):
                decline(f"argument {a.name} is {v}; compiled {a.constant!r}")
            if a.divisibility > 1 and not bool(v % a.divisibility == 0):
                decline(f"argument {a.name} is not a multiple of {a.divisibility}")

        # the launcher's own grid code, run on the values of the trace
        dims: list[Any] = []
        grid_launcher = types.FunctionType(
            code, scope | {"runner": lambda *a: dims.extend(a[:3])}
        )
        grid_launcher(*args, stream=stream)
        for axis, extent in enumerate(dims):
            if type(extent) not in (int, torch.SymInt):
                decline(f"grid axis {axis} is a {type(extent).__name__}")
        gx, gy, gz = dims
        record_launch(tr, name, int(function), owner, abi, values, (gx, gy, gz))


class _InductorMultiKernel:
    """Stands in for a MultiKernelCall: under a host trace, records the sub-
    kernel the call would run. A size-hint multi-kernel chooses by the
    shapes, so the trace guards each size, as its shape cache keys on them,
    and a replay selects the variant."""

    def __init__(self, call: MultiKernelCall) -> None:
        self.call = call

    def run(self, *args: Any, stream: int, **kwargs: Any) -> Any:
        tr = current_trace()
        call = self.call
        if tr is None:
            return call.run(*args, stream=stream, **kwargs)
        if type(call) is SizeHintMultiKernelCall:
            # SizeHintMultiKernelCall._select_kernel_by_shape, at guarded sizes
            key = tuple(tuple(int(v) for v in a.shape) for a in args if hasattr(a, "shape"))
            dists = [call._dist_heuristic(key, k) if k is not None else 2**62 for k in call._kernel_hints]
            picked = dists.index(min(dists))
        elif type(call) is MultiKernelCall and call.picked_kernel is not None:
            picked = call.picked_kernel
        else:
            e = tr.decline(f"{type(call).__name__} {call.multi_kernel_name} has not picked")
            e.retry = type(call) is MultiKernelCall
            raise e
        kernel = _InductorKernel(call.kernels[picked])
        return kernel.run(*call._get_filtered_args(args, picked), stream=stream, **kwargs)
