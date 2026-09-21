"""Isolated symbolic execution of warmed dispatch and selected launcher code."""

from dataclasses import dataclass
from dis import get_instructions
from types import FunctionType, MappingProxyType, MethodType, SimpleNamespace
from collections.abc import Callable

import torch
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedCudaKernel
from torch._inductor.runtime.triton_heuristics import CachingAutotuner


class TraceViewDeclined(ValueError):
    pass


_AUTOTUNER_RUN = CachingAutotuner.run
_MULTI_RUN = MultiKernelCall.run
_MULTI_FILTER = MultiKernelCall._get_filtered_args
_MULTI_KERNELS = MultiKernelCall.kernels
_STATIC_RUN = StaticallyLaunchedCudaKernel.run


def _reject_slow_path(*args, **kwargs):
    raise TraceViewDeclined("Symbolic launch entered the unwarmed autotuner path")


def _clone(function, replacements):
    result = FunctionType(function.__code__, {**function.__globals__, **replacements},
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    result.__dict__.update(function.__dict__)
    return result


@dataclass(frozen=True)
class _Receipt:
    owners: tuple
    facts: tuple

    def check(self, current):
        if (len(self.owners) != len(current.owners)
                or any(before is not after for before, after in zip(self.owners, current.owners))
                or self.facts != current.facts):
            raise TraceViewDeclined("Warmed provider state changed during host preparation")


def _provider_receipt(provider):
    if (type(provider) is not CachingAutotuner or len(provider.launchers) != 1
            or not provider._cache_eligible or provider._plugins
            or type(provider.run) is not MethodType or provider.run.__func__ is not _AUTOTUNER_RUN):
        raise TraceViewDeclined("Expected an unchanged warmed CachingAutotuner")
    launcher, = provider.launchers
    cached = provider._cached_launcher
    if (type(launcher) is not FunctionType or type(cached) is not FunctionType
            or cached.__code__ is not launcher.__code__ or cached.config is not launcher.config
            or launcher.__closure__ or cached.__closure__
            or launcher.__defaults__ or cached.__defaults__
            or launcher.__kwdefaults__ or cached.__kwdefaults__
            or getattr(launcher, "_is_static", False) is not True):
        raise TraceViewDeclined("Expected the selected static Python launcher and its cached code")
    runner = launcher.__globals__.get("runner")
    kernel = runner.__self__ if type(runner) is MethodType else None
    if (type(kernel) is not StaticallyLaunchedCudaKernel or runner.__func__ is not _STATIC_RUN
            or kernel.device_agnostic
            or type(kernel.module) is not int or type(kernel.function) is not int
            or kernel.profile_scratch_size or kernel._has_tensordesc
            or provider.inductor_meta.get("host_tma_descriptor_args") or torch.version.hip):
        raise TraceViewDeclined("Expected a loaded CUDA kernel without host descriptors or profile scratch")
    cached_runner = cached.__globals__.get("runner")
    if cached is not launcher and (
        type(cached_runner) is not torch._C._FastCudaLauncher
        or getattr(cached, "_static_kernel_owner", None) is not kernel
    ):
        raise TraceViewDeclined("Cached launcher lost its selected static kernel owner")
    if {op.argval for op in get_instructions(cached) if op.opname == "LOAD_GLOBAL"} - {"runner"}:
        raise TraceViewDeclined("Launcher uses a host global outside the demonstrated trace boundary")
    config = launcher.config
    if config.pre_hook is not None or getattr(config, "ir_override", None) is not None:
        raise TraceViewDeclined("Selected configuration has an unsupported host hook")
    config_values = config.all_kwargs()
    if any(type(value) not in (int, bool, str, type(None)) for value in config_values.values()):
        raise TraceViewDeclined("Selected configuration contains unsupported mutable values")
    results = [result for result in provider.compile_results
               if result.kernel is kernel and result.config is config]
    if len(results) != 1:
        raise TraceViewDeclined("Selected launcher has no unique compiled kernel/configuration owner")
    scope = _AUTOTUNER_RUN.__globals__
    debug_guard = scope["get_active_debug_mode"]
    return _Receipt(
        (provider, provider.run.__func__, provider.run.__func__.__code__, launcher, launcher.__code__,
         cached, cached.__code__, runner, runner.__func__, runner.__func__.__code__,
         cached_runner, kernel, config, results[0],
         kernel.cudagraph_formal_args, getattr(launcher, "_cudagraph_arg_info", None),
         scope["autograd_profiler"], debug_guard, getattr(debug_guard, "__code__", None)),
        (tuple(sorted(config_values.items())), kernel.module, kernel.function, kernel.hash,
         kernel.arg_tys, kernel.num_warps, kernel.shared, kernel.num_ctas, tuple(kernel.arg_names),
         tuple(kernel.full_constexprs), kernel.has_global_scratch, kernel.has_profile_scratch,
         kernel.global_scratch_size, kernel.global_scratch_align, kernel.profile_scratch_size,
         launcher.cache_hash, cached.cache_hash),
    )


def _dispatch_receipt(carrier):
    if (type(carrier) is not MultiKernelCall or type(carrier.picked_kernel) is not int
            or type(carrier._kernels) is not list or not carrier._recorded
            or not 0 <= carrier.picked_kernel < len(carrier._kernels)
            or any(type(child) is not CachingAutotuner for child in carrier._kernels)
            or type(carrier.run) is not MethodType or carrier.run.__func__ is not _MULTI_RUN
            or type(carrier._get_filtered_args) is not MethodType
            or carrier._get_filtered_args.__func__ is not _MULTI_FILTER
            or MultiKernelCall.kernels is not _MULTI_KERNELS
            or type(carrier.arg_index) is not dict
            or set(carrier.arg_index) != set(range(len(carrier._kernels)))):
        raise TraceViewDeclined("Expected an unchanged warmed fixed-choice MultiKernelCall")
    virtualized = _MULTI_FILTER.__globals__["V"]
    if getattr(virtualized.graph, "cpp_wrapper", False) is not False:
        raise TraceViewDeclined("Multi-kernel tracing requires the ordinary Python-wrapper context")
    indices = []
    for index in range(len(carrier._kernels)):
        parts = carrier.arg_index[index]
        if (type(parts) is not list or not parts
                or any(type(part) is not slice or type(part.start) is not int
                       or type(part.stop) is not int or not 0 <= part.start <= part.stop
                       or part.step is not None for part in parts)):
            raise TraceViewDeclined("Unsupported ordinary multi-kernel argument slices")
        indices.append(tuple(parts))
    return _Receipt(
        (carrier, carrier.run.__func__, carrier.run.__func__.__code__,
         carrier._get_filtered_args.__func__, carrier._get_filtered_args.__func__.__code__,
         _MULTI_KERNELS, _MULTI_KERNELS.fget, _MULTI_KERNELS.fget.__code__, virtualized, *carrier._kernels),
        (carrier.picked_kernel, tuple(indices)),
    )


@dataclass(frozen=True)
class _AutotunerState:
    _cached_launcher: FunctionType


@dataclass(frozen=True)
class _MultiState:
    _kernels: tuple
    picked_kernel: int
    arg_index: MappingProxyType
    _filter: FunctionType
    _recorded: bool = True

    kernels = _MULTI_KERNELS

    def _get_filtered_args(self, args, index):
        return self._filter(self, args, index)


@dataclass(frozen=True)
class _ChildCall:
    function: FunctionType
    state: _AutotunerState

    def run(self, *args, stream, **kwargs):
        scope = self.function.__globals__
        if (kwargs or scope["autograd_profiler"]._is_profiler_enabled
                or scope["get_active_debug_mode"]()):
            raise TraceViewDeclined("Symbolic launch requires the warmed fast dispatch path")
        return self.function(self.state, *args, stream=stream)


@dataclass(frozen=True)
class TraceView:
    carrier: object
    providers: tuple[CachingAutotuner, ...]
    _provider_receipts: tuple[_Receipt, ...]
    _dispatch: _Receipt | None
    _call: Callable

    def check(self):
        if self._dispatch is not None:
            self._dispatch.check(_dispatch_receipt(self.carrier))
        for provider, receipt in zip(self.providers, self._provider_receipts):
            receipt.check(_provider_receipt(provider))

    def run(self, *args, stream):
        self.check()
        try:
            return self._call(*args, stream=stream)
        finally:
            self.check()


def make_trace_view(carrier, sink):
    """Bind sink(provider, grid, stream, explicit_args) without changing live dispatch."""
    dispatch = None if type(carrier) is CachingAutotuner else _dispatch_receipt(carrier)
    provider = carrier if dispatch is None else carrier._kernels[carrier.picked_kernel]
    receipt = _provider_receipt(provider)
    cached = provider._cached_launcher

    def record(grid_x, grid_y, grid_z, stream, *arguments):
        return sink(provider, (grid_x, grid_y, grid_z), stream, arguments)

    launcher = _clone(cached, {"runner": record})
    state = _AutotunerState(launcher)
    # A profiler/debug transition must decline instead of reaching autotuning or allocation.
    function = _clone(_AUTOTUNER_RUN, {"triton": SimpleNamespace(set_allocator=_reject_slow_path)})
    child = _ChildCall(function, state)
    call = child.run
    if dispatch is not None:
        children = tuple(child if index == carrier.picked_kernel else None
                         for index in range(len(carrier._kernels)))
        indices = MappingProxyType({index: tuple(parts) for index, parts in carrier.arg_index.items()})
        virtualized = SimpleNamespace(graph=SimpleNamespace(cpp_wrapper=False))
        filtering = _clone(_MULTI_FILTER, {"V": virtualized})
        multi = _MultiState(children, carrier.picked_kernel, indices, filtering)
        call = MethodType(_clone(_MULTI_RUN, {}), multi)
    result = TraceView(carrier, (provider,), (receipt,), dispatch, call)
    result.check()
    return result
