"""Shared ordinary-first metadata adapter, separate from replay preparation."""

from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import Any, TYPE_CHECKING

from torch._inductor.runtime.cudagraph_arg_mapping import (
    bind_fixed_grid, bind_grid_recipe, bind_launcher_arguments, bind_wrapper_arguments,
    BufferSource, CallArgument, FIXED_GRID_ARGUMENTS, IntExpr, KernelCallRecord, LauncherArgument,
)


if TYPE_CHECKING:
    from torch._inductor.runtime.cudagraph_multikernel import MultiKernelCallRecord
    from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedCudaKernel
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torch._inductor.runtime.triton_parameter_analysis import SelectedUserKernelFacts


@dataclass(frozen=True)
class _SelectedCall:
    record: KernelCallRecord
    kernel: "CachingAutotuner"
    selected: FunctionType
    cached: FunctionType
    module: "StaticallyLaunchedCudaKernel"
    function: int
    arguments: tuple[CallArgument, ...]
    scratch: tuple[bytes, ...]
    grid_recipe: tuple[IntExpr, IntExpr, IntExpr] | None = None
    user_facts: "SelectedUserKernelFacts | None" = None
    dispatch: Any = None
    source_kernel: Any = None
    source_record: "MultiKernelCallRecord | None" = None

    def check(self) -> None:
        from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
        from torch._inductor.runtime.cudagraph_multikernel import MultiKernelCallRecord
        from torch._inductor.runtime.triton_compat import autograd_profiler
        from torch._inductor.runtime.triton_parameter_analysis import user_kernel_controls_supported
        from torch.utils._debug_mode import get_active_debug_mode

        if self.dispatch is not None:
            try:
                self.dispatch.check()
            except ValueError as error:
                raise UnsupportedCapture(str(error)) from error
            if (type(self.source_record) is not MultiKernelCallRecord
                    or self.source_kernel is not self.dispatch.owner
                    or self.kernel is not self.dispatch.kernels[self.dispatch.picked]
                    or self.dispatch.argument_indices
                    != self.source_record.alternatives[self.dispatch.picked].argument_indices):
                raise UnsupportedCapture("Selected call lost its original dispatch provenance")
        if (
            len(self.kernel.launchers) != 1 or self.kernel.launchers[0] is not self.selected
            or self.kernel._cached_launcher is not self.cached
            or self.module.function != self.function
            or autograd_profiler._is_profiler_enabled or get_active_debug_mode()
            or (self.user_facts is None and not self.record.generated_template and self.grid_recipe is not None and (
                getattr(self.selected, "_cudagraph_grid_recipe", None) is not self.grid_recipe
                or getattr(self.cached, "_cudagraph_grid_recipe", None) is not self.grid_recipe
            ))
        ):
            raise UnsupportedCapture("Selected launcher changed during preparation")
        if self.record.generated_template and (
            self.kernel.inductor_meta.get("cudagraph_generated_template") is not True
            or tuple(self.kernel.inductor_meta.get("extra_launcher_args", ())) != FIXED_GRID_ARGUMENTS
            or tuple(self.kernel.inductor_meta.get("fixed_grid", ())) != FIXED_GRID_ARGUMENTS
            or getattr(self.selected, "_cudagraph_grid_args", None) != FIXED_GRID_ARGUMENTS
            or getattr(self.cached, "_cudagraph_grid_args", None) != FIXED_GRID_ARGUMENTS
            or bind_fixed_grid(self.record) != self.grid_recipe
        ):
            raise UnsupportedCapture("Selected template grid changed during preparation")
        if self.user_facts is not None and (
            self.module.cudagraph_user_facts is not self.user_facts
            or type(self.module._cudagraph_user_loaded) is not tuple
            or len(self.module._cudagraph_user_loaded) != 3
            or self.module._cudagraph_user_loaded[0] is not self.user_facts
            or self.module._cudagraph_user_loaded[1:] != (self.module.module, self.function)
            or self.module.cudagraph_formal_args is not self.user_facts.arguments
            or self.module.hash != self.user_facts.analysis.compiled_hash
            or self.kernel.fn.src != self.user_facts.source
            or bind_launcher_arguments(
                self.user_facts.arguments, self.module.arg_names,
                [*(arg.formal for arg in self.record.arguments), *FIXED_GRID_ARGUMENTS],
                [arg.formal for arg in self.user_facts.arguments if arg.abi_index is not None],
                grid_args=FIXED_GRID_ARGUMENTS,
            ) != getattr(self.selected, "_cudagraph_arg_info", None)
            or getattr(self.selected, "_cudagraph_grid_args", None) != FIXED_GRID_ARGUMENTS
            or getattr(self.cached, "_cudagraph_grid_args", None) != FIXED_GRID_ARGUMENTS
            or self.grid_recipe is not None and bind_fixed_grid(self.record) != self.grid_recipe
            or bind_wrapper_arguments(self.record, getattr(self.selected, "_cudagraph_arg_info", None)) != self.arguments
            or not user_kernel_controls_supported()
        ):
            raise UnsupportedCapture("Selected user-kernel facts changed during preparation")


def _multi_kernel_state(owner):
    from torch._inductor.codegen.multi_kernel import MultiKernelCall
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    if (type(owner) is not MultiKernelCall or type(owner.picked_kernel) is not int
            or type(owner._kernels) is not list or not 0 <= owner.picked_kernel < len(owner._kernels)
            or any(type(kernel) is not CachingAutotuner for kernel in owner._kernels)
            or type(owner.run) is not MethodType or owner.run.__func__ is not MultiKernelCall.run
            or type(owner.arg_index) is not dict or any(type(index) is not int for index in owner.arg_index)
            or set(owner.arg_index) != set(range(len(owner._kernels)))):
        raise ValueError("Expected a warmed, fixed-choice MultiKernelCall")
    arguments = []
    for index in range(len(owner._kernels)):
        slices = owner.arg_index[index]
        if (type(slices) is not list or not slices
                or any(type(part) is not slice or type(part.start) is not int or type(part.stop) is not int
                       or not 0 <= part.start <= part.stop or part.step is not None for part in slices)):
            raise ValueError("Unsupported multi-kernel argument selection")
        arguments.append(tuple((part.start, part.stop) for part in slices))
    run = owner.run.__func__
    return tuple(owner._kernels), owner.picked_kernel, tuple(arguments), run, run.__code__


@dataclass(frozen=True)
class WarmedMultiKernel:
    owner: object
    kernels: tuple[object, ...]
    picked: int
    arguments: tuple[tuple[tuple[int, int], ...], ...]
    run: object
    code: object

    @property
    def argument_indices(self):
        return tuple(index for start, stop in self.arguments[self.picked] for index in range(start, stop))

    def check(self):
        kernels, picked, arguments, run, code = _multi_kernel_state(self.owner)
        if (len(kernels) != len(self.kernels) or any(left is not right for left, right in zip(kernels, self.kernels))
                or picked != self.picked or arguments != self.arguments or run is not self.run or code is not self.code):
            raise ValueError("Warmed multi-kernel selection changed")


def resolve_warmed_kernel(carrier):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    if type(carrier) is CachingAutotuner:
        return carrier, None
    state = _multi_kernel_state(carrier)
    receipt = WarmedMultiKernel(carrier, *state)
    return receipt.kernels[receipt.picked], receipt


def _grid_product_key(expression):
    arguments = tuple(_grid_product_key(arg) for arg in expression.args)
    if expression.op == "multiply":
        arguments = tuple(sorted(arguments))
    return expression.op, expression.value, arguments


@dataclass(frozen=True)
class SelectedKernel:
    arguments: tuple[LauncherArgument, ...]
    grid_recipe: tuple[IntExpr, IntExpr, IntExpr]
    pointer_writes: tuple[tuple[str, bool], ...]
    arg_tys: str
    identity: object
    compiler_call: object | None = None
    dispatch: WarmedMultiKernel | None = None

    @classmethod
    def from_compiler_call(cls, call):
        if type(call) is not _SelectedCall:
            raise ValueError("Expected the checked compiler call from ordinary selection")
        call.check()
        recipe = call.grid_recipe
        if call.user_facts is None:
            provenance = call.kernel.inductor_meta.get("cudagraph_parameter_provenance", {})
            writes = tuple((name, row[1]) for name, row in provenance.items())
        else:
            recipe = bind_fixed_grid(call.record)
            writes = tuple((row.name, row.written) for row in call.user_facts.analysis.pointer_formals)
        if recipe is None:
            raise ValueError("Selected call has no supported compiler grid")
        result = cls(call.selected._cudagraph_arg_info, recipe, writes, call.module.arg_tys,
                     (call.kernel, call.selected, call.cached, call.module), call, call.dispatch)
        result.check()
        return result

    def check(self):
        if self.dispatch is not None:
            self.dispatch.check()
            if self.identity[0] is not self.dispatch.kernels[self.dispatch.picked]:
                raise ValueError("Selected kernel differs from its warmed dispatch winner")
        call = self.compiler_call
        if call is None:
            return
        if type(call) is not _SelectedCall:
            raise ValueError("Selected projection lost its compiler call")
        call.check()
        if call.user_facts is None:
            provenance = call.kernel.inductor_meta.get("cudagraph_parameter_provenance", {})
            writes = tuple((name, row[1]) for name, row in provenance.items())
            grid = call.grid_recipe
        else:
            writes = tuple((row.name, row.written) for row in call.user_facts.analysis.pointer_formals)
            grid = bind_fixed_grid(call.record)
        identities = (call.kernel, call.selected, call.cached, call.module)
        if (type(self.identity) is not tuple or len(self.identity) != len(identities)
                or any(left is not right for left, right in zip(self.identity, identities))
                or self.arguments is not call.selected._cudagraph_arg_info
                or self.arg_tys != call.module.arg_tys or self.pointer_writes != writes
                or self.grid_recipe != grid or self.dispatch is not call.dispatch):
            raise ValueError("Selected projection changed after ordinary selection")

    def user_pointer_effects(self):
        self.check()
        call = self.compiler_call
        if call is None or call.user_facts is None:
            raise ValueError("User effects require the original selected receipt")
        indices = {row.formal: row.source_arg_index for row in self.arguments if row.abi_index is not None}
        return tuple((row.name, indices[row.name], row.read, row.written)
                     for row in call.user_facts.analysis.pointer_formals)

    def bind_call(self, record, buffer_sources=None):
        self.check()
        if type(record) is not KernelCallRecord:
            raise ValueError("Expected a frontend compiler call record")
        bound = bind_wrapper_arguments(record, self.arguments)
        if record.grid_type == "FixedGrid":
            if self.compiler_call is None or (
                self.compiler_call.user_facts is None
                and (record.generated_template is not True
                     or self.compiler_call.kernel.inductor_meta.get("cudagraph_generated_template") is not True)
            ):
                raise ValueError("FixedGrid requires a checked compiler-call projection")
            grid = bind_fixed_grid(record)
            if grid is None or (grid != self.grid_recipe and (
                not record.generated_template
                or tuple(_grid_product_key(axis) for axis in grid)
                != tuple(_grid_product_key(axis) for axis in self.grid_recipe)
            )):
                raise ValueError("Frontend grid differs from its original boxed sources")
            grid = self.grid_recipe
        else:
            grid = bind_grid_recipe(record, self.arguments, self.grid_recipe)
        abi = "".join("O" if row.triton_type.startswith("*") else {"i32": "i", "i64": "l"}.get(
            row.triton_type, "?") for row in self.arguments if row.abi_index is not None)
        if bound is None or grid is None or abi != self.arg_tys:
            raise ValueError("Trace and selected ABI/grid correspondence differ")
        if self.compiler_call is None:
            return grid
        original = self.compiler_call.record
        if (record.occurrence != original.occurrence or record.kernel_global != original.kernel_global
                or record.formals != original.formals or record.grid_type != original.grid_type
                or record.generated_template != original.generated_template
                or record.constexprs != original.constexprs or len(record.arguments) != len(original.arguments)
                or type(buffer_sources) is not dict
                or any(type(left) is not BufferSource or type(right) is not BufferSource
                       for left, right in buffer_sources.items())
                or len(set(buffer_sources.values())) != len(buffer_sources)):
            raise ValueError("Frontend call lost its compiler occurrence or buffer map")
        renamed = buffer_sources.copy()
        for expected, actual in zip(original.arguments, record.arguments):
            if (expected.formal != actual.formal or expected.source_arg_index != actual.source_arg_index
                    or expected.call_arg_index != actual.call_arg_index or expected.triton_type != actual.triton_type):
                raise ValueError("Frontend formal differs from its compiler operand")
            source = expected.source
            if type(source) is BufferSource:
                if type(actual.source) is not BufferSource:
                    raise ValueError("Frontend owned pointer changed source kind")
                if source in renamed:
                    if renamed[source] != actual.source:
                        raise ValueError("Frontend changed a previously bound buffer source")
                elif actual.source in renamed.values():
                    raise ValueError("Frontend collapsed distinct compiler buffers")
                else:
                    renamed[source] = actual.source
            elif type(actual.source) is not type(source) or actual.source != source:
                raise ValueError("Frontend operand differs from its original boxed source")
        self.check()
        buffer_sources.clear()
        buffer_sources.update(renamed)
        return grid

    @classmethod
    def from_warmed(cls, kernel):
        import torch
        from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedCudaKernel
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        kernel, dispatch = resolve_warmed_kernel(kernel)
        if (type(kernel) is not CachingAutotuner or len(kernel.launchers) != 1
                or not kernel._cache_eligible or kernel._plugins
                or type(kernel.run) is not MethodType or kernel.run.__func__ is not CachingAutotuner.run):
            raise ValueError("Ordinary execution must select one unchanged generated launcher first")
        selected, cached = kernel.launchers[0], kernel._cached_launcher
        if type(selected) is not FunctionType or type(cached) is not FunctionType:
            raise ValueError("Expected an already cached launcher")
        arguments = getattr(selected, "_cudagraph_arg_info", None)
        recipe = getattr(selected, "_cudagraph_grid_recipe", None)
        runner = selected.__globals__.get("runner")
        module = runner.__self__ if type(runner) is MethodType else None
        if (type(module) is not StaticallyLaunchedCudaKernel or module.device_agnostic
                or module.module is None or type(module.function) is not int or module.function <= 0
                or arguments is None or recipe is None
                or getattr(selected, "_is_static", None) is not True
                or getattr(cached, "_cudagraph_arg_info", None) is not arguments
                or getattr(cached, "_cudagraph_grid_recipe", None) is not recipe
                or getattr(cached, "config", None) is not selected.config
                or getattr(selected.config, "pre_hook", None) is not None
                or getattr(selected.config, "ir_override", None) is not None
                or kernel.inductor_meta.get("grid_type") != "Grid1D"
                or (cached is not selected and (
                    cached.__code__ is not selected.__code__
                    or type(cached.__globals__.get("runner")) is not torch._C._FastCudaLauncher
                    or getattr(cached, "_static_kernel_owner", None) is not module))):
            raise ValueError("Missing exact selected launcher correspondence")
        provenance = kernel.inductor_meta.get("cudagraph_parameter_provenance", {})
        writes = tuple((name, row[1]) for name, row in provenance.items())
        return cls(arguments, recipe, writes, module.arg_tys, (kernel, selected, cached, module), dispatch=dispatch)
