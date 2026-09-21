"""Observe ordinary Triton selection and record its symbolic host invocation."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import sympy
import torch
from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedCudaKernel
from torch.cuda._utils import _check_cuda_bindings
from triton.compiler import ASTSource, CompiledKernel
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction

from .direct_invocation import ACTIVE
from .triton_scratch import scratch_specs, TritonScratchDeclined
from .triton_tma import descriptor_guards, descriptor_specs, snapshot_descriptor, TritonTmaDeclined


_POINTER_DTYPES = {
    "*i1": torch.bool, "*i8": torch.int8, "*i16": torch.int16, "*i32": torch.int32,
    "*i64": torch.int64, "*u8": torch.uint8, "*u16": torch.uint16, "*u32": torch.uint32,
    "*u64": torch.uint64, "*fp16": torch.float16, "*bf16": torch.bfloat16,
    "*fp32": torch.float32, "*fp64": torch.float64,
}


class DirectTritonDeclined(ValueError):
    pass


@dataclass(frozen=True)
class PointerAlignment:
    formal: str
    source_arg_index: int
    alignment: int


@dataclass(frozen=True, eq=False)
class DirectTritonInvokeEvent:
    owner: object
    arguments: tuple
    grid: tuple
    guards: tuple


def _attributes(source):
    return tuple((path, tuple(tuple(attr) for attr in attrs)) for path, attrs in source.attrs.items())


class DirectTritonOwner:
    def __init__(self, binary, jit, device_index):
        from cuda.bindings import driver

        if (type(binary) is not CompiledKernel or type(binary.src) is not ASTSource
                or binary.src.fn is not jit or binary.metadata.target.backend != "cuda"
                or binary.metadata.target.warp_size != 32 or binary.metadata.num_ctas != 1
                or type(binary.kernel) is not bytes or binary.kernel != binary.asm.get("cubin")):
            raise DirectTritonDeclined("Ordinary Triton launch has no supported selected CUDA compilation")
        self.binary, self.jit, self.device_index = binary, jit, device_index
        self.module = None
        self.closed = False
        self._source = binary.src
        self._compiler = (binary.hash, binary.metadata, binary.kernel,
                          tuple(binary.src.signature.items()), tuple(binary.src.constants.items()),
                          _attributes(binary.src))
        self._jit = (jit.fn, jit.fn.__code__, jit.src, type(jit).run)
        self._globals = tuple((name, value, namespace)
                              for (name, _), (value, namespace) in jit.used_global_vals.items())
        with TemporaryDirectory(prefix="direct_triton_") as directory:
            cubin = Path(directory) / "selected.cubin"
            cubin.write_bytes(binary.kernel)
            selected = SimpleNamespace(src=binary.src, metadata=binary.metadata, hash=binary.hash,
                                       asm={"cubin": binary.kernel}, _cubin_path=str(cubin))
            module = StaticallyLaunchedCudaKernel(selected)
            try:
                if module.cudagraph_formal_args is None:
                    raise DirectTritonDeclined("Selected Triton launch has no supported compiler ABI")
                try:
                    self.scratch = scratch_specs(module)
                except TritonScratchDeclined as error:
                    raise DirectTritonDeclined(str(error)) from error
                self.formals = module.cudagraph_formal_args
                try:
                    self.descriptors = descriptor_specs(module, self.formals)
                except TritonTmaDeclined as error:
                    raise DirectTritonDeclined(str(error)) from error
                descriptors = dict(self.descriptors)
                expected_abi = "".join(
                    descriptors[row.source_arg_index].abi_types if row.source_arg_index in descriptors else
                    "O" if row.triton_type.startswith("*") else {"i32": "i", "i64": "l"}[row.triton_type]
                    for row in sorted((row for row in self.formals if row.abi_index is not None),
                                      key=lambda row: row.abi_index)
                )
                if module.arg_tys != expected_abi:
                    raise DirectTritonDeclined("Selected Triton ABI differs from the descriptor expansion")
                if tuple(row.formal for row in self.formals) != tuple(jit.arg_names):
                    raise DirectTritonDeclined("Selected Triton ABI lost its original formal ordering")
                pointers = []
                for row in self.formals:
                    if row.attributes is None:
                        raise DirectTritonDeclined("Selected Triton argument attributes are unavailable")
                    if row.source_arg_index in descriptors and row.attributes:
                        raise DirectTritonDeclined("Selected Triton descriptor attributes have no replay contract")
                    alignments = []
                    for name, value in row.attributes:
                        if (name != "tt.divisibility" or type(value) is not int or value <= 0
                                or value & (value - 1)):
                            raise DirectTritonDeclined("Unsupported selected Triton argument specialization")
                        alignments.append(value)
                    if row.abi_index is not None and row.triton_type.startswith("*"):
                        pointers.append(PointerAlignment(row.formal, row.source_arg_index,
                                                         max((1, *alignments))))
                self.pointers = tuple(pointers)
                self.scratch_abi_indices = tuple(range(len(module.arg_tys), len(module.arg_tys) + len(self.scratch)))
                self.null_abi_indices = tuple(index for index, spec in zip(self.scratch_abi_indices, self.scratch)
                                              if spec.size == 0)
                module.load_kernel(device_index)
                self.module = module
                self._module = (module.module, module.function, module.arg_tys, module.num_warps,
                                module.shared, module.has_global_scratch, module.has_profile_scratch,
                                module.num_ctas, module.global_scratch_size, module.global_scratch_align,
                                module.profile_scratch_size)
                self.abi_layout = []
                widths = [128 if kind == "M" else 8 if kind in ("O", "l") else 4 for kind in module.arg_tys]
                for index, width in enumerate((*widths, *(8 for _ in self.scratch_abi_indices))):
                    offset, size = _check_cuda_bindings(driver.cuFuncGetParamInfo(module.function, index))
                    if size != width:
                        raise DirectTritonDeclined("Loaded Triton parameter width differs from selected compiler ABI")
                    self.abi_layout.append((offset, size))
                result = driver.cuFuncGetParamInfo(module.function, len(self.abi_layout))
                if result[0] != driver.CUresult.CUDA_ERROR_INVALID_VALUE:
                    _check_cuda_bindings(result)
                    raise DirectTritonDeclined("Loaded Triton kernel has unexpected trailing parameters")
                self.abi_layout = tuple(self.abi_layout)
                self.check()
            except BaseException:
                module.close()
                self.closed = True
                raise

    def check(self):
        binary, jit, module = self.binary, self.jit, self.module
        if (self.closed or module is None or binary.src is not self._source or binary.src.fn is not jit
                or (binary.hash, binary.metadata, binary.kernel, tuple(binary.src.signature.items()),
                    tuple(binary.src.constants.items()), _attributes(binary.src)) != self._compiler
                or binary.asm.get("cubin") != self._compiler[2]
                or (jit.fn, jit.fn.__code__, jit.src, type(jit).run) != self._jit
                or module.cudagraph_formal_args is not self.formals
                or (module.module, module.function, module.arg_tys, module.num_warps, module.shared,
                    module.has_global_scratch, module.has_profile_scratch, module.num_ctas,
                    module.global_scratch_size, module.global_scratch_align, module.profile_scratch_size) != self._module):
            raise DirectTritonDeclined("The selected ordinary Triton compilation or owner changed")
        try:
            if descriptor_specs(module, self.formals) != self.descriptors:
                raise DirectTritonDeclined("Selected Triton descriptor metadata changed")
        except TritonTmaDeclined as error:
            raise DirectTritonDeclined(str(error)) from error
        for name, value, namespace in self._globals:
            if name not in namespace or namespace[name] != value:
                raise DirectTritonDeclined("A selected Triton compilation global changed")

    def close(self):
        if not self.closed:
            self.module.close()
            self.closed = True


@dataclass(frozen=True)
class _ObservedCall:
    owner: DirectTritonOwner
    options: tuple
    config: object
    config_values: tuple
    tuning_keys: tuple

    def check(self):
        self.owner.check()
        if self.config is not None and (
                tuple(self.config.all_kwargs().items()) != self.config_values
                or self.config.pre_hook is not None):
            raise DirectTritonDeclined("The observed Triton autotuner configuration changed")


def _bind(jit, args, kwargs):
    formal_kwargs = {name: value for name, value in kwargs.items() if name in jit.arg_names}
    bound = jit.signature.bind(*args, **formal_kwargs)
    bound.apply_defaults()
    return bound.arguments


def _integer(value):
    if type(value) is int:
        return sympy.Integer(value)
    if type(value) is torch.SymInt:
        return value.node.expr
    raise DirectTritonDeclined("Triton scalar/grid values must be host integers or SymInts")


def _guard(guards, expression):
    if expression is sympy.false:
        raise DirectTritonDeclined("Symbolic invocation contradicts its ordinary Triton selection")
    if expression is not sympy.true:
        guards.append(expression)


class _TraceView:
    providers = ()

    def __init__(self, adapter, sink):
        self.adapter, self.sink = adapter, sink
        self.calls = tuple(adapter._calls)
        self.position = 0

    def check(self):
        self.adapter.check()
        for call in self.calls:
            call.check()

    def finish(self):
        self.check()
        if self.position != len(self.calls):
            raise DirectTritonDeclined("Symbolic host omitted an observed ordinary Triton invocation")

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

    def run(self, *args, grid, warmup=False, **kwargs):
        self.check()
        if warmup is not False or self.position >= len(self.calls):
            raise DirectTritonDeclined("Symbolic host added an unobserved Triton invocation")
        call = self.calls[self.position]
        jit = call.owner.jit
        options = tuple((name, value) for name, value in kwargs.items() if name not in jit.arg_names)
        if options != call.options:
            raise DirectTritonDeclined("Symbolic Triton launch options differ from ordinary execution")
        overlap = kwargs.keys() & dict(call.config_values).keys()
        if overlap:
            raise DirectTritonDeclined("Explicit launch arguments overlap the selected autotuner configuration")
        bound = _bind(jit, args, {**kwargs, **dict(call.config_values)})
        arguments = []
        for row in call.owner.formals:
            value = bound[row.formal]
            if (row.abi_index is not None and row.triton_type.startswith("*")
                    and not isinstance(value, torch.Tensor)):
                from torch._native.const_tensor_wrapper import ConstTensorWrapper

                if type(value) is ConstTensorWrapper:
                    value = value._tensor
            arguments.append(value)
        arguments = tuple(arguments)
        descriptors = dict(call.owner.descriptors)
        guards = []
        for row, value in zip(call.owner.formals, arguments):
            if row.abi_index is None:
                if type(value) is torch.SymInt or type(value) is int and type(row.constant) is int:
                    _guard(guards, sympy.Eq(_integer(value), _integer(row.constant)))
                elif type(value) is not type(row.constant) or value != row.constant:
                    raise DirectTritonDeclined("Triton constant differs from selected compiler specialization")
            elif row.triton_type.startswith("*"):
                if (not isinstance(value, torch.Tensor) or value.dtype != _POINTER_DTYPES.get(row.triton_type)
                        or value.device != torch.device("cuda", call.owner.device_index)):
                    raise DirectTritonDeclined("Triton pointer formal differs from its traced dtype/device")
            elif row.source_arg_index in descriptors:
                continue
            else:
                expression = _integer(value)
                bits = {"i32": 32, "i64": 64}[row.triton_type]
                _guard(guards, sympy.Ge(expression, -(2 ** (bits - 1))))
                _guard(guards, sympy.Le(expression, 2 ** (bits - 1) - 1))
                for _, alignment in row.attributes:
                    _guard(guards, sympy.Eq(sympy.Mod(expression, alignment), 0))
        for name, expected in call.tuning_keys:
            _guard(guards, sympy.Eq(_integer(bound[name]), _integer(expected)))
        dimensions = grid(bound) if callable(grid) else grid
        if type(dimensions) not in (tuple, list) or not 1 <= len(dimensions) <= 3:
            raise DirectTritonDeclined("Triton grid must return one to three dimensions")
        dimensions = (*dimensions, *((1,) * (3 - len(dimensions))))
        for index, value in enumerate(dimensions):
            expression = _integer(value)
            _guard(guards, sympy.Ge(expression, 1))
            _guard(guards, sympy.Le(expression, 2**31 - 1 if index == 0 else 65535))
        if call.owner.descriptors:
            arguments = list(arguments)
            for index, spec in call.owner.descriptors:
                try:
                    arguments[index] = snapshot_descriptor(arguments[index], spec, call.owner.device_index)
                except TritonTmaDeclined as error:
                    raise DirectTritonDeclined(str(error)) from error
                for guard in descriptor_guards(arguments[index], spec):
                    _guard(guards, guard)
            arguments = tuple(arguments)
        self.position += 1
        self.sink(DirectTritonInvokeEvent(call.owner, arguments, dimensions, tuple(guards)))
        return call.owner.binary


class DirectTriton:
    """Substitute this proxy for a host's actual JITFunction or Autotuner reference."""

    def __init__(self, kernel):
        if type(kernel) is JITFunction:
            jit = kernel
        elif type(kernel) is Autotuner and type(kernel.fn) is JITFunction:
            jit = kernel.fn
        else:
            raise DirectTritonDeclined("Direct Triton tracing requires a JITFunction or its ordinary Autotuner")
        self.kernel, self.jit = kernel, jit
        self._binding = (kernel, jit, type(kernel).run, type(jit).run)
        self._calls = []
        self._owners = []
        self._observing = False
        self._completed = False
        self._decline_reason = None
        self.closed = False

    def check(self):
        kernel, jit, run, jit_run = self._binding
        if (self.closed or self.kernel is not kernel or self.jit is not jit
                or type(kernel).run is not run or type(jit).run is not jit_run
                or "run" in vars(kernel) or "run" in vars(jit)
                or type(kernel) is Autotuner and kernel.fn is not jit):
            raise DirectTritonDeclined("The direct Triton invocation target changed or closed")

    @contextmanager
    def observe(self):
        self.check()
        if self._observing:
            raise DirectTritonDeclined("Direct Triton observation is closed or already active")
        self._calls = []
        self._completed = False
        self._decline_reason = None
        self._observing = True
        try:
            yield self
            self._completed = True
        finally:
            self._observing = False

    def trace_view(self, sink):
        self.check()
        if self._decline_reason is not None:
            raise DirectTritonDeclined(self._decline_reason)
        if self._observing or not self._completed or not self._calls:
            raise DirectTritonDeclined("Symbolic Triton tracing requires completed ordinary observations")
        return _TraceView(self, sink)

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

    def run(self, *args, grid, warmup=False, **kwargs):
        active = ACTIVE.get()
        if active is not None:
            return active.triton(self, args, grid=grid, warmup=warmup, kwargs=kwargs)
        return self._run_ordinary(*args, grid=grid, warmup=warmup, **kwargs)

    def _run_ordinary(self, *args, grid, warmup=False, **kwargs):
        self.check()
        if not self._observing:
            return self.kernel.run(*args, grid=grid, warmup=warmup, **kwargs)
        recordable = warmup is False and not self.jit.pre_run_hooks and self.jit.launch_metadata is None
        binary = self.kernel.run(*args, grid=grid, warmup=warmup, **kwargs)
        if self._decline_reason is not None:
            return binary
        try:
            if not recordable:
                raise DirectTritonDeclined("Direct observation requires an ordinary launch without JIT host hooks")
            config = self.kernel.best_config if type(self.kernel) is Autotuner else None
            if config is not None and config.pre_hook is not None:
                raise DirectTritonDeclined("Selected Triton configuration has an untraced host pre-hook")
            config_values = () if config is None else tuple(config.all_kwargs().items())
            bound = _bind(self.jit, args, {**kwargs, **dict(config_values)})
            keys = tuple((name, bound[name]) for name in self.kernel.keys if name in bound) if (
                config is not None and len(self.kernel.configs) > 1) else ()
            if any(type(value) is not int for _, value in keys):
                raise DirectTritonDeclined("Direct autotuner selection keys must be original integer arguments")
            device = torch.cuda.current_device()
            owner = next((value for value in self._owners if value.binary is binary
                          and value.device_index == device), None)
            if owner is None:
                owner = DirectTritonOwner(binary, self.jit, device)
                self._owners.append(owner)
            options = tuple((name, value) for name, value in kwargs.items() if name not in self.jit.arg_names)
            self._calls.append(_ObservedCall(owner, options, config, config_values, keys))
        except DirectTritonDeclined as error:
            self._decline_reason = str(error)
        return binary

    def close(self):
        if self.closed:
            return
        first_error = None
        for owner in self._owners:
            try:
                owner.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        self.closed = True
