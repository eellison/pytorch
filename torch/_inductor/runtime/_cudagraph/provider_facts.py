"""Selected compiler ABI and specializations for opaque kernel calls."""

from dataclasses import dataclass
from types import MethodType

from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import FXTraceDeclined
from torch._inductor.runtime.cudagraph_arg_mapping import KernelArgument
from torch._inductor.runtime.hints import HeuristicType
from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedCudaKernel
from torch._inductor.runtime.triton_parameter_analysis import user_kernel_controls_supported


@dataclass(frozen=True)
class PointerAlignment:
    formal: str
    alignment: int


@dataclass(frozen=True)
class ProviderFacts:
    provider: object
    arguments: tuple[KernelArgument, ...]
    pointers: tuple[PointerAlignment, ...]
    user: bool
    _owners: tuple
    _values: tuple

    def check(self):
        current = read_provider_facts(self.provider)
        if (len(self._owners) != len(current._owners)
                or any(before is not after for before, after in zip(self._owners, current._owners))
                or self._values != current._values):
            raise FXTraceDeclined("Selected compiler ABI or specializations changed during preparation")


def read_provider_facts(provider):
    if len(provider.launchers) != 1:
        raise FXTraceDeclined("Expected one selected launcher for compiler ABI correspondence")
    launcher, = provider.launchers
    cached = provider._cached_launcher
    runner = launcher.__globals__.get("runner")
    module = runner.__self__ if type(runner) is MethodType else None
    if type(module) is not StaticallyLaunchedCudaKernel:
        raise FXTraceDeclined("Selected module has no supported compiler ABI correspondence")
    formals = module.cudagraph_formal_args
    if (type(formals) is not tuple or not formals
            or any(type(row) is not KernelArgument or type(row.formal) is not str
                   or type(row.triton_type) is not str or type(row.source_arg_index) is not int
                   or row.source_arg_index != index
                   or row.abi_index is not None and type(row.abi_index) is not int
                   for index, row in enumerate(formals))
            or len({row.formal for row in formals}) != len(formals)
            or tuple(row.formal for row in formals) != tuple(module.arg_names)):
        raise FXTraceDeclined("Selected module has no supported compiler ABI correspondence")
    if any(type(row.attributes) is not tuple
           or any(type(attr) is not tuple or len(attr) != 2 or type(attr[0]) is not str
                  or type(attr[1]) not in (int, float, bool, str, type(None)) for attr in row.attributes)
           for row in formals):
        raise FXTraceDeclined("Selected compiler ABI correspondence has no exact specialization attributes")
    arguments = tuple(sorted((row for row in formals if row.abi_index is not None),
                             key=lambda row: row.abi_index))
    if tuple(row.abi_index for row in arguments) != tuple(range(len(arguments))):
        raise FXTraceDeclined("Selected compiler ABI correspondence has missing or repeated slots")
    abi = "".join("O" if row.triton_type.startswith("*") else {"i32": "i", "i64": "l"}.get(
        row.triton_type, "?") for row in arguments)
    if abi != module.arg_tys:
        raise FXTraceDeclined("Compiler formals differ from the selected runner types")

    pointer_rows = tuple(row for row in arguments if row.triton_type.startswith("*"))
    user = provider.heuristic_type is HeuristicType.USER_AUTOTUNE
    marker = provider.inductor_meta.get("cudagraph_user_kernel")
    if user and (marker is not True or not user_kernel_controls_supported()
                 or provider.fn.pre_run_hooks or provider.fn.launch_metadata is not None):
        raise FXTraceDeclined("User kernel lost its compiler host/configuration contract")
    owners = (provider, provider.fn, launcher, cached, module, formals, *formals)
    values = (module.module, module.function, module.hash, module.arg_tys, module.num_ctas,
              module.has_global_scratch, module.global_scratch_size, module.global_scratch_align,
              module.has_profile_scratch, module.profile_scratch_size,
              provider.fn.src, provider.heuristic_type, marker,
              tuple((row.formal, row.source_arg_index, row.triton_type, row.abi_index, row.constant, row.attributes)
                    for row in formals))
    pointers = []
    for row in pointer_rows:
        divisors = [value for name, value in row.attributes if name == "tt.divisibility"]
        if any(type(value) is not int or value <= 0 or value & (value - 1) for value in divisors):
            raise FXTraceDeclined("Unsupported compiler pointer alignment")
        pointers.append(PointerAlignment(row.formal, max([1, *divisors])))
    if any(not 0 < pointer.alignment <= 16 or pointer.alignment & (pointer.alignment - 1)
           for pointer in pointers):
        raise FXTraceDeclined("Pointer specialization exceeds the supported allocator alignment")
    pointers = tuple(pointers)
    return ProviderFacts(provider, arguments, pointers, user, owners, (*values, user, pointers))
