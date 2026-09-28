"""The parameter ABI of a compiled Triton kernel, for host tracing: which of
the JIT function's arguments hold a kernel parameter slot, their types and
specializations, and where each slot lives in the kernel's parameter buffer.

Derived from the compilation's source (signature, constexprs, attributes) and
metadata, the same inputs Triton's CUDA launcher packs its arguments from,
then checked against the loaded function with cuFuncGetParamInfo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.cuda._host_trace import declined
from torch.cuda._utils import _check_cuda_bindings


# the scalar types a slot may hold, by width; every pointer is 8 bytes
_SCALAR_BYTES = {"i32": 4, "i64": 8, "fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8}
# the launcher appends a global and a profile scratch pointer to every launch
_LAUNCHER_SCRATCH_SLOTS = 2


@dataclass(frozen=True)
class TritonArg:
    name: str
    triton_type: str  # "*fp32", a _SCALAR_BYTES key or "constexpr"
    slot: int | None  # the kernel parameter index; None for a constexpr
    constant: Any = None  # a constexpr's compiled value
    divisibility: int = 1  # tt.divisibility: the value (or address) is a multiple

    @property
    def is_pointer(self) -> bool:
        return self.triton_type.startswith("*")


@dataclass(frozen=True)
class TritonABI:
    args: tuple[TritonArg, ...]  # in the JIT function's order
    num_slots: int  # including the launcher's trailing scratch pointers
    num_warps: int
    shared: int

    @property
    def scratch_slots(self) -> range:
        return range(self.num_slots - _LAUNCHER_SCRATCH_SLOTS, self.num_slots)

    def slot_bytes(self, slot: int) -> int:
        for arg in self.args:
            if arg.slot == slot:
                return 8 if arg.is_pointer else _SCALAR_BYTES[arg.triton_type]
        return 8


def triton_abi(src: Any, metadata: Any) -> TritonABI:
    """The ABI of the compilation `src` (an ASTSource) with `metadata`, or
    Declined for a launch this tracer does not describe."""
    name = src.fn.__name__

    def decline(why: str) -> None:
        raise declined(f"Triton kernel {name}: {why}")

    if metadata.num_ctas != 1:
        decline("thread block clusters (num_ctas > 1) are not traced")
    if metadata.launch_cooperative_grid or metadata.launch_pdl:
        decline("cooperative and programmatic-dependent launches are not traced")
    if metadata.global_scratch_size or metadata.profile_scratch_size:
        decline("the launcher's per-launch scratch allocation is not traced")
    if getattr(metadata, "tensordesc_meta", None):
        decline("TMA descriptor arguments are not traced")
    if "gsan" in getattr(metadata, "instrumentation_mode", ""):
        decline("gsan instrumentation is not traced")
    if list(src.signature) != list(src.fn.arg_names):
        decline("the compiled signature does not follow the function's arguments")
    args = []
    slot = 0
    for i, (arg, ty) in enumerate(src.signature.items()):
        attrs = src.attrs.get((i,), ())
        divisibility = 1
        for attr, value in attrs:
            if attr != "tt.divisibility":
                decline(f"argument {arg} has the unsupported attribute {attr}")
            divisibility = value
        if ty == "constexpr":
            args.append(TritonArg(arg, ty, None, src.constants[(i,)]))
            continue
        if not isinstance(ty, str) or not (ty.startswith("*") or ty in _SCALAR_BYTES):
            decline(f"argument {arg} of type {ty} has no traced slot")
        args.append(TritonArg(arg, ty, slot, divisibility=divisibility))
        slot += 1
    return TritonABI(
        tuple(args), slot + _LAUNCHER_SCRATCH_SLOTS, metadata.num_warps, metadata.shared
    )


def param_layout(abi: TritonABI, function: int) -> tuple[tuple[int, int], ...]:
    """(offset, size) of every slot of the loaded `function`, which must have
    exactly the ABI's slots at the ABI's widths."""
    from cuda.bindings import driver

    layout = []
    for slot in range(abi.num_slots):
        offset, size = _check_cuda_bindings(driver.cuFuncGetParamInfo(function, slot))
        if size != abi.slot_bytes(slot):
            raise declined(
                f"Triton parameter {slot} is {size} bytes; its ABI says {abi.slot_bytes(slot)}"
            )
        layout.append((offset, size))
    err, *_ = driver.cuFuncGetParamInfo(function, abi.num_slots)
    if err != driver.CUresult.CUDA_ERROR_INVALID_VALUE:
        raise declined("Triton parameters past the ABI")
    return tuple(layout)
