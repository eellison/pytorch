"""Host tracing (private): the launch record of a trace's kernels, and the
CUtensorMap parameters a launch encodes at each call."""

from __future__ import annotations

import ctypes
import functools
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
from torch.cuda._host_trace_tape import _PLACEHOLDER_LOW, _Root, _TracedTensor
from torch.cuda._utils import _check_cuda_bindings


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.cuda._host_trace_triton import TritonABI


# a bit of CUtensorMap byte 8 the CuTe DSL's inline encode sets and no encode
# option does; the kernel is correct either way
_DSL_ENCODE_BIT = 0x02
_GRID_LIMITS = (2**31 - 1, 65535, 65535)


@dataclass(frozen=True)
class KernelLaunch:
    """One kernel launch of a trace: Triton's, CuTe's or an opaque call's.
    Every value is an int or a SymInt over the trace's symbols."""

    name: str
    function: int  # the CUfunction
    abi: TritonABI | None  # None for a CuTe launch, which has `pointers`
    layout: tuple[tuple[int, int], ...]  # (offset, size) of each slot
    grid: tuple[Any, Any, Any]
    block: tuple[Any, Any, Any]
    smem: Any
    slots: tuple[Any, ...]  # each parameter slot's value, in ABI order
    roots: tuple[_Root, ...]  # the pointer arguments' roots; all may be written
    owner: Any = None  # keeps `function` loaded: a CompiledKernel or Inductor's result
    # per slot (parameter, byte offset, width) where a parameter packs several
    # slots (a CuTe memref: its address, then its dynamic shape and strides; an
    # opaque kernel's); the slots descriptors read come after them
    fields: tuple[tuple[int, int, int], ...] | None = None
    descriptors: tuple[TmaDescriptor, ...] = ()
    # the node's topology, fixed at capture: the launch attributes it is
    # captured with, (CUlaunchAttributeID, value) as explicit_attributes has
    # them, and whether the edge into it is programmatic (the launch allows
    # stream serialization)
    attributes: tuple[tuple[Any, Any], ...] = ()
    pointers: frozenset[int] | None = None  # the address slots, if not the ABI's
    images: tuple[bytes, ...] = ()  # each parameter's bytes `fields` are packed over; zeros if empty
    programmatic: bool = False
    # an opaque RNG call's generator state fields, (parameter, byte offset,
    # kind, delta) with kind as a harvest Slot's, packed at capture from the
    # capture's philox state; the call's philox offsets, on its first node
    rng: tuple[tuple[int, int, str, int], ...] = ()
    rng_increment: int = 0
    # the generator the RNG call draws from, a graph input under graphsafe
    # RNG; None for the default generator
    generator: torch.Generator | None = None
    # functor members eager reads from a 0-dim CPU tensor, (parameter, byte
    # offset, CPU scalar class, tensor): re-read from the tensor at each call
    cpu_scalars: tuple[tuple[int, int, str, torch.Tensor], ...] = field(default=(), compare=False)
    # the kernel takes one struct parameter that `layout` splits into pieces
    # (so descriptors are whole parameters); the pieces are passed contiguously
    packed: bool = False
    # the launch through its library's own launcher (Triton's, Inductor's) on
    # (raw stream, each slot's value at a call, the grid there), which the
    # capture checks the record against at the traced call
    witness: Callable[[int, Sequence[int], tuple[int, int, int]], None] | None = field(default=None, compare=False, repr=False)

    @property
    def cluster(self) -> tuple[int, int, int]:
        from cuda.bindings import driver

        key = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        return dict(self.attributes).get(key, (1, 1, 1))


def compiled_cluster(function: int) -> tuple[tuple[int, int, int], ...]:
    """The kernel's __cluster_dims__ as a 1-tuple, () if it has none. An exec
    fixes a node's cluster at instantiation: cuGraphExecKernelNodeSetParams
    into or out of such a function succeeds, then the node runs at the old
    cluster or launches nothing."""
    from cuda.bindings import driver

    attr = driver.CUfunction_attribute
    names = ("WIDTH", "HEIGHT", "DEPTH")
    dims = tuple(_check_cuda_bindings(driver.cuFuncGetAttribute(getattr(attr, f"CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_{n}"), function)) for n in names)
    return (dims,) if any(dims) else ()


@dataclass(frozen=True)
class TmaDescriptor:
    """A CUtensorMap parameter, encoded at each call from rank * 2 slots from
    `first` on: the global address, the extent of each dimension, the byte
    stride of each dimension but the first. The encode's other arguments are
    the CuTe DSL's constants: element strides 1, no interleave, L2 promotion
    128B; `fill` is the out-of-bounds fill."""

    param: int
    first: int
    dtype: int  # CUtensorMapDataType
    box: tuple[int, ...]
    swizzle: int  # CUtensorMapSwizzle
    fill: int = 0  # CUtensorMapFloatOOBfill
    # nonzero: Triton's encode of elements this size, without the CuTe DSL's
    # bit and with its bit-21 clear (_triton_clears_bit21)
    triton_elem: int = 0
    # with the bits the library's encode adds to the driver's (the CuTe DSL's
    # bit, Triton's bit-21 clear): eager's bytes; else the driver's as is
    # (tma_library_bits)
    edits: bool = True
    # the last encode's slot values and CUtensorMap: a change of the address
    # alone replaces it in place (a fraction of an encode's cost)
    last: list[Any] = field(default_factory=list, compare=False, repr=False)

    def encode(self, slots: Sequence[int], edits: bool | None = None) -> bytes:
        from cuda.bindings import driver

        first, *rest = slots[self.first : self.first + 2 * len(self.box)]
        # the driver rejects a placeholder's non-canonical top; a real
        # address has none
        values = (first & _PLACEHOLDER_LOW, *rest)
        last = self.last
        if not last or last[0][1:] != values[1:]:
            last[:] = [values, _encode_tma(self.dtype, self.box, self.swizzle, values, self.fill)]
        elif last[0][0] != values[0]:
            _check_cuda_bindings(driver.cuTensorMapReplaceAddress(last[1], values[0]))
            last[0] = values
        image = bytearray(ctypes.string_at(last[1].getPtr(), 128))
        if not (self.edits if edits is None else edits):
            return bytes(image)
        if not self.triton_elem:
            image[8] |= _DSL_ENCODE_BIT
        elif _triton_clears_bit21(self.triton_elem, values):
            image[10] &= ~0x20  # bit 21 of the second word
        return bytes(image)


def tma_edits() -> bool | None:
    """Whether a descriptor recorded now launches with its library's bits
    (tma_library_bits); None where the native replay cannot launch the
    driver's encode as is."""
    if torch.cuda._host_trace.tma_library_bits:
        return True
    return False if getattr(torch._C, "_host_trace_tma_raw", False) else None


@functools.cache
def _driver_version() -> int:
    from cuda.bindings import driver

    return _check_cuda_bindings(driver.cuDriverGetVersion())


def _triton_clears_bit21(elem: int, values: tuple[int, ...]) -> bool:
    """Whether Triton's launcher (as CUTLASS) clears bit 21 of the map's
    second word: on drivers <= 13010, where the C int max byte index + 1 of
    the encode's `values` (address, extents, byte strides) is under 128 KiB."""
    rank = len(values) // 2
    index = sum((x - 1) * b for x, b in zip(values[1 : rank + 1], (elem, *values[rank + 1 :]))) % 2**32
    return _driver_version() <= 13010 and (index - 2**32 if index >= 2**31 else index) < 2**17 - 1


def _encode_tma(
    dtype: int, box: tuple[int, ...], swizzle: int, values: tuple[int, ...], fill: int = 0
) -> Any:
    from cuda.bindings import driver

    rank = len(box)
    return _check_cuda_bindings(
        driver.cuTensorMapEncodeTiled(
            dtype,
            rank,
            values[0],
            [driver.cuuint64_t(v) for v in values[1 : rank + 1]],
            [driver.cuuint64_t(v) for v in values[rank + 1 :]],
            [driver.cuuint32_t(v) for v in box],
            [driver.cuuint32_t(1)] * rank,
            driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            driver.CUtensorMapFloatOOBfill(fill),
        )
    )


def _probe_address(t: _TracedTensor) -> Any:
    # the address Triton's specialization reads: an input's full address; an
    # allocation's or eager output's offset alone, its base a multiple of 256
    # by construction (its symbol is no input of the replay's guards)
    offset = t._sym_offset * t.element_size()
    return t._root.sym + offset if t._root.kind == "argument" else offset
