"""Host tracing (private): the cuBLAS, SDPA, cuDNN convolution and RNG provider
of opaque calls.

learn() harvests one key: four raw captures of the call (a cuBLAS op through
its out= overload), never replayed, on operands carved from an address-only
arena (reserved, never mapped: a capture only needs addresses); a call's
fresh outputs are allocated in the capture pool:

  A  operand set 1 on stream s, the stack smeared with 0xA5 below the call
  B  operand set 2 on s: every address bit between the operand's alignment
     class and 2 MiB flipped, and bit 21 flipped too
  C  set 1 on another stream
  D  set 1 on s, the stack smeared with 0x5A

A priming capture of A runs first (an uncaptured run's leftovers: cuDNN's
transient parameter image, and nvcc's host pointer at the end of an ATen
__host__ __device__ lambda's parameter, at another heap address in a harvest's
first capture). The four capture into a pool of the harvests', kept between
harvests while it caches at most _POOL_KEEP bytes. The allocator's history of
the pool (recorded from each capture on unless the process records its own)
says which allocations are the call's: the call's blocks are the pool's free
blocks holding one, and each is one scratch buffer as far as they reach in it
(the allocations a call frees coalesce). The blocks a capture's call allocated
are plugged before the next capture, so each capture's sit at their own
addresses. The call's cuBLAS workspace is no allocation: each capture's is its
own window of the arena past the operand sets, one more scratch buffer. Every
qword of every kernel's parameters is classified across the four: operand +
delta (moves with the operand set), scratch + delta (inside one of the call's
blocks in every capture), host (differs with the stack smear or is a stack
address: kept as A's bytes, as a graph replay keeps it) or constant. The
captures run with the harvest flag set, under which ATen zeroes the unused
bytes of the parameter structs it would otherwise leave uninitialized. A qword
that differs otherwise refuses the key. The binding is then launched on fresh
random operands at the key's layouts and alignments and must equal the eager
call on them bitwise; a GEMM's, on the call's own operands into its output,
which with every input must digest as eager left them (else eager reruns into
the output). Each check runs with the scratch filled 0x00, then 0xFF: a
binding right only on zeroed scratch (a semaphore or counter) gets a memset of
a block ahead of its nodes, and a launch that hangs has its scratch zeroed
from the host and fails. A constant qword that is a CUDA address (at an
8-aligned byte: a pointer member sits at none of the others) is launched into
a mirror of its allocation, memory of the harvest's own at a shift of a
multiple of 4 GiB (an int32 under a stale high half reads the same), filled
like the scratch and never written: passing so, the address is dead and kept.
Else each address in turn is made a slot into a scratch block of its own, the
size of its mirror (and zeroed if it must be); failing that the key refuses.
A live int32 under a stale high half that reads the mirror as it reads the
allocation is the gap. The captures run under the cuBLAS
settings the key holds (the call's at the trace); cuDNN's are the current
global ones, which the key does not hold.

A cuBLAS or SDPA key launching the kernels (by name: each cuDNN plan loads its
own copy) of a key harvested in full (another M: the heuristic's choice repeats
across shapes) is its sibling: one capture, A, read with the harvested key's
slots, each slot's qword re-based to its operand or the call's block here, and
checked like any binding. Any qword of another role (into an operand or the
call's blocks, or a CUDA address the harvested key does not hold as a
constant), or a failed check, harvests the key in full; with no harvested key
launching its kernels, A primes the full harvest.

A key harvested in full first tries two captures (_PAIR): A and set 2 on
the other stream at D's smear, B, C and D at once. A is the sibling attempt's,
or a cuBLAS or attention key's first capture (they need no priming one, a
convolution's backward does), else the one after the priming capture. A qword
that moves with the set is an operand's as in B; any other qword that differs
is host state or of no role (a convolution's stale bytes), as in the four
captures but for the smear's bytes (padding). Any refusal harvests the key in
four captures.

For a convolution (cuDNN builds its plan and parameter structs uncaptured on a
key's first call) each capture finds a free block of exactly each fresh
output's bytes (dgrad frees its workspace before grad_weight takes one).
cuDNN's structs keep fields of the uncaptured path, which may vary with the
operands: any stale bytes but a CUDA address whose low half varies are lenient,
checked by launching another capture's. CUTLASS 3 wgrad and strided dgrad
keep an old workspace's address, or an int32 under the high half of one, whose
low half is live (the stream-K reduction mode): the mirror check above tells
them apart. cuBLASLt's CUTLASS 3 parameter struct is built on the heap
too: a qword that is a heap address in every capture (none CUDA's) is stale
alike. A binding holds its cuDNN plan (the owner ATen notes at the
capture), whose kernels it launches.

An RNG call (randint, uniform_, SDPA with dropout) reads the default
generator's per-capture seed and offset, and an intragraph offset: capture x
first takes _PRETAKE[x] offsets, so the three are told apart from operands.
They become philox slots, the offsets the call took its rng_increment (equal
in the four), and the check launches from the generator's offset.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import struct
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._logging import trace_structured
from torch.cuda._host_trace_capture import attribute_value, launch_attributes, MemsetNode, reproducible
from torch.cuda._host_trace_opaque import (
    OpaqueBinding,
    OpaqueKernel,
    OpaqueKey,
    OpaqueMemset,
    Slot,
    library_state_as,
)
from torch.cuda._host_trace_tape import _cuda_address, _gc_hold, seed_offset_on_device
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._comparison import default_tolerances


log = logging.getLogger(__name__)
aten = torch.ops.aten

# the out= overload each accepted op is harvested through, into a preallocated
# output (nothing may allocate the output inside a raw capture); an out=
# call's output is an operand like the inputs, at its strides and alignment
_OUT = {
    aten.mm.default: "aten::mm",
    aten.addmm.default: "aten::addmm",
    aten.bmm.default: "aten::bmm",
    aten.mm.out: "aten::mm",
    aten.addmm.out: "aten::addmm",
    aten.bmm.out: "aten::bmm",
}

# the ops harvested through the op itself: functional ones allocate their
# fresh outputs in the capture pool
_ATTENTION = frozenset(
    {
        aten._scaled_dot_product_cudnn_attention.default,
        aten._scaled_dot_product_flash_attention.default,
        aten._scaled_dot_product_efficient_attention.default,
        aten._scaled_dot_product_cudnn_attention_backward.default,
        aten._scaled_dot_product_flash_attention_backward.default,
        aten._scaled_dot_product_efficient_attention_backward.default,
    }
)
_RNG = frozenset({aten.randint.low_out, aten.native_dropout.default, aten.uniform_.default})
_CONV = frozenset({aten.convolution.default, aten.convolution_backward.default})
_REDUCE = frozenset({aten.sum.dim_IntList})
# a size-1 dim's stride is never stepped: ATen hands cuDNN convolution a
# canonical one (fixSizeOneDimStride), and a fake kernel's output may have
# another than the CUDA kernel's (cuDNN attention's empty_like of a query with
# a size-1 dim), so these ops' keys ignore it
_CUDNN = _CONV | {aten._scaled_dot_product_cudnn_attention.default, aten._scaled_dot_product_cudnn_attention_backward.default}
# the fresh outputs eager accumulates in no fixed order (flash backward's dq by
# fp32 atomics, efficient backward's dq over key splits under a lock): one
# rerun of eager can agree bitwise by chance, so these are always held to the
# tolerance
_UNORDERED = {
    aten._scaled_dot_product_flash_attention_backward.default: (0,),
    aten._scaled_dot_product_efficient_attention_backward.default: (0,),
}
_FAMILIES = {"blas": frozenset(_OUT), "attention": _ATTENTION, "conv": _CONV, "rng": _RNG, "reduce": _REDUCE}


_WINDOW = 1 << 21
_QWORD = (1 << 64) - 1
_RETRIES = 3  # harvests of one key refused for a transient reason
_SMEAR = (0xA5, 0xA5, 0xA5, 0x5A)  # per capture A, B, C, D
# per capture, the philox offsets taken before an RNG call: its intragraph
# offset moves with them
_PRETAKE = (0, 4, 8, 12)
# seconds a checked launch may run before its scratch is zeroed from another
# stream (a kernel spinning on a semaphore it expects zeroed)
_HANG = 10.0
# a mirror of the allocation a constant address points into spans from this
# far below the lowest such address to this far past the highest
_MIRROR_BELOW, _MIRROR_ABOVE = 2 << 20, 64 << 20


class _Refused(Exception):
    pass


class _Transient(_Refused):
    # a refusal a later harvest of the key may not repeat
    pass


class _Disabled(Exception):
    # the provider cannot harvest in this process
    pass


class _NoSibling(Exception):
    # the key is harvested in full after all
    pass


# whether a blas key whose kernels a harvested key's match is harvested by
# one capture read with that key's slots, and whether that binding is checked
_SIBLINGS = True
_VERIFY_SIBLINGS = True
# whether a key harvested in full first tries two captures
_PAIR = True
# bytes of the harvest pool kept cached between harvests
_POOL_KEEP = 64 << 20


def _class(align: int) -> int:
    # the largest power of two dividing the address, up to 256 (ATen's
    # _getAlignment): all cuBLAS's kernel choice reads of an address
    return align & -align if align else 256


_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _host_slot(off: int, imgs: tuple, stack: tuple, smear: tuple) -> bool:
    # whether a qword that moved with the stack smear or the stream is host
    # state of the call: a stack address, or the smear's bytes in
    # uninitialized padding
    q = int.from_bytes(imgs[0][off : off + 8], "little")
    if stack[0] <= q < stack[1]:
        return True
    diff = [i for i in range(8) if len({img[off + i] for img in imgs}) > 1]
    return bool(diff) and all(tuple(img[off + i] for img in imgs) == smear for i in diff)


def _heap_bounds() -> tuple[int, int]:
    # the brk heap of the process
    with open("/proc/self/maps") as f:
        for line in f:
            if line.endswith("[heap]\n"):
                lo, hi = line.split(None, 1)[0].split("-")
                return int(lo, 16), int(hi, 16)
    return 0, 0


def _address_range(q: int) -> tuple[int, int]:
    # the start and bytes of the allocation holding a CUDA address
    from cuda.bindings import driver

    a = driver.CUpointer_attribute
    start = _check_cuda_bindings(driver.cuPointerGetAttribute(a.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, q))
    size = _check_cuda_bindings(driver.cuPointerGetAttribute(a.CU_POINTER_ATTRIBUTE_RANGE_SIZE, q))
    return int(start), int(size)


def _address_hits(nodes: Iterable[Any], stack: tuple[int, int], known: dict[int, bool]) -> tuple[list, list]:
    """Each qword of the kernels' parameters, but a slot's or a stack
    address, that is a CUDA address, as (node, parameter, byte, value): at an
    8-aligned byte, and at the others (4 mod 8), where no pointer member of a
    parameter struct sits. `known` memoizes the driver's answer."""
    aligned: list[tuple[int, int, int, int]] = []
    misaligned: list[tuple[int, int, int, int]] = []
    for x, n in enumerate(nodes):
        if not isinstance(n, OpaqueKernel):
            continue
        slots = {(s.param, s.offset) for s in n.slots}
        for param, image in enumerate(n.images):
            for first, hits in ((0, aligned), (4, misaligned)):
                for i, q in enumerate(_qwords(image, first)):
                    off = first + 8 * i
                    if not q >> 32 or stack[0] <= q < stack[1] or any((param, off + d) in slots for d in (-4, 0, 4)):
                        continue
                    if q not in known:
                        known[q] = _cuda_address(q)
                    if known[q]:
                        hits.append((x, param, off, q))
    return aligned, misaligned


def _patched(image: bytes, off: int, q: int) -> bytes:
    return image[:off] + q.to_bytes(8, "little") + image[off + 8 :]


class _Mirrors:
    """Memory of the harvest's own standing in for the allocations constant
    addresses point into: a window of each (_MIRROR_BELOW below the lowest
    such address in it to _MIRROR_ABOVE past the highest), mapped at a shift
    of a multiple of 4 GiB, so an address's low half is kept (an int32 field
    under a stale high half reads the same)."""

    def __init__(self, device: int, addresses: Iterable[int]) -> None:
        from cuda.bindings import driver

        self.shift: dict[int, int] = {}
        self.windows: dict[int, tuple[int, int]] = {}
        self.tensors: list[torch.Tensor] = []
        self._undo: list[Callable[[], Any]] = []
        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
        minimum = driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        access = driver.CUmemAccessDesc()
        access.location = prop.location
        access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        ranges: dict[int, list[int]] = {}
        for q in addresses:
            start, size = _address_range(q)
            ranges.setdefault(start, [start + size]).append(q)
        try:
            step = int(_check_cuda_bindings(driver.cuMemGetAllocationGranularity(prop, minimum)))
            for start, (end, *qs) in ranges.items():
                lo = max(start, min(qs) - _MIRROR_BELOW) // step * step
                hi = -(-max(min(end, max(qs) + _MIRROR_ABOVE), max(qs) + 8) // step) * step
                size, reserved = hi - lo, (1 << 32) + hi - lo
                va = int(_check_cuda_bindings(driver.cuMemAddressReserve(reserved, 1 << 32, 0, 0)))
                self._undo.append(functools.partial(driver.cuMemAddressFree, va, reserved))
                handle = _check_cuda_bindings(driver.cuMemCreate(size, prop, 0))
                self._undo.append(functools.partial(driver.cuMemRelease, handle))
                at = va + (lo & 0xFFFFFFFF)
                _check_cuda_bindings(driver.cuMemMap(at, size, 0, handle, 0))
                self._undo.append(functools.partial(driver.cuMemUnmap, at, size))
                _check_cuda_bindings(driver.cuMemSetAccess(at, size, [access], 1))
                storage = torch._C._construct_storage_from_data_pointer(at, torch.device("cuda", device), size)
                self.tensors.append(torch.empty(0, dtype=torch.uint8, device=device).set_(storage))
                for q in qs:
                    self.shift[q], self.windows[q] = at - lo, (lo, hi)
        except BaseException:
            self.close()
            raise

    def fill(self, value: int) -> None:
        for t in self.tensors:
            t.fill_(value)

    def touched(self, value: int) -> bool:
        return any(bool((t != value).any()) for t in self.tensors)

    def close(self) -> None:
        torch.cuda.synchronize()
        self.tensors.clear()
        while self._undo:
            _check_cuda_bindings(self._undo.pop()())


@functools.cache
def _attribute_ids() -> list[int]:
    from cuda.bindings import driver

    return [int(k) for k in driver.CUlaunchAttributeID]


@functools.cache
def _attributes(raw: tuple[tuple[int, bytes], ...], programmatic: bool) -> tuple:
    # a captured kernel's attributes as graph_nodes has them, but only those
    # that differ from a plain launch's (explicit_attributes'), from their
    # raw CUlaunchAttributeValues
    from cuda.bindings import driver

    out = []
    for i, pad in raw:
        key, v = driver.CUlaunchAttributeID(i), driver.CUlaunchAttributeValue()
        v.pad = pad
        out.append((key, attribute_value(key, v)))
    if programmatic:
        out.append((driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, 1))
    return tuple(sorted(out, key=lambda a: int(a[0])))


@functools.cache
def _raw_attributes(attributes: tuple) -> tuple[tuple[int, bytes], ...]:
    return tuple((int(a.id), bytes(a.value.pad)) for a in launch_attributes(attributes))


def _launch(
    binding: OpaqueBinding,
    addresses: list[int],
    scratch: list[int],
    stream: int,
    philox: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """The binding's nodes launched in order on `stream` at these addresses;
    `philox` is the generator state its RNG kernels read: the seed's and
    offset's addresses, and the intragraph offset the call starts at."""

    def value(s: Slot) -> int:
        if s.kind.startswith("philox"):
            return philox[("philox_seed", "philox_offset", "philox").index(s.kind)] + s.delta
        return (addresses if s.kind == "operand" else scratch)[s.index] + s.delta

    torch._C._cuda_hostTraceLaunch(
        stream,
        [
            (value(n.dst), n.value, n.element_size, n.width, n.height, n.pitch)
            if isinstance(n, OpaqueMemset)
            else (
                n.function,
                n.grid,
                n.block,
                n.smem,
                _raw_attributes(n.attributes),
                n.images,
                [(s.param, s.offset, value(s)) for s in n.slots],
            )
            for n in binding.nodes
        ],
    )


class _Device:
    """What every harvest on one device shares, used under _LOCK: the capture
    streams, the anchor and the arena; and the current harvest's pool."""

    def __init__(self, index: int) -> None:
        self.index = index
        dev = torch.device("cuda", index)
        self.streams = (torch.cuda.Stream(dev), torch.cuda.Stream(dev))
        self.anchor = torch.empty(1, device=dev)
        self.pool: Any = None
        # a hung verify's scratch zeroed from the host, on the other stream
        self.zeros = torch.zeros(_WINDOW, dtype=torch.uint8, pin_memory=True)
        self.keeper: Any = None
        self.va = 0
        self.va_size = 0

    def arena(self, size: int) -> int:
        """`size` bytes of address space, 2 MiB-aligned and never mapped: a
        capture only records the operands' addresses, and neither cuBLAS nor
        cuDNN queries them at capture."""
        from cuda.bindings import driver

        if size > self.va_size:
            if self.va:
                _check_cuda_bindings(driver.cuMemAddressFree(self.va, self.va_size))
                self.va = self.va_size = 0
            n = max(1 << 28, 1 << (size - 1).bit_length())
            va = _check_cuda_bindings(driver.cuMemAddressReserve(n, _WINDOW, 0, 0))
            self.va, self.va_size = int(va), n
        return self.va

    def blocks(self) -> list:
        """The pool's blocks as (stream, address, size, requested size,
        active, in the small pool)."""
        return torch._C._cuda_hostTracePool(self.index, self.pool.id, False)[0]

    def plug(self, keep: list) -> list[torch.Tensor]:
        """Tensors in the pool exactly covering the free blocks in keep.
        Every free block is plugged, smallest first, so each request's best
        fit is a piece of a free block."""
        blocks = self.blocks()
        streams = {st.cuda_stream: st for st in self.streams}
        wanted = {b[1] for b in keep}
        # (stream, address, bytes) -> the free block the piece is of: among
        # blocks of one size the best fit is the oldest segment's, not the
        # lowest address, so a plug may land on another piece of its size
        pieces = {}
        order = []
        for stream, address, size, _, active, small in sorted(
            blocks, key=lambda b: (b[0], b[2], b[1])
        ):
            if active:
                continue
            # a small-pool block serves requests of at most 1 MiB
            step = _WINDOW // 2 if small else size
            for at in range(address, address + size, step):
                piece = min(step, address + size - at)
                pieces[stream, at, piece] = address
                order.append((stream, piece))
        plugs, held = [], []  # held until every block is plugged
        with torch.cuda.use_mem_pool(self.pool, self.index):
            for stream, piece in order:
                with torch.cuda.stream(streams[stream]):
                    t = torch.empty(piece, dtype=torch.uint8, device=self.index)
                owner = pieces.pop((stream, t.data_ptr(), piece), None)
                if owner is None:
                    raise _Disabled(
                        f"a {piece} byte plug landed at {t.data_ptr():#x}, "
                        "on no free piece of its size"
                    )
                (plugs if owner in wanted else held).append(t)
        return plugs


def _argument(op: Any, args: tuple, kwargs: dict, name: str) -> Any:
    for i, a in enumerate(op._schema.arguments):
        if a.name == name:
            return args[i] if i < len(args) else kwargs.get(name, a.default_value)
    return None


# whether a capture's call runs with the harvest flag set (ATen zeroes its
# kernels' parameter padding)
_ZERO_INIT = True
_LOCK = threading.Lock()
_DEVICES: dict[int, _Device] = {}


def _qwords(image: bytes, first: int) -> tuple[int, ...]:
    # the image's qwords from byte `first` on
    n = (len(image) - first) // 8
    return struct.unpack_from(f"<{n}Q", image, first) if n > 0 else ()


def _slots(n: OpaqueKernel | OpaqueMemset) -> tuple[Slot, ...]:
    return (n.dst,) if isinstance(n, OpaqueMemset) else n.slots


def _signature(nodes: Iterable[Any]) -> tuple:
    # what a harvested key's slots are read against: each kernel by name (a
    # cuDNN plan loads its own copy of its kernels) with its parameter layout
    # and launch attributes, each memset's element
    nodes = list(nodes)
    memset = [isinstance(n, (OpaqueMemset, MemsetNode)) for n in nodes]
    names = iter(torch._C._cuda_hostTraceFunctionNames([n.function for n, m in zip(nodes, memset) if not m]))
    return tuple(
        ("memset", n.value, n.element_size) if m else (next(names), n.params, n.attributes)
        for n, m in zip(nodes, memset)
    )


def _renumbered(nodes: list, sizes: list[int], increment: int, owner: Any = None) -> OpaqueBinding:
    # only the scratch some slot points into, renumbered
    used = sorted({t.index for n in nodes for t in _slots(n) if t.kind == "scratch"})
    renumber = {j: i for i, j in enumerate(used)}

    def moved(t: Slot) -> Slot:
        if t.kind != "scratch":
            return t
        return dataclasses.replace(t, index=renumber[t.index])

    nodes = [
        dataclasses.replace(n, dst=moved(n.dst))
        if isinstance(n, OpaqueMemset)
        else dataclasses.replace(n, slots=tuple(map(moved, n.slots)))
        for n in nodes
    ]
    return OpaqueBinding(tuple(nodes), tuple(sizes[j] for j in used), increment, owner)


class HarvestProvider:
    """The OpaqueProvider for the ops of `families`: "blas", bf16, fp16 and
    fp32 mm, addmm and bmm through cuBLAS; "attention", the cuDNN, flash and
    memory-efficient SDPA forwards and backwards; "conv", cuDNN's convolution
    forward and backward; "rng", randint's out= form, native_dropout, uniform_
    and any other ATen op that draws from the default generator (bernoulli_,
    normal_, rand_like), and `rng_ops` (Inductor's inductor_seeds); "reduce",
    ATen's sum over dims. It harvests at most
    budget() keys, read at each harvest."""

    def __init__(
        self,
        families: tuple[str, ...] = ("blas",),
        budget: Callable[[], int] = lambda: 1024,
        rng_ops: tuple[Any, ...] = (),
    ) -> None:
        self.budget = budget
        self.rng = _RNG | frozenset(rng_ops) if "rng" in families else frozenset()
        self.ops = frozenset().union(*(_FAMILIES[f] for f in families)) | self.rng
        self.harvests = 0
        self.bindings: dict[tuple, OpaqueBinding] = {}
        # per (op, dtypes, device) of a blas or attention key: the harvested
        # bindings by _signature
        self.templates: dict[tuple, dict[tuple, OpaqueBinding]] = {}
        self.refused: dict[tuple, str] = {}
        self.transient: dict[tuple, int] = {}  # transient refusals per key
        # per constant address or zeroed scratch block of a harvested
        # binding: (kernel, parameter, byte, value, channel)
        self.addresses: list[tuple[str, int, int, int, str]] = []
        self.disabled: str | None = None
        self._lock = threading.Lock()

    def accepts(self, op: Any, args: tuple, kwargs: dict) -> str | None:
        if self.disabled is not None:
            return self.disabled
        if op not in self.ops and not self._draws(op):
            return f"{op} is not one of the harvested ops"
        leaves = pytree.tree_leaves((args, kwargs))
        tensors = [t for t in leaves if isinstance(t, torch.Tensor)]
        if any(t.device.type != "cuda" for t in tensors):
            return "an operand is not on a CUDA device"
        if self._draws(op):
            if _argument(op, args, kwargs, "generator") is not None:
                return f"{op} with a generator other than the default"
            return None
        dtypes = {t.dtype for t in tensors}
        if op in _ATTENTION:
            dtypes = {t.dtype for t in tensors[:3]}
        if len(dtypes) != 1 or not dtypes <= {torch.bfloat16, torch.float16, torch.float32}:
            return f"operand dtypes {sorted(map(str, dtypes))}: not all bf16, fp16 or fp32"
        # the bias epilogue cuBLAS fuses (addmm's Lt path) takes a 1-D bias
        # at alpha = beta = 1; other forms run as eager steps
        if _OUT.get(op) == "aten::addmm":
            if tensors[0].dim() != 1:
                return "addmm with a bias that is not 1-D"
            scalars = [kwargs.get(k, 1) for k in ("beta", "alpha")]
            if any(type(v) not in (int, float) or v != 1 for v in scalars):
                return "addmm with beta or alpha other than 1"
        return None

    def _draws(self, op: Any) -> bool:
        tags = getattr(op, "tags", ())
        seeded = torch.Tag.nondeterministic_seeded in tags and not {torch.Tag.dynamic_output_shape, torch.Tag.data_dependent_output} & set(tags)
        return op in self.rng or (bool(self.rng) and seeded and op.namespace == "aten")

    @staticmethod
    def _normal(key: OpaqueKey) -> tuple:
        return (
            _OUT.get(key.op, key.op),
            key.dtypes,
            key.sizes,
            tuple(tuple(0 if n == 1 else st for n, st in zip(z, t)) for z, t in zip(key.sizes, key.strides)) if key.op in _CUDNN else key.strides,
            tuple(map(_class, key.align)),
            key.scalars,
            key.device,
            key.state,
        )

    def bind(self, key: OpaqueKey) -> OpaqueBinding | None:
        return self.bindings.get(self._normal(key))

    def _refuse(self, key: OpaqueKey, why: str) -> None:
        # final for the process: a trace records the key as a plain eager step
        self.refused[self._normal(key)] = why
        msg = f"{key.op} at sizes {key.sizes}: {why}"
        log.info("host_trace: harvest refused %s", msg)
        trace_structured(
            "artifact",
            metadata_fn=lambda: {"name": "host_trace_harvest_refused", "encoding": "string"},
            payload_fn=lambda: msg,
        )

    def refusal(self, key: OpaqueKey) -> str | None:
        k = self._normal(key)
        if k in self.bindings:
            return None
        if k in self.refused:
            return self.refused[k]
        if self.harvests >= self.budget():
            return f"the harvest budget ({self.budget()} keys) is spent"
        return None

    def learn(
        self, key: OpaqueKey, args: tuple, kwargs: dict, operands: list[Any]
    ) -> OpaqueBinding | None:
        k = self._normal(key)
        with self._lock, torch.cuda.device(key.device):
            if k in self.bindings:
                return self.bindings[k]
            if self.disabled is not None or k in self.refused:
                return None
            if self.harvests >= self.budget():
                return None
            # not now, but maybe at a later call of the key
            if torch.cuda.is_current_stream_capturing():
                return None
            self.harvests += 1
            try:
                # the collector held off: a harvest makes many short-lived
                # objects, and a collection can finalize a capture's graph
                with _gc_hold, _LOCK, library_state_as(key.state):
                    binding = None
                    group = (_OUT.get(key.op, key.op), key.dtypes, key.device, key.state)
                    sibling = key.op in _OUT or key.op in _ATTENTION
                    siblings = self.templates.get(group) if sibling and _SIBLINGS else None
                    pair = _PAIR
                    # a refusal of the two captures is the four's to make
                    if siblings:
                        try:
                            binding = self._harvest(key, args, kwargs, operands, siblings, pair)
                        except _NoSibling:
                            pass
                        except _Refused as e:
                            if not pair or isinstance(e, _Transient):
                                raise
                            pair = False
                    if binding is None and pair:
                        try:
                            binding = self._harvest(key, args, kwargs, operands, pair=True)
                        except _Refused as e:
                            if isinstance(e, _Transient):
                                raise
                    if binding is None:
                        binding = self._harvest(key, args, kwargs, operands)
                    if sibling:
                        self.templates.setdefault(group, {}).setdefault(_signature(binding.nodes), binding)
            except _Disabled as e:
                self.disabled = str(e)
                return None
            except _Transient as e:
                self.transient[k] = self.transient.get(k, 0) + 1
                if self.transient[k] >= _RETRIES:
                    self._refuse(key, str(e))
                return None
            except _Refused as e:
                self._refuse(key, str(e))
                return None
            self.bindings[k] = binding
            return binding

    def _harvest(
        self,
        key: OpaqueKey,
        args: tuple,
        kwargs: dict,
        operands: list[torch.Tensor],
        siblings: dict[tuple, OpaqueBinding] | None = None,
        pair: bool = False,
    ) -> OpaqueBinding:
        if not hasattr(torch._C, "_cuda_hostTraceHarvestCapture") or not torch._C._cuda_hostTraceHarvestDriver():
            raise _Disabled("no harvest in this build")
        op = key.op
        leaves, spec = pytree.tree_flatten((args, kwargs))
        # the operands the arena holds: the inputs, and an out= overload's
        # output; the rest are fresh outputs the call allocates in the pool
        placed = sum(isinstance(a, torch.Tensor) for a in leaves)
        if op in _OUT:
            if _OUT[op] == "aten::addmm" and key.strides[0] != (1,):
                raise _Refused(f"addmm with a bias of stride {key.strides[0]}")
            placed = len(operands)
            name, overload = _OUT[op], "out"
        else:
            name, overload = op._schema.name, op._schema.overload_name
        if any(t.numel() == 0 for t in operands[:placed]):
            raise _Refused("an empty operand")
        dropout = _argument(op, args, kwargs, "dropout_p")
        rng = self._draws(op) or bool(dropout)
        device = operands[0].device.index
        gen = torch.cuda.default_generators[device]
        if device not in _DEVICES:
            _DEVICES[device] = _Device(device)
        dev = _DEVICES[device]
        s, other = dev.streams

        # the two operand sets, each operand in its own 2 MiB windows of the
        # arena: set 1 at its alignment class in the window, set 2 an odd
        # number of windows on with every bit between the class and the
        # window flipped
        classes = [_class(a) for a in key.align]
        spans = [
            (1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride())))
            * t.element_size()
            if t.numel()
            else 0
            for t in operands
        ]
        windows = [(span + 2 * _WINDOW - 1) // _WINDOW for span in spans[:placed]]
        total = sum(windows) | 1
        # past the sets, each capture's cuBLAS workspace (in place of the
        # allocator's, at the size the call asks for) in windows of its own
        workspace = max(torch._C._cuda_getCublasWorkspaceSize(), torch._C._cuda_getCublasLtWorkspaceSize())
        per_capture = -(-workspace // _WINDOW)
        arena = dev.arena((2 * total + 1 + 4 * per_capture) * _WINDOW)
        workspaces = arena + (2 * total + 1) * _WINDOW
        starts: list[list[int]] = [[], []]
        cursor = 0
        for c, n in zip(classes, windows):
            second = total * _WINDOW + (c ^ ((_WINDOW - 1) & ~(2 * c - 1)))
            starts[0].append(arena + cursor + c)
            starts[1].append(arena + cursor + second)
            cursor += n * _WINDOW
        sets = [torch._C._cuda_hostTraceArenaViews(at, operands[:placed]) for at in starts]
        # a GEMM verifies on the call's own tensors, unless its output overlaps an input
        in_place = op in _OUT
        if in_place:
            lo = operands[-1].data_ptr()
            hi = lo + spans[-1]
            in_place = all(t.data_ptr() >= hi or t.data_ptr() + n <= lo for t, n in zip(operands[:-1], spans))

        graphs = []  # every capture's graph, held until the harvest is over
        grown = False  # whether a capture allocated in the pool
        plugs: list[torch.Tensor] = []
        # what the captures left live, held until the harvest is over: their
        # fresh outputs, and an RNG call's per-capture seed and offset
        kept: list[torch.Tensor] = []
        returns = [r.alias_info is None for r in op._schema.returns]

        def capture(x: int, stream: torch.cuda.Stream, tensors: list, prime: bool = False) -> tuple:
            nonlocal grown
            ws = workspaces + x * per_capture * _WINDOW
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            graphs.append(graph)
            if op in _OUT:
                call_args, call_kwargs = tuple(tensors[:-1]), {"out": tensors[-1]}
            else:
                it = iter(tensors)
                filled = [next(it) if isinstance(a, torch.Tensor) else a for a in leaves]
                call_args, call_kwargs = pytree.tree_unflatten(filled, spec)
            # a convolution's fresh outputs each get a free block of exactly its
            # bytes, and the pool no other free block: cuDNN's dgrad frees its
            # workspace before it allocates grad_weight, which is then not
            # where the workspace was
            holes = []
            for o in operands[placed:] if op in _CONV else ():
                with torch.cuda.use_mem_pool(dev.pool, device), torch.cuda.stream(stream):
                    holes.append(torch.empty(o.untyped_storage().nbytes(), dtype=torch.uint8, device=device))
                plugs.extend(dev.plug([b for b in dev.blocks() if not b[4]]))
            del holes
            # the history from here on: restarted, or the process's own from its length
            if own:
                torch._C._cuda_hostTraceRecordAllocations(True)
            seen = 0 if own else len(torch._C._cuda_hostTracePool(device, dev.pool.id, True)[1])
            # TORCH_CUBLAS_WORKSPACE_CACHE=1: the stream's cached cuBLAS
            # workspace is released before the capture, so the call allocates
            # its own in the pool (its scratch), and after it, so none is left
            # live in the pool
            try:
                result, allocs, requested, philox, held = torch._C._cuda_hostTraceHarvestCapture(
                    graph,
                    dev.pool.id,
                    (stream.stream_id, stream.device_index, stream.device_type),
                    dev.anchor,
                    ws,
                    workspace,
                    _PRETAKE[x] if rng else None,
                    _ZERO_INIT,
                    name,
                    overload,
                    _SMEAR[x],
                    *call_args,
                    **call_kwargs,
                )
            except Exception as e:
                # a CUDA error is the capture's refusal; any other, the harvest's bug
                if torch.cuda._host_trace.raise_unexpected and not isinstance(e, torch.AcceleratorError):
                    raise
                raise _Refused(f"the call failed under capture: {e}") from None
            if philox is not None:
                seed, offset, taken = philox
                kept.extend((seed, offset))
                philox = (seed.data_ptr(), offset.data_ptr(), taken)
                del seed, offset
            grown |= allocs > 0
            pool, allocated = [], []
            if allocs:
                pool, history = torch._C._cuda_hostTracePool(device, dev.pool.id, True)
                # a ring buffer's wrap loses some, which the allocation count sees
                allocated = history[seen:]
            if prime:
                return ()
            rets = [] if op in _OUT else [result] if len(returns) == 1 else list(result or ())
            fresh = [
                o
                for r, fresh in zip(rets, returns)
                if fresh
                for o in pytree.tree_leaves(r)
                if isinstance(o, torch.Tensor)
            ]
            kept.extend(fresh)
            owned = [(o.untyped_storage().data_ptr(), o.untyped_storage().nbytes()) for o in fresh]
            outputs = [o.data_ptr() for o in fresh]
            # a refusal's traceback holds this frame: no tensor of the pool in it
            del result, rets, fresh
            if len(outputs) != len(operands) - placed:
                raise _Refused(f"{len(outputs)} fresh outputs under capture, {len(operands) - placed} eager")
            try:
                found = torch._C._cuda_hostTraceCapturedNodes(graph.raw_cuda_graph(), _attribute_ids())
            except ValueError as e:
                raise _Refused(str(e)) from None
            names = [n[0] if len(n) == 9 else "memset" for n in found]
            nodes = [
                OpaqueKernel(n[1], n[3], n[4], n[5], _attributes(n[7], n[6]), n[2], n[8], ())
                if len(n) == 9
                else MemsetNode(*n)
                for n in found
            ]
            kernels = [n for n in nodes if isinstance(n, OpaqueKernel)]
            if any(not reproducible(k) for n in kernels for k, _ in n.attributes):
                raise _Refused("a device-updatable kernel node")
            plugged = {t.data_ptr() for t in plugs} | {t.untyped_storage().data_ptr() for t in kept}
            if any(b[4] and b[1] not in plugged for b in pool):
                raise _Refused("the call left an allocation live")
            # the device-wide count: another thread's allocations refuse too
            if len(allocated) != allocs:
                raise _Transient(f"{allocs} allocations, {len(allocated)} of them the call's")
            # the free blocks the call allocated in: freed allocations
            # coalesce with their free neighbors, so each is one scratch
            # buffer, as far as the call's allocations in it reach
            ends = {b: [a + n for a, n in allocated if b[1] <= a < b[1] + b[2]] for b in pool if not b[4]}
            mine = [b for b, e in ends.items() if e]
            if allocs:
                plugs.extend(dev.plug(mine))
            # an output's own allocation, or a temporary freed before the
            # output took its bytes (the output's address, as bound)
            reused = sum(any(p <= a and a + n <= p + m for p, m in owned) for a, n in allocated)
            if sum(map(len, ends.values())) + reused != allocs:
                raise _Refused(f"the call's allocations {allocated} are not its blocks and outputs")
            extents = [max(ends[b]) - b[1] for b in mine]
            # the blocks in the order the call first allocated in them (the
            # allocator places them anywhere)
            order = [next(i for i, (a, _) in enumerate(allocated) if b[1] <= a < b[1] + b[2]) for b in mine]
            ordered = [(a, n) for _, a, n in sorted(zip(order, (b[1] for b in mine), extents))]
            if requested:
                ordered.insert(0, (ws, requested))
            return nodes, names, ordered, outputs, philox, _PRETAKE[x], held

        def verify(binding: OpaqueBinding, stale: list, names: list) -> OpaqueBinding:
            # the binding checked (and with stale bytes: with another
            # capture's there), each launch with its scratch filled 0x00, then
            # 0xFF, and each constant CUDA address pointing into a mirror
            # filled alike. It returns the binding, but for a failed check that
            # a channel explains: scratch read before it is written (a
            # semaphore or counter), zeroed by a memset ahead of the binding;
            # a live constant address, into a scratch block of the harvest's
            # own (zeroed if it must be)
            nodes = binding.nodes
            variants = [binding]
            if stale:
                images = [list(n.images) if isinstance(n, OpaqueKernel) else None for n in nodes]
                for x, param, off, other in stale:
                    image = bytearray(images[x][param])
                    image[off : off + 8] = other
                    images[x][param] = bytes(image)
                alt = [n if i is None else dataclasses.replace(n, images=tuple(i)) for n, i in zip(nodes, images)]
                variants.append(dataclasses.replace(binding, nodes=tuple(alt)))
            known: dict[int, bool] = {}
            hits = [_address_hits(b.nodes, stack, known)[0] for b in variants]
            misaligned = _address_hits(nodes, stack, known)[1]
            addresses = {h[3] for hs in hits for h in hs}
            try:
                mirrors = _Mirrors(device, addresses) if addresses else None
            except RuntimeError as e:
                raise _Transient(f"no mirror of the constant addresses {sorted(map(hex, addresses))}: {e}") from None
            try:
                return checked(binding, variants, hits, misaligned, mirrors, names)
            finally:
                if mirrors is not None:
                    mirrors.close()

        def checked(
            binding: OpaqueBinding,
            variants: list[OpaqueBinding],
            hits: list[list[tuple[int, int, int, int]]],
            misaligned: list[tuple[int, int, int, int]],
            mirrors: _Mirrors | None,
            names: list,
        ) -> OpaqueBinding:
            nodes, increment = binding.nodes, binding.rng_increment
            first = len(binding.scratch)

            def build(owned: tuple, zeroed: tuple[int, ...], mirrored: bool = True) -> list[OpaqueBinding]:
                # each variant with its constant addresses into the mirrors
                # (unless kept as they are), those of `owned` slots into
                # scratch blocks of their own, and a memset of each block in
                # `zeroed` ahead of its nodes
                windows = [mirrors.windows[h[3]] for h in owned] if mirrors else []
                scratch = binding.scratch + tuple(hi - lo for lo, hi in windows)
                at = {h[:3] for h in owned}
                built = []
                for b, hs in zip(variants, hits):
                    images = [list(n.images) if isinstance(n, OpaqueKernel) else None for n in b.nodes]
                    slots = [list(n.slots) if isinstance(n, OpaqueKernel) else None for n in b.nodes]
                    for x, param, off, q in hs:
                        if mirrored and (x, param, off) not in at:
                            images[x][param] = _patched(images[x][param], off, q + mirrors.shift[q])  # type: ignore[index, union-attr]
                    for j, ((x, param, off, q), (lo, _)) in enumerate(zip(owned, windows)):
                        images[x][param] = _patched(images[x][param], off, 0)  # type: ignore[index]
                        slots[x].append(Slot(param, off, "scratch", first + j, q - lo))  # type: ignore[union-attr]
                    zeros = [OpaqueMemset(Slot(0, 0, "scratch", j, 0), 0, 1, scratch[j], 1, scratch[j]) for j in zeroed]
                    kept = [
                        n if i is None else dataclasses.replace(n, images=tuple(i), slots=tuple(s))
                        for n, i, s in zip(b.nodes, images, slots)
                    ]
                    built.append(dataclasses.replace(b, nodes=tuple(zeros + kept), scratch=scratch))
                return built

            def launch(b: OpaqueBinding, addrs: list[int], fill: int, philox: tuple[int, int, int] = (0, 0, 0)) -> str | None:
                # the binding launched on scratch and mirrors filled with
                # `fill`; why it is wrong, if it hangs or writes a mirror
                scratch = [torch.empty(n, dtype=torch.uint8, device=device) for n in b.scratch]
                for t in scratch:
                    t.fill_(fill)
                if mirrors is not None:
                    mirrors.fill(fill)
                _launch(b, addrs, [t.data_ptr() for t in scratch], stream, philox)
                done = torch.cuda.Event()
                done.record()
                deadline = time.monotonic() + _HANG
                while not done.query():
                    if time.monotonic() > deadline:
                        break
                    time.sleep(1e-4)
                else:
                    if mirrors is not None and mirrors.touched(fill):
                        return f"{names} wrote into memory a constant address points to"
                    return None
                # a kernel spinning on memory it expects zeroed: zeros from
                # the host on another stream (a copy engine runs beside it)
                with torch.cuda.stream(other):
                    for t in scratch + (mirrors.tensors if mirrors else []):
                        for lo in range(0, t.numel(), dev.zeros.numel()):
                            piece = t[lo : lo + dev.zeros.numel()]
                            piece.copy_(dev.zeros[: piece.numel()], non_blocking=True)
                deadline = time.monotonic() + _HANG
                while not done.query():
                    if time.monotonic() > deadline:
                        raise _Disabled(f"a harvested launch of {names} hangs")
                    time.sleep(1e-3)
                torch.cuda.synchronize()
                return f"{names} did not finish on memory filled {fill:#x}"

            stream = torch.cuda.current_stream().cuda_stream
            if in_place:
                # a GEMM's binding on the call's own tensors, nothing copied:
                # every operand digested, the output filled with 0xFF and each
                # binding launched into it, every operand digested again and
                # compared. A mismatch at the end reruns eager into the output;
                # an input the binding wrote is past restoring. An input's
                # digest is its row sums: a stray write changes values, it
                # does not move them along a row
                out, inputs = operands[-1], operands[:-1]
                digests, columns = torch._C._cuda_hostTraceDigests, [False] * len(inputs) + [True]
                want, held = digests(operands, columns)
                ptrs = [t.data_ptr() for t in operands]
                eager = functools.partial(getattr(aten, name.removeprefix("aten::")).out, *inputs, out=out)

                def run(b: OpaqueBinding, fill: int) -> str | None:
                    out.view(_BITS[out.element_size()]).fill_(-1)
                    try:
                        why = launch(b, ptrs, fill)
                    except _Disabled:
                        eager()
                        raise
                    got, after = digests(operands, columns)
                    wrong = [i for i, (a, c) in enumerate(zip(held[:-1], after)) if not torch.equal(a, c)]
                    if wrong:
                        eager()
                        raise RuntimeError(f"host_trace: the harvested launch of {names} wrote input {wrong[0]} of {op}")
                    if why is None and not torch.equal(want, got):
                        why = f"the binding's launch differs from eager at operand {placed - 1} ({names})"
                    return why

            else:
                # the binding on fresh random operands at the key's layouts and
                # alignments, against the eager call on them (a zero or NaN
                # operand would pass any binding; unbounded ones overflow a
                # long reduction, and an _UNORDERED output's rounding grows
                # with its terms, not its result); a second copy of each placed
                # operand when eager writes one, restored before each launch,
                # the binding's fresh outputs filled with 0xFF. An RNG call
                # draws from the generator's offset for both, which is left
                # where it was. A drawing op's floating operands are
                # probabilities, means and scales: in (0, 1)
                mutates = op in _OUT or any(a.alias_info and a.alias_info.is_write for a in op._schema.arguments)
                filler = torch.Generator(operands[0].device).manual_seed(0)
                bases, copies = [], []
                for t, span, align in zip(operands[:placed], spans, key.align):
                    base = torch.empty(-(-(span + 256) // 8), dtype=torch.int64, device=t.device).view(torch.uint8)
                    flat = base[align : align + span].view(t.dtype)
                    if t.dtype.is_floating_point and self._draws(op):
                        flat.uniform_(0.25, 0.75, generator=filler)
                    elif t.dtype.is_floating_point:
                        flat.uniform_(-1, 1, generator=filler)
                    else:
                        flat.random_(0, 2 if t.dtype == torch.bool else 8, generator=filler)
                    twin = base.clone() if mutates else base
                    bases.append((base, twin))
                    copies.append(
                        [b[align : align + span].view(t.dtype).as_strided(t.shape, t.stride()) for b in (base, twin)]
                    )
                # without a second copy, neither eager nor the binding may change an input's digest
                inputs = [] if mutates else torch._C._cuda_hostTraceDigests([c[0] for c in copies], [True] * placed)[1]

                def same(a: torch.Tensor, b: torch.Tensor) -> bool:
                    # contiguous() keeps a size-1 dim's stride, which view(uint8) rejects
                    a, b = (t.clone(memory_format=torch.contiguous_format).view(-1).view(torch.uint8) for t in (a, b))
                    return torch.equal(a, b)

                def gap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    # elementwise; equal values (infinities, NaNs) are 0 apart: the
                    # random operands overflow a few elements of some outputs
                    a, b = a.double(), b.double()
                    return torch.where((a == b) | (a.isnan() & b.isnan()), 0.0, (a - b).abs())

                ref_in = [c[0] for c in copies]
                start = gen.get_offset()
                spread: list[float | None] = []
                if op in _OUT:
                    getattr(aten, name.removeprefix("aten::")).out(*ref_in[:-1], out=ref_in[-1])
                    ref_out = []
                else:
                    it = iter(ref_in)
                    call_args, call_kwargs = pytree.tree_unflatten(
                        [next(it) if isinstance(a, torch.Tensor) else a for a in leaves], spec
                    )

                    def fresh_outputs() -> list[torch.Tensor]:
                        result = op(*call_args, **call_kwargs)
                        rets = [result] if len(returns) == 1 else list(result or ())
                        fresh = [
                            o
                            for r, fresh in zip(rets, returns)
                            if fresh
                            for o in pytree.tree_leaves(r)
                            if isinstance(o, torch.Tensor)
                        ]
                        return seed_offset_on_device(op, fresh, operands[0].device)

                    ref_out = fresh_outputs()
                    # a floating output eager itself doesn't reproduce bitwise (an
                    # _UNORDERED one, or one this rerun differs in) is held to 4x
                    # eager's own spread plus the dtype's testing tolerance: a wrong
                    # address or size gives garbage, not rounding noise
                    if not mutates:
                        gen.set_offset(start)
                        unordered = _UNORDERED.get(op, ())
                        spread = [
                            gap(a, b).nan_to_num(0.0, 0.0).max().item()
                            if a.is_floating_point() and (i in unordered or not same(a, b))
                            else None
                            for i, (a, b) in enumerate(zip(ref_out, fresh_outputs()))
                        ]
                outs = [
                    torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
                    for t in operands[placed:]
                ]
                state = (0, 0, 0)
                if increment:
                    seed = gen.initial_seed()
                    seed = torch.tensor([seed - (seed >> 63 << 64)], device=operands[0].device)
                    offset = torch.tensor([start], device=operands[0].device)
                    state = (seed.data_ptr(), offset.data_ptr(), 0)
                addrs = [c[1].data_ptr() for c in copies] + [t.data_ptr() for t in outs]
                gen.set_offset(start)
                referenced = {t.index for n in nodes for t in _slots(n) if t.kind == "operand"}
                twins = [twin.clone() for _, twin in bases] if mutates else []
                tolerated = tuple(i for i in sorted(referenced) if i >= placed and spread and spread[i - placed] is not None)

                def close(want: torch.Tensor, got: torch.Tensor, noise: float) -> bool:
                    rtol, atol = default_tolerances(want)
                    d = gap(want, got)
                    return bool((d.isfinite() & (d <= 4 * noise + atol + rtol * want.double().abs())).all())

                def run(b: OpaqueBinding, fill: int) -> str | None:
                    for (_, twin), t in zip(bases, twins):
                        twin.copy_(t)
                    for t in outs:
                        t.untyped_storage().fill_(255)
                    why = launch(b, addrs, fill, state)
                    for i in sorted(referenced) if why is None else ():
                        if i < placed and not mutates:
                            ok = torch.equal(torch._C._cuda_hostTraceDigests([copies[i][0]], [True])[1][0], inputs[i])
                        elif i < placed:
                            ok = torch.equal(*(b.view(torch.int64) for b in bases[i]))
                        elif i in tolerated:
                            ok = close(ref_out[i - placed], outs[i - placed], spread[i - placed])
                        else:
                            ok = same(ref_out[i - placed], outs[i - placed])
                        if not ok:
                            return f"the binding's launch differs from eager at operand {i} ({names})"
                    return why

            def failure(bs: list[OpaqueBinding]) -> tuple[int, str] | None:
                for b in bs:
                    for fill in (0, 0xFF):
                        if (why := run(b, fill)) is not None:
                            return fill, why
                return None

            def zeroings(blocks: range) -> list[tuple[int, ...]]:
                # the blocks a memset may zero: one at a time, then all
                return [(j,) for j in blocks] + ([tuple(blocks)] if len(blocks) > 1 else [])

            owned: tuple = ()
            zeroed: tuple[int, ...] = ()
            failed = failure(build(owned, zeroed))
            if failed is not None and failed[0] == 0xFF:
                for z in zeroings(range(first)):
                    if failure(build(owned, z)) is None:
                        zeroed, failed = z, None
                        break
            for h in hits[0] if failed is not None else ():
                for z in ((), (first,)):
                    if failure(build((h,), z)) is None:
                        owned, zeroed, failed = (h,), z, None
                        break
                if failed is None:
                    break
            if failed is not None:
                if in_place:
                    eager()
                where = ", ".join(f"{names[x]} parameter {p} byte {o}: {q:#x}" for x, p, o, q in hits[0])
                raise _Refused(failed[1] + (f"; constant addresses no channel explains: {where}" if where else ""))
            owner = "owned+zeroed" if first in zeroed else "owned"
            records = [(names[x], p, o, q, owner if (x, p, o, q) in owned else "dead") for x, p, o, q in hits[0]]
            records += [(names[x], p, o, q, "misaligned") for x, p, o, q in misaligned]
            records += [("scratch", -1, j, binding.scratch[j], "zeroed") for j in zeroed if j < first]
            if records:
                self.addresses.extend(records)
                msg = f"{op} at sizes {key.sizes}: {records}"
                log.info("host_trace: harvest constant addresses and zeroed scratch %s", msg)
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {"name": "host_trace_harvest_addresses", "encoding": "string"},
                    payload_fn=lambda: msg,
                )
            return build(owned, zeroed, mirrored=False)[0]

        def adopt(cap: tuple, siblings: dict[tuple, OpaqueBinding], placed: list[int]) -> OpaqueBinding:
            # the capture read with the slots of the harvested key launching
            # the same kernels: each slot's qword is its operand or one of the
            # call's blocks here plus a delta, and every other qword a
            # constant: one into an operand or the call's blocks, or a CUDA
            # address the template does not hold, has a role the template's
            # slots do not say
            found, _, blocks, outputs, *_ = cap
            template = siblings[_signature(found)]
            here = placed + outputs
            # a block is its request rounded up to 512 bytes
            extents = [(n + 511) // 512 * 512 for _, n in blocks]
            live = [(a, a + n) for a, n in zip(here, spans)] + [(a, a + n) for (a, _), n in zip(blocks, extents)]

            def moved(t: Slot, q: int) -> Slot:
                if t.kind == "operand" and 0 <= q - here[t.index] < spans[t.index]:
                    return Slot(t.param, t.offset, t.kind, t.index, q - here[t.index])
                if t.kind == "scratch":
                    for j, ((a, _), n) in enumerate(zip(blocks, extents)):
                        if a <= q < a + n:
                            return Slot(t.param, t.offset, t.kind, j, q - a)
                raise _NoSibling(f"{q:#x} is no {t.kind} of the template's")

            nodes: list = []
            for n, t in zip(found, template.nodes):
                if isinstance(n, MemsetNode):
                    nodes.append(OpaqueMemset(moved(t.dst, n.dst), *dataclasses.astuple(n)[1:]))
                    continue
                at = [(s.param, s.offset) for s in t.slots]
                images, qwords, stray = torch._C._cuda_hostTraceSlotQwords(list(n.images), list(t.images), at, live, stack)
                slots = tuple(map(moved, t.slots, qwords))
                if stray is not None:
                    raise _NoSibling("parameter {} byte {}: {:#x} is no constant".format(*stray))
                nodes.append(dataclasses.replace(n, images=tuple(images), slots=slots))
            return _renumbered(nodes, [n for _, n in blocks], 0, cap[6])

        if dev.pool is None:
            with torch.cuda.device(device):
                dev.pool = torch.cuda.MemPool()
                # a graph kept on the pool: the pinned host allocator asserts
                # when a pool id is captured into after all its graphs are gone
                dev.keeper = torch.cuda.CUDAGraph()
                with torch.cuda.stream(s):
                    dev.keeper.capture_begin(pool=dev.pool.id, capture_error_mode="thread_local")
                    dev.anchor.fill_(1)
                    dev.keeper.capture_end()
        own = not torch._C._cuda_isHistoryEnabled()
        smear = _SMEAR
        try:
            # a priming capture, not checked: cuDNN's first capture after an
            # uncaptured run launches a transient parameter image (dead bytes of another host path), and
            # an ATen __host__ __device__ lambda's parameter ends in nvcc's
            # pointer to its host heap copy, at another address in a harvest's
            # first capture than in the ones after it. A sibling's one capture
            # is checked, and primes the full harvest when no harvested key
            # launches its kernels; the two captures of a cuBLAS or attention
            # key take it as their A (a refusal primes the four)
            unprimed = bool(siblings) or (pair and (op in _OUT or op in _ATTENTION))
            try:
                caps = [capture(0, s, sets[0], prime=not unprimed)]
            except _Refused as e:
                raise _NoSibling(str(e)) if siblings else e from None
            if not siblings or _signature(caps[0][0]) not in siblings:
                if pair:
                    caps = (caps if unprimed else [capture(0, s, sets[0])]) + [capture(3, other, sets[1])]
                    smear = (_SMEAR[0], _SMEAR[3])
                else:
                    caps = [
                        capture(0, s, sets[0]),
                        capture(1, s, sets[1]),
                        capture(2, other, sets[0]),
                        capture(3, s, sets[0]),
                    ]
                siblings = None
        finally:
            if own:
                torch._C._cuda_hostTraceRecordAllocations(False)
            # with its graphs reset (a refusal's traceback may hold one) and
            # its tensors gone, the pool's blocks are free: cached for the
            # next harvest up to _POOL_KEEP bytes, else returned to the
            # device by the pool's destructor
            for graph in graphs:
                graph.reset()
            graphs.clear()
            plugs.clear()
            kept.clear()
            if grown and sum(b[2] for b in dev.blocks()) > _POOL_KEEP:
                dev.keeper = None
                dev.pool = None
        blocks = [c[2] for c in caps]
        stack = torch._C._cuda_hostTraceStackBounds()
        heap = _heap_bounds()
        addresses = [[t.data_ptr() for t in sets[w]] + c[3] for w, c in zip((0, 1, 0, 0), caps)]
        if siblings:
            binding = adopt(caps[0], siblings, starts[0])
            del sets
            if _VERIFY_SIBLINGS:
                try:
                    binding = verify(binding, [], caps[0][1])
                except _Refused as e:
                    raise _NoSibling(str(e)) from None
            return binding

        philox = [c[4] for c in caps]
        increment = philox[0][2] if rng else 0
        if rng and any(p[2] != increment for p in philox):
            raise _Refused(f"the captures took {[p[2] for p in philox]} philox offsets")
        # a seeded op outside the rng family that took none draws from a
        # generator the captures don't see (a CPU one), which verify can't pin;
        # attention draws only at a forward with dropout, a backward reads its
        # forward's seed and offset
        if rng and not increment and op not in self.rng and dropout is None:
            raise _Refused(f"{op} is seeded and took no philox offsets")
        names = caps[0][1]
        sizes = [[n for _, n in b] for b in blocks]
        if any(z != sizes[0] for z in sizes):
            raise _Refused(f"the call's allocations differ between captures: {sizes}")
        if len({a for b in blocks for a, _ in b}) != sum(map(len, blocks)):
            raise _Refused("two captures' calls share a block")

        def shape(n: Any) -> Any:
            return (
                dataclasses.astuple(n)[1:]
                if isinstance(n, MemsetNode)
                else (n.function, n.grid, n.block, n.smem, n.attributes, n.params)
            )

        # a kernel's launch attributes are part of the binding's template: an
        # instantiated exec's are fixed (cuGraphExecKernelNodeSetParams sets
        # the function, dims, shared memory and parameters; only a whole-graph
        # cuGraphExecUpdate could change one), so one that moves with the call
        # (an access policy window over an operand) refuses
        varying = {
            k.name
            for c in caps
            for n, m in zip(c[0], caps[0][0])
            if isinstance(n, OpaqueKernel) and isinstance(m, OpaqueKernel)
            for k, _ in set(n.attributes) ^ set(m.attributes)
        }
        if varying:
            raise _Refused(f"the launch attributes {sorted(varying)} differ between captures")
        if any(list(map(shape, c[0])) != list(map(shape, caps[0][0])) for c in caps):
            raise _Refused(
                f"the captures launched different kernels: {[c[1] for c in caps]}"
            )

        def classify(qs: tuple) -> Any:
            # ("operand", i, delta), ("scratch", j, delta), a philox field
            # (the capture's seed or offset address, or its intragraph offset
            # + delta), "host", None for a constant, False for a value that
            # fits no role
            qa = qs[0]
            if increment:
                if all(q == p[0] for q, p in zip(qs, philox)):
                    return ("philox_seed", 0, 0)
                if all(q == p[1] for q, p in zip(qs, philox)):
                    return ("philox_offset", 0, 0)
                d = qa - _PRETAKE[0]
                if 0 <= d < increment and all(q - c[5] == d for q, c in zip(qs, caps)):
                    return ("philox", 0, d)
            for j, (ta, n) in enumerate(blocks[0]):
                # the block is the request rounded up to 512 bytes; one value
                # in every capture is a constant, if into the first's block
                if ta <= qa < ta + (n + 511) // 512 * 512 and len(set(qs)) > 1:
                    d = qa - ta
                    ok = all(q - blocks[x][j][0] == d for x, q in enumerate(qs))
                    return ("scratch", j, d) if ok else False
            if qa != qs[1]:
                roles = [
                    (i, qa - addresses[0][i])
                    for i in range(len(spans))
                    if 0 <= qa - addresses[0][i] < spans[i]
                    and all(
                        q - addresses[x][i] == qa - addresses[0][i]
                        for x, q in enumerate(qs)
                    )
                ]
                if len(roles) == 1:
                    return ("operand", *roles[0])
            # differs between captures of one operand set on one stream, or
            # with the stream alone, or a stack address in any: host state of
            # the call
            if any(q != qa for q in qs[2:]) or any(stack[0] <= q < stack[1] for q in qs):
                return "host"
            return False if qa != qs[1] else None

        nodes: list = []
        stale: list[tuple[int, int, int, bytes]] = []  # (node, parameter, byte, another capture's bytes)
        for x, ns in enumerate(zip(*(c[0] for c in caps))):
            a = ns[0]
            if isinstance(a, MemsetNode):
                role = classify(tuple(n.dst for n in ns))
                if not role or role == "host" or role[0] not in ("operand", "scratch"):
                    raise _Refused(f"a memset targets {a.dst:#x}, no operand or scratch")
                kind, index, delta = role
                nodes.append(OpaqueMemset(Slot(0, 0, kind, index, delta), *dataclasses.astuple(a)[1:]))
                continue
            images = []
            slots = []
            for param in range(len(a.params)):
                imgs = tuple(n.images[param] for n in ns)
                image = bytearray(imgs[0])
                covered = bytearray(len(image))
                # the bits where the captures' bytes differ: a qword equal in
                # all of them is host if a stack address, else a constant
                ints = [int.from_bytes(img, "little") for img in imgs]
                differ = 0
                for v in ints[1:]:
                    differ |= v ^ ints[0]
                for first in (0, 4):
                    for i, q in enumerate(_qwords(imgs[0], first)):
                        off = first + 8 * i
                        varies = increment or (differ >> (8 * off)) & _QWORD
                        if (varies or stack[0] <= q < stack[1]) and not any(covered[off : off + 8]):
                            qs = tuple(int.from_bytes(img[off : off + 8], "little") for img in imgs)
                            role = classify(qs) if varies else "host"
                        elif varies or any(covered[off : off + 8]):
                            continue
                        else:
                            role = None
                        if role is None:
                            # a constant CUDA address is verify's to explain
                            continue
                        if role is False and op not in _CONV:
                            continue
                        if role in ("host", False):
                            if not _host_slot(off, imgs, stack, smear):
                                # stale host bytes of a cuDNN call: its engines
                                # build their parameter structs on the heap,
                                # where the harvest flag zeroes nothing, fields
                                # of another path left over, which may differ
                                # with the operands; if no capture's value is
                                # an address CUDA knows, the check also
                                # launches another capture's (or if only the high
                                # half varies: an int32 under a stale high half,
                                # no pointer into memory that moves); likewise the
                                # padding after ATen's ReduceOp functor, which
                                # holds bytes TensorIterator's frames left there
                                where = f"{names[x]} parameter {param} byte {off}"
                                heaped = op in _OUT and all(heap[0] <= q < heap[1] for q in qs)
                                if op not in _CONV and op not in _REDUCE and not heaped:
                                    raise _Refused(f"{where}: unexplained varying parameter")
                                if len({q & 0xFFFFFFFF for q in qs}) > 1 and any(map(_cuda_address, qs)):
                                    seen = " / ".join(img[off : off + 8].hex() for img in imgs)
                                    raise _Refused(f"{where}: host state {seen}")
                                other = next(img for img in imgs if img[off : off + 8] != imgs[0][off : off + 8])
                                stale.append((len(nodes), param, off, other[off : off + 8]))
                        else:
                            kind, index, delta = role
                            slots.append(Slot(param, off, kind, index, delta))
                            image[off : off + 8] = bytes(8)
                        covered[off : off + 8] = b"\x01" * 8
                while differ:
                    off = ((differ & -differ).bit_length() - 1) // 8
                    differ &= ~(0xFF << (8 * off))
                    seen = tuple(img[off] for img in imgs)
                    # the smear's bytes: uninitialized padding
                    if not covered[off] and len(set(seen)) > 1 and seen != smear:
                        lo = off - off % 8
                        seen = " / ".join(img[lo : lo + 8].hex() for img in imgs)
                        where = f"{names[x]} parameter {param} byte {off}"
                        raise _Refused(f"{where}: per-call state of no role: {seen}")
                images.append(bytes(image))
            nodes.append(dataclasses.replace(a, images=tuple(images), slots=tuple(slots)))

        binding = _renumbered(nodes, sizes[0], increment, tuple(c[6] for c in caps if c[6] is not None) or None)
        kinds = {t.kind for n in binding.nodes for t in _slots(n)}
        if increment and not {"philox_seed", "philox_offset", "philox"} <= kinds:
            raise _Refused(f"{names} take {increment} philox offsets and read no captured generator state")
        del sets

        return verify(binding, stale, names)
