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
first capture). The four capture into a pool of the harvest's own, back to the
device once they are over (a binding's constant into it is checked as below);
verify's buffers come from a pool of their own, back once the harvest is over. The pool's own log of its
allocations during the capture says which are the call's: the call's blocks are the pool's free
blocks holding one, and each is one scratch buffer as far as they reach in it
(the allocations a call frees coalesce). The blocks a capture's call allocated
are plugged before the next capture, so each capture's sit at their own
addresses. The call's cuBLAS workspace is one of its allocations, as in
eager and at eager's size. Every
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

A cuBLAS, SDPA or cuDNN convolution key launching the kernels (by name: each
cuDNN plan loads its own copy) of a key harvested in full (another M or batch:
the heuristic's choice repeats across shapes) is its sibling: one capture, A, read with the harvested key's
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
alike, and so is one shaped like a device address in every capture (an
earlier call's workspace, which may be in a released pool: each call takes a
fresh workspace, as eager's per-call allocation), retired: verify mirrors and
poisons it whether or not it is mapped. A binding holds its cuDNN plan
(the owner ATen notes at the capture), whose kernels it launches.

An RNG call (randint, uniform_, SDPA with dropout) reads the default
generator's per-capture seed and offset, and an intragraph offset: capture x
first takes _PRETAKE[x] offsets, so the three are told apart from operands.
They become philox slots, the offsets the call took its rng_increment (equal
in the four), and the check launches from the generator's offset.

An "extern" op (a custom op of a kernel library, allowlisted by the caller)
is checked on its live read-only operands, whose values may index, and on
copies of the live bytes of the operands it writes, refused above the
provider's extern_cap bytes; its written floating operands are random in the copies, its other
written operands zeros (valid indices and counts; the call ran before the harvest: a binding that
writes nothing would match its result).
When eager on the random copy gives the live bytes back, the op writes those
operands without reading them: the binding is checked on that one copy against
the live bytes (B(R) == live == E(R)), else on a second copy against eager's.
Operands on one storage (OpaqueKey.alias, q and k views of one qkv) lie in
one arena block at their distance, in each set and copy: a qword in several
of them is the one it is fewest bytes into. A declared KV writer (`indexed`:
its index operand, the caches it indexes, and how many leading cache dims an
index spans, 1 if not given) is checked on caches of at least 2n + 1 slots
(dim 0 cut to fewer rows) at the live strides, random, and n distinct indices
into them: the binding captured at the live caches' sizes writes the slots it
is given. Caches on one storage (K and V interleaved in one page) are one
block at their live distance, so a write past a K slot into V is compared too.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import functools
import logging
import math
import os
import pickle
import struct
import threading
import time
import traceback
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch.utils._triton import has_triton
from torch._logging import trace_structured
from torch._ops import OpOverload
from torch.cuda._host_trace_capture import attribute_value, launch_attributes, MemsetNode, reproducible
from torch.cuda._host_trace_opaque import (
    fill_uniform,
    OpaqueBinding,
    OpaqueKernel,
    OpaqueKey,
    OpaqueMemset,
    Slot,
    library_state_as,
    span_bytes,
    takes_generator,
)
from torch.cuda._host_trace_tape import _cuda_address, _gc_hold, new_private_stream, seed_offset_on_device
from torch.cuda._host_trace_native import NativeBindings
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
    aten.mm.dtype: "aten::mm",
    aten.mm.dtype_out: "aten::mm",
}
# mm with an output dtype (bf16 / fp16 operands into a float32 output, as a
# router GEMM): harvested through mm.dtype_out, whose out_dtype is the output's
_OUT_DTYPE = frozenset({aten.mm.dtype, aten.mm.dtype_out})


def _out_call(op: Any, name: str, operands: Sequence[torch.Tensor]) -> functools.partial:
    """The out= overload an accepted op is harvested through, bound to its
    operands (the output last)."""
    *inputs, out = operands
    if op in _OUT_DTYPE:
        return functools.partial(aten.mm.dtype_out, *inputs, out.dtype, out=out)
    return functools.partial(getattr(aten, name.removeprefix("aten::")).out, *inputs, out=out)

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
_FAMILIES = {"blas": frozenset(_OUT), "attention": _ATTENTION, "conv": _CONV, "rng": _RNG, "reduce": _REDUCE, "extern": frozenset()}


_WINDOW = 1 << 21
_QWORD = (1 << 64) - 1
_RETRIES = 3  # harvests of one key refused for a transient reason
_SMEAR = (0xA5, 0xA5, 0xA5, 0x5A)  # per capture A, B, C, D
# per capture, the philox offsets taken before an RNG call: its intragraph
# offset moves with them
_PRETAKE = (0, 4, 8, 12)
# seconds a checked launch may run before its scratch is zeroed from another
# stream (a kernel spinning on a semaphore it expects zeroed); once a launch of
# the binding (or the eager op, timed before them) has finished, _HANG_FACTOR
# times its time (at least _HANG_MIN)
_HANG = 10.0
_HANG_FACTOR, _HANG_MIN = 20, 0.2
# a mirror of the allocation a constant address points into spans from this
# far below the lowest such address to this far past the highest
_MIRROR_BELOW, _MIRROR_ABOVE = 2 << 20, 64 << 20
# a mirror's address range past its 4 GiB is reserved in multiples of this
_MIRROR_SPAN_STEP = 128 << 20


class _Refused(Exception):
    def __init__(self, msg: str, structural: bool = False) -> None:
        super().__init__(msg)
        # a property of the call's kernels, not of its sizes: the op refuses at
        # every key of its layout class (HarvestProvider._layout_class)
        self.structural = structural


class _Transient(_Refused):
    # a refusal a later harvest of the key may not repeat
    pass


class _Disabled(Exception):
    # the provider cannot harvest in this process
    pass


class _NotBitwise(Exception):
    # eager does not give the call's outputs bitwise again: verify on copies,
    # with its spread
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
# bytes of the harvest pool kept cached between harvests: none, it goes back
# to the device after each harvest's captures
_POOL_KEEP = 0
# operand bytes an extern key's check copies, at most, for a provider whose
# extern_cap is None: a KV pool layer is past it, FlashInfer's 512 MiB
# trtllm-gen workspace is not
_EXTERN_CAP = 1 << 30
# an extern op that writes its operands without reading them checks its
# binding on one refilled copy against the live bytes; False checks it on
# fresh random copies like any mutating op
_EXTERN_FAST = True
# input bytes past which an extern key learns without its call only on a
# view of one of the provider's `lendable` tensors (a KV pool layer)
_LENT = 256 << 20
# a check on random operands digests the inputs it must leave unchanged once,
# after all its launches (a difference refuses the key); False digests them
# after each launch (a difference fails that binding)
_DIGEST_ONCE = True


def _class(align: int) -> int:
    # the largest power of two dividing the address, up to 256 (ATen's
    # _getAlignment): all cuBLAS's kernel choice reads of an address
    return align & -align if align else 256


_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
# bytes a byte comparison compares at once (torch.equal holds a bool per element)
_EQUAL_PIECE = 16 << 20


def _equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and all(torch.equal(a[j : j + _EQUAL_PIECE], b[j : j + _EQUAL_PIECE]) for j in range(0, a.numel(), _EQUAL_PIECE))


def _rows(t: torch.Tensor, contiguous: bool) -> torch.Tensor:
    # a long 1-D (with `contiguous`, any contiguous) operand in rows of 4096:
    # _cuda_hostTraceDigests copies a single row, and short rows digest at a
    # fraction of the bandwidth
    flat = t.is_contiguous() if contiguous else t.dim() == 1 and t.stride(0) == 1
    return t.view(-1, 4096) if flat and t.numel() > 4096 and t.numel() % 4096 == 0 else t


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
    # the start and bytes of the allocation holding a CUDA address; one no
    # longer mapped (a retired qword into a released pool) spans the mirror's
    # whole window
    from cuda.bindings import driver

    a = driver.CUpointer_attribute
    err, start = driver.cuPointerGetAttribute(a.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, q)
    if err != driver.CUresult.CUDA_SUCCESS:
        return q - _MIRROR_BELOW, _MIRROR_BELOW + _MIRROR_ABOVE
    size = _check_cuda_bindings(driver.cuPointerGetAttribute(a.CU_POINTER_ATTRIBUTE_RANGE_SIZE, q))
    return int(start), int(size)


def _classified(role: Any, qs: tuple[int, ...]) -> Any:
    """The role extraction takes for a varying qword (values qs, one per
    capture): classify's. A test seam."""
    return role


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
        # the bytes a slot's qword starts at, and 4 either side of it
        near = {(s.param, s.offset + d) for s in n.slots for d in (-4, 0, 4)}
        for param, image in enumerate(n.images):
            for first, hits in ((0, aligned), (4, misaligned)):
                for i, q in enumerate(_qwords(image, first)):
                    if not q >> 32 or stack[0] <= q < stack[1] or (param, off := first + 8 * i) in near:
                        continue
                    if q not in known:
                        known[q] = _cuda_address(q)
                    if known[q]:
                        hits.append((x, param, off, q))
    return aligned, misaligned


@contextlib.contextmanager
def _released(device: int) -> Iterator[None]:
    """Nothing of verify's outlives its harvest but blocks the caching
    allocator keeps for reuse: an exception's finished frames drop their
    locals (verify's tensors; the exiting with statement holds its
    traceback, its lines stay), and the cuBLASLt workspaces verify's GEMMs
    cached go."""
    try:
        yield
    except BaseException as e:
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        if device in _DEVICES:
            # verify's eager GEMMs on a stream with no cuBLASLt workspace yet
            # cached one (ATen keeps one per handle and stream)
            torch._C._cuda_clearCublasWorkspaces()


# _digest's words per chunk: its temporaries are a few int64 buffers this long
_DIGEST_CHUNK = 1 << 19
# splitmix64's constants (as int64), and the two lanes' seeds
_GOLDEN, _MIX1, _MIX2 = -0x61C8864680B583EB, -0x40A7B892E31B1A47, -0x6B2FB644ECCEEE15
_LANES = (0x2545F4914F6CDD1D, 0x1B873593A1E5F2C7)


def _chunks(x: torch.Tensor, n: int):
    # x's elements in index order as 1-D pieces of at most about n elements
    # (a piece of a non-contiguous tensor is a copy of that piece only)
    if x.numel() <= n or x.dim() == 0:
        yield x.reshape(-1)
        return
    d = next(i for i in range(x.dim()) if x.shape[i] > 1)
    rows = max(1, n * x.shape[d] // x.numel())
    for piece in x.split(rows, d):
        yield from _chunks(piece.squeeze(d) if rows == 1 else piece, n)


_DIGEST_DIMS = 6


@functools.cache
def _digest_kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x, out, n, base, golden, mix1, mix2, seed0, seed1, s0, s1, s2, s3, s4, s5, l0, l1, l2, l3, l4, l5, DIMS: tl.constexpr, BLOCK: tl.constexpr):  # noqa: N803
        # x's elements in memory order; each one's index-order position from
        # its coordinates (dims innermost first: sizes s, index strides l)
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        rem = offs
        index = (rem % s0) * l0
        rem = rem // s0
        if DIMS > 1:
            index += (rem % s1) * l1
            rem = rem // s1
        if DIMS > 2:
            index += (rem % s2) * l2
            rem = rem // s2
        if DIMS > 3:
            index += (rem % s3) * l3
            rem = rem // s3
        if DIMS > 4:
            index += (rem % s4) * l4
            rem = rem // s4
        if DIMS > 5:
            index += (rem % s5) * l5
        v0 = tl.load(x + offs, mask=mask, other=0).to(tl.int64) + (index + base) * golden
        for lane in tl.static_range(2):
            v = v0 + (seed0 if lane == 0 else seed1)
            v = (v ^ ((v >> 30) & 0x3FFFFFFFF)) * mix1
            v = (v ^ ((v >> 27) & 0x1FFFFFFFFF)) * mix2
            v = v ^ ((v >> 31) & 0x1FFFFFFFF)
            tl.atomic_add(out + lane, tl.sum(tl.where(mask, v, 0), axis=0))

    return kernel


def _dense_dims(x: torch.Tensor) -> list[tuple[int, int]] | None:
    # (size, index-order stride) per dim, innermost in memory first, when x's
    # elements fill its span in some order of its dims (contiguous, channels
    # last, a transpose); None for a view with gaps or overlaps
    index, stride = [1] * x.dim(), 1
    for d in reversed(range(x.dim())):
        index[d], stride = stride, stride * x.shape[d]
    dims = sorted(((x.stride(d), x.shape[d], index[d]) for d in range(x.dim()) if x.shape[d] > 1), key=lambda t: t[0])
    expect = 1
    for st, size, _ in dims:
        if st != expect:
            return None
        expect *= size
    return [(size, i) for _, size, i in dims] or [(1, 1)]


def _digest(t: torch.Tensor) -> torch.Tensor:
    """A strong digest of t's bits in index order: per element of its integer
    view of the same itemsize, splitmix64 of the element plus its index times
    the golden ratio, plus a lane's seed, summed per lane (two lanes, 128
    bits). Integer arithmetic throughout, so it does not depend on the order a
    reduction takes. On CUDA one fused kernel reads a dense tensor in memory
    order (any order of its dims); a view with gaps copies pieces of
    _DIGEST_CHUNK words for it; elsewhere torch ops on such pieces."""
    if t.is_complex():
        t = torch.view_as_real(t)
    x = t.view(_BITS[t.element_size()])
    out = torch.zeros(len(_LANES), dtype=torch.int64, device=t.device)
    if not x.numel():
        return out
    fused = x.is_cuda and has_triton()
    dims = _dense_dims(x) if fused else None
    if dims is not None and len(dims) <= _DIGEST_DIMS:
        pieces = [(x, dims)]
    else:
        # one copied piece at a time
        pieces = ((piece, [(piece.numel(), 1)]) for piece in _chunks(x, _DIGEST_CHUNK))
    base = 0
    for piece, dims in pieces:
        n = piece.numel()
        if fused:
            sizes, strides = zip(*(dims + [(1, 0)] * (_DIGEST_DIMS - len(dims))))
            _digest_kernel()[(-(-n // 1024),)](piece, out, n, base, _GOLDEN, _MIX1, _MIX2, *_LANES, *sizes, *strides, DIMS=len(dims), BLOCK=1024)
            base += n
            continue
        v0 = piece.to(torch.int64) + (torch.arange(base, base + n, device=t.device) * _GOLDEN)
        base += n
        for lane, seed in enumerate(_LANES):
            v = v0 + seed
            v = (v ^ ((v >> 30) & ((1 << 34) - 1))) * _MIX1
            v = (v ^ ((v >> 27) & ((1 << 37) - 1))) * _MIX2
            v = v ^ ((v >> 31) & ((1 << 33) - 1))
            out[lane] += v.sum()
    return out


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors of one shape hold the same bits in index order (+0
    and -0, and NaN payloads, told apart): their _digests, which allocate
    chunks, not a buffer the size of either."""
    return torch.equal(_digest(a), _digest(b))


def _gap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # elementwise distance in float64; equal values (infinities, NaNs) are 0
    # apart: random operands overflow a few elements of some outputs
    a, b = a.double(), b.double()
    return torch.where((a == b) | (a.isnan() & b.isnan()), 0.0, (a - b).abs())


def _gap_max(a: torch.Tensor, b: torch.Tensor) -> float:
    """The largest _gap of two tensors of one shape, over _DIGEST_CHUNK-element
    pieces (float64 temporaries of a piece, not of the tensor)."""
    pieces = zip(_chunks(a, _DIGEST_CHUNK), _chunks(b, _DIGEST_CHUNK))
    return max((_gap(x, y).nan_to_num(0.0, 0.0).max().item() for x, y in pieces if x.numel()), default=0.0)


def _close(want: torch.Tensor, got: torch.Tensor, noise: float) -> bool:
    """Whether got is want to within 4x eager's own spread `noise` plus the
    dtype's testing tolerance, piece by piece: a wrong address or size gives
    garbage, not rounding noise."""
    rtol, atol = default_tolerances(want)
    for x, y in zip(_chunks(want, _DIGEST_CHUNK), _chunks(got, _DIGEST_CHUNK)):
        d = _gap(x, y)
        if not bool((d.isfinite() & (d <= 4 * noise + atol + rtol * x.double().abs())).all()):
            return False
    return True

def _patched(image: bytes, off: int, q: int) -> bytes:
    return image[:off] + q.to_bytes(8, "little") + image[off + 8 :]


# where the harvest asks for its address ranges (the operand arena, mirror
# spans): among the caching allocator's segments, not at the low addresses the
# driver hands out first, which a kernel's small int pairs (a size less one
# under another) read as, past an operand of the arena or a mirror
_HIGH_VA = 0xFF00_0000_0000


# the harvest's reservations the driver placed at their hints, start -> bytes: their addresses' high halves are in
# no host or allocator range of the process (a reservation placed elsewhere may share its high half with host
# mappings), so a constant of a cuDNN parameter struct holding one is the harvest's leftover (_own_highs)
_HINTED: dict[int, int] = {}
# a reservation within this many bytes above its hint is at its hint
_HINT_BAND = 1 << 39


def _reserve_high(size: int, align: int, at: int = _HIGH_VA) -> int:
    from cuda.bindings import driver

    hint = -(-at // align) * align
    va = int(_check_cuda_bindings(driver.cuMemAddressReserve(size, align, hint, 0)))
    if hint <= va < hint + _HINT_BAND:
        _HINTED[va] = size
    return va


class _AddressOnly:
    """The captures' allocator: address space and no memory. A capture records
    addresses and its graph is reset unlaunched, so nothing it allocates is
    ever touched (the operand arena's reasoning, for the call's own
    allocations, outputs and the plugs). Per device a reserved range handed out
    upward, rewound once every piece is back."""

    _SPAN = 64 << 30

    def __init__(self) -> None:
        alloc_fn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)
        free_fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)
        self.ranges: dict[int, list[int]] = {}  # device -> [base, next, pieces out]
        self.c_alloc, self.c_free = alloc_fn(self._alloc), free_fn(self._free)
        self.allocator = torch._C._cuda_customAllocator(ctypes.cast(self.c_alloc, ctypes.c_void_p).value, ctypes.cast(self.c_free, ctypes.c_void_p).value)

    def _alloc(self, size: int, device: int, stream: Any) -> int:
        r = self.ranges.get(device)
        if r is None:
            base = _reserve_high(self._SPAN, 1 << 21, 0xFE00_0000_0000)
            r = self.ranges[device] = [base, base, 0]
        at = -(-r[1] // _WINDOW) * _WINDOW
        if at + size > r[0] + self._SPAN:
            return 0
        r[1], r[2] = at + size, r[2] + 1
        return at

    def _free(self, ptr: int, size: int, device: int, stream: Any) -> None:
        r = self.ranges[device]
        r[2] -= 1
        if r[2] == 0:
            r[1] = r[0]


@functools.cache
def _address_only() -> _AddressOnly:
    return _AddressOnly()


def _unfilled(nbytes: int, device: int) -> torch.Tensor:
    """`nbytes` from the current allocation pool, launching nothing: an address-only pool's blocks are never mapped, and
    torch.empty under use_deterministic_algorithms(True) (fill_uninitialized_memory) launches a fill into them, an
    illegal address. A storage of the allocator is no factory op and never filled."""
    return torch.empty(0, dtype=torch.uint8, device=device).set_(torch.UntypedStorage(nbytes, device=device))


# per device, the address ranges mirrors map in (4 GiB-aligned, free ones),
# reserved for the process: reserved and freed per check, their addresses (low
# ones, for the alignment) went to the caching allocator's next segments,
# where an int pair in a kernel's parameters reads as a mapped address
_MIRROR_SPANS: dict[int, list[tuple[int, int]]] = {}


class _Mirrors:
    """Memory of the harvest's own standing in for the allocations constant
    addresses point into: a window of each (_MIRROR_BELOW below the lowest
    such address in it to _MIRROR_ABOVE past the highest), mapped at a shift
    of a multiple of 4 GiB, so an address's low half is kept (an int32 field
    under a stale high half reads the same)."""

    def __init__(self, device: int, addresses: Iterable[int]) -> None:
        from cuda.bindings import driver

        self.device = device
        self.spans: list[tuple[int, int]] = []  # taken from _MIRROR_SPANS
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
                size = hi - lo
                free = _MIRROR_SPANS.setdefault(device, [])
                span = min((x for x in free if x[1] >= (1 << 32) + size), default=None, key=lambda x: x[1])
                if span is None:
                    reserved = (1 << 32) + -(-size // _MIRROR_SPAN_STEP) * _MIRROR_SPAN_STEP
                    span = (_reserve_high(reserved, 1 << 32), reserved)
                else:
                    free.remove(span)
                self.spans.append(span)
                va = span[0]
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
        # in slices: a window is up to 66 MiB, a whole compare's bool buffer as big
        return any(bool((c != value).any()) for t in self.tensors for c in t.split(1 << 22))

    def close(self) -> None:
        torch.cuda.synchronize()
        self.tensors.clear()
        while self._undo:
            _check_cuda_bindings(self._undo.pop()())
        _MIRROR_SPANS.setdefault(self.device, []).extend(self.spans)
        self.spans.clear()


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
        self.streams = (new_private_stream(dev), new_private_stream(dev))
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
                _HINTED.pop(self.va, None)
                self.va = self.va_size = 0
            n = max(1 << 28, 1 << (size - 1).bit_length())
            self.va, self.va_size = _reserve_high(n, _WINDOW), n
        return self.va

    def blocks(self) -> list:
        """The pool's blocks as (stream, address, size, requested size,
        active, in the small pool)."""
        return torch._C._cuda_hostTracePool(self.index, self.pool.id)

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
                    t = _unfilled(piece, self.index)
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


def _fresh(rets: list, returns: list[bool]) -> list[torch.Tensor]:
    """The tensors of the returns that are fresh outputs."""
    return [o for r, fresh in zip(rets, returns) if fresh for o in pytree.tree_leaves(r) if isinstance(o, torch.Tensor)]


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


def _libraries() -> dict[str, tuple[str, int]]:
    # the CUDA libraries loaded in the process: file name -> (path, size)
    out = {}
    with open("/proc/self/maps") as f:
        for line in f:
            path = line.split()[-1]
            name = os.path.basename(path)
            if name.startswith(("libcublas", "libcudnn", "libcuda.so", "libcudart")) and name not in out and os.path.exists(path):
                out[name] = (path, os.stat(path).st_size)
    return out


def _same_meta(saved: dict[str, Any], now: dict[str, Any]) -> bool:
    # a library saved but not loaded yet (CUDA loads lazily) is its file at the saved path
    if {k: v for k, v in saved.items() if k != "libraries"} != {k: v for k, v in now.items() if k != "libraries"}:
        return False
    loaded = now["libraries"]
    for name, (path, size) in saved["libraries"].items():
        if name in loaded:
            if loaded[name] != (path, size):
                return False
        elif not (os.path.exists(path) and os.stat(path).st_size == size):
            return False
    return True


def _meta(device: int, versions: dict[str, str] | None) -> dict[str, Any]:
    """What a stored binding holds under: the GPU, driver, torch and the CUDA
    libraries loaded (by file name and size), and the caller's `versions`."""
    from cuda.bindings import driver

    props = torch.cuda.get_device_properties(device)
    return {
        "gpu": (props.name, props.major, props.minor, props.multi_processor_count),
        "driver": _check_cuda_bindings(driver.cuDriverGetVersion()),
        "torch": (torch.__version__, torch.version.git_version, torch.version.cuda),
        "libraries": _libraries(),
        "versions": dict(versions or {}),
        # 2: Slot, OpaqueKernel and OpaqueMemset have __slots__, which a
        # version 1 file's dict states do not unpickle into
        "format": 2,
    }


def _portable(k: tuple) -> tuple:
    # a normal key with its operator by name
    op = k[1]
    return (k[0], op if isinstance(op, str) else (op._schema.name, op._overloadname), *k[2:])


class _RestoreOnly:
    """preload's provider for HostTraceReplay._learn: a stored key's restore, nothing harvested."""

    def __init__(self, provider: HarvestProvider) -> None:
        self.provider = provider

    def refusal(self, key: OpaqueKey) -> str | None:
        return self.provider.refusal(key)

    def operands(self, key: OpaqueKey, inputs: int) -> dict[int, torch.Tensor] | None:
        return self.provider.operands(key, inputs)

    def learn(self, key: OpaqueKey, args: tuple, kwargs: dict, operands: list[Any]) -> OpaqueBinding | None:
        p = self.provider
        k = p._normal(key)
        with p._lock, torch.cuda.device(key.device):
            return p._restore(key, k, args, kwargs) if k in p.stored else None


# a stored key's operator by its normal name (_OUT): the default overload, whose call the warm-up's learn makes
_DEFAULT_OF = {"aten::mm": aten.mm.default, "aten::addmm": aten.addmm.default, "aten::bmm": aten.bmm.default}


def _synthetic_call(k: tuple) -> tuple[OpaqueKey, tuple[Any, frozenset[int]]] | None:
    """A stored key as an OpaqueKey of a callable operator and its call's pytree spec (tensor leaves' positions; the key's
    scalars in the other arguments' places, in schema order), as the offline learn builds it; None for a key it cannot
    call (a mixed-dtype mm)."""
    alias, op, dtypes, sizes, strides, classes, scalars, device, state = _local(k)
    if isinstance(op, str):
        if op == "aten::mm" and dtypes[-1] != dtypes[0]:
            return None
        op = _DEFAULT_OF.get(op)
        if op is None:
            return None
    key = OpaqueKey(op, dtypes, sizes, strides, tuple(0 if c == 256 else c for c in classes), scalars, device, state, alias)
    tensors = len(dtypes) - sum(1 for r in op._schema.returns if r.alias_info is None)
    leaves: list[Any] = []
    positions: list[int] = []
    given = iter(scalars)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for arg in op._schema.arguments:
        is_tensor = isinstance(arg.type, torch.TensorType) or str(arg.type).startswith("Tensor")
        if is_tensor and len(positions) < tensors:
            positions.append(len(leaves))
            v: Any = object()  # a tensor leaf's place (_learn puts the synthetic tensor there)
            leaves.append(v)
        elif is_tensor:
            continue
        else:
            try:
                v = next(given)
            except StopIteration:
                continue
            leaves.append(v)
        if arg.kwarg_only:
            kwargs[arg.name] = v
        else:
            args.append(v)
    return key, (pytree.tree_flatten((tuple(args), kwargs))[1], frozenset(positions))


def _local(k: tuple) -> tuple:
    op = k[1]
    if not isinstance(op, str):
        ns, name = op[0].split("::")
        op = getattr(getattr(getattr(torch.ops, ns), name), op[1])
    return (k[0], op, *k[2:])


def _scratch_renumbered(nodes: Iterable[Any], renumber: dict[int, int]) -> tuple:
    def moved(t: Slot) -> Slot:
        return dataclasses.replace(t, index=renumber[t.index]) if t.kind == "scratch" else t

    return tuple(
        dataclasses.replace(n, dst=moved(n.dst)) if isinstance(n, OpaqueMemset) else dataclasses.replace(n, slots=tuple(map(moved, n.slots)))
        for n in nodes
    )


def _renumbered(nodes: list, sizes: list[int], increment: int, owner: Any = None, keep: bool = False) -> OpaqueBinding:
    # only the scratch some slot points into, renumbered; with keep every block (a cuBLAS call's
    # workspace, which eager allocates whether or not the kernel reads it)
    used = list(range(len(sizes))) if keep else sorted({t.index for n in nodes for t in _slots(n) if t.kind == "scratch"})
    renumber = {j: i for i, j in enumerate(used)}
    return OpaqueBinding(_scratch_renumbered(nodes, renumber), tuple(sizes[j] for j in used), increment, owner)


def _shared(binding: OpaqueBinding) -> OpaqueBinding:
    # scratch blocks eager has live one after the other in one block, at
    # offset 0, as its allocator gives a freed workspace to the next request:
    # a block joins an earlier one whose later nodes all take the new block
    # too (they are the new block's call, holding a stale pointer into the
    # freed one, as cuDNN's wgrad into dgrad's workspace); a block no node
    # reads keeps its own. verify judges the shared form
    users: dict[int, set[int]] = {}
    for x, n in enumerate(binding.nodes):
        for t in _slots(n):
            if t.kind == "scratch":
                users.setdefault(t.index, set()).add(x)
    groups: list[tuple[set[int], int]] = []  # (the nodes using a shared block, its bytes)
    renumber = {}
    for j, size in enumerate(binding.scratch):
        mine = users.get(j)
        i = next((i for i, (used, _) in enumerate(groups) if mine and used and all(x in mine for x in used if x >= min(mine))), len(groups))
        if i == len(groups):
            groups.append((set(mine or ()), size))
        else:
            groups[i] = (groups[i][0] | mine, max(groups[i][1], size))
        renumber[j] = i
    if len(groups) == len(binding.scratch):
        return binding
    return dataclasses.replace(binding, nodes=_scratch_renumbered(binding.nodes, renumber), scratch=tuple(n for _, n in groups))


_STALE_STRUCT = "the harvest's own address high half in a cuDNN parameter struct (refused, never rewritten)"


def _stale_constant(binding: OpaqueBinding, stale: Callable[[int], bool]) -> tuple[int, int, int, int] | None:
    # the first constant qword (alike in every capture, no slot's, not loose)
    # `stale` holds, as (node, parameter, byte, qword). cuDNN builds its
    # parameter structs on the heap before the captures and a field may hold
    # another's leftovers: the high half of one of the harvest's own addresses
    # above an int32 is a byte eager's struct would not hold (CUTLASS strided
    # dgrad reads that half as a flag and hangs on 0xff11's). Such a binding is
    # refused before any launch, never rewritten: a rewritten word may be one
    # the kernel reads (a CUtensorMap's, also inside a CUTLASS 3 Params
    # struct), and verify's one launch does not prove it never does
    for x, n in enumerate(binding.nodes):
        if not isinstance(n, OpaqueKernel):
            continue
        near = {(s.param, s.offset + d) for s in n.slots for d in range(-7, 8)}
        loose = {(p, o + i) for p, o, size in n.loose for i in range(size)}
        for param, image in enumerate(n.images):
            for i, q in enumerate(_qwords(image, 0)):
                off = 8 * i
                if q >> 32 and (param, off) not in near and not any((param, off + b) in loose for b in range(8)) and stale(q):
                    return x, param, off, q
    return None


def _own_highs() -> set[int]:
    # the high halves of the harvest's reservations at their hints (_HINTED: the operand arena, the captures'
    # address-only spans, the mirror spans): 0xFE00.. and 0xFF00.. as reserved today
    return {h for a, n in _HINTED.items() for h in range(a >> 32, ((a + n - 1) >> 32) + 1)}


class HarvestProvider:
    """The OpaqueProvider for the ops of `families`: "blas", bf16, fp16 and
    fp32 mm, addmm and bmm through cuBLAS; "attention", the cuDNN, flash and
    memory-efficient SDPA forwards and backwards; "conv", cuDNN's convolution
    forward and backward; "rng", randint's out= form, native_dropout, uniform_
    and any other ATen op that draws from the default generator (bernoulli_,
    normal_, rand_like), and `rng_ops` (Inductor's inductor_seeds); "reduce",
    ATen's sum over dims; "extern", `extern_ops`, and of them the KV writers
    `indexed` declares: op -> (the index argument's name, the names of the
    caches it indexes[, the leading cache dims an index spans]). An
    extern key's check copies at most extern_cap bytes of written operands
    (None: the module's _EXTERN_CAP, read at each harvest), and digests its
    read-only inputs' row and column sums, or with extern_digest "rows" their
    row sums only (a stray write that only permutes values within a row of
    4096 passes "rows"; it is what the in-place path checks). An extern key
    learned without its call takes its inputs past _LENT bytes as views of
    `lendable` tensors of their dtype, which it reads and does not write. A
    group on one storage (OpaqueKey.alias: K and V views of one KV layer) is
    views of one lendable tensor at the key's distances; without
    `lend_alias_groups` such a key refuses."""

    def __init__(
        self,
        families: tuple[str, ...] = ("blas",),
        rng_ops: tuple[Any, ...] = (),
        extern_ops: tuple[Any, ...] = (),
        indexed: dict[Any, tuple[str | int, ...]] | None = None,
        lendable: tuple[torch.Tensor, ...] = (),
        extern_cap: int | None = None,
        extern_digest: str = "full",
        lend_alias_groups: bool = True,
    ) -> None:
        if extern_digest not in ("full", "rows"):
            raise ValueError(f"extern_digest {extern_digest!r}")
        self.extern_cap = extern_cap
        self.extern_digest = extern_digest
        self.rng = _RNG | frozenset(rng_ops) if "rng" in families else frozenset()
        self.extern = frozenset(extern_ops) if "extern" in families else frozenset()
        self.aliased = self.extern
        self.indexed = {op: names for op, names in (indexed or {}).items() if op in self.extern}
        self.ops = frozenset().union(*(_FAMILIES[f] for f in families)) | self.rng | self.extern
        self.harvests = 0
        self.bindings: dict[tuple, OpaqueBinding] = {}
        # the bindings as native tables (_host_trace_native.NativeBindings): a normal key's parts as _normal has them
        self.native = NativeBindings(
            lambda k: None if k[0] else k[1:], lambda op: _OUT.get(op, op), align_class=True, size1_strides=lambda op: op in _CUDNN
        )
        # per (op, dtypes, device) of a blas or attention key: the harvested
        # bindings by _signature
        self.templates: dict[tuple, dict[tuple, OpaqueBinding]] = {}
        self.refused: dict[tuple, str] = {}
        # structural refusals by _layout_class: an op whose kernels refuse at
        # one key refuses at every key of the class, with no harvest or retrace
        self.refused_class: dict[tuple, str] = {}
        self.lendable = list(lendable)
        self.lend_alias_groups = lend_alias_groups
        self.transient: dict[tuple, int] = {}  # transient refusals per key
        # per constant address or zeroed scratch block of a harvested
        # binding: (kernel, parameter, byte, value, channel)
        self.addresses: list[tuple[str, int, int, int, str]] = []
        self.disabled: str | None = None
        self._lock = threading.Lock()
        # load's bindings, their functions by name until resolved; the
        # resolved functions by (device, name); the stored keys whose nodes
        # disagreed with their plain capture (then harvested)
        self.stored: dict[tuple, OpaqueBinding] = {}
        self.functions: dict[tuple[int, str], int] = {}
        self.restored = 0
        self.stale: list[str] = []
        # the stored keys by each (device, function name) they wait for, how many each still waits for, and per device
        # those whose functions are all loaded (_await): a restore visits only the keys its new functions complete; the
        # keys it visited (counter)
        self._awaiting: dict[tuple[int, str], list[tuple]] = {}
        self._missing: dict[tuple, int] = {}
        self._complete: dict[int, dict[tuple, None]] = {}
        self.index_visits = 0
        # each distinct slot, slot tuple, launch shape and parameter layout
        # of the bindings, held once (_compact)
        self._interned: dict[Any, Any] = {}

    def accepts(self, op: Any, args: tuple, kwargs: dict) -> str | None:
        if self.disabled is not None:
            return self.disabled
        if op not in self.ops and not self._draws(op):
            return ""
        leaves = pytree.tree_leaves((args, kwargs))
        tensors = [t for t in leaves if isinstance(t, torch.Tensor)]
        if any(t.device.type != "cuda" for t in tensors):
            return "an operand is not on a CUDA device"
        if self._draws(op):
            if _argument(op, args, kwargs, "generator") is not None:
                return f"{op} with a generator other than the default"
            return None
        if op in self.extern:
            return None
        if op in _OUT_DTYPE:
            ins, out_dtype = {t.dtype for t in tensors[:2]}, _argument(op, args, kwargs, "out_dtype")
            if len(ins) != 1 or not ins <= {torch.bfloat16, torch.float16} or out_dtype != torch.float32:
                return f"mm with out_dtype {out_dtype} of {sorted(map(str, ins))} operands: only bf16 or fp16 into float32"
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
        # the rng family's ops and, with it on, an aten op that takes a Generator; a capture of any
        # other op that takes philox offsets is refused (_harvest)
        tags = getattr(op, "tags", ())
        static = not {torch.Tag.dynamic_output_shape, torch.Tag.data_dependent_output} & set(tags)
        return op in self.rng or (bool(self.rng) and isinstance(op, OpOverload) and op.namespace == "aten" and static and takes_generator(op))

    def _normal(self, key: OpaqueKey) -> tuple:
        return (
            key.alias if key.op in self.extern else (),
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
        k = self._normal(key)
        binding = self.bindings.get(k)
        if binding is None and k in self.stored:
            binding = self._resolved(k, key.device)
        return binding

    def save(self, path: str, versions: dict[str, str] | None = None) -> int:
        """Writes the bindings another process can load: each without an
        owner (a cuDNN plan), RNG state or a constant address in its images,
        its functions by name. The count written."""
        constants = {a[3] for a in self.addresses}
        entries, device = [], None
        for k, b in self.bindings.items():
            kernels = [n for n in b.nodes if isinstance(n, OpaqueKernel)]
            if b.owner is not None or b.rng_increment or any(t.kind.startswith("philox") for n in kernels for t in n.slots):
                continue
            if constants and any(w in constants for n in kernels for image in n.images for w in _qwords(image, 0)):
                continue
            names = iter(torch._C._cuda_hostTraceFunctionNames([n.function for n in kernels]))
            nodes = tuple(dataclasses.replace(n, function=next(names)) if isinstance(n, OpaqueKernel) else n for n in b.nodes)
            entries.append((_portable(k), dataclasses.replace(b, nodes=nodes)))
            device = k[7]
        if device is None:
            return 0
        with open(path + ".tmp", "wb") as f:
            pickle.dump({"meta": _meta(device, versions), "entries": entries}, f)
        os.replace(path + ".tmp", path)
        return len(entries)

    def load(self, path: str, versions: dict[str, str] | None = None) -> int:
        """Reads save's bindings if they were saved under this process's GPU,
        driver, torch, CUDA libraries and `versions`. A stored key binds once
        its kernels' functions are resolved: each name from a plain capture
        (no harvest) at the first stored key with it that learns, whose nodes
        must equal the stored ones but for their slots. The count read."""
        if not os.path.exists(path):
            return 0
        with open(path, "rb") as f:
            data = pickle.load(f)
        entries = data["entries"]
        if not entries or not _same_meta(data["meta"], _meta(entries[0][0][7], versions)):
            return 0
        for k, b in entries:
            local = _local(k)
            self.stored[local] = self._compact(b)
            self._await(local, self.stored[local])
        return len(entries)

    def preload(self) -> int:
        """Loads, before any call, the kernel functions load's bindings wait for: per function, one stored key that waits for
        it is restored from a plain capture of its operator on synthetic operands at the key's metadata (the warm-up's
        _learn, its call from the operator's schema), then every stored key whose functions are all loaded is held. A call
        at a stored key then fills its row in C++, with no restore at the call. A key whose capture differs from it is put
        back for its own call to judge. The count restored."""
        import types

        from torch.cuda._host_trace_replay import HostTraceReplay

        before = self.restored
        restore_only = _RestoreOnly(self)
        while self._awaiting:
            want, keys = next(iter(self._awaiting.items()))
            k = next((k for k in keys if k in self.stored and (call := _synthetic_call(k)) is not None), None)
            if k is None:
                del self._awaiting[want]
                continue
            key, spec = call
            stored, stale = self.stored[k], len(self.stale)
            binding = HostTraceReplay._learn(types.SimpleNamespace(learned=0, learn_s=0.0), key.op, restore_only, spec, key)
            if binding is None:
                self.stored[k] = stored
                del self.stale[stale:]
            if want in self._awaiting:  # this key did not load the function: another key of it, if any
                self._awaiting[want] = [o for o in self._awaiting[want] if o != k]
                if not self._awaiting[want]:
                    del self._awaiting[want]
        return self.restored - before

    def _await(self, k: tuple, binding: OpaqueBinding) -> None:
        device = k[7]
        missing = {n.function for n in binding.nodes if isinstance(n, OpaqueKernel)} - {name for d, name in self.functions if d == device}
        if missing:
            self._missing[k] = len(missing)
            for name in missing:
                self._awaiting.setdefault((device, name), []).append(k)
        else:
            self._complete.setdefault(device, {})[k] = None

    def _compact(self, binding: OpaqueBinding) -> OpaqueBinding:
        """binding with its parts that other bindings repeat (slots, launch
        shapes and parameter layouts) shared with theirs: keys that differ in
        a size share all but their parameter images, so a binding holds a few
        objects of its own, for the memory and the collector's passes."""
        intern = self._interned.setdefault

        def kernel(n: OpaqueKernel) -> OpaqueKernel:
            return dataclasses.replace(
                n,
                grid=intern(n.grid, n.grid),
                block=intern(n.block, n.block),
                attributes=intern(n.attributes, n.attributes),
                params=intern(n.params, n.params),
                slots=intern(slots := tuple(intern(t, t) for t in n.slots), slots),
                loose=intern(n.loose, n.loose),
            )

        nodes = tuple(kernel(n) if isinstance(n, OpaqueKernel) else dataclasses.replace(n, dst=intern(n.dst, n.dst)) for n in binding.nodes)
        return dataclasses.replace(binding, nodes=nodes, scratch=intern(binding.scratch, binding.scratch))

    def _resolved(self, k: tuple, device: int) -> OpaqueBinding | None:
        stored = self.stored[k]
        names = [n.function for n in stored.nodes if isinstance(n, OpaqueKernel)]
        if not all((device, name) in self.functions for name in names):
            return None
        nodes = tuple(dataclasses.replace(n, function=self.functions[device, n.function]) if isinstance(n, OpaqueKernel) else n for n in stored.nodes)
        binding = dataclasses.replace(stored, nodes=nodes)
        self._hold(k, binding)
        del self.stored[k]
        self.restored += 1
        return binding

    def _restore(self, key: OpaqueKey, k: tuple, args: tuple, kwargs: dict) -> OpaqueBinding | None:
        """The stored binding of k, its functions those of a plain capture of
        the call; None if the capture's nodes differ from it but for the
        slots and the harvest's loose bytes."""
        from torch.cuda._host_trace_capture import capture_kernel_nodes, KernelNode

        stored = self.stored[k]
        harvesting = torch._C._cuda_hostTraceSetHarvesting(True) if _ZERO_INIT else None
        try:
            with library_state_as(key.state), torch._C._AutoDispatchBelowADInplaceOrView():
                nodes = capture_kernel_nodes(lambda s: key.op(*args, **kwargs))
        finally:
            if harvesting is not None:
                torch._C._cuda_hostTraceSetHarvesting(harvesting)
        why = None
        if len(nodes) != len(stored.nodes):
            why = f"{len(nodes)} nodes, {len(stored.nodes)} stored"
        for n, b in zip(nodes, stored.nodes) if why is None else ():
            if isinstance(b, OpaqueMemset):
                if not isinstance(n, MemsetNode):
                    why = "a memset stored, a kernel captured"
                    break
                continue
            if not isinstance(n, KernelNode) or (n.name, tuple(n.grid), tuple(n.block), n.smem, tuple(map(tuple, n.layout))) != (b.function, b.grid, b.block, b.smem, b.params):
                why = f"kernel {getattr(n, 'name', n)} differs from the stored {b.function}"
                break
            masked = [bytearray(image) for image in n.images]
            for t in b.slots:
                masked[t.param][t.offset : t.offset + 8] = b.images[t.param][t.offset : t.offset + 8]
            for p, o, size in b.loose:
                masked[p][o : o + size] = b.images[p][o : o + size]
            if tuple(map(bytes, masked)) != b.images:
                why = f"kernel {n.name}'s parameters differ from the stored ones"
                break
        if why is not None:
            self.stale.append(f"{key.op} at sizes {key.sizes}: {why}")
            del self.stored[k]
            return None
        for n in nodes:
            if isinstance(n, KernelNode):
                if (key.device, n.name) not in self.functions:
                    for waiting in self._awaiting.pop((key.device, n.name), ()):
                        self.index_visits += 1
                        self._missing[waiting] -= 1
                        if not self._missing[waiting]:
                            del self._missing[waiting]
                            self._complete.setdefault(key.device, {})[waiting] = None
                self.functions[key.device, n.name] = n.function
        binding = self._resolved(k, key.device)
        # the stored keys whose functions are now all loaded: held, so a keyed site's row at one fills in C++
        ready = [o for o in self._complete.pop(key.device, {}) if o in self.stored]
        if torch.cuda._host_trace.verify_redo_templates:
            scan = [o for o, b in self.stored.items() if o[7] == key.device and all((key.device, n.function) in self.functions for n in b.nodes if isinstance(n, OpaqueKernel))]
            if set(ready) != set(scan):
                raise AssertionError(f"host_trace: the restore index gives {len(ready)} stored keys complete, a scan {len(scan)}")
        for other in ready:
            self._resolved(other, key.device)
        return binding

    def _layout_class(self, key: OpaqueKey) -> tuple:
        """The key without its sizes and integer scalars: per operand its rank,
        the order of its strides and which are 1 or 0, and its alignment
        class; a structural refusal holds for every key of the class."""

        def layout(sizes: tuple, strides: tuple) -> tuple:
            order = tuple(sorted(range(len(strides)), key=lambda d: (-strides[d], d)))
            return order, tuple(st == 1 for st in strides), tuple(st == 0 for st in strides)

        scalars = tuple(None if type(v) is int else v for v in key.scalars)
        alias = tuple((i, lead) for i, lead, _ in key.alias) if key.op in self.extern else ()
        layouts = tuple(layout(z, st) for z, st in zip(key.sizes, key.strides))
        return (alias, _OUT.get(key.op, key.op), key.dtypes, layouts, tuple(map(_class, key.align)), scalars, key.device, key.state)

    def _refuse(self, key: OpaqueKey, why: str, structural: bool = False) -> None:
        # final for the process: a trace records the key as a plain eager step
        self.refused[self._normal(key)] = why
        if structural:
            self.refused_class[self._layout_class(key)] = why
        msg = f"{key.op} at sizes {key.sizes}: {why}"
        log.info("host_trace: harvest refused %s", msg)
        trace_structured(
            "artifact",
            metadata_fn=lambda: {"name": "host_trace_harvest_refused", "encoding": "string"},
            payload_fn=lambda: msg,
        )

    def refusal(self, key: OpaqueKey) -> str | None:
        k = self._normal(key)
        if k in self.bindings or k in self.stored:
            return None
        return self.refused.get(k) or (self.refused_class.get(self._layout_class(key)) if self.refused_class else None)

    def refuses_layout(self, key: OpaqueKey) -> bool:
        k = self._normal(key)
        return k not in self.bindings and k not in self.stored and self._layout_class(key) in self.refused_class

    def _span(self, key: OpaqueKey, i: int) -> int:
        return span_bytes(key.sizes[i], key.strides[i], key.dtypes[i].itemsize)

    def operands(self, key: OpaqueKey, inputs: int) -> dict[int, torch.Tensor] | None:
        if key.op not in self.extern:
            return {}
        # with lend_alias_groups, per alias group's lead: (input, distance) of each
        alias = key.alias if self.lend_alias_groups else ()
        groups: dict[int, list[tuple[int, int]]] = {}
        for j, lead, d in alias:
            groups.setdefault(lead, [(lead, 0)]).append((j, d))
        members = {j for j, _, _ in alias}
        lent: dict[int, torch.Tensor] = {}
        for i in range(inputs):
            if i in members:
                continue
            group = groups.get(i, [(i, 0)])
            dtype, align = key.dtypes[i], key.align[i]
            reach = max(d + self._span(key, j) for j, d in group)
            if reach <= _LENT:
                continue
            if not self.lend_alias_groups and any(i in (j, lead) for j, lead, _ in key.alias):
                return None
            if any(j >= inputs or key.dtypes[j] != dtype or d % dtype.itemsize for j, d in group):
                return None
            fits = [
                t
                for t in self.lendable
                if t.dtype == dtype and t.device.index == key.device and t.data_ptr() % 256 == 0
                and t.untyped_storage().nbytes() - t.storage_offset() * t.itemsize >= align + reach
            ]
            # a lendable tensor of its own per group, where there are enough
            taken = {u.untyped_storage().data_ptr() for u in lent.values()}
            fits.sort(key=lambda t: t.untyped_storage().data_ptr() in taken)
            if not fits:
                return None
            at = fits[0].storage_offset() + align // dtype.itemsize
            for j, d in group:
                lent[j] = fits[0].as_strided(key.sizes[j], key.strides[j], at + d // dtype.itemsize)
        return lent

    def learn(
        self, key: OpaqueKey, args: tuple, kwargs: dict, operands: list[Any]
    ) -> OpaqueBinding | None:
        k = self._normal(key)
        with self._lock, torch.cuda.device(key.device):
            if k in self.bindings:
                return self.bindings[k]
            if k in self.stored and (binding := self._restore(key, k, args, kwargs)) is not None:
                return binding
            if self.disabled is not None or self.refusal(key) is not None:
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
                    sibling = key.op in _OUT or key.op in _ATTENTION or key.op in _CONV or key.op in self.extern
                    siblings = self.templates.get(group) if sibling and _SIBLINGS else None
                    pair = _PAIR
                    # a refusal of the two captures is the four's to make
                    if siblings:
                        try:
                            with _released(key.device):
                                binding = self._harvest(key, args, kwargs, operands, siblings, pair)
                        except _NoSibling:
                            pass
                        except _Refused as e:
                            if not pair or isinstance(e, _Transient):
                                raise
                            pair = False
                    if binding is None and pair:
                        try:
                            with _released(key.device):
                                binding = self._harvest(key, args, kwargs, operands, pair=True)
                        except _Refused as e:
                            if isinstance(e, _Transient):
                                raise
                    if binding is None:
                        with _released(key.device):
                            binding = self._harvest(key, args, kwargs, operands)
                    binding = self._compact(binding)
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
                self._refuse(key, str(e), e.structural)
                return None
            self._hold(k, binding)
            return binding

    def _hold(self, k: tuple, binding: OpaqueBinding) -> None:
        # a binding at its normal key, and in the native tables (a keyed site's row filled in C++ at a new key)
        self.bindings[k] = binding
        self.native.add(k, binding)

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
            name, overload = _OUT[op], "dtype_out" if op in _OUT_DTYPE else "out"
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
        spans = [span_bytes(t.shape, t.stride(), t.element_size()) for t in operands]
        # an aliased operand lies in its lead's windows at its distance, the
        # lead's windows covering the group's union
        alias = {i: (lead, d) for i, lead, d in key.alias} if op in self.extern else {}
        reach = spans[:placed]
        for i, (lead, d) in alias.items():
            t, u = operands[i], operands[lead]
            if t.untyped_storage().data_ptr() != u.untyped_storage().data_ptr() or t.data_ptr() - u.data_ptr() != d:
                raise _Transient(f"operand {i} is not {d} bytes past operand {lead} on its storage")
            reach[lead] = max(reach[lead], d + spans[i])
            reach[i] = 0
        # an extern op verifies on copies of the alias groups it writes, the
        # rest its live operands
        written = [_argument(op, args, kwargs, a.name) for a in op._schema.arguments if a.alias_info and a.alias_info.is_write]
        copied = {alias.get(i, (i, 0))[0] for i, t in enumerate(operands[:placed]) if any(t is w for w in written)}
        # a declared KV writer verifies on caches of at least 2n + 1 slots at
        # their strides (rows of dim 0 times the slots of a row: the cache dims
        # the index spans after dim 0), its n indices a copy holding distinct
        # slots of them. Caches on one storage (K and V interleaved in one
        # page) stay in one block at their distance
        shapes = [t.shape for t in operands[:placed]]
        # the bytes verify holds of each operand, and of each block
        extent, block_bytes = spans[:placed], list(reach)
        indices, caches, index_range = None, (), 0
        if op in self.indexed:
            *names, dims = self.indexed[op] if isinstance(self.indexed[op][-1], int) else (*self.indexed[op], 1)
            positions = {id(t): i for i, t in enumerate(operands[:placed])}
            indices, *caches = (positions[id(_argument(op, args, kwargs, name))] for name in names)
            group = {j: alias.get(j, (j, 0))[0] for j in range(placed)}
            if any(group[j] == group[indices] for j in range(placed) if j != indices):
                raise _Refused(f"the index operand {indices} shares a storage with another operand")
            if any(group[j] in {group[c] for c in caches} for j in range(placed) if j not in caches):
                raise _Refused("an operand other than a cache shares a cache's storage")
            per_row = {math.prod(operands[j].shape[1:dims]) for j in caches}
            if len(per_row) != 1:
                raise _Refused(f"the caches' rows hold different slot counts {sorted(per_row)}")
            per_row = per_row.pop()
            rows = -(-(2 * operands[indices].numel() + 1) // per_row)
            index_range = rows * per_row
            for j in caches:
                t = operands[j]
                shapes[j] = (rows, *t.shape[1:])
                block_bytes[j] = extent[j] = (1 + sum((n - 1) * st for n, st in zip(shapes[j], t.stride()))) * t.element_size()
            for j in caches:
                if j in alias:
                    lead, d = alias[j]
                    block_bytes[lead] = max(block_bytes[lead], d + extent[j])
                    block_bytes[j] = 0
            copied.add(indices)
        cap = _EXTERN_CAP if self.extern_cap is None else self.extern_cap
        if op in self.extern and sum(block_bytes[i] for i in copied) > cap:
            raise _Refused(f"the written operands span {sum(block_bytes[i] for i in copied)} bytes, past the extern cap {cap}")
        windows = [(r + 2 * _WINDOW - 1) // _WINDOW if r else 0 for r in reach]
        total = sum(windows) | 1
        arena = dev.arena((2 * total + 1) * _WINDOW)
        starts: list[list[int]] = [[], []]
        cursor = 0
        for c, n in zip(classes, windows):
            second = total * _WINDOW + (c ^ ((_WINDOW - 1) & ~(2 * c - 1)))
            starts[0].append(arena + cursor + c)
            starts[1].append(arena + cursor + second)
            cursor += n * _WINDOW
        for i, (lead, d) in alias.items():
            starts[0][i], starts[1][i] = starts[0][lead] + d, starts[1][lead] + d
        sets = [torch._C._cuda_hostTraceArenaViews(at, operands[:placed]) for at in starts]
        # a GEMM verifies on the call's own tensors, unless its output overlaps an input
        in_place = op in _OUT
        if in_place:
            lo = operands[-1].data_ptr()
            hi = lo + spans[-1]
            in_place = all(t.data_ptr() >= hi or t.data_ptr() + n <= lo for t, n in zip(operands[:-1], spans))

        graphs = []  # every capture's graph, held until the harvest is over
        grown = False  # whether a capture allocated in the pool
        # the harvest's own memory, which no binding may hold a constant into:
        # (start, bytes, what), the most telling first
        ours: list[tuple[int, int, str]] = [(dev.va, dev.va_size, "the operand arena")]
        plugs: list[torch.Tensor] = []
        # what the captures left live, held until the harvest is over: their
        # fresh outputs, and an RNG call's per-capture seed and offset
        kept: list[torch.Tensor] = []
        returns = [r.alias_info is None for r in op._schema.returns]
        # each parameter's byte classes (LaunchLayout.h) of the kernels the
        # captures' ATen pointwise launch sites report, by CUfunction
        reported: dict[int, list[str]] = {}

        def capture(x: int, stream: torch.cuda.Stream, tensors: list, prime: bool = False) -> tuple:
            nonlocal grown
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            graphs.append(graph)
            if op in _OUT:
                call = _out_call(op, name, tensors)
                call_args, call_kwargs = call.args, call.keywords
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
                    holes.append(_unfilled(o.untyped_storage().nbytes(), device))
                plugs.extend(dev.plug([b for b in dev.blocks() if not b[4]]))
            del holes
            # the call allocates its cuBLAS workspace in the pool (its
            # scratch); TORCH_CUBLAS_WORKSPACE_CACHE=1: the stream's cached one
            # is released before the capture, and after it, so none is left
            # live in the pool
            torch._C._cuda_hostTraceRecordLaunches(True)
            try:
                result, allocated, _, philox, held = torch._C._cuda_hostTraceHarvestCapture(
                    graph,
                    dev.pool.id,
                    (stream.stream_id, stream.device_index, stream.device_type),
                    dev.anchor,
                    0,
                    0,
                    _PRETAKE[x] if rng else 0,
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
            finally:
                for function, _, _, layout, _ in torch._C._cuda_hostTraceRecordLaunches(False):
                    reported[function] = layout
            if philox is not None:
                seed, offset, taken = philox
                kept.extend((seed, offset))
                philox = (seed.data_ptr(), offset.data_ptr(), taken)
                del seed, offset
            grown |= bool(allocated)
            ours[:0] = [(a, n, f"capture {x}'s allocation") for a, n in allocated]
            ours.extend((b[1], b[2], "the capture pool") for b in dev.blocks())
            pool = dev.blocks() if allocated else []
            if prime:
                return ()
            rets = [] if op in _OUT else [result] if len(returns) == 1 else list(result or ())
            fresh = _fresh(rets, returns)
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
                raise _Refused(str(e), structural=True) from None
            names = [n[0] if len(n) == 9 else "memset" for n in found]
            nodes = [
                OpaqueKernel(n[1], n[3], n[4], n[5], _attributes(n[7], n[6]), n[2], n[8], ())
                if len(n) == 9
                else MemsetNode(*n)
                for n in found
            ]
            kernels = [n for n in nodes if isinstance(n, OpaqueKernel)]
            if any(not reproducible(k) for n in kernels for k, _ in n.attributes):
                raise _Refused("a device-updatable kernel node", structural=True)
            plugged = {t.data_ptr() for t in plugs} | {t.untyped_storage().data_ptr() for t in kept}
            if any(b[4] and b[1] not in plugged for b in pool):
                raise _Refused("the call left an allocation live")
            # the free blocks the call allocated in: freed allocations
            # coalesce with their free neighbors, so each is one scratch
            # buffer, as far as the call's allocations in it reach
            ends = {b: [a + n for a, n in allocated if b[1] <= a < b[1] + b[2]] for b in pool if not b[4]}
            mine = [b for b, e in ends.items() if e]
            if allocated:
                plugs.extend(dev.plug(mine))
            # an output's own allocation, or a temporary freed before the
            # output took its bytes (the output's address, as bound)
            reused = sum(any(p <= a and a + n <= p + m for p, m in owned) for a, n in allocated)
            if sum(map(len, ends.values())) + reused != len(allocated):
                raise _Refused(f"the call's allocations {allocated} are not its blocks and outputs")
            extents = [max(ends[b]) - b[1] for b in mine]
            # the blocks in the order the call first allocated in them (the
            # allocator places them anywhere)
            order = [next(i for i, (a, _) in enumerate(allocated) if b[1] <= a < b[1] + b[2]) for b in mine]
            ordered = [(a, n) for _, a, n in sorted(zip(order, (b[1] for b in mine), extents))]
            return nodes, names, ordered, outputs, philox, _PRETAKE[x], held

        def verify(binding: OpaqueBinding, stale: list, names: list, retired: Collection[int] = ()) -> OpaqueBinding:
            # the binding checked (and with stale bytes: with another
            # capture's there), each launch with its scratch filled 0x00, then
            # 0xFF, and each constant CUDA address (a `retired` qword, mapped
            # or not, among them) pointing into a mirror filled alike. It returns the binding, but for a failed check that
            # a channel explains: scratch read before it is written (a
            # semaphore or counter), zeroed by a memset ahead of the binding;
            # a live constant address, into a scratch block of the harvest's
            # own (zeroed if it must be)
            nodes = binding.nodes
            # a constant into the harvest's own memory (an earlier capture's
            # block, the operand arena) goes away with the pool: verify mirrors
            # and poisons it like a retired qword, so a live one (a workspace
            # the kernel writes or a value it reads) fails the check and refuses
            # the key; a stale leftover of a library's struct passes
            ours_hits = set()
            for x, n in enumerate(nodes):
                if not isinstance(n, OpaqueKernel):
                    continue
                near = {(t.param, t.offset + d) for t in n.slots for d in (-4, 0, 4)}
                for param, image in enumerate(n.images):
                    for first in (0, 4):
                        for i, q in enumerate(_qwords(image, first)):
                            if (param, first + 8 * i) not in near and any(a <= q < a + size for a, size, _ in ours):
                                ours_hits.add(q)
            retired = {*retired, *ours_hits}
            variants = [binding]
            if stale:
                images = [list(n.images) if isinstance(n, OpaqueKernel) else None for n in nodes]
                for x, param, off, other in stale:
                    image = bytearray(images[x][param])
                    image[off : off + 8] = other
                    images[x][param] = bytes(image)
                alt = [n if i is None else dataclasses.replace(n, images=tuple(i)) for n, i in zip(nodes, images)]
                variants.append(dataclasses.replace(binding, nodes=tuple(alt)))
            known = dict.fromkeys(retired, True)
            hits = [_address_hits(b.nodes, stack, known)[0] for b in variants]
            misaligned = _address_hits(nodes, stack, known)[1]
            addresses = {h[3] for hs in hits for h in hs}
            try:
                mirrors = _Mirrors(device, addresses) if addresses else None
            except RuntimeError as e:
                raise _Transient(f"no mirror of the constant addresses {sorted(map(hex, addresses))}: {e}") from None
            try:
                try:
                    return checked(binding, variants, hits, misaligned, mirrors, names)
                except _NotBitwise:
                    return checked(binding, variants, hits, misaligned, mirrors, names, live=False)
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
            live: bool = True,
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

            took: list[float] = []

            def launch(b: OpaqueBinding, addrs: list[int], fill: int, philox: tuple[int, int, int] = (0, 0, 0)) -> str | None:
                # the binding launched on scratch and mirrors filled with
                # `fill`; why it is wrong, if it hangs or writes a mirror
                scratch = [torch.empty(n, dtype=torch.uint8, device=device) for n in b.scratch]
                for t in scratch:
                    t.fill_(fill)
                if mirrors is not None:
                    mirrors.fill(fill)
                torch.cuda.current_stream().synchronize()
                start = time.monotonic()
                _launch(b, addrs, [t.data_ptr() for t in scratch], stream, philox)
                done = torch.cuda.Event()
                done.record()
                deadline = start + (min(_HANG, max(_HANG_MIN, _HANG_FACTOR * min(took))) if took else _HANG)
                while not done.query():
                    if time.monotonic() > deadline:
                        break
                    time.sleep(1e-4)
                else:
                    took.append(time.monotonic() - start)
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
            # floating inputs of one value (an all-zeros mask): any binding
            # passes on them, so they take random stand-ins
            def one_value(t: torch.Tensor) -> bool:
                bits = t.view(_BITS[t.element_size()])
                return bool(bits.amin() == bits.amax())

            constant = set() if op in self.extern else {i for i, t in enumerate(operands[:placed]) if t.is_floating_point() and t.numel() > 1 and one_value(t)}
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
                eager = _out_call(op, name, operands)

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

                def untouched() -> str | None:
                    return None

            elif live and not binding.rng and op not in self.extern and not any(a.alias_info and a.alias_info.is_write for a in op._schema.arguments):
                # the binding on the call's own tensors (live at a learn at the
                # call; a missed bind's; _learn's stand-ins and its probe's
                # outputs), nothing copied: the outputs' and inputs' digests
                # taken, each launch into the outputs (filled 0xFF first) and
                # everything digested again. A mismatch reruns eager into the
                # outputs, which also tells whether eager reproduces them
                # bitwise; if not, verify runs on copies (_NotBitwise). A
                # constant input (all zeros, say) would pass a wrong binding:
                # it takes a random stand-in at its layout and alignment, eager
                # on the stand-ins gives the outputs' digests, and the call's
                # outputs are put back from eager after the check. An _UNORDERED output (atomics: eager
                # does not reproduce it bitwise) is held to 4x eager's own
                # spread, from one eager run, against a copy of the call's
                # (the one copy, which also restores it); an op that reads its
                # forward's philox seed and offset takes no generator state
                # the outputs the binding writes (another, a CPU seed say, is the op's own)
                referenced = {t.index for n in nodes for t in _slots(n) if t.kind == "operand"}
                ins = list(operands[:placed])
                filler = torch.Generator(operands[0].device).manual_seed(0)
                for i in sorted(constant):
                    t = ins[i]
                    span = (sum((n - 1) * st for n, st in zip(t.shape, t.stride())) + 1) * t.element_size()
                    block = torch.empty(-(-(span + 256) // 8), dtype=torch.int64, device=t.device).view(torch.uint8)
                    flat = block[key.align[i] : key.align[i] + span].view(t.dtype)
                    fill_uniform(flat, -1, 1, filler)
                    ins[i] = flat.as_strided(t.shape, t.stride())
                positions = [j for j in range(len(operands) - placed) if placed + j in referenced]
                live_outs = [operands[placed + j] for j in positions]
                loose = {k for k, j in enumerate(positions) if j in _UNORDERED.get(op, ()) and live_outs[k].is_floating_point()}
                ptrs = [t.data_ptr() for t in ins] + [t.data_ptr() for t in operands[placed:]]
                held = [_digest(t) for t in ins]

                def rerun(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
                    it = iter(inputs)
                    call_args, call_kwargs = pytree.tree_unflatten([next(it) if isinstance(a, torch.Tensor) else a for a in leaves], spec)
                    result = op(*call_args, **call_kwargs)
                    fresh = _fresh([result] if len(returns) == 1 else list(result or ()), returns)
                    return [fresh[j] for j in positions]

                noise: dict[int, float] = {}
                if constant or loose:
                    torch.cuda.current_stream().synchronize()
                    start = time.monotonic()
                    reference = rerun(ins) if constant else live_outs
                    torch.cuda.current_stream().synchronize()
                    took.append(time.monotonic() - start)
                want = [_digest(o) for o in (reference if constant else live_outs)]
                saved = {k: (reference if constant else live_outs)[k].clone() for k in loose}
                if constant:
                    del reference
                if constant or loose:
                    # eager once more on the same inputs: the ordered outputs again bitwise, the unordered ones' spread
                    fresh = rerun(ins)
                    if not all(torch.equal(_digest(f), w) for k, (f, w) in enumerate(zip(fresh, want)) if k not in loose):
                        raise _NotBitwise
                    noise = {k: _gap_max(saved[k], fresh[k]) for k in loose}
                    del fresh

                def eager() -> bool:
                    # the outputs back as the call left them (from eager on its own inputs); whether eager gave the ordered ones bitwise again
                    fresh = rerun(list(operands[:placed]))
                    again = constant or all(torch.equal(_digest(f), w) for k, (f, w) in enumerate(zip(fresh, want)) if k not in loose)
                    for k, (o, f) in enumerate(zip(live_outs, fresh)):
                        o.copy_(saved[k] if k in loose and not constant else f)
                    return bool(again)

                def run(b: OpaqueBinding, fill: int) -> str | None:
                    for o in live_outs:
                        o.view(_BITS[o.element_size()]).fill_(-1)
                    try:
                        why = launch(b, ptrs, fill)
                    except _Disabled:
                        eager()
                        raise
                    wrong = [i for i, (t, d) in enumerate(zip(ins, held)) if not torch.equal(_digest(t), d)]
                    if wrong:
                        eager()
                        raise RuntimeError(f"host_trace: the harvested launch of {names} wrote input {wrong[0]} of {op}")
                    bad = next((k for k, (o, w) in enumerate(zip(live_outs, want)) if not (_close(saved[k], o, noise[k]) if k in loose else torch.equal(_digest(o), w))), None)
                    if why is None and bad is None:
                        return None
                    if not eager() and why is None:
                        raise _NotBitwise
                    return why or f"the binding's launch differs from eager at operand {placed + positions[bad]} ({names})"

                def untouched() -> str | None:
                    if constant:
                        eager()
                    return None

            else:
                # the binding on fresh random operands at the key's layouts and
                # alignments, against the eager call on them (a zero or NaN
                # operand would pass any binding; unbounded ones overflow a
                # long reduction, and an _UNORDERED output's rounding grows
                # with its terms, not its result); a second copy of each placed
                # operand when eager writes one, refilled before each launch,
                # the binding's fresh outputs filled with 0xFF. An RNG call
                # draws from the generator's offset for both, which is left
                # where it was. A drawing op's floating operands are
                # probabilities, means and scales: in (0, 1)
                mutates = op in _OUT or any(a.alias_info and a.alias_info.is_write for a in op._schema.arguments)
                # an extern op runs on its live read-only operands (its index
                # operands hold valid indices only there), checked unchanged,
                # and copies of the live bytes of the alias groups it writes,
                # a group's in one block, the written floating operands then
                # random and the rest zeros: their live values are the call's results
                extern = op in self.extern
                # each operand's block (its lead operand's) and byte in it; an
                # op that writes none of its operands runs on them live but for
                # the alias groups of its constant inputs
                randomized = {alias.get(i, (i, 0))[0] for i in constant}
                lies: list[tuple[int, int] | None] = []
                for i in range(placed):
                    lead, d = alias.get(i, (i, 0))
                    live_in = (extern and lead not in copied) or (not extern and not mutates and lead not in randomized)
                    lies.append(None if live_in else (lead, key.align[lead] + d))
                leads = sorted({lie[0] for lie in lies if lie})
                lives = {
                    i: torch.empty(0, dtype=torch.uint8, device=t.device).set_(t.untyped_storage(), t.storage_offset() * t.element_size(), (block_bytes[i],), (1,))
                    for i, t in enumerate(operands[:placed])
                    if extern and i in leads and i not in caches and block_bytes[i]
                }

                def blocks() -> dict[int, torch.Tensor]:
                    return {i: torch.zeros(-(-(block_bytes[i] + 256) // 8), dtype=torch.int64, device=operands[i].device).view(torch.uint8) for i in leads}

                def refill(blocks: dict[int, torch.Tensor]) -> None:
                    # the blocks as each launch starts them, the same bytes each time
                    filler = torch.Generator(operands[0].device).manual_seed(0)
                    for i, base in blocks.items():
                        if i in caches:
                            base.random_(0, 256, generator=filler)
                        elif i in lives:
                            base[key.align[i] : key.align[i] + block_bytes[i]].copy_(lives[i])
                    for i, t in enumerate(operands[:placed]):
                        if lies[i] is None or i in caches:
                            continue
                        lead, align = lies[i]
                        flat = blocks[lead][align : align + extent[i]].view(t.dtype)
                        if i == indices:
                            flat.as_strided(t.shape, t.stride()).copy_(torch.randperm(index_range, generator=filler, device=t.device)[: t.numel()].view(t.shape))
                        elif extern:
                            if t.dtype.is_floating_point and any(t is w for w in written):
                                fill_uniform(flat.as_strided(t.shape, t.stride()), -1, 1, filler)
                            elif any(t is w for w in written):
                                flat.as_strided(t.shape, t.stride()).zero_()
                        elif t.dtype.is_floating_point and self._draws(op):
                            fill_uniform(flat, 0.25, 0.75, filler)
                        elif t.dtype.is_floating_point:
                            fill_uniform(flat, -1, 1, filler)
                        else:
                            flat.random_(0, 2 if t.dtype == torch.bool else 8, generator=filler)

                def placed_at(blocks: dict[int, torch.Tensor]) -> list[torch.Tensor]:
                    return [
                        t if lie is None else blocks[lie[0]][lie[1] : lie[1] + span].view(t.dtype).as_strided(shape, t.stride())
                        for t, shape, span, lie in zip(operands[:placed], shapes, extent, lies)
                    ]

                refs = blocks()
                refill(refs)
                ref_in = placed_at(refs)
                # without a second copy, neither eager nor the binding may change an
                # input's digest
                rows_only = self.extern_digest == "rows"

                def digests(read: list[int]) -> list[torch.Tensor]:
                    return torch._C._cuda_hostTraceDigests([_rows(ref_in[i], rows_only) for i in read], [not rows_only] * len(read))[1] if read else []

                unchanged = [i for i, lie in enumerate(lies) if not mutates or lie is None]
                inputs = dict(zip(unchanged, digests(unchanged)))

                start = gen.get_offset()
                spread: list[float | None] = []
                if op in _OUT:
                    _out_call(op, name, ref_in)()
                    ref_out = []
                else:
                    it = iter(ref_in)
                    call_args, call_kwargs = pytree.tree_unflatten(
                        [next(it) if isinstance(a, torch.Tensor) else a for a in leaves], spec
                    )

                    def fresh_outputs() -> list[torch.Tensor]:
                        result = op(*call_args, **call_kwargs)
                        rets = [result] if len(returns) == 1 else list(result or ())
                        return seed_offset_on_device(op, _fresh(rets, returns), operands[0].device)

                    torch.cuda.current_stream().synchronize()
                    started = time.monotonic()
                    ref_out = fresh_outputs()
                    torch.cuda.current_stream().synchronize()
                    took.append(time.monotonic() - started)
                    # a floating output eager itself doesn't reproduce bitwise (an
                    # _UNORDERED one, or one this rerun differs in) is held to 4x
                    # eager's own spread plus the dtype's testing tolerance: a wrong
                    # address or size gives garbage, not rounding noise
                    if not mutates:
                        gen.set_offset(start)
                        unordered = _UNORDERED.get(op, ())
                        spread = [
                            _gap_max(a, b)
                            if a.is_floating_point() and (i in unordered or not _same(a, b))
                            else None
                            for i, (a, b) in enumerate(zip(ref_out, fresh_outputs()))
                        ]
                # at the fresh outputs' layouts (meta tensors past learn, HarvestProvider.learn)
                outs = [
                    torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=operands[0].device)
                    for t in operands[placed:]
                ]
                state = (0, 0, 0)
                if increment:
                    seed = gen.initial_seed()
                    seed = torch.tensor([seed - (seed >> 63 << 64)], device=operands[0].device)
                    offset = torch.tensor([start], device=operands[0].device)
                    state = (seed.data_ptr(), offset.data_ptr(), 0)
                # an extern op whose eager call on the random copies gave its
                # live results (it writes those operands, it does not read
                # them) checks the binding against the live bytes, on the one
                # copy, refilled
                fast = _EXTERN_FAST and extern and mutates and not rng and indices is None and all(i in lives and _equal(refs[i][key.align[i] : key.align[i] + block_bytes[i]], lives[i]) for i in leads)
                runs = refs if fast or not mutates else blocks()
                addrs = [t.data_ptr() for t in placed_at(runs)] + [t.data_ptr() for t in outs]
                gen.set_offset(start)
                referenced = {t.index for n in nodes for t in _slots(n) if t.kind == "operand"}
                tolerated = tuple(i for i in sorted(referenced) if i >= placed and spread and spread[i - placed] is not None)

                def run(b: OpaqueBinding, fill: int) -> str | None:
                    if mutates:
                        refill(runs)
                    for t in outs:
                        t.untyped_storage().fill_(255)
                    why = launch(b, addrs, fill, state)
                    once = inputs.keys() if _DIGEST_ONCE else set()
                    for i in sorted(referenced - once) if why is None else ():
                        lead = lies[i][0] if i < placed and lies[i] else None
                        if i in inputs:
                            ok = torch.equal(digests([i])[0], inputs[i])
                        elif i < placed and fast:
                            # the block's padding still the zeros it was allocated with
                            got, at, r = runs[lead], key.align[lead], block_bytes[lead]
                            ok = _equal(got[at : at + r], lives[lead]) and not got[:at].any() and not got[at + r :].any()
                        elif i < placed:
                            ok = torch.equal(refs[lead].view(torch.int64), runs[lead].view(torch.int64))
                        elif i in tolerated:
                            ok = _close(ref_out[i - placed], outs[i - placed], spread[i - placed])
                        else:
                            ok = _same(ref_out[i - placed], outs[i - placed])
                        if not ok:
                            return f"the binding's launch differs from eager at operand {i} ({names})"
                    return why

                def untouched() -> str | None:
                    # the inputs digested once, after every launch: an input
                    # can be a live KV cache layer, gigabytes to digest
                    read = sorted(referenced & inputs.keys()) if _DIGEST_ONCE else []
                    after = digests(read)
                    wrong = [i for i, d in zip(read, after) if not torch.equal(d, inputs[i])]
                    return f"the binding's launch differs from eager at operand {wrong[0]} ({names})" if wrong else None

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
            if failed is None and (why := untouched()) is not None:
                raise _Refused(why)
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
                nodes.append(dataclasses.replace(n, images=tuple(images), slots=slots, loose=t.loose))
            binding = _renumbered(nodes, [n for _, n in blocks], 0, cap[6], key.op in _OUT)
            if key.op in _CUDNN:
                highs = _own_highs()
                stale_at = _stale_constant(binding, lambda q: q >> 32 in highs and not _cuda_address(q))
                if stale_at is not None:
                    raise _NoSibling("parameter {1} byte {2}: {3:#x}, the harvest's own high half".format(*stale_at))
            shared = _shared(binding)
            # its blocks shared as the template's are (unshared if the template's shared form refused)
            return shared if len(shared.scratch) == len(template.scratch) else binding

        if dev.pool is None:
            with torch.cuda.device(device):
                dev.pool = torch.cuda.MemPool(allocator=_address_only().allocator)
                # a graph kept on the pool: the pinned host allocator asserts
                # when a pool id is captured into after all its graphs are gone
                dev.keeper = torch.cuda.CUDAGraph()
                with torch.cuda.stream(s):
                    dev.keeper.capture_begin(pool=dev.pool.id, capture_error_mode="thread_local")
                    dev.anchor.fill_(1)
                    dev.keeper.capture_end()
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
            # with its graphs reset (a refusal's traceback may hold one) and
            # its tensors gone, the pool's blocks are free: back to the device
            # by the pool's destructor
            for graph in graphs:
                graph.reset()
            graphs.clear()
            plugs.clear()
            kept.clear()
            dev.keeper = None
            dev.pool = None
        blocks = [c[2] for c in caps]
        stack = torch._C._cuda_hostTraceStackBounds()
        if siblings:
            binding = adopt(caps[0], siblings, starts[0])
            del sets
            if _VERIFY_SIBLINGS:
                try:
                    binding = verify(binding, [], caps[0][1])
                except _Refused as e:
                    raise _NoSibling(str(e)) from None
            return binding

        heap = _heap_bounds()
        addresses = [[t.data_ptr() for t in sets[w]] + c[3] for w, c in zip((0, 1, 0, 0), caps)]
        philox = [c[4] for c in caps]
        if not rng and any(p[2] for p in philox):
            raise _Refused(f"{op} takes {[p[2] for p in philox]} philox offsets and takes no Generator")
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
                # into an alias group's union: the operand it is fewest bytes into
                if len({alias.get(i, (i, 0))[0] for i, _ in roles}) == 1:
                    return ("operand", *min(roles, key=lambda r: r[1]))
            # differs between captures of one operand set on one stream, or
            # with the stream alone, or a stack address in any: host state of
            # the call
            if any(q != qa for q in qs[2:]) or any(stack[0] <= q < stack[1] for q in qs):
                return "host"
            return False if qa != qs[1] else None

        nodes: list = []
        stale: list[tuple[int, int, int, bytes]] = []  # (node, parameter, byte, another capture's bytes)
        retired_qwords: set[int] = set()
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
            loose = []
            for param in range(len(a.params)):
                imgs = tuple(n.images[param] for n in ns)
                image = bytearray(imgs[0])
                covered = bytearray(len(image))
                # bytes another call may hold otherwise: a stack address's,
                # and those that differ between captures with no slot
                free = bytearray(len(image))
                # bytes of no role by the parameter's type (padding, an empty
                # functor) hold whatever the call's stack left there
                layout = reported.get(a.function)
                if layout is not None and len(layout[param]) == len(image):
                    for i in range(len(image)):
                        if layout[param][i] == ".":
                            covered[i] = free[i] = 1
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
                            role = _classified(classify(qs), qs) if varies else "host"
                        elif varies or any(covered[off : off + 8]):
                            continue
                        else:
                            role = None
                        if role is None:
                            # a constant CUDA address is verify's to explain
                            continue
                        # cuBLASLt's CUTLASS 3 struct keeps an earlier call's workspace
                        # address, mapped or in a released pool. The address shape alone
                        # suffices only because cuBLAS is closed source and verify mirrors
                        # and poisons every retired qword at every learn: a live one
                        # mismatches there and refuses the key, never faults a replay
                        retired = role is False and op in _OUT and all(q >> 32 and q % 256 == 0 for q in qs)
                        if retired:
                            retired_qwords.update(qs)
                        if role is False and op not in _CONV and not retired:
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
                                if op not in _CONV and op not in _REDUCE and not heaped and not retired:
                                    raise _Refused(f"{where}: unexplained varying parameter", structural=True)
                                if len({q & 0xFFFFFFFF for q in qs}) > 1 and any(map(_cuda_address, qs)) and not retired:
                                    seen = " / ".join(img[off : off + 8].hex() for img in imgs)
                                    raise _Refused(f"{where}: host state {seen}", structural=True)
                                other = next(img for img in imgs if img[off : off + 8] != imgs[0][off : off + 8])
                                stale.append((len(nodes), param, off, other[off : off + 8]))
                            if any(stack[0] <= int.from_bytes(img[off : off + 8], "little") < stack[1] for img in imgs):
                                free[off : off + 8] = b"\x01" * 8
                            else:
                                free[off : off + 8] = bytes(len({img[i] for img in imgs}) > 1 for i in range(off, off + 8))
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
                        raise _Refused(f"{where}: per-call state of no role: {seen}", structural=True)
                    if not covered[off]:
                        free[off] = 1
                at = free.find(1)
                while at >= 0:
                    end = free.find(0, at)
                    end = len(free) if end < 0 else end
                    loose.append((param, at, end - at))
                    at = free.find(1, end)
                images.append(bytes(image))
            nodes.append(dataclasses.replace(a, images=tuple(images), slots=tuple(slots), loose=tuple(loose)))

        binding = _renumbered(nodes, sizes[0], increment, tuple(c[6] for c in caps if c[6] is not None) or None, key.op in _OUT)
        kinds = {t.kind for n in binding.nodes for t in _slots(n)}
        if increment and not {"philox_seed", "philox_offset", "philox"} <= kinds:
            raise _Refused(f"{names} take {increment} philox offsets and read no captured generator state")
        del sets

        # cuDNN builds its parameter structs on the heap before the captures: a
        # constant there whose high half is one of the harvest's own ranges'
        # (under an int32, or a stale address into them) is the harvest's
        # leftover, which eager's struct would not hold, and refuses
        # (_stale_constant). A cuBLAS / CUTLASS binding's constants are its own
        highs = _own_highs()

        def own_high(q: int) -> bool:
            return q >> 32 in highs and q not in retired_qwords and not _cuda_address(q)

        stale_at = _stale_constant(binding, own_high) if op in _CUDNN else None
        if stale_at is not None:
            x, param, off, q = stale_at
            raise _Refused(f"{names[x]} parameter {param} byte {off}: {q:#x}, {_STALE_STRUCT}")
        # the forms verify judges, the first it passes binding: its scratch
        # shared (a block that must start zeroed, or a node reaching a block
        # past its slots, refuses that), then each block its own
        shared = _shared(binding)
        forms = [shared, binding] if shared is not binding else [binding]
        for form in forms[:-1]:
            try:
                return verify(form, stale, names, retired_qwords)
            except _Refused as e:
                log.info("host_trace: %s at sizes %s: a form of its binding refused, trying the next: %s", op, key.sizes, e)
        return verify(forms[-1], stale, names, retired_qwords)
