"""The planned arena of a lowered host-trace tape (direct_hosttrace).

The tape records every allocation with a symbolic size and, through the launches,
regions, memsets and copies that touch it, a first and last use. The temporaries (the
allocations no output escapes through) are placed in one arena by first fit
decreasing at the preparation shapes; each block's offset is the expression
``offset(support) + rounded_size(support)`` along its support chain, so the plan is a
table of expressions the numeric plan evaluates per call, and the arena itself is one
boxed input the entry rebinds per call (E19: no address pins). The runtime's per-call
allocate loop and ``clear()`` then run over the escaping outputs only.

The plan's non-overlap holds where its class predicate holds: for every pair of
lifetime-overlapping blocks the condition ``offset(hi) >= offset(lo) + size(lo)`` is
proved for all shapes when ``lo`` sits on ``hi``'s support chain (structural), or when
the symbolic gap is provably non-negative over the tape's positive size symbols (the
rounding bounded by ``x <= 512*ceil(x/512) < x + 512``), or after unifying symbols that
share a hint value (each unification becomes an equality guard); what remains is a
guard term. A call outside the class is a topology-style miss: the same tape is built
again at the call's inputs with its own plan (no re-trace). A call whose arena
requirement exceeds the family's arena grows it (a rebind, not a rebuild).

The escaping outputs have their own arena (``plan_outputs``, ``OutputRing``): the
outputs of one call are placed together in one block (all live at once: a chain, no
class guard, the requirement the sum of the rounded sizes) and the runtime binds each
output as a typed view over the boxed block, so every view the caller keeps is a
reference on the block's storage. The family keeps a ring of such blocks; a block
whose storage only the ring references is free and is reused, a call with every block
held takes a new one (an allocation, never a wait), so the caller never sees a value
change under an output it holds.
"""

import math
from collections import Counter
from dataclasses import dataclass, field

import sympy

import torch
from torch.utils._sympy.functions import CeilDiv, FloorDiv


ALIGN = 512  # the plan's granularity: every offset is a multiple of it (cuBLAS / TMA need 256)
_SMALL_ROUND = 64 * 1024
_LARGE = 1 << 20
_LARGE_ROUND = 2 << 20


def round_capacity(nbytes):
    """The arena's capacity for a requirement: 64 KiB steps under 1 MiB, 2 MiB steps
    above (the caching allocator's large-block granularity), never zero."""
    step = _SMALL_ROUND if nbytes <= _LARGE else _LARGE_ROUND
    return max(step, -(-nbytes // step) * step)


@dataclass
class Block:
    name: str
    size: sympy.Expr  # bytes of the storage extent, as the tape's expressions
    rounded: sympy.Expr  # ALIGN * FloorDiv(size + ALIGN - 1, ALIGN)
    hint: int  # the rounded size at the plan's hints
    first: int  # seq of the first event that touches it
    last: int  # seq of the last
    key: str  # the size class: blocks with equal size expressions share it
    support: str | None = None
    hint_offset: int = 0
    offset: sympy.Expr = sympy.Integer(0)
    chain: Counter = field(
        default_factory=Counter
    )  # size class -> count along the support chain
    ancestors: frozenset = frozenset()  # the names along the support chain


def _overlaps(a, b):
    return a.first <= b.last and b.first <= a.last


def _strip_div1(e):
    return e.replace(
        lambda x: isinstance(x, FloorDiv) and x.args[1] == 1, lambda x: x.args[0]
    )


def _bound_divisions(e):
    """Every c*CeilDiv(x, k) / c*FloorDiv(x, k) term replaced by the bound in the
    direction that lowers the expression (k*CeilDiv(x, k) lies in [x, x + k - 1],
    k*FloorDiv(x, k) in [x - k + 1, x]): a lower bound of the whole."""
    e = sympy.expand(_strip_div1(e))
    out = sympy.Integer(0)
    for term, c in e.as_coefficients_dict().items():
        divisions = term.atoms(CeilDiv, FloorDiv)
        for f in divisions:
            x, k = f.args
            if isinstance(f, CeilDiv):
                term = term.xreplace({f: x / k if c > 0 else (x + k - 1) / k})
            else:
                term = term.xreplace({f: (x - k + 1) / k if c > 0 else x / k})
        out += c * term
    return sympy.expand(out)


def _symbol_floor(s):
    if s.is_positive:
        return 1
    if s.is_nonnegative:
        return 0
    return None


def nonnegative(e):
    """Whether `e >= 0` for every value of its symbols above their floors (positive
    integers for sizes, non-negative for strides), by coefficient signs after bounding
    the roundings: the minimum of a sum with non-negative coefficients is at the
    floors. False when it cannot be shown (not a statement that it fails)."""
    e = _bound_divisions(e)
    if e.is_Integer:
        return int(e) >= 0
    coefficients = e.as_coefficients_dict()
    if any(c < 0 for term, c in coefficients.items() if term != 1):
        return False
    floors = {}
    for s in e.free_symbols:
        floor = _symbol_floor(s)
        if floor is None:
            return False
        floors[s] = sympy.Integer(floor)
    return bool(e.xreplace(floors) >= 0)


@dataclass
class ArenaPlan:
    """The plan of one variant: its blocks with offset expressions, the arena
    requirement expression, the class guards, and the boxed position of the arena."""

    input_index: int
    blocks: dict
    arena: sympy.Expr
    class_guards: tuple
    capacity: sympy.Symbol  # stands for the arena input's size(0) in the predicate
    hints: dict
    equalities: int = 0
    proofs: dict = field(default_factory=dict)
    lower_bound: int = 0  # the largest live set at the hints: no plan is smaller there
    order: str = "size"  # the greedy order the plan came from

    def covers(self, name):
        return name in self.blocks

    def offset(self, name):
        return self.blocks[name].offset

    def required_bytes(self, values):
        v = self.arena.xreplace(values)
        if v.free_symbols:
            raise ValueError(
                f"arena requirement {self.arena} is not decided by {values}"
            )
        return int(v)

    def check(self, values, capacity):
        """The debug assertion: at the values of one call, no two lifetime-overlapping
        blocks intersect and every block lies within `capacity` bytes."""
        placed = []
        for b in self.blocks.values():
            offset = int(b.offset.xreplace(values))
            size = int(b.rounded.xreplace(values))
            if offset % ALIGN or offset < 0 or size < 0:
                raise AssertionError(
                    f"arena block {b.name} at {offset} of {size} bytes"
                )
            if offset + size > capacity:
                raise AssertionError(
                    f"arena block {b.name} [{offset}, {offset + size}) exceeds the arena's {capacity} bytes"
                )
            placed.append((b, offset, size))
        for i, (a, oa, sa) in enumerate(placed):
            for b, ob, sb in placed[i + 1 :]:
                if _overlaps(a, b) and oa < ob + sb and ob < oa + sa:
                    raise AssertionError(
                        f"arena blocks {a.name} [{oa}, {oa + sa}) and {b.name} [{ob}, {ob + sb}) overlap while both live"
                    )


_ORDERS = {
    "size": lambda b: (-b.hint, b.first, b.name),
    "first": lambda b: (b.first, -b.hint, b.name),
    "last": lambda b: (b.last, -b.hint, b.name),
    "span": lambda b: (-(b.last - b.first), -b.hint, b.name),
}


def _first_fit(blocks, order):
    """First fit with lifetimes in the given order: each block at the lowest offset
    where no lifetime-overlapping placed block intersects it. Returns the arena."""
    placed = []
    for b in sorted(blocks, key=_ORDERS[order]):
        live = sorted(
            (p for p in placed if _overlaps(p, b)), key=lambda p: p.hint_offset
        )
        cursor, support = 0, None
        for p in live:
            if p.hint_offset >= cursor + b.hint:
                break
            top = p.hint_offset + p.hint
            if top > cursor:
                cursor, support = top, p.name
        b.hint_offset = cursor
        b.support = support
        placed.append(b)
    return max((b.hint_offset + b.hint for b in blocks), default=0)


def _best_first_fit(blocks):
    """The smallest of the first-fit plans over the orders at the hints (size
    decreasing met the live-set bound on the planner's inference tapes; a
    forward-then-backward tape can do better in first-use order)."""
    bound = _live_peak(blocks)
    best = None
    for order in _ORDERS:
        arena = _first_fit(blocks, order)
        if best is None or arena < best[0]:
            best = (arena, order, [(b.hint_offset, b.support) for b in blocks])
        if arena == bound:
            break
    arena, order, placement = best
    for b, (offset, support) in zip(blocks, placement):
        b.hint_offset, b.support = offset, support
    return order


def plan_arena(allocations, uses, hints, input_index, capacity_symbol):
    """`allocations`: (name, size expression in bytes) rows of the temporaries to
    place; `uses`: name -> the seqs of the events touching it; `hints`: symbol ->
    value at the preparation inputs. An empty plan (nothing placed) needs 0 bytes."""
    blocks = {}
    for name, size in allocations:
        size = sympy.expand(size)
        if size.is_Integer and int(size) == 0:
            continue
        if size.is_Integer:
            rounded = sympy.Integer(ALIGN * math.ceil(int(size) / ALIGN))
        else:
            # not CeilDiv(size, ALIGN): its construction takes a polynomial gcd of the
            # size, exponential in the size's symbols (a cat of 100 inputs: 100
            # symbols); FloorDiv(size + ALIGN - 1, ALIGN) is the same value at 0.2 ms
            rounded = ALIGN * FloorDiv(size + ALIGN - 1, ALIGN)
        hint = rounded.xreplace(hints)
        if hint.free_symbols or int(hint) < 0:
            continue  # not decided by the inputs, or a negative extent: the runtime's buffer
        seqs = uses[name]
        blocks[name] = Block(
            name, size, rounded, int(hint), min(seqs), max(seqs), str(rounded)
        )
    if not blocks:
        return ArenaPlan(
            input_index, {}, sympy.Integer(0), (), capacity_symbol, dict(hints)
        )
    order = _best_first_fit(list(blocks.values()))
    rounded_of_key = {b.key: b.rounded for b in blocks.values()}

    def resolve(b):
        if b.ancestors or b.support is None:
            return
        s = blocks[b.support]
        resolve(s)
        b.chain = Counter(s.chain)
        b.chain[s.key] += 1
        b.ancestors = s.ancestors | {s.name}
        b.offset = sympy.expand(s.offset + s.rounded)

    for b in blocks.values():
        resolve(b)
    # equal-hint symbols, for the unified proofs
    by_value = {}
    for s, v in hints.items():
        by_value.setdefault(int(v), []).append(s)
    representative = {}
    for group in by_value.values():
        group = sorted(group, key=sympy.default_sort_key)
        for s in group:
            representative[s] = group[0]
    proofs = {}
    guards, equalities = [], set()
    seen_guards = set()

    def prove(gap_counter):
        # gap = sum of count * rounded(size class); memoized by the multiset
        key = tuple(sorted(gap_counter.items()))
        if key in proofs:
            return proofs[key]
        gap = sum((n * rounded_of_key[k] for k, n in key), sympy.Integer(0))
        if nonnegative(gap):
            result = ("symbolic", None, ())
        else:
            unified = gap.xreplace(
                {s: representative[s] for s in gap.free_symbols if s in representative}
            )
            if nonnegative(unified):
                pairs = tuple(
                    sympy.Eq(s, representative[s], evaluate=False)
                    for s in sorted(gap.free_symbols, key=sympy.default_sort_key)
                    if s in representative and representative[s] != s
                )
                result = ("unified", None, pairs)
            else:
                result = ("guard", sympy.Ge(sympy.expand(_strip_div1(gap)), 0), ())
        proofs[key] = result
        return result

    def add_guard(relation):
        text = str(relation)
        if text not in seen_guards:
            seen_guards.add(text)
            guards.append(relation)

    ordered = sorted(blocks.values(), key=lambda b: (b.hint_offset, b.name))
    counts = Counter()
    for i, lo in enumerate(ordered):
        for hi in ordered[i + 1 :]:
            if not _overlaps(lo, hi):
                continue
            if lo.name in hi.ancestors:
                counts["structural"] += 1
                continue
            gap = Counter(hi.chain)
            gap.subtract(lo.chain)
            gap[lo.key] -= 1
            kind, relation, pairs = prove(Counter({k: n for k, n in gap.items() if n}))
            counts[kind] += 1
            if relation is not None:
                add_guard(relation)
            for pair in pairs:
                if str(pair) not in equalities:
                    equalities.add(str(pair))
                    guards.append(pair)
    # the arena the call needs: the top at the hints, and every top not provably below it
    tops = {b.name: sympy.expand(b.offset + b.rounded) for b in blocks.values()}
    top = max(blocks.values(), key=lambda b: (b.hint_offset + b.hint, b.name))
    terms = [tops[top.name]]
    for b in blocks.values():
        if b is top:
            continue
        if not nonnegative(tops[top.name] - tops[b.name]):
            terms.append(tops[b.name])
    arena = terms[0] if len(terms) == 1 else sympy.Max(*terms)
    plan = ArenaPlan(
        input_index, blocks, arena, tuple(guards), capacity_symbol, dict(hints)
    )
    plan.equalities = len(equalities)
    plan.proofs = dict(counts)
    plan.lower_bound = _live_peak(blocks.values())
    plan.order = order
    return plan


def _live_peak(blocks):
    delta = Counter()
    for b in blocks:
        delta[b.first] += b.hint
        delta[b.last + 1] -= b.hint
    peak = live = 0
    for seq in sorted(delta):
        live += delta[seq]
        peak = max(peak, live)
    return peak


def plan_outputs(allocations, hints, input_index, capacity_symbol):
    """The escaping outputs of one call as the blocks of one output block: every one
    live at once, so first fit chains them (each block's support is the one before
    it), every pair condition is structural and the requirement is the sum of the
    rounded sizes. Rows whose size the inputs do not decide stay the runtime's."""
    uses = {name: [0] for name, _ in allocations}
    return plan_arena(allocations, uses, hints, input_index, capacity_symbol)


_use_count = torch._C._storage_Use_Count
_empty_cuda = torch._C._dynamo.guards._empty_strided_cuda


class OutputRing:
    """One family's output blocks: uint8 tensors from the caching allocator, 512-byte
    aligned with a storage offset of zero (the plan's offsets are over the block's
    data pointer). The runtime binds every escaping output as a typed view over the
    boxed block, so a block's storage use count is its count at creation (the ring's
    tensor and the storage's own Python object) plus the views the caller still holds:
    a block at its creation count is free. The ring keeps at most `k` blocks, most
    recently taken first: `take()` returns the first free one (the block taken last
    when the caller dropped the previous outputs: the same addresses, and the replay
    patches nothing), and when every kept block is held it allocates a block for this
    call alone, which the caller's views own and the allocator takes back when they
    go (an allocation, never a wait; a hold deeper than `k` costs one allocation per
    call, not one per output). `ensure()` raises the capacity floor on the miss path;
    a free block under the floor is dropped when a call needs one."""

    def __init__(self, device, k=2):
        self.device = device
        self.k = k
        self.blocks = []  # [tensor, storage impl address, capacity, base count], MRU first
        self.minimum = 0  # the capacity floor: what the largest call so far needed
        self.overflow = 0  # blocks allocated for one call because every kept block was held
        self.replaced = 0  # free blocks dropped for capacity

    @property
    def capacity(self):
        return self.blocks[0][2] if self.blocks else 0

    def held(self):
        return sum(_use_count(cdata) > base for _, cdata, _, base in self.blocks)

    def take(self):
        blocks = self.blocks
        minimum = self.minimum
        for i, block in enumerate(blocks):
            if _use_count(block[1]) <= block[3] and block[2] >= minimum:
                if i:
                    del blocks[i]
                    blocks.insert(0, block)
                return block[0]
        capacity = round_capacity(minimum)
        # a deep hold pays this per call: the bare allocator entry on the current
        # device (the entry's bound device, O29; 0.9 us hot), torch.empty(device=)
        # only when the call comes from elsewhere
        tensor = _empty_cuda((capacity,), (1,), torch.uint8)
        if tensor.device != self.device:
            tensor = torch.empty(capacity, dtype=torch.uint8, device=self.device)
        if tensor.data_ptr() % ALIGN:
            raise RuntimeError(
                f"host_trace replay: the caching allocator returned an output block at {tensor.data_ptr():#x}, not {ALIGN}-byte aligned"
            )
        if len(blocks) >= self.k:
            # free blocks under the floor never serve again: drop them for the new one
            kept = [b for b in blocks if b[2] >= minimum or _use_count(b[1]) > b[3]]
            self.replaced += len(blocks) - len(kept)
            blocks[:] = kept
        if len(blocks) < self.k:
            cdata = tensor.untyped_storage()._cdata
            blocks.insert(0, [tensor, cdata, capacity, _use_count(cdata)])
        else:
            self.overflow += 1
        return tensor

    def ensure(self, nbytes):
        """Raise the capacity floor to `nbytes` (a floor under 16 MiB at least
        doubles): the next `take()` returns a block of at least that. True when the
        floor rose."""
        if nbytes <= self.minimum:
            return False
        if self.minimum < 16 << 20:
            nbytes = max(nbytes, 2 * self.minimum)
        self.minimum = round_capacity(nbytes)
        return True

    def clear(self):
        self.blocks.clear()
