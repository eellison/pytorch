"""Local misses served by partitioning (plan item 38 stage 2 in its partition form, E41).

A call that fails only kernel-only guards of one op (or a few ops) of a variant's tape
is served without a re-trace and without re-lowering the whole tape: the tape's records
are split at those ops into segments, each segment lowered and prepared as its own
program (the same lowering and native replay as a variant, over the same symbols), the
cut ops run through eager at the call's real tensors (eager's own kernel choice, E36 by
construction), and the next segment takes the tensors the earlier segments and the
eager ops produced as inputs.

The boundary. A record belongs to the segment or cut op whose seq range holds it (the
op table's ``[seq0, seq1)``, LOCAL_MISS 3.1). An allocation belongs to where it was
made; one that a later segment or a later cut op reads is an escaping output of its
segment (a runtime buffer of that program, allocated per call) and, under E31, a
prebound root of every later segment that reads it: the sequence pass boxes it after
the tape's tensors and rebases every pointer over it to root + displacement, and its
program never allocates or frees it (``SequencePlan.prebound``). An allocation a cut
op made is eager's tensor, a prebound root of the segments that read it. The kept
guards split by their raw index against the cut op's guard range: the guards raised
before it are the earlier segment's, after it the later's; inside its range the
kernel-tagged ones are dropped (eager decides the kernel) and the metadata-tagged ones
go to the segment before the op (the op's validity and metadata checks, before eager
runs it). The swap check compares each eager output's metadata (sizes, strides, storage
offset, dtype) and root identity (a fresh storage, or the storage of the root the tape
recorded) with the tape's expressions at the call before the next segment reads it: a
mismatch means the guard that missed was metadata-affecting after all, and the call
misses as before (a trace) for a functional op, or raises for an op that wrote or
aliased an input (a re-run would repeat the write).

What is not served here (declined by name, the ordinary miss path continues): tapes
with random launches (the philox slots are per graph), host tables or pinned inputs,
cut ops whose outputs are not the only roots they made that a later record reads, and
ops whose arguments read an opaque rebind.
"""

import dataclasses
import itertools
import time
from types import SimpleNamespace
from typing import Any

import sympy

import torch
from torch.cuda import _host_trace
from torch.cuda._host_trace import _OutputRec, _SYM_TYPES, _TracedTensor
from torch.utils._pytree import tree_flatten, tree_map

from .hosttrace_allocseq import AllocatorSequence


class SwapMismatch(Exception):
    """An eager output of a cut op differs from the tape's expressions at the call."""


class PartitionDeclined(Exception):
    """The tape or the cut op is outside what the partition serves."""


_INF = 1 << 62


def _expr(v):
    if isinstance(v, _SYM_TYPES):
        return v.node._expr
    return v


def _free_symbols(v):
    e = _expr(v)
    return e.free_symbols if isinstance(e, sympy.Basic) else set()


def _raw_nbytes(rec):
    # an allocation's storage bytes over the raw expressions (computeStorageNbytes)
    extent = sympy.Integer(1)
    for size, stride in zip(rec.sizes, rec.strides):
        extent = extent + (sympy.sympify(_expr(size)) - 1) * sympy.sympify(
            _expr(stride)
        )
    return sympy.expand(rec.root.itemsize * extent)


class _PartTape:
    """A segment of a tape, in the shape ``lower_tape`` reads: the same symbols,
    inputs and argument contract as the tape, the records of the segment's seq range,
    the allocations the segment makes or reads, the guards assigned to it, and its
    outputs (the tape's outputs over its allocations, then its escaping allocations
    whole)."""

    def __init__(
        self, tape, launches, allocs, memsets, memcpys, regions, guards, outputs
    ):
        self.shape_env = tape.shape_env
        self.device = tape.device
        self.device_identity = tape.device_identity
        self.args = None
        self.warm_up_outputs = None
        self.nargs = tape.nargs
        self.positions = tape.positions
        self.constants = tape.constants
        self.inputs = tape.inputs
        self.allocs = allocs
        self.launches = launches
        self.opaque = tape.opaque
        self.rng_increment = None
        self.rng_slots = []
        self.all_on_capture_stream = tape.all_on_capture_stream
        self.written_roots = list(tape.written_roots)
        self.written_inputs = tape.written_inputs
        self.memsets = memsets
        self.host_buffers = []
        self.memcpys = memcpys
        self.regions = regions
        self.outputs = outputs
        self.guards = guards
        self.guard_notes = {}
        self.root_facts = list(tape.root_facts)
        self._tape = tape  # the symbols' owners stay alive with it

    @property
    def num_launches(self):
        return len(self.launches)

    def empty(self):
        return not (self.launches or self.memsets or self.memcpys or self.regions)


@dataclasses.dataclass
class _Segment:
    index: int
    lo: int  # records with lo <= seq < hi
    hi: int
    part: Any = None  # _PartTape, None for a segment with no records
    lowered: Any = None
    entry: Any = None
    dispatch: Any = None
    boxer: Any = None
    sequence: Any = None
    prebound: list = dataclasses.field(default_factory=list)  # (name, root index)
    escapes: list = dataclasses.field(default_factory=list)  # allocation names
    t_outputs: list = dataclasses.field(default_factory=list)  # tape output indices
    python_allocs: list = dataclasses.field(default_factory=list)  # _AllocRec rows
    lower_s: float = 0.0
    prepare_s: float = 0.0
    calls: int = 0

    def close(self):
        if self.dispatch is not None:
            self.dispatch.close()
        if self.sequence is not None:
            self.sequence.release()


@dataclasses.dataclass
class _Cut:
    op: Any  # _OpRec
    produced: set  # allocation names made inside the op's range
    args: Any = None  # the op's arguments with traced tensors replaced by _Ref rows
    kwargs: Any = None
    out_spec: Any = None
    out_leaves: list = dataclasses.field(default_factory=list)  # traced output leaves
    eager_s: float = 0.0


@dataclasses.dataclass(frozen=True)
class _Ref:
    """A traced tensor of the op table as the partition materializes it: its root and
    its metadata as expressions evaluated at the call."""

    root: str
    sizes: tuple
    strides: tuple
    offset: Any
    dtype: torch.dtype


def local_miss_ops(tape, lowered, box):
    """The top-level ops whose kernel-only guards are the only failing kept guards of
    `tape` at `box` (their indices, in tape order) and the failing guards' texts, or
    (None, why) when the miss is not local: no kept guard fails (an obligation, an
    opaque selector, a device term), or a failing guard is metadata-tagged or raised
    by the Python between ops."""
    from . import direct_hosttrace as dh

    if not getattr(tape, "ops", None) or not getattr(tape, "kept_raw", None):
        return None, "the tape has no op table"
    ok = lambda why: (None, why)  # noqa: E731
    values = dh._symbol_values(lowered, box)
    for symbol, rec in lowered.symbols.opaque.items():
        if rec["kind"] == "guard":
            continue
        try:
            args = [int(dh._expr(a).xreplace(values)) for a in rec["args"]]
            values[symbol] = sympy.Integer(rec["call"](args))
        except (ArithmeticError, TypeError, ValueError):
            return ok(f"opaque rebind {rec['fn']} cannot be evaluated here")
    for g in lowered.extra_guards:
        try:
            if dh._expr(g).xreplace(values) is sympy.false:
                return ok(f"obligation failed: {g}")
        except (ArithmeticError, TypeError, ValueError):
            return ok(f"obligation cannot be evaluated: {g}")
    for rec in lowered.symbols.opaque.values():
        if rec["kind"] != "guard":
            continue
        try:
            got = rec["call"]([int(dh._expr(a).xreplace(values)) for a in rec["args"]])
        except (ArithmeticError, TypeError, ValueError):
            return ok(f"opaque {rec['fn']} cannot be evaluated here")
        if got != int(rec["expected"]):
            return ok(
                f"opaque {rec['fn']} gives {got}, the tape traced {rec['expected']}"
            )
    ops = set()
    failing = []
    for k, g in enumerate(tape.guards):
        try:
            failed = dh._expr(g).xreplace(values) is sympy.false
        except (ArithmeticError, TypeError, ValueError):
            failed = True
        if not failed:
            continue
        a = tape.guard_attribution(k)
        failing.append(
            f"{g} (op {a['op']} {a['func']}: {a['phase']}, {a['origin']}{' + ' + str(len(a['also'])) + ' more raisers' if a['also'] else ''})"
        )
        # every raiser of the relation (the record keeps one row per relation; the
        # other ops that evaluated it are its `also` rows) must be a kernel choice
        # inside a top-level op; the cut is at every one of those ops
        rows = [(a["op"], a["depth"]), *((op, depth) for op, depth, _ in a["also"])]
        for op_index, depth in rows:
            if op_index is None or op_index < 0 or depth == 0:
                return ok(f"not local: {failing[-1]}")
            op = tape.ops[op_index]
            while op.depth > 0:
                op = tape.ops[op.parent]
            if op.declined:
                return ok(f"not local (a declined op): {failing[-1]}")
            ops.add(op.index)
    if not failing:
        return ok(
            "no kept guard fails by substitution (a pointer, device or opaque term)"
        )
    return sorted(ops), "; ".join(failing)


class Partition:
    """The variant's tape split at `op_indices` (top-level ops of its op table): the
    segments before, between and after them as programs of their own, the ops run
    through eager. Built lazily on its first `serve`: each segment is lowered and
    prepared when the call reaches it, at the tensors the earlier segments produced."""

    def __init__(self, owner, variant, op_indices):
        from . import direct_hosttrace as dh

        self.owner = owner
        self.variant = variant
        self.tape = tape = variant.tape
        self.lowered_t = variant.lowered
        self.device = variant.lowered.device
        self.op_indices = tuple(op_indices)
        self.serves = 0
        self.built = False
        self.build_s = 0.0
        self.evaluator = _host_trace._Evaluator()
        if tape.rng_increment is not None:
            raise PartitionDeclined(
                "the tape draws random numbers (per-graph philox slots)"
            )
        if tape.host_buffers or any(m["kind"] == "h2d" for m in tape.memcpys):
            raise PartitionDeclined(
                "the tape carries host tables or pinned-input copies"
            )
        if any(i.pinned or i.device.type == "cpu" for i in tape.inputs):
            raise PartitionDeclined("the tape has a pinned CPU input")
        cuts = []
        for i in self.op_indices:
            op = tape.ops[i]
            if op.depth != 0 or op.seq[1] < 0:
                raise PartitionDeclined(f"op {i} is not a completed top-level op")
            cuts.append(op)
        cuts.sort(key=lambda op: op.seq[0])
        for a, b in itertools.pairwise(cuts):
            if a.seq[1] > b.seq[0]:
                raise PartitionDeclined("the cut ops' record ranges overlap")
        # the allocation names by the symbol of their base, for the use scan
        by_q = {}
        self.alloc_by_name = {}
        # an allocation's record name ("alloc3") by its root's name ("a3"): the
        # boundary, the refs and the plans are keyed by the record name, an input by
        # its root's name
        self.root_alloc = {}
        for rec in tape.allocs:
            by_q[_expr(rec.q)] = rec.name
            self.alloc_by_name[rec.name] = rec
            self.root_alloc[rec.root.name] = rec.name
        # every event's uses of allocations, by seq
        uses: dict = {name: [] for name in self.alloc_by_name}

        def use(value, seq):
            for s in _free_symbols(value):
                name = by_q.get(s)
                if name is not None:
                    uses[name].append(seq)

        for L in tape.launches:
            for p in L["params"]:
                if p["kind"] == "ptr" and isinstance(p["value"], _SYM_TYPES):
                    use(p["value"], int(L["seq"]))
        for m in tape.memsets:
            use(m["dst"], int(m["seq"]))
        for m in tape.memcpys:
            use(m["src"], int(m["seq"]))
            use(m["dst"], int(m["seq"]))
        for r in tape.regions:
            for operand in (*r.inputs, *r.outputs):
                use(operand.address, int(r.seq))
        for op in cuts:
            for t in tree_flatten((op.args, op.kwargs))[0]:
                if isinstance(t, _TracedTensor) and t._root.allocation:
                    uses[self.root_alloc[t._root.name]].append(int(op.seq[0]))
        for o in tape.outputs:
            if o.root.allocation:
                uses[self.root_alloc[o.root.name]].append(_INF)
        self.uses = uses
        # the segments' ranges and the cuts
        bounds = [-1] + [s for op in cuts for s in op.seq] + [_INF]
        self.segments = [
            _Segment(j, bounds[2 * j], bounds[2 * j + 1]) for j in range(len(cuts) + 1)
        ]
        self.cuts = [
            _Cut(
                op,
                {rec.name for rec in tape.allocs if op.seq[0] <= rec.seq < op.seq[1]},
            )
            for op in cuts
        ]
        producer = {}
        for rec in tape.allocs:
            producer[rec.name] = self._where(rec.seq)
        self.producer = producer
        # the tape's outputs: where each comes from
        self.out_plan = [None] * len(tape.outputs)
        for seg in self.segments:
            self._plan_segment(seg, dh)
        for j, cut in enumerate(self.cuts):
            self._plan_cut(j, cut)
        for i, o in enumerate(tape.outputs):
            if self.out_plan[i] is None:
                self.out_plan[i] = self._plan_output(i, o)
        # the symbols the per-call evaluation reads, and how (input facts)
        needed = set()
        for cut in self.cuts:
            for leaf in tree_flatten((cut.args, cut.kwargs))[0]:
                needed |= self._ref_symbols(leaf)
            for leaf in cut.out_leaves:
                needed |= self._ref_symbols(leaf)
        for plan in self.out_plan:
            if plan[0] in ("view", "view_cut", "python_alloc"):
                needed |= self._ref_symbols(plan[1])
        for seg in self.segments:
            for rec in seg.python_allocs:
                for v in (*rec.sizes, *rec.strides):
                    needed |= _free_symbols(v)
        for name in self.alloc_by_name:
            needed |= _raw_nbytes(self.alloc_by_name[name]).free_symbols
        self.reads = []
        opaque = self.lowered_t.symbols.opaque
        for s in sorted(needed, key=sympy.default_sort_key):
            prop = self.lowered_t.symbols.by_symbol.get(s)
            if prop is None:
                raise PartitionDeclined(f"symbol {s} has no input source")
            if prop.kind == "opaque":
                rec = opaque[s]
                if rec["kind"] != "guard":
                    raise PartitionDeclined(
                        f"{s} is an opaque rebind the cut op's arguments read"
                    )
                self.reads.append((str(s), "const", int(rec["expected"]), 0))
            elif prop.kind in ("size", "stride", "offset"):
                self.reads.append((str(s), prop.kind, prop.index, prop.dim))
            else:
                raise PartitionDeclined(
                    f"symbol {s} ({prop.kind}) is not a metadata fact"
                )

    def _where(self, seq):
        for j, cut in enumerate(self.cuts):
            if cut.op.seq[0] <= seq < cut.op.seq[1]:
                return ("cut", j)
        for seg in self.segments:
            if seg.lo <= seq < seg.hi:
                return ("seg", seg.index)
        raise AssertionError(f"seq {seq} is in no segment")

    def _ref_symbols(self, leaf):
        if isinstance(leaf, _Ref):
            out = set()
            for v in (*leaf.sizes, *leaf.strides, leaf.offset):
                out |= _free_symbols(v)
            return out
        return _free_symbols(leaf)

    def _name(self, root):
        return self.root_alloc.get(root.name, root.name)

    def _ref(self, t):
        return _Ref(
            self._name(t._root),
            tuple(_expr(v) for v in t.shape),
            tuple(_expr(v) for v in t._sym_strides),
            _expr(t._sym_offset),
            t.dtype,
        )

    def _plan_segment(self, seg, dh):
        tape = self.tape
        in_range = lambda seq: seg.lo <= seq < seg.hi  # noqa: E731
        launches = [L for L in tape.launches if in_range(int(L["seq"]))]
        memsets = [m for m in tape.memsets if in_range(int(m["seq"]))]
        memcpys = [m for m in tape.memcpys if in_range(int(m["seq"]))]
        regions = [r for r in tape.regions if in_range(int(r.seq))]
        produced = [rec for rec in tape.allocs if in_range(rec.seq)]
        used_here = {
            name for name, seqs in self.uses.items() if any(in_range(s) for s in seqs)
        }
        # allocations made here that nothing here touches but a later segment or op
        # reads are made in Python at the call (eager's at::empty, written later);
        # one nothing ever touches is dropped, as the whole tape's lowering drops it
        seg.python_allocs = [
            rec for rec in produced if rec.name not in used_here and self.uses[rec.name]
        ]
        own = [rec for rec in produced if rec.name in used_here]
        prebound = [
            self.alloc_by_name[name]
            for name in used_here
            if self.producer[name] != ("seg", seg.index)
        ]
        prebound.sort(key=lambda rec: rec.seq)
        seg.escapes = [
            rec.name for rec in own if any(s >= seg.hi for s in self.uses[rec.name])
        ]
        outputs = []
        own_names = {rec.name for rec in own}
        for i, o in enumerate(tape.outputs):
            if o.root.allocation and self._name(o.root) in own_names:
                outputs.append(
                    _OutputRec(
                        o.name, o.root, o.sizes, o.strides, o.offset, o.dtype, None
                    )
                )
                self.out_plan[i] = ("seg", seg.index, len(seg.t_outputs))
                seg.t_outputs.append(i)
        for name in seg.escapes:
            rec = self.alloc_by_name[name]
            outputs.append(
                _OutputRec(
                    f"{name}_boundary",
                    rec.root,
                    rec.sizes,
                    rec.strides,
                    0,
                    rec.dtype,
                    None,
                )
            )
        guards = self._segment_guards(seg)
        part = _PartTape(
            tape, launches, own + prebound, memsets, memcpys, regions, guards, outputs
        )
        if part.empty():
            if outputs or prebound:
                raise PartitionDeclined(
                    f"segment {seg.index} has no records but outputs or prebound roots"
                )
            return
        seg.part = part
        seg.prebound = [rec.name for rec in prebound]

    def _segment_guards(self, seg):
        tape = self.tape
        j = seg.index
        lo = 0 if j == 0 else self.cuts[j - 1].op.guard_range[1]
        hi = _INF if j == len(self.cuts) else self.cuts[j].op.guard_range[0]
        out = []
        for k, g in enumerate(tape.guards):
            raw = tape.kept_raw[k]
            if lo <= raw < hi:
                out.append(g)
                continue
            # the metadata-tagged guards of the cut op after this segment (the op's
            # own validity and metadata checks, nested ops' included) are checked
            # here, before eager runs it; its kernel-tagged ones are dropped
            if j < len(self.cuts):
                op = self.cuts[j].op
                if op.guard_range[0] <= raw < op.guard_range[1]:
                    if tape.guard_rows[raw][1] == 0:
                        out.append(g)
        return out

    def _plan_cut(self, j, cut):
        op = cut.op

        def ref(x):
            if isinstance(x, _TracedTensor):
                return self._ref(x)
            if isinstance(x, torch.Tensor):
                raise PartitionDeclined(f"op {op.index} takes a real tensor argument")
            return _expr(x) if isinstance(x, _SYM_TYPES) else x

        cut.args = tree_map(ref, op.args)
        cut.kwargs = tree_map(ref, op.kwargs)
        leaves, cut.out_spec = tree_flatten(op.outputs)
        cut.out_leaves = [ref(x) for x in leaves]
        # every root the op made that a later record reads must be one of its outputs
        returned = {leaf.root for leaf in cut.out_leaves if isinstance(leaf, _Ref)}
        for name in cut.produced:
            if name not in returned and any(s >= op.seq[1] for s in self.uses[name]):
                raise PartitionDeclined(
                    f"op {op.index} made {name}, which a later record reads, but did not return it"
                )
        tape = self.tape
        for i, o in enumerate(tape.outputs):
            if (
                o.identity is not None
                or not o.root.allocation
                or self._name(o.root) not in cut.produced
            ):
                continue
            for k, leaf in enumerate(cut.out_leaves):
                if (
                    isinstance(leaf, _Ref)
                    and leaf.root == o.root.name
                    and leaf.dtype == o.dtype
                    and leaf.sizes == tuple(_expr(v) for v in o.sizes)
                    and leaf.strides == tuple(_expr(v) for v in o.strides)
                    and leaf.offset == _expr(o.offset)
                ):
                    self.out_plan[i] = ("cut", j, k)
                    break
            else:
                self.out_plan[i] = ("view_cut", self._out_ref(o))

    def _out_ref(self, o):
        return _Ref(
            self._name(o.root),
            tuple(_expr(v) for v in o.sizes),
            tuple(_expr(v) for v in o.strides),
            _expr(o.offset),
            o.dtype,
        )

    def _plan_output(self, i, o):
        if o.identity is not None:
            kind, index = o.identity
            if kind == "argument":
                return ("arg", index)
            return ("same", index)
        if not o.root.allocation:
            index = next(
                k for k, inp in enumerate(self.tape.inputs) if inp.root is o.root
            )
            if o.dtype != self.tape.inputs[index].dtype:
                raise PartitionDeclined("an output views an input under another dtype")
            return ("view", self._out_ref(o), index)
        if any(
            self._name(o.root) == rec.name
            for seg in self.segments
            for rec in seg.python_allocs
        ):
            return ("python_alloc", self._out_ref(o))
        raise PartitionDeclined(f"output {i} over {self._name(o.root)} has no producer")

    # ---- the call ----

    def _env(self, box):
        env = {}
        for name, kind, index, dim in self.reads:
            if kind == "size":
                env[name] = box[index].size(dim)
            elif kind == "stride":
                env[name] = box[index].stride(dim)
            elif kind == "offset":
                env[name] = box[index].storage_offset()
            else:
                env[name] = index
        return env

    def _ints(self, values, env):
        ev = self.evaluator.ev
        return tuple(int(ev(v, env)) for v in values)

    def _base(self, root, box, boundary):
        if root in boundary:
            return boundary[root]
        index = self._input_index.get(root)
        if index is None:
            raise PartitionDeclined(f"root {root} is not available at the cut")
        return box[index]

    def _materialize(self, leaf, env, box, boundary):
        if isinstance(leaf, _Ref):
            base = self._base(leaf.root, box, boundary)
            return torch.as_strided(
                base,
                self._ints(leaf.sizes, env),
                self._ints(leaf.strides, env),
                int(self.evaluator.ev(leaf.offset, env)),
            )
        if isinstance(leaf, sympy.Basic):
            v = self.evaluator.ev(leaf, env)
            return (
                bool(v) if isinstance(v, (bool, sympy.logic.boolalg.BooleanAtom)) else v
            )
        return leaf

    def _root_view(self, t):
        """A uint8 view of a tensor's whole storage: a prebound root's boxed tensor."""
        storage = t.untyped_storage()
        return torch.empty(0, dtype=torch.uint8, device=t.device).set_(
            storage, 0, (storage.nbytes(),), (1,)
        )

    def _build_segment(self, seg, args, boundary):
        from . import direct_hosttrace as dh

        owner = self.owner
        t0 = time.perf_counter()
        lowered = dh.lower_tape(
            seg.part,
            args,
            arena=False,
            output_arena=False,
            allocseq=True,
            device=self.device,
            prebound=tuple(seg.prebound),
        )
        seg.lower_s = time.perf_counter() - t0
        plan = lowered.sequence
        if plan is None and seg.prebound:
            raise PartitionDeclined(f"segment {seg.index} lowered without a sequence")
        for name in seg.prebound:
            if plan is None or not plan.covers(name):
                raise PartitionDeclined(
                    f"segment {seg.index}: prebound root {name} is not a sequence root"
                )
        seg.lowered = lowered
        t0 = time.perf_counter()
        sequence = None
        if plan is not None:
            sequence = AllocatorSequence(self.device, owner.allocseq_mode)
            sequence.bind(plan, lowered.tensors(args))
        seg.sequence = sequence
        seg.prebound = [
            (name, plan.root(name) - plan.input_index) for name in seg.prebound
        ]
        positions = lowered.symbols.positions
        every = positions == list(range(lowered.nargs))
        seg.boxer = torch._C._HostTraceBoxer(
            None if every else positions, lowered.written_positions
        )
        roots = self._roots(seg, boundary)
        seg.entry = dh.prepare_hosttrace(
            lowered,
            args,
            staging_depth=owner.staging_depth,
            arena=None,
            outputs=None,
            sequence=None if sequence is None else SimpleNamespace(roots=roots),
        )
        try:
            seg.dispatch = torch._C._cuda_make_boxed_dispatch(
                ((seg.entry, lowered.registration),), lambda box: dh._MISSED, ()
            )
        except BaseException:
            seg.entry.close()
            raise
        seg.prepare_s = time.perf_counter() - t0

    def _roots(self, seg, boundary):
        if seg.sequence is None:
            return []
        roots = list(seg.sequence.roots)
        for name, k in seg.prebound:
            roots[k] = self._root_view(boundary[name])
        return roots

    def _run_segment(self, seg, args, boundary):
        """The segment's program at `args` with the boundary bound: its outputs, or
        None when its predicate rejects the call."""
        from . import direct_hosttrace as dh

        box = seg.boxer(args, None)
        if seg.sequence is not None:
            seg.sequence.take(box)
            box.extend(self._roots(seg, boundary))
        outputs = seg.dispatch(box)
        if outputs is dh._MISSED:
            lowered = seg.lowered
            if not dh.check_predicate(lowered, box, regions=False):
                return None
            if seg.sequence is not None and not dh.check_predicate(
                lowered, box, mode=dh._MODE_ARENA
            ):
                # the hold-mode roots are too small for this call: rebind (no miss)
                seg.sequence.bind(lowered.sequence, box)
                self.owner.sequence_rebinds += 1
            else:
                # a closed region's new key: harvest it for the segment's site
                identity = tuple(self.tape.device_identity)
                registered = False
                for region in lowered.regions:
                    key = torch._C._cuda_kernel_template_take_miss(region.site)
                    if key is None:
                        continue
                    template = dh._region_template(
                        region, tuple(key), self.device, identity
                    )
                    dh._register_region_variant(
                        region,
                        tuple(key),
                        template,
                        lowered.region_nodes[region.site],
                        lowered.region_arena,
                    )
                    registered = True
                if not registered:
                    return None
            box = seg.boxer(args, None)
            if seg.sequence is not None:
                box.extend(self._roots(seg, boundary))
            outputs = seg.dispatch(box)
            if outputs is dh._MISSED:
                return None
        seg.calls += 1
        return list(outputs)

    def _run_cut(self, j, cut, env, box, boundary, known):
        op = cut.op
        mat = lambda leaf: self._materialize(leaf, env, box, boundary)  # noqa: E731
        args = tree_map(mat, cut.args)
        kwargs = tree_map(mat, cut.kwargs)
        for t in tree_flatten((args, kwargs))[0]:
            if isinstance(t, torch.Tensor):
                known.add(t.untyped_storage().data_ptr())
        t0 = time.perf_counter()
        with torch.no_grad():
            out = op.func(*args, **kwargs)
        cut.eager_s = time.perf_counter() - t0
        leaves = tree_flatten(out)[0]
        if len(leaves) != len(cut.out_leaves):
            raise SwapMismatch(
                f"op {op.index} {op.func} returned {len(leaves)} values, the tape recorded {len(cut.out_leaves)}"
            )
        fresh = {}
        for got, want in zip(leaves, cut.out_leaves):
            if not isinstance(want, _Ref):
                if isinstance(got, torch.Tensor) or (
                    want is not None
                    and got != self._materialize(want, env, box, boundary)
                ):
                    raise SwapMismatch(
                        f"op {op.index} {op.func}: a non-tensor output differs"
                    )
                continue
            if not isinstance(got, torch.Tensor):
                raise SwapMismatch(
                    f"op {op.index} {op.func}: an output is not a tensor"
                )
            sizes, strides = self._ints(want.sizes, env), self._ints(want.strides, env)
            offset = int(self.evaluator.ev(want.offset, env))
            if (
                tuple(got.shape) != sizes
                or got.stride() != strides
                or got.storage_offset() != offset
                or got.dtype != want.dtype
            ):
                raise SwapMismatch(
                    f"op {op.index} {op.func}: eager returned {want.dtype if got.dtype == want.dtype else got.dtype} "
                    f"{tuple(got.shape)} strides {got.stride()} offset {got.storage_offset()}, the tape's "
                    f"expressions give {sizes} strides {strides} offset {offset} at this call"
                )
            storage = got.untyped_storage().data_ptr()
            if want.root in cut.produced:
                if storage in known:
                    raise SwapMismatch(
                        f"op {op.index} {op.func}: an output the tape recorded as a fresh allocation ({want.root}) aliases an existing storage"
                    )
                if fresh.setdefault(want.root, storage) != storage:
                    raise SwapMismatch(
                        f"op {op.index} {op.func}: two outputs of one recorded root ({want.root}) have different storages"
                    )
                if got.untyped_storage().nbytes() < int(
                    self.evaluator.ev(self._alloc_nbytes[want.root], env)
                ):
                    raise SwapMismatch(
                        f"op {op.index} {op.func}: {want.root} is smaller than recorded"
                    )
            else:
                base = self._base(want.root, box, boundary)
                if base.untyped_storage().data_ptr() != storage:
                    raise SwapMismatch(
                        f"op {op.index} {op.func}: an output the tape recorded over {want.root} has another storage"
                    )
        # the fresh roots, by their first output; the same storage under two roots is
        # refused above, so a root's leaf stands for it
        for got, want in zip(leaves, cut.out_leaves):
            if (
                isinstance(want, _Ref)
                and want.root in cut.produced
                and want.root not in boundary
            ):
                boundary[want.root] = got
                known.add(got.untyped_storage().data_ptr())
        return leaves

    def serve(self, args, box):
        """Serve `args` (boxed as `box`, the tape's tensors first) by the partition:
        the outputs in the tape's order, or None when a segment's predicate rejects
        the call before anything ran. Raises SwapMismatch (a functional cut op; the
        caller misses as before) or RuntimeError (an op that wrote an input) after a
        swap check failed, PartitionDeclined from a build."""
        from . import direct_hosttrace as dh

        tape = self.tape
        if not hasattr(self, "_input_index"):
            self._input_index = {inp.root.name: k for k, inp in enumerate(tape.inputs)}
            self._alloc_nbytes = {
                name: _raw_nbytes(rec) for name, rec in self.alloc_by_name.items()
            }
        t0 = time.perf_counter()
        # every built segment's facts must hold before anything runs: a later
        # segment's miss after an eager op ran could not be re-traced safely
        for seg in self.segments:
            if seg.lowered is None:
                continue
            probe = seg.boxer(args, None)
            if seg.sequence is not None:
                probe.extend(seg.sequence.roots)
            if not dh.check_predicate(seg.lowered, probe, regions=False):
                return None
        env = self._env(box)
        boundary: dict = {}
        # the storages a cut op's fresh output must not alias: the boundary's and,
        # per op, its own arguments' (added at the cut)
        known: set = set()
        seg_outputs: list = [None] * len(self.segments)
        cut_outputs: list = [None] * len(self.cuts)
        ran = False  # GPU work issued: a rejection after it cannot fall back
        for seg in self.segments:
            for rec in seg.python_allocs:
                t = torch.empty_strided(
                    self._ints([_expr(v) for v in rec.sizes], env),
                    self._ints([_expr(v) for v in rec.strides], env),
                    dtype=rec.dtype,
                    device=torch.device("cuda", self.device),
                )
                boundary[rec.name] = t
                known.add(t.untyped_storage().data_ptr())
            if seg.part is not None:
                if seg.lowered is None:
                    self._build_segment(seg, args, boundary)
                outputs = self._run_segment(seg, args, boundary)
                if outputs is None:
                    if not ran:
                        return None
                    raise RuntimeError(
                        f"host_trace partition: segment {seg.index} rejected the call after earlier segments ran (a guard not among the tape's, or a region topology change)"
                    )
                ran = True
                seg_outputs[seg.index] = outputs
                n = len(seg.t_outputs)
                for k, name in enumerate(seg.escapes):
                    boundary[name] = outputs[n + k]
                    known.add(outputs[n + k].untyped_storage().data_ptr())
            if seg.index < len(self.cuts):
                cut = self.cuts[seg.index]
                ran = True
                try:
                    cut_outputs[seg.index] = self._run_cut(
                        seg.index, cut, env, box, boundary, known
                    )
                except SwapMismatch as e:
                    aliased = any(
                        isinstance(leaf, _Ref) and leaf.root not in cut.produced
                        for leaf in cut.out_leaves
                    )
                    if aliased:
                        raise RuntimeError(
                            f"host_trace partition: the swap check failed for an op that wrote or aliased an input; the guard the call missed was metadata-affecting, not kernel-only: {e}"
                        ) from e
                    raise
        results = []
        for i, plan in enumerate(self.out_plan):
            kind = plan[0]
            if kind == "seg":
                results.append(seg_outputs[plan[1]][plan[2]])
            elif kind == "cut":
                results.append(cut_outputs[plan[1]][plan[2]])
            elif kind == "arg":
                results.append(args[plan[1]])
            elif kind == "same":
                results.append(results[plan[1]])
            elif kind == "view":
                results.append(self._materialize(plan[1], env, box, boundary))
            else:  # view_cut, python_alloc
                results.append(self._materialize(plan[1], env, box, boundary))
        if not self.built:
            self.built = True
            self.build_s = time.perf_counter() - t0
        self.serves += 1
        return results

    def stats(self):
        return {
            "ops": [
                (op.index, str(op.func), list(op.seq), list(op.guard_range))
                for op in (c.op for c in self.cuts)
            ],
            "segments": [
                {
                    "index": s.index,
                    "range": [s.lo, s.hi],
                    "launches": 0 if s.part is None else len(s.part.launches),
                    "regions": 0 if s.part is None else len(s.part.regions),
                    "memsets": 0 if s.part is None else len(s.part.memsets),
                    "allocs": 0 if s.part is None else len(s.part.allocs),
                    "guards": 0 if s.part is None else len(s.part.guards),
                    "prebound": [n for n, _ in s.prebound]
                    if s.lowered
                    else list(s.prebound),
                    "escapes": list(s.escapes),
                    "python_allocs": [r.name for r in s.python_allocs],
                    "t_outputs": list(s.t_outputs),
                    "lower_s": s.lower_s,
                    "prepare_s": s.prepare_s,
                    "sequence_roots": 0
                    if s.lowered is None or s.lowered.sequence is None
                    else len(s.lowered.sequence.names),
                    "sequence_peak": 0
                    if s.lowered is None or s.lowered.sequence is None
                    else s.lowered.sequence.peak_at_hints,
                }
                for s in self.segments
            ],
            "serves": self.serves,
            "build_s": self.build_s,
            "eager_s": [c.eager_s for c in self.cuts],
        }

    def close(self):
        for seg in self.segments:
            seg.close()


def serve_existing(owner, family, args, box):
    """An existing partition of a variant of `family` whose segments accept the
    call: its outputs, else None (nothing ran)."""
    for partition in owner.partitions:
        if partition.variant.family is not family:
            continue
        outputs = partition.serve(args, box)
        if outputs is not None:
            owner.partition_serves += 1
            return outputs
    return None


def serve_partitioned(owner, family, args, box):
    """The entry's local-miss path (`HostTraceReplay._missed`, after the existing
    partitions were tried): the variant whose only failing kept guards are
    kernel-only guards of one or a few top-level ops, partitioned at them. Returns
    the outputs, or None when the miss is not local."""
    for variant in family.variants:
        ops, why = local_miss_ops(variant.tape, variant.lowered, box)
        if ops is None:
            owner.partition_why = why
            continue
        if any(
            p.variant is variant and p.op_indices == tuple(ops)
            for p in owner.partitions
        ):
            continue  # exists and did not take the call
        try:
            partition = Partition(owner, variant, ops)
        except PartitionDeclined as e:
            owner.miss_log.append(
                (
                    owner.calls,
                    f"local miss at ops {ops}, partition declined: {e}",
                    "miss",
                )
            )
            return None
        try:
            outputs = partition.serve(args, box)
        except SwapMismatch as e:
            # the guard was metadata-affecting after all: the call misses (a trace);
            # the ops' kernel-tagged guards are reclassified for this tape
            partition.close()
            tape = variant.tape

            def top(op_index):
                op = tape.ops[op_index]
                while op.depth > 0:
                    op = tape.ops[op.parent]
                return op.index

            for k in range(len(tape.guards)):
                raw = tape.kept_raw[k]
                op_index, depth, origin = tape.guard_rows[raw]
                if op_index >= 0 and depth > 0 and top(op_index) in ops:
                    tape.guard_rows[raw] = (op_index, 0, origin)
                tape.guard_also[raw] = [
                    (o, 0 if o >= 0 and top(o) in ops else d, g)
                    for o, d, g in tape.guard_also.get(raw, ())
                ]
            owner.miss_log.append(
                (owner.calls, f"swap check refused ops {ops}: {e}", "miss")
            )
            owner.swap_refusals += 1
            return None
        except PartitionDeclined as e:
            partition.close()
            owner.miss_log.append(
                (
                    owner.calls,
                    f"local miss at ops {ops}, partition declined: {e}",
                    "miss",
                )
            )
            return None
        except BaseException:
            partition.close()
            raise
        owner.partitions.append(partition)
        owner.partition_builds += 1
        if outputs is None:
            return None
        owner.partition_serves += 1
        names = ", ".join(f"op {i} {tape_op_name(variant.tape, i)}" for i in ops)
        owner.miss_log.append(
            (
                owner.calls,
                f"local miss: kernel-only guards of {names}: {why}",
                f"partition {len(owner.partitions)}",
            )
        )
        return outputs
    return None


def tape_op_name(tape, index):
    return str(tape.ops[index].func)
