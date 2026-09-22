"""Allocator-order replay of a lowered host-trace tape's temporaries (direct_hosttrace).

Eager's peak comes from allocating each temporary before its first use and freeing it at
its death; the runtime's replay allocates every buffer of a call up front and clears them
after the launch (the sum of the roots). This module replays the tape's allocation and
deallocation sequence through the caching allocator instead: before the launch, in tape
order, each temporary is allocated (``raw_alloc``) before its first use and freed at its
last use, in a MemPool private to the entry, so the addresses the sequence yields overlap
exactly as eager's did and the pool holds the sequence's peak, not the sum. The
temporaries are boxed as one root each (a uint8 alias of the block, no owner), so every
pointer parameter over a temporary is ``root + displacement``: the rebind the runtime
already runs per call (an unchanged address patches nothing).

Why the pool is private: after the sequence every block is free again in the allocator's
eyes, and the launch that reads them comes later on the same stream. Within the pool the
only allocations are the sequence's own, so a freed block can be taken only by a later
allocation of the same sequence (the intended reuse) or by a later call's sequence, which
is stream-ordered after this call's graph launch exactly as eager's next allocation is
ordered after the kernel that last used the block. The one hazard the argument depends on
is that nothing else allocates in the pool between the sequence's first allocation and the
launch: the pool is entered only inside ``bind`` (this thread, this stream) and is never a
last resort for foreign allocations (``use_on_oom`` stays False).

Two modes per family: ``hold`` (the default) keeps the blocks of the last sequence while
every call's sizes fit them (0 allocator calls per call; the held memory is the pool's
segments at the sequence's peak); ``replay`` runs the sequence every call (2 allocator
calls per temporary). Hold falls back to a sequence when a call's sizes exceed the bound
ones (the predicate's size terms fail and the miss path rebinds), and when the allocator
reported pressure since the last sequence (an OOM observer fired, or the device's retry /
OOM counters moved) the next sequence runs in a fresh pool and the previous one is dropped
(its segments are cudaFree'd, which waits for the launches that used them).
"""

import ctypes
import math
from dataclasses import dataclass, field

import torch


ALIGN = 512  # the caching allocator's block granularity (kMinBlockSize): every address is a multiple

_raw_alloc = torch._C._cuda_cudaCachingAllocator_raw_alloc
_raw_delete = torch._C._cuda_cudaCachingAllocator_raw_delete
_begin = torch._C._cuda_beginAllocateCurrentThreadToPool
_end = torch._C._cuda_endAllocateToPool
_release = torch._C._cuda_releasePool
_storage_from = torch._C._construct_storage_from_data_pointer
_current_raw_stream = torch._C._cuda_getCurrentRawStream
_device_stats = torch._C._cuda_memoryStats

# the out-of-memory observer: a count of hard OOMs anywhere in the process since import
# (a sequence compares it with the count at its last run)
_ooms = [0]
_observer_attached = [False]


def _note_oom(*_):
    _ooms[0] += 1


def _attach_observer():
    if not _observer_attached[0]:
        _observer_attached[0] = True
        torch._C._cuda_attach_out_of_memory_observer(_note_oom)


@dataclass
class SequencePlan:
    """One variant's sequence: the temporaries as roots (root k is boxed at
    ``input_index + k``), their byte-size expressions, the program (in tape order, ``k``
    allocates root k and ``~k`` frees it), the size and stride facts the sizes read (the
    compact fact vector the compiled ``seq_sizes`` takes), and the plan's figures at the
    preparation shapes."""

    input_index: int
    names: tuple
    sizes: tuple  # sympy byte sizes, per root
    program: tuple
    size_reads: tuple  # (boxed index, dim) rows: the fact vector's size slots
    stride_reads: tuple  # (boxed index, dim) rows after them
    hints: dict
    intervals: tuple = ()  # (allocate position, free position) in the program, per root
    peak_at_hints: int = 0  # the largest live set at the hints (bytes, ALIGN-rounded)
    sum_at_hints: int = 0  # the sum of the rounded sizes there (the allocate-all peak)
    # the seqs of the nodes behind programmatic edges the free points moved past (the
    # rule's input as planned, sorted)
    programmatic: tuple = ()
    # root indices the caller binds itself (a partition's boundary: a block another
    # program or an eager op allocated): in `names` and rebased like any root, never
    # allocated or freed by the program, outside the peak
    prebound: tuple = ()
    sizes_address: int = (
        0  # extern "C" int64_t seq_sizes(const int64_t* facts, int64_t* out)
    )
    _index: dict = field(default_factory=dict, repr=False)
    _reader: object = field(default=None, repr=False)
    _facts: object = field(default=None, repr=False)
    _out: object = field(default=None, repr=False)

    def __post_init__(self):
        self._index = {name: k for k, name in enumerate(self.names)}

    def covers(self, name):
        return name in self._index

    def root(self, name):
        return self.input_index + self._index[name]

    @property
    def signature(self):
        # what a family's variants must share for one sequence object to serve them all:
        # the program, the sizes and where their facts come from
        return (
            self.program,
            tuple(str(s) for s in self.sizes),
            self.size_reads,
            self.stride_reads,
            self.prebound,
        )

    def facts(self, box):
        return [box[i].size(d) for i, d in self.size_reads] + [
            box[i].stride(d) for i, d in self.stride_reads
        ]

    def sizes_at(self, facts):
        """Every root's bytes at a fact vector, by the compiled function (a size that
        evaluates to nothing is one byte: a block the launch never touches)."""
        if self._reader is None:
            self._reader = ctypes.CFUNCTYPE(
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
            )(self.sizes_address)
            self._facts = (ctypes.c_int64 * max(1, len(facts)))()
            self._out = (ctypes.c_int64 * len(self.names))()
        if facts:
            self._facts[:] = facts
        if self._reader(self._facts, self._out) != 0:
            raise OverflowError("host_trace replay: a temporary's size overflowed")
        return list(self._out)

    def check(self, sizes, addresses):
        """The debug assertion: no two roots whose lifetimes overlap in the program share
        bytes, every address is a multiple of ALIGN."""
        for k, address in enumerate(addresses):
            if address % ALIGN:
                raise AssertionError(
                    f"sequence root {k} at {address:#x}, not {ALIGN}-byte aligned"
                )
        for a, (fa, la) in enumerate(self.intervals):
            for b in range(a + 1, len(self.intervals)):
                fb, lb = self.intervals[b]
                if fa < lb and fb < la:
                    lo, hi = addresses[a], addresses[a] + sizes[a]
                    if lo < addresses[b] + sizes[b] and addresses[b] < hi:
                        raise AssertionError(
                            f"sequence roots {a} [{lo:#x}, {hi:#x}) and {b} [{addresses[b]:#x}, {addresses[b] + sizes[b]:#x}) overlap while both live"
                        )


def _free_points(last, programmatic, nodes):
    """Each temporary's free point: its last use, moved past the run of nodes after it
    whose incoming edge is programmatic (a consumer that may write before it waited for
    the work upstream), so the block is reusable from the first node that waited for
    everything upstream, or after the last node."""
    if not programmatic:
        return dict(last)
    if not nodes:
        raise ValueError(
            "host_trace lowering: programmatic edges need the launch order (nodes)"
        )
    missing = set(last.values()) - set(nodes)
    if missing:
        raise ValueError(
            f"host_trace lowering: uses at seqs {sorted(missing)} are not in the launch order"
        )
    programmatic = set(programmatic)
    # run_end: the end of the programmatic run the next node in launch order starts
    moved, run_end = {}, None
    for seq in sorted(nodes, reverse=True):
        moved[seq] = seq if run_end is None else run_end
        run_end = (seq if run_end is None else run_end) if seq in programmatic else None
    return {name: moved[seq] for name, seq in last.items()}


def plan_sequence(
    rows, uses, hints, input_index, order, programmatic=(), nodes=(), prebound=()
):
    """``rows``: (name, byte-size expression) of the temporaries to sequence; ``uses``:
    name -> the seqs of the events touching it; ``order``: name -> the tape's
    allocation index (eager's allocation order); ``hints``: symbol -> value at the
    preparation inputs; ``programmatic``: the seqs of the nodes whose incoming graph
    edge is programmatic, with ``nodes`` the seqs of every node in launch order (empty:
    every edge is a full-completion edge). Each temporary is allocated before the first
    event that touches it (in allocation order among those of one event) and freed
    after the last, or after the programmatic consumers that follow the last
    (``_free_points``). Names in ``prebound`` are roots the caller binds (a block
    allocated outside this program): rebased like the others, with no allocation or
    free in the program and no share of the peak."""
    prebound = set(prebound)
    first = {name: min(uses[name]) for name, _ in rows}
    last = {name: max(uses[name]) for name, _ in rows if name not in prebound}
    free = _free_points(last, programmatic, nodes)
    names = tuple(
        sorted((name for name, _ in rows), key=lambda n: (first[n], order[n]))
    )
    sizes = dict(rows)
    index = {name: k for k, name in enumerate(names)}
    program = []
    for seq in sorted({*(first[n] for n in last), *free.values()}):
        program.extend(index[n] for n in names if n in last and first[n] == seq)
        program.extend(~index[n] for n in names if n in last and free[n] == seq)
    intervals = [[0, 0] if n not in prebound else [-1, -1] for n in names]
    for position, op in enumerate(program):
        intervals[op if op >= 0 else ~op][0 if op >= 0 else 1] = position
    rounded = [ALIGN * math.ceil(int(sizes[n].xreplace(hints)) / ALIGN) for n in names]
    live = peak = 0
    for op in program:
        if op >= 0:
            live += rounded[op]
            peak = max(peak, live)
        else:
            live -= rounded[~op]
    return SequencePlan(
        input_index,
        names,
        tuple(sizes[n] for n in names),
        tuple(program),
        (),
        (),
        dict(hints),
        intervals=tuple(tuple(i) for i in intervals),
        peak_at_hints=peak,
        sum_at_hints=sum(r for r, n in zip(rounded, names) if n not in prebound),
        programmatic=tuple(sorted(programmatic)),
        prebound=tuple(k for k, n in enumerate(names) if n in prebound),
    )


class AllocatorSequence:
    """One family's roots and private pool. ``take(box)`` is the hot path: in hold mode
    with valid roots it returns them untouched; otherwise it runs the sequence at the
    call's sizes (``bind``). The roots are uint8 tensors over the blocks' addresses
    (no owner: the pool's segments stay reserved while the pool object lives), each
    ``set_`` again only when its address or size changed."""

    def __init__(self, device, mode="hold"):
        if mode not in ("hold", "replay"):
            raise ValueError(
                f"host_trace replay: sequence mode {mode!r} is not 'hold' or 'replay'"
            )
        _attach_observer()
        self.device = int(device)
        self.mode = mode
        self.plan = None
        self.pool = None
        self.roots = []
        self.addresses = []
        self.nbytes = []
        self.valid = False
        self.binds = 0  # sequences run
        self.rebinds = 0  # roots whose address or size changed over a sequence
        self.pools = 0  # pools created (one, plus one per pressure release)
        self.pressure_releases = 0
        self.peak = 0  # the live bytes the last sequence reached, as requested
        self._counters = (
            None  # (ooms seen, num_alloc_retries + num_ooms) at the last sequence
        )

    def take(self, box):
        if self.valid and self.mode == "hold":
            return self.roots
        self.bind(self.plan, box)
        return self.roots

    def _counters_now(self):
        stats = _device_stats(self.device)
        return (_ooms[0], stats["num_alloc_retries"] + stats["num_ooms"])

    def pressure(self):
        """Whether the allocator reported pressure since the pool was made: a hard
        OOM anywhere (the observer), or the device's retry / OOM counters moved."""
        return self._counters != self._counters_now()

    def bind(self, plan, box):
        """Run the sequence at the sizes of ``box`` and rebind the roots."""
        if plan is not self.plan:
            if self.plan is not None and plan.signature != self.plan.signature:
                raise ValueError(
                    "host_trace replay: a family's variants must share one sequence plan"
                )
            self.plan = plan
            n = len(plan.names)
            device = torch.device("cuda", self.device)
            while len(self.roots) < n:
                self.roots.append(torch.empty(0, dtype=torch.uint8, device=device))
                self.addresses.append(0)
                self.nbytes.append(0)
        sizes = plan.sizes_at(plan.facts(box))
        if self.pool is None or self.pressure():
            if self.pool is not None:
                self.pressure_releases += 1
            self.pool = None  # a dropped pool frees its segments at once (cudaFree waits for the launches)
            self.pool = torch.cuda.MemPool()
            self.pools += 1
            self._counters = self._counters_now()
        addresses = self.run(plan.program, sizes)
        device = torch.device("cuda", self.device)
        prebound = set(plan.prebound)
        for k, (address, n) in enumerate(zip(addresses, sizes)):
            if k in prebound:
                continue  # the caller's block: boxed by the caller, never ours
            if address != self.addresses[k] or n != self.nbytes[k]:
                self.roots[k].set_(_storage_from(address, device, n), 0, (n,), (1,))
                self.addresses[k] = address
                self.nbytes[k] = n
                self.rebinds += 1
        self.valid = True
        self.binds += 1

    def run(self, program, sizes):
        """The allocation / deallocation sequence in the pool on the current stream (the
        entry's bound stream at a served call): the addresses per root."""
        stream = _current_raw_stream(self.device)
        pool_id = self.pool.id
        addresses = [0] * len(sizes)
        live_set = set()
        live = peak = 0
        _begin(self.device, pool_id)
        try:
            for op in program:
                if op >= 0:
                    n = sizes[op]
                    addresses[op] = _raw_alloc(n, stream)
                    live_set.add(op)
                    live += n
                    if live > peak:
                        peak = live
                else:
                    k = ~op
                    _raw_delete(addresses[k])
                    live_set.discard(k)
                    live -= sizes[k]
        except BaseException:
            for k in live_set:
                _raw_delete(addresses[k])
            raise
        finally:
            _end(self.device, pool_id)
            _release(self.device, pool_id)
        self.peak = peak
        return addresses

    def reserved(self):
        """Bytes of the pool's segments (what hold mode holds)."""
        if self.pool is None:
            return 0
        return sum(
            s["total_size"]
            for s in torch.cuda.memory_snapshot(self.pool.id, include_traces=False)
        )

    def check(self, plan, box):
        n = len(plan.names)
        plan.check(self.nbytes[:n], self.addresses[:n])

    def release(self):
        self.valid = False
        self.pool = None
