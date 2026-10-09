"""Host tracing (private): a function served by replaying the CUDA graphs
captured from its tapes, patched per call.

A HostTraceReplay traces a call, lowers the tape and captures it once
(a variant). A later call is checked in two phases. Validation has no
effect: the argument contract selects the family of variants traced under
the same contract, and one evaluation of a variant's compiled program checks
every guard, allocation requirement and eager call's arguments; then that
no argument a traced step writes overlaps another it reads
(Tape.argument_pairs), or the call runs eagerly. Only a call
a variant holds commits, step by step: it allocates as eager would
(_host_trace_memory); for a run of launches it patches the kernel and
memset nodes whose parameters changed (a pointer to an eager output is known
only once its op has run, so each run is patched when it starts) and replays
the run's graph; an eager call it runs itself. A hit runs in C++ from the
call to its return (torch._C._HostTraceEntry, the base; _host_trace_native),
entering Python only for an eager step's op; a miss, a trace and every
fallback run here. A keyed site's key selects its nodes' parameters from
its table; a key the table lacks is bound here and added once (a binding
that does not fit the site declines the key for the variant). An opaque
call that runs eagerly binds its key, and on a miss its provider learns
from that run; a variant with one commits from here, which traces again at
a key that now binds, or that the provider refused (a plain eager step, and
the variant no longer learns). Everything runs on the caller's current
stream, the one stream the trace admitted, so stream order alone orders the
steps as eager's, with no events.

An eager call runs below ADInplaceOrView, where the trace saw it (after
autograd), so it records no autograd history and bumps no version counter,
like a graph's writes. Its outputs must match the trace's prediction (sizes,
strides, storage offset, dtype, device, and for an alias the argument
itself). A trace with a warm-up checks this at the traced call and
declines. A mismatch at a replay is a bug in the op's fake kernel: the
steps before it have run, so it cannot retrace; it raises AssertionError and
drops the variant. Either way a later trace of this HostTraceReplay with a
call to that op declines. A Triton launch whose
compilation the trace does not describe runs through Triton
(`JITFunction.run`), and why is logged once (trace_structured).
A call from inside one of its own eager steps needs no special case: its
buffers are its own allocations, and it leaves the kernel nodes it patches
recorded, so the outer call patches what it needs when its next run starts.
An entry's first call runs eagerly (cudagraph trees' warm-up). A later
call no variant holds is traced at its own inputs after a warm-up (the
call; under trust only the first trace warms up, and a later one's call is
the replay of its new capture) and joins its family, unless it
folds into a variant whose graph guards hold it but not the own guards of
some top-level ops (their selectors): each gains the trace's launches of its
op as an entry, where the two tapes agree otherwise
(_host_trace_lower_tape.fold). A segment whose capture fails is local: the
call traces again with the ops of the launches it failed at as eager calls
(SegmentFailed). A trace, lowering or capture that declines otherwise runs
the function eagerly, and that call's
exact class is not traced again; under trust a Declined is the graph's, and
no call is. Every eager fallback records its reason in `declines`, once per
reason. With `fullgraph` none happens: each raises EagerFallback.
"""

from __future__ import annotations

import builtins
import contextlib
import copy
import functools
import dataclasses
import inspect
import itertools
import logging
import queue
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
from torch._logging import trace_structured
from torch._prims.rng_prims import _impl_graphsafe_rng
from torch.cuda import _host_trace_cute  # noqa: F401  hooks cute.compile
from torch.cuda import _host_trace_ir as _ir
from torch.cuda._host_trace import _drop_tracebacks, Declined, declined, EagerFallback, ProcessHold
from torch.cuda._host_trace_capture import capture_tape, instantiate_form, plain_attributes, SegmentFailed
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.cuda._host_trace_lower_tape import fold, FoldRefused, lower_guards, lower_tape, PointerSlot, PredictedOutput, ScalarSlot
from torch.cuda._host_trace_memory import arena_addresses, auto_memory, check_plan, plan_memory, relocate, relocate_entries, size_classes, split_runs, SPLITS
from torch.cuda._host_trace_opaque import fill_uniform, key_guards, library_state_as, span_bytes, takes_generator
from torch.cuda._host_trace_program import Status
from torch.cuda._host_trace_redispatch import redispatch
from torch.cuda._host_trace_native import (
    binding_row,
    CAPTURE,
    flatten_variant,
    HIT,
    launch_row,
    MISALIGNED,
    MISS,
    native_variant,
)
from torch.cuda._host_trace_tape import (
    _active,
    _ALLOC_ALIGNMENT,
    _gc_hold,
    _placeholder,
    argument_contract,
    bind_opaque,
    current_trace,
    EagerCall,
    new_private_stream,
    OpaqueCall,
    seed_offset_on_device,
    trace,
)
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from torch._ops import OpOverload
    from torch.cuda._host_trace_capture import CapturedTape
    from torch.cuda._host_trace_lower_tape import (
        LoweredEagerCall,
        LoweredKeyedSite,
        LoweredLaunch,
        LoweredMemset,
        LoweredOpaqueCall,
    )
    from torch.cuda._host_trace_memory import MemoryPlan
    from torch.cuda._host_trace_opaque import KeyedSite, OpaqueBinding, OpaqueKey, OpaqueProvider
    from torch.cuda._host_trace_tape import Tape, TrustedInputs


log = logging.getLogger(__name__)

# a trace with opaque calls whose keys do not bind learns them out of band
# (_learn) and binds them in its own tape, a guard the bindings add kept
# graph-level, so its variant does not learn (no learner run, relower or second
# trace); the relower keeps such guards too. False: the variant learns
oob_learn = True

# a key a non-learning variant's provider refuses at the call is a plain eager
# step, guarding the key, of a variant built at the call from its tape before
# its opaque calls were bound (_eager_sites): no trace. False: the call traces
# again, as a trace guarding the refused key
refused_eager_steps = True


def _exact_class(contract: tuple, args: Sequence[Any]) -> tuple:
    # the contract plus every value a trace reads except addresses, and the
    # groups of arguments whose extents chain into one (Tape.argument_pairs)
    values: list[Any] = []
    extents = []
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            strided = a.layout == torch.strided
            values.append((tuple(a.shape), a.stride() if strided else None))
            values.append(a.storage_offset() if strided else None)
            if strided and a.numel():
                first = a.const_data_ptr()  # type: ignore[attr-defined]
                span = sum((n - 1) * s for n, s in zip(a.shape, a.stride()))
                extents.append((first, first + (span + 1) * a.element_size() - 1, i))
        elif type(a) is int:
            values.append(a)
    groups: list[list[int]] = []
    end = -1
    for first, last, i in sorted(extents):
        if first > end:
            groups.append([])
        groups[-1].append(i)
        end = max(end, last)
    return (contract, tuple(values), tuple(tuple(g) for g in groups if len(g) > 1))


def _piece(site: KeyedSite, binding: OpaqueBinding) -> bool:
    """Whether the site runs the binding on nodes of their own, not its nodes
    at other launch attributes: a node or a programmatic edge more or less, a
    memset's shape, or other attributes on another kernel (an attribute is set
    on the traced node, and checked against its kernel)."""
    from cuda.bindings import driver

    pdl = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    own, other = site.topology, binding.topology
    return len(own) != len(other) or any(
        a != b
        and (
            a[0] != "kernel"
            or b[0] != "kernel"
            or dict(a[1]).get(pdl) != dict(b[1]).get(pdl)
            or site.nodes[k].function != binding.nodes[k].function
        )
        for k, (a, b) in enumerate(zip(own, other))
    )


# size_classes modes: "planned_no_reuse" keeps size classes (no _holes / _offline),
# "planned_no_split" one arena per run step (no split around keyed-site scratch),
# "planned_scratch" each keyed site's scratch buffer a slot of its traced bytes in
# the arena (a run step allocates once; a key whose binding needs more is declined)
_PLANNED = ("planned", "planned_no_reuse", "planned_no_split", "packed", "planned_scratch")


def _capturing(device: torch.device) -> bool:
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _sym_minmax(builtin: Callable[..., Any], sym: Callable[[Any, Any], Any]) -> Callable[..., Any]:
    # builtin max or min, as torch.sym_max or sym_min where every operand is an
    # int and one is symbolic: no comparison, so no guard (an IR trace has no
    # float max, and a float operand would change the result's type)
    @functools.wraps(builtin)
    def call(*args: Any, **kwargs: Any) -> Any:
        if kwargs or not args:
            return builtin(*args, **kwargs)
        values = args
        if len(args) == 1:
            if not isinstance(args[0], Iterable):
                return builtin(*args)
            values = tuple(args[0])
            args = (values,)
        if any(isinstance(v, torch.SymInt) for v in values) and all(isinstance(v, (int, torch.SymInt)) for v in values):
            return functools.reduce(sym, values)
        return builtin(*args)

    return call


_builtin_max, _builtin_min = builtins.max, builtins.min


def _hold_minmax() -> None:
    builtins.max = _sym_minmax(_builtin_max, torch.sym_max)
    builtins.min = _sym_minmax(_builtin_min, torch.sym_min)


def _release_minmax() -> None:
    builtins.max, builtins.min = _builtin_max, _builtin_min


# held while any trace of a HostTraceReplay with trace_builtin_minmax runs
_builtin_minmax = ProcessHold(_hold_minmax, _release_minmax)


class _Disagreement(AssertionError):
    """A replay's own error after its first effect: the variant is dropped."""

    def __init__(self, msg: str, op: Any = None) -> None:
        super().__init__(f"host_trace: {msg}")
        self.op = op  # the eager call's operator whose metadata was wrong


class _Deferred(Exception):
    """A call's key went to the background learner: the call runs eagerly."""


class _StaleBinding(AssertionError):
    """A built variant's keyed site whose row at its key holds another key's binding."""


class _Background:
    """One worker thread that learns the keys HostTraceReplay(background=True)
    submits, each on a stream of its own per device, a key once at a time."""

    def __init__(self) -> None:
        self.jobs: queue.SimpleQueue = queue.SimpleQueue()
        self.pending: set[OpaqueKey] = set()
        self.lock = threading.Lock()
        # held while the worker learns: synchronizing the device is invalid
        # while any stream captures, so torch.cuda.synchronize waits for it
        self.learning = threading.RLock()
        self.thread: threading.Thread | None = None
        self.streams: dict[int, torch.cuda.Stream] = {}

    def submit(self, replay: HostTraceReplay, site: KeyedSite, key: OpaqueKey) -> None:
        with self.lock:
            if key in self.pending:
                return
            self.pending.add(key)
            if self.thread is None:
                sync = torch.cuda.synchronize

                def synchronize(device: Any = None) -> None:
                    with self.learning:
                        sync(device)

                torch.cuda.synchronize = synchronize
                self.thread = threading.Thread(target=self._run, name="host_trace_learn", daemon=True)
                self.thread.start()
        self.jobs.put((replay, site, key))

    def idle(self) -> bool:
        with self.lock:
            return not self.pending

    def _run(self) -> None:
        while True:
            replay, site, key = self.jobs.get()
            try:
                stream = self.streams.get(key.device)
                if stream is None:
                    stream = self.streams[key.device] = new_private_stream(key.device)
                pool = torch.cuda.MemPool() if replay.learn_pool else None
                with (
                    self.learning,
                    torch.cuda.device(key.device),
                    torch.cuda.stream(stream),
                    torch.cuda.use_mem_pool(pool, key.device) if pool is not None else contextlib.nullcontext(),
                ):
                    binding = replay._learn(site.op, site.provider, site.call, key, restore_rng=False)
                    stream.synchronize()
                del pool
                if binding is None and site.provider.refusal(key) is None:
                    # no binding and no verdict (a transient refusal): not silent
                    replay.background_failed.add(key)
                    replay.background_errors.append(f"{key.op} at sizes {key.sizes}: {type(site.provider).__name__} bound nothing and refused nothing")
            except Exception as e:
                replay.background_failed.add(key)
                replay.background_errors.append(f"{key.op} at sizes {key.sizes}: {e!r}")
            finally:
                with self.lock:
                    self.pending.discard(key)


_BACKGROUND = _Background()


@dataclass
class Handback:
    """A boxed call's arguments, handed back to call again (HostTraceReplay's
    `handback`)"""

    args: tuple


@dataclass
class _Variant:
    captured: CapturedTape
    memory: MemoryPlan
    native: torch._C._HostTraceVariant
    # the opaque calls' keys that bind or are refused yet stay opaque here: a
    # call traced again at them, or this variant's trace declined to record them
    tried: set[OpaqueKey] = field(default_factory=set)
    # per keyed site, a binding of each of its arms 1, 2, ...: its nodes at
    # other launch attributes, or a piece, nodes of their own in their place
    arms: dict[int, list[OpaqueBinding]] = field(default_factory=dict)
    # the launches of the traces folded into it, which keep their functions loaded
    folded: list[tuple[LoweredLaunch | LoweredMemset, ...]] = field(default_factory=list)
    # its tape before its opaque calls were bound (oob_learn's, a relower's) and
    # each keyed site's call in it, by the site's id, for _eager_sites
    unbound: tuple[Tape, dict[int, OpaqueCall]] | None = None
    # the variants built from it with refused keys (native site index, key) as eager steps
    eager_sites: dict[frozenset[tuple[int, OpaqueKey]], _Variant | None] = field(default_factory=dict)
    # per keyed site, a class shared by the sites whose keys are made alike
    # (_site_classes): sites of a class at equal key values have one OpaqueKey
    site_classes: list[int] | None = None

    @property
    def tape(self) -> Tape:
        return self.captured.lowered.tape

    @property
    def learns(self) -> bool:
        # an opaque call runs eagerly: its commits go through _call_slow
        return bool(self.captured.lowered.opaque)


if not hasattr(torch._C, "_HostTraceEntry"):

    class _HostTraceEntry:
        """A build without native replay: every call is a slow call, and every
        trace declines in _build."""

        replays = 0
        static_hits = 0
        slow_calls = 0

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            self.slow_calls += 1
            return self._call_slow(args, kwargs, False)  # pyrefly: ignore [missing-attribute]

        def call_boxed(self, inputs: list[Any]) -> Any:
            args = tuple(inputs)
            inputs.clear()
            return self(*args)

        def _native_init(self, *args: Any) -> None:
            pass

        def _native_key(self, args: tuple) -> list[int] | None:
            return None

        def _native_add(self, key: list[int], native: Any) -> None:
            pass

        def _native_remove(self, native: Any) -> None:
            pass

    torch._C.__dict__["_HostTraceEntry"] = _HostTraceEntry


class HostTraceReplay(torch._C._HostTraceEntry):
    """fn served by host-trace replay; call it as fn. With `trusted`, the
    caller vouches for every call's arguments (TrustedInputs): a call is
    checked only against its variants' dispatch guards. `memory` is how a
    replay allocates (_host_trace_memory): "auto" (the default), "eager",
    "run_buffer", "held" or one of _PLANNED; `splits`, where "auto" and "eager" split a run (SPLITS, split_runs).
    `opaque` is the keyed calls' providers (_host_trace_opaque); by default a
    HarvestProvider of its own, for cuBLAS's mm, addmm and bmm; () keys none.
    `freed_arguments` are the positions whose tensor a boxed call's caller
    holds no other reference to, which split_runs may free mid-tape. With
    `handback`, a slow call that traces a variant or adds to one's tables
    returns Handback instead of running the variant, which Python's reference
    to the arguments would hold to the end; the caller calls again, with
    call_boxed. `static_shapes` are the tensor positions whose layout is static (trace).
    Unless `trace_builtin_minmax` is False, a trace runs Python's max and min
    over ints, one of them symbolic, as torch.sym_max and sym_min, which guard
    on no comparison (as Dynamo's rewrite of them does).
    With `learn_pool`, the keys learned at a miss (_learn) allocate from a
    pool dropped after them, not the caching allocator's: no reserved growth
    from their operands, for a cudaMalloc and cudaFree per miss. With
    `background`, a call whose key would be learned runs eagerly instead and
    the key is learned on a worker thread (on a stream of its own); a later
    call at the key replays. A worker learn that binds nothing without a
    refusal is in `background_errors`, and the key's next call learns it as
    without `background`.
    A variant that does not learn learns a keyed site's missing key out of
    band (_learn); a learning variant (one with opaque calls) traces the call
    again, as it does where its selectors select nothing.
    The first `static_prefix` arguments are expected to be the same tensors
    at every call (a model's parameters): a call whose leading arguments are
    the last hit's objects with the same metadata (data_ptr, sizes, strides,
    dtype, ...) keys and evaluates only the other arguments, taking the
    leading ones' rows from that hit (static_hits counts these); any change
    takes the full path.

    With `fullgraph`, as torch.compile's, no call falls back and no tape
    breaks: where a call would run eagerly (a decline, an argument the
    contract cannot key, a misaligned allocation, an outer capture) or a tape
    would hold an eager step, it raises EagerFallback naming the op and why.
    The entry's first call, a trace's warm-up (the call itself), a retry
    and a host step (it reads no device memory and runs before the first
    graph) are not fallbacks. An opaque call whose key is not bound yet is
    not an eager step: its variant learns (counted in `learners` and
    `learner_calls`). With `forbid_learners` a trace learns its opaque calls'
    keys at once (_learn, as the relower would at the next call), so no
    variant learns, and one whose key neither binds nor learns (refused,
    graphsafe RNG, integer inputs) raises EagerFallback. A call that runs
    eagerly while its key learns in the background is not a fallback either:
    it is counted in `deferred` (and `eager`), a failed learn in
    `background_errors`. With `forbid_learners` such a key learns at the call
    instead, as without `background`.

    fn must be a pure function of its arguments. Torch's global settings
    (grad and inference mode, autocast, the TF32, reduced-precision, cuDNN
    and SDPA flags) are assumed unchanged between a trace and its replays
    unless `check_global_state`, which adds them to the argument contract;
    a torch.compile caller has these checks from Dynamo's guards. As with
    torch.cuda.graph, other Python state fn reads (a module's training flag
    or attributes, a global) is baked into its trace. The entry's first call
    runs eagerly and the second traces (_miss); fn's Python side effects run
    at each of a trace's runs and never at a replay, which is unavoidable; and
    it stores no tensor it makes (with `check_escapes` a trace that does declines).

    The call is the base's: a hit runs in C++, anything else is _call_slow."""

    # inside a trace a call runs fn inline (the base's, and _HostTraceEntry's)
    @property
    def fn(self) -> Callable[..., Any]:
        tr = current_trace()
        if tr is not None and tr.library_body is not None:
            raise tr.reentered()
        return self._fn

    @fn.setter
    def fn(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        trusted: TrustedInputs | None = None,
        opaque: Sequence[OpaqueProvider] | None = None,
        memory: str = "auto",
        freed_arguments: Collection[int] = (),
        static_shapes: Collection[int] = (),
        static_prefix: int = 0,
        check_global_state: bool = False,
        check_escapes: bool = False,
        handback: bool = False,
        splits: str = "peak",
        trace_builtin_minmax: bool = True,
        learn_pool: bool = False,
        background: bool = False,
        fullgraph: bool = False,
        forbid_learners: bool = False,
    ) -> None:
        if memory not in ("auto", "eager", "run_buffer", "held", *_PLANNED):
            raise ValueError(f"host_trace: memory {memory!r} is not 'auto', 'eager', 'run_buffer', 'held' or one of {_PLANNED}")
        if splits not in SPLITS:
            raise ValueError(f"host_trace: splits {splits!r} is not one of {SPLITS}")
        self.fn = fn
        self.memory = memory
        self.freed_arguments = frozenset(freed_arguments)
        self.handback = handback
        self.splits = splits
        self.static_shapes = frozenset(static_shapes)
        self.static_prefix = static_prefix
        self.trusted = trusted
        self.opaque = (HarvestProvider(),) if opaque is None else opaque
        self.check_global_state = check_global_state
        self.check_escapes = check_escapes
        self.trace_builtin_minmax = trace_builtin_minmax
        self.learn_pool = learn_pool
        self.background = background
        self.fullgraph = fullgraph
        self.forbid_learners = forbid_learners
        try:
            self._signature: inspect.Signature | None = inspect.signature(fn)
        except (TypeError, ValueError):
            self._signature = None
        self._families: dict[tuple, list[_Variant]] = {}
        # per contract, the redos' dispatches a redo of the same reads takes where their guards hold
        self._dispatches: dict[tuple, dict] = {}
        self._declined: set[tuple] = set()
        # per contract and overlapping argument groups, the guards of each trace
        # that found nothing to capture (keep_eager_regions): (program, row)
        self._eager_regions: dict[tuple, list[tuple[torch._C._HostTraceProgram, int]]] = {}
        # the classes whose warm-up's operator calls were not their trace's once
        self._witness_retried: set[tuple] = set()
        # the functions whose CuTe compile raised under a trace: retried once each
        self._declined_compiles: set[Any] = set()
        # the operators whose output disagreed with the trace's metadata
        self._meta_disagrees: set[Any] = set()
        # every fallback's reason, in order, each once
        self._reasons: dict[str, None] = {}
        # why each Triton launch that runs eagerly in a replay does, each once
        self.triton_fallbacks: dict[str, None] = {}
        # the entry's first call runs eagerly (_miss)
        self._called = False
        self.traces = 0
        # the learning variants' tapes rebuilt with their keys that bind bound, for no trace
        self.relowers = 0
        # the learning variants built, and the calls they served (replays
        # whose opaque calls ran eagerly)
        self.learners = 0
        self.learner_calls = 0
        self.replays = 0
        self.eager = 0
        # the traces that found nothing to capture (Declined.uncaptured)
        self.uncaptured = 0
        # the calls an eager region's guards held for, run eagerly with no trace
        self.eager_region_calls = 0
        # the traces whose decline holds for their class: not a retry, not the
        # call's own error, not uncaptured
        self.structural = 0
        # the guards of every variant's tape, summed
        self.guards = 0
        # the traces folded into a variant, and why each other one did not
        # fold into a variant it was a candidate for
        self.folds = 0
        self.fold_refusals: dict[str, int] = {}
        # the selector misses served by their ops dispatched again, no trace
        # (_host_trace_redispatch), their seconds, and why each other did not
        self.redispatches = 0
        self.redispatch_s = 0.0
        self.redispatch_refusals: dict[str, int] = {}
        # why each trace after the first traced: (class, op, guard, the user's line, refusal) -> count. Class "meta": a
        # graph guard an op recorded (its outputs' metadata); "dispatch": an op's own guard failed and its
        # redispatch refused (refusal); "graph": a guard no op recorded; "contract": no variant of the
        # call's argument contract; "other": a variant whose guards hold missed (a refused key, say)
        self.retrace_causes: dict[tuple[str, str | None, str, str | None, str | None], int] = {}
        # per variant (its native's id), its last redispatch refusal: (the op that refused (Tape.ops), why)
        self._last_refusal: dict[int, tuple[int | None, str]] = {}
        # the keys harvested at a keyed site's miss, no trace or relower
        # (_learn), and their seconds
        self.learned = 0
        self.learn_s = 0.0
        # the traces whose opaque calls bound out of band (oob_learn)
        self.oob_binds = 0
        # the variants built with refused keys as eager steps, no trace
        # (_eager_sites), their seconds, and why each other did not build
        self.eager_sites = 0
        self.eager_sites_s = 0.0
        self.eager_sites_refusals: dict[str, int] = {}
        # the calls run eagerly while their keys learn in the background, the
        # background learns' failures, and their keys (learned in the
        # foreground at their next call)
        self.deferred = 0
        self.background_errors: list[str] = []
        self.background_failed: set[OpaqueKey] = set()
        # one call at a time: a call patches the execs it replays (a native
        # call holds the base's lock too)
        self._lock = threading.RLock()
        if torch.cuda._host_trace.cpp_entry:
            # the parameters a keyword can pass by position, as _positional binds them
            params = self._signature.parameters.values() if self._signature is not None else ()
            positional = itertools.takewhile(lambda p: p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD), params)
            names = tuple(p.name if p.kind is p.POSITIONAL_OR_KEYWORD else None for p in positional)
            self._native_init(_active, _Disagreement, trusted is not None, check_global_state, static_prefix, names)

    @property
    def declines(self) -> list[str]:
        return list(self._reasons)

    @property
    def variants(self) -> list[_Variant]:
        return [v for family in self._families.values() for v in family]

    def _call_slow(
        self, args: tuple, kwargs: dict[str, Any] | None, searched: bool
    ) -> Any:
        """The call when the base's native path does not serve it; `searched`:
        the native path evaluated the call's family, none holds it and no
        keyed site missed, which leaves the variants that learn."""
        if current_trace() is not None:
            return self.fn(*args, **(kwargs or {}))
        if kwargs:
            bound = self._positional(args, kwargs)
            if bound is None:
                why = "keyword arguments that do not bind to positional parameters"
                return self._eager(args, kwargs, why)
            args = bound
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            return self._eager(args, {}, "an outer capture: fn is captured as eager")
        # trusted: Dynamo guarded the arguments' kinds and the global state
        contract = () if self.trusted is not None else argument_contract(args, self.check_global_state)
        try:
            family = self._families.get(contract)
        except TypeError:
            return self._eager(args, {}, "an argument without a hash")
        # the collector waits out a miss's trace, lowering and instantiations
        with self._lock, _gc_hold:
            try:
                return self._call_held(family, contract, args, searched)
            except _Deferred:
                self.deferred += 1
        return self._eager(args, {}, "a key learning in the background", fallback=False)

    def _call_held(self, family: list[_Variant] | None, contract: tuple, args: tuple, searched: bool) -> Any:
        # a variant that does not learn holds the call once its keys bind
        pending = False
        candidates: list[tuple[_Variant, list[int], list[int]]] = []
        for variant in tuple(family or ()):
            if searched and not variant.learns:
                continue
            held = len(candidates)
            refused: list[tuple[int, OpaqueKey]] = []
            filled = self._fill(variant, args, candidates, refused=refused)
            if filled is None and refused and refused_eager_steps and frozenset(refused) not in variant.eager_sites:
                twin = self._eager_sites(contract, variant, args, refused)
                if twin is not None and self._fill(twin, args, candidates) is not None:
                    outcome, result = self._call_native(twin, args)
                    if outcome == MISS:
                        raise AssertionError("host_trace: a call misses the variant built with its refused keys as eager steps")
                    return self._served(twin.native, outcome, result, args)
            if filled is None:
                if len(candidates) == held:
                    pending |= not variant.learns and variant.native.evaluate(args) is not None
                continue
            values, added = filled
            if variant.learns:
                # an opaque call whose key binds or is refused now is traced
                # again, as its launches or as a plain eager step, unless a
                # pending variant's keyed site takes the key
                lowered = variant.captured.lowered
                final = [
                    k
                    for o in lowered.opaque.values()
                    if (k := o.key(values)) not in variant.tried
                    and (o.provider.refusal(k) is not None or (not pending and o.provider.bind(k) is not None))
                ]
                if final:
                    variant.tried.update(final)
                    upgraded = self._relower(family, args, variant, values)
                    if upgraded is None:
                        try:
                            return self._miss(contract, args, candidates, before=variant)
                        except EagerFallback:
                            variant.tried.difference_update(final)  # the next call raises too
                            raise
                    if self._fill(upgraded, args, candidates) is not None:
                        variant = upgraded
            elif self.handback and added:
                return Handback(args)
            outcome, result = self._call_native(variant, args)
            if outcome != MISS:
                return self._served(variant.native, outcome, result, args)
        # no variant holds the call: one whose graph guards do gains its
        # failing selectors' entries from their ops dispatched again
        for j, (variant, values, unselected) in enumerate(candidates):
            if self._redispatch(variant, args, values, unselected, self._dispatches.setdefault(contract, {})) is not None:
                continue
            del candidates[j]
            if self._fill(variant, args, candidates) is not None:
                if self.handback:
                    return Handback(args)
                outcome, result = self._call_native(variant, args)
                if outcome != MISS:
                    return self._served(variant.native, outcome, result, args)
            break
        # a refused redispatch traces the graph again. Propagating the op's new metadata through its uses (as
        # respec did) could serve it where all kernel and tensor access goes through custom ops; engine code does not
        return self._miss(contract, args, candidates)

    def prepare(self, *args: Any) -> bool:
        """Learns what a call at `args` needs (its keyed sites' bindings, rows
        and forms) without running it: whether a variant that does not learn
        then holds the call. The arguments' contents are not read."""
        contract = () if self.trusted is not None else argument_contract(args, self.check_global_state)
        with self._lock, _gc_hold:
            return any(not v.learns and self._fill(v, args, [], background=False) is not None for v in self._families.get(contract, ()))

    def _relower(self, family: list[_Variant], args: tuple, variant: _Variant, values: list[int]) -> _Variant | None:
        """The learning variant's tape with its opaque calls bound (bind_opaque)
        at their keys at its traced call, each learned (_learn) unless it binds,
        or failing that only those whose keys bind at the call too, built and put
        before it in its family: the variant a trace at the call makes. None
        where the call must trace again: a key it refuses (a trace guards its
        values), or one that binds whose key at the traced call does not, or
        whose binding does not fit the site that key's binding records."""
        lowered = variant.captured.lowered
        if any(o.provider.refusal(o.key(values)) is not None for o in lowered.opaque.values()):
            return None
        calls = {id(lowered.steps[i].call): o for i, o in lowered.opaque.items()}
        wanted = {c: b for c, o in calls.items() if (b := o.provider.bind(o.key(values))) is not None}
        pool = torch.cuda.MemPool() if self.learn_pool else None
        with torch.cuda.use_mem_pool(pool, variant.tape.device) if pool is not None else contextlib.nullcontext():
            for ids, how in ((set(calls), self._learn), (set(wanted), None)):
                bound = bind_opaque(variant.tape, ids, how, keep_guards=oob_learn)
                if bound is not None and all(c in bound[1] and bound[1][c].fits(b) for c, b in wanted.items()):
                    break
            else:
                return None
        del pool
        try:
            upgraded = self._build(bound[0])
        except Declined:
            return None
        self.relowers += 1
        self.learners += upgraded.learns
        upgraded.unbound = _unbound(variant.tape, bound[1])
        family.insert(family.index(variant), upgraded)
        self._register(family, args)
        self.guards += upgraded.tape.guard_count
        return upgraded

    def _fill(
        self,
        variant: _Variant,
        args: tuple,
        candidates: list[tuple[_Variant, list[int], list[int]]],
        background: bool | None = None,
        refused: list[tuple[int, OpaqueKey]] | None = None,
    ) -> tuple[list[int], bool] | None:
        """The call's rows if the variant's program holds it, after adding
        each keyed site's missing key to its table and each segment's missing
        form, and whether it added any; None if a key does not bind and does
        not learn out of band (a learning variant's does not: its eager run
        harvests it), or if a selector selects nothing: then a variant that
        does not learn is a candidate to dispatch those selectors' ops again
        for, or to fold the call's trace into, (variant, rows, selectors) in
        `candidates`. A key that would be learned with `background` (default self.background) is submitted to
        the worker instead: _Deferred. Each site whose key the provider
        refuses goes in `refused`, (index, key)."""
        if background is None:
            background = self.background and not self.forbid_learners
        evaluated = variant.native.evaluate(args)
        if evaluated is None:
            return None
        values, missing, unformed, unselected = evaluated
        if unselected:
            if not variant.learns:
                candidates.append((variant, values, unselected))
            return None
        sites = variant.captured.lowered.sites
        if variant.site_classes is None:
            variant.site_classes = _site_classes(sites)
        bindings = []
        pool = None
        deferred = False
        # each (class, key values) built and bound once: the same GEMM in every layer is one key
        keys: dict[tuple, tuple[OpaqueKey, OpaqueBinding | None]] = {}
        for i, key in missing:
            site = sites[i].site
            known = keys.get(memo := (variant.site_classes[i], tuple(key)))
            opaque_key, binding = known if known is not None else (k := sites[i].key(key), site.provider.bind(k))
            # an RNG site on a graph's generator (graphsafe RNG) learns in its learning variant, as bind_opaque's do
            graphsafe = site.rng and any(getattr(n, "generator", None) is not None for n in site.nodes)
            if binding is None and not variant.learns and not graphsafe and site.call is not None:
                # an RNG key learns here: the worker's learn would race this thread's draws
                if background and not site.rng and site.provider.refusal(opaque_key) is None and opaque_key not in self.background_failed:
                    _BACKGROUND.submit(self, site, opaque_key)
                    deferred = True
                    continue
                if pool is None and self.learn_pool:
                    pool = torch.cuda.MemPool()
                with torch.cuda.use_mem_pool(pool, opaque_key.device) if pool is not None else contextlib.nullcontext():
                    binding = self._learn(site.op, site.provider, site.call, opaque_key)
            keys[memo] = (opaque_key, binding)
            bindings.append((i, key, binding))
        del pool
        if deferred:
            raise _Deferred
        # past a key that does not bind the others still learn: a retrace then sees every refusal
        if any(b is None for _, _, b in bindings):
            # a refused key is a row the variant refuses: its calls miss it in C++, not handed back here
            for i, key, binding in bindings:
                if binding is None and not variant.learns and sites[i].site.provider.refusal(sites[i].key(key)) is not None:
                    if not self._native_mutate(variant, "a row", variant.native.add_row, i, key, None):
                        return None
                    if refused is not None:
                        refused.append((i, sites[i].key(key)))
            return None
        for i, key, binding in bindings:
            fits = sites[i].site.fits(binding)
            if fits and self.memory == "planned_scratch":
                fits = all(n <= sites[i].site.scratch[j][1] for j, n in enumerate(binding.scratch))
            arm = self._arm(variant, i, binding) if fits else None
            if arm is None:
                row = (i, key, None)
            else:
                nodes, scratch = binding_row(sites[i], binding)
                row = (i, key, nodes, arm, _piece(sites[i].site, binding), scratch)
            if not self._native_mutate(variant, "a row", variant.native.add_row, *row):
                return None
        if bindings:
            evaluated = variant.native.evaluate(args)
            unformed = [] if evaluated is None else evaluated[2]
        for g, arms in unformed:
            if not self._native_mutate(variant, "a form", variant.native.add_form, g, arms, *self._form(variant, g, arms)):
                return None
        return values, bool(bindings or unformed)

    def _learn(self, op: OpOverload, provider: OpaqueProvider, call: tuple[Any, frozenset[int]], key: OpaqueKey, restore_rng: bool = True) -> OpaqueBinding | None:
        """The key's binding, harvested now from one eager run of `op` on
        buffers at the key's metadata, `call` its pytree spec and its tensor
        leaves' positions: no trace, no relower. Floating inputs hold uniform
        values, the rest zeros (in range as indices), operands on one storage
        (OpaqueKey.alias) views of one buffer, and the inputs the provider
        lends are its own: with one lent, the op does not run (it may write
        it). None for a key the provider refuses. An RNG op's run draws from
        the default generator, whose offset is put back (with restore_rng; the
        worker's learns draw none): eager's next call draws where it would
        have."""
        if provider.refusal(key) is not None:
            return None
        spec, positions = call
        inputs = len(positions)
        metadata = list(zip(key.dtypes, key.sizes, key.strides, key.align))
        lent = provider.operands(key, inputs)
        if lent is None or (lent and len(metadata) > inputs):
            return None
        if not all(align % dtype.itemsize == 0 for dtype, _, _, align in metadata[:inputs]):
            return None
        start = time.perf_counter()
        device = torch.device("cuda", key.device)
        generator = torch.Generator(device).manual_seed(0)

        def nbytes(i: int) -> int:
            dtype, sizes, strides, _ = metadata[i]
            return span_bytes(sizes, strides, dtype.itemsize)

        # each alias group in one buffer: its lead at the key's address % 256
        # (the allocator's blocks are 512-byte aligned), the rest at their distances
        lead = {i: (j, d) for i, j, d in key.alias}
        reach = [nbytes(i) for i in range(inputs)]
        for i, (j, d) in lead.items():
            reach[j] = max(reach[j], d + reach[i])
        buffers = {j: torch.empty(metadata[j][3] + reach[j], dtype=torch.uint8, device=device) for j in range(inputs) if j not in lead and j not in lent}
        # a drawing op's floating inputs are probabilities, rates or scales: in range, as the harvest's refill
        bounds = (0.25, 0.75) if takes_generator(op) else (-1, 1)
        tensors = []
        for i, (dtype, sizes, strides, align) in enumerate(metadata[:inputs]):
            if i in lent:
                tensors.append(lent[i])
                continue
            j, d = lead.get(i, (i, 0))
            at = metadata[j][3] + d
            flat = buffers[j][at : at + -(-nbytes(i) // dtype.itemsize) * dtype.itemsize].view(dtype)
            if dtype.is_floating_point:
                fill_uniform(flat, *bounds, generator)
            else:
                flat.zero_()
            tensors.append(flat.as_strided(sizes, strides))
        given, scalars = iter(tensors), iter(key.scalars)
        args, kwargs = pytree.tree_unflatten([next(given) if j in positions else next(scalars) for j in range(spec.num_leaves)], spec)
        fresh = []
        if not lent:
            gen = torch.cuda.default_generators[key.device]
            offset = gen.get_offset() if restore_rng else None
            try:
                with library_state_as(key.state), torch._C._AutoDispatchBelowADInplaceOrView():
                    out = op(*args, **kwargs)
            finally:
                if offset is not None:
                    gen.set_offset(offset)
            fresh = [o for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor) and not any(o is t for t in tensors)]
            fresh = seed_offset_on_device(op, fresh, device)
        want = [(dtype, sizes, strides, 0) for dtype, sizes, strides, _ in metadata[inputs:]]
        if [(o.dtype, tuple(o.shape), o.stride(), o.storage_offset()) for o in fresh] != want:
            return None
        binding = provider.learn(key, args, kwargs, tensors + fresh)
        self.learned += binding is not None
        self.learn_s += time.perf_counter() - start
        return binding

    def _arm(self, variant: _Variant, i: int, binding: OpaqueBinding) -> int:
        """Site i's arm for the binding: 0 its nodes' topology, k > 0 its k-th
        other."""
        topology = binding.topology
        if topology == variant.captured.lowered.sites[i].site.topology:
            return 0
        arms = variant.arms.setdefault(i, [])
        k = next((k for k, b in enumerate(arms) if b.topology == topology), None)
        if k is None:
            arms.append(binding)
            k = len(arms) - 1
        return k + 1

    def _form(self, variant: _Variant, g: int, arms: list[int]) -> tuple[int, int, list]:
        """An exec of a clone of segment g's graph with its sites' kernel
        nodes at their arms' launch attributes, a piece arm's site's nodes
        replaced by the arm's; the clone, and each piece's (site, nodes)."""
        plain = plain_attributes(torch.cuda.current_device())
        segments, launches = variant.captured.segments, variant.captured.launches
        first = sum(len(s.launches) for s in segments[:g])
        stop = first + len(segments[g].launches)
        sites = [(i, s) for i, s in enumerate(variant.captured.lowered.sites) if first <= s.nodes[0] < stop]
        changes, pieces = [], []
        for (i, site), arm in zip(sites, arms, strict=True):
            if arm == 0:
                continue
            binding = variant.arms[i][arm - 1]
            if _piece(site.site, binding):
                pieces.append((i, [launches[n].node for n in site.nodes], binding))
                continue
            for n, a, b, kernel in zip(site.nodes, site.site.topology, binding.topology, binding.nodes):
                if a != b:
                    own, other = dict(a[1]), dict(b[1])
                    keys = own.keys() | other.keys()
                    changes.append((launches[n].node, {k: other.get(k, plain[k]) for k in keys if own.get(k) != other.get(k)}, kernel.grid))
        exec_, clone, placed = instantiate_form(segments[g].graph.raw_cuda_graph(), changes, [p[1:] for p in pieces])
        return exec_, clone, [(i, nodes) for (i, _, _), nodes in zip(pieces, placed)]

    def _call_native(self, variant: _Variant, args: tuple) -> tuple[int, Any]:
        try:
            outcome, result = variant.native.call(args)
        except _Disagreement as e:
            self._disagreed(variant.native, e)
            raise
        self.learner_calls += variant.learns and outcome == HIT
        return outcome, result

    def _served(
        self, native: torch._C._HostTraceVariant, outcome: int, result: Any, args: tuple
    ) -> Any:
        if outcome == HIT:
            self.replays += 1
            return result
        return self._unreplayed(native, outcome, args)

    def _unreplayed(
        self, native: torch._C._HostTraceVariant, outcome: int, args: tuple
    ) -> Any:
        # a validated call the native commit ran no step of
        if outcome == MISALIGNED:
            why = f"an allocation that is not {_ALLOC_ALIGNMENT}-byte aligned"
            return self._eager(args, {}, why)
        if outcome != CAPTURE:
            raise AssertionError(f"host_trace: a native outcome {outcome}")
        device = next(v.tape.device for v in self.variants if v.native is native)
        return self._eager(args, {}, f"a capture on {device}")

    def _native_mutate(self, variant: _Variant, what: str, add: Callable[..., Any], *args: Any) -> bool:
        """add(*args), a native variant's row, form, entry or program; False
        where the native variant rejects it: the lowering's bug, which the
        suites raise (raise_unexpected) and a user's call notes, dropping the
        variant (its native state is partial), so the call goes on as a miss."""
        try:
            add(*args)
            return True
        except (TypeError, ValueError, IndexError, RuntimeError) as e:
            if torch.cuda._host_trace.raise_unexpected:
                raise AssertionError(f"host_trace: the native variant rejects {what}: {e}") from e
            self._note(f"host_trace: the native variant rejects {what} ({e}); the variant is dropped")
            for family in self._families.values():
                family[:] = [v for v in family if v is not variant]
            self._native_remove(variant.native)
            return False

    def _disagreed(self, native: torch._C._HostTraceVariant, e: _Disagreement) -> None:
        # a call from inside an eager step may have dropped it
        for family in self._families.values():
            family[:] = [v for v in family if v.native is not native]
        self._native_remove(native)
        if e.op is not None:
            self._meta_disagrees.add(e.op)

    def _register(self, family: list[_Variant], args: tuple) -> None:
        # the base serves the family's variants that do not learn, in order
        key = self._native_key(args)
        if key is None:
            return
        for v in family:
            self._native_remove(v.native)
        for v in family:
            if not v.learns:
                self._native_add(key, v.native)

    def _positional(self, args: tuple, kwargs: dict[str, Any]) -> tuple | None:
        if self._signature is None:
            return None
        try:
            bound = self._signature.bind(*args, **kwargs)
        except TypeError:
            return None
        return None if bound.kwargs else bound.args

    def _note(self, reason: str, warn: bool = False) -> None:
        if reason not in self._reasons:
            self._reasons[reason] = None
            (log.warning if warn else log.debug)("%s", reason)

    def _eager(
        self, args: Sequence[Any], kwargs: dict[str, Any], why: str | None = None, fallback: bool = True
    ) -> Any:
        if self.fullgraph and fallback:
            raise EagerFallback(f"host_trace: {why or 'a call of a class that declined'}; fullgraph=True does not run the call eagerly")
        if why is not None:
            self._note(f"host_trace: {why}; the call runs eagerly")
        self.eager += 1
        return self.fn(*args, **kwargs)

    @torch.cuda._host_trace.on_a_fresh_stack_chunk(32800)
    def _miss(
        self,
        contract: tuple,
        args: Sequence[Any],
        candidates: Sequence[tuple[_Variant, list[int], list[int]]] = (),
        before: _Variant | None = None,
    ) -> Any:
        if any(v.native.overlaps(tuple(args)) for v in self._families.get(contract, ())):
            # an assertion, not a dispatch: eager raises its error or runs it
            return self._eager(args, {}, "arguments overlap that a variant's steps take as disjoint")
        exact = _exact_class(contract, args)
        # under trust a decline is structural: one is the graph's
        structural = contract if self.trusted is not None else exact
        if contract in self._declined or exact in self._declined:
            return self._eager(args, {})
        device = next((a.device for a in args if isinstance(a, torch.Tensor)), None)
        if device is not None and device.type == "cuda" and _capturing(device):
            # not a decline of the class: the capture ends
            return self._eager(args, {}, f"a capture on {device}")
        for compiled, row in self._eager_regions.get((contract, exact[2]), ()):
            result = compiled.evaluate_inputs(tuple(args))
            if result is not None and result[0] == Status.SUCCESS and result[1][row] == 1:
                self.eager_region_calls += 1
                return self._eager(args, {})
        if not self._called:
            # as cudagraph trees' warm-up, an entry's first call runs eagerly:
            # first-use work (an autotuner's benchmark, a lazy initialization)
            # happens outside every trace, and fn runs once per call
            self._called = True
            return self._eager(args, {}, fallback=False)
        # a trace warms up: the warm-up is the call, its operator calls the
        # trace's witness, and at the entry's first lazy initialization happens
        # outside the trace. Under trust Dynamo's guards stand for the witness.
        first = self.traces == 0
        if not first:
            self._retrace_cause(contract, args)
        warm_up = first or self.trusted is None
        ran, result, tape, folded = False, None, None, None
        # the ops whose segments failed to capture, run eagerly by a trace again
        eager_ops: dict[int, str] = {}
        failed: SegmentFailed | None = None
        try:
            while True:
                self.traces += 1
                unbound = None
                with _builtin_minmax if self.trace_builtin_minmax else contextlib.nullcontext():
                    tape = trace(
                        self.fn,
                        tuple(args),
                        warm_up=warm_up and not ran,
                        trusted=self.trusted,
                        opaque=self.opaque,
                        static_shapes=self.static_shapes,
                        check_escapes=self.check_escapes,
                        eager_ops=eager_ops,
                        declined_compiles=self._declined_compiles,
                    )
                if not ran:
                    ran, result = warm_up, tape.warm_up_result
                tape.warm_up_result = None
                # an op the trace cannot run eagerly (a CuTe launch's) fails as the call's
                if failed is not None and any(k >= len(tape.ops) or not all(isinstance(tape.launches[i][1], EagerCall) for i in tape.ops[k].launches) for k in eager_ops):
                    raise failed
                traced = tape.contract if self.check_global_state or self.trusted is not None else (tape.contract[0], ())
                if traced != contract:
                    # a warm-up that initialized global state on first use (the
                    # SDPA priority order) leaves the state the trace ran under
                    if not first or traced[0] != contract[0]:
                        raise declined("the call changed the global state")
                    contract = traced
                if self.fullgraph:
                    steps = [r for _, r in tape.launches if isinstance(r, EagerCall) and not isinstance(r, OpaqueCall) and not r.host]
                    if steps:
                        why = "; ".join(f"{r.name} ({r.reason or 'no traced implementation'})" for r in steps)
                        raise EagerFallback(f"host_trace: eager steps in the tape: {why}; fullgraph=True does not run them eagerly")
                if self.forbid_learners:
                    calls = {id(rec) for _, rec in tape.launches if isinstance(rec, OpaqueCall)}
                    if calls and (bound := bind_opaque(tape, calls, self._learn)) is not None:
                        tape = bound[0]
                    if names := [r.name for _, r in tape.launches if isinstance(r, OpaqueCall)]:
                        raise EagerFallback(
                            f"host_trace: opaque calls whose keys neither bind nor learn at the trace: {', '.join(map(str, names))}; forbid_learners=True does not run them eagerly"
                        )
                folded = next((v for v, values, unselected in candidates if self._fold(v, values, unselected, tape, args)), None)
                if folded is not None:
                    break
                if oob_learn and (calls := {id(rec) for _, rec in tape.launches if isinstance(rec, OpaqueCall)}):
                    pool = torch.cuda.MemPool() if self.learn_pool else None

                    def learn(*key: Any) -> OpaqueBinding | None:
                        with torch.cuda.use_mem_pool(pool, tape.device) if pool is not None else contextlib.nullcontext():
                            return self._learn(*key)

                    bound = bind_opaque(tape, calls, learn, keep_guards=True)
                    del pool, learn
                    if bound is not None:
                        unbound = _unbound(tape, bound[1])
                        tape = bound[0]
                        self.oob_binds += 1
                try:
                    variant = self._build(tape)
                    break
                except SegmentFailed as e:
                    # as a trace-time decline is, local: the ops that made
                    # the launches it failed at run eagerly, the rest replays
                    at = {i for i, (seq, _) in enumerate(tape.launches) if seq in e.seqs}
                    ops = {k for k, op in enumerate(tape.ops) if any(i in op.launches for i in at)}
                    if len(at) != len(e.seqs) or not ops or not ops.isdisjoint(eager_ops):
                        raise
                    self._note(str(e))
                    _drop_tracebacks(e)
                    failed = e
                    eager_ops |= dict.fromkeys(ops, str(e))
                    tape.release_args()
        except Exception as e:
            if isinstance(e, EagerFallback) or (isinstance(e, AssertionError) and tape is not None):
                raise  # fullgraph's, or the lowering's or capture's own bug
            if torch.cuda._host_trace.raise_unexpected and not isinstance(e, (Declined, torch.OutOfMemoryError)):
                raise
            if isinstance(e, Declined):
                if tape is None:
                    ran, result = e.warm_up_ran, e.warm_up_result
                # a warm-up's own calls may be first-use work the first call did
                # not do (an autotuner's benchmark at a new size): once per class
                if e.witness and exact not in self._witness_retried:
                    self._witness_retried.add(exact)
                    e.retry = True
                if e.meta_op is not None:
                    self._meta_disagrees.add(e.meta_op)
            elif warm_up and tape is None:
                raise  # the warm-up's own error: the call's
            # the raise site's frame holds the exception, and its traceback holds
            # the frames up to this call's (its arguments, the tape): a cycle
            # only a gc frees, and the call's memory would outlive it
            _drop_tracebacks(e)
            if self.fullgraph and not (isinstance(e, Declined) and e.retry):
                why = e if isinstance(e, Declined) else f"host_trace: {type(e).__name__}: {e}"
                raise EagerFallback(f"{why}; fullgraph=True does not run the call eagerly") from e
            if not isinstance(e, Declined) or (isinstance(e, SegmentFailed) and not isinstance(e.__cause__, Declined)):
                self._declined.add(exact)  # an OOM or a capture's error, say: the call's, not the graph's
            elif not e.retry:
                self._declined.add(contract if e.side_stream else structural)
                self.structural += not e.uncaptured
            if isinstance(e, Declined) and e.uncaptured:
                self.uncaptured += 1
                self._note(str(e))
                if torch.cuda._host_trace.keep_eager_regions and not e.retry and not eager_ops and tape is not None:
                    try:
                        self._eager_regions.setdefault((contract, exact[2]), []).append(lower_guards(tape))
                    except Declined as d:
                        self._note(str(d))
            elif isinstance(e, Declined) and e.retry:
                self._note(f"{e} (retried)")
            elif isinstance(e, Declined):
                self._note(str(e), e.side_stream)
            else:
                self._note(f"host_trace: {type(e).__name__}: {e} (declined)")
            if ran:
                self.eager += 1
                return result
            return self._eager(args, {}, fallback=not (isinstance(e, Declined) and e.retry))
        if folded is not None:
            if ran:
                return result
            if self.handback:
                return Handback(tuple(args))
            outcome, result = self._call_native(folded, tuple(args))
            if outcome == MISS:
                raise AssertionError("host_trace: a call misses the variant its trace folded into")
            return self._served(folded.native, outcome, result, tuple(args))
        variant.unbound = unbound
        family = self._families.setdefault(contract, [])
        family.insert(family.index(before) if before in family else len(family), variant)
        self.learners += variant.learns
        self._register(family, tuple(args))
        self.guards += tape.guard_count
        lowered = variant.captured.lowered
        if lowered.opaque and (evaluated := variant.native.evaluate(tuple(args))) is not None:
            values = evaluated[0]
            # a key bound since the call was traced (a warm-up's harvest) is the relower's to take
            variant.tried.update(k for i, o in lowered.opaque.items() if lowered.steps[i].call.bound and o.provider.bind(k := o.key(values)) is not None)
        if ran:
            return result
        if self.handback and not variant.learns:
            return Handback(tuple(args))
        outcome, result = self._call_native(variant, tuple(args))
        if outcome == MISS:
            raise AssertionError("host_trace: a call misses the tape traced at it")
        return self._served(variant.native, outcome, result, tuple(args))

    def _retrace_cause(self, contract: tuple, args: Sequence[Any]) -> None:
        """Counts why the call traces again (retrace_causes) and logs it
        (trace_structured host_trace_retrace): the closest variant's first
        failing guard, the op that recorded it and the user's line that did
        (SLoc.maybe_user_loc), and for an op's own guard its redispatch's
        refusal. A miss path's."""
        causes = []
        for variant in self._families.get(contract, ()):
            low, tape = variant.captured.lowered, variant.tape
            env = tape.shape_env
            result = low.compiled.evaluate_inputs(tuple(args))
            if result is None or result[0] != Status.SUCCESS:
                causes.append(("other", None, "an input of another kind, or a failing status", None, None))
                continue
            values = result[1]
            if values[low.valid] == 1:
                op, why = self._last_refusal.get(id(variant.native), (None, None))
                sel = next((s for s in low.selectors if s.op == op), None)
                pairs = zip(tape.ops[op].guards, sel.rows) if sel is not None else ()
                kind = ("dispatch", None if op is None else str(tape.ops[op].func)) if why is not None else ("other", None)
            else:
                pairs, kind, why = zip(tape.graph, low.lowering.lowering.guard_rows[: len(tape.graph)]), None, None
            g = next((g for g, row in pairs if values[row] != 1), None)
            if g is None:
                text = "an allocation's requirement or a declared sign" if kind is None else "its guards hold" if why is None else ""
                causes.append((*(kind or ("other", None)), text, None, why))
                continue
            text = _ir.render(env.records[g][0]) if low.lowering.ir else str(env.guards[g].expr)
            where = (env.slocs[g] if low.lowering.ir else env.guards[g].sloc).maybe_user_loc
            if kind is None:
                owners = env.owner_sets[g] if env.owner_sets is not None else {env.owners[g]}
                op = next((str(tape.ops[o].func) for o in sorted(o for o in owners if o is not None)), None)
                kind = ("meta" if op is not None else "graph", op)
            causes.append((*kind, text, where, why))
        cause = next((c for c in causes if c[0] == "dispatch"), causes[0] if causes else ("contract", None, "", None, None))
        self.retrace_causes[cause] = self.retrace_causes.get(cause, 0) + 1
        trace_structured(
            "artifact",
            metadata_fn=lambda: {"name": "host_trace_retrace", "encoding": "string"},
            payload_fn=lambda: f"class {cause[0]}; op {cause[1]}; guard {cause[2]}; at {cause[3]}; refusal {cause[4]}",
        )

    def _fold(self, variant: _Variant, values: list[int], unselected: list[int], tape: Tape, args: Sequence[Any]) -> bool:
        """Whether the trace of the call `args` folds into the variant, whose
        graph guards hold the call (at rows `values`) but not the selectors
        `unselected`: each of those gains the trace's launches of its op as an
        entry (_host_trace_lower_tape.fold)."""
        if self._add_entries(variant, self.fold_refusals, args, fold, variant.captured.lowered, tape, args, values, unselected) is not None:
            return False
        self.folds += 1
        return True

    def _redispatch(self, variant: _Variant, args: tuple, values: list[int], unselected: list[int], dispatches: dict | None = None) -> FoldRefused | None:
        """None where the selectors `unselected` of the variant, whose graph
        guards hold the call, gain entries from their ops dispatched again at
        the call's metadata (_host_trace_redispatch): no trace; else why not."""
        start = time.perf_counter()
        refused = self._add_entries(variant, self.redispatch_refusals, args, redispatch, variant.captured.lowered, args, values, unselected, dispatches)
        # a refusal keeps the rows it appended, evaluated at the call, which a fold reads
        values += variant.captured.lowered.lowering.program.values[len(values) :]
        self.redispatch_s += time.perf_counter() - start
        self.redispatches += refused is None
        if refused is not None:
            self._last_refusal[id(variant.native)] = refused.op, str(refused)
        return refused

    def _eager_sites(self, contract: tuple, variant: _Variant, args: tuple, refused: list[tuple[int, OpaqueKey]]) -> _Variant | None:
        """The variant's tape before its opaque calls were bound, with the
        calls of keyed sites `refused` (index, the key the provider refuses at
        the call) as plain eager steps, each guarding its key (its op's twin
        guards), and the others bound; built at the call and put after it in
        its family: what a trace at the call builds, with no trace. None where
        the variant has no such tape or the build declines."""
        start = time.perf_counter()
        sites = variant.captured.lowered.sites
        try:
            if variant.unbound is None or any(id(sites[i].site) not in variant.unbound[1] for i, _ in refused):
                raise FoldRefused("a keyed site the trace recorded bound")
            # the twin's tape is the trace's, whose own op guards fail at a call an entry (a fold's or a redispatch's) selects
            if variant.captured.lowered.evaluate(args) is None:
                raise FoldRefused("an op's own guards fail at the call: an entry selects it")
            tape, calls = variant.unbound
            at = {id(r): j for j, (_, r) in enumerate(tape.launches)}
            eager = copy.copy(tape)
            eager.launches, eager.twin_guards, eager.args = list(tape.launches), dict(tape.twin_guards), tuple(args)
            for i, key in refused:
                r = calls[id(sites[i].site)]
                if (guards := key_guards(r, key)) is None:
                    raise FoldRefused(f"{key.op}'s key is not its call's")
                why = f"{key.op} at sizes {key.sizes}: {sites[i].site.provider.refusal(key)}"
                j = at[id(r)]
                eager.launches[j] = (tape.launches[j][0], EagerCall(r.target, r.args, r.kwargs, r.outputs, why, r.generator, r.state))
                k = next(k for k, op in enumerate(tape.ops) if j in op.launches)
                eager.twin_guards[k] = (*eager.twin_guards.get(k, ()), *guards)
            rest = {id(r) for _, r in eager.launches if isinstance(r, OpaqueCall)}
            # the other sites' keys at the trace bound when the variant was built
            bound = bind_opaque(eager, rest, keep_guards=oob_learn) if rest else (eager, {})
            if bound is None:
                raise FoldRefused("its other opaque calls do not bind")
            # its other sites are bound at the trace's keys, which its refused keys' guards may contradict
            twin = self._build(bound[0])
        except (Declined, FoldRefused, _StaleBinding) as e:
            reason = f"the build declined: {e}" if isinstance(e, Declined) else str(e)
            self.eager_sites_refusals[reason] = self.eager_sites_refusals.get(reason, 0) + 1
            variant.eager_sites[frozenset(refused)] = None
            return None
        finally:
            self.eager_sites_s += time.perf_counter() - start
        self.eager_sites += 1
        twin.unbound = _unbound(eager, bound[1])
        variant.eager_sites[frozenset(refused)] = twin
        family = self._families[contract]
        family.insert(family.index(variant) + 1, twin)
        self._register(family, args)
        self.guards += twin.tape.guard_count
        return twin

    def _add_entries(self, variant: _Variant, refusals: dict[str, int], args: tuple, make: Callable[..., Any], *make_args: Any) -> FoldRefused | None:
        try:
            program, entries = make(*make_args)
        except FoldRefused as e:
            if e.program is not None and not self._native_mutate(variant, "a program", variant.native.set_program, e.program):
                e = FoldRefused(f"the native variant rejects a program ({e})")
            refusals[str(e)] = refusals.get(str(e), 0) + 1
            _drop_tracebacks(e)
            return e
        relocated = variant.memory.relocated
        if any(isinstance(s, PointerSlot) and s.base in relocated for _, _, launches in entries for lo in launches for s in lo.slots):
            program, entries = relocate_entries(variant.captured.lowered, relocated, entries, args)
        ok = self._native_mutate(variant, "a program", variant.native.set_program, program)
        for site, predicate, launches in entries if ok else ():
            if not (ok := self._native_mutate(variant, "an entry", variant.native.add_entry, site, predicate, tuple(map(launch_row, launches)))):
                break
            variant.folded.append(launches)
        if not ok:
            why = "the native variant rejects an entry or its program"
            refusals[why] = refusals.get(why, 0) + 1
            return FoldRefused(why)
        return None

    def _build(self, tape: Tape) -> _Variant:
        if not hasattr(torch._C, "_HostTraceVariant"):
            raise declined("no native replay in this build")
        calls = [rec for _, rec in tape.launches if isinstance(rec, EagerCall)]
        for call in calls:
            # a Triton target, which returns nothing, holds its grid's SymInts
            if (
                not isinstance(call.target, tuple)
                and call.target in self._meta_disagrees
            ):
                raise declined(
                    f"{call.name} returned other metadata than its fake kernel's"
                )
        lowered = lower_tape(tape)
        # a keyed site's row at its key holds the nodes of that key's binding, never another key's
        for s in lowered.sites:
            if s.site.key is not None and (key := s.key([lowered.program.values[r] for r in s.rows])) != s.site.key:
                raise _StaleBinding(f"{s.site.op}'s row at {key.sizes} holds the binding at {s.site.key.sizes}")
        # eager order reads the native caching allocator's release count
        native = torch.cuda.get_allocator_backend() == "native"
        if native and self.memory == "auto":
            lowered, memory = auto_memory(lowered, self.freed_arguments, self.splits)
        elif native and self.memory == "eager":
            lowered, memory = split_runs(lowered, self.freed_arguments, splits=self.splits)
        elif self.memory == "held":
            memory = plan_memory(lowered, "held")
            # lower_tape dropped the program's inputs; only their count is compiled
            compiled = torch._C._HostTraceProgram(lowered.program.instructions, len(tape.args))
            lowered = dataclasses.replace(lowered, compiled=compiled)
        elif self.memory in _PLANNED:
            memory, arenas = size_classes(
                lowered, packed=self.memory == "packed", reuse=self.memory != "planned_no_reuse", split=self.memory != "planned_no_split", scratch=self.memory == "planned_scratch"
            )
        else:
            memory = plan_memory(lowered, self.memory if native else "run_buffer")
        check_plan(lowered, memory)
        # capture_tape checks the compiled program at the traced call against
        # these rows
        values = lowered.program.values
        # the trace's placeholders: the capture runs no kernel, and the first
        # replay patches every pointer
        addresses = [_placeholder(rec.root) for rec in tape.allocs]
        planned = self.memory in _PLANNED
        if planned:
            addresses = arena_addresses(lowered, arenas, addresses)
        captured = capture_tape(lowered, addresses)
        if planned:
            captured, memory = relocate(captured, memory, arenas)
        bases = [*addresses, *(_placeholder(r) for r in lowered.eager_roots)]
        tape.release_args()
        for call in calls:
            # a step a capture failure made is noted as that failure
            if not isinstance(call, OpaqueCall) and not call.host and call.reason not in self._reasons:
                self._note(f"host_trace: an eager step in a variant: {call.name} ({call.reason or 'no traced implementation'})")
            if not isinstance(call.target, tuple) or call.reason is None:
                continue
            if call.reason not in self.triton_fallbacks:
                self.triton_fallbacks[call.reason] = None
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "host_trace_triton_fallback",
                        "encoding": "string",
                    },
                    payload_fn=lambda why=call.reason: why,
                )
        index = tape.device.index
        device = torch.device(
            "cuda", torch.cuda.current_device() if index is None else index
        )
        opaque = {id(lowered.steps[i]): o for i, o in lowered.opaque.items()}
        spec = flatten_variant(
            captured,
            (values, bases),
            memory,
            lambda step: _eager_step(step, device, opaque.get(id(step))),
            _Disagreement,
        )
        return _Variant(captured, memory, native_variant(spec))


def _site_classes(sites: Sequence[LoweredKeyedSite]) -> list[int]:
    """Per keyed site, a class: equal for sites whose OpaqueKey is the same
    function of the key values (_opaque_key's other inputs equal)."""
    classes: dict[tuple, int] = {}
    out = []
    for i, s in enumerate(sites):
        scalars = tuple(None if isinstance(v, (torch.SymInt, ScalarSlot)) else v for v in s.site.scalars)
        sig = (id(s.site.provider), s.site.op, s.dtypes, s.ranks, scalars, s.device, s.site.state, s.alias)
        try:
            out.append(classes.setdefault(sig, len(classes)))
        except TypeError:  # an unhashable constant: a class of its own
            out.append(classes.setdefault(("site", i), len(classes)))
    return out


def _unbound(tape: Tape, bound: dict[int, KeyedSite]) -> tuple[Tape, dict[int, OpaqueCall]]:
    # bind_opaque's sites by their calls' ids -> each site's call, by the site's id;
    # the tape keeps no argument alive
    tape.release_args()
    calls = {id(r): r for _, r in tape.launches if isinstance(r, OpaqueCall)}
    return tape, {id(site): calls[c] for c, site in bound.items()}


def _eager_step(
    step: LoweredEagerCall, device: torch.device, opaque: LoweredOpaqueCall | None
) -> Callable[[list[Any], Sequence[int]], tuple]:
    """An eager step's op call for the native commit: run(leaves, values)
    calls it on the leaves the commit built, checks its outputs against the
    trace's prediction and returns each fresh output as (root, tensor). An
    opaque call's provider learns from a run whose key does not bind."""
    target = step.call.target

    def run(leaves: list[Any], values: Sequence[int]) -> tuple:
        key = bound = None
        # the library state eager's call read at the trace
        state = library_state_as(step.call.state) if step.call.state else contextlib.nullcontext()
        if opaque is not None:
            key = opaque.key(values)
            bound = opaque.provider.bind(key)
        if step.flat:
            call_args, call_kwargs = leaves, {}
        else:
            call_args, call_kwargs = pytree.tree_unflatten(leaves, step.spec)
        if isinstance(target, tuple) and target[0] == "host":
            target[1](*call_args, **call_kwargs)
            return ()
        if isinstance(target, tuple) and target[0] == "cute":
            _, compiled, streams, cute_args = target
            call_args = list(call_args)
            stream = torch.cuda.current_stream()
            for i, kind in streams:
                is_torch = issubclass(kind, torch.cuda.Stream)
                call_args[i] = stream if is_torch else kind(stream.cuda_stream)
            for i, arg in cute_args or ():
                call_args[i] = arg.make(call_args[i])
            compiled(*call_args, **call_kwargs)
            return ()
        if isinstance(target, tuple):
            _, jit, _, options = target
            grid = tuple(values[r] for r in step.grid or ())
            jit.run(*call_args, grid=grid, warmup=False, **options)
            return ()
        with state, torch._C._AutoDispatchBelowADInplaceOrView():
            if step.call.generator is None:
                out = target(*call_args, **call_kwargs)
            else:
                out = _impl_graphsafe_rng(target, *call_args, rng_state=step.call.generator, **call_kwargs)
        if isinstance(out, torch.Tensor):
            outs = [out]
        else:
            outs = [o for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor)]
            outs = seed_offset_on_device(target, outs, device)
        if len(outs) != len(step.outputs):
            raise _Disagreement(f"{target} returned {len(outs)} tensors", target)
        fresh = []
        for i, (o, p) in enumerate(zip(outs, step.outputs)):
            if not isinstance(p, PredictedOutput):
                if o is not leaves[p]:
                    raise _Disagreement(
                        f"{target} output {i} is not its argument", target
                    )
                continue
            want = (
                [values[r] for r in p.sizes],
                [values[r] for r in p.strides],
                values[p.offset],
                p.dtype,
                device,
            )
            got = (
                list(o.shape),
                # a size-1 dim's stride addresses nothing (_addressing), nor does a zero-element output's: eager's stands
                [w if n == 1 or o.numel() == 0 else s for n, s, w in zip(o.shape, o.stride(), want[1])],
                o.storage_offset(),
                o.dtype,
                o.device,
            )
            base = o.untyped_storage().data_ptr()
            if got != want or (o.numel() and base % _ALLOC_ALIGNMENT):
                raise _Disagreement(
                    f"{target} output {i} is (sizes, strides, storage offset, dtype, device) "
                    f"{got} at address {base:#x}; its fake kernel predicted {want}",
                    target,
                )
            fresh.append((p.root, o))
        if opaque is not None and bound is None:
            operands = [v for v in leaves if isinstance(v, torch.Tensor)]
            operands += [o for o, p in zip(outs, step.outputs) if isinstance(p, PredictedOutput) and not any(o is v for v in operands)]
            opaque.provider.learn(key, tuple(call_args), call_kwargs, operands)  # pyrefly: ignore [bad-argument-type]
        return tuple(fresh)

    return run
