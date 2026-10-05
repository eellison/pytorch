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
reason.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
from torch._logging import trace_structured
from torch._prims.rng_prims import _impl_graphsafe_rng
from torch.cuda import _host_trace_cute  # noqa: F401  hooks cute.compile
from torch.cuda._host_trace import Declined, declined
from torch.cuda._host_trace_capture import capture_tape, instantiate_form, plain_attributes, SegmentFailed
from torch.cuda._host_trace_lower_tape import fold, FoldRefused, lower_tape, PredictedOutput
from torch.cuda._host_trace_memory import auto_memory, plan_memory, split_runs, SPLITS
from torch.cuda._host_trace_opaque import library_state_as
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
    _hint,
    argument_contract,
    bind_opaque,
    current_trace,
    EagerCall,
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
        LoweredLaunch,
        LoweredMemset,
        LoweredOpaqueCall,
    )
    from torch.cuda._host_trace_memory import MemoryPlan
    from torch.cuda._host_trace_opaque import KeyedSite, OpaqueBinding, OpaqueKey, OpaqueProvider
    from torch.cuda._host_trace_tape import Tape, TrustedInputs


log = logging.getLogger(__name__)


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


def _capturing(device: torch.device) -> bool:
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


class _Disagreement(AssertionError):
    """A replay's own error after its first effect: the variant is dropped."""

    def __init__(self, msg: str, op: Any = None) -> None:
        super().__init__(f"host_trace: {msg}")
        self.op = op  # the eager call's operator whose metadata was wrong


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

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if current_trace() is not None:
                return self.fn(*args, **kwargs)  # pyrefly: ignore [missing-attribute]
            return self._call_slow(
                args, kwargs, False
            )  # pyrefly: ignore [missing-attribute]

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
    replay allocates (_host_trace_memory): "auto", "eager" or "run_buffer";
    `splits`, where "auto" and "eager" split a run (SPLITS, split_runs).
    `freed_arguments` are the positions whose tensor a boxed call's caller
    holds no other reference to, which split_runs may free mid-tape. With
    `handback`, a slow call that traces a variant or adds to one's tables
    returns Handback instead of running the variant, which Python's reference
    to the arguments would hold to the end; the caller calls again, with
    call_boxed. `static_shapes` are the tensor positions whose layout is static (trace).

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

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        trusted: TrustedInputs | None = None,
        opaque: Sequence[OpaqueProvider] = (),
        memory: str = "eager",
        freed_arguments: Collection[int] = (),
        static_shapes: Collection[int] = (),
        check_global_state: bool = False,
        check_escapes: bool = False,
        handback: bool = False,
        splits: str = "peak",
    ) -> None:
        if memory not in ("auto", "eager", "run_buffer"):
            raise ValueError(f"host_trace: memory {memory!r} is not 'auto', 'eager' or 'run_buffer'")
        if splits not in SPLITS:
            raise ValueError(f"host_trace: splits {splits!r} is not one of {SPLITS}")
        self.fn = fn
        self.memory = memory
        self.freed_arguments = frozenset(freed_arguments)
        self.handback = handback
        self.splits = splits
        self.static_shapes = frozenset(static_shapes)
        self.trusted = trusted
        self.opaque = opaque
        self.check_global_state = check_global_state
        self.check_escapes = check_escapes
        try:
            self._signature: inspect.Signature | None = inspect.signature(fn)
        except (TypeError, ValueError):
            self._signature = None
        self._families: dict[tuple, list[_Variant]] = {}
        self._declined: set[tuple] = set()
        # the classes whose warm-up's operator calls were not their trace's once
        self._witness_retried: set[tuple] = set()
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
        self.replays = 0
        self.eager = 0
        # the traces that found nothing to capture (Declined.uncaptured)
        self.uncaptured = 0
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
        # the keys harvested at a keyed site's miss, no trace or relower
        # (_learn), and their seconds
        self.learned = 0
        self.learn_s = 0.0
        # one call at a time: a call patches the execs it replays (a native
        # call holds the base's lock too)
        self._lock = threading.RLock()
        self._native_init(_active, _Disagreement, trusted is not None, check_global_state)

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
            # a variant that does not learn holds the call once its keys bind
            pending = False
            candidates: list[tuple[_Variant, list[int], list[int]]] = []
            for variant in tuple(family or ()):
                if searched and not variant.learns:
                    continue
                held = len(candidates)
                filled = self._fill(variant, args, candidates)
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
                            return self._miss(contract, args, candidates, before=variant)
                        if self._fill(upgraded, args, candidates) is not None:
                            variant = upgraded
                elif self.handback and added:
                    return Handback(args)
                outcome, result = self._call_native(variant, args)
                if outcome != MISS:
                    return self._served(variant.native, outcome, result, args)
            # no variant holds the call: one whose graph guards do gains its
            # failing selectors' entries from their ops dispatched again
            for j, (variant, values, unselected) in enumerate(candidates if self.trusted is None else ()):
                if not self._redispatch(variant, args, values, unselected):
                    continue
                del candidates[j]
                if self._fill(variant, args, candidates) is not None:
                    outcome, result = self._call_native(variant, args)
                    if outcome != MISS:
                        return self._served(variant.native, outcome, result, args)
                break
            return self._miss(contract, args, candidates)

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
        for ids, how in ((set(calls), self._learn), (set(wanted), None)):
            bound = bind_opaque(variant.tape, ids, how)
            if bound is not None and all(c in bound[1] and bound[1][c].fits(b) for c, b in wanted.items()):
                break
        else:
            return None
        try:
            upgraded = self._build(bound[0])
        except Declined:
            return None
        self.relowers += 1
        family.insert(family.index(variant), upgraded)
        self._register(family, args)
        self.guards += upgraded.tape.guard_count
        return upgraded

    def _fill(self, variant: _Variant, args: tuple, candidates: list[tuple[_Variant, list[int], list[int]]]) -> tuple[list[int], bool] | None:
        """The call's rows if the variant's program holds it, after adding
        each keyed site's missing key to its table and each segment's missing
        form, and whether it added any; None if a key does not bind yet (a learning variant's eager run
        harvests it), or if a selector selects nothing: then a variant that
        does not learn is a candidate to dispatch those selectors' ops again
        for, or to fold the call's trace into, (variant, rows, selectors) in
        `candidates`."""
        evaluated = variant.native.evaluate(args)
        if evaluated is None:
            return None
        values, missing, unformed, unselected = evaluated
        if unselected:
            if not variant.learns:
                candidates.append((variant, values, unselected))
            return None
        sites = variant.captured.lowered.sites
        bindings = []
        for i, key in missing:
            binding = sites[i].site.provider.bind(sites[i].key(key))
            site = sites[i].site
            if binding is None and not variant.learns and not site.rng and site.call is not None:
                binding = self._learn(site.op, site.provider, site.call, sites[i].key(key))
            bindings.append((i, key, binding))
        # past a key that does not bind the others still learn: a retrace then sees every refusal
        if any(b is None for _, _, b in bindings):
            return None
        for i, key, binding in bindings:
            arm = self._arm(variant, i, binding) if sites[i].site.fits(binding) else None
            if arm is None:
                variant.native.add_row(i, key, None)
            else:
                nodes, scratch = binding_row(sites[i], binding)
                piece = _piece(sites[i].site, binding)
                variant.native.add_row(i, key, nodes, arm, piece, scratch)
        if bindings:
            evaluated = variant.native.evaluate(args)
            unformed = [] if evaluated is None else evaluated[2]
        for g, arms in unformed:
            variant.native.add_form(g, arms, *self._form(variant, g, arms))
        return values, bool(bindings or unformed)

    def _learn(self, op: OpOverload, provider: OpaqueProvider, call: tuple[Any, frozenset[int]], key: OpaqueKey) -> OpaqueBinding | None:
        """The key's binding, harvested now from one eager run of `op` on
        buffers at the key's metadata, `call` its pytree spec and its tensor
        leaves' positions: no trace, no relower. None for a key the provider
        refuses, or an input that is not floating point (its values may index).
        The op draws no RNG: its run would draw from the generator."""
        if provider.refusal(key) is not None:
            return None
        spec, positions = call
        inputs = len(positions)
        metadata = list(zip(key.dtypes, key.sizes, key.strides, key.align))
        if not all(dtype.is_floating_point and align % dtype.itemsize == 0 for dtype, _, _, align in metadata[:inputs]):
            return None
        start = time.perf_counter()
        device = torch.device("cuda", key.device)
        generator = torch.Generator(device).manual_seed(0)
        tensors = []
        for dtype, sizes, strides, align in metadata[:inputs]:
            span = 1 + sum((n - 1) * s for n, s in zip(sizes, strides)) if all(sizes) else 0
            # the allocator's blocks are 512-byte aligned: the key's address % 256
            flat = torch.empty(align // dtype.itemsize + span, dtype=dtype, device=device).uniform_(-1, 1, generator=generator)
            tensors.append(flat.as_strided(sizes, strides, align // dtype.itemsize))
        given, scalars = iter(tensors), iter(key.scalars)
        args, kwargs = pytree.tree_unflatten([next(given) if j in positions else next(scalars) for j in range(spec.num_leaves)], spec)
        with library_state_as(key.state), torch._C._AutoDispatchBelowADInplaceOrView():
            out = op(*args, **kwargs)
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
            for n, a, b in zip(site.nodes, site.site.topology, binding.topology):
                if a != b:
                    own, other = dict(a[1]), dict(b[1])
                    keys = own.keys() | other.keys()
                    changes.append((launches[n].node, {k: other.get(k, plain[k]) for k in keys if own.get(k) != other.get(k)}))
        exec_, clone, placed = instantiate_form(segments[g].graph.raw_cuda_graph(), changes, [p[1:] for p in pieces])
        return exec_, clone, [(i, nodes) for (i, _, _), nodes in zip(pieces, placed)]

    def _call_native(self, variant: _Variant, args: tuple) -> tuple[int, Any]:
        try:
            return variant.native.call(args)
        except _Disagreement as e:
            self._disagreed(variant.native, e)
            raise

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

    def _note(self, reason: str) -> None:
        if reason not in self._reasons:
            self._reasons[reason] = None
            log.debug("%s", reason)

    def _eager(
        self, args: Sequence[Any], kwargs: dict[str, Any], why: str | None = None
    ) -> Any:
        if why is not None:
            self._note(f"host_trace: {why}; the call runs eagerly")
        self.eager += 1
        return self.fn(*args, **kwargs)

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
        if structural in self._declined or exact in self._declined:
            return self._eager(args, {})
        device = next((a.device for a in args if isinstance(a, torch.Tensor)), None)
        if device is not None and device.type == "cuda" and _capturing(device):
            # not a decline of the class: the capture ends
            return self._eager(args, {}, f"a capture on {device}")
        if not self._called:
            # as cudagraph trees' warm-up, an entry's first call runs eagerly:
            # first-use work (an autotuner's benchmark, a lazy initialization)
            # happens outside every trace, and fn runs once per call
            self._called = True
            return self._eager(args, {})
        # a trace warms up: the warm-up is the call, its operator calls the
        # trace's witness, and at the entry's first lazy initialization happens
        # outside the trace. Under trust Dynamo's guards stand for the witness.
        first = self.traces == 0
        warm_up = first or self.trusted is None
        ran, result, tape, folded = False, None, None, None
        # the ops whose segments failed to capture, run eagerly by a trace again
        eager_ops: dict[int, str] = {}
        failed: SegmentFailed | None = None
        try:
            while True:
                self.traces += 1
                tape = trace(
                    self.fn,
                    tuple(args),
                    warm_up=warm_up and not ran,
                    trusted=self.trusted,
                    opaque=self.opaque,
                    static_shapes=self.static_shapes,
                    check_escapes=self.check_escapes,
                    eager_ops=eager_ops,
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
                folded = next((v for v, values, unselected in candidates if self._fold(v, values, unselected, tape, args)), None)
                if folded is not None:
                    break
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
                    e.__traceback__ = None
                    failed = e
                    eager_ops |= dict.fromkeys(ops, str(e))
                    tape.release_args()
        except Exception as e:
            if isinstance(e, AssertionError) and tape is not None:
                raise  # the lowering's or capture's own bug
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
            e.__traceback__ = None
            if not isinstance(e, Declined) or (isinstance(e, SegmentFailed) and not isinstance(e.__cause__, Declined)):
                self._declined.add(exact)  # an OOM or a capture's error, say: the call's, not the graph's
            elif not e.retry:
                self._declined.add(structural)
                self.structural += not e.uncaptured
            if isinstance(e, Declined) and e.uncaptured:
                self.uncaptured += 1
                log.debug("%s", e)
            elif isinstance(e, Declined) and e.retry:
                log.debug("%s (retried)", e)
            elif isinstance(e, Declined):
                self._note(str(e))
            else:
                self._note(f"host_trace: {type(e).__name__}: {e} (declined)")
            if ran:
                self.eager += 1
                return result
            return self._eager(args, {})
        if folded is not None:
            if ran:
                return result
            if self.handback:
                return Handback(tuple(args))
            outcome, result = self._call_native(folded, tuple(args))
            if outcome == MISS:
                raise AssertionError("host_trace: a call misses the variant its trace folded into")
            return self._served(folded.native, outcome, result, tuple(args))
        family = self._families.setdefault(contract, [])
        family.insert(family.index(before) if before in family else len(family), variant)
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

    def _fold(self, variant: _Variant, values: list[int], unselected: list[int], tape: Tape, args: Sequence[Any]) -> bool:
        """Whether the trace of the call `args` folds into the variant, whose
        graph guards hold the call (at rows `values`) but not the selectors
        `unselected`: each of those gains the trace's launches of its op as an
        entry (_host_trace_lower_tape.fold)."""
        if not self._add_entries(variant, self.fold_refusals, fold, variant.captured.lowered, tape, args, values, unselected):
            return False
        self.folds += 1
        return True

    def _redispatch(self, variant: _Variant, args: tuple, values: list[int], unselected: list[int]) -> bool:
        """Whether the selectors `unselected` of the variant, whose graph
        guards hold the call, gain entries from their ops dispatched again at
        the call's metadata (_host_trace_redispatch): no trace."""
        start = time.perf_counter()
        added = self._add_entries(variant, self.redispatch_refusals, redispatch, variant.captured.lowered, args, values, unselected)
        # a refusal keeps the rows it appended, evaluated at the call, which a fold reads
        values += variant.captured.lowered.lowering.program.values[len(values) :]
        self.redispatch_s += time.perf_counter() - start
        self.redispatches += added
        return added

    def _add_entries(self, variant: _Variant, refusals: dict[str, int], make: Callable[..., Any], *make_args: Any) -> bool:
        try:
            program, entries = make(*make_args)
        except FoldRefused as e:
            if e.program is not None:
                variant.native.set_program(e.program)
            refusals[str(e)] = refusals.get(str(e), 0) + 1
            return False
        variant.native.set_program(program)
        for site, predicate, launches in entries:
            variant.native.add_entry(site, predicate, tuple(map(launch_row, launches)))
            variant.folded.append(launches)
        return True

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
        # eager order reads the native caching allocator's release count
        native = torch.cuda.get_allocator_backend() == "native"
        if native and self.memory == "auto":
            lowered, memory = auto_memory(lowered, self.freed_arguments, self.splits)
        elif native and self.memory == "eager":
            lowered, memory = split_runs(lowered, self.freed_arguments, splits=self.splits)
        else:
            memory = plan_memory(lowered, self.memory if native else "run_buffer")
        # capture_tape checks the compiled program at the traced call against
        # these rows
        values = lowered.program.values
        # the trace's placeholders: the capture runs no kernel, and the first
        # replay patches every pointer
        addresses = [_hint(rec.root.sym) for rec in tape.allocs]
        captured = capture_tape(lowered, addresses)
        bases = [*addresses, *(_hint(r.sym) for r in lowered.eager_roots)]
        tape.release_args()
        for call in calls:
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
            _, compiled, streams, _ = target
            call_args = list(call_args)
            stream = torch.cuda.current_stream()
            for i, kind in streams:
                is_torch = issubclass(kind, torch.cuda.Stream)
                call_args[i] = stream if is_torch else kind(stream.cuda_stream)
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
                # a size-1 dim's stride addresses nothing (_addressing)
                [w if n == 1 else s for n, s, w in zip(o.shape, o.stride(), want[1])],
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
