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
A call no variant holds is traced at its own inputs without a warm-up (the
replay of its new capture is the call) and joins its family. A trace,
lowering or capture that declines runs the function eagerly, and that call's
exact class is not traced again; under trust a Declined is the graph's, and
no call is. Every eager fallback records its reason in `declines`, once per
reason.
"""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
from torch._logging import trace_structured
from torch._prims.rng_prims import _impl_graphsafe_rng
from torch.cuda import _host_trace_cute  # noqa: F401  hooks cute.compile
from torch.cuda._host_trace import Declined, declined
from torch.cuda._host_trace_capture import capture_tape, instantiate_form, plain_attributes
from torch.cuda._host_trace_lower_tape import lower_tape, PredictedOutput
from torch.cuda._host_trace_memory import auto_memory, plan_memory, split_runs
from torch.cuda._host_trace_native import (
    binding_row,
    CAPTURE,
    flatten_variant,
    HIT,
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
    current_trace,
    EagerCall,
    trace,
)
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from torch.cuda._host_trace_capture import CapturedTape
    from torch.cuda._host_trace_lower_tape import LoweredEagerCall, LoweredOpaqueCall
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
    replay allocates (_host_trace_memory): "auto", "eager" or "run_buffer".
    `freed_arguments` are the positions whose tensor a boxed call's caller
    holds no other reference to, which split_runs may free mid-tape.

    The call is the base's: a hit runs in C++, anything else is _call_slow."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        max_variants: int = 16,
        trusted: TrustedInputs | None = None,
        opaque: Sequence[OpaqueProvider] = (),
        memory: str = "eager",
        freed_arguments: Collection[int] = (),
    ) -> None:
        if memory not in ("auto", "eager", "run_buffer"):
            raise ValueError(f"host_trace: memory {memory!r} is not 'auto', 'eager' or 'run_buffer'")
        self.fn = fn
        self.memory = memory
        self.freed_arguments = frozenset(freed_arguments)
        self.max_variants = max_variants
        self.trusted = trusted
        self.opaque = opaque
        try:
            self._signature: inspect.Signature | None = inspect.signature(fn)
        except (TypeError, ValueError):
            self._signature = None
        self._families: dict[tuple, list[_Variant]] = {}
        self._declined: set[tuple] = set()
        # the operators whose output disagreed with the trace's metadata
        self._meta_disagrees: set[Any] = set()
        # every fallback's reason, in order, each once
        self._reasons: dict[str, None] = {}
        # why each Triton launch that runs eagerly in a replay does, each once
        self.triton_fallbacks: dict[str, None] = {}
        self.traces = 0
        self.replays = 0
        self.eager = 0
        # the traces that found nothing to capture (Declined.uncaptured)
        self.uncaptured = 0
        # the traces whose decline holds for their class: not a retry, not the
        # call's own error, not uncaptured
        self.structural = 0
        # the guards of every variant's tape, summed
        self.guards = 0
        # one call at a time: a call patches the execs it replays (a native
        # call holds the base's lock too)
        self._lock = threading.RLock()
        self._native_init(_active, _Disagreement, trusted is not None)

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
        contract = () if self.trusted is not None else argument_contract(args)
        try:
            family = self._families.get(contract)
        except TypeError:
            return self._eager(args, {}, "an argument without a hash")
        # the collector waits out a miss's trace, lowering and instantiations
        with self._lock, _gc_hold:
            # a variant that does not learn holds the call once its keys bind
            pending = False
            for variant in family or ():
                if searched and not variant.learns:
                    continue
                values = self._fill(variant, args)
                if values is None:
                    pending |= not variant.learns and variant.native.evaluate(args) is not None
                    continue
                if variant.learns and len(self.variants) < self.max_variants:
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
                        return self._miss(contract, args, before=variant)
                outcome, result = self._call_native(variant, args)
                if outcome != MISS:
                    return self._served(variant.native, outcome, result, args)
            return self._miss(contract, args)

    def _fill(self, variant: _Variant, args: tuple) -> list[int] | None:
        """The call's rows if the variant's program holds it, after adding
        each keyed site's missing key to its table and each segment's missing
        form; None if a key does not bind yet (a learning variant's eager run
        harvests it)."""
        evaluated = variant.native.evaluate(args)
        if evaluated is None:
            return None
        values, missing, unformed = evaluated
        sites = variant.captured.lowered.sites
        bindings = []
        for i, key in missing:
            binding = sites[i].site.provider.bind(sites[i].key(key))
            if binding is None:
                return None
            bindings.append((i, key, binding))
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
        return values

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
        self, contract: tuple, args: Sequence[Any], before: _Variant | None = None
    ) -> Any:
        if any(v.native.overlaps(tuple(args)) for v in self._families.get(contract, ())):
            # an assertion, not a dispatch: eager raises its error or runs it
            return self._eager(args, {}, "arguments overlap that a variant's steps take as disjoint")
        exact = _exact_class(contract, args)
        # under trust a decline is structural: one is the graph's
        structural = contract if self.trusted is not None else exact
        if structural in self._declined or exact in self._declined:
            return self._eager(args, {})
        if any(isinstance(a, torch.Tensor) and a.numel() == 0 for a in args):
            # trace declines an empty argument, and a tape guards its nonempty
            return self._eager(args, {}, "an empty tensor argument")
        if len(self.variants) >= self.max_variants:
            return self._eager(args, {}, f"max_variants ({self.max_variants}) exist")
        device = next((a.device for a in args if isinstance(a, torch.Tensor)), None)
        if device is not None and device.type == "cuda" and _capturing(device):
            # not a decline of the class: the capture ends
            return self._eager(args, {}, f"a capture on {device}")
        # the entry's first trace warms up (lazy initialization happens outside
        # the trace, and the warm-up is the call); a later trace does not
        warm_up = self.traces == 0
        ran, result, tape = False, None, None
        self.traces += 1
        try:
            tape = trace(
                self.fn,
                tuple(args),
                warm_up=warm_up,
                trusted=self.trusted,
                opaque=self.opaque,
            )
            ran, result, tape.warm_up_result = warm_up, tape.warm_up_result, None
            if self.trusted is None and tape.contract != contract:
                raise declined("the call changed the global state")
            variant = self._build(tape)
        except Exception as e:
            if isinstance(e, AssertionError) and tape is not None:
                raise  # the lowering's or capture's own bug
            if torch.cuda._host_trace.raise_unexpected and not isinstance(e, (Declined, torch.OutOfMemoryError)):
                raise
            if isinstance(e, Declined):
                if tape is None:
                    ran, result = e.warm_up_ran, e.warm_up_result
                if e.meta_op is not None:
                    self._meta_disagrees.add(e.meta_op)
            elif warm_up and tape is None:
                raise  # the warm-up's own error: the call's
            if not isinstance(e, Declined):
                self._declined.add(exact)  # an OOM, say: the call's, not the graph's
            elif not e.retry:
                self._declined.add(structural)
                self.structural += not e.uncaptured
            if isinstance(e, Declined) and e.uncaptured:
                self.uncaptured += 1
                log.debug("%s", e)
            elif isinstance(e, Declined):
                self._note(str(e))
            else:
                self._note(f"host_trace: {type(e).__name__}: {e} (declined)")
            if ran:
                self.eager += 1
                return result
            return self._eager(args, {})
        family = self._families.setdefault(contract, [])
        family.insert(family.index(before) if before in family else len(family), variant)
        self._register(family, tuple(args))
        self.guards += len(tape.guards)
        lowered = variant.captured.lowered
        if lowered.opaque and (evaluated := variant.native.evaluate(tuple(args))) is not None:
            values = evaluated[0]
            variant.tried.update(k for o in lowered.opaque.values() if o.provider.bind(k := o.key(values)) is not None)
        if ran:
            return result
        outcome, result = self._call_native(variant, tuple(args))
        if outcome == MISS:
            raise AssertionError("host_trace: a call misses the tape traced at it")
        return self._served(variant.native, outcome, result, tuple(args))

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
            lowered, memory = auto_memory(lowered, self.freed_arguments)
        elif native and self.memory == "eager":
            lowered, memory = split_runs(lowered, self.freed_arguments)
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
        with torch._C._AutoDispatchBelowADInplaceOrView():
            if step.call.generator is None:
                out = target(*call_args, **call_kwargs)
            else:
                out = _impl_graphsafe_rng(target, *call_args, rng_state=step.call.generator, **call_kwargs)
        if isinstance(out, torch.Tensor):
            outs = [out]
        else:
            outs = [o for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor)]
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
