"""CuTe DSL kernels under a host trace.

A traced host may invoke a CuTe DSL kernel from Python through the runtime's
registered entry: `DirectCuTe(owner)` over an `ObservedOrdinaryEntry` (the
`@cute.jit` host, its kernel, a signature policy and the user's argument
conversion; torch/_inductor/runtime/_cudagraph/direct_cute.py). Such an
invocation enters the tape as launch records like any other launch (Tape.h
LaunchRec, as tape_records renders it), one record per launch site the host's
dispatch selects, each at its sequence position: the site's kernel symbol, the
pointer operands as values over the tape's roots, the integer operands as
values, the grid, block and shared bytes at the traced values, and under the
record's "cute" key what the runtime's binder needs to produce the physical
call at lowering (CuteLaunch). Nothing runs under the trace.

How the invocation is seen. trace()'s warm-up runs the call as written: the
owner's ordinary execution compiles the host once, observes its source (the
host MLIR, the function metadata, the accessors) and selects its executor. The
symbolic run sets the runtime's own dispatch seam (direct_invocation.ACTIVE,
which DirectCuTe.invoke consults before its compiled executor) to a handler of
the recorder, so the invocation reaches the recorder with its operands: the
traced tensors' fake twins (the same symbolic sizes, strides and offsets in
the trace's ShapeEnv) and its integers. The user's conversion runs on the
twins through the runtime's symbolic conversion trace (its views become
storage origins), the runtime's binder joins the operands to the observed
compilation (build_entry_signature -> bind_ordinary_metadata ->
CuTePreparation.bind, the correspondence receipt kept), and the host's
dispatch predicate is evaluated at the traced values to select the launch
sites. The physical fields, grid, shared bytes and the dispatch, resource,
range and TMA obligations come out at lowering, through the same binder
(torch/_inductor/runtime/_cudagraph/hosttrace_cute.py), as sympy relations
over the tape's symbols. The tensor operands' addresses are guarded at the
trace to the alignment the compiler assumes.

Declined by name: an entry that has not run once (trace(warm_up=True) runs it;
the recorder never compiles or observes a host itself), an entry whose
ordinary compilation's source observation declined (host structure beyond an
unconditional launch sequence or one two-arm dispatch), an argument the binder
does not take (a tensor from outside the trace, an integer derived from a data
pointer, a float), a conversion the runtime cannot trace, an invocation on
another device or stream than the trace's, a CuTe DSL launch that does not go
through a registered entry (its from_dlpack reads a traced tensor:
_TracedTensor.__dlpack__), a Python selection before the invocation that
reads tensor data (declines where it occurs), and a DirectTriton proxy (the
trace records the JITFunction's own launcher).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor


# a stand-in for the FX node the runtime's EntryCall names: the recorder has no
# graph, and the binder reads the node only by identity
_RECORDER_NODE = object()


def _host_trace() -> Any:
    from torch.cuda import _host_trace

    return _host_trace


@dataclass
class CuteTrace:
    """One trace's CuTe invocations: the capturing stream an invocation must be
    on, the records made (one per selected launch site), the roots the kernels
    may write, and the compiler templates bound in this trace."""

    stream: torch.cuda.Stream
    launches: list = field(default_factory=list)
    written_roots: list = field(default_factory=list)
    preparation: Any = (
        None  # cute_adapter.CuTePreparation, made at the first invocation
    )


@dataclass(frozen=True, eq=False)
class CuteInvocation:
    """One invocation of a registered observed CuTe entry under the trace, shared
    by the records of its launch sites: the adapter and its owner, the provenance
    object the binder reads (`local`), the entry call, the operands (the fake
    twins after the user's conversion, the integers), each twin's storage origin,
    the conversion trace, the ordinary compilation, the entry signature, the
    ordinary binding, the bound artifact, the arm the dispatch took at the traced
    values and its sites, and the alignment each tensor operand's address was
    guarded to at the trace."""

    adapter: Any
    owner: Any
    local: Any
    call: Any
    operands: tuple
    origins: dict  # id(twin) -> (root, offset in elements, dtype)
    conversion: Any
    compilation: Any
    signature: Any
    binding: Any
    artifact: Any
    arm: bool | None
    sites: tuple
    alignments: dict  # id(twin) -> guarded alignment in bytes
    read_only: frozenset = (
        frozenset()
    )  # tensor formals exported read-only: not written roots
    owner_provider: Any = None


@dataclass(frozen=True)
class CuteLaunch:
    """What a CuTe launch record carries beside the LaunchRec fields, under its
    "cute" key: the invocation and the artifact site the record stands for."""

    invocation: CuteInvocation
    site: Any
    site_index: int


class _NoIntegerInputs:
    """The tape has no boxed integer inputs: sizes are loads of the tensors."""

    integer_ranges: tuple = ()


class _RecorderInvocation:
    """The recorder-owned provenance the runtime's binder reads in place of its
    FX terminal's (_InvocationTrace): the trace's ShapeEnv and fake mode, the one
    entry call, no integer inputs, and a check that the adapter, its owner, the
    conversion trace and the operands are as recorded."""

    def __init__(
        self,
        adapter,
        owner,
        call,
        operands,
        converted,
        conversion,
        shape_env,
        fake_mode,
    ):
        from torch._inductor.runtime._cudagraph._compiler.entry_signature import (
            _number_state,
            _tensor_state,
        )

        self.adapter = adapter
        self.owner = owner
        self.call = call
        self.calls = (call,)
        self.operands = operands
        self.converted = converted  # the conversion's outputs, before specialization
        self.conversion = conversion
        self.shape_env = shape_env
        self.fake_mode = fake_mode
        self.input_contract = _NoIntegerInputs()
        # the ordinary borrow of the entry a lowering takes (the runtime's receipts
        # read it here); None until the tape is lowered
        self.borrow = None
        self._operand_state = tuple(
            _tensor_state(v) if type(v) is FakeTensor else _number_state(v)
            for v in operands
        )
        self._number_state, self._tensor_state = _number_state, _tensor_state

    def check(self):
        from torch._inductor.runtime._cudagraph.cute_adapter import CuTeDeclined

        self.adapter.check()
        self.owner.check()
        if self.borrow is not None:
            self.borrow.check()
        owner, call, operands = self.owner, self.call, self.operands
        conversion = self.conversion
        if conversion is None:
            if owner.conversion is not None:
                raise CuTeDeclined("CuTe invocation lost its user conversion trace")
        else:
            conversion.check()
            if (
                conversion.conversion is not owner.conversion
                or len(conversion.outputs) != len(operands)
                or any(
                    output.value is not operand
                    for output, operand in zip(conversion.outputs, self.converted)
                )
            ):
                raise CuTeDeclined(
                    "CuTe invocation lost its converted argument sources"
                )
        state = tuple(
            self._tensor_state(v) if type(v) is FakeTensor else self._number_state(v)
            for v in operands
        )
        values = (*call.arguments, *(value for _, value in call.keyword_arguments))
        paths = tuple(("args", index) for index in range(len(call.arguments)))
        paths += tuple(("kwargs", name) for name, _ in call.keyword_arguments)
        if (
            self.adapter.owner is not owner
            or call.entry is not owner.entry
            or call.target is not call.entry.target
            or call.config
            or call.entry_index != 0
            or call.node is not _RECORDER_NODE
            or len(values) != len(operands)
            or len(call.operands) != len(operands)
            or any(value is not original for value, original in zip(values, operands))
            or any(
                operand.value is not value
                or operand.path != path
                or operand.fx_argument is not value
                for operand, value, path in zip(call.operands, operands, paths)
            )
            or state != self._operand_state
        ):
            raise CuTeDeclined(
                "CuTe invocation lost its recorded operands or ordinary owner"
            )


class _Handler:
    """The recorder's handler on the runtime's dispatch seam (direct_invocation.
    ACTIVE) during the symbolic run: a DirectCuTe invocation is recorded, a
    DirectCudaHost entry runs under the trace mode as written, a DirectTriton
    proxy declines by name (the trace records the JITFunction's own launcher)."""

    def __init__(self, tr):
        self.trace = tr

    def cute(self, adapter, arguments):
        return _intercept(self.trace, adapter, arguments)

    def triton(self, adapter, arguments, *, grid, warmup, kwargs):
        name = getattr(getattr(adapter, "jit", None), "fn", None)
        name = getattr(name, "__name__", type(adapter).__name__)
        raise _host_trace().Declined(
            f"host_trace: Triton kernel {name}: launched through a DirectTriton proxy under a "
            "trace; the trace records the JITFunction's own launcher, kernel[grid](...) (declined)"
        )

    def cuda(self, adapter, arguments):
        return adapter.entry(*arguments)


@contextlib.contextmanager
def tracing(tr) -> Any:
    """The symbolic run: a registered observed CuTe entry invoked on this thread
    is recorded on `tr`. The runtime's dispatch seam carries the invocation to
    the recorder the way the runtime's mixed tracer takes it; it is restored
    when the run ends."""
    from torch._inductor.runtime._cudagraph.direct_invocation import activate

    tr.cute = CuteTrace(torch.cuda.current_stream(tr.device))
    with activate(_Handler(tr)):
        yield


def merge(tr, records: dict) -> None:
    """The trace's CuTe launch records into the recorder's records, in host order
    with the C++ launches; the roots they may write into written_roots."""
    ct = tr.cute
    if ct is None or not ct.launches:
        return
    _host_trace()._merge_launches(records, ct.launches)
    written = list(records["written_roots"])
    for name in ct.written_roots:
        if name not in written:
            written.append(name)
    records["written_roots"] = written


def _mentions_root(ht, tr, value) -> bool:
    if type(value) is not torch.SymInt:
        return False
    roots = {
        name
        for name in (
            [ht._symbol_name(i.root.sym) for i in tr.inputs]
            + [ht._symbol_name(a.q) for a in tr.allocs]
        )
        if name is not None
    }
    return bool(ht._free_symbols(value) & roots)


def _intercept(tr, adapter, arguments, *, read_only=frozenset(), name=None):
    ht = _host_trace()
    if getattr(ht._active, "trace", None) is not tr:
        # not the tracing thread: the runtime's ordinary path
        return adapter._invoke_ordinary(*arguments)
    from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import (
        CaptureIneligible,
        ObservedOrdinaryEntry,
    )
    from torch._inductor.runtime._cudagraph.direct_cute import DirectCuTe
    from torch.utils._python_dispatch import _disable_current_modes

    owner = getattr(adapter, "owner", None)
    target = getattr(getattr(owner, "entry", None), "target", None)
    if name is None:
        name = getattr(target, "__name__", type(adapter).__name__)

    def decline(why: str) -> Any:
        raise ht.Declined(f"host_trace: CuTe DSL kernel {name}: {why} (declined)")

    ct = tr.cute
    if type(adapter) is not DirectCuTe:
        decline(
            f"invoked through a {type(adapter).__name__}, not the runtime's DirectCuTe"
        )
    try:
        adapter.check()
    except RuntimeError as error:
        decline(f"the registered entry changed since its registration ({error})")
    if type(owner) is not ObservedOrdinaryEntry:
        decline(
            "its entry is not an ObservedOrdinaryEntry (the observed compilation the binder reads)"
        )
    if owner._selected is None or owner._owned_executor is None:
        decline(
            "the entry has not run once; trace(warm_up=True) runs the call as written first, and "
            "the owner's ordinary execution selects and observes its compilation"
        )
    device = tr.device
    if owner._device != device:
        decline(f"its entry ran on {owner._device}; the trace is on {device}")
    if torch.cuda.current_device() != device.index:
        decline(
            f"invoked on cuda:{torch.cuda.current_device()}; the trace is on {device}"
        )
    if torch.cuda.current_stream(device.index) != ct.stream:
        decline("invoked on a stream other than the trace's capturing stream")
    kinds, names = owner.argument_kinds, owner.argument_names
    if len(arguments) != len(kinds):
        decline(f"{len(arguments)} arguments for {len(kinds)} registered formals")
    twins: list = []
    origins: dict = {}
    for kind, formal, value in zip(kinds, names, arguments):
        if kind == "tensor":
            if type(value) is not ht._TracedTensor:
                decline(
                    f"tensor argument {formal} is a {type(value).__name__}, not a tensor of the trace"
                )
            if value.device != device:
                decline(f"tensor argument {formal} is on {value.device}")
            twin = value._fake
            origins[id(twin)] = (value._root, value._sym_offset, value.dtype)
            twins.append(twin)
            continue
        if type(value) is not int and type(value) is not torch.SymInt:
            decline(f"integer argument {formal} is a {type(value).__name__}")
        if type(value) is torch.SymInt and value.node.shape_env is not tr.shape_env:
            decline(f"integer argument {formal} belongs to another ShapeEnv")
        if _mentions_root(ht, tr, value):
            decline(
                f"integer argument {formal} is derived from a data pointer; the binder takes "
                "integers as values of the tape, not addresses"
            )
        twins.append(value)
    operands = tuple(twins)
    conversion = None
    with _disable_current_modes():
        if owner.conversion is not None:
            try:
                conversion = owner.conversion.trace(operands)
            except ValueError as error:
                decline(f"its argument conversion is not traceable ({error})")
            for event in conversion.events:
                source = origins.get(id(event.source))
                if source is None:
                    decline(
                        "its argument conversion viewed a tensor from outside the trace"
                    )
                root, offset, dtype = source
                origins[id(event.tensor)] = (root, offset + event.offset, dtype)
            operands = tuple(output.value for output in conversion.outputs)
            if any(type(v) is FakeTensor and id(v) not in origins for v in operands):
                decline("a converted tensor argument lost its storage origin")
        try:
            compilation = owner.compilation()
        except CaptureIneligible as error:
            decline(f"its ordinary compilation's source observation declined ({error})")
        except RuntimeError as error:
            decline(f"its ordinary compilation is not available ({error})")
        converted = operands
        operands = _specialize(
            tr, operands, compilation.metadata, names, origins, decline
        )
        invocation = _bind(
            tr,
            ct,
            adapter,
            owner,
            compilation,
            operands,
            converted,
            origins,
            conversion,
            decline,
            read_only,
        )
    _record(tr, ct, invocation, name)
    return None


def _specialize(tr, operands, metadata, names, origins, decline):
    """The operands as the compiled host takes them: a size or stride the
    compilation holds static (the user's conversion marked the axis static, the
    compiler baked the value in) is pinned by a guard of the tape and handed to
    the binder as that integer, on a twin of the same root, offset and dtype;
    shared dynamic symbols are unified only under an exact equality guard."""
    specialized = list(operands)
    symbols = {}
    parameters = tuple(
        row for row in metadata.params if row.kind not in ("Stream", "EnvStream")
    )
    for index, parameter in enumerate(parameters):
        if parameter.kind != "Tensor":
            continue
        twin = operands[index]
        if type(twin) is not FakeTensor:
            decline(
                f"tensor argument {names[index]} was converted to a {type(twin).__name__}"
            )
        sizes, strides = list(twin.shape), list(twin.stride())
        if len(sizes) != len(parameter.shape) or len(strides) != len(parameter.strides):
            decline(
                f"tensor argument {names[index]} has another rank than the compiled host's formal"
            )
        changed = False
        for values, dimensions, what in (
            (sizes, parameter.shape, "size"),
            (strides, parameter.strides, "stride"),
        ):
            for axis, (kind, value) in enumerate(dimensions):
                if kind == "symbol":
                    if value not in symbols:
                        symbols[value] = values[axis]
                        continue
                    expected = symbols[value]
                    if values[axis] is expected:
                        continue
                    contract = f"the compiled host shares compiler symbol {value}"
                elif kind == "constant" and type(values[axis]) is torch.SymInt:
                    expected = value
                    contract = f"the compiled host holds it static at {value}"
                else:
                    continue
                if not bool(values[axis] == expected):
                    decline(
                        f"tensor argument {names[index]}'s {what} {axis} is {values[axis]} at the traced call; "
                        f"{contract}"
                    )
                values[axis], changed = expected, True
        if not changed:
            continue
        elem = torch.empty_strided(sizes, strides, dtype=twin.dtype, device="meta")
        offset = twin.storage_offset()
        if not (isinstance(offset, int) and offset == 0):
            elem = elem.as_strided(sizes, strides, offset)
        pinned = FakeTensor(tr.fake_mode, elem, twin.device)
        origins[id(pinned)] = origins[id(twin)]
        specialized[index] = pinned
    return tuple(specialized)


def _bind(
    tr,
    ct,
    adapter,
    owner,
    compilation,
    operands,
    converted,
    origins,
    conversion,
    decline,
    read_only=frozenset(),
    *,
    binding_factory=None,
    artifact_factory=None,
    owner_provider=None,
):
    import sympy

    from torch._inductor.runtime._cudagraph._compiler.cute_bridge.lowering import (
        _Operands,
    )
    from torch._inductor.runtime._cudagraph._compiler.cute_bridge.numeric import (
        lower_numeric,
    )
    from torch._inductor.runtime._cudagraph._compiler.entry_signature import (
        build_entry_signature,
    )
    from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_reuse.binding import (
        bind_ordinary_metadata,
    )
    from torch._inductor.runtime._cudagraph._compiler.python_entry import (
        EntryCall,
        Operand,
    )
    from torch._inductor.runtime._cudagraph.cute_adapter import (
        _condition,
        CuTePreparation,
    )
    from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr

    ht = _host_trace()
    stream_index = next(
        index
        for index, row in enumerate(compilation.metadata.params)
        if row.name == owner.policy.stream_name
    )
    positional = operands[:stream_index]
    keywords = tuple(
        (row.name, value)
        for row, value in zip(
            compilation.metadata.params[stream_index + 1 :],
            operands[stream_index:],
            strict=True,
        )
    )
    call_operands = tuple(
        Operand(("args", i), value, value) for i, value in enumerate(positional)
    )
    call_operands += tuple(
        Operand(("kwargs", name), value, value) for name, value in keywords
    )
    call = EntryCall(
        0,
        owner.entry,
        owner.entry.target,
        owner.entry.config,
        _RECORDER_NODE,
        positional,
        keywords,
        call_operands,
    )
    local = _RecorderInvocation(
        adapter,
        owner,
        call,
        operands,
        converted,
        conversion,
        tr.shape_env,
        tr.fake_mode,
    )
    try:
        signature = build_entry_signature(
            local, call, policy=owner.policy, metadata=compilation.metadata
        )
        binding = (
            bind_ordinary_metadata(signature, compilation)
            if binding_factory is None
            else binding_factory(signature, owner)
        )
        if artifact_factory is None:
            if ct.preparation is None:
                ct.preparation = CuTePreparation()
            artifact = ct.preparation.bind(adapter, binding)
        else:
            artifact = artifact_factory(binding)
    except (ValueError, TypeError, RuntimeError) as error:
        decline(
            f"the runtime's binder does not express this invocation ({type(error).__name__}: {error})"
        )
    # the dispatch at the traced values: the compiler's recipes on the operands'
    # hints select the launch sites (the predicate as a relation over the tape's
    # symbols is a guard of the lowering)
    hints = _Operands(
        local,
        artifact,
        {id(v): None for v in operands if type(v) is FakeTensor},
        lambda v: IntExpr("constant", int(ht._hint(v))),
    )
    try:
        predicates = [
            c for c in artifact.consumers if c.site_id is None and c.role == "predicate"
        ]
        if artifact.sites and all(site.arm is None for site in artifact.sites):
            if predicates:
                decline("its unconditional artifact carries a dispatch predicate")
            arm, sites = None, tuple(artifact.sites)
        else:
            if len(predicates) != 1:
                decline("its artifact lacks one original dispatch predicate")
            decision = lower_numeric(predicates[0].numeric, hints.numeric)
            if len(decision.values) != 1:
                decline("its dispatch predicate has multiple results")
            condition = _condition(decision.values[0], lambda v: _constant(v, decline))
            if condition is sympy.true:
                arm = True
            elif condition is sympy.false:
                arm = False
            else:
                decline("its dispatch predicate has no value at the traced operands")
            sites = tuple(site for site in artifact.sites if site.arm is arm)
            if len(sites) != 1:
                decline("its dispatch does not select exactly one launch site")
    except ValueError as error:
        decline(f"its dispatch is not lowered by the runtime ({error})")
    # every tensor operand's address, guarded to the alignment the compiler
    # assumes (the policy's, or a stronger one the formal declares): a base
    # symbol plus the view's offset in bytes for an input, the offset alone for
    # an allocation (its base is 256-byte aligned by construction)
    alignments: dict = {}
    for formal in artifact.formals:
        if formal.kind != "Tensor":
            continue
        operand = signature.operands[formal.operand_index]
        twin = operand.tensor.value
        root, offset, dtype = origins[id(twin)]
        required = max(owner.policy.assumed_alignment, formal.data_alignment or 1)
        offset_bytes = offset * dtype.itemsize
        probe = offset_bytes if root.allocation else root.sym + offset_bytes
        if not bool(probe % required == 0):
            decline(
                f"tensor argument {operand.name} is not {required}-byte aligned at the traced "
                "call, which the compiled host assumes"
            )
        alignments[id(twin)] = required
    return CuteInvocation(
        adapter,
        owner,
        local,
        call,
        operands,
        origins,
        conversion,
        compilation,
        signature,
        binding,
        artifact,
        arm,
        sites,
        alignments,
        frozenset(read_only),
        owner_provider,
    )


def _constant(value, decline):
    # the hint evaluation's integers back as sympy numbers (the predicate folds)
    import sympy

    from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr

    if type(value) is int:
        return sympy.Integer(value)
    if type(value) is IntExpr and value.op == "constant" and not value.args:
        return sympy.Integer(value.value)
    return decline("its dispatch predicate did not fold at the traced operands")


def _record(tr, ct, invocation, name) -> None:
    """One launch record per selected site, at the invocation's sequence position."""
    import sympy

    from torch._inductor.runtime._cudagraph._compiler.cute_bridge.lowering import (
        _Operands,
    )
    from torch._inductor.runtime._cudagraph._compiler.cute_bridge.numeric import (
        lower_numeric,
        NumericSource,
    )
    from torch._inductor.runtime._cudagraph._compiler.cute_bridge.provider import (
        _constant_consumer,
    )
    from torch._inductor.runtime._cudagraph.address_scalars import symbolic_integer
    from torch._inductor.runtime._cudagraph.cute_adapter import _condition, CuTeDeclined
    from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr
    from torch.utils._sympy.numbers import int_oo

    ht = _host_trace()

    def decline(why: str) -> Any:
        raise ht.Declined(f"host_trace: CuTe DSL kernel {name}: {why} (declined)")

    artifact, signature = invocation.artifact, invocation.signature
    symbols = {}

    def require(condition):
        if not tr.shape_env.evaluate_expr(condition):
            decline(f"its launch configuration violates the integer domain {condition}")

    class SymbolicOperands(_Operands):
        def numeric_value(self, value):
            if type(value) is int:
                return super().numeric_value(value)
            if (
                type(value) is not torch.SymInt
                or value.node.shape_env is not tr.shape_env
            ):
                decline("its launch configuration lost its original symbolic integer")
            expression = value.node.expr
            interval = tr.shape_env.bound_sympy(expression)
            if not interval.is_int:
                decline("its launch configuration has no inherited integer domain")
            lower, upper = (
                None if bound in (-int_oo, int_oo) else int(bound)
                for bound in (interval.lower, interval.upper)
            )
            if lower is not None:
                require(sympy.Ge(expression, lower))
            if upper is not None:
                require(sympy.Le(expression, upper))
            index = len(symbols)
            symbols[index] = expression
            return NumericSource(IntExpr("boxed", index), lower, upper)

    operands = SymbolicOperands(
        invocation.local,
        artifact,
        {id(v): None for v in invocation.operands if type(v) is FakeTensor},
        lambda value: value,
    )
    formals = {formal.source_arg_index: formal for formal in artifact.formals}

    def symbolic(value):
        return symbolic_integer(value, symbols, CuTeDeclined)

    def lower(consumer):
        try:
            lowered = lower_numeric(consumer.numeric, operands.numeric)
            (value,) = lowered.values
        except ValueError as error:
            decline(f"its {consumer.role} is not lowered by the runtime ({error})")
        for obligation in lowered.obligations:
            expression = symbolic(obligation.expression)
            require(sympy.Ge(expression, obligation.lower))
            require(sympy.Le(expression, obligation.upper))
        return value

    if invocation.arm is not None:
        (predicate,) = (
            c for c in artifact.consumers if c.site_id is None and c.role == "predicate"
        )
        condition = _condition(lower(predicate), symbolic)
        require(condition if invocation.arm else sympy.Not(condition))

    def numeric(site, role, index):
        consumer = next(
            c
            for c in artifact.consumers
            if c.site_id == site.site_id and c.role == role and c.index == index
        )
        expression = symbolic(lower(consumer).expression)
        return tr.shape_env.create_symintnode(expression, hint=None)

    written = []
    for site_index, site in enumerate(invocation.sites):
        # the parameters' natural layout (the loaded kernel's is read at lowering)
        layout, cursor = [], 0
        for parameter in site.parameters:
            cursor = (
                (cursor + parameter.alignment - 1)
                // parameter.alignment
                * parameter.alignment
            )
            layout.append((cursor, parameter.size))
            cursor += parameter.size
        params = []
        for row in (*site.fields.pointers, *site.fields.integers):
            source = row.source
            dtype = getattr(row, "dtype", None)
            if source.kind == "compiler_constant":
                value, kind, label = (
                    source.constant,
                    dtype,
                    f"<{source.llvm_type} constant>",
                )
            elif source.kind in ("tensor_property", "scalar_formal"):
                formal = formals[source.ir_arg_index]
                operand = signature.operands[formal.operand_index]
                if formal.kind == "Var":
                    value, kind, label = operand.scalar.use.value, dtype, formal.name
                elif source.property == "pointer":
                    root, offset, tensor_dtype = invocation.origins[
                        id(operand.tensor.value)
                    ]
                    value, kind = root.sym + offset * tensor_dtype.itemsize, "ptr"
                    label = f"{formal.name}.data_ptr"
                    if (
                        formal.name not in invocation.read_only
                        and root.name not in written
                    ):
                        written.append(root.name)
                else:
                    (axis,) = source.property_path
                    uses = (
                        operand.tensor.shape
                        if source.property == "shape"
                        else operand.tensor.strides
                    )
                    value, kind = uses[axis].value, dtype
                    label = f"{formal.name}.{source.property}[{axis}]"
            else:
                # a compiler expression (a TMA descriptor word, a computed pointer):
                # the binder lowers it from the operands at preparation
                continue
            params.append(
                {
                    "offset": layout[row.parameter][0] + row.byte_offset,
                    "size": 8 if kind in ("ptr", "i64") else 4,
                    "kind": kind,
                    "value": value,
                    "name": label,
                    "access": ("r" if label[:-9] in invocation.read_only else "rw")
                    if kind == "ptr"
                    else "",
                }
            )
        image = bytearray(layout[-1][0] + layout[-1][1] if layout else 0)
        for parameter, offset, data in site.fields.constants:
            start = layout[parameter][0] + offset
            image[start : start + len(data)] = data
        for p in params:
            image[p["offset"] : p["offset"] + p["size"]] = ht._pack(
                p["kind"], ht._hint(p["value"])
            )
        block = tuple(
            _constant_consumer(artifact, site, "block", axis) for axis in range(3)
        )
        ct.launches.append(
            {
                "seq": tr.rec.next_seq(),
                "kernel": site.registration.kernel_symbol,
                # the function handle is the lowering's loaded copy of the kernel
                "func": None,
                "param_layout": layout,
                "grid": tuple(numeric(site, "grid", axis) for axis in range(3)),
                "block": block,
                "block_expr": block,
                "smem": numeric(site, "shared", 0),
                "params": params,
                "hint_image": bytes(image),
                "cute": CuteLaunch(invocation, site, site_index),
            }
        )
    for root in written:
        if root not in ct.written_roots:
            ct.written_roots.append(root)
