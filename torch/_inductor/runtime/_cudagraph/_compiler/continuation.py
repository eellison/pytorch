from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.accessors import _function, _tagged_arguments
from torch._inductor.runtime._cudagraph._compiler.cfg_values import CFGProgram, read_cfg_function
from torch._inductor.runtime._cudagraph._compiler.compiler_boundary import _snapshot
from torch._inductor.runtime._cudagraph._compiler.compiler_owner import TaggedProgram, _TaggedCollector
from torch._inductor.runtime._cudagraph._compiler.dispatch_join import JoinedDispatch
from torch._inductor.runtime._cudagraph._compiler.entry_signature import snapshot_metadata
from torch._inductor.runtime._cudagraph._compiler.helpers import DispatchHelpers, DispatchScalarHelper, ScalarRequest, emit_dispatch_helpers
from torch._inductor.runtime._cudagraph._compiler.source_dispatch import SourceLaunchSite, check_dispatch_source


CONTINUATION_VERSION = 1


def _uses(source: Any) -> tuple[tuple[Any, str, int, Any], ...]:
    uses = [] if source.predicate is None else [(None, "predicate", 0, source.predicate)]
    for site in source.sites:
        for role, values in (("grid", site.grid), ("block", site.block_dims), ("shared", (site.shared,)),
                             ("diagnostic", tuple(item.predicate for item in site.diagnostics)),
                             ("kernel_smem", (site.diagnostics[0].kernel_query.results[0],)),
                             ("tma_stride", tuple(item.tensor for item in site.tma_strides))):
            uses.extend((site, role, index, value) for index, value in enumerate(values))
        for index, requirement in enumerate(site.tma_dimensions):
            uses.append((site, "tma_shape", index, requirement.tensor))
            if requirement.grouped:
                uses.append((site, "tma_dimension_stride", index, requirement.tensor))
    return tuple(uses)


def _metadata_signature(metadata: Any) -> tuple[Any, ...]:
    return metadata.symbol_name, metadata.params, metadata.symbols, metadata.ret


class _DispatchCollector(_TaggedCollector):
    def __init__(self, owner: Any, signature: Any, arguments: tuple[Any, ...], keywords: dict[str, Any]) -> None:
        super().__init__(owner)
        self.signature = signature
        self.arguments = arguments
        self.keywords = keywords
        self.helpers: DispatchHelpers | None = None
        self.original_text = ""
        self.original_bytecode = b""

    def __call__(self, owner: Any, module: Any, function_name: str) -> None:
        from cutlass._mlir import ir
        from cutlass.cute.metadata import build_function_metadata

        super().__call__(owner, module, function_name)
        metadata = build_function_metadata(function_name=function_name, signature=self.signature,
                                           args=self.arguments, kwonlyargs=self.keywords)
        with module.context, ir.raw_values():
            self.original_text, self.original_bytecode = _snapshot(module)
            source = check_dispatch_source(module, function_name, snapshot_metadata(metadata))
            prefix = "__cudagraph_dispatch_" + sha256(function_name.encode()).hexdigest()[:16]
            requests = tuple(ScalarRequest(f"{prefix}_{number}", role, index, value, site)
                             for number, (site, role, index, value) in enumerate(_uses(source)))
            self.helpers = emit_dispatch_helpers(source, requests)
            self.text, self.bytecode = _snapshot(module)


@dataclass(frozen=True)
class BoundDispatchHelper:
    joined: JoinedDispatch = field(repr=False)
    helper: DispatchScalarHelper = field(repr=False)
    site: SourceLaunchSite | None = field(repr=False)
    role: str
    index: int
    source_value: Any = field(repr=False)
    cfg: CFGProgram = field(repr=False)
    source_order: tuple[int, ...]
    _seal: tuple[Any, ...] = field(repr=False)

    def _state(self) -> tuple[Any, ...]:
        return (id(self.joined), id(self.helper), id(self.site), self.role, self.index,
                id(self.source_value), id(self.cfg), self.source_order)

    def check(self) -> None:
        if self._state() != self._seal:
            raise RuntimeError("Dispatch helper consumer ownership changed")
        joined = self.joined
        joined.check()
        self._check_body(joined)

    def _check_body(self, joined: JoinedDispatch) -> None:
        from cutlass._mlir import ir

        if self.joined is not joined or self._state() != self._seal:
            raise RuntimeError("Dispatch helper consumer ownership changed")
        self.helper.check()
        self.cfg.check()
        source = joined.admission.source
        program = joined.mapping.program
        helper = self.helper
        if (helper.module is not program.source_module or self.cfg.module is not program.module
                or self.cfg.context is not program.context or helper.symbol != self.cfg.function_name
                or (helper.result_type,) != self.cfg.result_types
                or helper.role != self.role or helper.index != self.index
                or helper.source.host != source.host or helper.input_origin != source.arguments
                or helper.source_types != source.source_types
                or _metadata_signature(helper.source.metadata) != _metadata_signature(source.metadata)):
            raise ValueError("Compiled helper differs from its actual source owner, signature or result")
        if self.site is None:
            if helper.site is not None:
                raise ValueError("Root predicate helper changed its source scope")
        elif (not any(self.site is site for site in source.sites) or helper.site is None
                or helper.site.launch != self.site.launch or helper.site.kernel != self.site.kernel
                or helper.site.callee != self.site.callee):
            raise ValueError("Compiled helper changed its exact dispatch arm")
        with program.source_context, ir.raw_values():
            matches = [value for site, role, index, value in _uses(source)
                       if site is self.site and role == self.role and index == self.index]
            if len(matches) != 1 or matches[0] != self.source_value or helper.output_value != self.source_value:
                raise ValueError("Helper label does not identify its exact original consumer SSA value")
            original = _tagged_arguments(source.host)
            helper_arguments = _tagged_arguments(helper.operation)
            if (tuple(helper_arguments) != helper.source_ids or set(helper_arguments) != set(original)
                    or any(helper_arguments[index].type != original[index].type for index in original)):
                raise ValueError("Source helper changed an original formal ID or type")
        with program.context, ir.raw_values():
            host = _tagged_arguments(_function(program.module, program.function_name, "llvm.func"))
            lowered = _tagged_arguments(_function(program.module, helper.symbol, "llvm.func"))
            if (set(lowered) != set(host) or set(lowered) != set(helper.source_ids)
                    or tuple(lowered) != self.source_order
                    or tuple(str(value.type) for value in lowered.values()) != self.cfg.argument_types
                    or any(lowered[index].type != host[index].type for index in host)):
                raise ValueError("Lowered helper lost its original formal IDs or exact LLVM types")


def check_dispatch_consumers(joined: JoinedDispatch, consumers: tuple[BoundDispatchHelper, ...]) -> None:
    if (type(joined) is not JoinedDispatch or type(consumers) is not tuple
            or any(type(consumer) is not BoundDispatchHelper for consumer in consumers)):
        raise TypeError("Expected an exact joined dispatch and consumer tuple")
    if any(type(consumer.helper) is not DispatchScalarHelper or type(consumer.cfg) is not CFGProgram
           for consumer in consumers):
        raise TypeError("Expected exact scalar helper and CFG owners")
    seals = tuple(consumer._seal for consumer in consumers)
    joined.check()
    for consumer in consumers:
        BoundDispatchHelper._check_body(consumer, joined)
    joined.check()
    for consumer, seal in zip(consumers, seals):
        if consumer.joined is not joined or consumer._seal is not seal or consumer._state() != seal:
            raise RuntimeError("Dispatch helper consumer ownership changed")


def bind_dispatch_consumers(joined: JoinedDispatch, helpers: DispatchHelpers) -> tuple[BoundDispatchHelper, ...]:
    from cutlass._mlir import ir

    if type(joined) is not JoinedDispatch or type(helpers) is not DispatchHelpers:
        raise TypeError("Expected owned source/compiled dispatch and emitted helpers")
    joined.check()
    helpers.check()
    source, program = joined.admission.source, joined.mapping.program
    if helpers.source.module is not program.source_module or helpers.source.host != source.host:
        raise ValueError("Helpers belong to a different original source Module")
    bound, used = [], set()
    with program.context, ir.raw_values():
        for helper in helpers.helpers:
            site = None
            if helper.site is not None:
                matches = [item for item in source.sites if item.launch == helper.site.launch]
                if len(matches) != 1:
                    raise ValueError("Helper scope does not identify an admitted source site")
                site = matches[0]
            key = (None if site is None else site.launch, helper.role, helper.index)
            if key in used:
                raise ValueError("Duplicate helper for an original scalar consumer")
            used.add(key)
            cfg = read_cfg_function(program.module, helper.symbol)
            order = tuple(_tagged_arguments(_function(program.module, helper.symbol, "llvm.func")))
            item = BoundDispatchHelper(joined, helper, site, helper.role, helper.index, helper.output_value, cfg, order, ())
            item = replace(item, _seal=item._state())
            bound.append(item)
    with program.source_context, ir.raw_values():
        expected = {(None if site is None else site.launch, role, index) for site, role, index, _ in _uses(source)}
    if used != expected:
        raise ValueError("Dispatch helpers lack complete original consumer coverage")
    bound = tuple(bound)
    check_dispatch_consumers(joined, bound)
    return bound


@dataclass(frozen=True)
class DispatchCompilation:
    program: TaggedProgram
    helpers: DispatchHelpers
    joined: JoinedDispatch
    consumers: tuple[BoundDispatchHelper, ...]
    original_text: str = field(repr=False)
    original_bytecode: bytes = field(repr=False)
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        from cutlass._mlir import ir

        owned = self.program, self.helpers, self.joined, self.consumers, self.original_text, self.original_bytecode
        if len(owned) != len(self._owners) or any(actual is not owner for actual, owner in zip(owned, self._owners)):
            raise RuntimeError("Dispatch compilation ownership changed")
        self.program.check()
        self.helpers.check()
        self.joined.check()
        if (type(self.program) is not TaggedProgram or self.joined.mapping.program is not self.program
                or self.helpers.source.module is not self.program.source_module
                or len(self.consumers) != len(self.helpers.helpers)):
            raise ValueError("Dispatch compilation lost its actual compiler or helper owner")
        seen = set()
        with self.program.source_context, ir.raw_values():
            expected = {(None if site is None else site.launch, role, index)
                        for site, role, index, _ in _uses(self.joined.admission.source)}
        for bound, helper in zip(self.consumers, self.helpers.helpers):
            if bound.joined is not self.joined or bound.helper is not helper:
                raise ValueError("Bound dispatch helper changed its collection owner")
            bound.check()
            seen.add((None if bound.site is None else bound.site.launch, bound.role, bound.index))
        if seen != expected or len(seen) != len(self.consumers):
            raise ValueError("Dispatch compilation lost complete scalar consumer coverage")
