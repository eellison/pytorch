"""Export local host predicates and compile them for the native boxed dispatcher."""

import ctypes
from dataclasses import dataclass

import sympy

import torch
from torch._dynamo.source import LocalSource, TensorProperty, TensorPropertySource
from torch._guards import TracingContext
from torch._inductor.codecache import CppCodeCache
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    FXTraceDeclined,
    NormalizeEvent,
)
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    InputSource,
    IntExpr,
)
from torch._inductor.runtime.cudagraph_compiled_evaluation import STRICT_FLOAT_FLAGS
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.fx.experimental.symbolic_shapes import _CppShapeGuardsHelper
from torch.utils._sympy.value_ranges import ValueRanges

from .address_guard_printer import (
    _AddressPrinter,
    _BoundedPrinter,
    GuardExportDeclined as _PrinterDeclined,
)
from .extraction import ComputedIntegerBinding, TerminalTrace
from .host_trace_guards import (
    _emit_ordered_guards,
    _GuardCallback,
    _ordered_guard_source,
    _OrderedPrinter,
    prove_owned_guard,
)


class GuardExportDeclined(FXTraceDeclined):
    pass


@dataclass(frozen=True)
class GuardExpressions:
    expressions: tuple[str, ...]
    bindings: tuple[tuple[int, str], ...]
    discharged: tuple[sympy.Basic, ...]
    trace: TerminalTrace
    pointer_bindings: tuple[tuple[int, str], ...] = ()
    storage_offset_bindings: tuple[tuple[int, str], ...] = ()
    statements: tuple[str, ...] = ()
    callback_owners: tuple[object, ...] = ()


@dataclass(frozen=True)
class PreparedGuard:
    boxed_integer_indices: tuple[int, ...]
    function_address: int
    library_owner: object
    expressions: tuple[str, ...]
    cpp_source: str
    program: object
    boxed_pointer_indices: tuple[int, ...] = ()
    boxed_storage_offset_indices: tuple[int, ...] = ()

    @property
    def registration(self):
        result = self.boxed_integer_indices, self.function_address, self.library_owner
        if self.boxed_storage_offset_indices:
            return (
                *result,
                self.boxed_pointer_indices,
                self.boxed_storage_offset_indices,
            )
        return (
            (*result, self.boxed_pointer_indices)
            if self.boxed_pointer_indices
            else result
        )


class _TerminalPrinter(_BoundedPrinter):
    def __init__(
        self,
        *args,
        sources,
        source_domains,
        symbols,
        late_domains=None,
        owned=None,
        owned_domains=None,
    ):
        super().__init__(*args, domains={}, sources=sources)
        self.source_domains = source_domains
        self.original_symbols = symbols
        self.bound_sources = {}
        self.late_domains = {} if late_domains is None else late_domains
        self.owned = {} if owned is None else owned
        self.owned_domains = {} if owned_domains is None else owned_domains

    def doprint(self, expr):
        if not expr.free_symbols.issubset(self.original_symbols):
            raise GuardExportDeclined(
                "Guard arithmetic has an unresolved symbolic source"
            )
        for symbol in expr.free_symbols:
            if symbol in self.late_domains:
                self.domains[symbol] = self.late_domains[symbol]
                continue
            choices = self.symbol_to_source.get(symbol) or self.var_to_sources.get(
                symbol
            )
            if not choices or id(choices[0]) not in self.source_domains:
                raise GuardExportDeclined(
                    "Guard symbol has no exact original boxed source"
                )
            source = choices[0]
            if (
                symbol in self.bound_sources
                and self.bound_sources[symbol] is not source
            ):
                raise GuardExportDeclined("Guard source changed during export")
            self.bound_sources[symbol] = source
            # ShapeEnv can print an equality's surviving symbol through an alias.
            self.domains[symbol] = self.source_domains[id(source)]
        if expr.free_symbols.intersection(self.late_domains):
            terms = expr.args if isinstance(expr, sympy.And) else (expr,)
            pending = []
            for term in terms:
                if not term.free_symbols.intersection(self.late_domains):
                    pending.append(term)
                    continue
                if term.free_symbols.intersection(self.owned):
                    proof = _OrderedPrinter(
                        {},
                        lambda source: source.name(),
                        {},
                        domains={**self.domains, **self.owned_domains},
                        sources=(),
                    )
                    prove_owned_guard(proof, term, self.owned)
                else:
                    self.check_arithmetic(term)
                    bounds = self.bounds(term)
                    if bounds.lower is not sympy.true or bounds.upper is not sympy.true:
                        raise GuardExportDeclined(
                            "Allocation address predicate is not proven before allocation"
                        )
                self.discharged.append(term)
            expr = sympy.And(*pending)
        return super().doprint(expr)


class _AddressTerminalPrinter(_TerminalPrinter, _AddressPrinter):
    pass


def export_guards(trace, extra=()):
    if type(trace) is not TerminalTrace or TracingContext.try_get() is not None:
        raise GuardExportDeclined(
            "Expected an exact terminal trace outside an ambient compiler context"
        )
    if type(extra) is not tuple or any(
        not isinstance(expr, sympy.Basic) for expr in extra
    ):
        raise GuardExportDeclined(
            "Additional dispatch guards require exact symbolic expressions"
        )
    trace.compiler_binding.check()
    rows = trace.contract.integer_ranges
    placeholders, sources = trace.integer_placeholders, trace.integer_sources
    environment = trace.shape_env
    address_symbols = {binding.symbol for binding in trace.address_bindings}
    offset_symbols = {binding.symbol for binding in trace.storage_offset_bindings}
    computed = trace.computed_integer_bindings
    if any(type(binding) is not ComputedIntegerBinding for binding in computed):
        raise GuardExportDeclined("Computed integer requires its exact traced record")
    computed_symbols = {binding.symbol for binding in computed}
    original_symbols = (
        frozenset(trace.symbol_sources)
        | address_symbols
        | offset_symbols
        | computed_symbols
    )
    if (
        len(rows) != len(placeholders)
        or len(rows) != len(sources)
        or len({row.index for row in rows}) != len(rows)
        or len({id(source) for source in sources}) != len(sources)
        or environment.deferred_runtime_asserts
        or not set(environment.var_to_range).issubset(original_symbols)
    ):
        raise GuardExportDeclined(
            "Local guard inputs or assertion ownership are incomplete"
        )
    declared, symbols = {row.index: row for row in rows}, {}
    for symbol, origin in trace.symbol_sources.items():
        if (
            type(symbol) is not sympy.Symbol
            or type(origin) is not IntExpr
            or origin.op != "boxed"
            or origin.args
            or origin.value not in declared
            or origin.value in symbols
        ):
            raise GuardExportDeclined(
                "Local guard symbol lacks one original boxed source"
            )
        symbols[origin.value] = symbol
    source_domains = {}
    for row, placeholder, source in zip(rows, placeholders, sources, strict=True):
        if (
            type(placeholder) is not torch.SymInt
            or placeholder.node.shape_env is not environment
            or row.index not in symbols
            or not any(
                source is original
                for original in environment.var_to_sources[symbols[row.index]]
            )
        ):
            raise GuardExportDeclined("Local guard lost its exact ShapeEnv source")
        source_domains[id(source)] = ValueRanges(
            max(row.lower, -(2**63)),
            min(row.upper if row.upper is not None else 2**63 - 1, 2**63 - 1),
        )
    if len(computed_symbols) != len(computed):
        raise GuardExportDeclined(
            "Computed integers must retain distinct symbolic identities"
        )
    for binding in computed:
        if (
            type(binding) is not ComputedIntegerBinding
            or type(binding.value) is not torch.SymInt
            or binding.value.node.shape_env is not environment
            or binding.value.node._expr is not binding.symbol
            or type(binding.symbol) is not sympy.Symbol
            or binding.symbol in trace.symbol_sources
            or binding.symbol in address_symbols | offset_symbols
            or type(binding.source) is not LocalSource
            or not any(
                binding.source is source
                for source in environment.var_to_sources.get(binding.symbol, ())
            )
            or binding.kind not in ("guard", "rebind")
            or type(binding.expected) is not int
            or not -(2**63) <= binding.expected < 2**63
            or type(binding.address) is not int
            or not 0 < binding.address < 2**64
            or binding.owner is None
            or type(binding.arguments) is not tuple
            or any(not isinstance(arg, sympy.Expr) for arg in binding.arguments)
        ):
            raise GuardExportDeclined(
                "Computed integer lost its exact recorded callback or symbolic source"
            )
        source_domains[id(binding.source)] = ValueRanges(-(2**63), 2**63 - 1)
    tensor_indices = {row.index for row in trace.contract.tensor_inputs}
    offset_sources = []
    for binding in trace.storage_offset_bindings:
        if (
            binding.index not in tensor_indices
            or type(binding.symbol) is not sympy.Symbol
            or binding.symbol in trace.symbol_sources
            or binding.symbol in address_symbols
            or type(binding.source) is not TensorPropertySource
            or binding.source.prop is not TensorProperty.STORAGE_OFFSET
            or type(binding.source.base) is not LocalSource
            or binding.source.base.local_name != f"boxed_{binding.index}"
            or type(binding.value) is not torch.SymInt
            or binding.value.node.shape_env is not environment
            or binding.symbol not in environment.var_to_range
            or not any(
                binding.source is source
                for source in environment.var_to_sources[binding.symbol]
            )
        ):
            raise GuardExportDeclined(
                "Storage-offset predicate lost its original Tensor metadata source"
            )
        offset_sources.append((binding.index, binding.source))
        source_domains[id(binding.source)] = ValueRanges(0, 2**63 - 1)
    if len(offset_symbols) != len(trace.storage_offset_bindings) or len(
        {index for index, _ in offset_sources}
    ) != len(offset_sources):
        raise GuardExportDeclined(
            "Storage-offset sources must retain distinct input identities"
        )
    address_sources, input_addresses = [], []
    address_roots, late_domains = set(), {}
    owned, owned_domains = {}, {}
    allocated_roots = (
        set()
        if trace.tensor_roots is None
        else {resolution.root for resolution in trace.tensor_roots.allocations}
    )
    for binding in trace.address_bindings:
        if binding.generation != 0:
            raise GuardExportDeclined(
                "Normalized address generations require distinct traced tensor identities"
            )
        if (
            type(binding.symbol) is not sympy.Symbol
            or binding.symbol in trace.symbol_sources
            or binding.root in address_roots
            or type(binding.value) is not torch.SymInt
            or binding.value.node.shape_env is not environment
            or binding.symbol not in environment.var_to_range
        ):
            raise GuardExportDeclined(
                "Local address guard lost its exact ShapeEnv source"
            )
        address_roots.add(binding.root)
        if type(binding.root) is BufferSource and binding.root in allocated_roots:
            if binding.alignment != 256:
                raise GuardExportDeclined(
                    "Allocation address lacks its allocator alignment"
                )
            actual_range = environment.var_to_range[binding.symbol]
            if actual_range.lower > 0 or actual_range.upper < 2**64 - 1:
                raise GuardExportDeclined(
                    "Allocation address range was narrowed without a replay predicate"
                )
            late_domains[binding.symbol] = ValueRanges(0, 2**64 - 1)
            quotient = sympy.Dummy(integer=True, nonnegative=True)
            owned[binding.symbol] = binding.alignment * quotient
            owned_domains[quotient] = ValueRanges(0, (2**64 - 1) // binding.alignment)
            continue
        if (
            type(binding.root) is not InputSource
            or binding.root.index not in tensor_indices
            or not any(
                binding.source is original
                for original in environment.var_to_sources[binding.symbol]
            )
        ):
            raise GuardExportDeclined(
                "Address predicate lost its original input source"
            )
        address_sources.append((binding.root.index, binding.source))
        input_addresses.append(binding)
        source_domains[id(binding.source)] = ValueRanges(0, 2**64 - 1)
    placeholders = (
        *placeholders,
        *(binding.value for binding in input_addresses),
        *(binding.value for binding in trace.storage_offset_bindings),
        *(binding.value for binding in computed),
    )
    sources = (
        *sources,
        *(binding.source for binding in input_addresses),
        *(binding.source for binding in trace.storage_offset_bindings),
        *(binding.source for binding in computed),
    )
    if len({id(source) for source in sources}) != len(sources) or len(
        address_symbols
    ) != len(address_roots):
        raise GuardExportDeclined(
            "Local guard sources must retain distinct input identities"
        )
    normalized_roots = {
        trace.tensor_roots.event(event).root
        for event in trace.events
        if type(event) is NormalizeEvent
    }
    normalized_offsets = {
        binding.symbol
        for binding in trace.storage_offset_bindings
        if InputSource(binding.index) in normalized_roots
    }
    if late_domains or normalized_offsets:
        if any(
            environment.var_to_range[symbol].lower > 0
            or environment.var_to_range[symbol].upper < 2**63 - 1
            for symbol in normalized_offsets
        ):
            raise GuardExportDeclined(
                "Normalized storage-offset range was narrowed without a represented generation"
            )
        if any(
            symbol in late_domains or expression.free_symbols.intersection(late_domains)
            for symbol, expression in environment.replacements.items()
        ):
            raise GuardExportDeclined(
                "Allocation address substitution needs a replay predicate"
            )
        if any(
            symbol in normalized_offsets
            or expression.free_symbols.intersection(normalized_offsets)
            for symbol, expression in environment.replacements.items()
        ):
            raise GuardExportDeclined(
                "Storage-offset substitution crosses a normalization generation"
            )
        original_sources = {
            symbols[row.index]: [source]
            for row, source in zip(rows, trace.integer_sources, strict=True)
        }
        original_sources.update(
            {binding.symbol: [binding.source] for binding in input_addresses}
        )
        original_sources.update(
            {
                binding.symbol: [binding.source]
                for binding in trace.storage_offset_bindings
            }
        )
        original_sources.update(
            {binding.symbol: [binding.source] for binding in computed}
        )
        unavailable_domains = {
            **late_domains,
            **{symbol: ValueRanges(0, 2**63 - 1) for symbol in normalized_offsets},
        }
        proof = _AddressTerminalPrinter(
            original_sources,
            lambda source: source.name(),
            original_sources,
            sources=sources,
            source_domains=source_domains,
            symbols=original_symbols,
            late_domains=unavailable_domains,
            owned=owned,
            owned_domains=owned_domains,
        )
        try:
            for predicate in (*(guard.expr for guard in environment.guards), *extra):
                if predicate.free_symbols.intersection(unavailable_domains):
                    try:
                        proof.doprint(predicate)
                    except (GuardExportDeclined, _PrinterDeclined) as error:
                        if predicate.free_symbols.intersection(normalized_offsets):
                            raise GuardExportDeclined(
                                "Storage-offset guard crosses an unrepresented normalization generation"
                            ) from error
                        raise
        except _PrinterDeclined as error:
            raise GuardExportDeclined(str(error)) from error
    if computed:
        return _export_computed_guards(
            trace,
            extra,
            sources,
            source_domains,
            input_addresses,
            offset_sources,
            late_domains,
            normalized_offsets,
            owned,
            owned_domains,
        )
    printers = []

    def factory(*args):
        printer_type = (
            _AddressTerminalPrinter
            if trace.address_bindings or trace.storage_offset_bindings
            else _TerminalPrinter
        )
        printer = printer_type(
            *args,
            sources=sources,
            source_domains=source_domains,
            symbols=original_symbols,
            late_domains=late_domains,
            owned=owned,
            owned_domains=owned_domains,
        )
        printers.append(printer)
        return printer

    try:
        (parts,) = environment.produce_guards_verbose(
            placeholders,
            sources,
            langs=("cpp",),
            _simplified=False,
            ignore_static=False,
            _cpp_printer_factory=factory,
        )
    except _PrinterDeclined as error:
        raise GuardExportDeclined(str(error)) from error
    if type(parts) is not _CppShapeGuardsHelper or len(printers) != 1:
        raise GuardExportDeclined(
            "Expected one complete compiler-produced local predicate"
        )
    expressions = [expr for expr in parts.exprs if expr != "true"]
    try:
        for expr in extra:
            printed = printers[0].doprint(expr)
            if printed != "true" and printed not in expressions:
                expressions.append(printed)
    except _PrinterDeclined as error:
        raise GuardExportDeclined(str(error)) from error
    bindings, pointer_bindings, metadata_bindings = [], [], []
    for source, symbol in printers[0].source_to_symbol.items():
        matches = [
            (bindings, row.index)
            for row, original in zip(rows, trace.integer_sources, strict=True)
            if source is original
        ]
        matches.extend(
            (pointer_bindings, index)
            for index, original in address_sources
            if source is original
        )
        matches.extend(
            (metadata_bindings, index)
            for index, original in offset_sources
            if source is original
        )
        if (
            len(matches) != 1
            or not symbol.name.isascii()
            or not symbol.name.isidentifier()
        ):
            raise GuardExportDeclined(
                "Printed predicate lost its original boxed correspondence"
            )
        target, index = matches[0]
        target.append((index, symbol.name))
    trace.compiler_binding.check()
    return GuardExpressions(
        tuple(expressions),
        tuple(bindings),
        tuple(printers[0].discharged),
        trace,
        tuple(pointer_bindings),
        tuple(metadata_bindings),
    )


class _ComputedGuardPrinter(_OrderedPrinter):
    def print_source(self, source):
        return _BoundedPrinter.print_source(self, source)


def _export_computed_guards(
    trace,
    extra,
    sources,
    source_domains,
    input_addresses,
    offset_sources,
    late_domains,
    normalized_offsets,
    owned,
    owned_domains,
):
    environment = trace.shape_env
    computed = trace.computed_integer_bindings
    symbol_sources = {
        placeholder.node._expr: [source]
        for placeholder, source in zip(
            trace.integer_placeholders, trace.integer_sources, strict=True
        )
    }
    symbol_sources.update(
        {binding.symbol: [binding.source] for binding in input_addresses}
    )
    symbol_sources.update(
        {binding.symbol: [binding.source] for binding in trace.storage_offset_bindings}
    )
    symbol_sources.update({binding.symbol: [binding.source] for binding in computed})
    domains = {
        symbol: source_domains[id(values[0])]
        for symbol, values in symbol_sources.items()
    }
    domains.update(late_domains)
    domains.update(owned_domains)
    printer = _ComputedGuardPrinter(
        symbol_sources,
        lambda source: source.name(),
        symbol_sources,
        domains=domains,
        sources=sources,
    )
    predicates = []
    for symbol, bounds in environment.var_to_range.items():
        for bound, relation in ((bounds.lower, sympy.Ge), (bounds.upper, sympy.Le)):
            if isinstance(bound, sympy.Integer):
                predicates.append(relation(symbol, bound, evaluate=False))
    predicates.extend(guard.expr for guard in environment.guards)
    predicates.extend(
        sympy.Eq(symbol, value, evaluate=False)
        for symbol, value in environment.replacements.items()
    )
    predicates.extend(extra)
    if normalized_offsets:
        proof = _AddressTerminalPrinter(
            symbol_sources,
            lambda source: source.name(),
            symbol_sources,
            sources=sources,
            source_domains=source_domains,
            symbols=frozenset(domains),
            late_domains={
                **late_domains,
                **{symbol: domains[symbol] for symbol in normalized_offsets},
            },
            owned=owned,
            owned_domains=owned_domains,
        )
        pending = []
        for predicate in predicates:
            if predicate.free_symbols.intersection(normalized_offsets):
                if proof.doprint(predicate) != "true":
                    raise GuardExportDeclined(
                        "Storage-offset guard crosses an unrepresented normalization generation"
                    )
            else:
                pending.append(predicate)
        predicates = pending
    callbacks = tuple(
        _GuardCallback(
            binding.symbol,
            binding.arguments,
            binding.address,
            binding.owner,
            binding.kind,
            binding.expected,
        )
        for binding in computed
    )
    if any(
        arg.free_symbols.intersection(normalized_offsets)
        for binding in computed
        for arg in binding.arguments
    ):
        raise GuardExportDeclined(
            "Computed integer uses an unrepresented normalized storage offset"
        )
    try:
        expressions, statements = _emit_ordered_guards(
            printer,
            symbol_sources,
            predicates,
            callbacks,
            (symbol_sources.keys() | late_domains.keys())
            - {binding.symbol for binding in computed},
            owned,
        )
    except (_PrinterDeclined, UnsupportedCapture) as error:
        raise GuardExportDeclined(str(error)) from error
    bindings, pointers, offsets = [], [], []
    computed_sources = {id(binding.source) for binding in computed}
    for source, symbol in printer.source_to_symbol.items():
        if id(source) in computed_sources:
            continue
        matches = [
            (bindings, row.index)
            for row, original in zip(
                trace.contract.integer_ranges, trace.integer_sources, strict=True
            )
            if source is original
        ]
        matches.extend(
            (pointers, binding.root.index)
            for binding in input_addresses
            if source is binding.source
        )
        matches.extend(
            (offsets, index) for index, original in offset_sources if source is original
        )
        if (
            len(matches) != 1
            or not symbol.name.isascii()
            or not symbol.name.isidentifier()
        ):
            raise GuardExportDeclined(
                "Computed predicate lost its original boxed correspondence"
            )
        target, index = matches[0]
        target.append((index, symbol.name))
    trace.compiler_binding.check()
    return GuardExpressions(
        expressions,
        tuple(bindings),
        tuple(printer.discharged),
        trace,
        tuple(pointers),
        tuple(offsets),
        statements,
        tuple(binding.owner for binding in computed),
    )


def prepare_guard(program, example_inputs, *, required=False):
    from .frontend import TerminalProgram

    if (
        type(program) is not TerminalProgram
        or type(program.guards) is not GuardExpressions
        or program.guards.trace.compiler_binding is not program.origin
        or program.guards.trace.contract is not program.contract
        or TracingContext.try_get() is not None
    ):
        raise GuardExportDeclined("Local predicate lost its terminal program ownership")
    program.check()
    guards = program.guards
    if not guards.expressions and not guards.statements and not required:
        return None
    if type(example_inputs) not in (tuple, list) or len(example_inputs) != len(
        program.contract.kinds
    ):
        raise GuardExportDeclined(
            "Local predicate preparation requires the complete boxed inputs"
        )
    rows = {row.index: row for row in program.contract.integer_ranges}
    values = []
    for index, _ in guards.bindings:
        value, row = example_inputs[index], rows[index]
        if (
            type(value) is not int
            or not -(2**63) <= value < 2**63
            or value < row.lower
            or row.upper is not None
            and value > row.upper
        ):
            raise GuardExportDeclined(
                "Local predicate input exceeds its inherited integer domain"
            )
        values.append(value)
    for index, _ in guards.pointer_bindings:
        value = example_inputs[index].const_data_ptr()
        if type(value) is not int or not 0 <= value < 2**64:
            raise GuardExportDeclined(
                "Local predicate address exceeds its unsigned pointer domain"
            )
        values.append(value)
    for index, _ in guards.storage_offset_bindings:
        values.append(example_inputs[index].storage_offset())
    declarations = [
        f"  const int64_t {symbol} = int_values[{index}];"
        for index, (_, symbol) in enumerate(guards.bindings)
    ]
    declarations.extend(
        f"  const uint64_t {symbol} = static_cast<uint64_t>(int_values[{index}]);"
        for index, (_, symbol) in enumerate(
            guards.pointer_bindings, len(guards.bindings)
        )
    )
    declarations.extend(
        f"  const int64_t {symbol} = int_values[{index}];"
        for index, (_, symbol) in enumerate(
            guards.storage_offset_bindings,
            len(guards.bindings) + len(guards.pointer_bindings),
        )
    )
    declarations = "\n".join(declarations)
    condition = " && ".join(f"({expr})" for expr in guards.expressions) or "true"
    source = f"""#include <algorithm>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <c10/util/generic_math.h>
extern "C" int8_t guard(int64_t* int_values, double* float_values) {{
{declarations}
  return {condition};
}}
"""
    if guards.statements:
        source = _ordered_guard_source((declarations, *guards.statements))
        library = CppCodeCache.load(source, extra_flags=STRICT_FLOAT_FLAGS)
    else:
        library = CppCodeCache.load(source)
    address = ctypes.cast(library.guard, ctypes.c_void_p).value
    if address is None:
        raise GuardExportDeclined("Compiled local predicate has no function address")
    predicate = ctypes.CFUNCTYPE(
        ctypes.c_int8, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double)
    )(address)
    bits = (ctypes.c_uint64 * len(values))(*(value % (2**64) for value in values))
    if predicate(ctypes.cast(bits, ctypes.POINTER(ctypes.c_int64)), None) != 1:
        raise GuardExportDeclined(
            "Compiled local predicate rejects its preparation inputs"
        )
    program.check()
    return PreparedGuard(
        tuple(index for index, _ in guards.bindings),
        address,
        (library, *guards.callback_owners) if guards.callback_owners else library,
        guards.expressions,
        source,
        program,
        tuple(index for index, _ in guards.pointer_bindings),
        tuple(index for index, _ in guards.storage_offset_bindings),
    )
