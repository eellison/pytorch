"""Compile ordered live host-trace predicates for the native boxed dispatcher."""

import ctypes
import math
from dataclasses import dataclass

import sympy
from sympy.core.relational import Relational

import torch
from torch._dynamo.source import LocalSource
from torch._guards import TracingContext
from torch._inductor.codecache import CppCodeCache
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, InputSource
from torch._inductor.runtime.cudagraph_compiled_evaluation import STRICT_FLOAT_FLAGS
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.utils._sympy.functions import (
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    Max,
    Min,
    Mod,
    PythonMod,
    ToFloat,
)
from torch.utils._sympy.value_ranges import ValueRanges

from .address_guard_printer import _AddressPrinter, GuardExportDeclined


_FLOAT_NODES = (sympy.Float, ToFloat, FloatPow, FloatTrueDiv)


def prove_owned_guard(printer, predicate, substitutions):
    if predicate.has(*_FLOAT_NODES):
        raise GuardExportDeclined(
            "Owned address float guard is not proven before allocation"
        )
    divisions = predicate.atoms(FloorDiv, Mod, PythonMod, sympy.Mod)
    if any(printer._division_requirements(value, substitutions) for value in divisions):
        raise GuardExportDeclined(
            "Owned address operation domain is not proven before allocation"
        )
    proof = predicate.xreplace(substitutions)
    proof = proof.replace(
        lambda value: isinstance(value, (Mod, PythonMod)),
        lambda value: sympy.Mod(*value.args),
    )
    printer.check_arithmetic(proof)
    bounds = printer.bounds(proof)
    if bounds.lower is not sympy.true or bounds.upper is not sympy.true:
        raise GuardExportDeclined("Owned address guard is not proven before allocation")


@dataclass(frozen=True)
class HostTraceGuard:
    function_address: int
    library_owner: object
    cpp_source: str
    expressions: tuple[str, ...]
    mapping: object
    boxed_pointer_indices: tuple[int, ...]
    boxed_storage_offset_indices: tuple[int, ...]
    metadata_bindings: tuple[tuple[str, int, int | None], ...]

    @property
    def registration(self):
        return (
            (),
            self.function_address,
            self.library_owner,
            self.boxed_pointer_indices,
            self.boxed_storage_offset_indices,
            self.metadata_bindings,
        )


class _OrderedPrinter(_AddressPrinter):
    _print_Dummy = _AddressPrinter._print_Symbol

    def __init__(self, *args, domains, sources):
        super().__init__(*args, domains=domains, sources=sources)
        self.facts = {}
        self.runtime_checked = set()

    def check_arithmetic(self, expression):
        if expression in self.checked:
            return
        if isinstance(expression, sympy.Float):
            if expression._prec > 53 or not math.isfinite(float(expression)):
                raise GuardExportDeclined("Host float literal requires finite binary64")
            self.checked.add(expression)
            return
        if isinstance(expression, (ToFloat, FloatPow, FloatTrueDiv)):
            for argument in expression.args:
                self.check_arithmetic(argument)
            if isinstance(expression, ToFloat):
                valid = expression.args[0].is_integer is True
            else:
                valid = all(isinstance(arg, _FLOAT_NODES) for arg in expression.args)
            if not valid:
                raise GuardExportDeclined(
                    "Host float operation requires typed operands"
                )
            self.checked.add(expression)
            return
        if isinstance(expression, Relational) and any(
            isinstance(arg, _FLOAT_NODES) for arg in expression.args
        ):
            if not all(isinstance(arg, _FLOAT_NODES) for arg in expression.args):
                raise GuardExportDeclined(
                    "Host float comparison requires typed operands"
                )
        if isinstance(expression, sympy.Pow):
            if (
                not isinstance(expression.exp, sympy.Integer)
                or not 0 <= expression.exp < 2**64
                or expression.base.is_integer is not True
            ):
                raise GuardExportDeclined(
                    "Host integer power requires a natural uint64 exponent"
                )
            self.check_arithmetic(expression.base)
            self.checked.add(expression)
            return
        if isinstance(
            expression,
            (
                FloorDiv,
                Mod,
                PythonMod,
                sympy.Mod,
                Min,
                Max,
                sympy.Min,  # @allow-raw-sympy-minmax: type checks only
                sympy.Max,  # @allow-raw-sympy-minmax: type checks only
            ),
        ):
            for argument in expression.args:
                self.check_arithmetic(argument)
            if any(
                argument.is_Boolean or isinstance(argument, (Relational, *_FLOAT_NODES))
                for argument in expression.args
            ):
                raise GuardExportDeclined(
                    "Host selection and division require integers"
                )
            self.checked.add(expression)
            return
        if isinstance(expression, sympy.Piecewise):
            if not expression.args or expression.args[-1].cond is not sympy.true:
                raise GuardExportDeclined(
                    "Integer selection requires an unconditional final arm"
                )
            for value, condition in expression.args:
                self.check_arithmetic(value)
                self.check_arithmetic(condition)
                if value.is_Boolean or isinstance(value, (Relational, *_FLOAT_NODES)):
                    raise GuardExportDeclined("Host selection requires integer values")
            self.checked.add(expression)
            return
        if not isinstance(expression, (sympy.Add, sympy.Mul)):
            return super().check_arithmetic(expression)
        for argument in expression.args:
            self.check_arithmetic(argument)
        try:
            super().check_arithmetic(expression)
        except GuardExportDeclined:
            if not self.bounds(expression).is_int:
                raise
            self.runtime_checked.add(expression)
            self.checked.add(expression)

    def _checked_operation(self, expression, operation):
        values = [self._print(argument) for argument in expression.args]
        result = values[0]
        for value in values[1:]:
            result = f"guard_{operation}({result}, {value})"
        return result

    def _print_Add(self, expression, order=None):
        if expression in self.runtime_checked:
            return self._checked_operation(expression, "add")
        return super()._print_Add(expression, order=order)

    def _print_Mul(self, expression):
        if expression in self.runtime_checked:
            return self._checked_operation(expression, "multiply")
        return super()._print_Mul(expression)

    def _print_Float(self, expression):
        return float(expression).hex()

    def _print_FloatTrueDiv(self, expression):
        left, right = (self._print(arg) for arg in expression.args)
        return f"guard_float_divide({left}, {right})"

    def _print_FloatPow(self, expression):
        base, exponent = (self._print(arg) for arg in expression.args)
        return f"guard_float_power({base}, {exponent})"

    def _print_Pow(self, expression):
        return f"guard_power({self._print(expression.base)}, {int(expression.exp)}ULL)"

    def _print_FloorDiv(self, expression):
        left, right = (self._print(arg) for arg in expression.args)
        return f"guard_floor_divide({left}, {right})"

    def _print_Mod(self, expression):
        left, right = (self._print(arg) for arg in expression.args)
        nonnegative = ", true" if expression.func is Mod else ""
        return f"guard_remainder({left}, {right}{nonnegative})"

    _print_PythonMod = _print_Mod

    def _print_And(self, expression):
        return " && ".join(f"({self._print(argument)})" for argument in expression.args)

    @staticmethod
    def _domain_conjunction(conditions):
        return (
            " && ".join(f"({condition})" for condition in conditions if condition)
            or None
        )

    def _division_requirements(self, expression, substitutions=None):
        left, right = (
            argument.xreplace(substitutions) if substitutions else argument
            for argument in expression.args
        )
        divisor = self.bounds(right)
        requirements = []
        if expression.func is Mod:
            if self.bounds(left).lower < 0:
                requirements.append(sympy.Ge(left, 0, evaluate=False))
            if divisor.lower <= 0:
                requirements.append(sympy.Gt(right, 0, evaluate=False))
        elif divisor.lower <= 0 <= divisor.upper:
            requirements.append(sympy.Ne(right, 0, evaluate=False))
        return requirements

    def _operation_domain(self, expression):
        # Predicate simplification must preserve the domains of partial operations.
        if isinstance(expression, (sympy.And, sympy.Or)):
            tail = None
            for argument in reversed(expression.args):
                if tail:
                    value = self._print(argument)
                    tail = (
                        f"({value}) ? ({tail}) : true"
                        if isinstance(expression, sympy.And)
                        else f"({value}) ? true : ({tail})"
                    )
                tail = self._domain_conjunction(
                    (self._operation_domain(argument), tail)
                )
            return tail
        if isinstance(expression, sympy.Piecewise):
            tail = self._operation_domain(expression.args[-1].expr)
            for value, condition in reversed(expression.args[:-1]):
                branch = self._operation_domain(value)
                if branch != tail:
                    tail = f"({self._print(condition)}) ? ({branch or 'true'}) : ({tail or 'true'})"
                tail = self._domain_conjunction(
                    (self._operation_domain(condition), tail)
                )
            return tail
        conditions = [self._operation_domain(argument) for argument in expression.args]
        if isinstance(expression, (FloorDiv, Mod, PythonMod, sympy.Mod)):
            conditions.extend(
                self._print(requirement)
                for requirement in self._division_requirements(expression)
            )
        return self._domain_conjunction(conditions)

    def doprint(self, expression):
        self.check_arithmetic(expression)
        domain = None
        if expression.is_Boolean or isinstance(expression, Relational):
            domain = self._operation_domain(expression)
        value = (
            self._print(expression)
            if expression.has(*_FLOAT_NODES)
            else super().doprint(expression)
        )
        return f"({domain}) && ({value})" if domain else value

    def print_source(self, source):
        if not any(source is original for original in self.sources):
            raise GuardExportDeclined(
                "Host-trace guard source lost its native metadata binding"
            )
        return source.local_name

    def bounds(self, expression):
        if expression.has(*_FLOAT_NODES):
            raise GuardExportDeclined("Host float expressions have no integer proof")
        return super().bounds(expression.xreplace(self.facts))

    def assume(self, expression):
        if expression.has(*_FLOAT_NODES):
            return
        if isinstance(expression, sympy.And):
            for term in expression.args:
                self.assume(term)
            return
        if not isinstance(expression, Relational):
            return
        left, right = expression.lhs, expression.rhs
        if not isinstance(right, sympy.Integer):
            left, right = left - right, sympy.Integer(0)
        if not left.free_symbols:
            return
        current = self.bounds(left)
        lower, upper = current.lower, current.upper
        if isinstance(expression, sympy.Equality):
            lower, upper = max(lower, right), min(upper, right)
        elif isinstance(expression, (sympy.GreaterThan, sympy.StrictGreaterThan)):
            lower = max(
                lower, right + int(isinstance(expression, sympy.StrictGreaterThan))
            )
        elif isinstance(expression, (sympy.LessThan, sympy.StrictLessThan)):
            upper = min(
                upper, right - int(isinstance(expression, sympy.StrictLessThan))
            )
        elif isinstance(expression, sympy.Unequality):  # codespell:ignore unequality
            if lower == right:
                lower += 1
            elif upper == right:
                upper -= 1
        if lower > upper:
            raise UnsupportedCapture(
                "Host-trace guards have inconsistent integer domains"
            )
        if isinstance(left, sympy.Symbol):
            self.domains[left] = ValueRanges(lower, upper)
        else:
            symbol = self.facts.setdefault(left, sympy.Dummy(integer=True))
            self.domains[symbol] = ValueRanges(lower, upper)
        self.checked.clear()


@dataclass(frozen=True)
class _GuardCallback:
    symbol: sympy.Symbol
    arguments: tuple[sympy.Expr, ...]
    address: int
    owner: object
    kind: str
    expected: int


def _emit_ordered_guards(
    printer, symbol_sources, predicates, callbacks, available, owned
):
    expressions, statements = [], []
    available = set(available)
    pending = list(predicates)
    for callback in (*callbacks, None):
        symbol = None if callback is None else callback.symbol
        remaining = []
        for predicate in pending:
            if not predicate.free_symbols.issubset(available):
                remaining.append(predicate)
                continue
            if predicate.free_symbols.intersection(owned):
                prove_owned_guard(printer, predicate, owned)
                continue
            expression = printer.doprint(predicate)
            if expression != "true":
                expressions.append(expression)
                statements.append(f"  if (!({expression})) return 0;")
            printer.assume(predicate)
        pending = remaining
        if callback is None:
            break
        arguments = []
        for index, expression in enumerate(callback.arguments):
            if expression.is_integer is not True:
                raise UnsupportedCapture("Opaque guard arguments require integers")
            if not expression.free_symbols.issubset(available - owned.keys()):
                raise UnsupportedCapture(
                    "Opaque guard arguments must exist before allocation"
                )
            name = f"{printer.print_source(symbol_sources[symbol][0])}_arg{index}"
            statements.append(
                f"  const __int128 {name} = {printer.doprint(expression)};"
            )
            statements.append(
                f"  if ({name} < INT64_MIN || {name} > INT64_MAX) return 0;"
            )
            arguments.append(f"static_cast<int64_t>({name})")
        name = printer.print_source(symbol_sources[symbol][0])
        statements.append(
            f"  const int64_t {name} = reinterpret_cast<int64_t(*)(const std::vector<int64_t>&)>"
            f"(uintptr_t({callback.address}ULL))(std::vector<int64_t>{{{', '.join(arguments)}}});"
        )
        available.add(symbol)
        if callback.kind == "guard":
            predicate = sympy.Eq(symbol, callback.expected, evaluate=False)
            expression = printer.doprint(predicate)
            expressions.append(expression)
            statements.append(f"  if (!({expression})) return 0;")
            printer.assume(predicate)
    if pending:
        raise UnsupportedCapture(
            "Host-trace predicates depend on unavailable opaque results"
        )
    return tuple(expressions), tuple(statements)


_ORDERED_GUARD_PREFIX = """#include <algorithm>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <c10/util/generic_math.h>
static inline __int128 guard_add(__int128 a, __int128 b) {
  __int128 result;
  if (__builtin_add_overflow(a, b, &result)) {
    throw std::overflow_error("host guard addition overflow");
  }
  return result;
}
static inline __int128 guard_multiply(__int128 a, __int128 b) {
  __int128 result;
  if (__builtin_mul_overflow(a, b, &result)) {
    throw std::overflow_error("host guard multiplication overflow");
  }
  return result;
}
static inline __int128 guard_floor_divide(__int128 a, __int128 b) {
  constexpr __int128 minimum = -(static_cast<__int128>(1) << 126) * 2;
  if (b == 0 || (a == minimum && b == -1)) {
    throw std::domain_error("host guard division domain");
  }
  __int128 quotient = a / b;
  __int128 remainder = a % b;
  if (remainder != 0 && ((remainder < 0) != (b < 0))) {
    --quotient;
  }
  return quotient;
}
static inline __int128 guard_remainder(__int128 a, __int128 b,
                                       bool nonnegative = false) {
  if (b == 0 || (nonnegative && (a < 0 || b < 0))) {
    throw std::domain_error("host guard remainder domain");
  }
  if (b == -1) {
    return 0;
  }
  __int128 remainder = a % b;
  if (remainder != 0 && ((remainder < 0) != (b < 0))) {
    remainder += b;
  }
  return remainder;
}
static inline double guard_float_divide(double a, double b) {
  if (b == 0) {
    throw std::domain_error("host float division domain");
  }
  const double result = a / b;
  if (!std::isfinite(result)) {
    throw std::domain_error("host float division requires a finite result");
  }
  return result;
}
static inline double guard_float_power(double base, double exponent) {
  if ((base == 0 && exponent < 0) ||
      (base < 0 && std::trunc(exponent) != exponent)) {
    throw std::domain_error("host float power domain");
  }
  const double result = std::pow(base, exponent);
  if (!std::isfinite(result)) {
    throw std::domain_error("host float power requires a finite result");
  }
  return result;
}
static inline __int128 guard_power(__int128 base, uint64_t exponent) {
  __int128 result = 1;
  while (exponent) {
    if (exponent & 1) {
      result = guard_multiply(result, base);
    }
    exponent >>= 1;
    if (exponent) {
      base = guard_multiply(base, base);
    }
  }
  return result;
}
extern "C" int8_t guard(int64_t* int_values, double* float_values) {
  try {
"""


def _ordered_guard_source(statements):
    return (
        _ORDERED_GUARD_PREFIX
        + "\n".join(statements)
        + "\n  return 1;\n  } catch (...) { return 0; }\n}\n"
    )


def compile_host_trace_guard(tape, mapping, tensor_examples, extra_guards=()):
    r"""Compile the original local tape's reuse conditions into a native predicate.

    ``tensor_examples`` follow the mapping's compact input order; ``extra_guards``
    use its translated symbols. The result retains the compiled library and
    native callbacks needed by the predicate.
    """
    if mapping.tape is not tape or TracingContext.try_get() is not None:
        raise UnsupportedCapture("Host-trace guards require their original local tape")
    if tape.shape_env.deferred_runtime_asserts:
        raise UnsupportedCapture(
            "Host-trace deferred assertions have no ordered replay predicate"
        )
    if len(tensor_examples) != len(tape.inputs) or set(
        mapping.input_indices.values()
    ) != set(range(len(tensor_examples))):
        raise UnsupportedCapture(
            "Host-trace guard examples must cover compacted Tensor arguments"
        )
    original_symbols = (
        mapping.metadata_symbols.keys()
        | mapping.substitutions.keys()
        | mapping.opaque_symbols.keys()
    )
    if not set(tape.shape_env.var_to_range).issubset(original_symbols):
        raise UnsupportedCapture("Host-trace guard ranges contain unbound symbols")
    allowed = (
        mapping.metadata_symbols.keys()
        | mapping.address_symbols.keys()
        | mapping.host_address_symbols.keys()
        | mapping.opaque_symbols.keys()
    )
    if any(
        not isinstance(guard, sympy.Basic) or not guard.free_symbols.issubset(allowed)
        for guard in extra_guards
    ):
        raise UnsupportedCapture(
            "Additional host-trace guards need translated symbolic expressions"
        )

    native_metadata = torch._C._cuda_boxed_tensor_metadata
    bindings, values, static_checks = [], [], []
    symbol_sources, sources, domains = {}, [], {}
    pointer_indices, offset_indices = [], []
    owned = {}
    for symbol in mapping.host_address_symbols:
        owned[symbol] = symbol
        domains[symbol] = ValueRanges(-(1 << 63), (1 << 63) - 1)
    for symbol, root in mapping.address_symbols.items():
        if type(root) is BufferSource:
            alignment = mapping.root_alignments[root]
            if alignment != 256:
                raise UnsupportedCapture(
                    "Owned host-trace address lacks its allocator alignment"
                )
            quotient = sympy.Dummy(integer=True, nonnegative=True)
            owned[symbol] = alignment * quotient
            domains[quotient] = ValueRanges(0, ((1 << 64) - 1) // alignment)
            continue
        if type(root) is not InputSource:
            raise UnsupportedCapture("Host-trace guard address has no Tensor input")
        source = LocalSource(f"pointer_{len(pointer_indices)}")
        sources.append(source)
        symbol_sources[symbol] = [source]
        domains[symbol] = ValueRanges(0, (1 << 64) - 1)
        pointer_indices.append(root.index)
        values.append(tensor_examples[root.index].data_ptr())

    offset_symbols = [
        (symbol, source)
        for symbol, source in mapping.metadata_symbols.items()
        if source.property == "storage_offset"
    ]
    for symbol, metadata in offset_symbols:
        source = LocalSource(f"offset_{len(offset_indices)}")
        sources.append(source)
        symbol_sources[symbol] = [source]
        domains[symbol] = ValueRanges(-(1 << 63), (1 << 63) - 1)
        offset_indices.append(metadata.index)
        values.append(tensor_examples[metadata.index].storage_offset())

    for record in tape.inputs:
        index = mapping.input_indices[record.position]
        tensor = tensor_examples[index]
        if (
            type(tensor) not in (torch.Tensor, torch.nn.Parameter)
            or tensor.dtype != record.dtype
            or tensor.dim() != len(record.sizes)
            or tensor.layout is not torch.strided
            or tensor.is_nested
            or tensor.is_neg()
            or tensor.is_conj()
        ):
            raise UnsupportedCapture(
                "Host-trace guard example differs from its plain strided Tensor contract"
            )
        expected_device = -1 if record.device.type == "cpu" else tape.device.index
        if native_metadata(tensor, "device") != expected_device:
            raise UnsupportedCapture(
                "Host-trace guard example is on a different CUDA device"
            )
        for kind in ("dtype", "device", "rank", "neg", "conj", "layout", "pinned"):
            value = native_metadata(tensor, kind)
            name = f"metadata_{len(bindings)}"
            static_checks.append(f"  if ({name} != {value}) return 0;")
            bindings.append((kind, index, 0))
            values.append(value)
    for symbol, metadata in mapping.metadata_symbols.items():
        if metadata.property == "storage_offset":
            continue
        if metadata.property not in ("size", "stride"):
            raise UnsupportedCapture(
                "Host-trace metadata has no native Tensor accessor"
            )
        source = LocalSource(f"metadata_{len(bindings)}")
        sources.append(source)
        symbol_sources[symbol] = [source]
        domains[symbol] = ValueRanges(-(1 << 63), (1 << 63) - 1)
        bindings.append((metadata.property, metadata.index, metadata.dimension))
        values.append(
            native_metadata(
                tensor_examples[metadata.index], metadata.property, metadata.dimension
            )
        )

    for index, symbol in enumerate(mapping.opaque_symbols):
        source = LocalSource(f"opaque_{index}")
        sources.append(source)
        symbol_sources[symbol] = [source]
        domains[symbol] = ValueRanges(-(1 << 63), (1 << 63) - 1)

    printer = _OrderedPrinter(
        symbol_sources,
        lambda source: source.local_name,
        symbol_sources,
        domains=domains,
        sources=sources,
    )
    predicates = []
    for symbol, metadata in mapping.metadata_symbols.items():
        minimum = 1 if metadata.property == "size" else 0
        predicates.append(sympy.Ge(symbol, minimum, evaluate=False))
    for symbol, bounds in tape.shape_env.var_to_range.items():
        value = mapping.translate(symbol)
        for bound, relation in ((bounds.lower, sympy.Ge), (bounds.upper, sympy.Le)):
            if isinstance(bound, sympy.Integer):
                predicates.append(relation(value, bound, evaluate=False))
    predicates.extend(
        mapping.translate(guard, preserve_operations=True) for guard in tape.guards
    )
    predicates.extend(
        mapping.translate(
            sympy.Eq(symbol, value, evaluate=False), preserve_operations=True
        )
        for symbol, value in tape.shape_env.replacements.items()
    )
    predicates.extend(extra_guards)

    callbacks = tuple(
        _GuardCallback(
            symbol,
            tuple(
                mapping.translate(arg, preserve_operations=True)
                for arg in record["args"]
            ),
            record["impl"],
            record["call"],
            record["kind"],
            record["expected"],
        )
        for symbol, record in mapping.opaque_symbols.items()
    )
    try:
        expressions, statements = _emit_ordered_guards(
            printer,
            symbol_sources,
            predicates,
            callbacks,
            mapping.metadata_symbols.keys()
            | mapping.address_symbols.keys()
            | mapping.host_address_symbols.keys(),
            owned,
        )
    except GuardExportDeclined as error:
        raise UnsupportedCapture(
            f"Host-trace guard cannot use checked native arithmetic: {error}"
        ) from error

    declarations = []
    for index in range(len(pointer_indices)):
        declarations.append(
            f"  const uint64_t pointer_{index} = static_cast<uint64_t>(int_values[{index}]);"
        )
    start = len(pointer_indices)
    declarations.extend(
        f"  const int64_t offset_{index} = int_values[{start + index}];"
        for index in range(len(offset_indices))
    )
    start += len(offset_indices)
    declarations.extend(
        f"  const int64_t metadata_{index} = int_values[{start + index}];"
        for index in range(len(bindings))
    )
    checks = [*declarations, *static_checks, *statements]
    source = _ordered_guard_source(checks)
    library = CppCodeCache.load(source, extra_flags=STRICT_FLOAT_FLAGS)
    address = ctypes.cast(library.guard, ctypes.c_void_p).value
    if address is None:
        raise UnsupportedCapture("Compiled host-trace guard has no function address")
    predicate = ctypes.CFUNCTYPE(
        ctypes.c_int8, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double)
    )(address)
    bits = (ctypes.c_uint64 * len(values))(*(value % (1 << 64) for value in values))
    if predicate(ctypes.cast(bits, ctypes.POINTER(ctypes.c_int64)), None) != 1:
        raise UnsupportedCapture(
            "Compiled host-trace guard rejects its preparation inputs"
        )
    owner = (library, *(record["call"] for record in mapping.opaque_symbols.values()))
    return HostTraceGuard(
        address,
        owner,
        source,
        tuple(expressions),
        mapping,
        tuple(pointer_indices),
        tuple(offset_indices),
        tuple(bindings),
    )
