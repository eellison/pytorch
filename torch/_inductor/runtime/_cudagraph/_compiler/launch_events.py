from __future__ import annotations
import dis
import types
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CallSite:
    filename: str
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int


def callsite(code: types.CodeType, lasti: int) -> CallSite:
    if type(code) is not types.CodeType or type(lasti) is not int:
        raise TypeError("Expected an actual code object and instruction offset")
    instructions = list(dis.get_instructions(code, show_caches=True))
    indices = [i for i, instruction in enumerate(instructions) if instruction.offset == lasti]
    if len(indices) != 1:
        raise ValueError("The caller instruction is absent from its code object")
    index = indices[0]
    while index and instructions[index].opname == "CACHE":
        index -= 1
    instruction = instructions[index]
    if instruction.opname not in ("CALL", "CALL_FUNCTION_EX"):
        raise ValueError("Expected the currently executing Python call")
    position = instruction.positions
    values = (position.lineno, position.end_lineno, position.col_offset, position.end_col_offset)
    if any(type(value) is not int for value in values) or values[0] < 1 or values[1] < values[0]:
        raise ValueError("The call has no complete original source position")
    if values[2] < 0 or values[3] < 0:
        raise ValueError("The call has invalid source columns")
    if values[0] == values[1] and values[3] < values[2]:
        raise ValueError("The call has reversed source columns")
    return CallSite(code.co_filename, *values)


def original_callsites(code: types.CodeType) -> frozenset[CallSite]:
    sites = [callsite(code, instruction.offset) for instruction in dis.get_instructions(code)
             if instruction.opname in ("CALL", "CALL_FUNCTION_EX")]
    if not sites or len(set(sites)) != len(sites):
        raise ValueError("Original calls must have distinct complete source positions")
    return frozenset(sites)


@dataclass(frozen=True)
class _FunctionState:
    function: types.FunctionType
    code: types.CodeType
    wrapped: Any
    annotations: tuple[tuple[str, Any], ...]
    closure: tuple[Any, ...]

    def check(self) -> None:
        fn = self.function
        current = tuple(fn.__annotations__.items())
        if (fn.__code__ is not self.code or getattr(fn, "__wrapped__", None) is not self.wrapped
                or fn.__defaults__ or fn.__kwdefaults__
                or len(current) != len(self.annotations)
                or any(name != old_name or value is not old_value
                       for (name, value), (old_name, old_value) in zip(current, self.annotations))
                or len(fn.__closure__ or ()) != len(self.closure)
                or any(cell.cell_contents is not value for cell, value in zip(fn.__closure__ or (), self.closure))):
            raise RuntimeError("Original host or kernel code, annotations, or decorator association changed")


@dataclass
class CloneBundle:
    original_host: types.FunctionType
    original_kernel: types.FunctionType
    host_body: types.FunctionType
    kernel_body: types.FunctionType
    host: types.FunctionType
    kernel: types.FunctionType
    host_code: types.CodeType
    kernel_code: types.CodeType
    sites: frozenset[CallSite]
    _functions: tuple[_FunctionState, ...] = field(repr=False)
    _globals: tuple[tuple[dict[str, Any], dict[str, Any]], ...] = field(repr=False)

    @classmethod
    def create(cls, host: Any, kernel: Any) -> CloneBundle:
        import cutlass.cute as cute

        original_host = getattr(host, "__wrapped__", None)
        original_kernel = getattr(kernel, "__wrapped__", None)
        if any(type(fn) is not types.FunctionType for fn in (host, kernel, original_host, original_kernel)):
            raise TypeError("Expected the original decorated CuTe host and kernel")
        for fn in (original_host, original_kernel):
            if fn.__closure__ or fn.__defaults__ or fn.__kwdefaults__ or getattr(fn, "_preprocessed", False):
                raise ValueError("Only cold functions without closures or defaults can be cloned")
        host_globals = dict(original_host.__globals__)
        kernel_globals = dict(original_kernel.__globals__)
        kernel_body = types.FunctionType(original_kernel.__code__, kernel_globals, original_kernel.__name__)
        host_body = types.FunctionType(original_host.__code__, host_globals, original_host.__name__)
        for clone, original in ((host_body, original_host), (kernel_body, original_kernel)):
            clone.__annotations__ = dict(original.__annotations__)
            clone.__qualname__ = original.__qualname__
            clone.__module__ = original.__module__
            clone.__doc__ = original.__doc__
        cloned_kernel = cute.kernel(kernel_body)
        for namespace in (host_globals, kernel_globals):
            for name, value in tuple(namespace.items()):
                if value is kernel:
                    namespace[name] = cloned_kernel
        cloned_host = cute.jit(host_body)
        for clone, original in ((host_body, original_host), (kernel_body, original_kernel)):
            clone._decorator_location = original._decorator_location
        states = tuple(_FunctionState(
            fn, fn.__code__, getattr(fn, "__wrapped__", None), tuple(fn.__annotations__.items()),
            tuple(cell.cell_contents for cell in fn.__closure__ or ()),
        ) for fn in (host, original_host, kernel, original_kernel))
        globals_state = tuple((fn.__globals__, dict(fn.__globals__)) for fn in (original_host, original_kernel))
        result = cls(host, kernel, host_body, kernel_body, cloned_host, cloned_kernel,
                     original_host.__code__, original_kernel.__code__, original_callsites(original_host.__code__),
                     states, globals_state)
        result.check_originals()
        return result

    def check_originals(self) -> None:
        for state in self._functions:
            state.check()
        for namespace, expected in self._globals:
            if namespace.keys() != expected.keys() or any(namespace[name] is not value for name, value in expected.items()):
                raise RuntimeError("Original host/kernel globals changed after cloning")
        if any(getattr(fn, "_preprocessed", False)
               for fn in (self.original_host.__wrapped__, self.original_kernel.__wrapped__)):
            raise RuntimeError("The original functions were preprocessed instead of their clones")
