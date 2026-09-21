from __future__ import annotations

import builtins
import dis
import math
import types
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.node import has_side_effect
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv


class PythonHostUnsupported(ValueError):
    pass


_ACTIVE: ContextVar[Any] = ContextVar("cute_python_entry_trace", default=None)
_SYMBOLIC = (torch.SymInt, torch.SymFloat, torch.SymBool)
_TORCH_NAMES = {
    "empty", "empty_like", "empty_strided", "zeros", "zeros_like", "ones", "ones_like",
    "full", "full_like", "reshape", "transpose", "permute", "narrow", "select", "as_strided",
    "flatten", "unsqueeze", "squeeze", "numel", "nonzero", "sym_int", "sym_float", "sym_max", "sym_min",
    "bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64",
    "strided", "contiguous_format", "preserve_format",
}
_TORCH_MEMBERS = {name: getattr(torch, name) for name in _TORCH_NAMES}
_TENSOR_NAMES = {
    "shape", "size", "stride", "storage_offset", "numel", "dim", "ndimension", "ndim", "dtype",
    "device", "layout", "requires_grad", "is_cuda", "view", "reshape", "transpose", "permute", "narrow",
    "select", "as_strided", "flatten", "unsqueeze", "squeeze", "detach", "contiguous", "clone",
    "new_empty", "new_zeros", "new_full", "is_contiguous", "nonzero", "item",
}
_TENSOR_MEMBERS = {name: getattr(torch.Tensor, name) for name in _TENSOR_NAMES}
_BUILTINS = {name: getattr(builtins, name) for name in (
    "int", "bool", "float", "len", "range", "min", "max", "abs", "enumerate", "zip",
    "ValueError", "RuntimeError", "AssertionError",
)}
_FORBIDDEN = {
    "STORE_GLOBAL", "DELETE_GLOBAL", "STORE_DEREF", "DELETE_DEREF", "STORE_ATTR", "DELETE_ATTR",
    "STORE_SUBSCR", "DELETE_SUBSCR", "IMPORT_NAME", "IMPORT_FROM", "LOAD_BUILD_CLASS", "MAKE_FUNCTION",
    "YIELD_VALUE", "RETURN_GENERATOR", "BEFORE_WITH", "WITH_EXCEPT_START", "BEFORE_ASYNC_WITH",
    "GET_AWAITABLE", "IS_OP",
}


def _constant(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str, bytes):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is tuple:
        return tuple(_constant(item) for item in value)
    raise PythonHostUnsupported("Expected an immutable primitive entry configuration or host constant")


@dataclass(frozen=True, eq=False)
class PythonEntry:
    target: Any
    config: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if type(self.config) is not tuple:
            raise PythonHostUnsupported("Entry configuration must be an immutable tuple")
        _constant(self.config)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        active = _ACTIVE.get()
        if active is None:
            raise RuntimeError("PythonEntry only records an explicit Python host trace; it never invokes its target")
        active.record(self, args, kwargs)


@has_side_effect
def opaque_entry(entry_index: int, *args: Any, **kwargs: Any) -> None:
    raise RuntimeError("The opaque entry node requires an explicit downstream consumer")


@dataclass(frozen=True, eq=False)
class Operand:
    path: tuple[Any, ...]
    value: Any
    fx_argument: Any


@dataclass(frozen=True, eq=False)
class EntryCall:
    entry_index: int
    entry: PythonEntry
    target: Any
    config: tuple[Any, ...]
    node: Node
    arguments: tuple[Any, ...]
    keyword_arguments: tuple[tuple[str, Any], ...]
    operands: tuple[Operand, ...]


def _leaves(value: Any, path: tuple[Any, ...] = ()):
    if type(value) in (tuple, list):
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))
    elif type(value) is dict:
        if any(type(key) is not str for key in value):
            raise PythonHostUnsupported("Entry dictionaries require string keys")
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    else:
        yield path, value


class _Dependencies:
    def __init__(self, host: types.FunctionType, entries: tuple[PythonEntry, ...]) -> None:
        self.functions: list[Any] = []
        self.references: list[Any] = []
        self.cells: list[Any] = []
        self.tensor_attributes: set[str] = set()
        self.entries = entries
        self.entry_states = tuple((entry, entry.target, entry.config) for entry in entries)
        self.entry_call = PythonEntry.__call__
        self._visit(host)

    def _value(self, value: Any) -> None:
        if value is torch or any(value is entry for entry in self.entries):
            return
        if any(value is item for item in (*_TORCH_MEMBERS.values(), *_BUILTINS.values())):
            return
        if type(value) is types.FunctionType:
            self._visit(value)
            return
        _constant(value)

    def _visit(self, function: types.FunctionType) -> None:
        if any(function is saved[0] for saved in self.functions):
            return
        defaults = function.__defaults__
        kwdefaults = function.__kwdefaults__
        self.functions.append((function, function.__code__, defaults, kwdefaults,
                               tuple((kwdefaults or {}).items()), function.__closure__))
        for value in (*(() if defaults is None else defaults), *(kwdefaults or {}).values()):
            self._value(value)
        for cell in function.__closure__ or ():
            value = cell.cell_contents
            self.cells.append((cell, value))
            self._value(value)
        if any(type(value) is types.CodeType for value in function.__code__.co_consts):
            raise PythonHostUnsupported("Nested function/comprehension code is outside the initial host subset")
        for instruction in dis.get_instructions(function):
            name = instruction.argval
            if instruction.opname in _FORBIDDEN:
                raise PythonHostUnsupported(f"Unsupported host effect: {instruction.opname}")
            if instruction.opname == "LOAD_GLOBAL":
                fallback = None if name in function.__globals__ else function.__globals__
                namespace = function.__globals__ if fallback is None else function.__builtins__
                if name not in namespace:
                    raise PythonHostUnsupported(f"Unresolved host dependency: {name}")
                value = namespace[name]
                self.references.append((namespace, name, value, fallback))
                self._value(value)
            if instruction.opname in ("LOAD_ATTR", "LOAD_METHOD"):
                if name not in _TORCH_MEMBERS and name not in _TENSOR_NAMES and name not in ("type", "index"):
                    raise PythonHostUnsupported(f"Unsupported host attribute: {name}")
                if name in _TORCH_MEMBERS:
                    self.references.append((vars(torch), name, _TORCH_MEMBERS[name], None))
                if name in _TENSOR_MEMBERS:
                    self.tensor_attributes.add(name)

    def check(self) -> None:
        if PythonEntry.__call__ is not self.entry_call:
            raise RuntimeError("Python entry adapter changed")
        for entry, target, config in self.entry_states:
            if entry.target is not target or entry.config is not config:
                raise RuntimeError("Selected entry identity or configuration changed")
        for function, code, defaults, kwdefaults, items, closure in self.functions:
            if (function.__code__ is not code or function.__defaults__ is not defaults
                    or function.__kwdefaults__ is not kwdefaults or function.__closure__ is not closure
                    or list(kwdefaults or {}) != [key for key, _ in items]
                    or any(kwdefaults[key] is not value for key, value in items)):
                raise RuntimeError("Original Python host/helper code or defaults changed")
        for namespace, name, value, fallback in self.references:
            if (name not in namespace or namespace[name] is not value
                    or fallback is not None and name in fallback):
                raise RuntimeError(f"Original Python host dependency changed: {name}")
        if any(cell.cell_contents is not value for cell, value in self.cells):
            raise RuntimeError("Original Python host closure changed")
        if any(getattr(torch.Tensor, name) is not _TENSOR_MEMBERS[name] for name in self.tensor_attributes):
            raise RuntimeError("Tensor metadata/view helper changed")


class _Recording:
    def __init__(self, entries: tuple[PythonEntry, ...], mode: FakeTensorMode, environment: ShapeEnv) -> None:
        self.entries = entries
        self.mode = mode
        self.environment = environment
        self.calls: list[EntryCall] = []

    def record(self, entry: PythonEntry, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        proxy_mode = get_proxy_mode()
        indices = [index for index, candidate in enumerate(self.entries) if candidate is entry]
        if proxy_mode is None or len(indices) != 1:
            raise RuntimeError("Entry call escaped its registered symbolic host trace")
        tree = {"args": args, "kwargs": kwargs}
        leaves = list(_leaves(tree))
        for _, value in leaves:
            if isinstance(value, torch.Tensor):
                if not isinstance(value, FakeTensor) or value.fake_mode is not self.mode:
                    raise RuntimeError("Entry Tensor operand escaped the local FakeTensorMode")
            elif isinstance(value, _SYMBOLIC):
                if value.node.shape_env is not self.environment:
                    raise RuntimeError("Entry symbolic operand belongs to a foreign ShapeEnv")
            elif value is not None and type(value) not in (int, bool, float):
                raise PythonHostUnsupported("Entry operands must be Tensors, symbolic/numeric scalars, or None")
        proxies = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, tree)
        proxy = proxy_mode.tracer.create_proxy(
            "call_function", opaque_entry, (indices[0], *proxies["args"]), proxies["kwargs"], name="cute_entry",
        )
        arguments = tuple(proxy.node.args[1:])
        keyword_arguments = tuple(proxy.node.kwargs.items())
        mapped = {"args": arguments, "kwargs": dict(keyword_arguments)}
        operands = []
        for path, value in leaves:
            fx_argument = mapped
            for key in path:
                fx_argument = fx_argument[key]
            if (isinstance(value, torch.Tensor)
                    or isinstance(value, _SYMBOLIC) and value.node.expr.free_symbols):
                if not isinstance(fx_argument, Node):
                    raise RuntimeError("A symbolic entry operand lost its exact FX dependency")
            operands.append(Operand(path, value, fx_argument))
        self.calls.append(EntryCall(indices[0], entry, entry.target, entry.config, proxy.node,
                                    arguments, keyword_arguments, tuple(operands)))
