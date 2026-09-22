"""The persisted launch descriptor of a CuTe DSL program (A410, A411).

A program compiled through cute.compile launches its kernel from generated
host code: the loaded object (QuACK's on-disk cache, a tvm_ffi.Function) and
the in-process JitCompiledFunction alike keep the kernel handle and the
parameter packing to themselves. What the recorder needs to record such a
launch on the tape is read once, at the compile, from the host function's own
IR at the DSL's trace finalize (the cute dialect, before lowering): the
kernel's declared parameters (memref types with their dynamic leaves), the
launch's operands as programs over the host formals (a formal itself, a view
over a formal's pointer with a layout of expressions, a scalar formal, a
static object), the grid, block and dynamic shared bytes as expressions, and
the compile-time formals (dtype, shape and stride entries as ints or symbol
IDs, assumed alignment; per symbol its width and divisibility) beside the
compile recipe. That is the descriptor; QuACK's jit_cache writes it next to
the .o, and a warm load returns it with the object.

Under a trace (torch/cuda/_host_trace_cute_dsl.py routes the call here):

  - the warm-up reads the kernel node off a capture of eager's own call on
    the real arguments (nothing launched, the graph discarded): the function
    handle an eager capture holds (E36), the driver's parameter layout, the
    launch image, grid, block and shared bytes;
  - the symbolic run binds the traced arguments to the formals, raises the
    descriptor's guards on exactly those facts (dtype, static sizes and
    strides, divisibility, alignment: E40), evaluates the operand programs
    over the tape's symbols, and emits a plain launch record with eager's
    function handle; the record's image at the traced values must reproduce
    eager's launch image byte for byte, else the call declines by name.

Declined (the closed-region route serves, torch/cuda/_host_trace_native.py):
a host whose launch operands the walk does not express (a TMA descriptor, a
computed pointer, a host dispatch), a program without a descriptor and no
recipe to recompile, a kernel whose driver layout differs from the declared
one.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from dataclasses import dataclass, field
from typing import Any

import torch


VERSION = 1


class Unexpressed(Exception):
    """The walk met a host construct the descriptor does not express."""


@dataclass(frozen=True)
class Formal:
    name: str
    kind: str  # "tensor" | "int" | "float" | "stream" | "none" | "constexpr"
    runtime: bool = True
    dtype: str | None = None
    shape: tuple = ()  # ints or symbol IDs ("s<n>")
    stride: tuple = ()
    assumed_align: int | None = None
    width: int | None = None  # an int formal's width; a float's 32
    constant: Any = None
    # the tvm-ffi environment stream: a stream formal not passed at the call
    env_stream: bool = False


@dataclass(frozen=True)
class Symbol:
    width: int
    divisibility: int
    name: str | None = None


@dataclass(frozen=True)
class Operand:
    """One kernel parameter of a launch, as a program over the host formals."""

    kind: str  # "formal" | "view" | "scalar" | "static"
    formal: int | None = None
    # a view's layout: the dynamic leaves as expressions (static ones from the type)
    shape: tuple = ()
    stride: tuple = ()
    # the kernel parameter's memref layout string, or the scalar's IR type
    type: str = ""


@dataclass(frozen=True)
class Launch:
    kernel: str
    operands: tuple
    grid: tuple  # three expressions
    block: tuple  # three ints
    smem: Any  # an expression ("smem_size" of the kernel, or a constant)


@dataclass
class Descriptor:
    version: int
    function_name: str
    kernels: tuple  # symbols, in kernel_info order
    formals: tuple  # Formal, in the compiled signature's order
    symbols: dict  # id -> Symbol
    launches: tuple  # Launch, in host order; empty when `declined` is set
    recipe: dict  # options, kwargs, the jit callable's identity
    declined: str | None = None
    cubin_sha256: str | None = None
    fingerprint: str | None = None

    def to_json(self) -> str:
        return json.dumps(_encode(self), sort_keys=True, indent=1)

    @staticmethod
    def from_json(text: str) -> Descriptor:
        return _decode(json.loads(text))


def _encode(value: Any) -> Any:
    if isinstance(value, (Descriptor, Formal, Symbol, Operand, Launch)):
        return {"__" + type(value).__name__: _encode(vars(value))}
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return value


_KINDS = {c.__name__: c for c in (Descriptor, Formal, Symbol, Operand, Launch)}


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if len(value) == 1:
            (key,) = value
            if key.startswith("__") and key[2:] in _KINDS:
                fields = {k: _decode(v) for k, v in value[key].items()}
                for k, v in fields.items():
                    if isinstance(v, list):
                        fields[k] = tuple(v)
                return _KINDS[key[2:]](**fields)
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(_decode(v) for v in value)
    return value


# ---------------------------------------------------------------------------
# the compile: the descriptor from the compile arguments and the host IR
# ---------------------------------------------------------------------------


def formals_of(compiled: Any, function: Any, args: tuple, kwargs: dict) -> tuple:
    """The compile-time formals (as _host_trace_cute_dsl reads them) with
    symbol IDs by SymInt identity, and the symbol table."""
    import inspect
    import types

    import cutlass
    from cutlass.cute.runtime import _FakeTensor
    from cutlass.cute.typing import SymInt

    signature = getattr(getattr(compiled, "execution_args", None), "original_signature", None)
    if signature is None:
        raise Unexpressed("the compiled program carries no signature")
    names = list(signature.parameters)
    if names and names[0] == "self" and not isinstance(function, types.FunctionType):
        names = names[1:]
    signature = signature.replace(parameters=[signature.parameters[n] for n in names])
    bound = signature.bind(*args, **{n: kwargs[n] for n in names if n in kwargs})
    bound.apply_defaults()
    runtime_names = set(compiled.execution_args.signature.parameters)
    symbols: dict = {}
    ids: dict = {}

    def symbol_id(s: Any) -> Any:
        if isinstance(s, int):
            return int(s)
        if not isinstance(s, SymInt):
            raise Unexpressed(f"a layout entry of type {type(s).__name__}")
        key = id(s)
        if key not in ids:
            ids[key] = f"s{len(ids)}"
            symbols[ids[key]] = Symbol(int(s.width), int(s.divisibility), s.symbol)
        return ids[key]

    formals = []
    for name, value in bound.arguments.items():
        passed = name in runtime_names
        annotation = signature.parameters[name].annotation
        # a Python literal for an Int32 / Int64 / Float32 formal is that runtime formal
        if type(value) in (int, float) and type(value) is not bool:
            if annotation is cutlass.Int32:
                value = cutlass.Int32(value)
            elif annotation is cutlass.Int64:
                value = cutlass.Int64(value)
            elif annotation is cutlass.Float32:
                value = cutlass.Float32(value)
        if isinstance(value, _FakeTensor):
            formals.append(
                Formal(
                    name,
                    "tensor",
                    passed,
                    dtype=str(value.element_type),
                    shape=tuple(symbol_id(s) for s in value.shape),
                    stride=tuple(symbol_id(s) for s in value.stride),
                    assumed_align=int(value._assumed_align or 1),
                )
            )
        elif type(value).__name__ == "_FakeStream":
            env = bool(getattr(value, "use_tvm_ffi_env_stream", False))
            formals.append(Formal(name, "stream", passed and not env, env_stream=env))
        elif value is None:
            formals.append(Formal(name, "none", passed))
        elif type(value) in (cutlass.Int32, cutlass.Int64):
            formals.append(
                Formal(name, "int", passed, width=32 if type(value) is cutlass.Int32 else 64)
            )
        elif type(value) is cutlass.Float32 or type(value) is float:
            formals.append(Formal(name, "float", passed, width=32))
        elif type(value) in (int, bool, str):
            formals.append(Formal(name, "constexpr", passed, constant=value))
        else:
            raise Unexpressed(f"compile-time argument {name} is a {type(value).__name__}")
    return tuple(formals), symbols


def recipe_of(function: Any, kwargs: dict, compile_options: Any) -> dict:
    """The compile recipe: the options as given, the CompileCallable's resolved
    options (every option's value, so a non-default callable is not mistaken
    for the default one), the explicit keywords, the jit callable's identity."""
    import types

    target = function if isinstance(function, types.FunctionType) else type(function)
    resolved = {}
    options = getattr(compile_options, "options", None) or {}
    for kind, option in options.items():
        value = getattr(option, "value", None)
        resolved[kind.__name__] = value if isinstance(value, (int, str, bool, float, type(None))) else str(value)
    return {
        "options": kwargs.get("options"),
        "resolved_options": resolved,
        "kwargs": tuple(sorted(k for k in kwargs if k not in ("options", "trace_finalize_hooks"))),
        "callable": {
            "module": getattr(target, "__module__", None),
            "qualname": getattr(target, "__qualname__", None),
            "instance": not isinstance(function, types.FunctionType),
        },
    }


def cubin_sha256(compiled: Any) -> str | None:
    """The kernel binary's hash: the lowered module's `kernels_binary` global (the
    cubin the host loads with cuda_load_to_device), as its printed attribute."""
    module = getattr(compiled, "ir_module", None)
    if module is None:
        return None
    found: list = []
    try:
        with raw_values():
            for op in module.body.operations:
                op = op.operation
                if op.name != "llvm.mlir.global":
                    continue
                name = str(op.attributes["sym_name"]).strip('"')
                if name.endswith("_binary") and "value" in op.attributes:
                    found.append(str(op.attributes["value"]))
    except Exception:
        return None
    if not found:
        return None
    return hashlib.sha256("".join(found).encode()).hexdigest()


_LAYOUT_RE = re.compile(r'"([^"]*)"\s*>\s*$')


def _memref_layout(type_string: str) -> str | None:
    """The layout string of a !cute.memref type (`(?,2048):(?{i64 div=8},1)`)."""
    if not type_string.startswith("!cute.memref<"):
        return None
    m = _LAYOUT_RE.search(type_string)
    return m.group(1) if m else None


def _split_modes(text: str) -> list:
    """Top-level comma-separated modes of a tuple string without its parentheses."""
    modes, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            modes.append(text[start:i])
            start = i + 1
    modes.append(text[start:])
    return [m.strip() for m in modes if m.strip()]


def _strip(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")") and _balanced(text[1:-1]):
        text = text[1:-1].strip()
    return text


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def parse_tuple(text: str) -> list:
    """A cute int tuple string as nested lists: an int, or ("?", width) for a
    dynamic leaf (`?`, `?{div=4}`, `?{i64 div=8}`)."""
    text = text.strip()
    if text.startswith("(") and text.endswith(")") and _balanced(text[1:-1]):
        return [parse_tuple(m) for m in _split_modes(text[1:-1])]
    if text.startswith("?"):
        width = 64 if re.search(r"\bi64\b", text) else 32
        return ("?", width)
    if "@" in text:  # a basis stride (a TMA coordinate tensor): not a value
        raise Unexpressed(f"a basis stride {text}")
    try:
        return int(text)
    except ValueError as error:
        raise Unexpressed(f"a tuple entry {text!r}") from error


def _tile_extent(type_string: str) -> int:
    """The extent of a one-mode static tile (`!cute.tile<"4:1">`, `<"[4:1]">`)."""
    m = re.search(r'<"([^"]*)"', type_string)
    text = (m.group(1) if m else type_string).strip().strip("[]")
    modes = [t for t in text.split(";") if t.strip()]
    if len(modes) != 1:
        raise Unexpressed(f"ceil_div by a tile of {len(modes)} modes")
    shape = modes[0].split(":")[0].strip()
    leaves = _leaves(parse_tuple(shape))
    if len(leaves) != 1 or isinstance(leaves[0], tuple):
        raise Unexpressed("ceil_div by a non-scalar tile")
    return int(leaves[0])


def _leaves(tree: Any) -> list:
    if isinstance(tree, list):
        out: list = []
        for t in tree:
            out.extend(_leaves(t))
        return out
    return [tree]


def _layout_parts(layout: str) -> tuple:
    shape, _, stride = layout.partition(":")
    if not _:
        raise Unexpressed(f"a layout without a stride {layout!r}")
    return parse_tuple(shape), parse_tuple(stride)


def dynamic_leaves(layout: str) -> list:
    """The dynamic leaves of a memref layout, in the order the kernel parameter
    struct holds them (the shape's, then the stride's): [(which, flat index, width)]."""
    shape, stride = _layout_parts(layout)
    out = []
    for which, tree in (("shape", shape), ("stride", stride)):
        for index, leaf in enumerate(_leaves(tree)):
            if isinstance(leaf, tuple):
                out.append((which, index, leaf[1]))
    return out


def struct_layout(fields: list) -> tuple:
    """Natural layout of scalar fields [(size, align)]: ([(offset, size)], total, align)."""
    offset, out, align = 0, [], 1
    for size, a in fields:
        offset = (offset + a - 1) // a * a
        out.append((offset, size))
        offset += size
        align = max(align, a)
    total = (offset + align - 1) // align * align if fields else 0
    return out, total, align


def parameter_fields(operand: Operand) -> list:
    """The scalar fields of a kernel parameter: [(name, kind, size, align)]."""
    if operand.kind in ("formal", "view"):
        fields = [("ptr", "ptr", 8, 8)]
        for which, index, width in dynamic_leaves(operand.type):
            fields.append((f"{which}[{index}]", f"i{width}", width // 8, width // 8))
        return fields
    if operand.kind == "scalar":
        kind = {"f32": ("f32", 4), "i32": ("i32", 4), "i64": ("i64", 8), "f64": ("f64", 8)}.get(
            operand.type
        )
        if kind is None:
            raise Unexpressed(f"a scalar kernel parameter of type {operand.type}")
        return [("value", kind[0], kind[1], kind[1])]
    return []


# ---------------------------------------------------------------------------
# the host walk
# ---------------------------------------------------------------------------


def _raw(value: Any) -> Any:
    """The plain ir.Value behind an operand or result: the DSL downcasts typed
    values to wrappers (a memref operand is a cute _Tensor carrying the value; a
    layout an unhashable _Layout subclassing ir.Value), so the value is recast
    to its plain class, whose hash is the underlying value's."""
    from cutlass._mlir import ir

    if not isinstance(value, ir.Value):
        inner = getattr(value, "value", None)
        if not isinstance(inner, ir.Value):
            raise Unexpressed(f"an operand of type {type(value).__name__} without an IR value")
        value = inner
    if isinstance(value.owner, ir.Block):
        return ir.BlockArgument(value)
    return ir.OpResult(value)


def _type_of(value: Any) -> str:
    return str(_raw(value).type)


class _Walk:
    """Resolves the host function's SSA values to programs over its formals."""

    def __init__(self, host: Any, formal_types: list) -> None:
        self.host = host
        self.formal_types = formal_types
        block = host.regions[0].blocks[0]
        self.arguments = {hash(_raw(a)): (i, _type_of(a)) for i, a in enumerate(block.arguments)}
        self.values: dict = {}
        self.launches: list = []

    def key(self, value: Any) -> Any:
        return hash(_raw(value))

    def resolve(self, value: Any) -> Any:
        k = self.key(value)
        if k in self.values:
            return self.values[k]
        if k in self.arguments:
            index, type_string = self.arguments[k]
            return ("formal", index, type_string)
        raise Unexpressed(f"a value the walk did not define ({_type_of(value)[:60]})")

    def define(self, op: Any, program: Any) -> None:
        for result in op.results:
            self.values[self.key(result)] = program

    def const_of(self, type_string: str) -> Any:
        m = re.search(r'<"([^"]*)"', type_string)
        if m is None:
            raise Unexpressed(f"a static value of type {type_string[:60]}")
        return parse_tuple(m.group(1))

    def scalar(self, program: Any) -> Any:
        """A program as a scalar expression (an int leaf), or raise."""
        if isinstance(program, tuple) and program and program[0] in (
            "const",
            "size",
            "stride",
            "scalar",
            "add",
            "sub",
            "mul",
            "div",
            "ceil_div",
            "smem_size",
        ):
            return program
        if isinstance(program, int):
            return ("const", program)
        if isinstance(program, tuple) and program and program[0] == "formal" and program[2] in ("i32", "i64", "f32", "f64"):
            return ("scalar", program[1])
        if isinstance(program, tuple) and program and program[0] == "tuple" and len(program[1]) == 1:
            return self.scalar(program[1][0])
        raise Unexpressed(f"a non-scalar program {str(program)[:80]}")

    def visit(self, op: Any) -> None:
        op = op.operation  # a dialect OpView's .name may be its symbol, not the op name
        name = op.name
        results = list(op.results)
        operands = list(op.operands)
        if name == "arith.constant":
            value = op.attributes["value"]
            text = str(value).split(":")[0].strip()
            try:
                self.define(op, ("const", int(text)))
            except ValueError:
                try:
                    self.define(op, ("fconst", float(text)))
                except ValueError as error:
                    raise Unexpressed(f"a constant {text!r}") from error
            return
        if name == "cute.get_iter":
            src = self.resolve(operands[0])
            if src[0] == "formal":
                self.define(op, ("iter", src[1]))
                return
            if src[0] == "view":
                self.define(op, ("iter", src[1]))
                return
            raise Unexpressed("get_iter of a computed tensor")
        if name == "cute.get_layout":
            src = self.resolve(operands[0])
            if src[0] == "formal":
                self.define(op, ("layout_of", src[1], _memref_layout(src[2])))
                return
            if src[0] == "view":
                self.define(op, ("layout", src[3], src[4], src[5]))
                return
            raise Unexpressed("get_layout of a computed tensor")
        if name in ("cute.get_shape", "cute.get_stride"):
            src = self.resolve(operands[0])
            which = "shape" if name == "cute.get_shape" else "stride"
            if src[0] == "static":
                self.define(op, self._static_tuple_of(_type_of(results[0])))
                return
            if src[0] == "layout_of":
                shape, stride = _layout_parts(src[2])
                tree = shape if which == "shape" else stride
                leaves = _leaves(tree)
                kind = "size" if which == "shape" else "stride"
                exprs = tuple(
                    (kind, src[1], i) if isinstance(leaf, tuple) else ("const", leaf)
                    for i, leaf in enumerate(leaves)
                )
                self.define(op, ("tuple", exprs, _type_of(results[0])))
                return
            if src[0] == "layout":
                exprs = src[1] if which == "shape" else src[2]
                self.define(op, ("tuple", tuple(exprs), _type_of(results[0])))
                return
            raise Unexpressed(f"{name} of a computed layout")
        if name == "cute.get_leaves":
            src = self.resolve(operands[0])
            if src[0] == "static":
                try:
                    src = self._static_tuple_of(src[1])
                except Unexpressed:
                    # the leaves of a static tile or layout: static objects themselves
                    for result in results:
                        self.values[self.key(result)] = ("static", _type_of(result))
                    return
            if src[0] != "tuple":
                raise Unexpressed("get_leaves of a non-tuple")
            exprs = src[1]
            if len(exprs) != len(results):
                raise Unexpressed("get_leaves arity")
            for result, expr in zip(results, exprs):
                self.values[self.key(result)] = ("tuple", (expr,), _type_of(result))
            return
        if name in ("cute.to_int_tuple", "cute.get_scalars"):
            src = self.resolve(operands[0])
            if src[0] == "tuple" and len(src[1]) == 1:
                self.define(op, src if name == "cute.to_int_tuple" else self.scalar(src[1][0]))
                return
            if name == "cute.get_scalars" and src[0] == "tuple":
                if len(results) != len(src[1]):
                    raise Unexpressed("get_scalars arity")
                for result, expr in zip(results, src[1]):
                    self.values[self.key(result)] = self.scalar(expr)
                return
            self.define(op, src)
            return
        if name in ("cute.make_int_tuple", "cute.make_shape", "cute.make_stride", "cute.make_tile"):
            type_string = _type_of(results[0])
            if not operands and name == "cute.make_tile":
                self.define(op, ("static", type_string))
                return
            tree = self.const_of(type_string)
            leaves = _leaves(tree)
            dynamic = [i for i, leaf in enumerate(leaves) if isinstance(leaf, tuple)]
            if len(dynamic) != len(operands):
                raise Unexpressed(f"{name}: {len(operands)} operands for {len(dynamic)} dynamic leaves")
            exprs = []
            it = iter(operands)
            for leaf in leaves:
                if isinstance(leaf, tuple):
                    src = self.resolve(next(it))
                    exprs.append(self.scalar(src[1][0] if src[0] == "tuple" else src))
                else:
                    exprs.append(("const", leaf))
            self.define(op, ("tuple", tuple(exprs), type_string))
            return
        if name == "cute.make_layout":
            shape = self.resolve(operands[0])
            m = re.search(r'<"([^"]*)"', _type_of(results[0]))
            layout = m.group(1) if m else ""
            shape_text, _, stride_text = layout.partition(":")
            if shape[0] != "tuple":
                raise Unexpressed("make_layout of a non-tuple shape")
            if len(operands) > 1:
                stride = self.resolve(operands[1])
                if stride[0] != "tuple":
                    raise Unexpressed("make_layout of a non-tuple stride")
                stride_exprs = stride[1]
            else:
                # a compact layout inferred from the shape: its leaves from the type
                leaves = _leaves(parse_tuple(stride_text))
                if any(isinstance(leaf, tuple) for leaf in leaves):
                    raise Unexpressed("a compact layout with dynamic strides")
                stride_exprs = tuple(("const", leaf) for leaf in leaves)
            self.define(op, ("layout", tuple(shape[1]), tuple(stride_exprs), layout))
            return
        if name == "cute.make_view":
            it = self.resolve(operands[0])
            layout = self.resolve(operands[1])
            if it[0] != "iter" or layout[0] != "layout":
                raise Unexpressed("make_view of a computed iterator or layout")
            self.define(
                op,
                ("view", it[1], _type_of(results[0]), layout[1], layout[2], layout[3]),
            )
            return
        if name in ("cute.tuple_add", "cute.tuple_sub", "cute.tuple_mul", "cute.tuple_div"):
            a, b = (self.resolve(v) for v in operands)
            a = self.scalar(a[1][0] if a[0] == "tuple" else a)
            b = self.scalar(b[1][0] if b[0] == "tuple" else b)
            self.define(op, ("tuple", ((name.split("_")[1], a, b),), _type_of(results[0])))
            return
        if name == "cute.ceil_div":
            a, b = (self.resolve(v) for v in operands)
            a = self.scalar(a[1][0] if a[0] == "tuple" else a)
            if b[0] == "tuple":
                b = self.scalar(b[1][0])
            elif b[0] == "static":
                b = ("const", _tile_extent(b[1]))
            else:
                b = self.scalar(b)
            self.define(op, ("tuple", (("ceil_div", a, b),), _type_of(results[0])))
            return
        if name in ("arith.addi", "arith.subi", "arith.muli", "arith.divsi", "arith.divui", "arith.ceildivsi"):
            a, b = (self.scalar(self.resolve(v)) for v in operands)
            kind = {"arith.addi": "add", "arith.subi": "sub", "arith.muli": "mul", "arith.divsi": "div", "arith.divui": "div", "arith.ceildivsi": "ceil_div"}[name]
            self.define(op, (kind, a, b))
            return
        if name in ("arith.index_cast", "arith.extsi", "arith.extui", "arith.trunci", "arith.index_castui"):
            self.define(op, self.scalar(self.resolve(operands[0])))
            return
        if name == "cute.kernel_smem_size":
            kernel = str(op.attributes["kernel"]) if "kernel" in op.attributes else str(op)
            m = re.search(r"@kernels::@([A-Za-z0-9_]+)", str(op))
            self.define(op, ("smem_size", m.group(1) if m else kernel))
            return
        if name == "cute.static" or (not operands and results and name.startswith("cute")):
            self.define(op, ("static", _type_of(results[0]) if results else ""))
            return
        if name.startswith("cute.") and results:
            # a static object built from static objects (a tiled copy, a layout
            # product): no runtime value
            srcs = [self.resolve(v) for v in operands]
            if all(self._static_tuple(s) for s in srcs):
                self.define(op, ("static", _type_of(results[0])))
                return
            raise Unexpressed(f"{name} over runtime values")
        if name == "cuda.launch_cfg.create":
            named = self._launch_cfg(op)
            self.define(op, ("cfg", named))
            return
        if name.startswith("cuda.launch_cfg."):
            return
        if name == "cuda.launch_ex":
            cfg = self.resolve(operands[0])
            if cfg[0] != "cfg":
                raise Unexpressed("launch_ex without a launch_cfg")
            m = re.search(r"@kernels::@([A-Za-z0-9_]+)", str(op))
            if m is None:
                raise Unexpressed("launch_ex without a kernel symbol")
            self.launches.append((m.group(1), [self.resolve(v) for v in operands[1:]], cfg[1]))
            return
        if name in ("scf.if", "cute.print", "cuda.cast", "cuda.return_if_error", "func.return", "arith.cmpi"):
            # the shared-memory bound check and the return: no value the launch reads
            if results:
                self.define(op, ("host", name))
            return
        if results:
            self.define(op, ("host", name))

    def _static_tuple_of(self, type_string: str) -> tuple:
        """A static int tuple / shape / stride type as a tuple of constants."""
        leaves = _leaves(self.const_of(type_string))
        if any(isinstance(leaf, tuple) for leaf in leaves):
            raise Unexpressed(f"a static value with dynamic leaves {type_string[:60]}")
        return ("tuple", tuple(("const", int(leaf)) for leaf in leaves), type_string)

    def _static_tuple(self, program: Any) -> bool:
        if program[0] in ("static", "const", "fconst"):
            return True
        if program[0] == "tuple":
            return all(e[0] == "const" for e in program[1])
        if program[0] == "layout":
            return all(e[0] == "const" for e in (*program[1], *program[2]))
        return False

    def _launch_cfg(self, op: Any) -> dict:
        operands = list(op.operands)
        # the operand segments: blockDim (3), dynamicSmemBytes (1), gridDim (3), stream (1)
        segments = op.attributes["operandSegmentSizes"] if "operandSegmentSizes" in op.attributes else None
        text = str(op)
        order = [m.group(1) for m in re.finditer(r"(blockDim|dynamicSmemBytes|gridDim|stream|clusterDim)\s*=", text)]
        sizes = {"blockDim": 3, "gridDim": 3, "clusterDim": 3, "dynamicSmemBytes": 1, "stream": 1}
        if segments is not None:
            counts = [int(x) for x in re.findall(r"-?\d+", str(segments))]
        else:
            counts = [sizes[k] for k in order]
        if len(counts) != len(order) or sum(counts) != len(operands):
            raise Unexpressed("launch_cfg operand segments")
        named: dict = {}
        cursor = 0
        for key, count in zip(order, counts):
            named[key] = operands[cursor : cursor + count]
            cursor += count
        out: dict = {}
        out["block"] = tuple(self._const(self.resolve(v)) for v in named.get("blockDim", ()))
        out["grid"] = tuple(self.scalar(self.resolve(v)) for v in named.get("gridDim", ()))
        smem = named.get("dynamicSmemBytes", ())
        out["smem"] = self.scalar(self.resolve(smem[0])) if smem else ("const", 0)
        if named.get("clusterDim"):
            raise Unexpressed("a cluster launch")
        return out

    def _const(self, program: Any) -> int:
        program = self.scalar(program)
        if program[0] != "const":
            raise Unexpressed("a block dimension that is not a constant")
        return int(program[1])


def _kernel_parameter_types(module: Any) -> dict:
    """kernel symbol -> the declared parameter types (strings), from the gpu module."""
    out: dict = {}
    for op in module.body.operations:
        op = op.operation
        if op.name != "gpu.module":
            continue
        for kernel in op.regions[0].blocks[0].operations:
            kernel = kernel.operation
            if kernel.name not in ("cuda.kernel", "gpu.func"):
                continue
            sym = str(kernel.attributes["sym_name"]).strip('"')
            types = [_type_of(a) for a in kernel.regions[0].blocks[0].arguments]
            out[sym] = types
    return out


def raw_values() -> Any:
    """The runtime SDK's raw value reads (torch/_inductor/runtime/_cudagraph/_sdk.py,
    active once _sdk.activate() ran before cutlass was imported): inside it the
    DSL's value casters return plain ir.Values. Without it a read of a memref
    value builds a cute.Tensor wrapper, whose constructor emits a get_iter op
    into the module under construction: a walk must not run then."""
    from cutlass._mlir import ir

    context = getattr(ir, "raw_values", None)
    if context is None:
        raise Unexpressed(
            "the walk needs the runtime SDK's raw value reads (torch._inductor.runtime._cudagraph._sdk.activate() "
            "before cutlass is imported)"
        )
    return context()


def available() -> bool:
    """Whether descriptors can be built in this process (the SDK is activated)."""
    module = sys.modules.get("cutlass._mlir.ir")
    return module is not None and getattr(module, "raw_values", None) is not None


def extract_ir(module: Any, function_name: str) -> tuple:
    """The launches of the host `function_name` in the finalize-time module (walked
    inside the DSL's finalize hook: the lowering then rewrites the module in place),
    over the host's IR formal indices: (launches, IR formal types). Raises
    Unexpressed with the construct the walk does not express."""
    with raw_values():
        return _extract_ir(module, function_name)


def _extract_ir(module: Any, function_name: str) -> tuple:
    host = None
    for op in module.body.operations:
        op = op.operation
        if op.name == "func.func" and str(op.attributes["sym_name"]).strip('"') == function_name:
            host = op
    if host is None:
        raise Unexpressed("the host function is not in the finalize module")
    block = host.regions[0].blocks[0]
    formal_types = [_type_of(a) for a in block.arguments]
    kernel_types = _kernel_parameter_types(module)
    walk = _Walk(host, formal_types)
    for op in block.operations:
        walk.visit(op)
    if not walk.launches:
        raise Unexpressed("no kernel launch in the host")
    launches = []
    for kernel, operands, cfg in walk.launches:
        types = kernel_types.get(kernel)
        if types is None or len(types) != len(operands):
            raise Unexpressed(f"kernel {kernel[:60]} declares {types and len(types)} parameters, launched with {len(operands)}")
        out = []
        for program, type_string in zip(operands, types):
            if program[0] == "formal":
                layout = _memref_layout(type_string)
                if layout is None:
                    if type_string not in ("f32", "i32", "i64", "f64"):
                        raise Unexpressed(f"a formal of type {type_string} as a kernel parameter")
                    out.append(Operand("scalar", program[1], type=type_string))
                    continue
                if _memref_layout(program[2]) != layout:
                    raise Unexpressed("a formal passed to a kernel parameter of another layout")
                out.append(Operand("formal", program[1], type=layout))
            elif program[0] == "view":
                layout = _memref_layout(type_string)
                if layout is None or program[5] != layout:
                    raise Unexpressed("a view passed to a kernel parameter of another layout")
                out.append(Operand("view", program[1], shape=tuple(program[3]), stride=tuple(program[4]), type=layout))
            elif program[0] == "static":
                if _memref_layout(type_string) is not None:
                    raise Unexpressed("a static object for a memref parameter")
                out.append(Operand("static", type=type_string))
            elif program[0] in ("const", "fconst"):
                raise Unexpressed("a constant kernel parameter")
            else:
                raise Unexpressed(f"a kernel parameter computed by {program[0]}")
        for operand in out:
            parameter_fields(operand)  # raises for a layout the struct does not express
        launches.append(Launch(kernel, tuple(out), tuple(cfg["grid"]), tuple(cfg["block"]), cfg["smem"]))
    return tuple(launches), tuple(formal_types)


def rebase(launches: tuple, formal_types: tuple, formals: tuple) -> tuple:
    """IR formal indices -> the compiled signature's formal indices."""
    ir_formals = [f for f in formals if f.kind in ("tensor", "int", "float", "stream")]
    if len(ir_formals) != len(formal_types):
        raise Unexpressed(f"{len(formal_types)} IR formals for {len(ir_formals)} signature formals")
    index_of = {i: formals.index(f) for i, f in enumerate(ir_formals)}
    out = []
    for L in launches:
        operands = tuple(
            Operand(
                o.kind,
                None if o.formal is None else index_of[o.formal],
                tuple(_rebase(e, index_of) for e in o.shape),
                tuple(_rebase(e, index_of) for e in o.stride),
                o.type,
            )
            for o in L.operands
        )
        out.append(Launch(L.kernel, operands, tuple(_rebase(e, index_of) for e in L.grid), L.block, _rebase(L.smem, index_of)))
    return tuple(out)


def _rebase(expr: Any, index_of: dict) -> Any:
    """IR formal indices in an expression -> signature formal indices."""
    if not isinstance(expr, tuple):
        return expr
    head = expr[0]
    if head in ("size", "stride"):
        return (head, index_of[expr[1]], expr[2])
    if head == "scalar":
        return (head, index_of[expr[1]])
    if head == "formal":
        return ("scalar", index_of[expr[1]])
    if head in ("add", "sub", "mul", "div", "ceil_div"):
        return (head, _rebase(expr[1], index_of), _rebase(expr[2], index_of))
    return expr


def build(compiled: Any, function: Any, args: tuple, kwargs: dict, extracted: Any, compile_options: Any) -> Descriptor:
    """The descriptor of a compile the recorder observed: `extracted` is what
    extract_ir returned at the finalize hook, or the Unexpressed it raised."""
    from torch._vendor.quack.cache.jit import _compute_source_fingerprint

    formals, symbols = formals_of(compiled, function, args, kwargs)
    declined = None
    launches: tuple = ()
    if isinstance(extracted, Exception):
        declined = str(extracted)
    elif extracted is None:
        declined = "the compile's finalize was not observed"
    else:
        try:
            launches = rebase(extracted[0], extracted[1], formals)
        except Unexpressed as error:
            declined = str(error)
    try:
        fingerprint = _compute_source_fingerprint()
    except Exception:
        fingerprint = None
    return Descriptor(
        VERSION,
        compiled.function_name,
        tuple(compiled.kernel_info or ()),
        formals,
        symbols,
        launches,
        recipe_of(function, kwargs, compile_options),
        declined,
        cubin_sha256(compiled),
        fingerprint,
    )


# ---------------------------------------------------------------------------
# the trace: evaluation over the tape's values
# ---------------------------------------------------------------------------


def evaluate(expr: Any, env: Any) -> Any:
    """An expression at the values `env` gives: env.size(formal, axis),
    env.stride(formal, axis), env.scalar(formal), env.smem_size(kernel)."""
    head = expr[0]
    if head == "const":
        return int(expr[1])
    if head == "size":
        return env.size(expr[1], expr[2])
    if head == "stride":
        return env.stride(expr[1], expr[2])
    if head == "scalar":
        return env.scalar(expr[1])
    if head == "smem_size":
        return env.smem_size(expr[1])
    a, b = evaluate(expr[1], env), evaluate(expr[2], env)
    if head == "add":
        return a + b
    if head == "sub":
        return a - b
    if head == "mul":
        return a * b
    if head == "div":
        return a // b
    if head == "ceil_div":
        return (a + b - 1) // b
    raise Unexpressed(f"expression {head}")


def image_of(fields: list, values: list, total: int) -> bytes:
    """The parameter image of one launch at concrete values."""
    from torch.cuda._host_trace import _pack

    image = bytearray(total)
    for (offset, size), (kind, value) in zip(fields, values):
        image[offset : offset + size] = _pack(kind, value)
    return bytes(image)


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


# ---------------------------------------------------------------------------
# the trace: eager's launch at the warm-up, the record at the symbolic run
# ---------------------------------------------------------------------------


@dataclass
class Program:
    """A program the recorder knows by its descriptor: the object it launches
    through (a loaded tvm_ffi.Function or the in-process compiled function,
    kept alive with the module that owns it) and its display identity."""

    descriptor: Descriptor
    name: str
    module: str
    keep_alive: Any = None


@dataclass(frozen=True)
class DescLaunch:
    """What a descriptor-route launch record carries under its "cute_desc" key
    for the lowering: the program (its module stays loaded), eager's function
    handle read at the warm-up, and the launch's index in the descriptor."""

    program: Program
    function: int
    index: int


_side_streams: dict = {}


def read_eager_launches(call: Any, device: int) -> list:
    """The kernel nodes of eager's own call: a capture of `call` on a side
    stream of `device` (the call's launch goes to the current stream, which is
    the capture's; nothing runs, the graph is discarded after the read), as
    torch._C._host_trace_harvest_nodes reports them: func (the CUfunction an
    eager capture's node holds), name, grid, block, smem, image, layout."""
    if torch.cuda.is_current_stream_capturing():
        raise Unexpressed("the warm-up ran inside a stream capture")
    stream = _side_streams.get(device)
    if stream is None:
        stream = _side_streams[device] = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    done = False
    with torch.cuda.device(device), torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode="thread_local")
        try:
            call()
            done = True
        finally:
            try:
                graph.capture_end()
            except Exception:
                if done:
                    raise
    nodes = torch._C._host_trace_harvest_nodes(graph.raw_cuda_graph(), -1, False)
    kernels = [n for n in nodes if n["kind"] == "kernel"]
    if len(kernels) != len(nodes):
        raise Unexpressed("eager's call captured a node that is not a kernel")
    return kernels


_CUTE_DTYPES = {
    "Float32": torch.float32,
    "Float16": torch.float16,
    "BFloat16": torch.bfloat16,
    "Float64": torch.float64,
    "Int8": torch.int8,
    "Int16": torch.int16,
    "Int32": torch.int32,
    "Int64": torch.int64,
    "Uint8": torch.uint8,
    "Boolean": torch.bool,
}


def _strides(tensor: Any) -> list:
    return list(tensor._sym_strides) if hasattr(tensor, "_root") else list(tensor.stride())


def _address(tensor: Any) -> Any:
    """A tensor's data address as a value: the root's symbol plus the view's
    offset for a traced tensor, the real address for the warm-up's."""
    if hasattr(tensor, "_root"):
        return tensor._root.sym + tensor._sym_offset * tensor.element_size()
    return tensor.data_ptr()


def _alignment_probe(tensor: Any) -> Any:
    # an allocation's base is 256-byte aligned by construction: its offset decides
    if hasattr(tensor, "_root") and tensor._root.allocation:
        return tensor._sym_offset * tensor.element_size()
    return _address(tensor)


class _Bound:
    """A call's arguments bound to the descriptor's formals: the traced ones at
    the symbolic run, the real ones at the warm-up (the verification)."""

    def __init__(self) -> None:
        self.tensors: dict = {}  # formal index -> (tensor, read_only)
        self.scalars: dict = {}  # formal index -> value (int / SymInt / float)
        self.eager_smem: dict = {}  # kernel -> smem of eager's node

    def size(self, formal: int, axis: int) -> Any:
        return self.tensors[formal][0].shape[axis]

    def stride(self, formal: int, axis: int) -> Any:
        return _strides(self.tensors[formal][0])[axis]

    def scalar(self, formal: int) -> Any:
        return self.scalars[formal]

    def smem_size(self, kernel: str) -> Any:
        return self.eager_smem[kernel]


def _bind_call(ht: Any, tr: Any, descriptor: Descriptor, args: tuple, kwargs: dict, operand: Any, decline: Any) -> _Bound:
    """The call's arguments as the formals; the descriptor's guards raised on
    exactly the facts the compile fixed: dtype, static sizes and strides, a
    symbol's divisibility and width, a recurring symbol's equality, the
    assumed alignment of each tensor's address."""
    import sympy

    runtime = [f for f in descriptor.formals if f.runtime]
    values: dict = {}
    if len(args) > len(runtime):
        decline(f"{len(args)} positional arguments for {len(runtime)} runtime formals")
    for formal, value in zip(runtime, args):
        values[formal.name] = value
    names = {f.name for f in runtime}
    for key, value in kwargs.items():
        if key not in names or key in values:
            decline(f"argument {key} is not a runtime formal of the compiled program, or given twice")
        values[key] = value
    missing = [f.name for f in runtime if f.name not in values]
    if missing:
        decline(f"arguments {missing} are missing")
    bound = _Bound()
    seen: dict = {}  # symbol id -> the value it stands for (the first)

    def fact(name: str, value: Any, entry: Any, symbols: dict) -> None:
        if isinstance(entry, int):
            if not bool(value == entry):
                decline(f"{name} is {ht._hint(value)} at the traced call; the compiled program holds {entry}")
            return
        symbol = symbols[entry]
        if symbol.divisibility > 1 and not bool(value % symbol.divisibility == 0):
            decline(f"{name} is {ht._hint(value)} at the traced call; the compiled program assumes a multiple of {symbol.divisibility}")
        if symbol.width == 32 and not bool(value < 1 << 31):
            decline(f"{name} does not fit the compiled program's 32-bit symbol")
        first = seen.get(entry)
        if first is None:
            seen[entry] = value
        elif not bool(first == value):
            decline(f"{name} is {ht._hint(value)} at the traced call; the compiled program shares its symbol with a value of {ht._hint(first)}")

    for index, formal in enumerate(descriptor.formals):
        if not formal.runtime:
            continue
        value = values[formal.name]
        if formal.kind == "tensor":
            tensor, read_only = operand(value)
            dtype = _CUTE_DTYPES.get(formal.dtype)
            if dtype is None or tensor.dtype != dtype:
                decline(f"tensor argument {formal.name} is {tensor.dtype}; the compiled program takes {formal.dtype}")
            if tr is not None and tensor.device != tr.device:
                decline(f"tensor argument {formal.name} is on {tensor.device}")
            if tensor.dim() != len(formal.shape):
                decline(f"tensor argument {formal.name} has rank {tensor.dim()}; the compiled program takes {len(formal.shape)}")
            for axis, entry in enumerate(formal.shape):
                fact(f"{formal.name}.shape[{axis}]", tensor.shape[axis], entry, descriptor.symbols)
            strides = _strides(tensor)
            for axis, entry in enumerate(formal.stride):
                fact(f"{formal.name}.stride[{axis}]", strides[axis], entry, descriptor.symbols)
            align = int(formal.assumed_align or 1)
            if align > 1 and not bool(_alignment_probe(tensor) % align == 0):
                decline(f"tensor argument {formal.name} is not {align}-byte aligned at the traced call, which the compiled program assumes")
            bound.tensors[index] = (tensor, read_only)
        elif formal.kind == "int":
            if hasattr(value, "value") and not isinstance(value, (int, torch.SymInt)):
                value = value.value
            if type(value) is not int and type(value) is not torch.SymInt:
                decline(f"integer argument {formal.name} is a {type(value).__name__}")
            if type(value) is torch.SymInt and value.node.shape_env is not tr.shape_env:
                decline(f"integer argument {formal.name} belongs to another ShapeEnv")
            from torch.cuda import _host_trace_cute

            if tr is not None and _host_trace_cute._mentions_root(ht, tr, value):
                decline(f"integer argument {formal.name} is derived from a data pointer; the tape takes integers as values, not addresses")
            if formal.width == 32 and not (bool(value < 1 << 31) and bool(value >= -(1 << 31))):
                decline(f"integer argument {formal.name} does not fit the compiled program's 32-bit formal")
            bound.scalars[index] = value
        elif formal.kind == "float":
            value = value.value if hasattr(value, "value") else value
            if isinstance(value, (torch.SymFloat, torch.SymInt)):
                decline(f"float argument {formal.name} is symbolic; the tape has no float symbols")
            try:
                value = float(value)
            except (TypeError, ValueError):
                decline(f"float argument {formal.name} is a {type(value).__name__}")
            if value != value or value in (float("inf"), float("-inf")):
                decline(f"float argument {formal.name} is not finite")
            bound.scalars[index] = value
        elif formal.kind == "stream":
            if tr is not None and int(value) != torch.cuda.current_stream().cuda_stream:
                decline("invoked on a stream other than the trace's capturing stream")
        elif formal.kind == "none":
            if value is not None:
                decline(f"argument {formal.name} was None at the compile and is not now")
        elif formal.kind == "constexpr":
            if value != formal.constant:
                decline(f"argument {formal.name} is {value!r} at this call; the program was compiled with {formal.constant!r}")
    return bound


def _launch_params(descriptor: Descriptor, launch: Launch, bound: _Bound, decline: Any) -> tuple:
    """The parameter rows of one launch over a binding, with the declared
    layout: ([rows], [(offset, size)] per driver parameter, the image's extent,
    the written roots' tensors)."""
    fields: list = []
    sizes: list = []
    for operand_ in launch.operands:
        entry = parameter_fields(operand_)
        if not entry:
            continue
        offsets, total, align = struct_layout([(size, a) for _, _, size, a in entry])
        fields.append((operand_, entry, offsets))
        sizes.append((total, align))
    layout, _, _ = struct_layout(sizes)
    # the driver's image ends with the last parameter (no tail padding)
    total = layout[-1][0] + layout[-1][1] if layout else 0
    params: list = []
    written: list = []
    for (operand_, entry, offsets), (base, _) in zip(fields, layout):
        if operand_.kind in ("formal", "view"):
            tensor, read_only = bound.tensors[operand_.formal]
            label = descriptor.formals[operand_.formal].name
            leaves = dynamic_leaves(operand_.type)
            for (_, kind, size, _), (inner, _), leaf in zip(entry, offsets, [None, *leaves]):
                if kind == "ptr":
                    params.append({"offset": base + inner, "size": size, "kind": "ptr", "value": _address(tensor), "name": f"{label}.data_ptr", "access": "r" if read_only else "rw"})
                    if not read_only:
                        written.append(tensor)
                    continue
                which, axis, _ = leaf
                if operand_.kind == "formal":
                    value = bound.size(operand_.formal, axis) if which == "shape" else bound.stride(operand_.formal, axis)
                else:
                    value = evaluate((operand_.shape if which == "shape" else operand_.stride)[axis], bound)
                params.append({"offset": base + inner, "size": size, "kind": kind, "value": value, "name": f"{label}.{which}[{axis}]", "access": ""})
        else:  # a scalar formal
            (_, kind, size, _), (inner, _) = entry[0], offsets[0]
            params.append({"offset": base + inner, "size": size, "kind": kind, "value": bound.scalars[operand_.formal], "name": descriptor.formals[operand_.formal].name, "access": ""})
    return params, layout, total, written


def _check_launch(descriptor: Descriptor, launch: Launch, index: int, node: dict, bound: _Bound, decline: Any, pointers: bool, values: bool = True) -> tuple:
    """One launch's rows against eager's node: the kernel, the driver's layout,
    the block, and with `values` the grid, shared bytes and every named byte of
    the image at the binding's values (the pointer leaves only when `pointers`:
    at the warm-up the binding is the real one, at the symbolic run an address
    hint is not an address; `values` is off for a trace without a warm-up,
    whose node was read and verified at another call's values)."""
    from torch.cuda import _host_trace as ht

    if node["name"] != launch.kernel:
        decline(f"eager launched {node['name'][:60]}; the descriptor's launch {index} is {launch.kernel[:60]}")
    params, layout, total, written = _launch_params(descriptor, launch, bound, decline)
    if [tuple(x) for x in node["layout"]] != layout:
        decline(f"the driver's parameter layout {node['layout']} differs from the declared one {layout}")
    grid = tuple(evaluate(e, bound) for e in launch.grid)
    for axis, (extent, limit) in enumerate(zip(grid, (2**31 - 1, 65535, 65535))):
        if not (bool(extent >= 1) and bool(extent <= limit)):
            decline(f"grid axis {axis} is outside the launch bounds at the traced call")
    block = tuple(int(b) for b in launch.block)
    smem = evaluate(launch.smem, bound)
    if tuple(node["block"]) != block:
        decline(f"eager's block {node['block']} differs from the descriptor's {block}")
    if values and (int(node["smem"]) != int(ht._hint(smem)) or tuple(node["grid"]) != tuple(int(ht._hint(g)) for g in grid)):
        decline(f"eager's launch configuration grid {node['grid']} smem {node['smem']} differs from the descriptor's at the call's values ({[ht._hint(g) for g in grid]}, {ht._hint(smem)})")
    image = bytes(node["image"])
    if len(image) != total:
        decline(f"eager's parameter image is {len(image)} bytes; the declared layout is {total}")
    for p in params:
        if not values or (p["kind"] == "ptr" and not pointers):
            continue
        packed = ht._pack(p["kind"], ht._hint(p["value"]))
        if image[p["offset"] : p["offset"] + p["size"]] != packed:
            decline(f"the descriptor's {p['name']} does not reproduce eager's launch image at byte {p['offset']} (eager {image[p['offset']:p['offset'] + p['size']].hex()}, descriptor {packed.hex()})")
    return params, layout, total, grid, block, smem, written


def verify_eager(descriptor: Descriptor, args: tuple, kwargs: dict, eager: list, operand: Any, name: str) -> None:
    """At the warm-up: eager's own launch (the nodes read off its capture)
    against the descriptor over the real arguments, every named byte of each
    image included. Raises Unexpressed with the first mismatch."""
    from torch.cuda import _host_trace as ht

    def decline(why: str) -> Any:
        raise Unexpressed(why)

    if descriptor.declined is not None or not descriptor.launches:
        decline(f"its descriptor does not express the launch ({descriptor.declined})")
    if len(eager) != len(descriptor.launches):
        decline(f"eager's call launched {len(eager)} kernels; the descriptor holds {len(descriptor.launches)}")
    bound = _bind_call(ht, None, descriptor, args, kwargs, operand, decline)
    for node, launch in zip(eager, descriptor.launches):
        bound.eager_smem[launch.kernel] = int(node["smem"])
    for index, (node, launch) in enumerate(zip(eager, descriptor.launches)):
        _check_launch(descriptor, launch, index, node, bound, decline, pointers=True)


def record(tr: Any, program: Program, args: tuple, kwargs: dict, eager: list, operand: Any, name: str, verify: bool = True) -> None:
    """The launch records of one call of `program` on the traced arguments:
    one plain record per launch of the descriptor, with eager's function
    handle (`eager`: the nodes read at the warm-up, verified there against
    the real arguments, in launch order)."""
    from torch.cuda import _host_trace as ht

    descriptor = program.descriptor

    def decline(why: str) -> Any:
        raise ht.Declined(f"host_trace: CuTe DSL kernel {name}: {why} (declined)")

    if descriptor.declined is not None or not descriptor.launches:
        decline(f"its descriptor does not express the launch ({descriptor.declined})")
    if len(eager) != len(descriptor.launches):
        decline(f"eager's call launched {len(eager)} kernels; the descriptor holds {len(descriptor.launches)}")
    bound = _bind_call(ht, tr, descriptor, args, kwargs, operand, decline)
    for node, launch in zip(eager, descriptor.launches):
        bound.eager_smem[launch.kernel] = int(node["smem"])
    ct = tr.cute
    for index, (node, launch) in enumerate(zip(eager, descriptor.launches)):
        params, layout, total, grid, block, smem, written = _check_launch(descriptor, launch, index, node, bound, decline, pointers=False, values=verify)
        image = bytearray(bytes(node["image"]))
        for p in params:
            image[p["offset"] : p["offset"] + p["size"]] = ht._pack(p["kind"], ht._hint(p["value"]))
        ct.launches.append(
            {
                "seq": tr.rec.next_seq(),
                "kernel": launch.kernel,
                "func": int(node["func"]),
                "param_layout": list(layout),
                "grid": grid,
                "block": block,
                "block_expr": block,
                "smem": int(smem) if not isinstance(smem, torch.SymInt) else smem,
                "params": params,
                "hint_image": bytes(image),
                "cute_desc": DescLaunch(program, int(node["func"]), index),
            }
        )
        for tensor in written:
            if tensor._root.name not in ct.written_roots:
                ct.written_roots.append(tensor._root.name)
