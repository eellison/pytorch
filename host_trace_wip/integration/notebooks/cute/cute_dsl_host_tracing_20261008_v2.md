# Dynamic Shape CUDA Graphs with CuTe DSL

> **AI-generated draft** (Claude Code, for Elias to review and edit). Nothing here has been sent to the CuTe DSL team.

**Background:** this is a short example of how dynamic-shape CUDA graphs can trace through CuTe DSL kernels, detect guards, and replay at varying sizes and memory addresses. A companion notebook (dynamic_shape_cudagraphs_triton_20261008.ipynb) does the same for Triton.

We start from a standard user CuTe DSL kernel. From there, we need three things:

- A tracer that runs user code with symbolic shapes and memory addresses, and records guards.
- Introspection into CuTe DSL's compile, to determine which guards the compiled function relies on and how our symbolic arguments map to kernel parameters in the CUDA graph. Unlike Triton, a CuTe DSL host function is compiled code, so both come from what the compile produces: the guards from TVM-FFI's argument spec (what the compiled function's argument checker, its *binder*, checks), and the launch from the host function's MLIR, read through MLIR's Python bindings, with CuTe layouts evaluated by the DSL's own layout algebra.
- A runtime that checks guards, does memory allocations, computes the size arithmetic, and then updates the CUDA graph's parameters.

We intentionally don't handle CuTe DSL's full feature set, to keep this concise. In the future we'll work with the CuTe DSL team on the best APIs CuTe DSL could expose for the same functionality (section 8).

The first call at new inputs traces the user code: it runs it once as a warm-up (which compiles what these inputs need and gives the call's result), then runs it again on symbolic inputs under CUDA graph capture. Later calls whose guards hold replay that graph. A call whose guards fail is traced again at its own inputs, as another replay. Only a trace that can't read something (it *declines*) leaves the call eager, with the reason.

Five files, each written by a cell:

1. `user_code.py`: the user code, untouched
2. `cute_hook.py`: the only code that touches CuTe DSL: it keeps each compile's host function, layouts and argument spec, and intercepts and makes calls
3. `cute_ir.py`: reads a compile: the binder's checks as guards, and the launch from the host function
4. `tracing.py`: the tracer
5. `replay.py`: the runtime; `DynamicCudaGraph` puts trace and replay together

Needs a GPU, `nvidia-cutlass-dsl` (4.6.2 here), `apache-tvm-ffi` and `cuda.bindings`. Section 7 also needs our host-tracing build. The outputs are from a GB300, on our build of 2026-10-08 (internal snapshot "candidate 3").

## 1. User kernel

A CuTe DSL kernel that adds two vectors, its `@cute.jit` host function, and a Python wrapper. The wrapper compiles the host function on first use with TVM-FFI, so the compiled function takes torch tensors and launches on torch's current stream. It keeps one compile per divisibility of `n`, the way libraries keep compile caches keyed on shapes.

```python
%%writefile user_code.py
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream


@cute.kernel
def add_kernel(gx: cute.Tensor, gy: cute.Tensor, gout: cute.Tensor, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 128 + tidx
    if i < n:
        gout[i] = gx[i] + gy[i]


@cute.jit
def add_host(mx: cute.Tensor, my: cute.Tensor, mout: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
    add_kernel(mx, my, mout, n).launch(grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream)


compiled = {}  # one compile per divisibility of n, made at first use


def add(x, y):
    out = torch.empty_like(x)
    n = out.numel()
    div = 16 if n % 16 == 0 else 1
    if div not in compiled:
        t = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int32(divisibility=div),), assumed_align=16)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)  # TVM-FFI launches on torch's current stream
        compiled[div] = cute.compile(add_host, t, t, t, cutlass.Int32(0), stream, options="--enable-tvm-ffi")
    compiled[div](x, y, out, n)
    return out
```

## 2. CuTe DSL hook

Here we keep what each compile produces, and intercept each call of a compiled function.

In this example we register a trace-finalize hook with the DSL, wrap `CutlassBaseDSL.compile_and_cache` and patch the compiled classes' `__call__`. I'll work with the CuTe DSL team on the best non-intrusive API for this (section 8). Two options:

1. A launch descriptor on the compiled function: each launch's kernel, grid and parameters as expressions over the arguments.
2. The argument spec as public data on the compiled function, and structured accessors on CuTe's MLIR types.

Per compile, keyed by the function's name (the only link back from what the DSL hands us to the compiled function):

- `host_functions`: the host function as MLIR in generic form, MLIR's own lossless serialization, which `cute_ir` parses back. The hook only prints the live module: reading its values from Python runs the DSL's value casters, which add ops to it.
- `layouts`: each layout, shape, stride or int tuple in the host function, and each tensor's layout, as the DSL's own layout algebra evaluates it (`evaluate_layouts`): nested tuples of static ints and dynamic leaves with their width. It works on a copy of the module, and only takes values of those types, since taking a value runs its type's caster, which must not run on others (an MMA or TMA atom's).
- `binder_specs`: TVM-FFI's argument spec. The DSL builds it from the compile's arguments, and generates the binder from it, but doesn't keep it on the compiled function, so we run its converter in our wrapper of `compile_and_cache`.

`intercept_calls` sends each call to the tracer, and `launch` makes a call at given values. TVM-FFI reads only a tensor's metadata, so each tensor is placed in a small scratch buffer at the same alignment; the kernel never runs during capture. A function compiled before `observe_compiles()` runs is never seen, and can't be traced.

```python
%%writefile cute_hook.py
import contextlib
from dataclasses import dataclass

import torch
from cutlass import Integer
from cutlass._mlir import ir
from cutlass._mlir._mlir_libs._cutlass_ir import _cute
from cutlass._mlir.dialects import cute as cute_ops
from cutlass.cute import core
from cutlass.cute._tvm_ffi_args_spec_converter import _tvm_ffi_args_spec_converter
from cutlass.cutlass_dsl import CuTeDSL
from cutlass.cutlass_dsl.cutlass import CutlassBaseDSL
from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunction, TVMFFIJitCompiledFunctionWithKwargs
from torch.utils._python_dispatch import _disable_current_modes

# What we keep per compile, by function name: the only link back from what the DSL hands us.
host_functions = {}  # the host function, in MLIR's generic form (lossless)
binder_specs = {}  # TVM-FFI's argument spec: what its binder checks at every call
layouts = {}  # each layout-like value of the host function, as the DSL's layout algebra evaluates it

LAYOUT_TYPES = (_cute.MemRefType, _cute.LayoutType, _cute.IntTupleType, _cute.ShapeType, _cute.StrideType)


def evaluate_layouts(module, name):
    """Each top-level value of the host function whose type is a CuTe layout, int tuple, shape or
    stride, or a tensor (its layout), as the DSL's own layout algebra evaluates it: ("tuple", t) or
    ("layout", (shape, stride)), nested tuples whose leaves are ints (static) or ("dyn", bits).
    Keyed ("arg", i) for a function argument, (k, j) for result j of top-level op k.
    Evaluating adds ops, so it works on a copy of the module. Only values of these types are taken:
    taking a value runs the DSL's caster for its type, which must not run on other values (an MMA
    or TMA atom's, say)."""
    copy = ir.Module.parse(module.operation.get_asm(print_generic_op_form=True), module.context)
    fn = next(o.operation for o in copy.body.operations if o.operation.name == "func.func" and o.operation.attributes["sym_name"].value == name)
    entry = fn.regions[0].blocks[0]
    ops = list(entry.operations)
    # (key, the values, index): the types are read first, and a value is taken only for these types
    found = [(("arg", i), entry.arguments, i, t) for i, t in enumerate(entry.arguments.types)]
    found += [((k, j), op.operation.results, j, t) for k, op in enumerate(ops) for j, t in enumerate(op.operation.results.types)]

    def tree(x):
        if isinstance(x, tuple):
            return tuple(map(tree, x))
        if type(x) is int:
            return x
        if isinstance(x, Integer):
            return ("dyn", x.width)
        raise TypeError(f"a leaf {x!r}")

    def layout(v):
        return ("layout", (tree(core._unpack_x_tuple(cute_ops.get_shape(v))), tree(core._unpack_x_tuple(cute_ops.get_stride(v)))))

    out = {}
    with ir.InsertionPoint(ops[-1]), ir.Location.unknown(module.context):
        for key, values, i, t in found:
            if not any(kind.isinstance(t) for kind in LAYOUT_TYPES):
                continue
            try:
                v = values[i]
                v = v if isinstance(v, ir.Value) else v.value  # the DSL wraps some values
                if _cute.MemRefType.isinstance(t):
                    out[key] = layout(cute_ops.get_layout(v))
                elif _cute.LayoutType.isinstance(t):
                    out[key] = layout(v)
                else:
                    out[key] = ("tuple", tree(core._unpack_x_tuple(v)))
            except Exception as e:
                out[key] = ("error", f"{type(e).__name__}: {e}")
    return out


def observe(dsl, module, name):
    # Print only: reading the live module's values from Python runs the DSL's value casters, which emit IR.
    for op in module.body.operations:
        op = op.operation
        if op.name == "func.func" and op.attributes["sym_name"].value == name:
            host_functions[name] = op.get_asm(print_generic_op_form=True)
            layouts[name] = evaluate_layouts(module, name)


original_compile_and_cache = CutlassBaseDSL.compile_and_cache


# TODO(upstream API): the spec on the compiled function (today: wrapping compile_and_cache).
def compile_and_cache(self, module, module_hash, function_name, pipeline, signature, *args, full_args=None, full_kwargs=None, **kwargs):
    # The DSL builds the binder from this spec, made from the compile's arguments; a cute.SymInt used twice is one spec.Var.
    binder_specs[function_name] = _tvm_ffi_args_spec_converter(function_name, signature, list(full_args), full_kwargs or {})[0]
    return original_compile_and_cache(self, module, module_hash, function_name, pipeline, signature, *args, full_args=full_args, full_kwargs=full_kwargs, **kwargs)


def observe_compiles():
    """From now on, keep the host function, its layouts and the argument spec of every cute.compile.
    A compile made earlier is never seen."""
    CuTeDSL._get_dsl().register_trace_finalize_hook(observe)
    CutlassBaseDSL.compile_and_cache = compile_and_cache


original_calls = {cls: cls.__call__ for cls in (TVMFFIJitCompiledFunction, TVMFFIJitCompiledFunctionWithKwargs)}


# TODO(upstream API): a call hook on compiled functions (today: patching __call__).
@contextlib.contextmanager
def intercept_calls(on_call):
    """Call on_call(compiled, args) in place of every call of a TVM-FFI compiled function."""
    for cls in original_calls:
        cls.__call__ = lambda self, *args: on_call(self, args)
    try:
        yield
    finally:
        for cls, call in original_calls.items():
            cls.__call__ = call


@dataclass
class TensorArg:
    """A tensor argument as TVM-FFI reads it: its metadata and address."""

    dtype: torch.dtype
    shape: tuple
    stride: tuple
    ptr: int


def launch(compiled, args, scratch):
    """Call compiled with these argument values, under the current capture. TVM-FFI reads a tensor
    argument's DLPack metadata only, so each TensorArg is a tensor at an address in scratch, a small
    device buffer, at the address's alignment (all the compile specializes on). The capture records
    that address, and each replay writes over it."""

    def tensor(a):
        ptr = scratch.data_ptr() + a.ptr % 16
        extent = 1 + sum((n - 1) * s for n, s in zip(a.shape, a.stride)) if all(a.shape) else 0
        storage = torch._C._construct_storage_from_data_pointer(ptr, scratch.device, extent * a.dtype.itemsize)
        return torch.empty(0, dtype=a.dtype, device=scratch.device).set_(storage, 0, a.shape, a.stride)

    with _disable_current_modes():  # outside the tracer's fake and dispatch modes
        original_calls[type(compiled)](compiled, *[tensor(a) if isinstance(a, TensorArg) else a for a in args])
```

## 3. Reading a compile

`binder_checks` turns the spec into conditions over a call's arguments, one per check the binder makes:
- a static size or stride equals its value;
- a symbol's first use gets its int range and divisibility;
- each later use of the symbol equals the first (a `cute.SymInt` passed for several tensors is one symbol);
- the address is aligned.

`HostProgram` parses the host function with MLIR's parser in a context of ours where the DSL's dialects aren't registered, so no caster runs, and reads ops, operands, attributes and integer types through the bindings. It evaluates each `cuda.launch_ex`: the grid from `cuda.launch_cfg.create`, and the kernel arguments. A tensor parameter's fields come from the spec: its address, then each size and stride the spec marks dynamic, at the spec's width. A CuTe layout or int tuple is the DSL's evaluation, with each dynamic leaf filled by its operand or by the traced tensor's size or stride, so it becomes a symbolic expression like any other size. Each integer op adds a no-wraparound condition at its width. Any other op declines with its name.

```python
%%writefile cute_ir.py
import math
from dataclasses import dataclass

import sympy
from cutlass._mlir._mlir_libs._cutlass_ir._mlir import ir
from cutlass.base_dsl.tvm_ffi_builder import spec
from sympy import And, Eq, Or
from torch.utils._sympy.functions import FloorDiv

from cute_hook import binder_specs, host_functions, layouts


def binder_checks(compiled, args):
    """Each check TVM-FFI's binder makes of a call's arguments, as conditions over them. args: per
    argument, a dict of ptr, shape and stride for a tensor, or an expression for an integer."""
    seen, conditions = {}, []

    def value(v, x, unless=sympy.false):
        if isinstance(v, int):  # a static size or stride
            conditions.append(Or(unless, Eq(x, v)))
        elif id(v) in seen:  # a symbol used before: the binder checks the two are equal
            conditions.append(Or(unless, Eq(x, seen[id(v)])))
        else:
            seen[id(v)] = x
            if v.dtype.bits == 32:
                conditions.append(And(x >= -(2**31), x < 2**31))
            if (v.divisibility or 1) > 1:
                conditions.append(Eq(x % v.divisibility, 0))

    params = [p for p in binder_specs[compiled.function_name] if not isinstance(p, spec.EnvStream)]
    for p, a in zip(params, args, strict=True):
        if isinstance(p, spec.Tensor):
            for v, x in zip(p.shape, a["shape"], strict=True):
                value(v, x)
            for v, x, n in zip(p.strides, a["stride"], a["shape"], strict=True):
                value(v, x, unless=Eq(n, 1))  # the binder skips the stride of a size-1 dim
            if (p.data_alignment or 1) > 1:
                conditions.append(Eq(a["ptr"] % p.data_alignment, 0))
        elif isinstance(p, spec.Var):
            value(p, a)
        else:
            raise NotImplementedError(f"a binder parameter of kind {type(p).__name__}")
    return conditions


@dataclass
class Launch:
    kernel: str
    grid: tuple  # three expressions
    params: list  # per kernel parameter, its fields (expression, offset, bytes)


ARITH = {  # integer ops, on scalars
    "arith.addi": lambda a, b: a + b,
    "arith.subi": lambda a, b: a - b,
    "arith.muli": lambda a, b: a * b,
    "arith.floordivsi": FloorDiv,
}
TUPLE = {  # CuTe int tuple ops, on single-leaf tuples (all this example needs)
    "cute.tuple_add": lambda a, b: a + b,
    "cute.tuple_sub": lambda a, b: a - b,
    "cute.tuple_mul": lambda a, b: a * b,
    "cute.tuple_div": FloorDiv,  # C division, of nonnegative sizes here
}
LAUNCH_ATTRIBUTES = {"cuda.launch_cfg.programmatic_stream_serialization_allowed", "cuda.launch_cfg.cooperative"}  # must be 0


def leaves(t):
    return [x for e in t for x in leaves(e)] if isinstance(t, tuple) and not is_dyn(t) else [t]


def is_dyn(leaf):
    return isinstance(leaf, tuple) and leaf[:1] == ("dyn",)


def fill(t, values):
    """The tree t with each ("dyn", bits) leaf replaced by the next of values."""
    if is_dyn(t):
        return next(values)
    return tuple(fill(e, values) for e in t) if isinstance(t, tuple) else t


class HostProgram:
    """A compile's host function, parsed by MLIR's parser into a context of ours where the DSL's dialects
    aren't registered (so none of its value casters runs), and read through the bindings: ops, operands,
    attributes and builtin integer types. Layouts and int tuples come from the DSL's evaluation at
    compile time (cute_hook.evaluate_layouts); kernel parameters from the binder spec of the formal passed."""

    def __init__(self, text, binder_spec, evaluated):
        self.spec, self.evaluated = binder_spec, evaluated
        self.context = ir._Context()
        self.context.allow_unregistered_dialects = True
        with self.context:
            self.module = ir.Module.parse(text)
            body = self.module.body.operations[0].operation.regions[0].blocks[0]
            self.formal_types = [a.type for a in body.arguments]
            self.ops = [op.operation for op in body.operations]  # the top level only

    def described(self, key):
        kind, v = self.evaluated.get(key, ("error", "not a CuTe layout or int tuple"))
        if kind == "error":
            raise NotImplementedError(f"the DSL did not evaluate value {key}: {v}")
        return kind, v

    def evaluate(self, formals):
        """Each launch over the formals' values (None for the stream), and the conditions it holds under."""
        if len(formals) != len(self.spec):
            raise NotImplementedError(f"{len(formals)} formals for {len(self.spec)} spec parameters")
        cache, conditions = {}, []

        def defining(v):
            return None if isinstance(v.owner, ir.Block) else ir.OpResult(v).owner.operation

        def value(v):
            if (op := defining(v)) is None:
                i = ir.BlockArgument(v).arg_number
                return self.formal(i, formals[i])
            k = next(i for i, o in enumerate(self.ops) if o == op)
            if k not in cache:
                cache[k] = self.op(k, op, [value(u) for u in op.operands], conditions)
            return cache[k]

        launches = []
        for op in self.ops:
            if op.name != "cuda.launch_ex":
                continue
            config = defining(op.operands[0])
            if config is None or config.name != "cuda.launch_cfg.create":
                raise NotImplementedError("a launch config of another form")
            stream = config.operands[7]
            if defining(stream) is not None or not isinstance(self.spec[ir.BlockArgument(stream).arg_number], (spec.EnvStream, spec.Stream)):
                raise NotImplementedError("a launch on a stream the call does not pass")
            for o in self.ops:  # the other ops that take the config: its attributes
                if o.name != "cuda.launch_ex" and len(o.operands) and (d := defining(o.operands[0])) is not None and d == config:
                    if o.name not in LAUNCH_ATTRIBUTES or any(value(u) != 0 for u in list(o.operands)[1:]):
                        raise NotImplementedError(f"a launch attribute {o.name}")
            # the replay keeps the captured block and shared memory size: they must not depend on the arguments
            block = [sympy.sympify(value(config.operands[i])) for i in range(3)]
            if any(b.free_symbols for b in block) or defining(config.operands[3]).name != "cute.kernel_smem_size":
                raise NotImplementedError("a block or shared memory size computed from the arguments")
            params = []
            for v in list(op.operands)[1:]:
                if defining(v) is not None:
                    raise NotImplementedError("a kernel argument the host function computes (a view): this example passes only the formals")
                i = ir.BlockArgument(v).arg_number
                params.append(self.fields(self.spec[i], self.formal_types[i], formals[i]))
            kernel = ir.SymbolRefAttr(op.attributes["callee"]).value[-1]
            launches.append(Launch(kernel, tuple(sympy.sympify(value(config.operands[i])) for i in range(4, 7)), params))
        return launches, conditions

    def formal(self, i, v):
        """A formal's value in the evaluation: a tensor's is its layout, the DSL's evaluated tree with each
        dynamic leaf the traced tensor's size or stride, in order."""
        if not isinstance(self.spec[i], spec.Tensor):
            return v
        _, (shape, stride) = self.described(("arg", i))
        if len(leaves(shape)) != len(v["shape"]) or len(leaves(stride)) != len(v["stride"]):
            raise NotImplementedError(f"argument {i}'s layout {shape}:{stride} is not one leaf per dim")
        dims = iter(x for x, leaf in zip(v["shape"], leaves(shape)) if is_dyn(leaf))
        steps = iter(x for x, leaf in zip(v["stride"], leaves(stride)) if is_dyn(leaf))
        return {"ptr": v["ptr"], "shape": fill(shape, dims), "stride": fill(stride, steps)}

    @staticmethod
    def fields(p, ty, v):
        """A kernel parameter's fields, at natural alignment: a tensor's address, then each dynamic size and
        stride (the spec's Vars, at their width); an integer's value."""
        if isinstance(p, spec.Tensor):
            fields = [(v["ptr"], 8)]
            fields += [(x, s.dtype.bits // 8) for s, x in zip(p.shape, v["shape"]) if isinstance(s, spec.Var)]
            fields += [(x, s.dtype.bits // 8) for s, x in zip(p.strides, v["stride"]) if isinstance(s, spec.Var)]
        elif isinstance(p, spec.Var) and ir.IntegerType.isinstance(ty) and ir.IntegerType(ty).width == p.dtype.bits:
            fields = [(v, p.dtype.bits // 8)]
        else:
            raise NotImplementedError(f"a kernel parameter for a {type(p).__name__} of type {ty}")
        out, offset = [], 0
        for x, size in fields:
            offset += -offset % size
            out.append((x, offset, size))
            offset += size
        return out

    def op(self, k, op, a, conditions):
        n = op.name
        evaluated = self.evaluated.get((k, 0), (None, None))
        if evaluated[0] == "tuple" and not any(is_dyn(x) for x in leaves(evaluated[1])):
            return evaluated[1]  # a static int tuple: its value, as the DSL evaluated it
        if n == "arith.constant":
            return ir.IntegerAttr(op.attributes["value"]).value
        if n in ARITH:
            r = ARITH[n](*a)
            bits = ir.IntegerType(op.results[0].type).width
            conditions.append(And(r >= -(2 ** (bits - 1)), r < 2 ** (bits - 1)))  # no wraparound
            return r
        if n in ("cute.get_leaves", "cute.get_scalars"):
            flat = leaves(a[0])
            return flat[0] if len(flat) == 1 else tuple(flat)
        if n in TUPLE:
            if isinstance(a[0], tuple) or isinstance(a[1], tuple):
                raise NotImplementedError(f"{n} of tuples with more than one leaf")
            return self.no_wraparound(k, TUPLE[n](a[0], a[1]), conditions)
        if n == "cute.make_shape":  # the DSL's evaluated shape, its dynamic leaves the operands in order
            return fill(self.described((k, 0))[1], iter(x for v in a for x in leaves(v)))
        if n == "cute.make_layout" and len(a) == 1:  # a compact layout of this shape: the DSL computed its strides
            _, (_, stride) = self.described((k, 0))
            if any(is_dyn(x) for x in leaves(stride)):
                raise NotImplementedError("a layout whose strides depend on the arguments")
            return {"shape": a[0], "stride": stride}
        if n == "cute.size":
            shape = a[0]["shape"]
            for m in ir.DenseI32ArrayAttr(op.attributes["mode"]):
                shape = (shape if isinstance(shape, tuple) and not is_dyn(shape) else (shape,))[m]
            return self.no_wraparound(k, math.prod(leaves(shape)), conditions)
        raise NotImplementedError(f"unsupported host op {n}")

    def no_wraparound(self, k, r, conditions):
        # op k's result has one dynamic leaf; the DSL's evaluation gives its width
        (_, bits), = leaves(self.described((k, 0))[1])
        conditions.append(And(r >= -(2 ** (bits - 1)), r < 2 ** (bits - 1)))
        return r


def host_program(compiled):
    name = compiled.function_name
    if name not in host_functions or name not in binder_specs:
        raise NotImplementedError(f"{name} was compiled before observe_compiles: no host function or spec")
    return HostProgram(host_functions[name], binder_specs[name], layouts[name])
```

We compile the user code once, observed, and read it. The three tensors share one size symbol, `s0`, because `user_code` passed one `cute.SymInt` for all three, so the binder checks `y`'s size equals `x`'s.

```python
import importlib, sympy, torch
import user_code, cute_hook, cute_ir
from cutlass.base_dsl.tvm_ffi_builder import spec

cute_hook.observe_compiles()
torch.manual_seed(0)
x, y = torch.randn(2, 8192, device="cuda")
user_code.add(x[:4096], y[:4096])  # compiles div=16, observed
compiled = user_code.compiled[16]

names = {}  # spec.Var -> s0, s1, ...: one name per symbol


def show(v):
    if isinstance(v, int):
        return str(v)
    div = f", divisible by {v.divisibility}" if (v.divisibility or 1) > 1 else ""
    return f"{names.setdefault(id(v), f's{len(names)}')} ({v.dtype}{div})"


print("the spec:")
for i, p in enumerate(cute_hook.binder_specs[compiled.function_name]):
    if isinstance(p, spec.Tensor):
        print(f"  argument {i}: {p.dtype} tensor, shape [{', '.join(map(show, p.shape))}], strides [{', '.join(map(show, p.strides))}], address aligned to {p.data_alignment}")
    elif isinstance(p, spec.Var):
        print(f"  argument {i}: integer {show(p)}")
    else:
        print(f"  argument {i}: {type(p).__name__}")

S = lambda name: sympy.Symbol(name, integer=True, nonnegative=True)
args = [
    {"ptr": S("x_ptr"), "shape": [S("x_size0")], "stride": [1]},
    {"ptr": S("y_ptr"), "shape": [S("y_size0")], "stride": [1]},
    {"ptr": S("out_ptr"), "shape": [S("x_size0")], "stride": [1]},  # out = torch.empty_like(x)
    S("n"),
]
print("the binder's checks, over symbolic arguments:")
for cond in cute_ir.binder_checks(compiled, args):
    if cond != sympy.true:  # a stride of 1 checked against a static 1
        print(f"  {cond}")
```

Output:

```
the spec:
  argument 0: float32 tensor, shape [s0 (int32, divisible by 16)], strides [1], address aligned to 16
  argument 1: float32 tensor, shape [s0 (int32, divisible by 16)], strides [1], address aligned to 16
  argument 2: float32 tensor, shape [s0 (int32, divisible by 16)], strides [1], address aligned to 16
  argument 3: integer s1 (int32)
  argument 4: EnvStream
the binder's checks, over symbolic arguments:
  x_size0 < 2147483648
  Eq(Mod(x_size0, 16), 0)
  Eq(Mod(x_ptr, 16), 0)
  Eq(y_size0, x_size0)
  Eq(Mod(y_ptr, 16), 0)
  Eq(Mod(out_ptr, 16), 0)
  n < 2147483648
```

The launch, over the same symbolic arguments. The grid is `(n + 127)//128`, each tensor parameter is its address plus its one dynamic size, and the two integer ops in the grid may not wrap around. Only the ops the launch's operands reach are evaluated.

```python
program = cute_ir.host_program(compiled)
print("host function ops:", ", ".join(op.name for op in program.ops))
print("the arguments' layouts, as the DSL evaluated them:", [v for key, (kind, v) in cute_hook.layouts[compiled.function_name].items() if key[0] == "arg"])
(launch,), conditions = program.evaluate(args + [None])  # None: the stream TVM-FFI passes
print(f"kernel: {launch.kernel}")
print(f"grid: {launch.grid}")
for i, fields in enumerate(launch.params):
    print(f"parameter {i}: " + ", ".join(f"{e} ({size} bytes at {offset})" for e, offset, size in fields))
print("no wraparound:", conditions)
```

Output:

```
host function ops: cute.get_iter, cute.get_iter, cute.get_iter, cute.get_iter, cute.get_iter, cute.get_iter, arith.constant, arith.addi, arith.constant, arith.floordivsi, cute.get_layout, cute.get_layout, cute.get_layout, cute.kernel_smem_size, arith.constant, arith.cmpi, scf.if, arith.constant, arith.constant, cuda.launch_cfg.create, arith.constant, cuda.launch_cfg.programmatic_stream_serialization_allowed, arith.constant, cuda.launch_cfg.cooperative, cuda.launch_ex, cuda.cast, cuda.return_if_error, arith.constant, func.return
the arguments' layouts, as the DSL evaluated them: [((('dyn', 32),), (1,)), ((('dyn', 32),), (1,)), ((('dyn', 32),), (1,))]
kernel: kernel_cutlass_add_kernel_tensorptrf32gmemalign16odiv161_tensorptrf32gmemalign16odiv161_tensorptrf32gmemalign16odiv161__0
grid: (((n + 127)//128), 1, 1)
parameter 0: x_ptr (8 bytes at 0), x_size0 (4 bytes at 8)
parameter 1: y_ptr (8 bytes at 0), y_size0 (4 bytes at 8)
parameter 2: out_ptr (8 bytes at 0), x_size0 (4 bytes at 8)
parameter 3: n (4 bytes at 0)
no wraparound: [n + 127 < 2147483648, (((n + 127)//128)) < 2147483648]
```

## 4. Tracing

- Each input tensor gets mapped to a fake tensor with symbolic sizes, strides, and a symbolic `data_ptr()`. The sizes have hints; the `data_ptr()` is unbacked, so only its alignment gets guarded.
- A tensor the user code allocates (`aten.empty`, `aten.empty_like`) is caught by a `TorchDispatchMode`, which records its size expressions and gives it a fresh unbacked address symbol (`buf0_ptr`).
- At each call of a compiled function, `on_call` makes the call at the traced values, so the capture holds a kernel node to update, guards on the binder's checks, and evaluates the host function.

Guards come from:
- the user code's decisions on a symbol, such as `n % 16 == 0`, which `ShapeEnv` records with the user's line. They go first, since they decide which compile is called;
- the binder's checks, computed in section 3;
- the host function's no-wraparound conditions;
- the launch: a non-empty grid, and each 4-byte parameter field in range.

```python
%%writefile tracing.py
import os
import weakref
from dataclasses import dataclass

import sympy
import torch
from sympy import And, Eq
from torch._dynamo.source import ConstantSource
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._sympy.numbers import int_oo

import cute_hook
import cute_ir
from cutlass.base_dsl.tvm_ffi_builder import spec


@dataclass
class Allocation:
    sizes: list  # expressions
    dtype: torch.dtype
    device: torch.device
    ptr: sympy.Symbol


def symbol(name):
    return sympy.Symbol(name, integer=True, nonnegative=True)


def symbol_values(**tensors):
    """The symbols' values for real tensors."""
    values = {}
    for name, t in tensors.items():
        values.update({symbol(f"{name}_size{d}"): n for d, n in enumerate(t.shape)})
        values[symbol(f"{name}_ptr")] = t.data_ptr()
    return values


class FakeInput(FakeTensor):
    """A fake input whose data_ptr() is its address symbol (data_ptr is not an ATen op)."""

    ptr: torch.SymInt

    def data_ptr(self):
        return self.ptr


class Allocations(TorchDispatchMode):
    """Make each tensor the eager code allocates a FakeInput with a fresh address symbol.
    This example handles only allocations: any other ATen op raises."""

    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket not in (torch.ops.aten.empty, torch.ops.aten.empty_like):
            raise NotImplementedError(f"only allocations are handled, not {func}")
        out = func(*args, **(kwargs or {}))
        trace = self.trace
        name = f"buf{len(trace.allocations)}"
        with unset_fake_temporarily():
            t = FakeInput(trace.mode, torch.empty(out.shape, dtype=out.dtype, device="meta"), out.device)
        # No replay uses the hint (each allocates its own), so it only needs the allocator's alignment.
        t.ptr = trace.address(f"{name}_ptr", 0)
        sizes = [trace.expr(s) for s in out.shape]
        trace.allocations.append(Allocation(sizes, out.dtype, out.device, symbol(f"{name}_ptr")))
        # Weak, as the trace holds every tensor: a strong reference makes the factory
        # (make_variable) return a detached copy, without .ptr.
        trace.allocated.append(weakref.ref(t))
        return t


class Trace:
    def __init__(self, fn, graph, **examples):
        self.hints = {}  # symbol -> its value in this trace's launches: the eager run's, but for allocations
        self.rename = {}  # ShapeEnv symbol -> {name}_size{d} or {name}_ptr, so guards read by name
        self.guards = []  # (condition, its value at the hints, where it comes from)
        self.launches = []
        self.allocations = []
        self.allocated = []  # weak references to the FakeInputs the allocations return
        self.addresses = set()  # the address symbols

        self.shape_env = ShapeEnv()
        self.mode = FakeTensorMode(shape_env=self.shape_env)
        inputs = {}
        for name, t in examples.items():
            sizes = [self.symint(f"{name}_size{d}", n) for d, n in enumerate(t.shape)]
            inputs[name] = FakeInput(self.mode, torch.empty(sizes, dtype=t.dtype, device="meta"), t.device)
            inputs[name].ptr = self.address(f"{name}_ptr", t.data_ptr())
        # For cute_hook.launch. Made before capture, so it is not graph memory.
        self.scratch = torch.empty(16, dtype=torch.uint8, device="cuda")

        with torch.cuda.graph(graph), self.mode, Allocations(self), cute_hook.intercept_calls(self.on_call):
            result = fn(**inputs)
        # Which allocation fn returns.
        self.output = next(k for k, t in enumerate(self.allocated) if t() is result)
        # The host code's own branches on symbols, which ShapeEnv recorded with the user's frame. They
        # go first: they decide which compile the user code calls.
        branches = []
        for g in self.shape_env.guards:
            frame = g.sloc.framework_loc
            branches.append((g.expr.xreplace(self.rename), True, f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}"))
        self.guards = branches + self.guards

    def symint(self, name, hint):
        # >= 0, but not sympy-positive like from_tensor's sizes, so a branch on n == 0 is guarded
        s = self.shape_env.create_unspecified_symint_and_symbol(hint, ConstantSource(name), DimDynamic.DYNAMIC)
        self.shape_env.constrain_symbol_range(s.node.expr, 0, int_oo)
        self.rename[s.node.expr] = symbol(name)
        self.hints[symbol(name)] = hint
        return s

    def address(self, name, value):
        # Unbacked, so host code cannot branch on it. Its only guards are the binder's
        # alignment checks, which on_call adds.
        s = self.shape_env.create_unbacked_symint()
        self.shape_env.constrain_symbol_range(s.node.expr, 0, int_oo)
        self.rename[s.node.expr] = symbol(name)
        self.hints[symbol(name)] = value
        self.addresses.add(symbol(name))
        return s

    def expr(self, v):
        if isinstance(v, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return v.node.expr.xreplace(self.rename)
        return sympy.sympify(v)

    def at_hint(self, v):
        if isinstance(v, torch.Tensor):
            shape, stride = (tuple(map(self.at_hint, s)) for s in (v.shape, v.stride()))
            return cute_hook.TensorArg(v.dtype, shape, stride, self.at_hint(v.data_ptr()))
        return int(self.expr(v).subs(self.hints))

    def view(self, t):
        return {"ptr": self.expr(t.data_ptr()), "shape": [self.expr(s) for s in t.shape], "stride": [self.expr(s) for s in t.stride()]}

    def launched(self):
        """The symbols' values in the trace's launches: an address is scratch's, at its alignment."""
        return {s: self.scratch.data_ptr() + v % 16 if s in self.addresses else v for s, v in self.hints.items()}

    def on_call(self, compiled, args):
        def guard(cond, source):
            value = bool(cond.subs(self.hints))
            if cond not in (sympy.true, sympy.false) and (cond, value, source) not in self.guards:
                self.guards.append((cond, value, source))

        # Launch at the hints, as eager did, so the capture has a real launch for replay to patch.
        cute_hook.launch(compiled, [self.at_hint(a) for a in args], self.scratch)

        # Each argument's value: a tensor is a dict of expressions.
        values = [self.view(a) if isinstance(a, torch.Tensor) else self.expr(a) for a in args]
        # TVM-FFI's binder checks, guarded at their outcome. The binder raises on a call that breaks
        # one, so a guard that fails sends the call to eager, which raises.
        for cond in cute_ir.binder_checks(compiled, values):
            guard(cond, "binder")

        # The host function's formals: the arguments, and None for the stream TVM-FFI passes (torch's current one).
        program = cute_ir.host_program(compiled)
        given = iter(values)
        formals = [None if isinstance(p, spec.EnvStream) else next(given) for p in program.spec]
        launches, conditions = program.evaluate(formals)
        for cond in conditions:  # no integer wraparound in the host function
            guard(cond, "host function")
        for launch in launches:
            guard(launch.grid[0] * launch.grid[1] * launch.grid[2] > 0, "launch")  # CUDA rejects an empty grid
            for e, _, size in (f for fields in launch.params for f in fields):
                if size < 8:  # a 32-bit field holds the value only in i32 range
                    guard(And(e >= -(2 ** (8 * size - 1)), e < 2 ** (8 * size - 1)), "launch")
            self.launches.append(launch)
```

## 5. Runtime

The trace captures the user code into `Replay`'s graph, so the graph holds the launch the trace made. From the spec and the host function, we know which bytes of each kernel parameter are an address, a size or a stride, and their expressions. A call allocates what the user code allocates, at the sizes evaluated on the new inputs, checks the guards, writes the new values into those bytes and the grid into the kernel node, and replays. (Once per trace, a check compares our evaluation at the traced values with the captured bytes; it is a test, not needed to replay.)

`DynamicCudaGraph` tries its replays in order and returns the first hit's output. If none hits, it traces again at the call's inputs, and records why (`retrace_causes`): the guard that failed, and where it came from.

```python
%%writefile replay.py
import ctypes

import torch
from cuda.bindings import driver
from torch.cuda._utils import _check_cuda_bindings

from tracing import Trace, symbol_values


def pack(params, values, images):
    """The parameters' bytes: each field's value at the symbols' values, over images (whatever is in
    the padding between fields)."""
    out = [bytearray(image) for image in images]
    for buf, fields in zip(out, params):
        for e, offset, size in fields:
            buf[offset : offset + size] = int(e.subs(values) if hasattr(e, "subs") else e).to_bytes(size, "little", signed=size < 8)
    return out


class Replay:
    def __init__(self, fn, **tensors):
        self.graph = torch.cuda.CUDAGraph(keep_graph=True)  # keep the graph to read its node
        trace = Trace(fn, self.graph, **tensors)
        self.graph.instantiate()
        self.guards = trace.guards
        self.allocations, self.output = trace.allocations, trace.output
        (self.launch,) = trace.launches  # this example records one launch

        # The graph's one node is the launch.
        (self.node,), _ = _check_cuda_bindings(driver.cuGraphGetNodes(self.graph.raw_cuda_graph(), 1))
        p = _check_cuda_bindings(driver.cuGraphKernelNodeGetParams(self.node))
        name = _check_cuda_bindings(driver.cuFuncGetName(p.func)).decode()
        if name != self.launch.kernel:
            raise AssertionError(f"the call launched {name}; its host function {self.launch.kernel}")

        # Each parameter's size. cuFuncGetParamInfo returns (error, offset, size)
        # and errors past the last parameter.
        sizes = []
        while (info := driver.cuFuncGetParamInfo(p.func, len(sizes)))[0] == driver.CUresult.CUDA_SUCCESS:
            sizes.append(info[2])
        captured = [ctypes.string_at(ptr, size) for ptr, size in zip((ctypes.c_void_p * len(sizes)).from_address(int(p.kernelParams)), sizes)]

        # A check, not needed to replay: at the values the capture launched with, the host function's
        # evaluation gives the captured parameter bytes (padding aside) and grid.
        if len(self.launch.params) != len(sizes) or pack(self.launch.params, trace.launched(), captured) != captured:
            raise AssertionError("the host function's parameters are not the captured ones")
        if tuple(int(g.subs(trace.launched())) for g in self.launch.grid) != (p.gridDimX, p.gridDimY, p.gridDimZ):
            raise AssertionError("the host function's grid is not the captured one")

        # Writable copies of the captured parameter values, which each call
        # overwrites, passed as kernelParams: an array of one pointer per parameter.
        self.buffers = [ctypes.create_string_buffer(image, len(image)) for image in captured]
        self.pointers = (ctypes.c_void_p * len(sizes))(*map(ctypes.addressof, self.buffers))
        p.kernelParams = ctypes.addressof(self.pointers)
        self.node_params = p

    def __call__(self, **tensors):
        values = symbol_values(**tensors)
        # Allocate as eager would, before the guards, which can involve the addresses.
        allocated = []
        for a in self.allocations:
            allocated.append(torch.empty([int(s.subs(values)) for s in a.sizes], dtype=a.dtype, device=a.device))
            values[a.ptr] = allocated[-1].data_ptr()
        for cond, expected, where in self.guards:
            if bool(cond.subs(values)) != expected:
                return f"miss: ({cond}) is {not expected}, from {where}", None
        for buf, data in zip(self.buffers, pack(self.launch.params, values, [b.raw for b in self.buffers])):
            buf.raw = bytes(data)
        p = self.node_params
        p.gridDimX, p.gridDimY, p.gridDimZ = (int(g.subs(values)) for g in self.launch.grid)
        _check_cuda_bindings(driver.cuGraphExecKernelNodeSetParams(self.graph.raw_cuda_graph_exec(), self.node, p))
        self.graph.replay()
        return "hit", allocated[self.output]


class DynamicCudaGraph:
    """Call fn through CUDA graphs. A call replays the first graph whose guards hold. Otherwise it
    traces fn at its inputs, as another replay: the trace first runs fn once as a warm-up, which
    compiles what these inputs need and gives the call's result, then runs fn again on symbolic
    inputs under capture. A trace that declines (NotImplementedError) leaves fn eager from then on,
    with the reason."""

    def __init__(self, fn):
        self.fn = fn
        self.replays = []
        self.declined = None
        self.retrace_causes = []  # per trace after the first: the guard that failed, and where it came from

    def __call__(self, **tensors):
        """Returns fn's result, and how the call ran."""
        if self.declined is not None:
            return self.fn(**tensors), f"eager (declined: {self.declined})"
        misses = []
        for i, replay in enumerate(self.replays):
            how, result = replay(**tensors)
            if how == "hit":
                return result, f"replay {i} hit"
            misses.append(f"replay {i} {how}")
        result = self.fn(**tensors)  # the trace's warm-up
        try:
            replay = Replay(self.fn, **tensors)
        except NotImplementedError as e:
            self.declined = str(e)
            return result, "; ".join(misses + [f"the trace declined: {e}; eager"])
        if self.replays:
            self.retrace_causes.append(misses[0].split(": ", 1)[1])
        self.replays.append(replay)
        return result, "; ".join(misses + [f"traced as replay {len(self.replays) - 1}"])
```

## 6. Calls

The first call traces as replay 0. Each guard prints with where it comes from. No guard pins the size.

```python
import tracing, replay

f = replay.DynamicCudaGraph(user_code.add)


def call(name, xi, yi):
    out, how = f(x=xi, y=yi)
    torch.cuda.synchronize()
    print(f"{name}: {how}; out == x + y: {torch.equal(out, xi + yi)}")


call("n=4096", x[:4096], y[:4096])
r0 = f.replays[0]
print("replay 0's guards (condition, its value, where it comes from):")
for cond, value, source in r0.guards:
    print(f"  {source}: ({cond}) is {value}")
print(f"grid: {r0.launch.grid}")
```

Output:

```
n=4096: traced as replay 0; out == x + y: True
replay 0's guards (condition, its value, where it comes from):
  user_code.py:28 in add: (Eq(Mod(x_size0, 16), 0)) is True
  binder: (x_size0 < 2147483648) is True
  binder: (Eq(Mod(x_size0, 16), 0)) is True
  binder: (Eq(Mod(x_ptr, 16), 0)) is True
  binder: (Eq(y_size0, x_size0)) is True
  binder: (Eq(Mod(y_ptr, 16), 0)) is True
  binder: (Eq(Mod(buf0_ptr, 16), 0)) is True
  host function: (x_size0 + 127 < 2147483648) is True
  host function: ((((x_size0 + 127)//128)) < 2147483648) is True
  launch: ((((x_size0 + 127)//128)) > 0) is True
  launch: (x_size0 < 2147483648) is True
  launch: (y_size0 < 2147483648) is True
grid: (((x_size0 + 127)//128), 1, 1)
```

### Other sizes

n=8192 and n=4112 hit replay 0, with no new trace or compile. n=4097 is not divisible by 16, so it fails the user code's branch: it is traced again as replay 1 (its warm-up compiles a `div=1` version), and the cause names the guard and the line. n=4100 then hits replay 1, and n=4096 still hits replay 0.

```python
call("n=8192", x[:8192], y[:8192])
call("n=4112", x[:4112], y[:4112])
call("n=4097", x[:4097], y[:4097])
call("n=4100", x[:4100], y[:4100])
call("n=4096", x[:4096], y[:4096])
print(f"compiles: div={sorted(user_code.compiled)}")
print(f"retrace causes: {f.retrace_causes}")
```

Output:

```
n=8192: replay 0 hit; out == x + y: True
n=4112: replay 0 hit; out == x + y: True
n=4097: replay 0 miss: (Eq(Mod(x_size0, 16), 0)) is False, from user_code.py:28 in add; traced as replay 1; out == x + y: True
n=4100: replay 1 hit; out == x + y: True
n=4096: replay 0 hit; out == x + y: True
compiles: div=[1, 16]
retrace causes: ['(Eq(Mod(x_size0, 16), 0)) is False, from user_code.py:28 in add']
```

### A call the binder rejects

With `y` shorter than `x`, replay 0 fails the binder's `y_size0 == x_size0`. The trace's warm-up then raises the binder's own error, as eager does. Without that guard, the replay would run the kernel over `x`'s size, reading past the end of `y`. An earlier version of our implementation did exactly that: it took guards from each tensor's printed type alone, which doesn't say that sizes share a symbol.

```python
print(f"replay 0, y shorter than x: {f.replays[0](x=x[:4096], y=y[:2048])[0]}")
try:
    call("y shorter than x", x[:4096], y[:2048])
except ValueError as e:
    print(f"the trace's warm-up: ValueError: {str(e).splitlines()[0]}")
```

Output:

```
replay 0, y shorter than x: miss: (Eq(y_size0, x_size0)) is False, from binder
the trace's warm-up: ValueError: Mismatched my.shape[0] on argument #1 when calling: `add_host(mx: Tensor([n0], float32), my: Tensor([n0], float32), mout: Tensor([n0], float32), n: int32)`, expected to match mx.shape[0]
```

### A grid computed from layouts

The same kernel, with the grid computed from layouts instead of from `n`:
- `cute.size(mx, mode=[0])`, the tensor's size;
- a nested layout `(4,(8,blocks))` of the blocks;
- the grid as its size divided by 32.

The next cell prints what the DSL's layout algebra gave at the compile: the tensors' `(?):(1)`, the static int tuples, and the nested layout with the strides the DSL computed.

```python
%%writefile layout_code.py
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from user_code import add_kernel


@cute.jit
def add_host_from_layout(mx: cute.Tensor, my: cute.Tensor, mout: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
    # add_host's launch, with the grid computed from layouts: the tensor's size, and a nested layout of the blocks
    m = cute.size(mx, mode=[0])
    blocks = cute.make_layout((4, (8, (m + 127) // 128)))  # (4,(8,?)), 32 per block
    add_kernel(mx, my, mout, n).launch(grid=[cute.size(blocks) // 32, 1, 1], block=[128, 1, 1], stream=stream)


compiled = {}


def add(x, y):
    out = torch.empty_like(x)
    if "add" not in compiled:
        t = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int32(),), assumed_align=16)
        compiled["add"] = cute.compile(add_host_from_layout, t, t, t, cutlass.Int32(0), make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    compiled["add"](x, y, out, out.numel())
    return out
```

```python
import layout_code
layout_code.add(x[:4096], y[:4096])  # compiles, observed
from_layout = layout_code.compiled["add"]
program = cute_ir.host_program(from_layout)
for key, (kind, v) in cute_hook.layouts[from_layout.function_name].items():
    where = f"argument {key[1]}" if key[0] == "arg" else program.ops[key[0]].name
    if key[0] == "arg" or where in ("cute.make_int_tuple", "cute.make_shape", "cute.make_layout"):
        print(f"{where}: {kind} {v}")
```

Output:

```
argument 0: layout ((('dyn', 32),), (1,))
argument 1: layout ((('dyn', 32),), (1,))
argument 2: layout ((('dyn', 32),), (1,))
cute.make_int_tuple: tuple 127
cute.make_int_tuple: tuple 128
cute.make_shape: tuple (4, (8, ('dyn', 32)))
cute.make_layout: layout ((4, (8, ('dyn', 32))), (1, (4, 32)))
cute.make_int_tuple: tuple 32
```

Tracing fills in the dynamic leaves: `blocks` is `(x_size0 + 127)//128` and the nested layout's size is `32*blocks`, so the grid simplifies back to `(x_size0 + 127)//128`. One replay serves every size, and each size and product gets its no-wraparound guard.

```python
g = replay.DynamicCudaGraph(layout_code.add)
big_x, big_y = torch.randn(2, 100000, device="cuda")
for n in (4096, 8192, 777, 100000):
    out, how = g(x=big_x[:n], y=big_y[:n])
    torch.cuda.synchronize()
    print(f"n={n}: {how}; out == x + y: {torch.equal(out, big_x[:n] + big_y[:n])}")
print(f"grid: {g.replays[0].launch.grid}")
```

Output:

```
n=4096: traced as replay 0; out == x + y: True
n=8192: replay 0 hit; out == x + y: True
n=777: replay 0 hit; out == x + y: True
n=100000: replay 0 hit; out == x + y: True
grid: (((x_size0 + 127)//128), 1, 1)
```

## 7. Our implementation on the same calls

The same user code through our implementation's entry point, `HostTraceReplay` (a private API in our build). Its first call runs eagerly and its second traces; `redispatches` counts ops dispatched again without a trace; `retrace_causes` maps each reason for a trace after the first, as (class, op, guard, the user's line, a redispatch's refusal), to its count.

- **Plain `user_code.add`.** Which compile runs is decided by the wrapper's Python (`n % 16 == 0`), outside any op, so that guard belongs to the graph: n=4097 traces again, and the cause names the guard and the line. Its class is `meta`: a guard of the graph's that an op recorded too (the CuTe call's binder checks the same divisibility).
- **The same wrapper inside `torch.cuda._host_trace.dispatch_unit`.** A dispatch unit declares that its body is one op whose effects are its launches, allocations and outputs. Its guards are the op's own, so a flip dispatches the unit again, without a trace.

Each output is compared with eager's (`x + y`) with `torch.equal`, a bitwise comparison here. In our current snapshot, the layouts of section 6 are still read from the printed types; the evaluation shown above, by the DSL's layout algebra on layout-typed values only, is queued for the next snapshot.

```python
from torch.cuda._host_trace_replay import HostTraceReplay  # arms its compile hooks: cutlass is loaded


def run(fn, label):
    importlib.reload(user_code)  # compile again, observed by the hooks
    h = HostTraceReplay(fn)
    bitwise = True
    for n in (4096, 4096, 8192, 4112, 4097, 4100, 4096):
        out = h(x[:n], y[:n])
        torch.cuda.synchronize()
        bitwise &= torch.equal(out, x[:n] + y[:n])
    print(f"{label}: traces {h.traces}, replays {h.replays}, redispatches {h.redispatches}, eager {h.eager}; every output equal to eager's: {bitwise}")
    for (cls, op, guard, line, refusal), count in h.retrace_causes.items():
        print(f"  {count} retrace: class {cls}, guard {guard}, at {line}, refusal {refusal}")


run(lambda x, y: user_code.add(x, y), "user_code.add")
run(torch.cuda._host_trace.dispatch_unit(lambda x, y: user_code.add(x, y)), "user_code.add in a dispatch_unit")
```

Output:

```
user_code.add: traces 2, replays 4, redispatches 0, eager 1; every output equal to eager's: True
  1 retrace: class meta, guard ((s61 % 16) == 0), at run_v2/user_code.py:28, refusal None
user_code.add in a dispatch_unit: traces 1, replays 5, redispatches 1, eager 1; every output equal to eager's: True
```

## 8. What remains hard, and what we'd ask of the CuTe DSL team

What makes CuTe DSL kernels harder to trace than Triton's (the companion document, `cute_dsl_difficulties_20261008.md`, has each with our workaround and its status):

- **The host function's ops.** Every library brings new host ops, and the evaluator has to learn each one, or decline. FlashInfer's mxfp8 GEMM needs about 9 more; our implementation interprets about 45.
- **Types without structured accessors.** MLIR's Python bindings for CuTe's types have no per-leaf accessors, and `LayoutType.stride` on a nested layout `(4,(8,?))` raises `std::get: wrong index for variant`. So layouts go through the DSL's layout algebra at compile time, on a copy of the module, with positional keys. Two facts have no structured source yet, so a host function that needs them declines: `cute.assume`'s divisibility (`ConstrainedIntType` has no accessors) and a TMA atom's fields.
- **The DSL's value casters.** Taking a value from Python runs its type's caster, which adds ops, and on some values (MMA and TMA atoms) has corrupted the heap. We take values only of the layout types, on a copy.
- **DSL internals.** The hook, the private `CuTeDSL._get_dsl()`, the signature of `compile_and_cache`, `_tvm_ffi_args_spec_converter`, and the compiled classes' `__call__`. A change there makes calls decline, not run wrongly.
- **Re-implemented binder checks.** The binder has no check-only entry point: its checks are inlined in the generated wrapper. If the DSL adds a check, the guards miss it until updated.
- **Objects loaded from disk carry nothing**: no host function, no spec. We write our own files beside the objects, versioned and tied to the object's digest.

Requests, ordered by how much each removes:

1. **A launch descriptor on the compiled function**: for each launch, the kernel, the grid, block, shared memory and cluster as expressions over the arguments, the launch attributes, the stream argument, and each kernel parameter's fields with their source. The evaluator and its op list disappear. A smaller fallback: `compiled.prepare(*args)`, returning the launches for concrete arguments without launching.
2. **The argument spec as public data on the compiled function**, e.g. `compiled.args_spec`, and a check-only binder entry, `compiled.check_args(*args)`, so the guards can be tested against the binder.
3. **Both carried through `export_to_c` and loading**, so a function loaded from a compile cache can be traced.
4. **Structured accessors on CuTe's MLIR types**: per-leaf static values or symbols, divisibility and widths on layouts and tuples; `ConstrainedIntType`'s divisibility; a TMA atom's fields; and a fix for `LayoutType.stride` on nested layouts. Small, and the repro needs no GPU.
5. **CuTe tensors from metadata with a deferred address**, for `from_dlpack`, fake tensors and `make_ptr`.
6. **A supported call hook on compiled functions**, instead of patching `__call__`.
7. **Nice to have:** a compile callback given the compiled object; stream handles kept out of the mangled function name (so a jit call doesn't compile once per stream); a documented TMA descriptor layout or re-encode.
