# Dynamic-shape CUDA graphs for CuTe DSL kernels

> **AI-generated draft** (Claude Code, for Elias to review and edit). Not reviewed yet; nothing here has been sent to the CuTe DSL team. Refreshed 2026-10-08 for our internal snapshot "candidate 3".

This notebook shows how a CUDA graph can be replayed at sizes and memory addresses other than the ones it was captured at, for a CuTe DSL kernel. A short version comes first. The long version then builds the mechanism step by step, on one small kernel, and prints what each step produces. It ends with the difficulties we hit and what we would ask of the CuTe DSL team.

You should know PyTorch and the basics of CuTe DSL (`@cute.kernel`, `@cute.jit`, `cute.compile`). You don't need to know host tracing; it is defined below. A companion Triton notebook (dynamic_shape_cudagraphs_triton_20261008.ipynb) does the same for Triton kernels.

Needs a GPU, `nvidia-cutlass-dsl` (4.6.2 here), `apache-tvm-ffi` and `cuda.bindings`. Section 8 also needs our host-tracing build. The outputs are from a GB300 (sm_100), on our build of 2026-10-08.

## Short version

- **The problem.** A CUDA graph replays the exact launches it captured: the same grid, and the same kernel parameters, which for a tensor are its address and sizes. A call at another size or with other tensors can't use it.
- **Host tracing.** We run the code that launches kernels with symbolic sizes and addresses. For each launch we record the grid and each parameter as an expression, and a **guard** for every assumption made on the way. A later call whose guards hold evaluates the expressions, writes them into the graph's kernel node, and replays. A call whose guards fail runs eagerly.
- **For a CuTe DSL kernel, the facts come from the compile, as structured data:**
  - the guards from **TVM-FFI's argument spec**, from which the DSL generates its argument checker (the *binder*): static sizes and strides, shared size symbols, divisibility, alignment, integer ranges;
  - the launch (kernel, grid, parameters) from **the host function's MLIR**, read through MLIR's Python bindings (ops, operands, attributes, builtin types), never by matching printed text;
  - a capture of a real launch, to check the evaluation byte for byte.
- **Anything without a structured source is declined:** the call runs eagerly, with the reason. Here, that includes CuTe layouts and int tuples, whose structure is only in their printed MLIR types. Declining keeps the replay sound: it never runs a launch that eager wouldn't.
- **On the example kernel:** one trace serves every size whose compile is the same, a call with mismatched sizes goes to eager and gets the binder's own error, and a host function that computes the grid from layouts declines.
- **What would make this simpler** (section 10): a launch descriptor on the compiled function, the argument spec as public data, both carried through export and load, and structured accessors on CuTe's MLIR types.

The same rule holds in our full implementation: Triton, CuTe DSL and ATen launches are traced symbolically, guarded, or declined. Only closed-source kernels whose host code we can't trace (cuBLAS/cuBLASLt, cuDNN, closed attention kernels) are learned from captures and checked empirically.

# Long version

## 1. What a CUDA graph freezes

A CUDA graph records the kernels a piece of code launches. Replaying the graph runs those kernels again, with less CPU work than the original launches. Each recorded launch keeps the exact values it was launched with: its grid, and every kernel parameter. For a tensor, the parameters are its address and its sizes.

Below is the user code: a CuTe DSL kernel that adds two vectors, its `@cute.jit` host function, and a Python wrapper. The wrapper compiles the host function on first use with TVM-FFI, so the compiled function takes torch tensors and launches on torch's current stream. It keeps one compile per divisibility of `n`, the way libraries keep compile caches keyed on shapes.

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

We capture one call at n=4096 into a CUDA graph and read back the kernel launch it recorded.

```python
import ctypes, struct, torch
from cuda.bindings import driver
from torch.cuda._utils import _check_cuda_bindings
import user_code


def kernel_node(graph):
    """The graph's one kernel node: its grid, and each kernel parameter's bytes."""
    (node,), _ = _check_cuda_bindings(driver.cuGraphGetNodes(graph.raw_cuda_graph(), 1))
    p = _check_cuda_bindings(driver.cuGraphKernelNodeGetParams(node))
    sizes = []  # cuFuncGetParamInfo returns (error, offset, size) and errors past the last parameter
    while (info := driver.cuFuncGetParamInfo(p.func, len(sizes)))[0] == driver.CUresult.CUDA_SUCCESS:
        sizes.append(info[2])
    pointers = (ctypes.c_void_p * len(sizes)).from_address(int(p.kernelParams))
    return (p.gridDimX, p.gridDimY, p.gridDimZ), [ctypes.string_at(ptr, n) for ptr, n in zip(pointers, sizes)]


torch.manual_seed(0)
x, y = torch.randn(2, 8192, device="cuda")
user_code.add(x[:4096], y[:4096])  # compile and warm up, outside the capture
graph = torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(graph):
    out = user_code.add(x[:4096], y[:4096])
graph.instantiate()

grid, params = kernel_node(graph)
print(f"grid: {grid}")
for name, t, b in zip(("x", "y", "out"), (x, y, out), params):
    address, size = struct.unpack("<Qi", b[:12])  # a tensor parameter: its address, then its size
    print(f"{name}: address {address:#x} (its data_ptr() is {t.data_ptr():#x}), size {size}")
print(f"n: {struct.unpack('<i', params[3])[0]}")
```

Output:

```
grid: (32, 1, 1)
x: address 0xfff99dc00000 (its data_ptr() is 0xfff99dc00000), size 4096
y: address 0xfff99dc08000 (its data_ptr() is 0xfff99dc08000), size 4096
out: address 0x333000000 (its data_ptr() is 0x333000000), size 4096
n: 4096
```

The graph holds the grid `(32, 1, 1)`, the three tensors' addresses, and the size 4096. Replaying it always runs exactly that launch: same sizes, same memory. A call at n=8192 needs grid 64, and a call with other tensors needs other addresses, so the graph can't serve them. The usual workarounds are to capture one graph per size, or to copy inputs into fixed buffers.

## 2. What host tracing records instead

**Host tracing** runs the code that launches kernels (the "host" code) with symbolic sizes and addresses instead of concrete ones. For each launch, it records:

- the grid and each kernel parameter as **expressions** over the call's sizes and addresses, e.g. `grid = (x_size0 + 127)//128`;
- a **guard** for every assumption made along the way: a condition on the sizes and addresses that must hold for the launch to be this one. Examples are a branch the host code took (`n % 16 == 0` picked the `div=16` compile), or a fact the compiled kernel relies on (the address is a multiple of 16).

It also captures the call into a CUDA graph once. At a later call, a **replay** checks the guards against the new sizes and addresses. If they all hold, it evaluates the expressions, writes the results into the graph's kernel node, and replays. If a guard fails (a **miss**), the call runs eagerly and is traced again at its own sizes, as another replay.

## 3. Where each fact comes from, for a CuTe DSL kernel

With Triton, the launch is computed in Python, where a tracer can watch it. A CuTe DSL host function is compiled together with its kernel, and calling the compiled function runs compiled code. So we take each fact from what the compile produces:

| fact | source |
|---|---|
| guards on the arguments | **TVM-FFI's argument spec**. The compiled function's argument checker (its *binder*) is generated from this spec. It lists each argument's static sizes and strides, its size *symbols* (a `cute.SymInt` passed for several tensors is one symbol, and the binder checks those sizes are equal), divisibility, alignment, int ranges and dtype. |
| the launch: kernel, grid, parameters | **the host function's IR**. The trace-finalize hook gets the `@cute.jit` function's MLIR after tracing. We read its ops, operands, attributes and builtin integer types through MLIR's Python bindings. |
| a tensor parameter's fields | **the spec again**: a tensor is passed as its address, then each size and stride the spec marks dynamic, at the spec's width. |
| layouts and int tuples computed in the host function | **none**. Their structure, e.g. `(4,(8,?)):(1,(4,32))`, is part of their MLIR type, and MLIR's bindings don't take it apart. A host function that computes the launch from them is declined. |
| the kernel node to update, and a check | **the capture**. We record a real launch at the traced call's values, then check that our evaluation of the host function gives exactly the captured grid and parameter bytes. |

**The rule this notebook follows:** take every fact from structured data captured at compile time (the spec and the IR), and decline whatever isn't provided that way. Declining means the call runs eagerly, with the reason stated. Nothing is guessed from printed text or from bytes. Decline, rather than guess, keeps the replay sound: it never runs a launch that eager wouldn't.

## 4. Seeing what the compile gives us

Four small files are the only code that touches CuTe DSL. The first, `cute_compile.py`, keeps two things per compile, keyed by the function's name, because that name is the only link back from what the DSL hands us to the compiled function:

- **The host function, as MLIR text in generic form.** This is MLIR's own lossless serialization, and we parse it back with MLIR's parser, so it is read as IR, not as text. The hook only prints the live module, because reading its values from Python runs the DSL's value casters, which add ops to the module.
- **TVM-FFI's argument spec.** The DSL builds it from the compile's arguments in `_tvm_ffi_args_spec_converter`, but doesn't keep it on the compiled function. So we wrap `CutlassBaseDSL.compile_and_cache`, which receives those arguments, and run the converter there.

A function compiled before `observe_compiles()` runs is never seen, and can't be traced.

```python
%%writefile cute_compile.py
from cutlass.cute._tvm_ffi_args_spec_converter import _tvm_ffi_args_spec_converter
from cutlass.cutlass_dsl import CuTeDSL
from cutlass.cutlass_dsl.cutlass import CutlassBaseDSL

host_functions = {}  # function name -> its host function, in MLIR's generic form (lossless)
binder_specs = {}  # function name -> TVM-FFI's argument spec: what its binder checks at every call


def observe(dsl, module, name):
    # Print only: reading the live module's values from Python runs the DSL's value casters, which emit IR.
    for op in module.body.operations:
        op = op.operation
        if op.name == "func.func" and op.attributes["sym_name"].value == name:
            host_functions[name] = op.get_asm(print_generic_op_form=True)


original_compile_and_cache = CutlassBaseDSL.compile_and_cache


# TODO(upstream API): the spec on the compiled function (today: wrapping compile_and_cache).
def compile_and_cache(self, module, module_hash, function_name, pipeline, signature, *args, full_args=None, full_kwargs=None, **kwargs):
    # The DSL builds the binder from this spec, made from the compile's arguments; a cute.SymInt used twice is one spec.Var.
    binder_specs[function_name] = _tvm_ffi_args_spec_converter(function_name, signature, list(full_args), full_kwargs or {})[0]
    return original_compile_and_cache(self, module, module_hash, function_name, pipeline, signature, *args, full_args=full_args, full_kwargs=full_kwargs, **kwargs)


def observe_compiles():
    """From now on, keep the host function and the argument spec of every cute.compile.
    A compile made earlier is never seen."""
    CuTeDSL._get_dsl().register_trace_finalize_hook(observe)
    CutlassBaseDSL.compile_and_cache = compile_and_cache
```

We compile again, now observed, and print the spec. The three tensors share one size symbol, `s0`, because `user_code` passed one `cute.SymInt` for all three. The integer `n` has its own symbol, `s1`.

```python
import importlib, cute_compile
from cutlass.base_dsl.tvm_ffi_builder import spec
importlib.reload(user_code)  # forget the compile above, which no hook saw
cute_compile.observe_compiles()
user_code.add(x[:4096], y[:4096])  # compiles div=16 again, now observed
compiled = user_code.compiled[16]

names = {}  # spec.Var -> s0, s1, ...: one name per symbol


def show(v):
    if isinstance(v, int):
        return str(v)
    div = f", divisible by {v.divisibility}" if (v.divisibility or 1) > 1 else ""
    return f"{names.setdefault(id(v), f's{len(names)}')} ({v.dtype}{div})"


for i, p in enumerate(cute_compile.binder_specs[compiled.function_name]):
    if isinstance(p, spec.Tensor):
        print(f"argument {i}: {p.dtype} tensor, shape [{', '.join(map(show, p.shape))}], strides [{', '.join(map(show, p.strides))}], address aligned to {p.data_alignment}")
    elif isinstance(p, spec.Var):
        print(f"argument {i}: integer {show(p)}")
    else:
        print(f"argument {i}: {type(p).__name__}")
```

Output:

```
argument 0: float32 tensor, shape [s0 (int32, divisible by 16)], strides [1], address aligned to 16
argument 1: float32 tensor, shape [s0 (int32, divisible by 16)], strides [1], address aligned to 16
argument 2: float32 tensor, shape [s0 (int32, divisible by 16)], strides [1], address aligned to 16
argument 3: integer s1 (int32)
argument 4: EnvStream
```

## 5. Guards from the spec

`binder_checks` (in `cute_guards.py`) turns the spec into conditions over a call's arguments. Each condition is one check the binder makes:
- a static size or stride must equal its value;
- a symbol's first use gets its int range and divisibility;
- each later use of the symbol must equal the first;
- the address must be aligned.

A call that breaks any of these is rejected by the binder in eager. So a replay guards on all of them: when one fails, the call goes to eager, which raises the binder's own error.

```python
%%writefile cute_guards.py
import sympy
from cutlass.base_dsl.tvm_ffi_builder import spec
from sympy import And, Eq, Or

from cute_compile import binder_specs


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
```

The checks over symbolic arguments (`out = torch.empty_like(x)`, so `out`'s size is `x_size0`). `y`'s size must equal `x`'s, because they share `s0`.

```python
import sympy
import cute_guards

x_size0, y_size0, n = (sympy.Symbol(s, integer=True, nonnegative=True) for s in ("x_size0", "y_size0", "n"))
x_ptr, y_ptr, out_ptr = (sympy.Symbol(s, integer=True, nonnegative=True) for s in ("x_ptr", "y_ptr", "out_ptr"))
args = [
    {"ptr": x_ptr, "shape": [x_size0], "stride": [1]},
    {"ptr": y_ptr, "shape": [y_size0], "stride": [1]},
    {"ptr": out_ptr, "shape": [x_size0], "stride": [1]},  # out = torch.empty_like(x)
    n,
]
for cond in cute_guards.binder_checks(compiled, args):
    if cond != sympy.true:  # a stride of 1 checked against a static 1
        print(cond)
```

Output:

```
x_size0 < 2147483648
Eq(Mod(x_size0, 16), 0)
Eq(Mod(x_ptr, 16), 0)
Eq(y_size0, x_size0)
Eq(Mod(y_ptr, 16), 0)
Eq(Mod(out_ptr, 16), 0)
n < 2147483648
```

## 6. The launch, from the host function's IR

`HostProgram` (in `cute_ir.py`) parses the host function with MLIR's parser, in an MLIR context where the DSL's dialects aren't registered, so no value caster can run. It reads the IR through the bindings:
- ops by name;
- operands, either as the op that defines them or as the function argument;
- attributes such as `IntegerAttr` constants and the callee `SymbolRefAttr`, and builtin integer types' widths.

It evaluates the operands of each `cuda.launch_ex`: the grid, from `cuda.launch_cfg.create`, and the kernel arguments. A kernel parameter's layout comes from the spec of the function argument passed: a tensor is its address, then each dynamic size and stride, at the spec's width.

The example evaluates integer `arith` ops. Anything else declines with the op's name. In particular, an op whose result is a CuTe int tuple, shape or layout declines: its static leaves and widths are only in its type, which the bindings don't take apart.

```python
%%writefile cute_ir.py
from dataclasses import dataclass

import sympy
from cutlass._mlir._mlir_libs._cutlass_ir._mlir import ir
from cutlass.base_dsl.tvm_ffi_builder import spec
from sympy import And
from torch.utils._sympy.functions import FloorDiv

from cute_compile import binder_specs, host_functions


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
LAUNCH_ATTRIBUTES = {"cuda.launch_cfg.programmatic_stream_serialization_allowed", "cuda.launch_cfg.cooperative"}  # must be 0


class HostProgram:
    """A compile's host function, parsed by MLIR's parser into a context of ours where the DSL's dialects
    aren't registered (so none of its value casters runs), and read through the bindings: ops, operands,
    attributes and builtin types. Kernel parameters come from the binder spec of the formal passed."""

    def __init__(self, text, binder_spec):
        self.spec = binder_spec
        self.context = ir._Context()
        self.context.allow_unregistered_dialects = True
        with self.context:
            self.module = ir.Module.parse(text)
            body = self.module.body.operations[0].operation.regions[0].blocks[0]
            self.formal_types = [a.type for a in body.arguments]
            self.ops = [op.operation for op in body.operations]  # the top level only

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
                return formals[i]
            k = next(i for i, o in enumerate(self.ops) if o == op)
            if k not in cache:
                cache[k] = self.op(op, [value(u) for u in op.operands], conditions)
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

    def op(self, op, a, conditions):
        n = op.name
        if n == "arith.constant":
            return ir.IntegerAttr(op.attributes["value"]).value
        if n in ARITH:
            r = ARITH[n](*a)
            bits = ir.IntegerType(op.results[0].type).width
            conditions.append(And(r >= -(2 ** (bits - 1)), r < 2 ** (bits - 1)))  # no wraparound
            return r
        t = op.results[0].type if len(op.results) else None
        if t is not None and ir.OpaqueType.isinstance(t) and ir.OpaqueType(t).dialect_namespace == "cute":
            # a CuTe int tuple, shape or layout: its static leaves and widths are in its type, which
            # MLIR's bindings don't take apart
            raise NotImplementedError(f"{n}: its result is a CuTe value whose structure is only in its type, {t}")
        raise NotImplementedError(f"unsupported host op {n}")


def host_program(compiled):
    name = compiled.function_name
    if name not in host_functions or name not in binder_specs:
        raise NotImplementedError(f"{name} was compiled before observe_compiles: no host function or spec")
    return HostProgram(host_functions[name], binder_specs[name])
```

The host function of our compile, evaluated over the same symbolic arguments. The grid is `(n + 127)//128`. Each tensor parameter is its address plus its one dynamic size. The two integer ops in the grid may not wrap around, so they add two conditions. Only the ops the launch's operands reach are evaluated; the others (`cute.get_iter`, `cute.get_layout`, ...) are never read.

```python
import cute_ir

program = cute_ir.host_program(compiled)
print("host function ops:", ", ".join(op.name for op in program.ops))
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
kernel: kernel_cutlass_add_kernel_tensorptrf32gmemalign16odiv161_tensorptrf32gmemalign16odiv161_tensorptrf32gmemalign16odiv161__0
grid: (((n + 127)//128), 1, 1)
parameter 0: x_ptr (8 bytes at 0), x_size0 (4 bytes at 8)
parameter 1: y_ptr (8 bytes at 0), y_size0 (4 bytes at 8)
parameter 2: out_ptr (8 bytes at 0), x_size0 (4 bytes at 8)
parameter 3: n (4 bytes at 0)
no wraparound: [n + 127 < 2147483648, (((n + 127)//128)) < 2147483648]
```

## 7. Tracing and replaying

The rest is the Triton notebook's tracer and runtime, with one CuTe-specific step in the middle.

- **`cute_calls.py`** intercepts calls of compiled functions, and makes a real launch at given values. Each tensor is placed in a small scratch buffer at the same alignment, because TVM-FFI reads only a tensor's metadata and the kernel never runs during capture.
- **`tracing.py` runs the user code under a capture.** The inputs are fake tensors with symbolic sizes and an unbacked symbolic `data_ptr()`. Allocations get fresh address symbols. `ShapeEnv` records the host code's branches on sizes, with the user's line. At each intercepted call, `on_call` launches at the traced values (so the capture has a node to update), guards on the binder's checks, and evaluates the host function. Every decision made from the traced call's values is a guard, so no value of that call is used without one.
- **`replay.py` checks the evaluation against the capture**, then serves later calls. Each call checks the guards, writes the parameters and grid into the kernel node, and replays. A trace that declines (`NotImplementedError`) leaves the function eager from then on, with the reason.

```python
%%writefile cute_calls.py
import contextlib
from dataclasses import dataclass

import torch
from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunction, TVMFFIJitCompiledFunctionWithKwargs
from torch.utils._python_dispatch import _disable_current_modes


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

import cute_calls
import cute_guards
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
        # For cute_calls.launch. Made before capture, so it is not graph memory.
        self.scratch = torch.empty(16, dtype=torch.uint8, device="cuda")

        with torch.cuda.graph(graph), self.mode, Allocations(self), cute_calls.intercept_calls(self.on_call):
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
            return cute_calls.TensorArg(v.dtype, shape, stride, self.at_hint(v.data_ptr()))
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
        cute_calls.launch(compiled, [self.at_hint(a) for a in args], self.scratch)

        # Each argument's value: a tensor is a dict of expressions.
        values = [self.view(a) if isinstance(a, torch.Tensor) else self.expr(a) for a in args]
        # TVM-FFI's binder checks, guarded at their outcome. The binder raises on a call that breaks
        # one, so a guard that fails sends the call to eager, which raises.
        for cond in cute_guards.binder_checks(compiled, values):
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

        # The host function was read from text, so check it: at the values the capture launched
        # with, its evaluation must give the captured bytes (padding aside) and grid.
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
    """Call fn through CUDA graphs. A call replays the first graph whose guards hold.
    Otherwise it runs fn eagerly, which warms up these inputs and gives the call's
    result, and before returning traces at the same inputs under capture. A trace that
    declines (NotImplementedError) leaves fn eager from then on, with the reason."""

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
        result = self.fn(**tensors)
        try:
            replay = Replay(self.fn, **tensors)
        except NotImplementedError as e:
            self.declined = str(e)
            return result, "; ".join(misses + [f"eager; the trace declined: {e}"])
        if self.replays:
            self.retrace_causes.append(misses[0].split(": ", 1)[1])
        self.replays.append(replay)
        return result, "; ".join(misses + [f"eager, then traced replay {len(self.replays) - 1}"])
```

The first call runs eagerly, then is traced as replay 0. Each guard prints with where it comes from:
- the user code's branch, which `ShapeEnv` recorded with its line. It goes first, because it decides which compile the user code calls;
- the binder, as computed in section 5;
- the host function's no-wraparound conditions;
- the launch: a non-empty grid, and each 4-byte field in range.

No guard pins the size.

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
n=4096: eager, then traced replay 0; out == x + y: True
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

- n=8192 and n=4112 hit replay 0, with no new trace or compile.
- n=4097 is not divisible by 16, so it misses replay 0 on the user code's branch, runs eagerly (which compiles a `div=1` version), and is traced as replay 1. The retrace cause names the guard and the user's line.
- n=4100 then hits replay 1, and n=4096 still hits replay 0. A hit doesn't list the replays it missed first.

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
n=4097: replay 0 miss: (Eq(Mod(x_size0, 16), 0)) is False, from user_code.py:28 in add; eager, then traced replay 1; out == x + y: True
n=4100: replay 1 hit; out == x + y: True
n=4096: replay 0 hit; out == x + y: True
compiles: div=[1, 16]
retrace causes: ['(Eq(Mod(x_size0, 16), 0)) is False, from user_code.py:28 in add']
```

### A call the binder rejects

With `y` shorter than `x`, replay 0 misses on `y_size0 == x_size0`, so the call runs eagerly, and the binder raises its own error. Without that guard, the replay would run the kernel over `x`'s size, reading past the end of `y`. An earlier version of our implementation did exactly that, because it took guards from each tensor's printed type alone, which doesn't say that sizes share a symbol.

```python
print(f"replay 0, y shorter than x: {f.replays[0](x=x[:4096], y=y[:2048])[0]}")
try:
    call("y shorter than x", x[:4096], y[:2048])
except ValueError as e:
    print(f"eager: ValueError: {str(e).splitlines()[0]}")
```

Output:

```
replay 0, y shorter than x: miss: (Eq(y_size0, x_size0)) is False, from binder
eager: ValueError: Mismatched my.shape[0] on argument #1 when calling: `add_host(mx: Tensor([n0], float32), my: Tensor([n0], float32), mout: Tensor([n0], float32), n: int32)`, expected to match mx.shape[0]
```

### A grid computed from layouts: declined

The same kernel, with the grid computed from layouts instead of from `n`:
- `cute.size(mx, mode=[0])`, the tensor's size;
- a nested layout `(4,(8,blocks))` of the blocks;
- the grid as its size divided by 32.

These values are CuTe layouts and int tuples. The next cell prints their types. The static parts (`127`, `128`, the nested shape, the strides the DSL computed) appear only inside the printed types, and MLIR's Python bindings for CuTe's types have no accessors for them. Reading them would mean parsing the printed type, which this notebook doesn't do, so `HostProgram` declines at the first such op.

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
import importlib, layout_code
importlib.reload(layout_code)
layout_code.add(x[:4096], y[:4096])  # compiles, observed
program = cute_ir.host_program(layout_code.compiled["add"])
for op in program.ops:  # the CuTe values the grid is computed from, and their printed types
    if op.name in ("cute.size", "cute.make_int_tuple", "cute.make_shape", "cute.make_layout"):
        print(f"{op.name} -> {op.results[0].type}")
try:
    program.evaluate(args + [None])
except NotImplementedError as e:
    print(f"declined: {e}")
```

Output:

```
cute.size -> !cute.int_tuple<"?">
cute.make_int_tuple -> !cute.int_tuple<"127">
cute.make_int_tuple -> !cute.int_tuple<"128">
cute.make_shape -> !cute.shape<"(4,(8,?))">
cute.make_layout -> !cute.layout<"(4,(8,?)):(1,(4,32))">
cute.size -> !cute.int_tuple<"?{div=32}">
cute.make_int_tuple -> !cute.int_tuple<"32">
declined: cute.size: its result is a CuTe value whose structure is only in its type, !cute.int_tuple<"?">
```

Called through `DynamicCudaGraph`, the first call's trace declines, and every call runs eagerly, with the reason. The results are eager's.

Our full implementation tried a structured route for these values: evaluating each layout with the DSL's own layout algebra on a copy of the module at compile time. On GB300 it corrupted the heap in the sm100 GEMM tests, because it ran the DSL's value casters on every value of the module, MMA and TMA atoms included, so it was pulled. A version that evaluates only layout-typed values is pending, for a later snapshot ("candidate 4"). It is not shown here. The current snapshot still reads these layout types from their printed form (listed in section 9), which is what that work replaces.

```python
g = replay.DynamicCudaGraph(layout_code.add)
big_x, big_y = torch.randn(2, 100000, device="cuda")
for n in (4096, 8192, 777):
    out, how = g(x=big_x[:n], y=big_y[:n])
    torch.cuda.synchronize()
    print(f"n={n}: {how}; out == x + y: {torch.equal(out, big_x[:n] + big_y[:n])}")
```

Output:

```
n=4096: eager; the trace declined: cute.size: its result is a CuTe value whose structure is only in its type, !cute.int_tuple<"?">; out == x + y: True
n=8192: eager (declined: cute.size: its result is a CuTe value whose structure is only in its type, !cute.int_tuple<"?">); out == x + y: True
n=777: eager (declined: cute.size: its result is a CuTe value whose structure is only in its type, !cute.int_tuple<"?">); out == x + y: True
```

## 8. Our implementation on the same calls

The same user code, through our full implementation's entry point, `HostTraceReplay` (a private API in our build). Its first call runs eagerly and its second call traces; `redispatches` counts ops dispatched again without a trace; `retrace_causes` maps each reason for a trace after the first, as (class, op, guard, the user's line, a redispatch's refusal), to its count.

- **Plain `user_code.add`.** Which compile runs is decided by the wrapper's Python (`n % 16 == 0`), outside any op, so that guard belongs to the graph: n=4097 traces again, and the cause names the guard and the user's line. Its class is `meta`: a guard of the graph's that an op recorded too (the CuTe call's binder checks the same divisibility). Class `graph` would be a guard no op recorded; class `dispatch` an op's own guard whose redispatch was refused.
- **The same wrapper inside `torch.cuda._host_trace.dispatch_unit`.** A dispatch unit declares that its body is one op whose effects are its launches, allocations and outputs. Its guards are the op's own, so a flip dispatches the unit again, without a trace.

Each output is compared with eager's (`x + y`) with `torch.equal`, a bitwise comparison here.

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
  1 retrace: class meta, guard ((s61 % 16) == 0), at run_20261008/user_code.py:28, refusal None
user_code.add in a dispatch_unit: traces 1, replays 5, redispatches 1, eager 1; every output equal to eager's: True
```

## 9. Difficulties

What makes CuTe DSL kernels harder to trace than Triton's, in short; the companion document (`cute_dsl_difficulties_20261008.md`) has each one with our workaround, its size and its status.

- **The host function's ops.** Every library brings new host ops, and the evaluator has to learn each one, or decline. FlashInfer's mxfp8 GEMM needs about 9 more. Our full implementation interprets about 45.
- **Types without structured accessors.** MLIR's Python bindings for CuTe's types have no per-leaf accessors, and `LayoutType.stride` on a nested layout `(4,(8,?))` raises `std::get: wrong index for variant`. So this notebook declines layouts and int tuples. Our current snapshot still reads them, `cute.assume`'s divisibility (`ConstrainedIntType` has no accessors) and a TMA atom's fields from their printed types; these are the places left to move onto structured data. The DSL-evaluated route for layouts is pending (section 7).
- **Reading live IR values runs the DSL's casters.** They add ops to the module, and on some values they have corrupted the heap. We only print the live module and read a parsed copy in a context without the DSL's dialects.
- **DSL internals.** We rely on the hook, the private `CuTeDSL._get_dsl()`, the signature of `compile_and_cache`, `_tvm_ffi_args_spec_converter`, and the compiled classes' `__call__`. A DSL change there makes calls decline, not run wrongly. The check against the capture catches a wrong evaluation.
- **Re-implemented binder checks.** `binder_checks` mirrors what the binder checks. The binder has no check-only entry point: its checks are inlined in the generated wrapper. If the DSL adds a check, the guards miss it until updated.
- **Objects loaded from disk carry nothing.** A function loaded from a library's compile cache has neither its host function nor its spec, so we write our own files beside the objects, versioned and tied to the object's digest, and decline a load without them.
- **Tensors and streams.** A traced tensor has no storage for `from_dlpack`, so we swap a module global during traces. A launch's stream is implicit unless it is a formal, so we decline other launches.

## 10. What we'd ask of the CuTe DSL team

Each request removes a part of the `cute_*.py` files (or of our full implementation) that today depends on DSL internals. They are ordered by how much each removes.

1. **A launch descriptor on the compiled function.** For each launch: the kernel, the grid, block, shared memory and cluster as expressions over the arguments, the launch attributes, which argument is the stream, and each kernel parameter's fields with their source (an argument's address, a size or stride, a scalar, a TMA descriptor's encode arguments).
   - *Simpler because:* section 6's evaluator and its op list disappear, layouts stop declining, and the capture becomes a debug check.
   - *Cost for them:* the DSL computes all of this when it lowers the host function; this is a serialization of it, with a version.
   - *Smaller fallback:* `compiled.prepare(*args)`, returning the launches for concrete arguments without launching.
2. **The argument spec as public data on the compiled function**, e.g. `compiled.args_spec`.
   - *Simpler because:* section 4's wrapper of `compile_and_cache` and the private converter call go away.
   - *Also useful:* a check-only binder entry, `compiled.check_args(*args)`, so the guards can be tested against the binder instead of mirroring it.
3. **Both carried through `export_to_c` and loading.**
   - *Why:* a function loaded from disk, e.g. from a library's compile cache, has neither today, so it can't be traced. We keep our own files beside the objects.
4. **Structured accessors on the CuTe types in the MLIR bindings.** These are per-leaf static values or symbols, divisibility and widths on the layout and tuple types; the divisibility of `ConstrainedIntType`; a TMA atom's fields; and a fix for `LayoutType.stride` on nested layouts.
   - *Simpler because:* layouts and int tuples could be read straight from the IR, and `cute.assume` and TMA kernels would stop depending on printed types.
   - *When:* needed unless request 1 lands. They are small, and the repro needs no GPU.
5. **CuTe tensors from metadata with a deferred address**, for `from_dlpack`, fake tensors and `make_ptr`.
   - *Why:* a traced tensor has no storage to export. Today we swap a module global during traces.
6. **A supported call hook on compiled functions**, instead of patching `__call__` on three classes.
7. **Nice to have:**
   - a compile callback that is given the compiled object (moot if requests 1 and 2 are attributes of the compiled object);
   - stream handles kept out of the mangled function name (the regex strips only 8 to 16 hex digits, and this machine's handles have 7), so a jit call doesn't compile once per stream;
   - a documented TMA descriptor layout or re-encode, if request 1 doesn't describe TMA parameters.
