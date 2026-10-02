# Owner(s): ["module: cuda graphs"]

import ctypes
import hashlib
import inspect
import itertools
import types
import unittest

import sympy

import torch
from torch.cuda import _host_trace_replay
from torch.cuda._host_trace import Declined
from torch.cuda._host_trace_capture import capture_tape
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_lower import Lowering
from torch.cuda._host_trace_lower_tape import lower_tape
from torch.cuda._host_trace_program import compile_program, IntegerProgram
from torch.cuda._host_trace_tape import EagerCall, trace
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    TEST_CUDA,
    TestCase,
)
from torch.utils._triton import has_triton


class HostTraceReplay(_host_trace_replay.HostTraceReplay):
    # traces at its first call: these are tests of the trace; an entry's first
    # call runs eagerly (test_the_first_call_runs_eagerly in
    # test_cuda_host_trace_replay)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._called = True


HAS_TRITON = has_triton()
if HAS_TRITON:
    import triton
    import triton.language as tl
    from triton import knobs
    from triton._C.libtriton import native_specialize_impl
    from triton.backends.compiler import BaseBackend
    from triton.runtime.autotuner import Autotuner, Config
    from triton.runtime.jit import (
        compute_cache_key,
        create_function_from_signature,
        JITFunction,
        KernelParam,
        serialize_specialization_data,
    )

    from torch.cuda import _host_trace_triton_launch as htl

    _TRITON_RUN = JITFunction.run

    @triton.jit
    def _body(x_ptr, y_ptr, n, s, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) + s, mask=m)

    @triton.jit
    def _add(x_ptr, y_ptr, n, s, B: tl.constexpr):
        _body(x_ptr, y_ptr, n, s, B)

    @triton.jit(do_not_specialize=["s"])
    def _add_dns(x_ptr, y_ptr, n, s, B: tl.constexpr):
        _body(x_ptr, y_ptr, n, s, B)

    @triton.jit(do_not_specialize_on_alignment=["s"])
    def _add_dnsa(x_ptr, y_ptr, n, s, B: tl.constexpr):
        _body(x_ptr, y_ptr, n, s, B)

    @triton.jit
    def _add_i64(x_ptr, y_ptr, n, s: tl.int64, B: tl.constexpr):
        _body(x_ptr, y_ptr, n, s, B)

    @triton.jit
    def _add_fresh(x_ptr, y_ptr, n, s, B: tl.constexpr):
        _body(x_ptr, y_ptr, n, s + 1, B)

    @triton.heuristics(
        {
            "B": lambda a: triton.next_power_of_2(a["n"]),
            "EVEN": lambda a: a["n"] % 16 == 0,
        }
    )
    @triton.jit
    def _add_heuristic(x_ptr, y_ptr, n, s, B: tl.constexpr, EVEN: tl.constexpr):
        _body(x_ptr, y_ptr, n, s, B)

    @triton.autotune(configs=[triton.Config({"B": 128})], key=["n"])
    @triton.jit
    def _add_tuned(x_ptr, y_ptr, n, s, B: tl.constexpr):
        _body(x_ptr, y_ptr, n, s, B)

    _CONFIGS = [triton.Config({"B": 128}), triton.Config({"B": 256}, num_warps=8)]

    @triton.autotune(configs=_CONFIGS, key=["n"])
    @triton.jit
    def _add_tuned2(x_ptr, y_ptr, n, s, B: tl.constexpr):
        _body(x_ptr, y_ptr, n, s, B)

    @triton.autotune(configs=_CONFIGS, key=["n"], reset_to_zero=["y_ptr"])
    @triton.jit
    def _add_tuned_reset(x_ptr, y_ptr, n, s, B: tl.constexpr):
        _body(x_ptr, y_ptr, n, s, B)

    @triton.autotune(
        configs=[triton.Config({"B": 128}, pre_hook=lambda nargs: None)], key=["n"]
    )
    @triton.jit
    def _add_tuned_hook(x_ptr, y_ptr, n, s, B: tl.constexpr):
        _body(x_ptr, y_ptr, n, s, B)

    @triton.jit
    def _scale_body(x_ptr, y_ptr, n, a, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) * a, mask=m)

    @triton.jit
    def _scale(x_ptr, y_ptr, n, a, B: tl.constexpr):
        _scale_body(x_ptr, y_ptr, n, a, B)

    @triton.jit
    def _scale_fp16(x_ptr, y_ptr, n, a: tl.float16, B: tl.constexpr):
        _scale_body(x_ptr, y_ptr, n, a, B)

    @triton.jit
    def _scale_bf16(x_ptr, y_ptr, n, a: tl.bfloat16, B: tl.constexpr):
        _scale_body(x_ptr, y_ptr, n, a, B)

    @triton.jit
    def _scale_fp64(x_ptr, y_ptr, n, a: tl.float64, B: tl.constexpr):
        _scale_body(x_ptr, y_ptr, n, a, B)

    @triton.jit
    def _layer_norm(X, Y, W, Bias, stride, N, eps, BLOCK: tl.constexpr):
        # the layer-norm tutorial's forward, one row per program
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
        mean = tl.sum(x, axis=0) / N
        xc = tl.where(mask, x - mean, 0.0)
        rstd = 1 / tl.sqrt(tl.sum(xc * xc, axis=0) / N + eps)
        w = tl.load(W + cols, mask=mask)
        b = tl.load(Bias + cols, mask=mask)
        tl.store(Y + row * stride + cols, xc * rstd * w + b, mask=mask)

    @triton.jit
    def _dropout(x_ptr, y_ptr, n, p, seed, B: tl.constexpr):
        # the low-memory dropout tutorial's seeded kernel
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        x = tl.load(x_ptr + i, mask=m)
        keep = tl.rand(seed, i) > p
        tl.store(y_ptr + i, tl.where(keep, x / (1 - p), 0.0), mask=m)

    @torch.library.custom_op("host_trace_test::add_one", mutates_args=(), device_types="cuda")
    def _add_one(x: torch.Tensor) -> torch.Tensor:
        # the block is picked from the size after the fake kernel fixed the output
        y = torch.empty_like(x)
        n = x.numel()
        b = 1024 if n >= 4096 else 128
        _add[(triton.cdiv(n, b),)](x, y, n, 1, B=b)
        return y

    _add_one.register_fake(lambda x: torch.empty_like(x))

    @triton.jit
    def _neg(x_ptr, y_ptr, n, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(y_ptr + i, -tl.load(x_ptr + i, mask=m), mask=m)

    def _native_neg(x):
        y = torch.empty_like(x)
        n = x.numel()
        b = 1024 if n >= 4096 else 128
        _neg[(triton.cdiv(n, b),)](x, y, n, B=b)
        return y


def _launches(tape):
    return [record for _, record in tape.launches]


def _named(tape, v):
    # an int, SymInt or sympy expression with each symbol named by its source
    e = v.node.expr if isinstance(v, torch.SymInt) else sympy.sympify(v)
    sources = tape.shape_env.var_to_sources
    names = {s: sympy.Symbol(sources[s][0].name) for s in e.free_symbols}
    return str(e.xreplace(names))


def _holds(tape, **values):
    # whether every guard holds with the named symbols (arg1, arg0_base, ...)
    # at `values` and every other symbol at its hint
    env = tape.shape_env
    m = {}
    for s, hint in env.backed_var_to_val.items():
        name = env.var_to_sources[s][0].name
        name = name.replace(".base", "_base").replace(".storage_offset()", "_offset")
        m[s] = sympy.Integer(values.get(name, hint))
    return all(g.xreplace(m) is sympy.true for g in tape.guards)


def _add_call(kernel):
    def f(x, n, s):
        y = torch.empty_like(x)
        kernel[(triton.cdiv(n, 128),)](x, y, n, s, B=128)
        return y

    return f


def _tuned_call(kernel):
    def f(x, n, s):
        y = torch.empty_like(x)
        kernel[lambda meta: (triton.cdiv(n, meta["B"]),)](x, y, n, s)
        return y

    return f


def _layer_norm_call(x, w, b, eps):
    y = torch.empty_like(x)
    M, N = x.shape
    block = triton.next_power_of_2(N)
    num_warps = min(max(block // 256, 1), 8)
    _layer_norm[(M,)](x, y, w, b, x.stride(0), N, eps, BLOCK=block, num_warps=num_warps)
    return y


def _dropout_call(x, p, seed):
    y = torch.empty_like(x)
    n = x.numel()
    _dropout[(triton.cdiv(n, 1024),)](x, y, n, p, seed, B=1024)
    return y


def _node_bytes(node, layout):
    from cuda.bindings import driver

    p = driver.cuGraphKernelNodeGetParams(node)[1]
    held = (ctypes.c_void_p * len(layout)).from_address(int(p.kernelParams))
    return tuple(ctypes.string_at(ptr, size) for ptr, (_, size) in zip(held, layout))


@unittest.skipIf(not (TEST_CUDA and HAS_TRITON), "requires CUDA and triton")
@requires_cuda_python_bindings
class TestTritonLaunch(TestCase):
    def setUp(self):
        self.x = torch.randn(2048, device="cuda")

    def test_launch_record(self):
        tape = trace(_add_call(_add), (self.x, 1000, 3))
        self.assertIs(JITFunction.run, _TRITON_RUN)
        (launch,) = _launches(tape)
        self.assertEqual(launch.name, "_add")
        eager = _add[(1,)](self.x, self.x.clone(), 1000, 3, B=128)
        # the tape's own load of the kernel's module
        self.assertIs(launch.owner, htl.owned_module(eager))
        self.assertEqual(launch.function, launch.owner.function)
        slots = [_named(tape, v) for v in launch.slots]
        x_ptr, y_ptr = "arg0.base + 4*arg0.storage_offset()", "256*alloc0.base/256"
        self.assertEqual(slots, [x_ptr, y_ptr, "arg1", "arg2", "0", "0"])
        grid = [_named(tape, v) for v in launch.grid]
        self.assertEqual(grid, ["((arg1 + 127)//128)", "1", "1"])
        self.assertEqual(launch.block, (launch.abi.num_warps * 32, 1, 1))
        self.assertEqual(launch.smem, eager.metadata.shared)
        self.assertEqual(len(launch.layout), launch.abi.num_slots)
        self.assertEqual([r.name for r in launch.roots], ["p0", "a0"])
        # the output reads the kernel's result
        out = _add_call(_add)(self.x, 1000, 3)
        self.assertEqual(out[:1000], self.x[:1000] + 3)

    @parametrize(
        "kernel,value,same,other",
        [
            ("_add", 1, [], [2]),  # == 1: a constexpr
            ("_add", 32, [48, 2**31 - 16], [33, 1, 2**31]),  # % 16, i32 width
            ("_add", 2**31, [2**31 + 16], [2**31 - 16]),  # i64 width
            ("_add_dns", 32, [1, 33], [2**31]),  # the width alone
            ("_add_dnsa", 32, [33], [1, 2**31]),  # == 1 and the width
            ("_add_i64", 32, [2**40], [33, 1]),  # annotated: no width class
        ],
    )
    def test_integer_axes(self, kernel, value, same, other):
        tape = trace(_add_call(globals()[kernel]), (self.x, 1000, value))
        self.assertTrue(_holds(tape, arg2=value))
        for v in same:
            self.assertTrue(_holds(tape, arg2=v), v)
        for v in other:
            self.assertFalse(_holds(tape, arg2=v), v)

    def test_integer_specializations_flip(self):
        def signature(s):
            (launch,) = _launches(trace(_add_call(_add), (self.x, 1000, s)))
            (arg,) = [a for a in launch.abi.args if a.name == "s"]
            return arg.triton_type, arg.divisibility

        self.assertEqual(signature(1), ("constexpr", 1))
        self.assertEqual(signature(32), ("i32", 16))
        self.assertEqual(signature(33), ("i32", 1))
        self.assertEqual(signature(2**31 + 16), ("i64", 16))

    def test_pointer_axes(self):
        base = torch.randn(4096, device="cuda")
        f = _add_call(_add)
        tape = trace(f, (base[4:], 1000, 3))
        (launch,) = _launches(tape)
        # an input's address is guarded
        self.assertEqual(launch.abi.args[0].divisibility, 16)
        self.assertTrue(_holds(tape, arg0_offset=8))
        self.assertFalse(_holds(tape, arg0_offset=5))
        hint = tape.inputs[0].root.sym.node.hint
        self.assertFalse(_holds(tape, arg0_base=hint + 4))
        # an allocation's is decided by its offset alone (0 until views are
        # traced), with no guard
        self.assertEqual(launch.abi.args[1].divisibility, 16)
        self.assertFalse(any("alloc0" in _named(tape, g) for g in tape.guards))
        misaligned = trace(f, (base[1:], 1000, 3))
        self.assertEqual(_launches(misaligned)[0].abi.args[0].divisibility, 1)
        self.assertFalse(_holds(misaligned, arg0_offset=4))

    def test_heuristics_run_on_symints(self):
        def f(x, n, s):
            y = torch.empty_like(x)
            _add_heuristic[(1,)](x, y, n, s)
            return y

        tape = trace(f, (self.x, 1000, 3))
        (launch,) = _launches(tape)
        self.assertEqual(launch.abi.args[4].constant, 1024)
        self.assertIs(launch.abi.args[5].constant, False)
        self.assertTrue(_holds(tape, arg1=1001))
        self.assertFalse(_holds(tape, arg1=1008))  # EVEN flips
        self.assertFalse(_holds(tape, arg1=1025))  # B flips
        # next_power_of_2's guard lowers into the compiled predicate
        env = tape.shape_env
        (n,) = (s for s, src in env.var_to_sources.items() if src[0].name == "arg1")
        program = IntegerProgram([self.x, 1000, 3])
        guards = [g for g in tape.guards if g.free_symbols == {n}]
        predicate = Lowering(program, {n: ("boxed", 1)}).predicate(guards)
        compiled = compile_program(program)
        results = [compiled.evaluate_inputs([self.x, v, 3]) for v in (1001, 1008, 1025)]
        self.assertEqual([(status, values[predicate]) for status, values in results], [(0, 1), (0, 0), (0, 0)])

    def test_zero_grid_records_no_launch(self):
        tape = trace(_add_call(_add), (self.x, 0, 3))
        self.assertEqual(tape.launches, [])
        self.assertFalse(_holds(tape, arg1=5))

    def test_without_warm_up(self):
        tape = trace(_add_call(_add_fresh), (self.x, 1000, 3), warm_up=False)
        (launch,) = _launches(tape)
        self.assertEqual(launch.name, "_add_fresh")

    @parametrize("kernel", ["_scale", "_scale_fp16", "_scale_bf16", "_scale_fp64"])
    def test_float_slot_bytes_match_tritons_own_launch(self, kernel):
        from cuda.bindings import runtime

        def fn(x, y, a):
            globals()[kernel][(8,)](x, y, 1000, a, B=128)
            return y

        x, y = self.x, torch.empty_like(self.x)
        # 1/3 rounds in fp32 and truncates differently in bf16
        for a in (1 / 3, -2.5, 1e-5):
            # y is an argument: the tape allocates nothing
            captured = capture_tape(lower_tape(trace(fn, (x, y, a))), [])
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                fn(x, y, a)
            g = graph.raw_cuda_graph()
            _, (node,), _ = runtime.cudaGraphGetNodes(g, numNodes=1)
            (ours,) = captured.launches
            layout = ours.launch.launch.layout
            want = _node_bytes(int(node), layout)
            self.assertEqual(_node_bytes(ours.node, layout), want, (kernel, a))

    def test_layer_norm_with_eps(self):
        f = HostTraceReplay(_layer_norm_call)
        for eps in (1e-5, 1e-3):
            for M, N in ((64, 1000), (32, 1000), (64, 1020), (16, 700)):
                x = torch.randn(M, N, device="cuda")
                w, b = torch.randn(N, device="cuda"), torch.randn(N, device="cuda")
                want = torch.nn.functional.layer_norm(x, (N,), w, b, eps)
                self.assertEqual(f(x, w, b, eps), want, atol=1e-4, rtol=1e-4)
        # one trace per eps: num_warps is a function of N's power of two
        self.assertEqual((f.traces, f.replays, f.declines), (2, 6, []))

    def test_dropout_with_float_p(self):
        f = HostTraceReplay(_dropout_call)
        for p in (0.5, 0.25):
            for n in (1000, 5000, 1000):
                x = torch.randn(n, device="cuda")
                self.assertEqual(f(x, p, 7), _dropout_call(x, p, 7))
        self.assertEqual(f.declines, [])
        self.assertGreater(f.replays, 0)

    def test_launch_option_from_sizes(self):
        def f(x, n, s):
            y = torch.empty_like(x)
            _add[(triton.cdiv(n, 128),)](x, y, n, s, B=128, num_warps=n // 256)
            return y

        tape = trace(f, (self.x, 1024, 3))
        (launch,) = _launches(tape)
        self.assertEqual(launch.abi.num_warps, 4)
        self.assertTrue(_holds(tape, arg1=1040))
        self.assertFalse(_holds(tape, arg1=1280))
        self.assertFalse(_holds(tape, arg1=1008))

    def test_autotune_single_config(self):
        tape = trace(_tuned_call(_add_tuned), (self.x, 1000, 3))
        (launch,) = _launches(tape)
        self.assertEqual(launch.abi.args[4].constant, 128)
        self.assertTrue(_holds(tape, arg1=1016))

    def test_autotune_multi_config(self):
        f = _tuned_call(_add_tuned2)
        # the autotuner's benchmark (its scratch) at a warm-up is not the trace's
        f(self.x, 1000, 3)
        tape = trace(f, (self.x, 1000, 3))
        (launch,) = _launches(tape)
        config = _add_tuned2.cache[(1000, "torch.float32", "torch.float32")]
        self.assertEqual(launch.abi.args[4].constant, config.kwargs["B"])
        self.assertEqual(launch.abi.num_warps, config.num_warps)
        self.assertFalse(_holds(tape, arg1=1016))  # the key is pinned
        with self.assertRaisesRegex(Declined, "not in the cache"):
            trace(f, (self.x, 999, 3), warm_up=False)
        # a later trace does not warm up: its key must have run eagerly
        f(self.x, 2000, 3)
        r = HostTraceReplay(f)
        for n in (1000, 2000, 1000, 2000):
            self.assertEqual(r(self.x, n, 3)[:n], self.x[:n] + 3)
        self.assertEqual((r.traces, r.replays, r.declines), (2, 2, []))

    def test_warmup_call_records_no_launch(self):
        compiled = []

        def f(x, n, s):
            compiled.append(_add.warmup(x, torch.float32, n, s, B=128, grid=(1,)))
            return x

        tape = trace(f, (self.x, 1000, 3), warm_up=False)
        self.assertEqual(tape.launches, [])
        eager = _add[(1,)](self.x, self.x.clone(), 1000, 3, B=128)
        self.assertIs(compiled[0], eager)
        self.assertFalse(_holds(tape, arg1=1008))  # its specialization is guarded

    def test_declines(self):
        z = torch.empty_like(self.x)

        def untracked(x, n, s):
            _add[(1,)](x, z, n, s, B=128)
            return x

        def other_stream(x, n, s):
            with torch.cuda.stream(torch.cuda.Stream()):
                _add[(1,)](x, x, n, s, B=128)
            return x

        def sym_float(x, n, s):
            _add[(1,)](x, x, n, n / 2, B=128)
            return x

        cases = {
            "does not track": (untracked, 3),
            "reset_to_zero": (_tuned_call(_add_tuned_reset), 3),
            "config pre_hook": (_tuned_call(_add_tuned_hook), 3),
            "stream other than": (other_stream, 3),
            "float argument s is a SymFloat": (sym_float, 3),
        }
        for msg, (f, s) in cases.items():
            with self.assertRaisesRegex(Declined, msg):
                trace(f, (self.x, 1000, s))
            self.assertIs(JITFunction.run, _TRITON_RUN)

        def tensor_option(x, n, s):
            _add[(1,)](x, x, n, s, B=128, num_warps=x)
            return x

        with self.assertRaisesRegex(Declined, "launch option num_warps is a tensor"):
            trace(tensor_option, (self.x, 1000, 3), warm_up=False)

    def test_a_launch_whose_compilation_declines_is_an_eager_call(self):
        def pdl(x, n, s):
            y = torch.empty_like(x)
            _add[(triton.cdiv(n, 128),)](x, y, n, s, B=128, launch_pdl=True)
            return y

        tape = trace(pdl, (self.x, 1000, 3))
        ((_, call),) = tape.launches
        self.assertIsInstance(call, EagerCall)
        kind, jit, grid, options = call.target
        self.assertEqual((kind, jit, options), ("triton", _add, {"launch_pdl": True}))
        self.assertEqual(_named(tape, grid[0]), "((arg1 + 127)//128)")
        self.assertEqual(grid[1:], (1, 1))
        x, y, n, s, b = call.args
        self.assertEqual((_named(tape, n), s, b), ("arg1", 3, 128))
        self.assertEqual(call.outputs, ())
        self.assertIn("programmatic-dependent", call.reason)
        # launch hooks run at every eager launch
        hook = lambda metadata: None  # noqa: E731
        knobs.runtime.launch_enter_hook.add(hook)
        try:
            tape = trace(_add_call(_add), (self.x, 1000, 3), warm_up=False)
        finally:
            knobs.runtime.launch_enter_hook.remove(hook)
        ((_, call),) = tape.launches
        self.assertIsInstance(call, EagerCall)

        # a compile-only call has no launch to run eagerly: its decline is the
        # trace's
        def warmup(x, n, s):
            _add.warmup(x, torch.float32, n, s, B=128, grid=(1,))
            return x

        _add.pre_run_hooks.append(lambda *args, **kwargs: None)
        try:
            with self.assertRaisesRegex(Declined, "pre-run hooks"):
                trace(warmup, (self.x, 1000, 3), warm_up=False)
        finally:
            _add.pre_run_hooks.clear()

    def test_an_eager_call_between_launches(self):
        def f(x, w, n, s):
            y = torch.empty_like(x)
            _add[(triton.cdiv(n, 128),)](x, y, n, s, B=128)
            z = y.view(-1, w.shape[0]) @ w
            out = torch.empty_like(z)
            _add[(triton.cdiv(z.numel(), 128),)](z, out, z.numel(), s, B=128)
            return out

        w = torch.randn(16, 8, device="cuda")
        tape = trace(f, (self.x, w, 2048, 3))
        first, call, second = (r for _, r in tape.launches)
        self.assertEqual(call.target, torch.ops.aten.mm.default)
        (z,) = call.outputs
        self.assertEqual(_named(tape, z.shape[0]), "(arg0.size(0)//arg1.size(0))")
        self.assertEqual(_named(tape, z.shape[1]), "arg1.size(1)")
        # the second launch reads the eager output: its address is e0's base
        self.assertIs(second.roots[0], z._root)
        self.assertEqual(second.slots[0].node.expr, z._root.sym.node.expr)
        # whose symbol no guard reads
        (q,) = z._root.sym.node.expr.free_symbols
        self.assertFalse(any(q in g.free_symbols for g in tape.guards))
        # which runs between the runs of the launches before and after it
        before, mm, after = lower_tape(tape).steps
        self.assertEqual((before, after), (range(1), range(1, 2)))
        self.assertIs(mm.call, call)

    def test_matmul_folds_its_batch_with_a_view(self):
        def f(x, w, n, s):
            y = torch.empty_like(x)
            _add[(triton.cdiv(n, 128),)](x, y, n, s, B=128)
            z = torch.matmul(y.view(4, -1, w.shape[0]), w)
            out = torch.empty_like(z)
            _add[(triton.cdiv(z.numel(), 128),)](z, out, z.numel(), s, B=128)
            return out

        w = torch.randn(16, 8, device="cuda")
        tape = trace(f, (self.x, w, 2048, 3))
        _, call, second = (r for _, r in tape.launches)
        self.assertEqual(call.target, torch.ops.aten.mm.default)
        # the second launch reads the mm's result through _unsafe_view, a view
        (mm,) = call.outputs
        self.assertIs(second.roots[0], mm._root)
        lower_tape(tape)

    def test_decline_is_sticky(self):
        def f(x, n):
            y = torch.empty_like(x)
            try:
                _add[(1,)](x, y, n, n / 2, B=128)
            except Declined:
                pass
            return y

        with self.assertRaisesRegex(Declined, "SymFloat"):
            trace(f, (self.x, 1000))

    def test_a_specialization_flip_dispatches_again_into_one_variant(self):
        # n's divisibility is the launch's own guard: n % 16 != 0 is another
        # binary, an entry of the one variant from the launch dispatched again
        f = _add_call(_add)
        (op,) = trace(f, (self.x, 1024, 3)).ops
        self.assertEqual(op.kind, "traced")
        self.assertTrue(op.guards)
        r = HostTraceReplay(f)
        for n in (1024, 1000, 2048, 1000):
            self.assertEqual(r(self.x, n, 3)[:n], self.x[:n] + 3)
        self.assertEqual((r.traces, r.folds, r.redispatches, len(r.variants), r.eager), (1, 0, 1, 1, 0))

    def test_a_custom_ops_kernel_choice_dispatches_again_into_one_variant(self):
        tape = trace(_add_one, (self.x,))
        self.assertEqual([op.kind for op in tape.ops], ["traced"])
        self.assertIsInstance(_launches(tape)[0], KernelLaunch)
        r = HostTraceReplay(_add_one)
        for n in (8192, 1024, 8192, 1024, 4096):
            x = torch.randn(n, device="cuda")
            self.assertEqual(r(x), x + 1, atol=0, rtol=0)
        self.assertEqual((r.traces, r.folds, r.redispatches, len(r.variants), r.eager), (1, 0, 1, 1, 0))

    def test_a_torch_native_routers_kernel_choice_dispatches_again_into_one_variant(self):
        # the router's condition, the override and its launch are aten.neg's host
        from torch._native import registry

        graphs = dict(registry._graphs)
        registry.register_op_override("host_trace_test", "aten", "neg", "CUDA", lambda x: x.dim() == 1, _native_neg)
        registry._register_overrides_from_graph("neg", "CUDA", registry._graphs[("neg", "CUDA")])
        try:
            tape = trace(torch.neg, (self.x,))
            self.assertEqual([op.kind for op in tape.ops], ["traced"])
            self.assertIsInstance(_launches(tape)[0], KernelLaunch)
            r = HostTraceReplay(torch.neg)
            for n in (8192, 1024, 8192, 1024, 4096):
                x = torch.randn(n, device="cuda")
                self.assertEqual(r(x), -x, atol=0, rtol=0)
            self.assertEqual((r.traces, r.folds, r.redispatches, len(r.variants), r.eager), (1, 0, 1, 1, 0))
        finally:
            registry._destroy_aten_override("neg", "CUDA")
            registry._graphs.clear()
            registry._graphs.update(graphs)


_REVIEWED = {
    "3.8.0": "f17f11b395889ea4ed4e98de50b6533ab191eb6f07962df9c1d538d31735735e",
}
_BOUNDS = (0, 1, 2, 15, 16, 17, 2**31 - 1, 2**31, 2**32, 2**63 - 1, 2**63, 2**64 - 1)
_STEPS = (-16, -1, 0, 1, 16)
_INT_GRID = sorted({s * (v + d) for v in _BOUNDS for d in _STEPS for s in (1, -1)})
_FLAGS = list(itertools.product((False, True), repeat=3))


def _int_specialization(v, specialize, align):
    # the symbolic run's integer rules on a plain int
    if specialize and v == 1:
        return ("constexpr", 1)
    ty = htl._int_type(v)
    if not specialize:
        return (ty, None)
    return (ty, BaseBackend.get_int_specialization(v, align=align))


class _Reads:
    """A tensor stand-in recording what native_specialize_impl reads."""

    def __init__(self, dtype, address):
        object.__setattr__(self, "reads", [])
        object.__setattr__(self, "_dtype", dtype)
        object.__setattr__(self, "_address", address)

    def __getattribute__(self, name):
        if name not in ("reads", "_dtype", "_address", "__class__"):
            object.__getattribute__(self, "reads").append(name)
        if name == "dtype":
            return object.__getattribute__(self, "_dtype")
        return object.__getattribute__(self, name)

    def data_ptr(self):
        return object.__getattribute__(self, "_address")


@unittest.skipIf(not HAS_TRITON, "requires triton")
class TestTritonSpecializationAxes(TestCase):
    """The installed Triton's specialization against what the symbolic binder
    run relies on. CPU only."""

    def test_reviewed_triton_version(self):
        pieces = [
            inspect.getsource(create_function_from_signature),
            inspect.getsource(JITFunction.run),
            inspect.getsource(JITFunction._pack_args),
            inspect.getsource(compute_cache_key),
            inspect.getsource(serialize_specialization_data),
            inspect.getsource(BaseBackend.parse_attr),
            inspect.getsource(BaseBackend.get_int_specialization),
            inspect.getsource(BaseBackend.get_tensor_specialization),
            inspect.getsource(KernelParam),
            inspect.getsource(Autotuner.__init__),
            inspect.getsource(Autotuner.run),
            inspect.getsource(Config.all_kwargs),
        ]
        digest = hashlib.sha256("\n".join(pieces).encode()).hexdigest()
        self.assertEqual(
            _REVIEWED.get(triton.__version__),
            digest,
            f"Triton {triton.__version__}'s specialization is not the reviewed one: re-derive what "
            "_host_trace_triton_launch relies on and record the new hash",
        )

    def test_integer_classes_agree_with_the_native_specialization(self):
        for v in _INT_GRID:
            for flags in _FLAGS:
                _, specialize, align = flags
                if v < -(2**63) or v > 2**64 - 1:
                    with self.assertRaises(OverflowError):
                        native_specialize_impl(BaseBackend, v, *flags)
                    with self.assertRaises(OverflowError):
                        _int_specialization(v, specialize, align)
                    continue
                self.assertEqual(
                    native_specialize_impl(BaseBackend, v, *flags),
                    _int_specialization(v, specialize, align),
                    (v, flags),
                )
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        sym = ShapeEnv().create_unbacked_symint()
        with self.assertRaisesRegex(TypeError, "failed to specialize .* SymInt"):
            native_specialize_impl(BaseBackend, sym, False, True, True)

    def test_tensor_specialization_in_python_is_the_native_one(self):
        view = htl._TensorSpecializationInPython(BaseBackend)
        for dtype in (torch.float32, torch.bfloat16, torch.int64, torch.bool):
            for address in (0, 4, 8, 16, 2**40 + 8, 2**40 + 16):
                for flags in _FLAGS:
                    native = _Reads(dtype, address)
                    want = native_specialize_impl(BaseBackend, native, *flags)
                    self.assertEqual(set(native.reads) - {"dtype"}, {"data_ptr"})
                    got = native_specialize_impl(view, _Reads(dtype, address), *flags)
                    self.assertEqual(got, want, (dtype, address, flags))

    def test_binder_flags_are_the_declarations(self):
        @triton.jit(do_not_specialize=["dns"], do_not_specialize_on_alignment=["dnsa"])
        def fn(p, k: tl.const, dns, dnsa, a: tl.int32, f: tl.float32, B: tl.constexpr):
            pass

        calls = []

        def recording(backend, arg, is_const, specialize, align):
            calls.append((arg, is_const, specialize, align))
            return (f"T{arg}", f"S{arg}")

        bind = create_function_from_signature(fn.signature, fn.params, BaseBackend)
        globals_ = bind.__globals__ | {"specialize_impl": recording}
        symbolic = types.FunctionType(bind.__code__, globals_, None, bind.__defaults__)
        names = ["p", "k", "dns", "dnsa", "a", "f", "B"]
        _, specialization, options = symbolic(*names, num_warps=2)
        self.assertEqual(options, {"num_warps": 2})
        want_calls = [
            ("p", False, True, True),
            ("k", True, True, True),
            ("dns", False, False, True),
            ("dnsa", False, True, False),
            ("a", False, True, True),
        ]
        self.assertEqual(calls, want_calls)
        want = [
            ("Tp", "Sp"),
            ("Tk", "Sk"),
            ("Tdns", "Sdns"),
            ("Tdnsa", "Sdnsa"),
            ("i32", "Sa"),
            ("fp32", None),
            ("constexpr", "B"),
        ]
        self.assertEqual(specialization, want)
        # the options _intercept passes to the binder are JITFunction.run's
        run = inspect.getsource(JITFunction.run)
        debug = 'kwargs.get("debug", self.debug) or knobs.runtime.debug'
        self.assertIn(f'kwargs["debug"] = {debug}', run)
        mode = "knobs.compilation.instrumentation_mode"
        self.assertIn(f'kwargs["instrumentation_mode"] = {mode}', run)

    def test_symbolic_specialization_on_plain_values_is_the_native_one(self):
        spec = htl._SymbolicSpecialization(BaseBackend)
        for v in (0, 1, 16, 17, 2**31, True, 2.5, None):
            for flags in _FLAGS:
                self.assertEqual(
                    spec(BaseBackend, v, *flags),
                    native_specialize_impl(BaseBackend, v, *flags),
                )


instantiate_parametrized_tests(TestTritonLaunch)


def setUpModule():
    import torch.cuda._host_trace_capture as capture

    capture.raise_trace_disagreements = True
    torch.cuda._host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
