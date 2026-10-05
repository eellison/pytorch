# Owner(s): ["module: cuda graphs"]

import copy
import functools
import os
import re
import subprocess
import sys
import types
import unittest
from unittest import mock

import sympy

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codecache import PyCodeCache
from torch._inductor.cudagraph_trees import get_manager, reset_cudagraph_trees
from torch._inductor.cudagraph_utils import BoxedDeviceIndex, CUDAGraphPolicy
from torch._inductor.output_code import maybe_handle_backward_generation
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache, run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.triton_utils import requires_cuda_and_triton


if HAS_CUDA_AND_TRITON:
    from torch._inductor.codegen.multi_kernel import SizeHintMultiKernelCall
    from torch._inductor.cudagraph_host_trace import (
        _InductorKernel,
        _Installation,
        _namespace,
        _trusted_inputs,
        HostTracePolicy,
    )
    from torch._inductor.runtime.triton_heuristics import (
        CachingAutotuner,
        DebugAutotuner,
        StaticTritonCompileResult,
        TritonCompileResult,
    )
    from torch.cuda._host_trace import Declined
    import torch.cuda._host_trace_replay as host_trace_replay
    from torch.cuda._host_trace_replay import HostTraceReplay
    from torch.cuda._host_trace_tape import EagerCall, Memset, OpaqueCall, trace
    from torch.cuda._host_trace_triton_launch import owned_module, OwnedModule


def _pointwise(x):
    return x.sin() + 1


def _reduction(x):
    return x.sum(-1)


def _mm(x):
    return x @ x


def _two(x, y):
    return x.sin() + y.cos() * 2


def _cumsum(x):
    # a split scan: its workspace is zeroed at each call
    return x.cumsum(0)


_CASES = {
    "pointwise": (_pointwise, (1024,), {}),
    "reduction": (_reduction, (16, 128), {}),
    "mm": (_mm, (64, 64), {}),
    "dynamic": (_pointwise, (1024,), {"dynamic": True}),
    "graph_partition": (_mm, (64, 64), {"graph_partition": True}),
}


@requires_cuda_and_triton
@instantiate_parametrized_tests
class TestFlagOffParity(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        reset_cudagraph_trees()
        counters.clear()

    def _code(self, case, host_trace):
        fn, shape, opts = _CASES[case]
        x = torch.randn(shape, device="cuda")
        torch._dynamo.reset()
        patch = {
            "triton.cudagraph_host_trace": host_trace,
            "graph_partition": opts.get("graph_partition", False),
        }
        with config.patch(patch):
            compiled = torch.compile(
                fn, mode="reduce-overhead", dynamic=opts.get("dynamic")
            )
            out, code = run_and_get_code(compiled, x)
        self.assertEqual(out, fn(x))
        # drop the AOT ID (a per-process counter) and kernel paths (a
        # temporary directory per compile)
        return [re.sub(r"# (AOT ID|kernel path): .*", "", c) for c in code]

    @parametrize("case", list(_CASES))
    @config.patch(fx_graph_cache=False)
    def test_codegen_is_the_same(self, case):
        with fresh_cache():
            self.assertEqual(self._code(case, False), self._code(case, True))

    def test_cache_config_is_the_same(self):
        off = config.save_config_portable()
        with config.patch({"triton.cudagraph_host_trace": True}):
            self.assertEqual(config.save_config_portable(), off)

    def test_flag_selects_the_wrapper(self):
        x = torch.randn(1024, device="cuda")
        torch.compile(_pointwise, mode="reduce-overhead")(x)
        self.assertIsNotNone(get_manager(0, create_if_none_exists=False))
        host_trace = [k for k in counters["inductor"] if "host_trace" in k]
        self.assertEqual(host_trace, [])
        torch._dynamo.reset()
        reset_cudagraph_trees()
        with config.patch({"triton.cudagraph_host_trace": True}):
            torch.compile(_pointwise, mode="reduce-overhead")(x)
        self.assertIsNone(get_manager(0, create_if_none_exists=False))

    @config.patch({"triton.cudagraph_host_trace": True})
    def test_backward_without_a_trees_manager(self):
        x = torch.randn(1024, device="cuda", requires_grad=True)
        compiled = torch.compile(_pointwise, mode="reduce-overhead")
        compiled(x).sum().backward()
        grad, x.grad = x.grad, None
        _pointwise(x).sum().backward()
        self.assertEqual(grad, x.grad)
        # a backward that is not cudagraphed does not look for a trees manager
        graph = types.SimpleNamespace(
            current_callable=_pointwise, fx_kwargs={"is_backward": True}
        )
        maybe_handle_backward_generation(graph, BoxedDeviceIndex(0))
        self.assertIs(graph.current_callable, _pointwise)

    @config.patch({"triton.cudagraph_host_trace": True})
    def test_explicit_policy_conflicts(self):
        x = torch.randn(1024, device="cuda")
        with config.patch(cudagraph_policy=CUDAGraphPolicy()):
            with self.assertRaisesRegex(Exception, "are both set"):
                torch.compile(_pointwise, mode="reduce-overhead")(x)

    def test_flag_off_imports_no_host_trace(self):
        script = """
import sys
import torch
f = torch.compile(lambda x: x.sin() + 1, mode="reduce-overhead")
x = torch.randn(1024, device="cuda")
f(x)
f(x)
names = ("torch.cuda._host_trace", "torch._inductor.cudagraph_host_trace")
print([m for m in sys.modules if m.startswith(names)])
"""
        out = subprocess.check_output(
            [sys.executable, "-c", script], env=os.environ.copy(), text=True
        )
        self.assertEqual(out.strip().splitlines()[-1], "[]")


def _traced_call(mod):
    # the generated `call` over the installation's namespace
    call = types.FunctionType(mod.call.__func__.__code__, _namespace(vars(mod)))
    return lambda *xs: call(mod.runner, list(xs))


def _is_guards_helper(v):
    return callable(v) and getattr(v, "__module__", None) == "torch._C._dynamo.guards"


@requires_cuda_and_triton
@instantiate_parametrized_tests
class TestInductorLaunches(TestCase):
    def _module(self, fn, *args, dynamic=False):
        torch._dynamo.reset()
        _, (code,) = run_and_get_code(torch.compile(fn, dynamic=dynamic), *args)
        return PyCodeCache.load(code)

    @parametrize("static_launcher", [True, False])
    @parametrize("fn,shape", [(_pointwise, (1000,)), (_reduction, (16, 128))])
    def test_launches_match_eager(self, fn, shape, static_launcher):
        with fresh_cache(), config.patch(use_static_cuda_launcher=static_launcher):
            x = torch.randn(shape, device="cuda")
            mod = self._module(fn, x)
            # the first call runs eagerly, so the fresh module's first-use
            # autotuning is outside the trace
            replay = HostTraceReplay(_traced_call(mod), trusted=_trusted_inputs([x]))
            for _ in range(3):
                x = torch.randn(shape, device="cuda")
                (out,) = replay(x)
                self.assertEqual(out, fn(x))
        self.assertEqual(replay.declines, [])
        self.assertEqual((replay.traces, replay.replays, replay.eager), (1, 1, 1))
        (launch,) = [r for _, r in replay.variants[0].tape.launches]
        # a static launcher's cubin is gone once loaded; else the tape loads its own
        kind = StaticTritonCompileResult if static_launcher else OwnedModule
        self.assertIs(type(launch.owner), kind)

    def test_dynamic_shapes_one_capture(self):
        with fresh_cache():
            mod = self._module(_pointwise, torch.randn(64, device="cuda"), dynamic=True)
            replay = HostTraceReplay(_traced_call(mod))
            for n in (1024, 4096, 2048):
                x = torch.randn(n, device="cuda")
                (out,) = replay(
                    *[x if torch.is_tensor(a) else n for a in mod.get_args()]
                )
                self.assertEqual(out, _pointwise(x))
        self.assertEqual(replay.declines, [])
        self.assertEqual((replay.traces, len(replay.variants)), (1, 1))

    def test_workspace_memset(self):
        with fresh_cache():
            x = torch.randn(100000, device="cuda")
            mod = self._module(_cumsum, x, dynamic=True)
        replay = HostTraceReplay(_traced_call(mod))
        held = set()
        # the first call runs eagerly (the fresh module's first-use
        # autotuning) and the next traces
        for n in (100000, 100000, 300000, 200000, 300000):
            x = torch.randn(n, device="cuda")
            args = [x if torch.is_tensor(a) else n for a in mod.get_args()]
            (out,) = replay(*args)
            # a split scan's lookback is not deterministic, even eagerly
            self.assertEqual(out, mod.call(list(args))[0], atol=1e-3, rtol=1e-3)
            held.update(torch._C._host_trace_held_images(v.native)[0] for v in replay.variants)
        (variant,) = replay.variants
        self.assertEqual((replay.eager, replay.declines), (1, []))
        self.assertIsInstance(variant.tape.launches[0][1], Memset)
        # the memset node is patched to each size's workspace
        self.assertEqual(len({memset[1] for memset in held}), 3)

    def test_two_inputs(self):
        x, y = (torch.randn(1000, device="cuda") for _ in range(2))
        with fresh_cache():
            mod = self._module(_two, x, y)
        self.assertIn("assert_size_stride_grouped", mod.call.__func__.__code__.co_names)
        replay = HostTraceReplay(_traced_call(mod), trusted=_trusted_inputs([x, y]))
        for _ in range(3):
            x, y = (torch.randn(1000, device="cuda") for _ in range(2))
            (out,) = replay(x, y)
            self.assertEqual(out, _two(x, y))
        self.assertEqual(replay.declines, [])
        self.assertEqual((replay.traces, replay.replays, replay.eager), (1, 1, 1))

    @config.patch(coordinate_descent_tuning=True)
    def test_coordinate_descent(self):
        x = torch.randn(16, 128, device="cuda")
        with fresh_cache():
            mod = self._module(_reduction, x)
        (autotuner,) = [
            v for v in vars(mod).values() if isinstance(v, CachingAutotuner)
        ]
        mod.call([x])
        (launcher,) = autotuner.launchers
        # a config coordinate descent moves to is compiled for its launcher
        # alone, never stored in compile_results
        autotuner.compile_results = []
        replay = HostTraceReplay(_traced_call(mod))
        for _ in range(3):
            x = torch.randn(16, 128, device="cuda")
            (out,) = replay(x)
            self.assertEqual(out, _reduction(x))
        self.assertEqual(replay.declines, [])
        (launch,) = [r for _, r in replay.variants[0].tape.launches]
        result = launcher.compile_result
        if type(result) is TritonCompileResult:
            self.assertIs(launch.owner, owned_module(result.kernel))
        else:
            self.assertIs(launch.owner, result)

    def test_declines(self):
        x = torch.randn(1000, device="cuda")
        with fresh_cache():
            mod = self._module(_pointwise, x)
        (autotuner,) = [
            v for v in vars(mod).values() if isinstance(v, CachingAutotuner)
        ]
        debug = DebugAutotuner.__new__(DebugAutotuner)
        debug.__dict__.update(autotuner.__dict__)
        untuned = CachingAutotuner.__new__(CachingAutotuner)
        untuned.__dict__.update(autotuner.__dict__, launchers=[])

        def launch(kernel, **kwargs):
            def f(x):
                y = torch.empty_like(x)
                stream = torch.cuda.current_stream().cuda_stream
                _InductorKernel(kernel).run(x, y, 1000, stream=stream, **kwargs)
                return y

            return f

        for kernel, kwargs, why in [
            (autotuner, {"extra": 1}, "keyword arguments"),
            (debug, {}, "DebugAutotuner is not traced"),
            (untuned, {}, "not tuned yet"),
        ]:
            with self.assertRaisesRegex(Declined, why) as cm:
                trace(launch(kernel, **kwargs), (x,), warm_up=False)
            self.assertEqual(cm.exception.retry, kernel is untuned)


@requires_cuda_and_triton
class TestWrapperSeams(TestCase):
    def _guards_scope(self):
        guards = torch._C._dynamo.guards
        module = {k: getattr(guards, k) for k in dir(guards)}
        module = {k: v for k, v in module.items() if _is_guards_helper(v)}
        return module, _namespace(module)

    def test_every_guards_helper_is_seamed(self):
        # the C++ helpers abort the process on a traced tensor; each one a
        # generated module binds, or might, must be a seam
        x, y = (torch.randn(64, 32, device="cuda") for _ in range(2))
        with fresh_cache():
            torch._dynamo.reset()
            _, (code,) = run_and_get_code(torch.compile(_two, dynamic=True), x, y)
        mod = PyCodeCache.load(code)
        helpers = [k for k, v in vars(mod).items() if _is_guards_helper(v)]
        self.assertIn("assert_size_stride_grouped", helpers)
        self.assertIn("copy_if_misaligned", helpers)
        scope = _namespace(vars(mod))
        self.assertEqual([k for k, v in scope.items() if _is_guards_helper(v)], [])
        module, scope = self._guards_scope()
        self.assertEqual([k for k, v in scope.items() if _is_guards_helper(v)], [])

        def call(name, *args):
            def f(x):
                scope[name](*args)
                return x

            return f

        for name, args in [
            ("_empty_strided_cpu_pinned", ((4,), (1,), torch.float32)),
            ("check_obj_id", (x, id(x))),
        ]:
            with self.assertRaisesRegex(Declined, f"{name} is not traced"):
                trace(call(name, *args), (x,), warm_up=False)

    def test_untraced_seams_are_the_originals(self):
        module, scope = self._guards_scope()
        x = torch.randn(64, device="cuda")
        args = (x, (4, 4), (1, 4), 8)
        self.assertEqual(
            scope["_reinterpret_tensor"](*args), module["_reinterpret_tensor"](*args)
        )
        out = scope["_empty_strided_cuda"]((4, 8), (1, 4), torch.float16)
        self.assertEqual(
            (out.shape, out.stride(), out.dtype), ((4, 8), (1, 4), torch.float16)
        )
        self.assertIs(scope["copy_if_misaligned"](x), x)
        misaligned = x[1:]
        copy = scope["copy_if_misaligned"](misaligned)
        self.assertEqual(copy, misaligned)
        self.assertEqual(copy.data_ptr() % 16, 0)
        with self.assertRaisesRegex(AssertionError, "expected size 64==3"):
            scope["assert_size_stride"](x, (3,), (1,))

    def test_reinterpret_tensor(self):
        _, scope = self._guards_scope()
        reinterpret = scope["_reinterpret_tensor"]
        replay = HostTraceReplay(lambda x: reinterpret(x, (4, 8), (1, 4), 3))
        base = torch.randn(100, device="cuda")
        for x in (base[2:], base[2:], base[7:]):
            out = replay(x)
            self.assertEqual(out, x.as_strided((4, 8), (1, 4), x.storage_offset() + 3))
            self.assertEqual(out.data_ptr(), x.data_ptr() + 3 * x.element_size())
        self.assertEqual((replay.traces, replay.replays, replay.eager), (1, 1, 1))

    def test_copy_if_misaligned(self):
        # the repair is a dispatch on the address; the misaligned side is a
        # variant of its own (TestInstallation.test_misaligned_input_is_a_variant)
        _, scope = self._guards_scope()
        realign = scope["copy_if_misaligned"]
        base = torch.randn(100, device="cuda")
        tape = trace(realign, (base[4:68],))
        (guard,) = tape.guards
        self.assertTrue(_alignment_guard(guard), guard)
        replay = HostTraceReplay(realign)
        for offset in (4, 1, 8, 3):
            x = base[offset : offset + 64]
            out = replay(x)
            self.assertEqual(out, x)
            self.assertEqual(out.data_ptr() % 16, 0)
            self.assertEqual(out.data_ptr() == x.data_ptr(), offset % 4 == 0)


def _alignment_guard(g):
    # copy_if_misaligned's: Eq(PythonMod(address, 16), 0)
    return isinstance(g, sympy.Eq) and g.lhs.args[1:] == (16,) and g.rhs == 0


class _Params(torch.nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.ps = torch.nn.ParameterList(torch.randn(32) for _ in range(n))

    def forward(self, x, y):
        out = y
        for p in self.ps:
            out = out + (x * p).relu()
        return out.sum(-1)


@requires_cuda_and_triton
@config.patch({"triton.cudagraph_host_trace": True})
@instantiate_parametrized_tests
class TestInstallation(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        counters.clear()
        self.installed = []
        self.calls = []
        cudagraphify = HostTracePolicy.cudagraphify

        def spy(*args, **kwargs):
            out = cudagraphify(*args, **kwargs)
            self.installed.append(out)
            self.calls.append((args, kwargs))
            return out

        patch = mock.patch.object(HostTracePolicy, "cudagraphify", spy)
        patch.start()
        self.addCleanup(patch.stop)

    @parametrize("fn,shape", [(_pointwise, (1024,)), (_reduction, (16, 128))])
    def test_dynamic_sizes_one_trace(self, fn, shape):
        compiled = torch.compile(fn, mode="reduce-overhead", dynamic=True)
        sizes = [8, 16, 33, 100, 1000, 7]
        for n in sizes:
            x = torch.randn(n, *shape[1:], device="cuda")
            self.assertEqual(compiled(x), fn(x))
        (installation,) = self.installed
        replay = installation.replay
        self.assertEqual(replay.declines, [])
        self.assertEqual(
            (replay.traces, replay.replays, replay.eager), (1, len(sizes) - 2, 1)
        )
        # no guard restates the input; only copy_if_misaligned's dispatch on
        # the live input's alignment remains
        (variant,) = replay.variants
        (guard,) = variant.tape.guards
        self.assertTrue(_alignment_guard(guard), guard)
        self.assertEqual(
            counters["inductor"]["cudagraph_host_trace_dispatch_guards"], 1
        )
        self.assertEqual(counters["inductor"]["cudagraph_host_trace_declined"], 0)

    @parametrize("memory", ["auto", "eager", "run_buffer"])
    def test_replay_memory_knob(self, memory):
        def fn(x):
            # the column means are a temporary of the run
            return (x - x.mean(0)).relu()

        with config.patch({"triton.cudagraph_host_trace_replay_memory": memory}):
            compiled = torch.compile(fn, mode="reduce-overhead")
            for _ in range(3):
                x = torch.randn(4096, 64, device="cuda")
                self.assertEqual(compiled(x), fn(x))
        (installation,) = self.installed
        replay = installation.replay
        self.assertEqual((replay.memory, replay.eager, replay.declines), (memory, 1, []))
        (variant,) = replay.variants
        self.assertTrue(any(s.temporaries for s in variant.memory.steps))
        # "auto" takes the run buffer: its peak is within the margin
        self.assertEqual(any(s.order for s in variant.memory.steps), memory == "eager")

    def test_static_inputs_are_read_at_every_call(self):
        # static_input_idxs does not promise the same tensor at every call:
        # two instances share one compile, and a `.data` swap moves a param
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(32))
                self.register_buffer("b", torch.randn(32))

            def forward(self, x):
                return (x * self.w).relu() + self.b

        m1, m2 = M().cuda(), M().cuda()
        c1, c2 = (torch.compile(m, mode="reduce-overhead") for m in (m1, m2))
        with torch.no_grad():
            for m, c in [(m1, c1), (m1, c1), (m2, c2), (m2, c2), (m1, c1)]:
                x = torch.randn(8, 32, device="cuda")
                self.assertEqual(c(x), m(x))
            m1.w.data = torch.randn(32, device="cuda")
            x = torch.randn(8, 32, device="cuda")
            self.assertEqual(c1(x), m1(x))
        (installation,) = self.installed
        replay = installation.replay
        self.assertEqual((replay.traces, replay.eager, replay.declines), (1, 1, []))
        rows = replay.variants[0].captured.lowered.program.instructions
        self.assertEqual({row[1] for row in rows if row[0] == "pointer"}, {0, 1, 2})

    def test_training_with_a_saved_activation(self):
        # the backward's saved matmul output is static under trees only
        class Lin(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(32, 32))

            def forward(self, x):
                return (x @ self.w).relu() + 1

        m = Lin().cuda()
        ref = Lin().cuda()
        ref.load_state_dict(m.state_dict())
        compiled = torch.compile(m, mode="reduce-overhead")
        opts = [torch.optim.SGD(p.parameters(), lr=0.1) for p in (m, ref)]
        for _ in range(4):
            x = torch.randn(8, 32, device="cuda")
            compiled(x).sum().backward()
            ref(x).sum().backward()
            self.assertEqual(m.w.grad, ref.w.grad)
            for opt in opts:
                opt.step()
                opt.zero_grad()
            self.assertEqual(m.w, ref.w)
        self.assertEqual(len(self.installed), 2)
        # the learner harvests the fp32 matmuls, then its tape relowered with them bound serves
        for installation in self.installed:
            replay = installation.replay
            self.assertEqual(replay.declines, [])
            self.assertEqual((replay.traces, replay.relowers, len(replay.variants), replay.eager), (1, 1, 2, 1))

    @config.patch(freezing=True)
    def test_frozen_constants_are_arguments(self):
        m = _Params().cuda()
        compiled = torch.compile(m, mode="reduce-overhead")
        with torch.no_grad():
            for _ in range(3):
                x, y = (torch.randn(8, 32, device="cuda") for _ in range(2))
                self.assertEqual(compiled(x, y), m(x, y))
        (installation,) = self.installed
        self.assertGreater(len(installation._constants), 0)
        replay = installation.replay
        self.assertEqual((replay.traces, replay.eager, replay.declines), (1, 1, []))

    def test_misaligned_input_is_a_variant(self):
        m = _Params().cuda()
        compiled = torch.compile(m, mode="reduce-overhead", dynamic=True)
        with torch.no_grad():
            for n, offset in [(8, 0), (8, 0), (50, 1), (16, 0), (9, 1)]:
                base = torch.randn(n * 64 + offset, device="cuda")
                x, y = base[offset:].view(2, n, 32)
                self.assertEqual(compiled(x, y), m(x, y))
        (installation,) = self.installed
        replay = installation.replay
        self.assertEqual((replay.traces, len(replay.variants), replay.eager), (2, 2, 1))

    def test_training_step(self):
        m = _Params().cuda()
        ref = _Params().cuda()
        ref.load_state_dict(m.state_dict())
        compiled = torch.compile(m, mode="reduce-overhead")
        for _ in range(3):
            x, y = (torch.randn(8, 32, device="cuda") for _ in range(2))
            compiled(x, y).sum().backward()
            ref(x, y).sum().backward()
        for p, q in zip(m.parameters(), ref.parameters()):
            self.assertEqual(p.grad, q.grad)
        self.assertEqual(len(self.installed), 2)
        for installation in self.installed:
            self.assertIsInstance(installation, _Installation)
            self.assertEqual(installation.replay.declines, [])
            self.assertEqual(installation.replay.traces, 1)

    @config.patch({"triton.cudagraph_host_trace_backward_frees_saved": True})
    def test_a_backward_frees_its_saved_tensors(self):
        # a backward's static inputs are its saved tensors too, whose
        # references its boxed caller hands over: it holds only the static
        # ones that are the forward's own inputs (primals), its parameters.
        # Its matmuls' sizes are its own: a harvested key outlives the test
        class Lin(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(48, 48))

            def forward(self, x):
                return (x @ self.w).relu() @ self.w

        compiled = torch.compile(Lin().cuda(), mode="reduce-overhead")
        for _ in range(3):
            compiled(torch.randn(16, 48, device="cuda")).sum().backward()
        (_, _, _, static), kwargs = self.calls[1]
        names = [p.name for p in kwargs["placeholders"]]
        held = {i for i in static if names[i].startswith("primals_")}
        self.assertEqual(len(held), 1)
        self.assertLess(len(held), len(static))
        self.assertEqual(self.installed[1].replay.freed_arguments, set(range(len(names))) - held)

    @config.patch(
        {
            "triton.cudagraph_host_trace_backward_frees_saved": False,
            "triton.cudagraph_host_trace_handback": False,
            "triton.cudagraph_host_trace_replay_splits": "peak",
        }
    )
    def test_a_backward_holding_its_static_inputs(self):
        # the switches off: a backward holds every static input to its end,
        # a slow call runs its arguments in place, and runs split only at the
        # replay's peak. Its matmuls' sizes are its own
        class Lin(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(40, 40))

            def forward(self, x):
                return (x @ self.w).relu() @ self.w

        compiled = torch.compile(Lin().cuda(), mode="reduce-overhead")
        for _ in range(3):
            compiled(torch.randn(24, 40, device="cuda")).sum().backward()
        (_, _, _, static), kwargs = self.calls[1]
        replay = self.installed[1].replay
        self.assertEqual(replay.freed_arguments, set(range(len(kwargs["placeholders"]))) - set(static))
        self.assertEqual((replay.handback, replay.splits), (False, "peak"))

    @unittest.skipIf(torch.compiler.config.force_disable_caches, "caches are disabled")
    @config.patch(fx_graph_cache=True)
    @torch._functorch.config.patch(enable_autograd_cache=True)
    def test_training_from_the_autograd_cache(self):
        # a hit hands post_compile the forward's example inputs for the backward
        w = torch.randn(64, 64, device="cuda", requires_grad=True)

        def fn(x):
            return (x @ w).relu()

        x = torch.randn(32, 64, device="cuda")
        with fresh_cache():
            for _ in range(2):
                torch._dynamo.reset()
                compiled = torch.compile(fn, mode="reduce-overhead", dynamic=True)
                for _ in range(3):
                    w.grad = None
                    compiled(x).sum().backward()
                    self.assertEqual(w.grad, torch.autograd.grad(fn(x).sum(), w)[0])
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(len(self.installed), 4)
        for installation in self.installed:
            self.assertEqual(installation.replay.declines, [])

    @unittest.skipIf(torch.compiler.config.force_disable_caches, "caches are disabled")
    @config.patch(fx_graph_cache=True)
    def test_a_cache_entry_compiled_without_the_flag(self):
        # the flag is not in the FX graph cache key, and a flag-off entry
        # holds no trusted inputs
        x = torch.randn(1024, device="cuda")
        with fresh_cache():
            with config.patch({"triton.cudagraph_host_trace": False}):
                torch.compile(_pointwise, mode="reduce-overhead")(x)
            torch._dynamo.reset()
            reset_cudagraph_trees()
            compiled = torch.compile(_pointwise, mode="reduce-overhead")
            for _ in range(3):
                self.assertEqual(compiled(x), _pointwise(x))
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        (installed,) = self.installed
        self.assertNotIsInstance(installed, _Installation)
        self.assertEqual(counters["inductor"]["cudagraph_host_trace_declined"], 1)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

    def test_int_outputs(self):
        def fn(x):
            return x.sin(), x.shape[0] * 2

        compiled = torch.compile(fn, mode="reduce-overhead", dynamic=True)
        for n in (8, 9, 8, 100):
            x = torch.randn(n, device="cuda")
            self.assertEqual(compiled(x), fn(x))
        (installation,) = self.installed
        replay = installation.replay
        self.assertEqual((replay.traces, replay.eager, replay.declines), (1, 1, []))

    def test_fallback_output_asserts_under_the_trace(self):
        # size_asserts (the default) checks a fallback op's output with
        # assert_tensor_metadata, which aborted on the trace's symbolic sizes
        def fn(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v) * 2

        compiled = torch.compile(fn, mode="reduce-overhead", dynamic=True)
        # flash: the cuDNN and efficient forwards decline for other reasons
        with config.patch(size_asserts=True), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for b in (2, 3, 4, 2):
                q, k, v = (torch.randn(b, 4, 128, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
                self.assertEqual(compiled(q, k, v), fn(q, k, v))
        (installation,) = self.installed
        replay = installation.replay
        self.assertEqual((replay.traces, replay.declines), (1, []))

    def test_declined_graph_matches_eager(self):
        def decline(self, tr, *args):
            raise tr.decline("a structural decline")

        compiled = torch.compile(_pointwise, mode="reduce-overhead", dynamic=True)
        with mock.patch.object(_InductorKernel, "_record", decline):
            for n in (8, 9, 8, 100):
                x = torch.randn(n, device="cuda")
                self.assertEqual(compiled(x), _pointwise(x))
        (installation,) = self.installed
        replay = installation.replay
        # under trust a decline is the graph's: no size retraces
        self.assertEqual((replay.traces, replay.eager), (1, 4))
        self.assertIn("a structural decline", replay.declines[0])
        self.assertEqual(counters["inductor"]["cudagraph_host_trace_declined"], 1)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

    @config.patch("triton.cudagraph_or_error", True)
    def test_decline_under_cudagraph_or_error_raises(self):
        def decline(self, tr, *args):
            raise tr.decline("a structural decline")

        compiled = torch.compile(_pointwise, mode="reduce-overhead")
        with mock.patch.object(_InductorKernel, "_record", decline):
            # the first call runs eagerly; the second traces
            compiled(torch.randn(8, device="cuda"))
            with self.assertRaisesRegex(RuntimeError, "a structural decline"):
                compiled(torch.randn(8, device="cuda"))

    def test_a_graph_without_inputs_runs_uncaptured(self):
        # HF's causal mask after a graph break: a fill from constants
        def fn():
            return torch.full((64, 64), float("-inf"), device="cuda").triu(1)

        compiled = torch.compile(fn, mode="reduce-overhead")
        for _ in range(3):
            self.assertEqual(compiled(), fn())
        (installed,) = self.installed
        self.assertNotIsInstance(installed, _Installation)
        self.assertEqual(counters["inductor"]["cudagraph_host_trace_declined"], 0)

    def test_a_gemm_only_graph_is_captured(self):
        # an accepted cuBLAS call binds at a replay: the graph is not all eager
        compiled = torch.compile(_mm, mode="reduce-overhead")
        x = torch.randn(64, 64, device="cuda")
        for _ in range(3):
            self.assertEqual(compiled(x), _mm(x))
        (installation,) = self.installed
        replay = installation.replay
        self.assertTrue(replay.variants)
        self.assertEqual((replay.uncaptured, replay.declines), (0, []))

    @config.patch("triton.cudagraph_host_trace_harvest", ("attention", "conv", "rng"))
    def test_an_all_eager_graph_is_not_a_decline(self):
        compiled = torch.compile(_mm, mode="reduce-overhead")
        x = torch.randn(64, 64, device="cuda")
        for _ in range(3):
            self.assertEqual(compiled(x), _mm(x))
        (installation,) = self.installed
        replay = installation.replay
        self.assertEqual((replay.variants, replay.declines), ([], []))
        self.assertGreater(replay.uncaptured, 0)
        self.assertEqual(counters["inductor"]["cudagraph_host_trace_declined"], 0)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

    def test_empty_input_is_its_own_variant(self):
        compiled = torch.compile(_pointwise, mode="reduce-overhead")
        for n, offset in [(64, 0), (0, 0), (64, 0), (64, 1), (0, 0), (128, 1)]:
            x = torch.randn(n + offset, device="cuda")[offset:]
            torch._dynamo.decorators.mark_unbacked(x, 0)
            self.assertEqual(compiled(x), _pointwise(x))
        (installation,) = self.installed
        replay = installation.replay
        # a size's zero-ness is a guard: the empty call and the misaligned call each trace
        self.assertEqual((replay.traces, len(replay.variants), replay.eager), (3, 3, 1))

    def test_capture_error_declines_only_the_call(self):
        capture = host_trace_replay.capture_tape

        def oom_once(*args, **kwargs):
            if not oom_once.raised:
                oom_once.raised = True
                raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
            return capture(*args, **kwargs)

        oom_once.raised = False
        compiled = torch.compile(_pointwise, mode="reduce-overhead", dynamic=True)
        with mock.patch.object(host_trace_replay, "capture_tape", oom_once):
            for n in (8, 16, 8, 100):
                x = torch.randn(n, device="cuda")
                self.assertEqual(compiled(x), _pointwise(x))
        (installation,) = self.installed
        replay = installation.replay
        self.assertEqual((replay.traces, len(replay.variants), replay.eager), (2, 1, 2))
        self.assertIn("OutOfMemoryError", replay.declines[0])

    def test_workspace_is_a_memset(self):
        compiled = torch.compile(_cumsum, mode="reduce-overhead", dynamic=True)
        for n in (100000, 200000, 300000, 100000):
            x = torch.randn(n, device="cuda")
            self.assertEqual(compiled(x), _cumsum(x), atol=1e-3, rtol=1e-3)
        (installation,) = self.installed
        replay = installation.replay
        self.assertEqual((replay.traces, replay.eager, replay.declines), (1, 1, []))
        (variant,) = replay.variants
        self.assertIsInstance(variant.tape.launches[0][1], Memset)

    @config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
            "multi_kernel_hints": [64, 4096],
            "coordinate_descent_tuning": False,
        }
    )
    def test_size_dispatch_selects_the_variant(self):
        def fn(x, y):
            return x @ y

        compiled = torch.compile(fn, mode="reduce-overhead")
        y = torch.randn(256, 256, device="cuda")
        x = torch.randn(64, 256, device="cuda")
        torch._dynamo.mark_dynamic(x, 0)
        _, (code,) = run_and_get_code(compiled, x, y)
        self.assertIn("size_hint_multi_kernel", code)
        (call,) = [
            v
            for v in vars(PyCodeCache.load(code)).values()
            if isinstance(v, SizeHintMultiKernelCall)
        ]
        # the kernel tuned at the smallest hint, and at the largest
        sizes = [sum(d for shape in hint for d in shape) for hint in call._kernel_hints]
        small, large = (
            call.kernels[sizes.index(f(sizes))].inductor_meta["kernel_name"]
            for f in (min, max)
        )
        (installation,) = self.installed
        replay = installation.replay
        # each size is guarded: a new size traces, a seen one replays
        for n, name in [(64, small), (4000, large), (64, small), (4000, large)]:
            traces = replay.traces
            x = torch.randn(n, 256, device="cuda")
            self.assertEqual(compiled(x, y), fn(x, y), atol=1e-2, rtol=1e-2)
            if replay.traces > traces:
                (launch,) = [r for _, r in replay.variants[-1].tape.launches]
                self.assertEqual(launch.name, name)
        self.assertEqual(replay.declines, [])
        self.assertEqual((replay.traces, len(replay.variants), replay.eager), (2, 2, 1))


class _Block(torch.nn.Module):
    # an HF-style attention block: pre-norm, fused QKV, SDPA, residual
    def __init__(self, d=256, heads=4, p=0.0):
        super().__init__()
        self.heads, self.p = heads, p
        self.ln = torch.nn.LayerNorm(d, dtype=torch.bfloat16)
        self.qkv = torch.nn.Linear(d, 3 * d, dtype=torch.bfloat16)
        self.o = torch.nn.Linear(d, d, dtype=torch.bfloat16)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(self.ln(x)).view(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        p = self.p if self.training else 0.0
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        return x + self.o(y.transpose(1, 2).reshape(b, n, d))


_BACKENDS = ("CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION")


class _ConvNet(torch.nn.Module):
    # a MobileNet-style stem: strided conv-bn-relu, depthwise, a 1x1 with a
    # bias, a dilated conv
    def __init__(self):
        super().__init__()
        conv = functools.partial(torch.nn.Conv2d, dtype=torch.bfloat16)
        self.stem = conv(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(32, dtype=torch.bfloat16)
        self.dw = conv(32, 32, 3, padding=1, groups=32, bias=False)
        self.pw = conv(32, 64, 1)
        self.dilated = conv(64, 64, 3, padding=2, dilation=2)

    def forward(self, x):
        x = self.dw(self.bn(self.stem(x)).relu())
        return self.dilated(self.pw(x).relu()).mean((2, 3))


@requires_cuda_and_triton
@config.patch({"triton.cudagraph_host_trace": True})
@instantiate_parametrized_tests
class TestInstalledAttentionAndRng(TestCase):
    """Attention, convolutions and RNG replayed inside the graph, bitwise
    against the generated code run eagerly from the same generator offset."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.installed = []
        self.calls = []
        cudagraphify = HostTracePolicy.cudagraphify
        call = _Installation.__call__

        def spy(*args, **kwargs):
            out = cudagraphify(*args, **kwargs)
            self.installed.append(out)
            return out

        def spy_call(installation, new_inputs):
            args = (*new_inputs, *installation._constants)
            gen = torch.cuda.default_generators[torch.cuda.current_device()]
            start = gen.get_offset()
            out = call(installation, new_inputs)
            outs = [o.clone() if isinstance(o, torch.Tensor) else o for o in out]
            self.calls.append((installation, args, outs, start, gen.get_offset()))
            return out

        for target, name, fn in ((HostTracePolicy, "cudagraphify", spy), (_Installation, "__call__", spy_call)):
            patch = mock.patch.object(target, name, fn)
            patch.start()
            self.addCleanup(patch.stop)

    def _run(self, fn, make, calls=8, warm_up=3):
        compiled = torch.compile(fn, mode="reduce-overhead")
        with torch.no_grad():
            for i in range(calls):
                compiled(make())
                if i == warm_up - 1:
                    (installation,) = self.installed
                    replay = installation.replay
                    eager, replays = replay.eager, replay.replays
        self.assertEqual(replay.declines, [])
        # no eager step after the warm-up: every call replays the graph
        self.assertEqual((replay.eager - eager, replay.replays - replays), (0, calls - warm_up))
        # the newest variant, traced once every key bound
        records = [r for _, r in replay.variants[0].tape.launches]
        self.assertEqual([r.name for r in records if isinstance(r, (EagerCall, OpaqueCall))], [])
        self.assertEqual(self._check(self.calls[warm_up:]), calls - warm_up)
        return compiled

    def _check(self, calls, floats_only=False):
        # each call against the generated code run eagerly from its
        # generator offset, which it leaves where the call did
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        for installation, args, outs, start, end in calls:
            gen.set_offset(start)
            with torch.no_grad():
                ref = installation.replay.fn(*args)
            self.assertEqual(gen.get_offset(), end)
            for a, b in zip(outs, ref, strict=True):
                if floats_only and not a.is_floating_point():
                    continue
                self.assertTrue(torch.equal(a.flatten().view(torch.uint8), b.flatten().view(torch.uint8)))
        return len(calls)

    @parametrize("backend", _BACKENDS)
    def test_attention_block(self, backend):
        from torch.nn.attention import sdpa_kernel, SDPBackend

        m = _Block().cuda().eval()
        with sdpa_kernel(getattr(SDPBackend, backend)):
            self._run(m, lambda: torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16))

    def test_cudnn_attention_training(self):
        from torch.nn.attention import sdpa_kernel, SDPBackend

        m = _Block().cuda()
        compiled = torch.compile(m, mode="reduce-overhead")
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            for i in range(8):
                x = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                compiled(x).float().sum().backward()
                if i == 3:
                    eager = [r.replay.eager for r in self.installed]
        # the forward and the backward (cuDNN's backward kernels) replay
        self.assertEqual(len(self.installed), 2)
        self.assertEqual([r.replay.eager for r in self.installed], eager)
        for installation in self.installed:
            replay = installation.replay
            self.assertEqual(replay.declines, [])
            records = [r for _, r in replay.variants[0].tape.launches]
            self.assertEqual([r.name for r in records if isinstance(r, (EagerCall, OpaqueCall))], [])
        # without dropout cuDNN's saved philox seed and offset are torch.empty
        self.assertEqual(self._check(self.calls[-8:], floats_only=True), 8)

    @parametrize("backend", _BACKENDS)
    def test_attention_dropout(self, backend):
        from torch.nn.attention import sdpa_kernel, SDPBackend

        # cuDNN's dropout resolution is 1/16
        m = _Block(p=0.125).cuda().train()
        x = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16)
        with sdpa_kernel(getattr(SDPBackend, backend)):
            compiled = self._run(m, lambda: x)
            # replays from one seed repeat themselves (and eager, _check)
            outs = []
            for _ in range(2):
                torch.manual_seed(0)
                with torch.no_grad():
                    outs.append([compiled(x).clone() for _ in range(5)])
        self.assertEqual(outs[0], outs[1], atol=0, rtol=0)
        self.assertNotEqual(outs[0][0], outs[0][1])
        self.assertEqual(self._check(self.calls[-10:]), 10)

    # size_asserts: a dynamic graph's fallback aborts
    @config.patch(size_asserts=False)
    def test_attention_dropout_new_key(self):
        # a sequence length first seen after capture: its RNG binding is a
        # row of the captured graph's site, bitwise against the generated
        # code from each call's generator offset
        from torch.nn.attention import sdpa_kernel, SDPBackend

        m = _Block(p=0.125).cuda().train()
        xs = [torch.randn(2, n, 256, device="cuda", dtype=torch.bfloat16) for n in (128, 256)]
        compiled = torch.compile(m, mode="reduce-overhead", dynamic=True)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.no_grad():
            for x in [xs[0]] * 4 + [xs[1]] * 4 + xs * 3:
                compiled(x)
        (installation,) = self.installed
        replay = installation.replay
        self.assertEqual(replay.declines, [])
        served = [v for v in replay.variants if not v.captured.lowered.opaque]
        self.assertEqual(len(served), 1)
        self.assertGreaterEqual(replay.replays, 8)
        self.assertEqual(self._check(self.calls[-6:]), 6)

    # size_asserts: a dynamic graph's fallback aborts
    @config.patch(size_asserts=False)
    def test_attention_dropout_new_shapes(self):
        # training steps at sequence lengths first seen after capture, with an
        # eager draw between two compiled blocks: no trace after the first
        # length, and from one seed the grads, the generator state and its
        # offset equal the same compile without graphs; a width no other test
        # uses, as a GEMM key another test harvested changes what traces
        from torch.nn.attention import sdpa_kernel, SDPBackend

        blocks = {k: _Block(d=320, heads=5, p=0.125) for k in "ab"}
        m = torch.nn.ModuleDict(blocks).cuda().train()
        ref = copy.deepcopy(m)
        graphs = [torch.compile(m[k], mode="reduce-overhead", dynamic=True) for k in "ab"]
        plain = [torch.compile(ref[k], dynamic=True) for k in "ab"]
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        xs = [torch.randn(2, n, 320, device="cuda", dtype=torch.bfloat16) for n in (128,) * 4 + (256, 192, 64, 256, 128)]

        def step(model, a, b, x):
            model.zero_grad(set_to_none=True)
            h = a(x)
            b(h * torch.rand_like(h)).float().sum().backward()
            return [w.grad.clone() for w in model.parameters()]

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            torch.cuda.manual_seed(0)
            got = [step(m, *graphs, x) for x in xs[:4]]
            traces = [i.replay.traces for i in self.installed]
            got += [step(m, *graphs, x) for x in xs[4:]]
            state, offset = torch.cuda.get_rng_state(), gen.get_offset()
            torch.cuda.manual_seed(0)
            want = [step(ref, *plain, x) for x in xs]
        self.assertEqual(got, want, atol=0, rtol=0)
        self.assertEqual((state, offset), (torch.cuda.get_rng_state(), gen.get_offset()))
        self.assertEqual(len(self.installed), 4)
        self.assertEqual([i.replay.traces for i in self.installed], traces)
        self.assertEqual([i.replay.declines for i in self.installed], [[]] * 4)

    def test_attention_training_from_host_seeds(self):
        # a block called four times a step, its second graph three: that
        # forward's first call runs eagerly and returns eager's host seed and
        # offset, its replays the device's; its backward, traced at a device
        # seed, takes the host one too. Grads bitwise against the same
        # compile without graphs from the same generator offset
        from torch.nn.attention import sdpa_kernel, SDPBackend

        m = _Block(p=0.125).cuda().train()
        ref = copy.deepcopy(m)
        compiled, plain = torch.compile(m, mode="reduce-overhead"), torch.compile(ref)
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        x = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16)

        def step(model, f):
            model.zero_grad(set_to_none=True)
            f(f(f(f(x)))).float().sum().backward()
            return [w.grad.clone() for w in model.parameters()]

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            for _ in range(4):
                start = gen.get_offset()
                grads = step(m, compiled)
                gen.set_offset(start)
                self.assertEqual(grads, step(ref, plain), atol=0, rtol=0)
        self.assertEqual(len(self.installed), 4)
        self.assertEqual([i.replay.declines for i in self.installed], [[]] * 4)
        # each graph's first call runs eagerly, its second traces, the rest replay
        self.assertEqual([(i.replay.eager, i.replay.traces) for i in self.installed], [(1, 1)] * 4)

    # size_asserts: a dynamic graph's fallback aborts
    @config.patch(size_asserts=False)
    @parametrize("backend", _BACKENDS[1:])
    @parametrize("p", [0.0, 0.125])
    def test_attention_training(self, backend, p):
        # the forward and the attention backward replay with no eager step
        # once each batch size is warm; grads bitwise against the same
        # compile without graphs from the same generator offset
        from torch.nn.attention import sdpa_kernel, SDPBackend

        m = _Block(p=p).cuda().train()
        compiled = torch.compile(m, mode="reduce-overhead", dynamic=True)
        expected = torch.compile(copy.deepcopy(m), dynamic=True)
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        xs = [torch.randn(b, 128, 256, device="cuda", dtype=torch.bfloat16) for b in (2, 3, 4)]

        def step(model, x):
            model.zero_grad(set_to_none=True)
            model(x).float().sum().backward()
            return [w.grad.clone() for w in model.parameters()]

        with sdpa_kernel(getattr(SDPBackend, backend)):
            for x in xs * 3:
                step(compiled, x)
            counts = [(r.replay.eager, r.replay.replays) for r in self.installed]
            for x in xs:
                start = gen.get_offset()
                grads = step(compiled, x)
                gen.set_offset(start)
                self.assertEqual(grads, step(expected, x), atol=0, rtol=0)
        self.assertEqual(len(self.installed), 2)
        for installation, (eager, replays) in zip(self.installed, counts):
            replay = installation.replay
            self.assertEqual(replay.declines, [])
            self.assertEqual((replay.eager - eager, replay.replays - replays), (0, len(xs)))
            records = [r for _, r in replay.variants[0].tape.launches]
            self.assertEqual([r.name for r in records if isinstance(r, (EagerCall, OpaqueCall))], [])
        # the attention backward's kernels are the backward graph's launches
        self.assertTrue(any("attention_backward" in r.name for r in records))

    # size_asserts: a dynamic graph's fallback aborts
    @config.patch(size_asserts=False)
    def test_checkpointed_dropout(self):
        # AOT hands checkpointed RNG its own generators as graph inputs
        # (graphsafe RNG), seeded from the CPU generator at the first call:
        # from one seed, the steps bitwise against the same compile without
        # graphs, with no eager step once each batch size is warm
        from torch.utils.checkpoint import checkpoint

        m = torch.nn.Sequential(
            torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(512, 256)
        ).to("cuda", torch.bfloat16)
        models = (m, copy.deepcopy(m))
        fns = [functools.partial(lambda m, x: checkpoint(m, x, use_reentrant=False).square().mean(), w) for w in models]
        compiled = torch.compile(fns[0], mode="reduce-overhead", dynamic=True)
        expected = torch.compile(fns[1], dynamic=True)
        xs = [torch.randn(b, 256, device="cuda", dtype=torch.bfloat16) for b in (32, 48, 64)]

        def step(fn, model, x):
            model.zero_grad(set_to_none=True)
            loss = fn(x)
            loss.backward()
            return [loss.detach(), *(w.grad for w in model.parameters())]

        runs = []
        for fn, model in ((compiled, models[0]), (expected, models[1])):
            torch.manual_seed(0)
            runs.append([step(fn, model, x) for x in xs * 3])
            if fn is compiled:
                counts = [(r.replay.eager, r.replay.replays) for r in self.installed]
            runs[-1] += [step(fn, model, x) for x in xs]
        # the bias grads' reductions may round differently,
        # under trees as well
        for (loss, *grads), (want, *expected) in zip(*runs):
            self.assertEqual([loss, *grads[::2]], [want, *expected[::2]], atol=0, rtol=0)
            self.assertEqual(grads[1::2], expected[1::2])
        self.assertEqual(len(self.installed), 2)
        for installation, (eager, replays) in zip(self.installed, counts):
            replay = installation.replay
            self.assertEqual(replay.declines, [])
            self.assertEqual((replay.eager - eager, replay.replays - replays), (0, len(xs)))
            records = [r for _, r in replay.variants[0].tape.launches]
            self.assertEqual([r.name for r in records if isinstance(r, (EagerCall, OpaqueCall))], [])
            self.assertTrue(any(getattr(r, "generator", None) is not None for r in records))

    def test_mix_order_reduction_training(self):
        # LayerNorm's backward over at least 5 Mi elements fuses its row and
        # column reductions (a mix-order reduction), whose column partials the
        # wrapper sums with ATen's sum.dim_IntList: harvested, no eager step
        m = torch.nn.LayerNorm(768, device="cuda")
        compiled = torch.compile(m, mode="reduce-overhead")
        expected = torch.compile(copy.deepcopy(m))
        x = torch.randn(8192, 768, device="cuda", requires_grad=True)

        def step(model):
            model.zero_grad(set_to_none=True)
            x.grad = None
            model(x).square().sum().backward()
            return [x.grad.clone(), *(w.grad.clone() for w in model.parameters())]

        with fresh_cache():
            _, (_, backward) = run_and_get_code(step, compiled)
            self.assertIn(".sum(dim=0)", backward)
            step(compiled)
            step(compiled)
            counts = [(r.replay.eager, r.replay.replays) for r in self.installed]
            self.assertEqual(step(compiled), step(expected), atol=0, rtol=0)
        self.assertEqual(len(self.installed), 2)
        for installation, (eager, replays) in zip(self.installed, counts):
            replay = installation.replay
            self.assertEqual(replay.declines, [])
            self.assertEqual((replay.eager - eager, replay.replays - replays), (0, 1))
            records = [r for _, r in replay.variants[0].tape.launches]
            self.assertEqual([r.name for r in records if isinstance(r, (EagerCall, OpaqueCall))], [])

    def test_conv_net(self):
        # channels_last (Inductor's layout optimization) cuDNN convolutions
        m = _ConvNet().cuda().eval()
        self._run(m, lambda: torch.randn(8, 3, 64, 64, device="cuda", dtype=torch.bfloat16))

    # size_asserts: a dynamic graph's fallback aborts
    @config.patch(size_asserts=False)
    def test_conv_net_training(self):
        # the forward and the convolution backwards replay with no eager step
        # once each batch size is warm; grads bitwise against the same compile
        # without graphs (cuDNN deterministic)
        m = _ConvNet().cuda().train()
        compiled = torch.compile(m, mode="reduce-overhead", dynamic=True)
        expected = torch.compile(copy.deepcopy(m), dynamic=True)
        xs = [torch.randn(b, 3, 64, 64, device="cuda", dtype=torch.bfloat16) for b in (4, 6, 8)]

        def step(model, x):
            model.zero_grad(set_to_none=True)
            model(x).float().sum().backward()
            return [w.grad.clone() for w in model.parameters()]

        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True):
            for x in xs * 3:
                step(compiled, x)
            counts = [(r.replay.eager, r.replay.replays) for r in self.installed]
            for x in xs:
                self.assertEqual(step(compiled, x), step(expected, x), atol=0, rtol=0)
        self.assertEqual(len(self.installed), 2)
        for installation, (eager, replays) in zip(self.installed, counts):
            replay = installation.replay
            self.assertEqual(replay.declines, [])
            self.assertEqual((replay.eager - eager, replay.replays - replays), (0, len(xs)))
            records = [r for _, r in replay.variants[0].tape.launches]
            self.assertEqual([r.name for r in records if isinstance(r, (EagerCall, OpaqueCall))], [])

    @parametrize("case", ["dropout", "rand"])
    def test_rng(self, case):
        fns = {
            "dropout": lambda x: torch.nn.functional.dropout(x, 0.1, True) * 2,
            "rand": lambda x: x + torch.rand_like(x),
        }
        self._run(fns[case], lambda: torch.randn(4096, device="cuda"))

    def _host_steps(self):
        records = [r for i in self.installed for _, r in i.replay.variants[0].tape.launches]
        return [r for r in records if isinstance(r, EagerCall) and r.host]

    @parametrize("fallback_random", [False, True])
    def test_cpu_rand_is_a_host_step(self, fallback_random):
        # HF LayerDrop: a CPU draw every call, which the replay makes in order
        def fn(x):
            return x.sin() * torch.rand([]).to(x.device)

        x = torch.randn(1024, device="cuda")
        with config.patch(fallback_random=fallback_random):
            graphs, plain = torch.compile(fn, mode="reduce-overhead"), torch.compile(fn)
            for i in range(6):
                torch.manual_seed(i)
                got = graphs(x)
                state = torch.get_rng_state()
                torch.manual_seed(i)
                self.assertEqual(got, plain(x), atol=0, rtol=0)
                self.assertEqual(state, torch.get_rng_state())
        (installation,) = self.installed
        self.assertEqual(installation.replay.declines, [])
        self.assertEqual(installation.replay.eager, 1)
        self.assertGreater(installation.replay.replays, 0)
        self.assertTrue(self._host_steps())

    def test_cpu_constant_feeds_the_device(self):
        # XLNet's positional embedding: a C++ kernel into a CPU buffer, copied over
        def fn(x):
            pos = torch.arange(x.shape[0], dtype=torch.float32)[:, None]
            freq = torch.arange(x.shape[1], dtype=torch.float32)[None, :] / 256
            return (x * (pos @ freq).sin().to(x.device)).sum()

        compiled = torch.compile(fn, mode="reduce-overhead")
        for _ in range(4):
            x = torch.randn(64, 256, device="cuda", requires_grad=True)
            compiled(x).backward()
            y = x.detach().requires_grad_()
            fn(y).backward()
            self.assertEqual(x.grad, y.grad)
        for installation in self.installed:
            self.assertEqual(installation.replay.declines, [])
        self.assertTrue(self._host_steps())

    def test_fft_forward_and_backward(self):
        # GoogleFnet's mixing: cuFFT over complex views of float buffers
        def fn(x):
            z = torch.view_as_complex((x * 2).unflatten(-1, (-1, 2)))
            return torch.view_as_real(torch.fft.fft(torch.fft.fft(z, dim=-1), dim=-2)).sin()

        graphs, plain = torch.compile(fn, mode="reduce-overhead"), torch.compile(fn)
        for _ in range(5):
            x = torch.randn(8, 16, 64, device="cuda", requires_grad=True)
            y = x.detach().requires_grad_()
            got, want = graphs(x), plain(y)
            got.sum().backward()
            want.sum().backward()
            self.assertEqual(got, want, atol=0, rtol=0)
            self.assertEqual(x.grad, y.grad, atol=0, rtol=0)
        self.assertEqual(len(self.installed), 2)
        for installation in self.installed:
            self.assertEqual(installation.replay.declines, [])
            self.assertGreater(installation.replay.replays, 0)


def setUpModule():
    import torch.cuda._host_trace_capture as capture

    capture.raise_trace_disagreements = True
    torch.cuda._host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
