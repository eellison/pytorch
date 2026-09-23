# Owner(s): ["module: inductor"]
"""The tape as Inductor's cudagraph mechanism: torch.compile through the host-trace policy."""

import copy

import torch
import torch._functorch.config as aot_config
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.cudagraph_utils import active_cudagraph_policy
from torch._inductor.runtime._cudagraph.hosttrace_policy import (
    CompiledKernelOwner,
    host_trace_policy,
    HostTracePolicy,
    Installation,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


TAPE = {"triton.cudagraphs": True, "triton.cudagraph_host_trace": True}


def pointwise(x, w):
    return ((x.sin() * w + 1).relu(),)


def pointwise_graph_break(x, w):
    y = x.sin()
    torch._dynamo.graph_break()
    return ((y * w + 1).relu(),)


def in_place(x, w):
    x.add_(w)
    return (x * 2,)


class Block(torch.nn.Module):
    def __init__(self, n=64):
        super().__init__()
        self.lin = torch.nn.Linear(n, n)
        self.ln = torch.nn.LayerNorm(n)

    def forward(self, x):
        return self.ln(torch.nn.functional.gelu(self.lin(x))) + x


class TestHostTracePolicy(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        self.enterContext(torch._dynamo.config.patch(caching_precompile=False))
        self.enterContext(aot_config.patch(enable_autograd_cache=False))
        self.enterContext(config.patch(force_disable_caches=True, fx_graph_cache=False))
        counters.clear()

    def _policy(self):
        policy = HostTracePolicy()
        self.addCleanup(policy.close)
        self.enterContext(
            config.patch(cudagraph_policy=policy, **{"triton.cudagraphs": True})
        )
        return policy

    def _ready(self, policy, n=1):
        installations = policy.installations
        self.assertEqual(
            [i.status for i in installations],
            ["ready"] * n,
            [i.decline for i in installations],
        )
        return installations

    def test_config_value_selects_the_policy(self, device):
        self.assertIsNone(active_cudagraph_policy())
        with config.patch("triton.cudagraph_host_trace", True):
            self.assertIs(active_cudagraph_policy(), host_trace_policy())
            self.assertIs(copy.deepcopy(host_trace_policy()), host_trace_policy())
            explicit = HostTracePolicy()
            self.addCleanup(explicit.close)
            with config.patch(cudagraph_policy=explicit):
                self.assertIs(active_cudagraph_policy(), explicit)

    def test_dynamic_sizes_serve_from_one_trace(self, device):
        policy = self._policy()
        compiled = torch.compile(pointwise, dynamic=True)
        w = torch.randn(1, 64, device=device)
        outputs = []
        for n in (8, 8, 24, 40, 8):
            x = torch.randn(n, 64, device=device)
            (got,) = compiled(x, w)
            (want,) = pointwise(x, w)
            self.assertEqual(got, want)
            outputs.append(got)
        (installation,) = self._ready(policy)
        self.assertIsInstance(installation, Installation)
        summary = installation.summary()
        self.assertEqual(
            (summary["live"], summary["hidden"], summary["ints"]), (2, 0, 2)
        )
        self.assertEqual(
            (summary["traces"], summary["variants"], summary["misses"]), (1, 1, 0)
        )
        self.assertEqual(summary["calls"], 4)
        self.assertEqual(len({o.data_ptr() for o in outputs}), len(outputs))
        self.assertEqual(counters["inductor"]["cudagraph_host_trace_installations"], 1)

    def test_static_shapes_are_one_installation_per_artifact(self, device):
        policy = self._policy()
        compiled = torch.compile(pointwise, dynamic=False)
        w = torch.randn(1, 64, device=device)
        for n in (8, 8, 24, 24):
            x = torch.randn(n, 64, device=device)
            self.assertEqual(compiled(x, w), pointwise(x, w))
        installations = self._ready(policy, 2)
        self.assertEqual([i.summary()["calls"] for i in installations], [1, 1])

    @parametrize("graph_break", (False, True))
    def test_module_parameters_are_hidden_inputs(self, device, graph_break):
        policy = self._policy()
        module = Block().to(device).eval()
        fn = module
        if graph_break:

            def fn(x):  # noqa: E306
                y = module(x)
                torch._dynamo.graph_break()
                return module(y)

        compiled = torch.compile(fn, dynamic=True)
        with torch.no_grad():
            for n in (8, 8, 16):
                x = torch.randn(n, 64, device=device)
                self.assertEqual(compiled(x), fn(x))
        installations = self._ready(policy, 2 if graph_break else 1)
        for installation in installations:
            summary = installation.summary()
            self.assertEqual(summary["hidden"], 4)
            self.assertEqual(summary["live"], 1)
            self.assertEqual(summary["regions"], 1)
            self.assertEqual(
                len(installation._bound),
                4 + (1 if installation._family.arena is not None else 0),
            )

    def test_input_mutation_is_served(self, device):
        policy = self._policy()
        compiled = torch.compile(in_place, dynamic=True)
        w = torch.full((1, 64), 0.5, device=device)
        for n in (8, 8, 32):
            x = torch.randn(n, 64, device=device)
            y = x.clone()
            (got,) = compiled(x, w)
            (want,) = in_place(y, w)
            self.assertEqual(got, want)
            self.assertEqual(x, y)
        (installation,) = self._ready(policy)
        self.assertEqual(installation.summary()["misses"], 0)

    def test_training_step_traces_forward_and_backward(self, device):
        policy = self._policy()
        module = Block().to(device)
        reference = copy.deepcopy(module)
        compiled = torch.compile(module, dynamic=True)
        for step, n in enumerate((8, 8, 16)):
            x = torch.randn(n, 64, device=device)
            compiled(x).sum().backward()
            reference(x).sum().backward()
            for p, q in zip(module.parameters(), reference.parameters()):
                self.assertEqual(p.grad, q.grad)
                p.grad = q.grad = None
        installations = self._ready(policy, 2)
        self.assertEqual(sorted(i.is_backward for i in installations), [False, True])
        for installation in installations:
            summary = installation.summary()
            # the third step's new M is a cuBLAS key of the mm region (a harvest or a
            # topology rebuild inside the entry), not a re-trace
            self.assertEqual((summary["traces"], summary["calls"]), (1, 2), summary)
            self.assertEqual(
                summary["non_tensor_outputs"], 1 if not installation.is_backward else 2
            )

    def test_close_restores_the_artifact(self, device):
        policy = self._policy()
        compiled = torch.compile(pointwise, dynamic=True)
        w = torch.randn(1, 64, device=device)
        x = torch.randn(8, 64, device=device)
        compiled(x, w)
        (installation,) = self._ready(policy)
        artifact = installation.artifact()
        self.assertIs(artifact.current_callable, installation)
        policy.close()
        self.assertEqual(installation.status, "closed")
        self.assertIs(artifact.current_callable, installation.model)
        self.assertEqual(compiled(x, w), pointwise(x, w))

    def test_decline_by_name_serves_the_ordinary_wrapper(self, device):
        policy = self._policy()

        def fft(x):
            return (torch.fft.rfft(x).abs(),)

        compiled = torch.compile(fft, dynamic=True)
        for n in (8, 8, 16):
            x = torch.randn(n, 64, device=device)
            self.assertEqual(compiled(x), fft(x))
        (installation,) = policy.installations
        self.assertEqual(installation.status, "declined", installation.summary())
        self.assertIn("declined", installation.decline)
        self.assertEqual(len(policy.declines), 1)
        self.assertGreaterEqual(counters["inductor"]["cudagraph_skips"], 1)
        self.assertIs(installation.artifact().current_callable, installation.model)

    def test_extern_gemm_into_a_padded_buffer(self, device):
        # Inductor's mm padding widens N and comprehensive_padding pads the buffer's
        # stride: the extern mm writes into a buffer whose leading dimension exceeds its
        # N, which the region takes as eager's host does (the stride in its key)
        policy = self._policy()

        def f(x, w):
            return ((x @ w).relu(),)

        compiled = torch.compile(f, dynamic=False)
        w = torch.randn(64, 1030, device=device)
        for _ in range(2):
            x = torch.randn(64, 64, device=device)
            self.assertEqual(compiled(x, w), f(x, w))
        (installation,) = self._ready(policy)
        (region,) = installation.replay.lowered.regions
        sizes, strides = region.metas[-1][1:]
        self.assertGreater(int(strides[0]), int(sizes[1]), (sizes, strides))

    def test_a_kernel_through_tritons_own_launcher_is_recorded(self, device):
        # a kernel Inductor's static launcher bypasses (here every kernel: the static
        # launcher off; TinyLlama's mask kernel, whose name Triton truncates in its cache
        # so Inductor finds no cubin) launches through Triton's own launcher over the
        # compilation Inductor selected; the record reads that compilation's ABI through
        # CompiledKernelOwner and the tape's node holds the function Triton loaded
        policy = self._policy()
        self.enterContext(config.patch(use_static_cuda_launcher=False))
        compiled = torch.compile(pointwise, dynamic=True)
        w = torch.randn(1, 64, device=device)
        for n in (8, 8, 24):
            x = torch.randn(n, 64, device=device)
            self.assertEqual(compiled(x, w), pointwise(x, w))
        (installation,) = self._ready(policy)
        (launch,) = installation.replay.tape.launches
        self.assertIsInstance(launch["triton"].owner, CompiledKernelOwner)
        binary = launch["triton"].binary
        self.assertEqual(launch["func"], int(binary.function))
        self.assertEqual(installation.summary()["misses"], 0)

    @parametrize("fallback_random", [False, True])
    def test_dropout_traces_the_training_pair(self, device, fallback_random):
        # Inductor's own dropout draws its seeds with `aten.randint.low_out` (one
        # int64 buffer of seeds per call, the kernels then compute tl.rand(seed,
        # offset) themselves): a traced host with an rng slot since the distribution
        # template's conversion, so the forward + backward pair traces as Inductor
        # fuses it (E43: no fallback_random needed); with config.fallback_random the
        # dropout is ATen's `native_dropout` instead, a traced host with its own rng
        # slot. Either way each replay draws what the artifact's own call draws from
        # the same generator state
        policy = self._policy()
        module = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Dropout(0.1)).to(
            device
        )
        x = torch.randn(32, 64, device=device)
        self.enterContext(config.patch(fallback_random=fallback_random))
        compiled = torch.compile(module, dynamic=False)
        for _ in range(3):
            torch.manual_seed(3)
            compiled(x).sum().backward()
        installations = self._ready(policy, 2)
        self.assertEqual(sorted(i.is_backward for i in installations), [False, True])
        (forward,) = [i for i in installations if not i.is_backward]
        tape = forward.replay.tape
        kernels = " ".join(rec["kernel"] for rec in tape.launches)
        if fallback_random:
            self.assertIn("fused_dropout", kernels)
        else:
            self.assertIn("distribution_elementwise_grid_stride_kernel", kernels)
        self.assertGreaterEqual(len(tape.rng_slots), 1)
        torch.manual_seed(3)
        want = compiled(x).detach().clone()
        for installation in installations:
            installation.artifact().current_callable = installation.model
        torch.manual_seed(3)
        got = compiled(x).detach().clone()
        for installation in installations:
            installation.artifact().current_callable = installation
        self.assertEqual(got, want, atol=0, rtol=0)

    def test_an_integer_output_that_is_an_expression_of_a_size(self, device):
        # a compiled frame returning an int derived from a dynamic size (HF's
        # DynamicCache reports the cache length + 1 as its seen tokens): the artifact
        # returns a SymInt expression over the inputs' sizes, recomputed per call
        policy = self._policy()

        def fn(x, w):
            return (x * w).relu(), x.size(0) + 1

        compiled = torch.compile(fn, dynamic=True)
        w = torch.randn(1, 64, device=device)
        for n in (8, 8, 24, 40):
            x = torch.randn(n, 64, device=device)
            got, count = compiled(x, w)
            self.assertEqual(got, (x * w).relu())
            self.assertEqual(count, n + 1)
        (installation,) = self._ready(policy)
        summary = installation.summary()
        self.assertEqual(summary["non_tensor_outputs"], 1)
        self.assertEqual((summary["calls"], summary["misses"]), (3, 0))


instantiate_device_type_tests(TestHostTracePolicy, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
