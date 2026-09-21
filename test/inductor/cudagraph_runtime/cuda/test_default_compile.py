"""Default torch.compile configuration can share one native runtime policy."""

import copy
from unittest import mock

import torch
import torch._functorch.config as aot_config
from torch._inductor import config
from torch._inductor.runtime._cudagraph.api import NativeTerminalPolicy
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def pointwise(source):
    return (source.sin() * 2 + 1).relu()


def pointwise_graph_break(source):
    value = source.sin()
    torch._dynamo.graph_break()
    return (value * 2 + 1).relu()


class TestDefaultCompile(TestCase):
    @parametrize("graph_break", (False, True))
    def test_default_compile_with_native_policy(self, device, graph_break):
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        self.enterContext(torch.cuda.device(device))
        self.enterContext(torch._dynamo.config.patch(caching_precompile=False))
        self.enterContext(aot_config.patch(enable_autograd_cache=False))
        policy = NativeTerminalPolicy()
        self.addCleanup(policy.close)
        self.enterContext(config.patch(cudagraph_policy=policy, force_disable_caches=True, fx_graph_cache=False))
        preparations = self.enterContext(mock.patch.object(policy, "prepare", wraps=policy.prepare))
        compiled = torch.compile(pointwise_graph_break if graph_break else pointwise, dynamic=True)
        samples = [torch.randn(size, device=device) for size in (100, 300, 37, 100)]
        self.assertEqual(len({source.data_ptr() for source in samples}), len(samples))
        held = []
        for source in samples:
            actual = compiled(source)
            expected = pointwise(source)
            self.assertEqual(actual, expected)
            held.append((actual, expected))
            self.assertEqual(policy.declines, ())
            self.assertEqual(len(policy.installations), 2 if graph_break else 1)
            self.assertEqual(preparations.call_count, len(policy.installations))
            self.assertIs(copy.deepcopy(policy), policy)
            self.assertIs(config.get_config_copy()["cudagraph_policy"], policy)
            for installation in policy.installations:
                self.assertEqual(installation.status, "ready")
                self.assertEqual(len(installation.variants), 1)
                self.assertIs(installation.artifact.current_callable, installation.entry)
        self.assertEqual(len({actual.data_ptr() for actual, _ in held}), len(held))
        installations = policy.installations
        policy.close()
        for installation in installations:
            self.assertIs(installation.artifact.current_callable, installation.original)
        for actual, expected in held:
            self.assertEqual(actual, expected)


instantiate_device_type_tests(TestDefaultCompile, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
