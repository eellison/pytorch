"""Default graph partition configuration admits a generated runner with no partitions."""

import json
import sys
from types import MethodType
from unittest import mock


import torch
import torch._functorch.config as aot_config
from torch._inductor.runtime._cudagraph._compiler.host_program import Normalize
from torch._inductor.runtime._cudagraph.api import NativeTerminalPolicy
from torch._inductor.runtime._cudagraph.metadata import check_terminal_attachment, MetadataDeclined
from torch._inductor import config
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


def pointwise(source):
    return (source.sin() * 2 + 1).relu()


class TestDefaultRunner(TestCase):
    def test_native_installation_without_config_override(self, device):
        self.assertTrue(config.graph_partition)
        with torch.cuda.device(device), torch._dynamo.config.patch(caching_precompile=False), \
             aot_config.patch(enable_autograd_cache=False):
            policy = NativeTerminalPolicy()
            self.addCleanup(policy.close)
            with config.patch(cudagraph_policy=policy, force_disable_caches=True, fx_graph_cache=False), \
                 mock.patch.object(policy, "prepare", wraps=policy.prepare) as preparations:
                compiled = torch.compile(pointwise, fullgraph=True, dynamic=True)
                samples = [torch.randn(size, device=device) for size in (100, 300, 37, 100)]
                actual = compiled(samples[0])
                self.assertTrue(config.graph_partition)
                self.assertEqual(policy.declines, ())
                self.assertEqual(len(policy.installations), 1)
                installation, = policy.installations
                self.assertEqual(installation.status, "ready")
                self.assertEqual(preparations.call_count, 1)
                self.assertEqual(len(installation.variants), 1)
                original = installation.original
                self.assertIs(type(original), MethodType)
                self.assertEqual(original.__self__.partitions, [])
                self.assertIs(original.__globals__["call"], original)
                self.assertIs(original.__globals__["runner"], original.__self__)
                attachment = original.__func__.__dict__["_cudagraph_terminal_attachment"]
                self.assertIs(attachment.function(), original)
                self.assertIs(check_terminal_attachment(original, attachment), installation.metadata)
                self.assertIs(installation.artifact.current_callable, installation.entry)
                self.assertEqual(len({source.data_ptr() for source in samples}), len(samples))
                held, frames = [], []

                def profile(frame, event, result):
                    if event == "call":
                        frames.append((frame.f_code.co_filename, frame.f_code.co_name))

                for index, source in enumerate(samples):
                    box = [source.numel() if kind == "integer" else source
                           for kind in installation.metadata.inputs.kinds]
                    if index:
                        try:
                            sys.setprofile(profile)
                            actual, = installation.entry(box)
                        finally:
                            sys.setprofile(None)
                        self.assertEqual(box, [])
                        self.assertEqual(frames, [])
                    ordinary_box = [source.numel() if kind == "integer" else source
                                    for kind in installation.metadata.inputs.kinds]
                    ordinary, = original(ordinary_box)
                    self.assertEqual(ordinary_box, [])
                    self.assertEqual(actual, pointwise(source))
                    self.assertEqual(actual, ordinary)
                    held.append((actual, actual.clone()))
                    self.assertEqual(len(installation.variants), 1)
                self.assertEqual(preparations.call_count, 1)
                program = installation.variants[0].program
                self.assertIs(program.origin.wrapper, original)
                self.assertEqual(len([event for event in program.events if type(event) is Normalize]), 1)
                original.__self__.partitions.append(None)
                try:
                    with self.assertRaises(MetadataDeclined):
                        check_terminal_attachment(original, attachment)
                finally:
                    original.__self__.partitions.clear()
                scope = original.__globals__
                with mock.patch.dict(scope, {"call": original.__self__.call}):
                    with self.assertRaises(MetadataDeclined):
                        check_terminal_attachment(original, attachment)
                self.assertIs(check_terminal_attachment(original, attachment), installation.metadata)
                policy.close()
                self.assertIs(installation.artifact.current_callable, original)
                for output, expected in held:
                    self.assertEqual(output, expected)
                print("DEFAULT_RUNNER_RESULT=" + json.dumps({
                    "accepted": True,

                    "graph_partition_default": config.graph_partition,
                    "actual_partitions": 0, "callable_type": "MethodType",
                    "installations": 1, "preparations": 1, "variants": 1,
                    "native_hits": 3, "python_frames": frames,
                    "sizes": [source.numel() for source in samples],
                    "ordinary_references": 4, "torch_references": 4,
                    "held_outputs_after_close": len(held),
                    "ownership_rejections": 2,
                }, sort_keys=True))


instantiate_device_type_tests(TestDefaultRunner, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
