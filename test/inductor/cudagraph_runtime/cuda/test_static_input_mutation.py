"""Registered-buffer mutation uses native replay without an alignment copy."""

import json
from pathlib import Path
import sys
from unittest import mock

import torch
import torch._functorch.config as aot_config
from torch._inductor import config
from torch._inductor.runtime._cudagraph import policy as terminal_policy
from torch._inductor.runtime._cudagraph._compiler.host_program import Normalize
from torch._inductor.runtime._cudagraph.api import NativeTerminalPolicy
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


class Accumulator(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.register_buffer("cache", torch.zeros(64, device=device))

    def forward(self, value):
        self.cache.add_(value)
        return self.cache, self.cache[1:], self.cache * 2


class TestStaticInputMutation(TestCase):
    def test_registered_buffer_mutation(self, device):
        if torch.version.hip:
            self.skipTest("requires CUDA parameterized graph replay")
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        self.enterContext(torch.cuda.device(device))
        self.enterContext(torch.no_grad())
        self.enterContext(torch._dynamo.config.patch(caching_precompile=False))
        self.enterContext(aot_config.patch(enable_autograd_cache=False))
        policy = NativeTerminalPolicy()
        self.addCleanup(policy.close)
        self.enterContext(config.patch(
            cudagraph_policy=policy, force_disable_caches=True, fx_graph_cache=False,
            use_static_triton_launcher=True, graph_partition=False, freezing=False,
            normalize_static_input_alignment=False,
        ))
        model, reference = Accumulator(device), Accumulator(device)
        cache_pointer = model.cache.data_ptr()
        self.assertEqual(cache_pointer % 16, 0)
        captured_static = []
        prepare = policy.prepare

        def observe_prepare(wrapper, inputs, **options):
            indices = options["static_inputs"]
            self.assertTrue(indices)
            cache_indices = tuple(index for index in indices if inputs[index] is model.cache)
            self.assertEqual(len(cache_indices), 1)
            snapshots = tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in inputs)
            variant = prepare(wrapper, inputs, **options)
            self.assertEqual(tuple(inputs), snapshots)
            copies = {event.input_index for event in variant.program.events if type(event) is Normalize}
            self.assertNotIn(cache_indices[0], copies)
            captured_static.append(cache_indices[0])
            return variant

        preparations = self.enterContext(mock.patch.object(policy, "prepare", side_effect=observe_prepare))
        compiled = torch.compile(model, fullgraph=True)
        samples = [torch.randn_like(model.cache) for _ in range(4)]
        self.assertEqual(len({value.data_ptr() for value in samples}), len(samples))
        held, aliases = [], []
        installation = None
        runtime_directory = Path(terminal_policy.__file__).parent
        for index, value in enumerate(samples):
            frames = []

            def profile(frame, event, result):
                if event == "call" and (
                    frame.f_code is installation.original.__code__
                    or Path(frame.f_code.co_filename).is_relative_to(runtime_directory)
                    or frame.f_code.co_filename.endswith("/cudagraph_boxed_replay.py")
                ):
                    frames.append((frame.f_code.co_filename, frame.f_code.co_name))

            if index:
                self.assertIs(installation.artifact.current_callable, installation.entry)
                try:
                    sys.setprofile(profile)
                    actual = compiled(value)
                finally:
                    sys.setprofile(None)
                self.assertEqual(frames, [])
            else:
                actual = compiled(value)
                self.assertEqual(policy.declines, ())
                self.assertEqual(len(policy.installations), 1)
                installation, = policy.installations
                self.assertEqual(installation.status, "ready", installation.decline)
                self.assertIn(captured_static[0], installation.artifact.mutated_input_idxs)
                self.assertNotIn(captured_static[0], installation.artifact.inputs_to_check)
            expected = reference(value)
            self.assertEqual(model.cache, reference.cache)
            self.assertEqual(actual, expected)
            self.assertIs(actual[0], model.cache)
            self.assertTrue(torch._C._is_alias_of(actual[1], model.cache))
            self.assertEqual(actual[1].data_ptr(), cache_pointer + model.cache.element_size())
            self.assertEqual(model.cache.data_ptr(), cache_pointer)
            self.assertFalse(torch._C._is_alias_of(actual[2], model.cache))
            self.assertEqual(len(installation.variants), 1)
            held.append((actual[2], expected[2]))
            aliases.append(actual[:2])
        self.assertEqual(preparations.call_count, 1)
        self.assertEqual(len({value.data_ptr() for value, _ in held}), len(held))
        policy.close()
        for value, expected in held:
            self.assertEqual(value, expected)
        for root, view in aliases:
            self.assertIs(root, model.cache)
            self.assertEqual(root, reference.cache)
            self.assertEqual(view, reference.cache[1:])
        print("STATIC_INPUT_MUTATION_RESULT=" + json.dumps({
            "static_mutated_slot": captured_static[0], "variants": 1,
            "native_hits": len(samples) - 1, "references": len(samples),
            "held_outputs_after_close": len(held), "held_alias_pairs": len(aliases),
        }, sort_keys=True), flush=True)


instantiate_device_type_tests(TestStaticInputMutation, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
