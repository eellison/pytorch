"""Ordinary input mutation composes with alignment copies and native replay."""

import json
import sys
from unittest import mock

import torch
import torch._functorch.config as aot_config
from torch._inductor import config
from torch._inductor.runtime._cudagraph import policy as terminal_policy
from torch._inductor.runtime._cudagraph.api import NativeTerminalPolicy
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


def mutate(source, value):
    source.add_(value)
    return source, source[1:], source * 2


class TestInputMutation(TestCase):
    def test_alignment_copyback_and_native_reuse(self, device):
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
            use_static_triton_launcher=True, graph_partition=False,
        ))
        preparations = self.enterContext(mock.patch.object(policy, "prepare", wraps=policy.prepare))
        captures = self.enterContext(mock.patch.object(
            terminal_policy, "prepare_terminal", wraps=terminal_policy.prepare_terminal,
        ))
        compiled = torch.compile(mutate)
        samples = []
        for offset in (0, 4, 1, 2, 0):
            storage = torch.randn(64 + offset + 1, device=device)
            source = storage[offset:offset + 64]
            value = torch.randn_like(source)
            self.assertEqual(source.storage_offset(), offset)
            self.assertEqual(source.data_ptr(), storage.data_ptr() + offset * source.element_size())
            samples.append((storage, source, value))
        self.assertEqual(len({source.data_ptr() for _, source, _ in samples}), len(samples))
        installation = None
        held, aliases, body_calls = [], [], []
        for index, (storage, source, value) in enumerate(samples):
            offset = source.storage_offset()
            expected_storage = storage.clone()
            expected_source = expected_storage[offset:offset + source.numel()]
            expected = mutate(expected_source, value)
            calls = []
            preparation_count = preparations.call_count

            def profile(frame, event, result):
                if (event == "call" and frame.f_code is installation.original.__code__
                        and frame.f_globals is installation.original.__globals__):
                    calls.append(frame.f_code.co_name)

            if index:
                try:
                    sys.setprofile(profile)
                    actual = compiled(source, value)
                finally:
                    sys.setprofile(None)
            else:
                actual = compiled(source, value)
                self.assertEqual(tuple(reason for _, reason in policy.declines), ())
                self.assertEqual(len(policy.installations), 1)
                installation, = policy.installations
                self.assertEqual(installation.status, "ready", installation.decline)
                self.assertIsNot(installation.ordinary, installation.original)
                self.assertEqual(installation._static_inputs, ())
                self.assertTrue(installation.artifact.mutated_input_idxs)
                self.assertTrue(set(installation.artifact.mutated_input_idxs).intersection(
                    installation.artifact.inputs_to_check,
                ))
            self.assertEqual(storage, expected_storage)
            self.assertEqual(actual, expected)
            self.assertIs(actual[0], source)
            self.assertTrue(torch._C._is_alias_of(actual[1], source))
            self.assertEqual(actual[1].storage_offset(), offset + 1)
            self.assertEqual(actual[1].data_ptr(), source.data_ptr() + source.element_size())
            self.assertFalse(torch._C._is_alias_of(actual[2], source))
            self.assertEqual(len(policy.installations), 1)
            self.assertEqual(len(installation.variants), 1)
            self.assertEqual(captures.call_count, 1)
            if index:
                aligned = source.data_ptr() % 16 == 0
                self.assertEqual(len(calls), 0 if aligned else 1)
                if aligned:
                    self.assertEqual(preparations.call_count, preparation_count)
            body_calls.append(len(calls))
            held.append((actual[2], expected[2]))
            aliases.append((actual[0], actual[1], source, expected_source))
        self.assertEqual(len({actual.data_ptr() for actual, _ in held}), len(held))
        policy.close()
        self.assertIs(installation.artifact.current_callable, installation.ordinary)
        for actual, expected in held:
            self.assertEqual(actual, expected)
        for root, view, source, expected in aliases:
            self.assertIs(root, source)
            self.assertEqual(root, expected)
            self.assertEqual(view, expected[1:])
        print("INPUT_MUTATION_RESULT=" + json.dumps({
            "samples": len(samples), "variants": 1, "prepare_attempts": preparations.call_count,
            "native_captures": captures.call_count, "native_hits": 2, "ordinary_copyback_calls": 2,
            "input_element_offsets": [source.storage_offset() for _, source, _ in samples],
            "generated_body_calls": body_calls, "references": len(samples),
            "held_outputs_after_close": len(held), "held_alias_pairs": len(aliases),
        }, sort_keys=True), flush=True)


instantiate_device_type_tests(TestInputMutation, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
