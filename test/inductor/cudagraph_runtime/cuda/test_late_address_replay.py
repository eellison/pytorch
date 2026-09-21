"""Store integer address values without dereferencing them as pointers."""

import json
import struct
import sys
from unittest import mock


import sympy
import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import DirectHost, DirectTriton, InputContract, IntegerRange, TensorInput
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
from torch._inductor.runtime._cudagraph.guard_export import export_guards, GuardExportDeclined
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, InputSource, ParameterSource, PointerSource
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._sympy.value_ranges import ValueRanges


@triton.jit(do_not_specialize=["owned_bits", "count"],
            do_not_specialize_on_alignment=["owned_bits", "count"])
def store_words(output, input_address, owned_bits, count):
    tl.store(output, input_address.to(tl.int64))
    tl.store(output + 1, owned_bits.to(tl.int64))
    tl.store(output + 2, count.to(tl.int64))


STORE = None


def host(box):
    count, source = box
    box.clear()
    owned = torch.empty_strided((count,), (1,), dtype=torch.float32, device=source.device)
    output = torch.empty_strided((3,), (1,), dtype=torch.int64, device=source.device)
    STORE[(1,)](output, source.data_ptr(), owned.const_data_ptr() % (1 << 31), count)
    return owned, output


class TestLateAddressReplay(TestCase):
    def test_scalar_only_roots_and_selected_abi(self, device):
        with torch.cuda.device(device):
            adapter = DirectTriton(store_words)
            self.addCleanup(adapter.close)
            self.enterContext(mock.patch.dict(globals(), {"STORE": adapter}))
            contract = InputContract(("integer", "tensor"),
                (TensorInput(1, torch.float32, (8,), (1,)),),
                (IntegerRange(0, 1, 1024),), device_index=torch.cuda.current_device())
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            observations = self.enterContext(mock.patch.object(direct_host, "_observe_direct",
                                                                wraps=direct_host._observe_direct))
            preparations = self.enterContext(mock.patch.object(direct_host, "_prepare_observed",
                                                               wraps=direct_host._prepare_observed))
            captures = []
            original_make_replay = replay._make_replay

            def inspect_capture(*args, **kwargs):
                _, _, allocations, _, copies, calls, launches, buffers, _ = args
                self.assertEqual(copies, ())
                self.assertEqual(len(allocations), 2)
                call, = calls
                launch, = launches
                late = {argument.formal: argument for argument in call.arguments
                        if type(argument.source) is ParameterSource}
                self.assertEqual(set(late), {"input_address", "owned_bits"})
                self.assertEqual(late["input_address"].triton_type, "i64")
                self.assertEqual(late["owned_bits"].triton_type, "i32")
                owned_root = allocations[0].source
                expected = {"input_address": kwargs["capture_inputs"][1].data_ptr(),
                            "owned_bits": buffers[owned_root].data_ptr() % (1 << 31)}
                for name, argument in late.items():
                    bits = 64 if name == "input_address" else 32
                    self.assertEqual(argument.source.width, bits)
                    data = launch.argument_bytes[argument.call_arg_index]
                    self.assertEqual(data, struct.pack("q" if bits == 64 else "i", expected[name]))
                self.assertEqual({pointer.root for pointer in late["input_address"].source.pointers},
                                 {InputSource(1)})
                self.assertEqual({pointer.root for pointer in late["owned_bits"].source.pointers}, {owned_root})
                pointer_roots = {argument.source.root for argument in call.arguments
                                 if type(argument.source) is PointerSource}
                self.assertNotIn(owned_root, pointer_roots)
                self.assertNotIn(InputSource(1), pointer_roots)
                result = original_make_replay(*args, **kwargs)
                captures.append({"scalar_slots": [late[name].call_arg_index
                                                   for name in ("input_address", "owned_bits")],
                                 "scalar_widths": [64, 32]})
                return result

            self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=inspect_capture))
            samples = []
            for count, offset in zip((5, 37, 100, 256, 11, 5), (0, 1, 4, 5, 8, 9)):
                storage = torch.empty(32, device=device)
                self.assertEqual(storage.data_ptr() % 16, 0)
                source = storage[offset:offset + 8]
                self.assertLess(source.data_ptr(), 1 << 63)
                samples.append((count, source))
            self.assertEqual(len({source.data_ptr() for _, source in samples}), 6)
            held, frames = [], []

            def profile(frame, event, result):
                if event == "call":
                    frames.append((frame.f_code.co_filename, frame.f_code.co_name))

            for index, (count, source) in enumerate(samples):
                box = [count, source]
                if index < 2:
                    owned, actual = runtime(box)
                else:
                    try:
                        sys.setprofile(profile)
                        owned, actual = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                self.assertEqual(tuple(owned.shape), (count,))
                expected = torch.tensor((source.data_ptr(), owned.data_ptr() % (1 << 31), count),
                                        dtype=torch.int64, device=device)
                self.assertEqual(actual, expected)
                ordinary_owned, ordinary = host([count, source])
                ordinary_expected = torch.tensor((source.data_ptr(), ordinary_owned.data_ptr() % (1 << 31), count),
                                                 dtype=torch.int64, device=device)
                self.assertEqual(ordinary, ordinary_expected)
                held.append((owned, actual, expected))
                self.assertEqual(len(runtime.variants), min(index + 1, 2))
            self.assertEqual(observations.call_count, 2)
            self.assertEqual(preparations.call_count, 2)
            self.assertEqual(len(captures), 2)
            self.assertEqual(len({owned.data_ptr() for owned, _, _ in held}), 6)
            selections = []
            for variant in runtime.variants:
                call, = [event for event in variant.program.events if type(event) is DirectKernelCall]
                row, = [row for row in call.owner.formals if row.formal == "input_address"]
                selections.append(tuple(row.attributes))
                owned_row, = [row for row in call.owner.formals if row.formal == "owned_bits"]
                self.assertEqual(owned_row.attributes, ())
                trace = variant.program.guards.trace
                binding, = [binding for binding in trace.address_bindings if type(binding.root) is BufferSource]
                with mock.patch.dict(trace.shape_env.var_to_range, {binding.symbol: ValueRanges(0, (1 << 63) - 1)}):
                    with self.assertRaises(GuardExportDeclined):
                        export_guards(trace)
                with mock.patch.dict(trace.shape_env.replacements, {binding.symbol: sympy.Integer(0)}):
                    with self.assertRaises(GuardExportDeclined):
                        export_guards(trace)
            self.assertIn(("tt.divisibility", 16), selections[0])
            self.assertEqual(selections[1], ())
            runtime.close()
            adapter.close()
            for owned, actual, expected in held:
                self.assertEqual(actual, expected)
                self.assertEqual(actual[1].item(), owned.data_ptr() % (1 << 31))
            print("LATE_ADDRESS_RESULT=" + json.dumps({
                "accepted": True,

                "samples": 6, "ordinary_observations": 2, "preparations": 2,
                "captures": captures, "variants": 2, "native_hits": 4,
                "python_frames": frames, "ordinary_references": 6, "own_pointer_references": 6,
                "input_scalar_specializations": selections, "allocations_per_variant": 2,
                "owned_root_used_only_through_scalar": True,
                "held_output_pairs_after_close": len(held), "owned_guard_preflight_rejections": 4,
            }, sort_keys=True))


instantiate_device_type_tests(TestLateAddressReplay, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
