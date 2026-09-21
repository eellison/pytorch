"""Validate compiler-derived CuTe scalar fields and fresh-address native replay."""

import json
import struct
import sys
from unittest import mock

import model
import torch
from torch._inductor.runtime._cudagraph import cute_adapter, direct_host, replay
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, InputSource, ParameterSource
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._inductor.runtime._cudagraph._compiler.user_triton.linking import LinkDeclined


class TestCuTeLateScalars(TestCase):
    def test_compiler_composition_and_scalar_root_lifetime(self, device):
        with torch.cuda.device(device):
            runtime, owner = model.make_example(torch.cuda.current_device())
            self.addCleanup(owner.close)
            self.addCleanup(runtime.close)
            observations = self.enterContext(mock.patch.object(direct_host, "_observe_direct",
                                                                wraps=direct_host._observe_direct))
            preparations = self.enterContext(mock.patch.object(direct_host, "_prepare_observed",
                                                               wraps=direct_host._prepare_observed))
            operand_sets, captures = [], []
            original_operands = cute_adapter._Operands
            original_make_replay = replay._make_replay

            def observe_operands(*args, **kwargs):
                result = original_operands(*args, **kwargs)
                operand_sets.append(result)
                return result

            def inspect_capture(*args, **kwargs):
                _, _, allocations, _, copies, calls, launches, buffers, _ = args
                self.assertEqual(copies, ())
                self.assertEqual(len(allocations), 2)
                call, = calls
                launch, = launches
                late = [field for field in call.fields if type(field.source) is ParameterSource]
                self.assertEqual(len(late), 2)
                owned_root = allocations[0].source
                expected = {InputSource(1): kwargs["capture_inputs"][1].data_ptr(),
                            owned_root: buffers[owned_root].data_ptr() % (1 << 30) + 7}
                fields = []
                for field in late:
                    root, = {pointer.root for pointer in field.source.pointers}
                    self.assertIn(root, expected)
                    width = 64 if type(root) is InputSource else 32
                    self.assertEqual(field.kind, f"i{width}")
                    self.assertEqual(field.source.width, width)
                    image = launch.argument_bytes[field.parameter]
                    self.assertEqual(image[field.byte_offset:field.byte_offset + width // 8],
                                     struct.pack("q" if width == 64 else "i", expected[root]))
                    fields.append((field.parameter, field.byte_offset, width))
                result = original_make_replay(*args, **kwargs)
                captures.append(fields)
                return result

            self.enterContext(mock.patch.object(cute_adapter, "_Operands", side_effect=observe_operands))
            self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=inspect_capture))
            samples = []
            for count, offset in zip((5, 37, 100, 256, 5), (0, 1, 4, 5, 8)):
                storage = torch.empty(32, device=device)
                source = storage[offset:offset + 8]
                self.assertLess(source.data_ptr(), 1 << 63)
                samples.append((count, source))
            self.assertEqual(len({source.data_ptr() for _, source in samples}), 5)
            held, frames = [], []

            def profile(frame, event, result):
                if event == "call":
                    frames.append((frame.f_code.co_filename, frame.f_code.co_name))

            for index, (count, source) in enumerate(samples):
                box = [count, source]
                if index == 0:
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
                expected = torch.tensor((source.data_ptr(), owned.data_ptr() % (1 << 30) + 7),
                                        dtype=torch.int64, device=device)
                self.assertEqual(actual, expected)
                ordinary_owned, ordinary = model.host([count, source])
                ordinary_expected = torch.tensor((source.data_ptr(), ordinary_owned.data_ptr() % (1 << 30) + 7),
                                                 dtype=torch.int64, device=device)
                self.assertEqual(ordinary, ordinary_expected)
                held.append((owned, actual, expected))
                self.assertEqual(len(runtime.variants), 1)
            self.assertEqual(observations.call_count, 1)
            self.assertEqual(preparations.call_count, 1)
            self.assertEqual(owner._capture.calls, 1)
            self.assertEqual(len(captures), 1)
            self.assertEqual(len({owned.data_ptr() for owned, _, _ in held}), 5)
            program = runtime.variants[0].program
            call, = [event for event in program.events if type(event) is CuTeCall]
            self.assertEqual(tuple(pointer.root for pointer in call.pointers), (program.allocations[1].source,))
            operands, = operand_sets
            artifact = call.bound.module.artifact
            self.assertIs(operands.artifact, artifact)
            scalar_formals = {formal.name: formal for formal in artifact.formals if formal.kind == "Var"}
            self.assertEqual(set(scalar_formals), {"input_address", "owned_bits"})
            for name, width in (("input_address", 64), ("owned_bits", 32)):
                formal = scalar_formals[name]
                leaf, = formal.leaves
                self.assertEqual(leaf.llvm_type, f"i{width}")
                with self.assertRaises(LinkDeclined):
                    operands.numeric(formal.source_arg_index, leaf.path)
            computed = [field.source.expression for field in call.bound.module.site.fields.integers
                        if field.source.kind == "compiler_expression"]
            self.assertEqual(len(computed), 1)
            self.assertEqual(computed[0].kind, "llvm.add")
            self.assertEqual(computed[0].llvm_type, "i32")
            runtime.close()
            self.assertEqual(owner._native_borrows, set())
            owner.close()
            for owned, actual, expected in held:
                self.assertEqual(actual, expected)
                self.assertEqual(actual[1].item(), owned.data_ptr() % (1 << 30) + 7)
            print("CUTE_LATE_SCALARS_RESULT=" + json.dumps({
                "accepted": True,

                "samples": 5, "ordinary_observations": 1, "preparations": 1,
                "cute_compilations": 1, "captures": captures, "variants": 1,
                "native_hits": 4, "python_frames": frames,
                "ordinary_references": 5, "own_pointer_references": 5,
                "scalar_widths": [64, 32], "compiler_scalar_operation": "llvm.add",
                "allocations": 2, "owned_root_used_only_through_scalar": True,
                "late_numeric_rejections": 2, "held_output_pairs_after_close": len(held),
            }, sort_keys=True))


instantiate_device_type_tests(TestCuTeLateScalars, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
