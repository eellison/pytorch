"""Late mixed-width scalars update a fixed-address captured kernel."""

import hashlib
import json
from pathlib import Path
import struct
import sys


import torch
import triton
import triton.language as tl
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@triton.jit(
    do_not_specialize=["narrow", "wide"],
    do_not_specialize_on_alignment=["narrow", "wide", "output"],
)
def store_scalars(narrow, wide, output):
    tl.store(output, narrow.to(tl.int64))
    tl.store(output + 1, wide.to(tl.int64))


class TestScalarOnlyUpdates(TestCase):
    def test_fixed_pointer_mixed_width_updates(self, device):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        self.addCleanup(stream.synchronize)
        with torch.cuda.stream(stream):
            output = torch.empty((2,), dtype=torch.int64, device=device)
            ordinary = torch.empty_like(output)
            base = 1 << 40
            warm = base + 17
            binary = store_scalars[(1,)](17, warm, output)
            signature = tuple(binary.src.signature.items())
            self.assertEqual(signature, (("narrow", "i32"), ("wide", "i64"), ("output", "*i64")))
            self.assertEqual((binary.metadata.global_scratch_size, binary.metadata.profile_scratch_size), (0, 0))
            self.assertEqual(binary.metadata.shared, 0)
            self.assertFalse(binary.run.gsan_enabled)
            slots = {name: index for index, (name, _) in enumerate(signature)}
            self.assertEqual(output, torch.tensor((17, warm), dtype=torch.int64, device=device))
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            self.addCleanup(graph.reset)
            with torch.cuda.graph(graph, stream=stream):
                store_scalars[(1,)](17, warm, output)
                frontier = torch._C._cuda_get_capture_frontier(stream.cuda_stream)
                self.assertEqual(len(frontier[3]), 1)
                node = frontier[3][0][0]
            graph.instantiate()
            snapshot, = graph._inspect_captured_kernel_nodes((node,))[3]
            fields = snapshot[8]
            self.assertEqual(snapshot[4], (1, 1, 1))
            self.assertEqual(snapshot[6], 0)
            self.assertEqual(len(fields), len(signature) + 2)
            self.assertEqual(tuple(field[1] for field in fields[:len(signature)]), (4, 8, 8))
            self.assertEqual(tuple(field[2] for field in fields[len(signature):]), (bytes(8), bytes(8)))
            self.assertEqual(struct.unpack("i", fields[slots["narrow"]][2])[0], 17)
            self.assertEqual(struct.unpack("q", fields[slots["wide"]][2])[0], warm)
            self.assertEqual(struct.unpack("Q", fields[slots["output"]][2])[0], output.data_ptr())

            early = (("boxed", 0),)
            parameters = ("parameter_v1", (("value", 64, 0), ("trunc", 32, 0)), (1, 0))
            scalar_bindings = ((node, slots["narrow"], 4, 1), (node, slots["wide"], 8, 2))
            batch = graph._prepare_kernel_replay_updates(
                ((node, slots["output"], 1),), 2, scalar_bindings, (), 3,
            )
            entry = graph._make_boxed_replay(
                batch, stream, (binary, output), 2, (1,), (), None,
                (("input", 1),), ((0,), early, parameters),
            )
            self.addCleanup(entry.close)
            pointer = output.data_ptr()
            samples = tuple(base + low for low in
                            (17, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x100000011, 0x100000011, 17))
            observed = []
            for raw in samples:
                low = raw & 0xFFFFFFFF
                narrow = low if low < 0x80000000 else low - 0x100000000
                expected = torch.tensor((narrow, raw), dtype=torch.int64, device=device)
                output.fill_(-777)
                ordinary.fill_(-888)
                reference = store_scalars[(1,)](narrow, raw, ordinary)
                self.assertEqual(tuple(reference.src.signature.items()), signature)
                frames = []

                def profile(frame, event, result):
                    if event == "call":
                        frames.append(frame.f_code.co_name)

                box = [raw, output]
                try:
                    sys.setprofile(profile)
                    result, = entry(box)
                finally:
                    sys.setprofile(None)
                self.assertEqual(frames, [])
                self.assertEqual(box, [])
                self.assertIs(result, output)
                self.assertEqual(result.data_ptr(), pointer)
                self.assertEqual(result, expected)
                self.assertEqual(ordinary, expected)
                observed.append(result.tolist())
            self.assertFalse(entry.failed)
            entry.close()
            self.assertTrue(entry.closed)
            self.assertEqual(result, torch.tensor((17, warm), dtype=torch.int64, device=device))
            print("SCALAR_ONLY_UPDATES_RESULT=" + json.dumps({


                "raw_values": samples, "observed": observed, "native_calls": len(samples),
                "ordinary_references": len(samples), "python_frames_on_calls": 0,
                "fixed_output_pointer": True, "fixed_grid": [1, 1, 1], "fixed_shared_bytes": 0,
                "parameter_layout": [(field[0], field[1]) for field in fields],
                "scalar_widths": [32, 64], "output_survives_close": True,
            }, sort_keys=True))


instantiate_device_type_tests(TestScalarOnlyUpdates, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
