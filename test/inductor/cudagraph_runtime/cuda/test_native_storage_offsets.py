"""Exercise signed early arithmetic and mixed native predicate selectors."""

import ctypes
import json
import sys


import torch
import triton
import triton.language as tl
from torch._inductor.codecache import CppCodeCache
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


MIN = -(1 << 63)
MAX = (1 << 63) - 1
SENTINEL = 424242


@triton.jit(do_not_specialize=["value"], do_not_specialize_on_alignment=["value", "output"])
def store_value(value, output):
    tl.store(output, value)


class TestNativeStorageOffsets(TestCase):
    def capture_entry(self, device, instructions):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            captured_output = torch.empty((1,), dtype=torch.int64, device=device)
            binary = store_value[(1,)](-(1 << 40), captured_output)
            signature = tuple(binary.src.signature.items())
            self.assertEqual(signature, (("value", "i64"), ("output", "*i64")))
            slots = {name: index for index, (name, _) in enumerate(signature)}
            self.assertEqual((binary.metadata.global_scratch_size, binary.metadata.profile_scratch_size), (0, 0))
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                store_value[(1,)](-(1 << 40), captured_output)
                frontier = torch._C._cuda_get_capture_frontier(stream.cuda_stream)
                self.assertEqual(len(frontier[3]), 1)
                node = frontier[3][0][0]
            graph.instantiate()
            batch = graph._prepare_kernel_replay_updates(
                ((node, slots["output"], 2),), 3,
                ((node, slots["value"], 8, len(instructions) - 1),), (), len(instructions),
            )
            entry = graph._make_boxed_replay(
                batch, stream, (binary, captured_output), 3, (2,), (), None,
                (("input", 2),), ((0, 1), instructions),
            )
        self.addCleanup(entry.close)
        self.addCleanup(stream.synchronize)
        return entry, stream

    @parametrize("operation,left,right,valid_right", (
        ("add", MAX, 1, 0),
        ("add", MIN, -1, 0),
        ("multiply", MIN, -1, 1),
        ("multiply", MAX, 2, 1),
        ("multiply", MIN, 2, 0),
    ))
    def test_signed_overflow_refuses_before_replay(self, device, operation, left, right, valid_right):
        instructions = (("boxed", 0), ("boxed", 1), (operation, 0, 1))
        entry, stream = self.capture_entry(device, instructions)
        with torch.cuda.stream(stream):
            output = torch.full((1,), SENTINEL, dtype=torch.int64, device=device)
            box = [left, valid_right, output]
            result, = entry(box)
            self.assertEqual(box, [])
            self.assertIs(result, output)
            expected = left + valid_right if operation == "add" else left * valid_right
            self.assertEqual(result, torch.full_like(output, expected))
            output.fill_(SENTINEL)
            message = "addition" if operation == "add" else "multiplication"
            with self.assertRaisesRegex(ValueError, f"Numeric {message} overflowed int64"):
                entry([left, right, output])
            self.assertEqual(output, torch.full_like(output, SENTINEL))
            entry.close()
            self.assertEqual(result, torch.full_like(output, SENTINEL))

    def test_integer_pointer_and_offset_slots_together(self, device):
        instructions = (("boxed", 0), ("boxed", 1), ("storage_offset", 2),
                        ("constant", 0), ("add", 2, 3))
        entry, stream = self.capture_entry(device, instructions)
        library = CppCodeCache.load('''#include <cstdint>
extern "C" int8_t guard(int64_t* values, double*) {
  return static_cast<uint64_t>(values[1]) == static_cast<uint64_t>(values[2])
      && values[0] == values[3];
}
''')
        address = ctypes.cast(library.guard, ctypes.c_void_p).value
        misses = []

        def ordinary(box):
            pointer_bits, expected_offset, output = box
            box.clear()
            misses.append((pointer_bits, expected_offset, output.storage_offset()))
            store_value[(1,)](output.storage_offset(), output)
            return (output,)

        registration = ((1, 0), address, library, (2,), (2,))
        dispatcher = torch._C._cuda_make_boxed_dispatch(((entry, registration),), ordinary)
        self.addCleanup(dispatcher.close)
        held, addresses, hits = [], set(), 0
        with torch.cuda.stream(stream):
            for offset, mismatch in ((0, None), (1, None), (3, None), (2, "pointer"), (4, "offset"), (1, None)):
                storage = torch.full((8,), SENTINEL, dtype=torch.int64, device=device)
                output = storage[offset:offset + 1]
                pointer = output.data_ptr()
                self.assertNotIn(pointer, addresses)
                addresses.add(pointer)
                pointer_bits = ctypes.c_int64(pointer).value
                expected_offset = offset
                if mismatch == "pointer":
                    pointer_bits ^= 8
                elif mismatch == "offset":
                    expected_offset += 1
                frames = []

                def profile(frame, event, result):
                    if event == "call":
                        frames.append(frame.f_code.co_name)

                box = [pointer_bits, expected_offset, output]
                before = len(misses)
                try:
                    sys.setprofile(profile)
                    result, = dispatcher(box)
                finally:
                    sys.setprofile(None)
                self.assertEqual(box, [])
                self.assertIs(result, output)
                self.assertEqual(result, torch.full_like(output, offset))
                if mismatch is None:
                    self.assertEqual(frames, [])
                    self.assertEqual(len(misses), before)
                    hits += 1
                else:
                    self.assertEqual(len(misses), before + 1)
                    self.assertEqual(misses[-1], (pointer_bits, expected_offset, offset))
                self.assertEqual(storage[:offset], torch.full_like(storage[:offset], SENTINEL))
                self.assertEqual(storage[offset + 1:], torch.full_like(storage[offset + 1:], SENTINEL))
                held.append((result, offset))
            self.assertEqual(hits, 4)
            self.assertEqual(len(misses), 2)
            dispatcher.close()
            for result, offset in held:
                self.assertEqual(result, torch.full_like(result, offset))
        print("NATIVE_MIXED_OFFSET_GUARDS=" + json.dumps({
            "native_hits": hits, "ordinary_misses": len(misses), "held_outputs": len(held),
            "integer_indices": [1, 0], "pointer_indices": [2], "storage_offset_indices": [2],
        }, sort_keys=True))


instantiate_device_type_tests(TestNativeStorageOffsets, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
