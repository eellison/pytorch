# Owner(s): ["module: inductor"]
"""The native numeric plan's n-ary max and min, replayed through a boxed entry."""

import triton
import triton.language as tl

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


SENTINEL = 424242


@triton.jit(
    do_not_specialize=["value"], do_not_specialize_on_alignment=["value", "output"]
)
def store_value(value, output):
    tl.store(output, value)


class TestNativeMinMax(TestCase):
    def capture_entry(self, device, instructions, value_index):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            captured_output = torch.empty((1,), dtype=torch.int64, device=device)
            binary = store_value[(1,)](-(1 << 40), captured_output)
            slots = {
                name: index
                for index, (name, _) in enumerate(binary.src.signature.items())
            }
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                store_value[(1,)](-(1 << 40), captured_output)
                frontier = torch._C._cuda_get_capture_frontier(stream.cuda_stream)
                node = frontier[3][0][0]
            graph.instantiate()
            batch = graph._prepare_kernel_replay_updates(
                ((node, slots["output"], 0),),
                1,
                ((node, slots["value"], 8, value_index),),
                (),
                len(instructions),
            )
            entry = graph._make_boxed_replay(
                batch,
                stream,
                (binary, captured_output),
                1,
                (0,),
                (),
                None,
                (("input", 0),),
                ((), instructions),
            )
        self.addCleanup(entry.close)
        self.addCleanup(stream.synchronize)
        return entry, stream

    def test_max_and_min_over_sizes_strides_and_constants(self, device):
        # value = max(size(0), stride(0), 5) * 1000 + min(size(0), stride(0), 5)
        instructions = (
            ("size", 0, 0),
            ("stride", 0, 0),
            ("constant", 5),
            ("max", 0, 1, 2),
            ("min", 0, 1, 2),
            ("constant", 1000),
            ("multiply", 3, 5),
            ("add", 6, 4),
        )
        entry, stream = self.capture_entry(device, instructions, 7)
        with torch.cuda.stream(stream):
            base = torch.full((64,), SENTINEL, dtype=torch.int64, device=device)
            for view, expected in (
                (base[:3], 5001),  # size 3, stride 1: max 5, min 1
                (base[::4], 16004),  # size 16, stride 4: max 16, min 4
                (base[:8], 8001),  # size 8, stride 1
                (base[::64], 64001),  # size 1, stride 64: max 64, min 1
            ):
                (result,) = entry([view])
                self.assertIs(result, view)
                self.assertEqual(int(result[0].item()), expected)

    def test_wide_max_over_many_operands(self, device):
        # 300 constants and the size load: the max is the size when it is largest
        constants = tuple(("constant", i) for i in range(300))
        instructions = (
            ("size", 0, 0),
            *constants,
            ("max", *range(301)),
        )
        entry, stream = self.capture_entry(device, instructions, 301)
        with torch.cuda.stream(stream):
            base = torch.full((1024,), SENTINEL, dtype=torch.int64, device=device)
            for view, expected in ((base[:1000], 1000), (base[:7], 299)):
                (result,) = entry([view])
                self.assertEqual(int(result[0].item()), expected)

    def test_plan_rejects_a_one_operand_max(self, device):
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            captured_output = torch.empty((1,), dtype=torch.int64, device=device)
            binary = store_value[(1,)](-(1 << 40), captured_output)
            slots = {
                name: index
                for index, (name, _) in enumerate(binary.src.signature.items())
            }
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                store_value[(1,)](-(1 << 40), captured_output)
                node = torch._C._cuda_get_capture_frontier(stream.cuda_stream)[3][0][0]
            graph.instantiate()
            batch = graph._prepare_kernel_replay_updates(
                ((node, slots["output"], 0),), 1, ((node, slots["value"], 8, 1),), (), 2
            )
            with self.assertRaisesRegex(
                ValueError, "Unsupported numeric instruction or arity"
            ):
                graph._make_boxed_replay(
                    batch,
                    stream,
                    (binary, captured_output),
                    1,
                    (0,),
                    (),
                    None,
                    (("input", 0),),
                    ((), (("size", 0, 0), ("max", 0))),
                )
        stream.synchronize()


instantiate_device_type_tests(TestNativeMinMax, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
