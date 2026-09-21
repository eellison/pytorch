# Owner(s): ["module: inductor"]

import ctypes

import triton
import triton.language as tl

import torch
from torch._inductor.codecache import CppCodeCache
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@triton.jit(
    do_not_specialize=["value"], do_not_specialize_on_alignment=["value", "output"]
)
def fill(output, value):
    index = tl.arange(0, 4)
    tl.store(output + index, value + index)


class TestNativeOutputReferences(TestCase):
    def capture(self, device, stream=None):
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            self.addCleanup(stream.synchronize)
        with torch.cuda.stream(stream):
            captured = torch.empty((4,), dtype=torch.int64, device=device)
            binary = fill[(1,)](captured, -(1 << 40))
            slots = {
                name: index
                for index, (name, _) in enumerate(binary.src.signature.items())
            }
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                fill[(1,)](captured, -(1 << 40))
                node = torch._C._cuda_get_capture_frontier(stream.cuda_stream)[3][0][0]
            graph.instantiate()
            batch = graph._prepare_kernel_replay_updates(
                ((node, slots["output"], 1),),
                2,
                ((node, slots["value"], 8, 0),),
                (),
                1,
            )
        return graph, batch, stream, (binary, captured), node

    def bind(self, capture, outputs, release=False):
        graph, batch, stream, resources, node = capture
        release_plan = None
        if release:
            release_plan = ((("drop", 0), ("allocate", 0), ("kernel", 0)), (node,))
        entry = graph._make_boxed_replay(
            batch,
            stream,
            resources,
            1,
            (),
            ((torch.int64, (4,), (1,)),),
            None,
            outputs,
            ((), (("constant", 7),)),
            release_plan,
        )
        self.addCleanup(entry.close)
        return entry

    @parametrize("kind", ("buffer", "input", "view"))
    def test_reference_chains_preserve_identity_and_lifetime(self, device, kind):
        capture = self.capture(device)
        stream = capture[2]
        first = {
            "buffer": 0,
            "input": ("input", 0),
            "view": ("view", 1, (3,), (1,), 1),
        }[kind]
        outputs = (None, first, ("literal", 19), ("output", 1), ("output", 3))
        entry = self.bind(capture, outputs)
        held = []
        with torch.cuda.stream(stream):
            for offset in (20, 40):
                source = torch.arange(4, dtype=torch.int64, device=device) + offset
                box = [source]
                actual = entry(box)
                self.assertEqual(box, [])
                self.assertIsNone(actual[0])
                self.assertEqual(actual[2], 19)
                self.assertIs(actual[1], actual[3])
                self.assertIs(actual[3], actual[4])
                expected = (
                    source if kind == "input" else torch.arange(7, 11, device=device)
                )
                if kind == "view":
                    expected = expected[1:]
                self.assertEqual(actual[1], expected)
                if kind == "input":
                    self.assertIs(actual[1], source)
                held.append((actual, expected))
            entry.close()
            self.assertIsNot(held[0][0][1], held[1][0][1])
            for actual, expected in held:
                self.assertEqual(actual[1], expected)
                self.assertIs(actual[1], actual[4])

    @parametrize(
        "kind", ("self", "forward", "none", "literal", "value", "duplicate_buffer")
    )
    def test_invalid_references_are_rejected_at_construction(self, device, kind):
        outputs = {
            "self": (("output", 0),),
            "forward": (("output", 1), 0),
            "none": (None, ("output", 0)),
            "literal": (("literal", 7), ("output", 0)),
            "value": (("value", 0), ("output", 0)),
            "duplicate_buffer": (0, 0),
        }[kind]
        capture = self.capture(device)
        message = (
            "Output indices must be distinct and in range"
            if kind == "duplicate_buffer"
            else "Output references must name preceding Tensor outputs"
        )
        with self.assertRaisesRegex(ValueError, message):
            self.bind(capture, outputs)

    @parametrize("index", (-1, True, 0.0))
    def test_reference_index_must_be_an_exact_nonnegative_integer(self, device, index):
        capture = self.capture(device)
        error = ValueError if type(index) is int else TypeError
        with self.assertRaisesRegex(error, "Output reference index must be"):
            self.bind(capture, (0, ("output", index)))

    @parametrize("kind", ("input", "view"))
    def test_reference_chains_cannot_release_a_returned_input(self, device, kind):
        first = ("input", 0) if kind == "input" else ("view", 0, (3,), (1,), 1)
        capture = self.capture(device)
        with self.assertRaisesRegex(
            ValueError, "Returned Tensor inputs cannot be released"
        ):
            self.bind(capture, (first, ("output", 0), ("output", 1)), release=True)

    def test_dispatch_accepts_repeated_and_distinct_tensor_outputs(self, device):
        first_capture = self.capture(device)
        stream = first_capture[2]
        second_capture = self.capture(device, stream)
        view = ("view", 1, (3,), (1,), 1)
        repeated = self.bind(first_capture, (view, ("output", 0)))
        distinct = self.bind(second_capture, (view, view))
        library = CppCodeCache.load("""#include <cstdint>
extern "C" int8_t four(int64_t* values, double*) { return values[0] == 4; }
extern "C" int8_t five(int64_t* values, double*) { return values[0] == 5; }
""")
        facts = (("size", 0, 0),)
        four = ctypes.cast(library.four, ctypes.c_void_p).value
        five = ctypes.cast(library.five, ctypes.c_void_p).value
        misses = []
        dispatch = torch._C._cuda_make_boxed_dispatch(
            (
                (repeated, ((), four, library, (), (), facts)),
                (distinct, ((), five, library, (), (), facts)),
            ),
            lambda box: misses.append(box) or (),
        )
        self.addCleanup(dispatch.close)
        with torch.cuda.stream(stream):
            for size in (4, 5, 4):
                source = torch.empty((size,), dtype=torch.int64, device=device)
                box = [source]
                actual = dispatch(box)
                self.assertEqual(box, [])
                self.assertEqual(actual[0], torch.arange(8, 11, device=device))
                self.assertEqual(actual[0], actual[1])
                if size == 4:
                    self.assertIs(actual[0], actual[1])
                else:
                    self.assertIsNot(actual[0], actual[1])
            self.assertEqual(misses, [])


instantiate_device_type_tests(TestNativeOutputReferences, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
