# Owner(s): ["module: inductor"]

import ctypes

import triton
import triton.language as tl

import torch
from torch._inductor.codecache import CppCodeCache
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@triton.jit(
    do_not_specialize=["value"], do_not_specialize_on_alignment=["value", "output"]
)
def fill(output, value):
    index = tl.arange(0, 4)
    tl.store(output + index, value + index)


PREDICATES = CppCodeCache.load("""#include <cstdint>
extern "C" int8_t always(int64_t*, double*) { return 1; }
extern "C" int8_t four(int64_t* values, double*) { return values[0] == 4; }
""")


def address(symbol):
    return ctypes.cast(symbol, ctypes.c_void_p).value


class TestBoundDispatch(TestCase):
    """`_cuda_make_boxed_dispatch(..., bound=)` binds trailing Tensor inputs the
    dispatcher owns. A plan without inputs takes the default empty tuple, and every
    input of a plan may be bound: the caller's box is then empty."""

    def capture(self, device, input_count):
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
            # the kernel writes the owned allocation (pointer index input_count)
            batch = graph._prepare_kernel_replay_updates(
                ((node, slots["output"], input_count),),
                input_count + 1,
                ((node, slots["value"], 8, 0),),
                (),
                1,
            )
        entry = graph._make_boxed_replay(
            batch,
            stream,
            (binary, captured),
            input_count,
            (),
            ((torch.int64, (4,), (1,)),),
            None,
            (0,),
            ((), (("constant", 7),)),
            None,
        )
        self.addCleanup(entry.close)
        return entry, stream

    def test_a_plan_without_inputs_takes_the_default_empty_bound(self, device):
        entry, stream = self.capture(device, 0)
        misses = []
        predicate = ((), address(PREDICATES.always), PREDICATES, (), (), ())
        dispatch = torch._C._cuda_make_boxed_dispatch(
            ((entry, predicate),), lambda box: misses.append(box) or ()
        )
        self.addCleanup(dispatch.close)
        expected = torch.arange(7, 11, device=device)
        with torch.cuda.stream(stream):
            for _ in range(3):
                box = []
                actual = dispatch(box)
                self.assertEqual(box, [])
                self.assertEqual(actual[0], expected)
        self.assertEqual(misses, [])
        with self.assertRaisesRegex(ValueError, "exceed the prepared input count"):
            dispatch.bind((torch.empty((4,), dtype=torch.int64, device=device),))
        with self.assertRaisesRegex(ValueError, "exceed the prepared input count"):
            torch._C._cuda_make_boxed_dispatch(
                ((entry, predicate),),
                lambda box: (),
                bound=(torch.empty((4,), dtype=torch.int64, device=device),),
            )

    def test_every_input_bound_serves_an_empty_box_and_rebinds(self, device):
        entry, stream = self.capture(device, 1)
        four = torch.empty((4,), dtype=torch.int64, device=device)
        five = torch.empty((5,), dtype=torch.int64, device=device)
        misses = []
        predicate = ((), address(PREDICATES.four), PREDICATES, (), (), (("size", 0, 0),))
        dispatch = torch._C._cuda_make_boxed_dispatch(
            ((entry, predicate),), lambda box: misses.append(list(box)) or (), bound=(four,)
        )
        self.addCleanup(dispatch.close)
        expected = torch.arange(7, 11, device=device)
        with torch.cuda.stream(stream):
            # the caller's box omits the bound input, or carries it
            for box in ([], [four], []):
                actual = dispatch(box)
                self.assertEqual(box, [])
                self.assertEqual(actual[0], expected)
            self.assertEqual(misses, [])
            dispatch.bind((five,))
            self.assertEqual(dispatch([]), ())
            self.assertEqual(len(misses), 1)
            dispatch.bind((four,))
            self.assertEqual(dispatch([])[0], expected)
            self.assertEqual(len(misses), 1)
        with self.assertRaisesRegex(ValueError, "exceed the prepared input count"):
            dispatch.bind((four, four))


instantiate_device_type_tests(TestBoundDispatch, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
