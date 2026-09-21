# Owner(s): ["module: cuda"]

import ctypes
import sys
import unittest

import torch
from torch._inductor.codecache import CppCodeCache
from torch._inductor.runtime.triton_compat import tl, triton
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfRocm,
    TestCase,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


@triton.jit(
    do_not_specialize=["value"], do_not_specialize_on_alignment=["value", "output"]
)
def store_metadata(value, output):
    tl.store(output, value)


@requires_cuda_and_triton
@unittest.skipIf(_get_torch_cuda_version() < (12, 8), "requires CUDA 12.8 or newer")
@skipIfRocm
class TestCUDAGraphHostTraceMetadata(TestCase):
    instructions = (
        ("boxed", 0),
        ("size", 1, 0),
        ("size", 1, 1),
        ("stride", 1, 0),
        ("stride", 1, 1),
        ("add", 1, 3),
        ("multiply", 2, 4),
        ("add", 5, 6),
    )

    def capture_entry(self, device, instructions=None):
        if instructions is None:
            instructions = self.instructions
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            output = torch.empty((1,), dtype=torch.int64, device=device)
            binary = store_metadata[(1,)](-(1 << 40), output)
            self.assertEqual(
                tuple(binary.src.signature.items()),
                (("value", "i64"), ("output", "*i64")),
            )
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                store_metadata[(1,)](-(1 << 40), output)
                frontier = torch._C._cuda_get_capture_frontier(stream.cuda_stream)
                self.assertEqual(len(frontier[3]), 1)
                node = frontier[3][0][0]
            graph.instantiate()
            batch = graph._prepare_kernel_replay_updates(
                ((node, 1, 2),),
                3,
                ((node, 0, 8, len(instructions) - 1),),
                (),
                len(instructions),
            )
            entry = graph._make_boxed_replay(
                batch,
                stream,
                (binary, output),
                3,
                (2,),
                (),
                None,
                (("input", 2),),
                ((0,), instructions),
            )
        self.addCleanup(entry.close)
        self.addCleanup(stream.synchronize)
        return entry, stream

    def make_dispatcher(self, entry, expression, bindings, misses):
        library = CppCodeCache.load(
            '#include <cstdint>\nextern "C" int8_t guard(int64_t* values, double*) {\n'
            f"  return {expression};\n"
            "}\n"
        )
        address = ctypes.cast(library.guard, ctypes.c_void_p).value

        def ordinary(box):
            misses.append(tuple(box))
            result = box[2]
            box.clear()
            return (result,)

        predicate = ((0,), address, library, (2,), (1,), bindings)
        dispatcher = torch._C._cuda_make_boxed_dispatch(((entry, predicate),), ordinary)
        self.addCleanup(dispatcher.close)
        return dispatcher

    @parametrize(
        "size,stride",
        (((2, 3), (3, 1)), ((3, 2), (1, 3)), ((4, 3), (9, 2)), ((0, 5), (9, 1))),
    )
    def test_numeric_sizes_and_strides(self, device, size, stride):
        entry, stream = self.capture_entry(device)
        with torch.cuda.stream(stream):
            source = torch.empty_strided(size, stride, device=device)
            output = torch.full((1,), -1, dtype=torch.int64, device=device)
            box = [0, source, output]
            (result,) = entry(box)
            self.assertEqual(box, [])
            self.assertIs(result, output)
            expected = (
                source.size(0) + source.stride(0) + source.size(1) * source.stride(1)
            )
            self.assertEqual(result, torch.full_like(output, expected))

    def test_runtime_rank_refusal_before_replay(self, device):
        entry, stream = self.capture_entry(device)
        with torch.cuda.stream(stream):
            source = torch.empty((6,), device=device)
            output = torch.full((1,), -1, dtype=torch.int64, device=device)
            box = [0, source, output]
            with self.assertRaisesRegex(ValueError, "in-range strided Tensor"):
                entry(box)
            self.assertEqual(output, torch.full_like(output, -1))
            self.assertEqual(len(box), 3)
            self.assertFalse(entry.failed)
            (result,) = entry([0, source.view(2, 3), output])
            self.assertEqual(result, torch.full_like(output, 8))

    def test_mixed_predicate_slots_and_rank_misses(self, device):
        entry, stream = self.capture_entry(device)
        source = torch.empty((2, 3), device=device)
        bindings = (
            ("size", 1, 1),
            ("stride", 1, 1),
            ("dtype", 1, 0),
            ("device", 1, 0),
            ("neg", 1, 0),
            ("conj", 1, 0),
            ("layout", 1, 0),
            ("rank", 1, 0),
        )
        expected = [
            torch._C._cuda_boxed_tensor_metadata(source, kind, dim)
            for kind, _, dim in bindings
        ]
        condition = " && ".join(
            f"values[{index + 3}] == {value}" for index, value in enumerate(expected)
        )
        misses = []
        dispatcher = self.make_dispatcher(
            entry,
            f"values[0] == values[2] && values[1] != 0 && {condition}",
            bindings,
            misses,
        )
        with torch.cuda.stream(stream):
            for candidate, hit in (
                (source, True),
                (source.view(-1), False),
                (source[:1, :1].reshape(()), False),
                (source.to(torch.float64), False),
                (torch.empty_strided((2, 3), (1, 2), device=device), False),
                (torch._neg_view(source), False),
                (source, True),
            ):
                output = torch.full((1,), -1, dtype=torch.int64, device=device)
                box = [candidate.storage_offset(), candidate, output]
                before = len(misses)
                frames = []

                def profile(frame, event, result):
                    if event == "call":
                        frames.append(frame.f_code.co_name)

                try:
                    sys.setprofile(profile)
                    (result,) = dispatcher(box)
                finally:
                    sys.setprofile(None)
                self.assertEqual(box, [])
                self.assertIs(result, output)
                self.assertEqual(result, torch.full_like(output, 8 if hit else -1))
                self.assertEqual(len(misses), before + (not hit))
                if hit:
                    self.assertEqual(frames, [])
            self.assertFalse(dispatcher.failed)

    @parametrize("kind", ("dtype", "device", "rank", "neg", "conj", "layout"))
    def test_scalar_metadata_guard(self, device, kind):
        entry, stream = self.capture_entry(device, (("boxed", 0), ("constant", 17)))
        source = torch.empty((2, 3), dtype=torch.complex64, device=device)
        changed = {
            "dtype": lambda: source.to(torch.complex128),
            "device": lambda: source.cpu(),
            "rank": lambda: source.view(-1),
            "neg": lambda: torch._neg_view(source),
            "conj": lambda: source.conj(),
            "layout": lambda: source.to_sparse(),
        }[kind]()
        expected = torch._C._cuda_boxed_tensor_metadata(source, kind)
        self.assertNotEqual(
            torch._C._cuda_boxed_tensor_metadata(changed, kind), expected
        )
        misses = []
        dispatcher = self.make_dispatcher(
            entry, f"values[3] == {expected}", ((kind, 1, 0),), misses
        )
        with torch.cuda.stream(stream):
            for candidate, hit in ((source, True), (changed, False), (source, True)):
                output = torch.full((1,), -1, dtype=torch.int64, device=device)
                (result,) = dispatcher([0, candidate, output])
                self.assertEqual(result, torch.full_like(output, 17 if hit else -1))
            self.assertEqual(len(misses), 1)

    @parametrize(
        "bindings,error,match",
        (
            ((("size", 0, 0),), ValueError, "declared Tensor"),
            ((("size", 3, 0),), ValueError, "declared Tensor"),
            ((("size", 1, -1),), ValueError, "nonnegative"),
            ((("size", 1, True),), TypeError, "exact integer"),
            ((("rank", 1, 1),), ValueError, "dimension zero"),
            ((("unknown", 1, None),), ValueError, "Unsupported Tensor metadata"),
            ((("size", 1),), TypeError, "kind, input, dimension"),
            ([("size", 1, 0)], TypeError, "exact tuple"),
        ),
    )
    def test_invalid_metadata_registration(self, device, bindings, error, match):
        entry, _ = self.capture_entry(device)
        with self.assertRaisesRegex(error, match):
            self.make_dispatcher(entry, "1", bindings, [])

    @parametrize(
        "instruction,error,match",
        (
            (("size", 0, 0), ValueError, "declared Tensor"),
            (("stride", 3, 0), ValueError, "declared Tensor"),
            (("size", 1, -1), ValueError, "nonnegative"),
            (("stride", 1, None), TypeError, "exact integer"),
        ),
    )
    def test_invalid_numeric_registration(self, device, instruction, error, match):
        with self.assertRaisesRegex(error, match):
            self.capture_entry(device, (("boxed", 0), instruction))


instantiate_device_type_tests(
    TestCUDAGraphHostTraceMetadata, globals(), only_for="cuda"
)

if __name__ == "__main__":
    run_tests()
