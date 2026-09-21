# Owner(s): ["module: inductor"]
import ctypes
import gc
import unittest
import weakref

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


CPP = r"""
#include <cstdint>
#include <cstring>

static int64_t stats[7]{};
extern "C" void reset_stats() {
  for (auto& value : stats) value = 0;
}
extern "C" int64_t read_stat(int index) { return stats[index]; }
extern "C" int32_t early_scalar(const int64_t* leaves, int64_t* values) {
  ++stats[0];
  values[0] = leaves[0];
  return 0;
}
extern "C" int32_t early_dynamic(const int64_t* leaves, int64_t* values) {
  ++stats[0];
  stats[2] = values[0] = leaves[0];
  stats[3] = values[1] = leaves[1];
  stats[4] = values[2] = leaves[2];
  values[3] = values[0] + values[2];
  values[4] = leaves[0];
  values[5] = 0;
  return 0;
}
extern "C" int32_t late_dynamic(
    const int64_t*, const uintptr_t* pointers, int64_t* outputs) {
  static_assert(sizeof(uintptr_t) == sizeof(int64_t));
  ++stats[1];
  std::memcpy(&stats[5], &pointers[0], sizeof(int64_t));
  std::memcpy(&stats[6], &pointers[2], sizeof(int64_t));
  outputs[0] = stats[6];
  return 0;
}
extern "C" int32_t late_zero_outputs(
    const int64_t* early, const uintptr_t*, int64_t*) {
  ++stats[1];
  if (early[0] == 0) return 3;
  return 0;
}
"""

EMPTY_LATE = ("parameter_v1", (), ())
DIVISION_LATE = (
    "parameter_v1",
    (("constant", 64, 7), ("value", 64, 0), ("udiv", 64, 0, 1, ())),
    (),
)
DYNAMIC_INSTRUCTIONS = (
    ("size", 0, 0),
    ("stride", 0, 0),
    ("boxed", 1),
    ("add", 0, 2),
    ("size", 0, 0),
    ("constant", 0),
)


class LibraryOwner:
    def __init__(self, library):
        self.library = library


@triton.jit(
    do_not_specialize=["value"], do_not_specialize_on_alignment=["value", "output"]
)
def store_scalar(value, output):
    tl.store(output, value)


@triton.jit(
    do_not_specialize=["size", "stride", "value", "address"],
    do_not_specialize_on_alignment=[
        "source",
        "output",
        "size",
        "stride",
        "value",
        "address",
    ],
)
def store_dynamic(source, output, size, stride, value, address):
    offsets = tl.arange(0, 64)
    values = tl.load(source + offsets * stride, offsets < size, other=0)
    tl.store(output + offsets, values + value + address, offsets < size)


@requires_cuda_and_triton
@unittest.skipIf(_get_torch_cuda_version() < (12, 8), "requires CUDA 12.8 or newer")
@skipIfRocm
class TestCompiledRegistration(TestCase):
    def setUp(self):
        super().setUp()
        self.library = CppCodeCache.load(CPP)
        self.library.read_stat.argtypes = (ctypes.c_int,)
        self.library.read_stat.restype = ctypes.c_int64
        self.library.reset_stats()

    def registration(self, *, early="early_scalar", late=None, owner=None, **counts):
        fields = dict(leaf_count=1, early_count=1, late_count=0, pointer_count=2)
        fields.update(counts)
        return torch._C._CUDAGraphCompiledEvaluation(
            early_address=ctypes.cast(
                getattr(self.library, early), ctypes.c_void_p
            ).value,
            late_address=0
            if late is None
            else ctypes.cast(getattr(self.library, late), ctypes.c_void_p).value,
            library_owner=self.library if owner is None else owner,
            **fields,
        )

    def capture_scalar(self, device, registration, late=EMPTY_LATE):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            output = torch.empty((1,), dtype=torch.int64, device=device)
            binary = store_scalar[(1,)](-(1 << 40), output)
            self.assertEqual(
                tuple(binary.src.signature.items()),
                (("value", "i64"), ("output", "*i64")),
            )
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                store_scalar[(1,)](-(1 << 40), output)
                frontier = torch._C._cuda_get_capture_frontier(stream.cuda_stream)
                self.assertEqual(len(frontier[3]), 1)
                node = frontier[3][0][0]
            graph.instantiate()
            batch = graph._prepare_kernel_replay_updates(
                ((node, 1, 0),),
                2,
                ((node, 0, 8, 0),),
                (),
                1,
            )
            entry = graph._make_boxed_replay(
                batch,
                stream,
                (binary, output),
                2,
                (0,),
                (),
                None,
                (("input", 0),),
                ((1,), (("boxed", 1),), late),
                compiled_evaluation=registration,
            )
        self.addCleanup(entry.close)
        self.addCleanup(stream.synchronize)
        return entry, stream

    def capture_dynamic(self, device, registration):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            source = torch.arange(4, dtype=torch.int64, device=device)
            output = torch.empty_like(source)
            arguments = (source, output, 4, 1, -(1 << 40), -(1 << 40))
            binary = store_dynamic[(1,)](*arguments)
            self.assertEqual(
                tuple(binary.src.signature.items()),
                (
                    ("source", "*i64"),
                    ("output", "*i64"),
                    ("size", "i32"),
                    ("stride", "i32"),
                    ("value", "i64"),
                    ("address", "i64"),
                ),
            )
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                store_dynamic[(1,)](*arguments)
                frontier = torch._C._cuda_get_capture_frontier(stream.cuda_stream)
                self.assertEqual(len(frontier[3]), 1)
                node = frontier[3][0][0]
            graph.instantiate()
            batch = graph._prepare_kernel_replay_updates(
                ((node, 0, 0), (node, 1, 2)),
                3,
                ((node, 2, 4, 0), (node, 3, 4, 1), (node, 4, 8, 3), (node, 5, 8, 6)),
                (),
                7,
            )
            late = ("parameter_v1", (("pointer", 64, 2, 5),), (0,))
            entry = graph._make_boxed_replay(
                batch,
                stream,
                (binary, source, output),
                2,
                (0,),
                ((torch.int64, (("value", 0),), (1,)),),
                None,
                (0,),
                ((1,), DYNAMIC_INSTRUCTIONS, late),
                compiled_evaluation=registration,
            )
        self.addCleanup(entry.close)
        self.addCleanup(stream.synchronize)
        return entry, stream

    @parametrize(
        "field,count",
        (
            ("leaf_count", 0),
            ("early_count", 2),
            ("late_count", 1),
            ("pointer_count", 3),
        ),
    )
    def test_abi_counts_must_match_plan(self, device, field, count):
        registration = self.registration(late="late_zero_outputs", **{field: count})
        message = (
            "Compiled leaf bindings"
            if field == "leaf_count"
            else "Compiled evaluation counts"
        )
        with self.assertRaisesRegex(ValueError, message):
            self.capture_scalar(device, registration)
        self.assertEqual((self.library.read_stat(0), self.library.read_stat(1)), (0, 0))

    def test_library_owner_retained_until_close(self, device):
        owner = LibraryOwner(self.library)
        reference = weakref.ref(owner)
        registration = self.registration(owner=owner)
        entry, stream = self.capture_scalar(device, registration)
        del owner, registration
        gc.collect()
        self.assertIsNotNone(reference())
        with torch.cuda.stream(stream):
            output = torch.full((1,), -1, dtype=torch.int64, device=device)
            (result,) = entry([output, 19])
            self.assertEqual(result, torch.full_like(output, 19))
        entry.close()
        gc.collect()
        self.assertIsNone(reference())

    def test_compiled_callbacks_use_fresh_metadata_and_allocations(self, device):
        registration = self.registration(
            early="early_dynamic",
            late="late_dynamic",
            leaf_count=3,
            early_count=6,
            late_count=1,
            pointer_count=3,
        )
        entry, stream = self.capture_dynamic(device, registration)
        held, source_addresses, output_addresses = [], set(), set()
        with torch.cuda.stream(stream):
            for call, (size, stride, bias) in enumerate(
                ((4, 1, 9), (7, 2, 13), (12, 3, -5)), 1
            ):
                source = torch.arange(size * stride, dtype=torch.int64, device=device)[
                    ::stride
                ]
                box = [source, bias]
                (result,) = entry(box)
                self.assertEqual(box, [])
                self.assertEqual(result.shape, source.shape)
                address = ctypes.c_int64(result.data_ptr()).value
                self.assertEqual(result, source + size + bias + address)
                self.assertEqual(
                    tuple(self.library.read_stat(index) for index in range(5)),
                    (call, call, size, stride, bias),
                )
                self.assertEqual(
                    self.library.read_stat(5), ctypes.c_int64(source.data_ptr()).value
                )
                self.assertEqual(self.library.read_stat(6), address)
                self.assertNotIn(source.data_ptr(), source_addresses)
                self.assertNotIn(result.data_ptr(), output_addresses)
                source_addresses.add(source.data_ptr())
                output_addresses.add(result.data_ptr())
                held.append((source, result))

    def test_nonempty_zero_output_program_requires_late_callback(self, device):
        registration = self.registration()
        with self.assertRaisesRegex(
            ValueError, "nonempty parameter program requires a compiled late function"
        ):
            self.capture_scalar(device, registration, DIVISION_LATE)
        self.assertEqual((self.library.read_stat(0), self.library.read_stat(1)), (0, 0))

    @parametrize("divisor", (0, 2))
    def test_nonempty_zero_output_program_preserves_hard_error(self, device, divisor):
        registration = self.registration(late="late_zero_outputs")
        self.assertEqual(registration.late_count, 0)
        entry, stream = self.capture_scalar(device, registration, DIVISION_LATE)
        with torch.cuda.stream(stream):
            output = torch.full((1,), -1, dtype=torch.int64, device=device)
            box = [output, divisor]
            if divisor == 0:
                with self.assertRaisesRegex(RuntimeError, "division by zero"):
                    torch._C._cuda_evaluate_parameter_program(
                        DIVISION_LATE, (divisor,), (0, 0)
                    )
                with self.assertRaisesRegex(
                    ValueError, "Compiled parameter evaluation failed with status 3"
                ):
                    entry(box)
                stream.synchronize()
                self.assertEqual(output, torch.full_like(output, -1))
            else:
                self.assertEqual(
                    torch._C._cuda_evaluate_parameter_program(
                        DIVISION_LATE, (divisor,), (0, 0)
                    ),
                    (),
                )
                (result,) = entry(box)
                self.assertIs(result, output)
                self.assertEqual(result, torch.full_like(output, divisor))
            self.assertEqual(box, [])
            self.assertEqual(
                (self.library.read_stat(0), self.library.read_stat(1)), (1, 1)
            )

    def test_empty_late_program_is_noop(self, device):
        entry, stream = self.capture_scalar(device, self.registration())
        with torch.cuda.stream(stream):
            output = torch.full((1,), -1, dtype=torch.int64, device=device)
            (result,) = entry([output, 23])
            self.assertEqual(result, torch.full_like(output, 23))
            self.assertEqual(
                (self.library.read_stat(0), self.library.read_stat(1)), (1, 0)
            )


instantiate_device_type_tests(TestCompiledRegistration, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
