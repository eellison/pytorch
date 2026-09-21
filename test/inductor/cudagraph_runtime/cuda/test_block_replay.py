# Owner(s): ["module: inductor"]
"""One CUDA function observes changing block axes through native replay."""

import unittest
from types import SimpleNamespace

import torch
from torch._inductor.runtime.cudagraph_arg_mapping import IntegerInput, IntExpr
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.cudagraph_compiled_evaluation import compile_evaluation
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    skipIfRocm,
    TestCase,
)


_SOURCE = r"""
__global__ __launch_bounds__(128) void observe_block(long long* out) {
  if (blockIdx.x || blockIdx.y || blockIdx.z || threadIdx.x || threadIdx.y || threadIdx.z)
    return;
  out[0] = blockDim.x;
  out[1] = blockDim.y;
  out[2] = blockDim.z;
  out[3] = gridDim.x;
  out[4] = gridDim.y;
  out[5] = gridDim.z;
  out[6] = blockDim.x * blockDim.y * blockDim.z;
}
"""


@requires_cuda_python_bindings
@unittest.skipIf(_get_torch_cuda_version() < (12, 8), "requires CUDA 12.8 or newer")
@skipIfRocm
class TestBlockReplay(TestCase):
    def test_fixture_eager_block_axes_are_not_fixed(self, device):
        from cuda.bindings import driver

        from torch.cuda import _compile_kernel
        from torch.cuda._utils import _check_cuda_bindings

        kernel = _compile_kernel(_SOURCE, "observe_block")
        limit = _check_cuda_bindings(
            driver.cuFuncGetAttribute(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                kernel.func.value,
            )
        )
        self.assertEqual(limit, 128)
        output = torch.empty(7, dtype=torch.int64, device=device)
        for block in ((32, 1, 1), (8, 4, 1), (4, 2, 4), (1, 1, 1)):
            kernel(grid=(2, 1, 1), block=block, args=[output])
            expected = (*block, 2, 1, 1, block[0] * block[1] * block[2])
            self.assertEqual(output, torch.tensor(expected, device=device))

    def capture(self, device):
        from torch.cuda import _compile_kernel

        kernel = _compile_kernel(_SOURCE, "observe_block")
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            output = torch.empty(7, dtype=torch.int64, device=device)
            kernel(grid=(1, 1, 1), block=(32, 1, 1), shared_mem=8, args=[output])
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                kernel(grid=(1, 1, 1), block=(32, 1, 1), shared_mem=8, args=[output])
                frontier = torch._C._cuda_get_capture_frontier(stream.cuda_stream)
                self.assertEqual(len(frontier[3]), 1)
                node = frontier[3][0][0]
            snapshot = graph._inspect_captured_kernel_nodes((node,))[3][0]
            self.assertEqual(snapshot[1], kernel.func.value)
            self.assertEqual(snapshot[5], (32, 1, 1))
            graph.instantiate()
        self.addCleanup(stream.synchronize)
        return graph, stream, kernel, node, output

    def make_entry(self, captured, width, compiled):
        graph, stream, kernel, node, output = captured
        names = ("gx", "gy", "gz", "bx", "by", "bz", "shared")
        numeric = _NumericProgram(
            SimpleNamespace(
                input_names=("out", *names),
                integer_inputs=tuple(
                    IntegerInput(name, index + 1) for index, name in enumerate(names)
                ),
            ),
            (output, 1, 1, 1, 32, 1, 1, 8),
        )
        indices = tuple(numeric.add(IntExpr("boxed", index)) for index in range(1, 8))
        row = (node, *indices[:3])
        if width in (5, 8):
            row += (indices[6],)
        if width in (7, 8):
            row += indices[3:6]
        batch = graph._prepare_kernel_replay_updates(
            ((node, 0, 0),), 8, (), (row,), len(indices)
        )
        registration = None
        if compiled:
            evaluation = compile_evaluation(numeric, pointer_count=8)
            registration = torch._C._CUDAGraphCompiledEvaluation(
                **evaluation.registration_kwargs
            )
        entry = graph._make_boxed_replay(
            batch,
            stream,
            (kernel, output),
            8,
            (0,),
            (),
            None,
            (("input", 0),),
            (numeric.integer_indices, tuple(numeric.instructions)),
            compiled_evaluation=registration,
        )
        self.addCleanup(entry.close)
        return entry, stream

    @parametrize("width", (4, 5, 7, 8))
    @parametrize("compiled", (False, True))
    def test_one_function_replays_changed_launch_dimensions(
        self, device, width, compiled
    ):
        entry, stream = self.make_entry(self.capture(device), width, compiled)
        outputs = []
        with torch.cuda.stream(stream):
            for grid, requested, shared in (
                ((1, 1, 1), (32, 1, 1), 8),
                ((2, 1, 1), (8, 4, 1), 16),
                ((1, 2, 1), (4, 2, 4), 32),
                ((1, 1, 2), (1, 1, 1), 8),
                ((2, 2, 1), (16, 2, 2), 16),
            ):
                output = torch.full((7,), -1, dtype=torch.int64, device=device)
                (result,) = entry([output, *grid, *requested, shared])
                self.assertIs(result, output)
                block = requested if width in (7, 8) else (32, 1, 1)
                outputs.append(
                    (result, (*block, *grid, block[0] * block[1] * block[2]))
                )
        stream.synchronize()
        for result, expected in outputs:
            self.assertEqual(
                result, torch.tensor(expected, device=device, dtype=torch.int64)
            )

    @parametrize(
        "block", ((0, 1, 1), (-1, 1, 1), (1025, 1, 1), (16, 16, 1), (64, 32, 1))
    )
    def test_invalid_axes_and_total_threads_fail_before_submission(self, device, block):
        entry, stream = self.make_entry(self.capture(device), 8, True)
        with torch.cuda.stream(stream):
            output = torch.full((7,), -1, dtype=torch.int64, device=device)
            box = [output, 1, 1, 1, *block, 8]
            with self.assertRaisesRegex(ValueError, "range|maximum thread count"):
                entry(box)
            self.assertEqual(box, [output, 1, 1, 1, *block, 8])
            stream.synchronize()
            self.assertEqual(output, torch.full_like(output, -1))
            (result,) = entry([output, 1, 1, 1, 8, 4, 1, 8])
            self.assertEqual(
                result, torch.tensor((8, 4, 1, 1, 1, 1, 32), device=device)
            )

    @parametrize(
        "fault",
        (
            "short",
            "long",
            "negative_index",
            "bool_index",
            "out_of_range",
            "duplicate",
            "foreign_node",
        ),
    )
    def test_native_binding_admission(self, device, fault):
        graph, _, _, node, _ = self.capture(device)
        row = (node, 0, 1, 2, 3, 4, 5)
        expected = (TypeError, ValueError, IndexError)
        if fault == "short":
            row = row[:-1]
        elif fault == "long":
            row += (6, 6)
        elif fault == "negative_index":
            row = (*row[:4], -1, *row[5:])
        elif fault == "bool_index":
            row = (*row[:4], True, *row[5:])
        elif fault == "out_of_range":
            row = (*row[:4], 7, *row[5:])
        elif fault == "foreign_node":
            other, _, _, other_node, _ = self.capture(device)
            self.addCleanup(other.reset)
            row = (other_node, *row[1:])
        rows = (row, row) if fault == "duplicate" else (row,)
        with self.assertRaises(expected):
            graph._prepare_kernel_replay_updates(((node, 0, 0),), 8, (), rows, 7)

    def test_block_binding_cannot_reference_late_parameter_output(self, device):
        graph, stream, kernel, node, output = self.capture(device)
        batch = graph._prepare_kernel_replay_updates(
            ((node, 0, 0),), 8, (), ((node, 0, 1, 2, 7, 4, 5),), 8
        )
        late = ("parameter_v1", (("constant", 64, 32),), (0,))
        with self.assertRaisesRegex(
            ValueError, "Block bindings require early numeric values"
        ):
            graph._make_boxed_replay(
                batch,
                stream,
                (kernel, output),
                8,
                (0,),
                (),
                None,
                (("input", 0),),
                (
                    tuple(range(1, 8)),
                    tuple(("boxed", index) for index in range(1, 8)),
                    late,
                ),
            )


instantiate_device_type_tests(TestBlockReplay, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
