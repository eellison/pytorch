# Owner(s): ["module: inductor"]
"""Byte-memset nodes share native graph replay's numeric and pointer bindings."""

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
__global__ void observe_bytes(const unsigned char* src, unsigned char* dst) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 256) dst[i] = src[i];
}
"""


@requires_cuda_python_bindings
@unittest.skipIf(_get_torch_cuda_version() < (12, 8), "requires CUDA 12.8 or newer")
@skipIfRocm
class TestMemsetReplay(TestCase):
    def capture(self, device, format="byte", before=False):
        from cuda.bindings import runtime

        from torch.cuda import _compile_kernel
        from torch.cuda._utils import _check_cuda_bindings

        kernel = _compile_kernel(_SOURCE, "observe_bytes")
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            source = torch.full((256,), 0x19, dtype=torch.uint8, device=device)
            output = torch.empty_like(source)
            prior = torch.empty_like(source) if before else None
            prior_node = None
            kernel(grid=(8, 1, 1), block=(32, 1, 1), args=[source, output])
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                if prior is not None:
                    kernel(grid=(8, 1, 1), block=(32, 1, 1), args=[source, prior])
                    prior_node = torch._C._cuda_get_capture_frontier(
                        stream.cuda_stream
                    )[3][0][0]
                _check_cuda_bindings(
                    runtime.cudaMemsetAsync(
                        source.data_ptr() + 16, 0xA5, 32, stream.cuda_stream
                    )
                )
                memset_node = torch._C._cuda_get_capture_frontier(stream.cuda_stream)[
                    3
                ][0][0]
                kernel(grid=(8, 1, 1), block=(32, 1, 1), args=[source, output])
                kernel_node = torch._C._cuda_get_capture_frontier(stream.cuda_stream)[
                    3
                ][0][0]
            if format != "byte":
                params = _check_cuda_bindings(
                    runtime.cudaGraphMemsetNodeGetParams(memset_node)
                )
                if format == "wide":
                    params.elementSize = 2
                    params.width = 16
                elif format == "2d":
                    params.height = 2
                    params.pitch = 64
                _check_cuda_bindings(
                    runtime.cudaGraphMemsetNodeSetParams(memset_node, params)
                )
            graph.instantiate()
        self.addCleanup(stream.synchronize)
        return SimpleNamespace(
            graph=graph,
            stream=stream,
            kernel=kernel,
            source=source,
            output=output,
            memset=memset_node,
            node=kernel_node,
            prior=prior,
            prior_node=prior_node,
        )

    def make_entry(self, c, compiled):
        numeric = _NumericProgram(
            SimpleNamespace(
                input_names=("source", "offset", "count"),
                integer_inputs=(IntegerInput("offset", 1), IntegerInput("count", 2)),
            ),
            (c.source, 16, 32),
        )
        offset = numeric.add(IntExpr("boxed", 1))
        count = numeric.add(IntExpr("boxed", 2))
        pointers = ((c.node, 0, 0), (c.node, 1, 3))
        layouts = ((torch.uint8, (256,), (1,)),)
        if c.prior is not None:
            pointers += ((c.prior_node, 0, 0), (c.prior_node, 1, 4))
            layouts *= 2
        pointer_count = 3 + len(layouts)
        batch = c.graph._prepare_kernel_replay_updates(
            pointers,
            pointer_count,
            (),
            (),
            len(numeric.instructions),
            memset_bindings=((c.memset, 0, offset, count),),
        )
        registration = None
        if compiled:
            evaluation = compile_evaluation(numeric, pointer_count=pointer_count)
            registration = torch._C._CUDAGraphCompiledEvaluation(
                **evaluation.registration_kwargs
            )
        entry = c.graph._make_boxed_replay(
            batch,
            c.stream,
            (c.kernel, c.source, c.output, c.prior),
            3,
            (0,),
            layouts,
            None,
            tuple(range(len(layouts))),
            (numeric.integer_indices, tuple(numeric.instructions)),
            compiled_evaluation=registration,
        )
        self.addCleanup(entry.close)
        return entry

    def test_fixture_has_real_byte_memset_and_independent_reference(self, device):
        c = self.capture(device, before=True)
        with torch.cuda.stream(c.stream):
            c.graph.replay()
        c.stream.synchronize()
        expected = torch.full((256,), 0x19, dtype=torch.uint8)
        self.assertEqual(c.prior.cpu(), expected)
        expected[16:48] = 0xA5
        self.assertEqual(c.output.cpu(), expected)

    @parametrize("compiled", (False, True))
    def test_kernel_memset_kernel_order(self, device, compiled):
        c = self.capture(device, before=True)
        entry = self.make_entry(c, compiled)
        with torch.cuda.stream(c.stream):
            source = torch.full((256,), 0x37, dtype=torch.uint8, device=device)
            after, before = entry([source, 64, 32])
        c.stream.synchronize()
        expected = torch.full((256,), 0x37, dtype=torch.uint8)
        self.assertEqual(before.cpu(), expected)
        expected[64:96] = 0xA5
        self.assertEqual(after.cpu(), expected)

    @parametrize("compiled", (False, True))
    def test_rebinding_preserves_bytes_and_output_lifetime(self, device, compiled):
        c = self.capture(device)
        entry = self.make_entry(c, compiled)
        retained = []
        with torch.cuda.stream(c.stream):
            for offset in (16, 32, 48, 64):
                source = torch.full((256,), 0x19, dtype=torch.uint8, device=device)
                (output,) = entry([source, offset, 32])
                expected = torch.full((256,), 0x19, dtype=torch.uint8)
                expected[offset : offset + 32] = 0xA5
                retained.append((output, expected))
        c.stream.synchronize()
        self.assertEqual(len({out.data_ptr() for out, _ in retained}), len(retained))
        for output, expected in retained:
            self.assertEqual(output.cpu(), expected)

    @parametrize("compiled", (False, True))
    def test_unchanged_bindings_still_reset_every_replay(self, device, compiled):
        c = self.capture(device)
        entry = self.make_entry(c, compiled)
        with torch.cuda.stream(c.stream):
            for sentinel in (0x19, 0x37, 0x51):
                c.source.fill_(sentinel)
                (output,) = entry([c.source, 16, 32])
                expected = torch.full((256,), sentinel, dtype=torch.uint8)
                expected[16:48] = 0xA5
                self.assertEqual(output.cpu(), expected)

    @parametrize(
        "offset,count",
        ((16, 0), (16, -1), (-(1 << 63), 32), ((1 << 63) - 1, (1 << 63) - 1)),
    )
    def test_invalid_domain_fails_before_gpu_work(self, device, offset, count):
        c = self.capture(device)
        entry = self.make_entry(c, True)
        with torch.cuda.stream(c.stream):
            c.source.fill_(0x19)
            with self.assertRaisesRegex(ValueError, "range|overflow|underflow"):
                entry([c.source, offset, count])
        c.stream.synchronize()
        self.assertEqual(c.source, torch.full_like(c.source, 0x19))

    @parametrize(
        "fault",
        (
            "short",
            "long",
            "bool_node",
            "bool_pointer",
            "negative_offset",
            "pointer_range",
            "offset_range",
            "count_range",
            "duplicate",
            "foreign",
            "kernel",
            "list",
        ),
    )
    def test_binding_admission(self, device, fault):
        c = self.capture(device)
        row = (c.memset, 0, 0, 1)
        if fault == "short":
            row = row[:-1]
        elif fault == "long":
            row += (0,)
        elif fault == "bool_node":
            row = (True, *row[1:])
        elif fault == "bool_pointer":
            row = (row[0], True, *row[2:])
        elif fault == "negative_offset":
            row = (*row[:2], -1, row[3])
        elif fault == "pointer_range":
            row = (row[0], 4, *row[2:])
        elif fault == "offset_range":
            row = (*row[:2], 2, row[3])
        elif fault == "count_range":
            row = (*row[:3], 2)
        elif fault == "foreign":
            other = self.capture(device)
            row = (other.memset, *row[1:])
        elif fault == "kernel":
            row = (c.node, *row[1:])
        rows = (row, row) if fault == "duplicate" else (row,)
        if fault == "list":
            rows = list(rows)
        with self.assertRaises((TypeError, ValueError, IndexError)):
            c.graph._prepare_kernel_replay_updates(
                ((c.node, 0, 0), (c.node, 1, 3)), 4, (), (), 2, memset_bindings=rows
            )

    @parametrize("format", ("wide", "2d"))
    def test_only_byte_nodes_are_admitted(self, device, format):
        c = self.capture(device, format)
        with self.assertRaisesRegex(ValueError, "one-dimensional byte memset"):
            c.graph._prepare_kernel_replay_updates(
                ((c.node, 0, 0), (c.node, 1, 3)),
                4,
                (),
                (),
                2,
                memset_bindings=((c.memset, 0, 0, 1),),
            )

    @parametrize("field", ("offset", "count"))
    def test_memset_cannot_reference_late_parameter_outputs(self, device, field):
        c = self.capture(device)
        row = (c.memset, 0, 2 if field == "offset" else 0, 2 if field == "count" else 1)
        batch = c.graph._prepare_kernel_replay_updates(
            ((c.node, 0, 0), (c.node, 1, 3)), 4, (), (), 3, memset_bindings=(row,)
        )
        late = ("parameter_v1", (("constant", 64, 32),), (0,))
        with self.assertRaisesRegex(
            ValueError, "Memset bindings require early numeric values"
        ):
            c.graph._make_boxed_replay(
                batch,
                c.stream,
                (c.kernel, c.source, c.output),
                3,
                (0,),
                ((torch.uint8, (256,), (1,)),),
                None,
                (0,),
                ((1, 2), (("boxed", 1), ("boxed", 2)), late),
            )

    def test_memset_destination_must_be_tensor_or_allocation(self, device):
        c = self.capture(device)
        batch = c.graph._prepare_kernel_replay_updates(
            ((c.node, 0, 0), (c.node, 1, 3)),
            4,
            (),
            (),
            2,
            memset_bindings=((c.memset, 1, 0, 1),),
        )
        with self.assertRaisesRegex(
            ValueError, "Memset pointers must name Tensor inputs or allocations"
        ):
            c.graph._make_boxed_replay(
                batch,
                c.stream,
                (c.kernel, c.source, c.output),
                3,
                (0,),
                ((torch.uint8, (256,), (1,)),),
                None,
                (0,),
                ((1, 2), (("boxed", 1), ("boxed", 2))),
            )


instantiate_device_type_tests(TestMemsetReplay, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
