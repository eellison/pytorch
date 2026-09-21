# Owner(s): ["module: inductor"]
import gc
import math
import struct
import unittest
import weakref
from types import SimpleNamespace

from cudagraph_runtime.cpu.test_host_evaluation import _CallOwner
from test_cudagraph_compiled_registration import store_scalar

import torch
from torch._inductor.runtime.cudagraph_arg_mapping import IntegerInput, IntExpr
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.cudagraph_compiled_evaluation import compile_evaluation
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfRocm,
    TestCase,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


@requires_cuda_and_triton
@unittest.skipIf(_get_torch_cuda_version() < (12, 8), "requires CUDA 12.8 or newer")
@skipIfRocm
class TestHostEvaluationRegistration(TestCase):
    def capture(self, device, owner, *, compiled=True, floating=False):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            output = torch.empty(1, dtype=torch.int64, device=device)
            binary = store_scalar[(1,)](-(1 << 40), output)
            stream.synchronize()
            records = SimpleNamespace(
                input_names=("output", "size", "divisor"),
                integer_inputs=(IntegerInput("size", 1), IntegerInput("divisor", 2)),
            )
            numeric = _NumericProgram(records, (output, 24, 8))
            value = IntExpr(
                "call",
                (owner.address, owner),
                (IntExpr("boxed", 1), IntExpr("boxed", 2)),
            )
            if floating:
                scale = IntExpr(
                    "fdiv",
                    args=(
                        IntExpr("fconst", struct.unpack("q", struct.pack("d", 1.0))[0]),
                        IntExpr("fsqrt", args=(IntExpr("ffromint", args=(value,)),)),
                    ),
                )
                rounded = IntExpr("fround32", args=(scale,))
                log2e = struct.unpack("q", struct.pack("d", math.log2(math.e)))[0]
                value = IntExpr(
                    "ftobits32",
                    args=(IntExpr("fmul", args=(rounded, IntExpr("fconst", log2e))),),
                )
            value_index = numeric.add(value)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                store_scalar[(1,)](-(1 << 40), output)
                frontier = torch._C._cuda_get_capture_frontier(stream.cuda_stream)
                self.assertEqual(len(frontier[3]), 1)
                node = frontier[3][0][0]
            graph.instantiate()
            batch = graph._prepare_kernel_replay_updates(
                ((node, 1, 0),),
                3,
                ((node, 0, 8, value_index),),
                (),
                len(numeric.instructions),
            )
            registration = None
            if compiled:
                evaluation = compile_evaluation(numeric, pointer_count=3)
                registration = torch._C._CUDAGraphCompiledEvaluation(
                    **evaluation.registration_kwargs
                )
            entry = graph._make_boxed_replay(
                batch,
                stream,
                (binary, output),
                3,
                (0,),
                (),
                None,
                (("input", 0),),
                ((1, 2), tuple(numeric.instructions)),
                compiled_evaluation=registration,
            )
        self.addCleanup(entry.close)
        self.addCleanup(stream.synchronize)
        return entry, stream

    @parametrize("compiled", (False, True))
    @parametrize("floating", (False, True))
    def test_fresh_callback_values_and_float_bits(self, device, compiled, floating):
        owner = _CallOwner()
        entry, stream = self.capture(
            device, owner, compiled=compiled, floating=floating
        )
        held = []
        preparation_calls = owner.preparation_calls
        with torch.cuda.stream(stream):
            for size, divisor in ((24, 8), (1024, 32), (65, 5)):
                output = torch.full((1,), -1, dtype=torch.int64, device=device)
                (result,) = entry([output, size, divisor])
                expected = size // divisor + 7
                if floating:
                    scale = struct.unpack(
                        "f", struct.pack("f", 1.0 / math.sqrt(expected))
                    )[0]
                    expected = struct.unpack(
                        "i", struct.pack("f", scale * math.log2(math.e))
                    )[0]
                self.assertEqual(result, torch.full_like(output, expected))
                self.assertIs(result, output)
                held.append(result)
        self.assertEqual(owner.preparation_calls, preparation_calls)

    def test_callback_exception_precedes_submission_and_does_not_poison_entry(
        self, device
    ):
        owner = _CallOwner()
        entry, stream = self.capture(device, owner)
        with torch.cuda.stream(stream):
            output = torch.full((1,), -1, dtype=torch.int64, device=device)
            box = [output, 10, 0]
            with self.assertRaisesRegex(ValueError, "evaluation failed with status 5"):
                entry(box)
            self.assertEqual(box, [output, 10, 0])
            stream.synchronize()
            self.assertEqual(output, torch.full_like(output, -1))
            (result,) = entry([output, 24, 8])
            self.assertEqual(result, torch.full_like(output, 10))

    @parametrize("compiled", (False, True))
    def test_callback_owner_retained_until_close(self, device, compiled):
        owner = _CallOwner()
        reference = weakref.ref(owner)
        entry, stream = self.capture(device, owner, compiled=compiled)
        del owner
        gc.collect()
        self.assertIsNotNone(reference())
        with torch.cuda.stream(stream):
            output = torch.full((1,), -1, dtype=torch.int64, device=device)
            (result,) = entry([output, 24, 8])
            self.assertEqual(result, torch.full_like(output, 10))
        entry.close()
        gc.collect()
        self.assertIsNone(reference())


instantiate_device_type_tests(
    TestHostEvaluationRegistration, globals(), only_for="cuda"
)

if __name__ == "__main__":
    run_tests()
