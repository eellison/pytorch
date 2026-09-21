# Owner(s): ["module: inductor"]

import sys
import unittest
from unittest import mock

import triton
from direct_example import example

import torch
from torch._inductor.runtime._cudagraph import replay
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe,
    DirectHost,
    DirectTriton,
    InputContract,
    IntegerRange,
    IntExpr,
    ObservedOrdinaryEntry,
    PythonEntry,
    SignaturePolicy,
    TensorInput,
)
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime._cudagraph.direct_cuda_host import DirectCudaHost
from torch._inductor.runtime._cudagraph.frontend import (
    DirectKernelCall,
    DirectPhysicalCall,
)
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


ADD = CUTE = SILU = None


def host(box):
    rows, source = box
    box.clear()
    temporary = torch.empty_like(source)
    affine = torch.empty_like(source)
    count = rows * 128
    ADD[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](
        source, temporary, count, BLOCK=128
    )
    CUTE(temporary, affine, rows)
    activated = SILU(affine)
    output = torch.empty_like(activated)
    ADD[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](
        activated, output, count, BLOCK=128
    )
    return (output,)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and _get_torch_cuda_version() >= (12, 8),
    "requires NVIDIA CUDA >= 12.8",
)
class TestMixedThreeFrontends(TestCase):
    @parametrize("input_offset", (0, 4))
    def test_one_graph_dynamic_replay(self, device, input_offset):
        from cuda.bindings import runtime as cudart

        with torch.cuda.device(device):
            owner = ObservedOrdinaryEntry(
                PythonEntry(example.launch_affine),
                example.affine,
                policy=SignaturePolicy(32, 64, 16, "stream"),
                conversion=example.convert_arguments,
            )
            add = DirectTriton(example.add_one)
            self.addCleanup(owner.close)
            self.addCleanup(add.close)
            self.enterContext(
                mock.patch.dict(
                    globals(),
                    ADD=add,
                    CUTE=DirectCuTe(owner),
                    SILU=DirectCudaHost(torch.nn.functional.silu),
                )
            )
            rows = IntExpr("boxed", 0)
            contract = InputContract(
                ("integer", "tensor"),
                (TensorInput(1, torch.float32, (rows, 128), (128, 1)),),
                (IntegerRange(0, 2, 127),),
                device_index=torch.cuda.current_device(),
            )
            runner = DirectHost(host, contract)
            self.addCleanup(runner.close)
            captures = []
            make_replay = replay._make_replay

            def capture(*args, **kwargs):
                graph, calls, launches = args[0], args[5], args[6]
                self.assertEqual(len(calls), 4)
                self.assertEqual(len(launches), 4)
                raw = graph.raw_cuda_graph()
                _, count = _check_cuda_bindings(cudart.cudaGraphGetNodes(raw))
                self.assertEqual(count, 4)
                nodes, _ = _check_cuda_bindings(cudart.cudaGraphGetNodes(raw, count))
                self.assertEqual(
                    {int(node) for node in nodes},
                    {launch.after[3][0][0] for launch in launches},
                )
                captures.append(raw)
                return make_replay(*args, **kwargs)

            self.enterContext(mock.patch.object(replay, "_make_replay", capture))
            sequence = (5, 7, 35, 96, 5)
            samples = []
            for rows in sequence:
                storage = torch.randn(rows * 128 + input_offset, device=device)
                source = storage[input_offset:].view(rows, 128)
                samples.append(source)
            self.assertEqual(
                len({source.data_ptr() for source in samples}), len(samples)
            )
            held, frames = [], []

            def profile(frame, event, arg):
                if event == "call":
                    frames.append((frame.f_code.co_filename, frame.f_code.co_name))

            for index, (rows, source) in enumerate(zip(sequence, samples, strict=True)):
                box = [rows, source]
                if index == 0:
                    (output,) = runner(box)
                    self.assertIn(
                        type(runner.entry),
                        (
                            torch._C._CUDAGraphBoxedReplay,
                            torch._C._CUDAGraphBoxedDispatch,
                        ),
                    )
                else:
                    try:
                        sys.setprofile(profile)
                        (output,) = runner.entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                expected = torch.nn.functional.silu((source + 1) * 2 + rows) + 1
                self.assertEqual(output, expected)
                self.assertEqual(len(runner.variants), 1)
                held.append((output, expected))

            self.assertEqual(len(captures), 1)
            calls = tuple(
                event
                for event in runner.variants[0].program.events
                if type(event) in (DirectKernelCall, CuTeCall, DirectPhysicalCall)
            )
            self.assertEqual(
                tuple(type(call) for call in calls),
                (DirectKernelCall, CuTeCall, DirectPhysicalCall, DirectKernelCall),
            )
            self.assertEqual(owner._capture.calls, 1)
            self.assertEqual(len({output.data_ptr() for output, _ in held}), len(held))
            runner.close()
            add.close()
            owner.close()
            for output, expected in held:
                self.assertEqual(output, expected)


instantiate_device_type_tests(TestMixedThreeFrontends, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
