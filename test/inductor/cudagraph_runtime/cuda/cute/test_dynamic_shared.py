# Owner(s): ["module: inductor"]
"""CuTe shared-byte recipes reach native replay with fixed block and cluster."""

import ctypes
import json
import sys
from unittest import mock

from torch._inductor.runtime._cudagraph import _sdk


_sdk.activate()

import cutlass
import cutlass.utils
from cuda.bindings import driver
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

import torch
from torch._inductor.runtime._cudagraph import replay
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.provider import (
    _constant_consumer,
)
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe,
    DirectHost,
    InputContract,
    IntegerRange,
    ObservedOrdinaryEntry,
    PythonEntry,
    SignaturePolicy,
    TensorInput,
)
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@cute.kernel
def shared_transform(source: cute.Tensor, destination: cute.Tensor):
    index = cute.arch.thread_idx()[0]
    allocator = cutlass.utils.SmemAllocator()
    shared = allocator.allocate_tensor(
        element_type=cutlass.Float32, layout=cute.make_layout(32), byte_alignment=16
    )
    shared[index] = source[index]
    cute.arch.sync_threads()
    destination[index] = shared[(index + 1) % 32] * 2 + 1
    if index == 0:
        destination[32] = cutlass.Float32(cute.arch.dynamic_smem_size())


@cute.jit
def launch_shared(
    source: cute.Tensor,
    destination: cute.Tensor,
    units: cutlass.Int32,
    stream: driver.CUstream,
):
    shared_transform(source, destination).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
        smem=units * cutlass.Int32(1024),
        stream=stream,
    )


def convert_arguments(source, destination, units):
    return (
        from_dlpack(source, assumed_align=4, use_32bit_stride=False),
        from_dlpack(destination, assumed_align=4, use_32bit_stride=False),
        cutlass.Int32(units),
    )


CUTE = None


def host(box):
    units, source, destination = box
    box.clear()
    CUTE(source, destination, units)
    return (destination,)


class TestDynamicShared(TestCase):
    def test_symbolic_request_and_resource_only_hit(self, device):
        with torch.cuda.device(device):
            owner = ObservedOrdinaryEntry(
                PythonEntry(launch_shared),
                shared_transform,
                policy=SignaturePolicy(32, 64, 4, "stream"),
                conversion=convert_arguments,
            )
            self.addCleanup(owner.close)
            self.enterContext(mock.patch.dict(globals(), {"CUTE": DirectCuTe(owner)}))
            contract = InputContract(
                ("integer", "tensor", "tensor"),
                (
                    TensorInput(1, torch.float32, (32,), (1,)),
                    TensorInput(2, torch.float32, (33,), (1,)),
                ),
                (IntegerRange(0, 1, 512),),
                device_index=torch.cuda.current_device(),
            )
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            captures = []
            make_replay = replay._make_replay

            def capture(*args, **kwargs):
                graph, _, allocations, _, copies, calls, launches, _, _ = args
                self.assertEqual(allocations, ())
                self.assertEqual(copies, ())
                (call,) = calls
                (launch,) = launches
                self.assertIsNone(call.module.shared)
                self.assertIs(type(call.shared), IntExpr)
                numeric = kwargs["numeric"]
                actual = numeric.prepared_value(call.shared)
                self.assertEqual(actual, 1024)
                node = launch.after[3][0][0]
                snapshot = graph._inspect_captured_kernel_nodes((node,))[3][0]
                self.assertEqual(snapshot[4], (1, 1, 1))
                self.assertEqual(snapshot[5], (32, 1, 1))
                self.assertEqual(snapshot[6], actual)
                captures.append(
                    {
                        "shared": actual,
                        "block": list(snapshot[5]),
                        "grid": list(snapshot[4]),
                    }
                )
                return make_replay(*args, **kwargs)

            self.enterContext(
                mock.patch.object(replay, "_make_replay", side_effect=capture)
            )
            first_source = torch.arange(32, dtype=torch.float32, device=device)
            first_destination = torch.empty(33, dtype=torch.float32, device=device)
            samples = [
                (1, first_source, first_destination),
                (2, first_source, first_destination),
            ]
            for step, units in enumerate((32, 4, 1), 1):
                samples.append(
                    (
                        units,
                        torch.arange(32, dtype=torch.float32, device=device)
                        + step * 100,
                        torch.empty(33, dtype=torch.float32, device=device),
                    )
                )
            held = []
            for step, (units, source, destination) in enumerate(samples):
                box, frames = [units, source, destination], []

                def profile(frame, event, result):
                    if event == "call":
                        frames.append((frame.f_code.co_filename, frame.f_code.co_name))

                if step == 0:
                    (actual,) = runtime(box)
                else:
                    try:
                        sys.setprofile(profile)
                        (actual,) = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                self.assertIs(actual, destination)
                expected = torch.cat(
                    (
                        torch.roll(source, -1) * 2 + 1,
                        torch.tensor(
                            [units * 1024], dtype=torch.float32, device=device
                        ),
                    )
                )
                self.assertEqual(actual, expected)
                reference_destination = torch.empty_like(destination)
                (ordinary,) = host([units, source, reference_destination])
                self.assertEqual(ordinary, expected)
                if step:
                    held.append((actual, expected))
            self.assertIs(samples[0][1], samples[1][1])
            self.assertIs(samples[0][2], samples[1][2])
            self.assertEqual(len({source.data_ptr() for _, source, _ in samples}), 4)
            self.assertEqual(
                len({destination.data_ptr() for _, _, destination in samples}), 4
            )
            (variant,) = runtime.variants
            (call,) = [
                event for event in variant.program.events if type(event) is CuTeCall
            ]
            self.assertIsNotNone(call.bound.shared)
            self.assertIsNone(call.bound.module.shared)
            self.assertEqual(call.bound.module.block, (32, 1, 1))
            self.assertIsNone(call.bound.module.cluster)
            requirement = _constant_consumer(
                call.bound.module.artifact, call.bound.module.site, "kernel_smem", 0
            )
            self.assertGreater(requirement, 0)
            self.assertLessEqual(requirement, 1024)
            maximum = call.bound.module.max_dynamic_shared
            self.assertGreaterEqual(maximum, 32768)
            guard = variant.guard
            self.assertEqual(guard.boxed_integer_indices, (0,))
            self.assertEqual(guard.boxed_pointer_indices, ())
            self.assertEqual(guard.boxed_storage_offset_indices, ())
            predicate = ctypes.CFUNCTYPE(
                ctypes.c_int8,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_double),
            )(guard.function_address)
            self.assertEqual(predicate((ctypes.c_int64 * 1)(32), None), 1)
            over_capacity = maximum // 1024 + 1
            self.assertLessEqual(over_capacity, 512)
            self.assertEqual(predicate((ctypes.c_int64 * 1)(over_capacity), None), 0)
            self.assertEqual(owner._capture.calls, 1)
            self.assertEqual(len(captures), 1)
            runtime.close()
            owner.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)
            print(
                "DYNAMIC_SHARED_RESULT="
                + json.dumps(
                    {
                        "samples": 5,
                        "requests": [units * 1024 for units, _, _ in samples],
                        "native_hits": 4,
                        "resource_only_hits": 1,
                        "variants": 1,
                        "ordinary_references": 5,
                        "compilations": 1,
                        "captures": captures,
                        "compiler_kernel_smem": requirement,
                        "loaded_capacity": maximum,
                        "held_outputs": len(held),
                        "capacity_guard_rejections": 1,
                    },
                    sort_keys=True,
                )
            )


instantiate_device_type_tests(TestDynamicShared, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
