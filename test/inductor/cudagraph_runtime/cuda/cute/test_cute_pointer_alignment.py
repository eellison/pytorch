"""Guard an ordinary valid CuTe view alignment without normalizing the input."""

import ctypes
import hashlib
import json
from pathlib import Path
import sys
from unittest import mock


from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver
import torch
from torch._inductor.runtime._cudagraph import replay
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe, DirectHost, InputContract, ObservedOrdinaryEntry, PythonEntry, SignaturePolicy, TensorInput,
)
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, PointerSource
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@cute.kernel
def copy_absolute(absolute: cute.Tensor, output: cute.Tensor):
    index = cute.arch.thread_idx()[0]
    output[index] = absolute[index] * 2 + 1


@cute.jit
def launch_absolute(absolute: cute.Tensor, output: cute.Tensor, stream: driver.CUstream):
    copy_absolute(absolute, output).launch(grid=(1, 1, 1), block=(8, 1, 1), smem=0, stream=stream)


def convert_arguments(absolute, output):
    return tuple(from_dlpack(tensor, assumed_align=16, use_32bit_stride=False)
                 for tensor in (absolute, output))


CUTE = None


def host(box):
    source, = box
    box.clear()
    absolute = source.as_strided((8,), (1,), storage_offset=0)
    output = torch.empty_strided((8,), (1,), dtype=source.dtype, device=source.device)
    CUTE(absolute, output)
    return (output,)


class TestCuTePointerAlignment(TestCase):
    def test_valid_aligned_view_from_shifted_input(self, device):
        with torch.cuda.device(device):
            owner = ObservedOrdinaryEntry(PythonEntry(launch_absolute), copy_absolute,
                policy=SignaturePolicy(32, 64, 16, "stream"), conversion=convert_arguments)
            self.addCleanup(owner.close)
            self.enterContext(mock.patch.dict(globals(), {"CUTE": DirectCuTe(owner)}))
            contract = InputContract(("tensor",), (TensorInput(0, torch.float32, (8,), (1,)),), (),
                                     device_index=torch.cuda.current_device())
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            captures = []
            original_make_replay = replay._make_replay

            def capture(*args, **kwargs):
                _, _, allocations, _, copies, calls, launches, _, _ = args
                self.assertEqual(len(allocations), 1)
                self.assertEqual(copies, ())
                self.assertEqual(len(launches), 1)
                call, = calls
                pointer, = [field.source for field in call.fields
                            if type(field.source) is PointerSource and field.source.root == InputSource(0)]
                numeric = kwargs["numeric"]
                offset = numeric.values[numeric.add(pointer.byte_offset)]
                self.assertEqual(offset, -4 * kwargs["capture_inputs"][0].storage_offset())
                captures.append(offset)
                return original_make_replay(*args, **kwargs)

            self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=capture))
            samples = []
            for step, offset in enumerate((0, 1, 4, 5)):
                storage = torch.arange(32, dtype=torch.float32, device=device) + step * 100
                self.assertEqual(storage.data_ptr() % 16, 0)
                source = storage[offset:offset + 8]
                absolute = source.as_strided((8,), (1,), storage_offset=0)
                self.assertEqual(absolute.data_ptr(), storage.data_ptr())
                self.assertEqual(absolute.data_ptr() % 16, 0)
                samples.append((source, storage))
            self.assertEqual(len({source.data_ptr() for source, _ in samples}), 4)
            held = []
            for step, (source, storage) in enumerate(samples):
                box, frames = [source], []

                def profile(frame, event, result):
                    if event == "call":
                        frames.append((frame.f_code.co_filename, frame.f_code.co_name))

                if step == 0:
                    actual, = runtime(box)
                else:
                    try:
                        sys.setprofile(profile)
                        actual, = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                expected = storage[:8] * 2 + 1
                self.assertEqual(actual, expected)
                ordinary, = host([source])
                self.assertEqual(ordinary, expected)
                self.assertEqual(len(runtime.variants), 1)
                held.append((actual, expected))
            variant, = runtime.variants
            call, = [event for event in variant.program.events if type(event) is CuTeCall]
            alignments = tuple(formal.data_alignment for formal in call.bound.module.artifact.formals
                               if formal.kind == "Tensor")
            self.assertEqual(alignments, (16, 16))
            self.assertEqual(owner._capture.calls, 1)
            self.assertEqual(captures, [0])
            guard = variant.guard
            self.assertEqual(guard.boxed_integer_indices, ())
            self.assertEqual(guard.boxed_pointer_indices, (0,))
            self.assertEqual(guard.boxed_storage_offset_indices, (0,))
            predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                        ctypes.POINTER(ctypes.c_double))(guard.function_address)
            base = alignments[0] * 64
            self.assertEqual(predicate((ctypes.c_int64 * 2)(base, 0), None), 1)
            self.assertEqual(predicate((ctypes.c_int64 * 2)(base + 4, 1), None), 1)
            self.assertEqual(predicate((ctypes.c_int64 * 2)(base + 4, 0), None), 0)
            self.assertEqual(predicate((ctypes.c_int64 * 2)(base, 1), None), 0)
            runtime.close()
            owner.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)
            print("CUTE_POINTER_ALIGNMENT_RESULT=" + json.dumps({


                "samples": 4, "offsets": [source.storage_offset() for source, _ in samples],
                "native_hits": 3, "variants": 1, "compilations": 1, "captures": captures,
                "ordinary_references": 4, "compiler_alignments": alignments,
                "held_outputs": len(held), "normalization_copies": 0,
                "synthetic_guard_acceptances": 2, "synthetic_guard_rejections": 2,
            }, sort_keys=True))


instantiate_device_type_tests(TestCuTePointerAlignment, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
