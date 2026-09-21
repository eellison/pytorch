"""Check the ordinary owner's actual signature projection across row counts."""

import inspect
import json

from torch._inductor.runtime._cudagraph._sdk import activate

activate()

from cuda.bindings import driver
from cutlass import cute
from host import convert_arguments
import torch
from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import _metadata
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


def invocation(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream):
    pass


class TestCompactMetadata(TestCase):
    def test_selected_signature(self, device):
        snapshots = []
        for rows in (128, 256):
            tensors = tuple(torch.empty_strided(shape, stride, dtype=torch.float16, device=device)
                            for shape, stride in (
                                ((1, rows, 128), (0, 128, 1)),
                                ((1, 128, 128), (16384, 128, 1)),
                                ((1, rows, 128), (rows * 128, 128, 1)),
                            ))
            values = convert_arguments(*tensors)
            self.assertEqual([value.data_ptr for value in values], [tensor.data_ptr() for tensor in tensors])
            stream = driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
            snapshot = _metadata("compact_probe", inspect.signature(invocation), (*values, stream), {})
            snapshots.append(snapshot)
            print("COMPACT_METADATA=" + json.dumps({"rows": rows, "snapshot": repr(snapshot)},
                                                   sort_keys=True), flush=True)
        self.assertEqual(snapshots[0], snapshots[1])


instantiate_device_type_tests(TestCompactMetadata, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
