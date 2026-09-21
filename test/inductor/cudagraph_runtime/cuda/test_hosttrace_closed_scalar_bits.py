# Owner(s): ["module: inductor"]

import math
import struct

import torch
from torch._inductor.runtime._cudagraph.direct_hosttrace import HostTraceReplay

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def closed_scalar(kind, alternate):
    bits = (
        0x7FF8000000000001 + int(alternate)
        if kind == "nan_payload"
        else int(alternate) << 63
    )
    value = struct.unpack("!d", struct.pack("!Q", bits))[0]
    if kind == "complex_real":
        return complex(value, 1.0)
    if kind == "complex_imag":
        return complex(1.0, value)
    return value


class TestHostTraceClosedScalarBits(TestCase):
    @parametrize("kind", ("float", "complex_real", "complex_imag", "nan_payload"))
    def test_closed_bits_and_shape_misses(self, device, kind):
        modes = []

        def fn(x, scalar):
            modes.append(torch._C._host_trace_tracing())
            y = torch.ops.aten.native_layer_norm(x, [128], None, None, 1e-5)[0]
            if x.shape[0] >= 8:
                y = torch.ops.aten.native_layer_norm(y, [128], None, None, 1e-5)[0]
            if kind == "nan_payload":
                transpose = struct.unpack("!Q", struct.pack("!d", scalar))[0] & 1 == 0
            else:
                value = scalar.imag if kind == "complex_imag" else scalar.real
                transpose = math.copysign(1.0, value) < 0
            return y.t() if transpose else y

        example = torch.randn(4, 128, device=device)
        runtime = HostTraceReplay(
            fn, (example, closed_scalar(kind, False)), warm_up=False
        )
        self.addCleanup(runtime.close)
        inputs = [example]
        held = []
        for rows, alternate, expected_modes, variants in (
            (4, False, [], 1),
            (4, True, [True], 2),
            (12, True, [True], 3),
            (12, False, [True], 4),
            (12, False, [], 4),
            (4, False, [], 4),
            (4, True, [], 4),
            (4, False, [], 4),
        ):
            x = torch.randn(rows, 128, device=device)
            self.assertNotIn(x.data_ptr(), [value.data_ptr() for value in inputs])
            inputs.append(x)
            expected = torch.nn.functional.layer_norm(x, (128,))
            if rows >= 8:
                expected = torch.nn.functional.layer_norm(expected, (128,))
            if alternate:
                expected = expected.t()
            modes.clear()
            misses = runtime.misses
            actual = runtime(x, closed_scalar(kind, alternate))
            detail = (
                f"kind={kind}, rows={rows}, alternate={alternate}, "
                f"modes={modes}, misses={runtime.misses - misses}, "
                f"variants={len(runtime.variants)}"
            )
            self.assertEqual(actual.shape, expected.shape, msg=detail)
            self.assertEqual(actual, expected, msg=detail)
            self.assertEqual(modes, expected_modes, msg=detail)
            self.assertEqual(runtime.misses - misses, int(bool(expected_modes)))
            self.assertEqual(len(runtime.variants), variants)
            held.append((actual, expected))
        self.assertEqual((runtime.calls, runtime.misses, runtime.served), (8, 3, 8))
        self.assertEqual(runtime.ordinary, 0)
        runtime.close()
        for actual, expected in held:
            self.assertEqual(actual, expected)


instantiate_device_type_tests(TestHostTraceClosedScalarBits, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
