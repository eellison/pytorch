# Owner(s): ["module: inductor"]

import unittest
from unittest import mock

import torch
from torch._inductor.runtime._cudagraph.host_trace import (
    HostTraceReplay,
    prepare_host_trace,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@unittest.skipIf(
    not hasattr(torch._C, "_HostTraceRecorder")
    or not hasattr(torch._C, "_CUDAGraphCompiledEvaluation")
    or not hasattr(torch._C, "_CUDAGraphBoxedDispatch")
    or torch.version.cuda is None
    or tuple(map(int, torch.version.cuda.split("."))) < (12, 8),
    "requires NVIDIA CUDA 12.8 or later with host tracing",
)
class TestCudaHostTraceReplay(TestCase):
    def test_fresh_output_changes_between_empty_and_nonempty(self, device):
        def fn(x):
            y = torch.ops.aten.native_layer_norm(x, [128], None, None, 1e-5)[0]
            return y, torch.empty((x.shape[0] - 4,), device=device)

        replay = HostTraceReplay(fn)
        self.addCleanup(replay.close)
        # Wrapper storage sizing branches on zero extent; both traces then reuse.
        for rows, variants in ((8, 1), (4, 2), (12, 2), (4, 2)):
            x = torch.randn(rows, 128, device=device)
            y, extra = replay([x])
            self.assertEqual(y, fn(x)[0])
            self.assertEqual(extra.shape, (rows - 4,))
            self.assertEqual(len(replay.variants), variants)

    @parametrize("empty", (False, True))
    def test_fresh_output_without_a_kernel_use(self, device, empty):
        def fn(x):
            y = torch.ops.aten.native_layer_norm(x, [128], None, None, 1e-5)[0]
            extra = torch.empty(
                (0 if empty else x.shape[0],), device=device, dtype=torch.uint64
            )
            return y, extra

        replay = HostTraceReplay(fn)
        self.addCleanup(replay.close)
        retained = []
        for rows in (8, 4, 16):
            x = torch.randn(rows, 128, device=device)
            y, extra = replay([x])
            self.assertEqual(y, fn(x)[0])
            self.assertEqual(extra.shape, (0 if empty else rows,))
            self.assertEqual(extra.dtype, torch.uint64)
            if not empty:
                self.assertNotIn(
                    extra.data_ptr(), [value.data_ptr() for value, _ in retained]
                )
                extra.fill_(rows)
            retained.append((extra, rows))
        self.assertEqual(len(replay.variants), 1)
        replay.close()
        for extra, rows in retained:
            self.assertEqual(extra, torch.full_like(extra, rows))

    @dtypes(torch.float32, torch.bfloat16)
    def test_symbolic_allocations_and_grid(self, device, dtype):
        def fn(x, weight, bias):
            return torch.ops.aten.native_layer_norm(x, [4096], weight, bias, 1e-5)

        replay = HostTraceReplay(fn)
        retained = []
        try:
            for rows in (8, 4, 16, 4):
                args = (
                    torch.randn(rows, 4096, device=device, dtype=dtype),
                    torch.randn(4096, device=device, dtype=dtype),
                    torch.randn(4096, device=device, dtype=dtype),
                )
                expected = fn(*args)
                box = list(args)
                actual = replay(box)
                self.assertEqual(box, [])
                self.assertEqual(actual, expected)
                retained.append((actual, expected))
            self.assertEqual(len(replay.variants), 1)
            self.assertIsInstance(replay.entry, torch._C._CUDAGraphBoxedDispatch)
        finally:
            replay.close()
        for actual, expected in retained:
            self.assertEqual(actual, expected)

    @parametrize("offset", (0, 4, 32))
    def test_view_addresses_and_outputs(self, device, offset):
        def fn(x):
            y = torch.ops.aten.native_layer_norm(x, [128], None, None, 1e-5)[0]
            return x[1:], y[1:], y.transpose(0, 1)

        replay = HostTraceReplay(fn)
        try:
            for rows in (8, 4, 8):
                storage = torch.randn(offset + rows * 128, device=device)
                x = storage[offset:].view(rows, 128)
                outputs = replay([x])
                self.assertEqual(outputs, fn(x))
                self.assertEqual(outputs[0].data_ptr(), x[1:].data_ptr())
                self.assertTrue(torch._C._is_alias_of(outputs[1], outputs[2]))
                self.assertEqual(outputs[2].stride(), (1, 128))
            self.assertEqual(len(replay.variants), 1)
        finally:
            replay.close()

    def test_kernel_local_branch_retraces(self, device):
        def fn(x):
            y = torch.ops.aten.native_layer_norm(x, [128], None, None, 1e-5)[0]
            if x.shape[0] > 8:
                y = torch.ops.aten.native_layer_norm(y, [128], None, None, 1e-5)[0]
            return (y,)

        replay = HostTraceReplay(fn)
        retained = []
        try:
            for rows, variants in ((4, 1), (8, 1), (16, 2), (4, 2), (12, 2)):
                x = torch.randn(rows, 128, device=device)
                actual, expected = replay([x]), fn(x)
                self.assertEqual(actual, expected)
                retained.append((actual, expected))
                self.assertEqual(len(replay.variants), variants)
        finally:
            replay.close()
        for actual, expected in retained:
            self.assertEqual(actual, expected)

    def test_misses_execute_once_then_trace_and_hits_do_neither(self, device):
        from torch.cuda import _host_trace as ht

        def reference(x):
            y = torch.ops.aten.native_layer_norm(x, [128], None, None, 1e-5)[0]
            if x.shape[0] > 8:
                y = torch.ops.aten.native_layer_norm(y, [128], None, None, 1e-5)[0]
            return (y,)

        modes = []

        def fn(x):
            modes.append(torch._C._host_trace_tracing())
            x.add_(1)
            return reference(x)

        replay = HostTraceReplay(fn)
        self.addCleanup(replay.close)
        with mock.patch.object(ht, "trace", wraps=ht.trace) as trace:
            for rows, miss, variants in (
                (4, True, 1),
                (8, False, 1),
                (16, True, 2),
                (12, False, 2),
                (4, False, 2),
            ):
                modes.clear()
                trace.reset_mock()
                x = torch.randn(rows, 128, device=device)
                expected_input = x + 1
                expected = reference(expected_input)
                box = [x]
                self.assertEqual(replay(box), expected)
                self.assertEqual(x, expected_input)
                self.assertEqual(box, [])
                self.assertEqual(modes, [True] if miss else [])
                self.assertEqual(len(replay.variants), variants)
                if miss:
                    trace.assert_called_once()
                    self.assertIs(trace.call_args.args[0], fn)
                    self.assertIs(trace.call_args.args[1][0], x)
                    self.assertEqual(trace.call_args.kwargs, {"warm_up": False})
                else:
                    trace.assert_not_called()

    @parametrize("warmed", (False, True))
    @parametrize("kind", ("empty", "negative"))
    def test_declined_miss_returns_ordinary_result_once(self, device, warmed, kind):
        def reference(x):
            return torch.ops.aten.native_layer_norm(x, [128], None, None, 1e-5)[0]

        modes = []

        def fn(x):
            modes.append(torch._C._host_trace_tracing())
            return reference(x)

        replay = HostTraceReplay(fn)
        self.addCleanup(replay.close)
        example = torch.randn(8, 128, device=device)
        if warmed:
            replay([example])
        variants = len(replay.variants)
        for _ in range(2):
            value = example[:0] if kind == "empty" else torch._neg_view(example)
            box = [value]
            modes.clear()
            self.assertEqual(replay(box), (reference(value),))
            self.assertEqual(box, [])
            self.assertEqual(modes, [False])
            self.assertEqual(len(replay.variants), variants)
        modes.clear()
        fresh = torch.randn_like(example)
        self.assertEqual(replay([fresh]), (reference(fresh),))
        self.assertEqual(modes, [] if warmed else [True])
        self.assertEqual(len(replay.variants), 1)

    def test_unexpected_trace_error_is_not_suppressed(self, device):
        from torch.cuda import _host_trace as ht

        def fn(x):
            return x.add_(1)

        replay = HostTraceReplay(fn)
        self.addCleanup(replay.close)
        x = torch.zeros(8, device=device)
        box = [x]
        error = RuntimeError("trace failed")
        with mock.patch.object(ht, "trace", side_effect=error) as trace:
            with self.assertRaisesRegex(RuntimeError, "trace failed") as raised:
                replay(box)
        self.assertIs(raised.exception, error)
        trace.assert_called_once()
        self.assertEqual(x, torch.zeros_like(x))
        self.assertEqual(box, [x])
        self.assertEqual(replay.variants, [])

    def test_complex_input_view_keeps_byte_offsets(self, device):
        def fn(z):
            real = torch.view_as_real(z)
            return (torch.ops.aten.native_layer_norm(real, [2], None, None, 1e-5)[0],)

        replay = HostTraceReplay(fn)
        try:
            for rows, offset in ((8, 0), (4, 2), (8, 4)):
                storage = torch.randn(
                    offset + rows * 16, device=device, dtype=torch.complex64
                )
                z = storage[offset:].view(rows, 16)
                self.assertEqual(replay([z]), fn(z))
            self.assertEqual(len(replay.variants), 1)
        finally:
            replay.close()

    def test_parameter_inputs(self, device):
        def fn(x, weight):
            return torch.ops.aten.native_layer_norm(x, [128], weight, None, 1e-5)

        replay = HostTraceReplay(fn)
        try:
            for rows in (8, 4):
                x = torch.randn(rows, 128, device=device)
                weight = torch.nn.Parameter(
                    torch.randn(128, device=device), requires_grad=False
                )
                self.assertEqual(replay([x, weight]), fn(x, weight))
            self.assertEqual(len(replay.variants), 1)
        finally:
            replay.close()

    def test_trace_constants_are_closed_over(self, device):
        from torch.cuda import _host_trace as ht

        def fn(x, keep):
            return torch.ops.aten.native_layer_norm(x, [128], None, None, 1e-5)[0][
                :keep
            ]

        x = torch.randn(8, 128, device=device)
        tape = ht.trace(fn, (x, 2))
        with self.assertRaises(ht.Miss):
            prepare_host_trace(tape, (x, 3))
        variant = prepare_host_trace(tape, (x, 2))
        try:
            dispatch = torch._C._cuda_make_boxed_dispatch(
                ((variant.entry, variant.guard.registration),), lambda box: None
            )
            y = torch.randn_like(x)
            self.assertEqual(dispatch([y]), (fn(y, 2),))
            self.assertIsNone(dispatch([torch._neg_view(y)]))
            dispatch.close()
        finally:
            variant.close()

    @parametrize("operation", ("prepare", "close"))
    def test_reentrant_preparation_rejects_without_deadlock(self, device, operation):
        import threading

        def fn(x):
            if operation == "prepare":
                return replay([x])
            replay.close()

        replay = HostTraceReplay(fn)
        x = torch.empty(0, device=device)
        box = [x]
        errors = []

        def run():
            try:
                replay(box)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), "reentrant preparation deadlocked")
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("Host trace preparation is busy", str(errors[0]))
        self.assertEqual(box, [x])
        self.assertFalse(replay.closed)
        self.assertFalse(torch._C._host_trace_tracing())
        replay.close()
        self.assertTrue(replay.closed)

    def test_concurrent_build_and_close_reject_during_cold_trace(self, device):
        import threading

        from torch.cuda import _host_trace as ht

        entered = threading.Event()
        release = threading.Event()
        modes = []

        def reference(x):
            return torch.ops.aten.native_layer_norm(x, [128], None, None, 1e-5)[0]

        def fn(x):
            modes.append(torch._C._host_trace_tracing())
            entered.set()
            if not release.wait(timeout=30):
                raise RuntimeError("cold trace release timed out")
            return reference(x)

        x = torch.randn(8, 128, device=device)
        expected = reference(x)
        tape = ht.trace(reference, (x,), warm_up=False)
        torch.cuda.synchronize(device)
        replay = HostTraceReplay(fn)
        self.addCleanup(replay.close)
        outputs = []
        errors = []

        def run():
            try:
                with torch.cuda.device(device):
                    outputs.append(replay([x]))
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=30), "cold trace did not start")
            with self.assertRaisesRegex(RuntimeError, "Host trace preparation is busy"):
                replay.build_variant(tape, (x,))
            self.assertFalse(replay.closed)
            self.assertEqual(replay.variants, [])
            with self.assertRaisesRegex(RuntimeError, "Host trace preparation is busy"):
                replay.close()
            self.assertFalse(replay.closed)
            self.assertEqual(replay.variants, [])
        finally:
            release.set()
            worker.join(timeout=30)
        self.assertFalse(worker.is_alive(), "cold trace did not finish")
        self.assertEqual(errors, [])
        self.assertEqual(modes, [True])
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0], (expected,))
        self.assertEqual(len(replay.variants), 1)
        replay.close()
        self.assertTrue(replay.closed)
        self.assertEqual(outputs[0][0] + 1, expected + 1)

    def test_close_releases_capture_authority(self, device):
        import gc
        import weakref

        def fn(x, keep):
            y = torch.ops.aten.native_layer_norm(x, [128], None, None, 1e-5)[0]
            if x.shape[0] > 8:
                y = torch.ops.aten.native_layer_norm(y, [128], None, None, 1e-5)[0]
            return y[:keep]

        replay = HostTraceReplay(fn)
        retained = []
        try:
            for keep, rows in ((2, 4), (2, 16), (3, 4), (3, 16), (2, 4)):
                x = torch.randn(rows, 128, device=device)
                expected = fn(x, keep)
                actual = replay([x, keep])[0]
                self.assertEqual(actual, expected)
                retained.append((actual, expected))
            self.assertEqual(len(replay._families), 2)
            self.assertEqual(len(replay.variants), 4)
            tape_refs = [
                weakref.ref(variant.program.tape) for variant in replay.variants
            ]
            module_refs = [
                weakref.ref(call.module)
                for variant in replay.variants
                for call in variant.program.calls
            ]
            gc.collect()
            for owner in (*tape_refs, *module_refs):
                self.assertIsNotNone(owner())
            replay.close()
            replay.close()
            gc.collect()
            self.assertEqual(replay.variants, [])
            self.assertEqual(replay._families, [])
            for owner in (*tape_refs, *module_refs):
                self.assertIsNone(owner())
            for actual, expected in retained:
                self.assertEqual(actual, expected)
                self.assertEqual(actual + 1, expected + 1)
        finally:
            replay.close()


instantiate_device_type_tests(TestCudaHostTraceReplay, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
