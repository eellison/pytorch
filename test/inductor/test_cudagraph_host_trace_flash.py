# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._inductor.runtime._cudagraph.direct_hosttrace import _function_by_symbol
from torch._inductor.runtime._cudagraph.host_trace import HostTraceReplay
from torch.testing._internal.common_cuda import (
    _get_torch_cuda_version,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


_HOST_TRACE_SUPPORTED = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and _get_torch_cuda_version() >= (12, 8)
    and hasattr(torch._C, "_HostTraceRecorder")
    and hasattr(torch._C, "_CUDAGraphBoxedDispatch")
    and hasattr(torch._C, "_CUDAGraphCompiledEvaluation")
)


def flash(q, k, v, *, causal=False, scale=0.125):
    return torch.ops.aten._flash_attention_forward.default(
        q, k, v, None, None, 0, 0, 0.0, causal, False, scale=scale
    )


@unittest.skipUnless(
    _HOST_TRACE_SUPPORTED, "requires NVIDIA CUDA >= 12.8 and native host replay"
)
@unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "flash attention not supported")
class TestCudaHostTraceFlashReplay(TestCase):
    def _qkv(
        self,
        device,
        shape,
        *,
        heads=32,
        kv_heads=None,
        head_dim=128,
        dtype=torch.bfloat16,
        offset=0,
    ):
        batch, query, cache = shape
        kv_heads = heads if kv_heads is None else kv_heads

        def make(sequence, count):
            size = batch * sequence * count * head_dim
            storage = torch.randn(size + offset, device=device, dtype=dtype)
            return storage[offset:].view(batch, sequence, count, head_dim)

        return make(query, heads), make(cache, kv_heads), make(cache, kv_heads)

    def _start(self, device, shape, *, fn=None, causal=False, scale=0.125, **kwargs):
        if fn is None:
            replay = HostTraceReplay(
                lambda q, k, v: flash(q, k, v, causal=causal, scale=scale)
            )
        else:
            replay = HostTraceReplay(fn)
        self.addCleanup(replay.close)
        args = self._qkv(device, shape, **kwargs)
        self._check(replay, args)
        self.assertEqual(len(replay.variants), 1)
        self.assertIsInstance(replay.entry, torch._C._CUDAGraphBoxedDispatch)
        return replay, args

    def _check(self, replay, args):
        expected = replay.fn(*args)
        box = list(args)
        actual = replay(box)
        self.assertEqual(box, [])
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(actual[:2], expected[:2], atol=0, rtol=0)
        # The no-dropout auxiliary tensors are uninitialized, but still owned outputs.
        for output, reference in zip(actual, expected):
            self.assertEqual(output.shape, reference.shape)
            self.assertEqual(output.stride(), reference.stride())
            self.assertEqual(output.dtype, reference.dtype)
            self.assertEqual(output.device, reference.device)
        return actual

    def _selector(self, replay, args):
        tape = replay.variants[0].program.tape
        self.assertEqual(
            [record["fn"] for record in tape.opaque], ["num_splits_heuristic"]
        )
        record = tape.opaque[0]
        substitutions = {}
        batches = []
        for source in tape.inputs:
            tensor = args[source.position]
            symbolic = (*source.sizes, *source.strides, source.offset)
            concrete = (*tensor.shape, *tensor.stride(), tensor.storage_offset())
            substitutions.update(
                (value.node._expr, actual)
                for value, actual in zip(symbolic, concrete)
                if isinstance(value, torch.SymInt)
            )
            batches.append(source.sizes[0].node._expr)

        def arguments(batch):
            values = substitutions | dict.fromkeys(batches, batch)
            result = []
            for value in record["args"]:
                expression = (
                    value.node._expr if isinstance(value, torch.SymInt) else value
                )
                if hasattr(expression, "xreplace"):
                    expression = expression.xreplace(values)
                    self.assertFalse(expression.free_symbols)
                result.append(int(expression))
            return result

        return record, arguments

    def test_exact_physical_launch(self, device):
        replay, _ = self._start(device, (4, 512, 512))
        program = replay.variants[0].program
        calls = program.calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].module.parameter_layout), 1)
        self.assertTrue(
            {"pointer", "i32", "i64"} <= {field.kind for field in calls[0].fields}
        )
        launch = program.tape.launches[0]
        self.assertEqual(
            calls[0].module.function, _function_by_symbol(int(launch["func"]))
        )
        self.assertEqual(
            calls[0].module.parameter_layout, tuple(launch["param_layout"])
        )

    @parametrize(
        "shape",
        [
            (4, 512, 512),
            (1, 512, 512),
            (8, 512, 512),
            (2, 1024, 256),
            (4, 384, 1024),
            (3, 512, 64),
        ],
    )
    def test_shapes_and_fresh_addresses(self, device, shape):
        replay, original = self._start(device, (4, 512, 512))
        args = self._qkv(device, shape)
        self.assertTrue(
            all(a.data_ptr() != b.data_ptr() for a, b in zip(args, original))
        )
        self._check(replay, args)
        self._check(replay, self._qkv(device, shape))
        self.assertEqual(len(replay.variants), 1)

    @parametrize("change", ("kernel_branch", "dtype"))
    def test_guard_miss_adds_reusable_variant(self, device, change):
        replay, original = self._start(device, (4, 512, 512))
        shape = (4, 500, 512) if change == "kernel_branch" else (4, 512, 512)
        dtype = torch.float16 if change == "dtype" else torch.bfloat16
        self._check(replay, self._qkv(device, shape, dtype=dtype))
        self.assertEqual(len(replay.variants), 2)
        self._check(replay, self._qkv(device, shape, dtype=dtype))
        self._check(replay, original)
        self.assertEqual(len(replay.variants), 2)

    def test_host_causal_branch_retraces(self, device):
        def fn(q, k, v):
            return flash(q, k, v, causal=q.shape[1] == k.shape[1])

        replay, original = self._start(device, (4, 512, 512), fn=fn)
        self._check(replay, self._qkv(device, (4, 512, 1024)))
        self.assertEqual(len(replay.variants), 2)
        self._check(replay, self._qkv(device, (4, 512, 1024)))
        self._check(replay, original)
        self.assertEqual(len(replay.variants), 2)

    @parametrize("offset", (8, 64))
    def test_aligned_input_offsets_reuse(self, device, offset):
        replay, _ = self._start(device, (4, 512, 512))
        self._check(replay, self._qkv(device, (2, 512, 512), offset=offset))
        self.assertEqual(len(replay.variants), 1)

    @parametrize(
        "initial_dim,next_dim,variants", [(128, 128, 1), (112, 120, 1), (112, 128, 2)]
    )
    def test_scale_from_dynamic_head_dimension(
        self, device, initial_dim, next_dim, variants
    ):
        replay, _ = self._start(device, (4, 512, 512), scale=None, head_dim=initial_dim)
        tape = replay.variants[0].program.tape
        self.assertTrue(
            any(
                field["kind"] == "f32"
                and isinstance(field["value"], torch.SymFloat)
                and bool(field["value"].node._expr.free_symbols)
                for launch in tape.launches
                for field in launch["params"]
            )
        )
        self._check(replay, self._qkv(device, (2, 512, 512), head_dim=next_dim))
        self.assertEqual(len(replay.variants), variants)
        self._check(replay, self._qkv(device, (2, 512, 512), head_dim=next_dim))
        self.assertEqual(len(replay.variants), variants)

    def test_split_selector_boundary_retraces_once(self, device):
        replay, original = self._start(device, (1, 1, 8192), heads=8)
        record, arguments = self._selector(replay, original)
        self.assertGreater(record["expected"], 1)
        calls = replay.variants[0].program.calls
        self.assertEqual(len(calls), 2)
        self.assertEqual(record["call"](arguments(1)), record["expected"])
        batch = next(
            (
                batch
                for batch in range(2, 65)
                if record["call"](arguments(batch)) != record["expected"]
            ),
            None,
        )
        self.assertIsNotNone(
            batch, "expected a split-selection boundary in the batch sweep"
        )
        expected = record["call"](arguments(batch))
        self._check(replay, self._qkv(device, (batch, 1, 8192), heads=8))
        self.assertEqual(len(replay.variants), 2)
        selected = replay.variants[-1].program.tape.opaque[0]["expected"]
        self.assertEqual(selected, expected)
        self._check(replay, self._qkv(device, (batch, 1, 8192), heads=8))
        self._check(replay, original)
        self.assertEqual(len(replay.variants), 2)

    def test_split_selector_exact_wave_tie(self, device):
        replay, original = self._start(device, (1, 1, 4096), heads=8)
        record, arguments = self._selector(replay, original)
        tie = None
        for batch in range(2, 65):
            values = arguments(batch)
            selected = record["call"](values)
            if values[0] * selected % values[1] == 0:
                tie = batch, selected
                break
        if tie is None:
            self.skipTest(
                "no exact selected wave-count tie in this device's bounded batch sweep"
            )
        batch, selected = tie
        expected_variants = 1 + (selected != record["expected"])
        self._check(replay, self._qkv(device, (batch, 1, 4096), heads=8))
        self.assertEqual(len(replay.variants), expected_variants)
        self._check(replay, self._qkv(device, (batch, 1, 4096), heads=8))
        self._check(replay, original)
        self.assertEqual(len(replay.variants), expected_variants)

    def test_auxiliary_outputs_are_fresh_and_survive_close(self, device):
        replay, _ = self._start(device, (4, 512, 512))
        first = self._check(replay, self._qkv(device, (2, 512, 512)))
        saved = tuple(output.clone() for output in first[:2])
        for output in first[2:]:
            output.fill_(7)
        second = self._check(replay, self._qkv(device, (1, 512, 512)))
        self.assertEqual(len(replay.variants), 1)
        for previous, current in zip(first, second):
            self.assertIsNot(previous, current)
            if previous.numel() and current.numel():
                self.assertFalse(torch._C._is_alias_of(previous, current))
        replay.close()
        self.assertEqual(first[:2], saved, atol=0, rtol=0)
        for output in first[2:]:
            self.assertEqual(output, torch.full_like(output, 7))

    @parametrize("shape", [(1, 512, 512), (8, 512, 512), (2, 1024, 256)])
    def test_agrees_with_interim_replay(self, device, shape):
        from torch.cuda import _host_trace

        replay, original = self._start(device, (4, 512, 512))
        tape = replay.variants[0].program.tape
        interim = _host_trace.build(tape, replay.fn, original)
        args = self._qkv(device, shape)
        native = self._check(replay, args)
        self.assertEqual(native[:2], interim.replay(args)[:2], atol=0, rtol=0)
        self.assertEqual(len(replay.variants), 1)


instantiate_device_type_tests(TestCudaHostTraceFlashReplay, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
