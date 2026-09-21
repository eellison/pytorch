# Owner(s): ["module: inductor"]

from unittest import mock

import triton
import triton.language as tl

import torch
from torch._inductor.runtime._cudagraph import direct_host
from torch._inductor.runtime._cudagraph.api import (
    DirectHost,
    DirectTriton,
    InputContract,
    TensorInput,
)
from torch._inductor.runtime._cudagraph.direct_triton import DirectTritonDeclined
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@triton.jit(do_not_specialize=["delta"])
def increment(source, delta):
    offset = tl.arange(0, 8)
    value = tl.load(source + offset)
    tl.store(source + offset, value + delta)


OP = None
GRID = None
DELTA = None
WARMUP = False
TAIL = None


def tail(source):
    source.add_(10)
    return source


def host(box):
    (source,) = box
    box.clear()
    OP.run(source, DELTA, grid=GRID, warmup=WARMUP)
    OP.run(source, DELTA, grid=GRID, warmup=WARMUP)
    return (TAIL(source),)


class TestDirectTritonDeclines(TestCase):
    def contract(self):
        return InputContract(
            ("tensor",),
            (TensorInput(0, torch.float32, (8,), (1,)),),
            (),
            device_index=torch.cuda.current_device(),
        )

    @parametrize(
        "stage",
        ("zero_grid", "float_abi", "warmup", "jit_hook", "metadata", "config_hook"),
    )
    def test_decline_finishes_ordinary_once(self, device, stage):
        self.enterContext(torch.cuda.device(device))
        grid = (0,) if stage == "zero_grid" else (1,)
        delta = 1.5 if stage == "float_abi" else 1
        warmup = stage == "warmup"
        hook_calls = []

        def hook(source, *args, **kwargs):
            hook_calls.append(None)
            source.add_(100)

        kernel = increment
        if stage == "jit_hook":
            self.enterContext(mock.patch.object(increment, "pre_run_hooks", [hook]))
        elif stage == "metadata":
            self.enterContext(
                mock.patch.object(increment, "launch_metadata", lambda *args: {})
            )
        elif stage == "config_hook":
            config = triton.Config({}, pre_hook=lambda args: hook(args["source"]))
            kernel = triton.autotune(configs=[config], key=[])(increment)
        change = 10
        if stage not in ("zero_grid", "warmup"):
            change += 2 * delta
        if stage in ("jit_hook", "config_hook"):
            change += 200
        source = torch.arange(8, dtype=torch.float32, device=device)
        expected = source + change
        ordinary = source.clone()
        self.enterContext(
            mock.patch.dict(
                globals(), OP=kernel, GRID=grid, DELTA=delta, WARMUP=warmup, TAIL=tail
            )
        )
        (reference,) = host([ordinary])
        self.assertIs(reference, ordinary)
        self.assertEqual(reference, expected)
        hook_calls.clear()

        adapter = DirectTriton(kernel)
        self.addCleanup(adapter.close)
        tail_calls = mock.Mock(wraps=tail)
        self.enterContext(mock.patch.dict(globals(), OP=adapter, TAIL=tail_calls))
        runtime = DirectHost(host, self.contract())
        self.addCleanup(runtime.close)
        for index in range(2):
            box = [source]
            (result,) = runtime(box)
            self.assertEqual(tail_calls.call_count, index + 1)
            self.assertEqual(source, expected + index * change)
            self.assertIs(result, source)
            self.assertEqual(box, [])
            self.assertEqual(runtime.variants, [])
            self.assertIsNone(runtime.entry)
        self.assertEqual(
            len(hook_calls), 4 if stage in ("jit_hook", "config_hook") else 0
        )

    @parametrize("error_type", (RuntimeError, DirectTritonDeclined))
    def test_ordinary_error_propagates_without_retry(self, device, error_type):
        self.enterContext(torch.cuda.device(device))
        hook_calls = []

        def fail(source, *args, **kwargs):
            hook_calls.append(None)
            source.add_(1)
            raise error_type("ordinary JIT run failed")

        self.enterContext(mock.patch.object(increment, "pre_run_hooks", [fail]))
        adapter = DirectTriton(increment)
        self.addCleanup(adapter.close)
        tail_calls = mock.Mock(wraps=tail)
        self.enterContext(
            mock.patch.dict(
                globals(), OP=adapter, GRID=(1,), DELTA=1, WARMUP=False, TAIL=tail_calls
            )
        )
        runtime = DirectHost(host, self.contract())
        self.addCleanup(runtime.close)
        source = torch.zeros(8, device=device)
        for index in range(2):
            box = [source]
            with self.assertRaisesRegex(error_type, "ordinary JIT run failed"):
                runtime(box)
            self.assertEqual(source, torch.full_like(source, index + 1))
            self.assertEqual(len(hook_calls), index + 1)
            self.assertEqual(box, [])
            self.assertEqual(runtime.variants, [])
            self.assertIsNone(runtime.entry)
        tail_calls.assert_not_called()

    def test_new_observation_resets_decline(self, device):
        self.enterContext(torch.cuda.device(device))
        adapter = DirectTriton(increment)
        self.addCleanup(adapter.close)
        self.enterContext(
            mock.patch.dict(
                globals(), OP=adapter, GRID=(1,), DELTA=1, TAIL=lambda source: source
            )
        )
        with mock.patch.dict(globals(), WARMUP=True):
            runtime = DirectHost(host, self.contract())
            try:
                source = torch.zeros(8, device=device)
                (result,) = runtime([source])
                self.assertIs(result, source)
                self.assertEqual(result, torch.zeros_like(source))
                self.assertEqual(runtime.variants, [])
                self.assertIsNone(runtime.entry)
            finally:
                runtime.close()

        with mock.patch.dict(globals(), WARMUP=False):
            runtime = DirectHost(host, self.contract())
            self.addCleanup(runtime.close)
            with mock.patch.object(
                direct_host, "_observe_direct", wraps=direct_host._observe_direct
            ) as observe:
                sources = [torch.zeros(8, device=device) for _ in range(2)]
                self.assertNotEqual(sources[0].data_ptr(), sources[1].data_ptr())
                for source in sources:
                    box = [source]
                    (result,) = runtime(box)
                    self.assertIs(result, source)
                    self.assertEqual(result, torch.full_like(source, 2))
                    self.assertEqual(box, [])
                    self.assertEqual(len(runtime.variants), 1)
                self.assertEqual(observe.call_count, 1)


instantiate_device_type_tests(TestDirectTritonDeclines, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
