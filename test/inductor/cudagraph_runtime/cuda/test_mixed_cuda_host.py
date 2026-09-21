# Owner(s): ["module: inductor"]

import struct
import sys
import unittest
from unittest import mock

import triton
import triton.language as tl

import torch
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    InputContract,
    IntegerRange,
    TensorInput,
)
from torch._inductor.runtime._cudagraph.direct_cuda_host import DirectCudaHost
from torch._inductor.runtime._cudagraph.direct_triton import DirectTriton
from torch._inductor.runtime._cudagraph.frontend import DirectPhysicalCall
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    IntExpr,
    PointerSource,
)
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["N"])
def before(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, value + 1, i < N)


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["N"])
def after(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, value * 2, i < N)


FIRST = SECOND = SILU = None


def host(box):
    n, source = box
    box.clear()
    backing = torch.empty_strided(
        (n + 4,), (1,), dtype=source.dtype, device=source.device
    )
    view = backing[2 : n + 2]
    block = 128 if n < 2048 else 256
    FIRST[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](source, view, n, BLOCK=block)
    middle = SILU(view)
    output = torch.empty_like(middle)
    SECOND[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
        middle, output, n, BLOCK=block
    )
    return (output,)


def view_dispatch(source):
    if source.shape[0] < 2048:
        return source[: source.shape[0] // 2]
    return source


def view_host(box):
    n, source = box
    box.clear()
    backing = torch.empty_strided(
        (n + 4,), (1,), dtype=source.dtype, device=source.device
    )
    view = backing[2 : n + 2]
    FIRST[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](source, view, n, BLOCK=128)
    middle = SILU(view)
    count = middle.numel()
    output = torch.empty_like(middle)
    SECOND[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](
        middle, output, count, BLOCK=128
    )
    return (output,)


def silu_and_empty(source):
    output = torch.nn.functional.silu(source)
    return output, torch.empty_like(source)


def late_output_host(box):
    n, source = box
    box.clear()
    backing = torch.empty_strided(
        (n + 4,), (1,), dtype=source.dtype, device=source.device
    )
    view = backing[2 : n + 2]
    FIRST[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](source, view, n, BLOCK=128)
    middle, extra = SILU(view)
    output = torch.empty_like(middle)
    SECOND[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](middle, output, n, BLOCK=128)
    return output, extra


def repeated_host(box):
    n, source = box
    box.clear()
    backing = torch.empty_strided(
        (n + 4,), (1,), dtype=source.dtype, device=source.device
    )
    view = backing[2 : n + 2]
    FIRST[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](source, view, n, BLOCK=128)
    first = SILU(view)
    middle = SILU(first[2:])
    count = middle.numel()
    output = torch.empty_like(middle)
    SECOND[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](
        middle, output, count, BLOCK=128
    )
    return (output,)


def identity_or_view(source):
    return source if source.numel() < 2048 else source[:]


def duplicate_identity_or_views(source):
    if source.numel() == 512:
        return source, source
    return source[:], source[:]


def duplicate_output(source):
    output = torch.nn.functional.silu(source)
    return output, output


def identity_host(box):
    n, source = box
    box.clear()
    backing = torch.empty_strided(
        (n + 4,), (1,), dtype=source.dtype, device=source.device
    )
    view = backing[2 : n + 2]
    FIRST[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](source, view, n, BLOCK=128)
    middle = SILU(view)
    count = n if middle is view else n // 2
    output = torch.empty_strided(
        (count,), (1,), dtype=source.dtype, device=source.device
    )
    SECOND[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](
        middle, output, count, BLOCK=128
    )
    return (output,)


def duplicate_host(box):
    n, source = box
    box.clear()
    backing = torch.empty_strided(
        (n + 4,), (1,), dtype=source.dtype, device=source.device
    )
    view = backing[2 : n + 2]
    FIRST[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](source, view, n, BLOCK=128)
    left, right = SILU(view)
    count = n if left is right else n // 2
    output = torch.empty_strided(
        (count,), (1,), dtype=source.dtype, device=source.device
    )
    SECOND[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](
        left, output, count, BLOCK=128
    )
    return (output,)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and _get_torch_cuda_version() >= (12, 8),
    "requires NVIDIA CUDA >= 12.8",
)
class TestMixedCudaHost(TestCase):
    def runtime(self, host_fn, entry):
        first, second = DirectTriton(before), DirectTriton(after)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        self.enterContext(
            mock.patch.dict(
                globals(), FIRST=first, SECOND=second, SILU=DirectCudaHost(entry)
            )
        )
        count = IntExpr("boxed", 0)
        contract = InputContract(
            ("integer", "tensor"),
            (TensorInput(1, torch.float32, (count,), (1,)),),
            (IntegerRange(0, 32, 16384),),
            torch.cuda.current_device(),
        )
        runtime = direct_host.DirectHost(host_fn, contract)
        self.addCleanup(runtime.close)
        return runtime

    def test_input_output_identity_selects_same_later_kernel(self, device):
        with torch.cuda.device(device):
            runtime = self.runtime(identity_host, identity_or_view)
            for n, variants in ((512, 1), (1024, 1), (4096, 2), (8192, 2), (512, 2)):
                source = torch.randn(n, device=device)
                (output,) = runtime([n, source])
                expected = (source + 1) * 2
                if n >= 2048:
                    expected = expected[: n // 2]
                self.assertEqual(output, expected)
                self.assertEqual(len(runtime.variants), variants)

    def test_equality_specialized_offset_input_identity(self, device):
        with torch.cuda.device(device):
            runtime = self.runtime(duplicate_host, duplicate_identity_or_views)
            held = []
            for n, variants, hit in (
                (512, 1, False),
                (512, 1, True),
                (1024, 2, False),
                (2048, 2, True),
                (512, 2, True),
                (1024, 2, True),
            ):
                source = torch.randn(n, device=device)
                box = [n, source]
                if hit:
                    with mock.patch.object(
                        direct_host,
                        "_observe_direct",
                        side_effect=AssertionError("A native hit must not retrace"),
                    ):
                        (output,) = runtime.entry(box)
                else:
                    (output,) = runtime(box)
                expected = (source + 1) * 2
                if n != 512:
                    expected = expected[: n // 2]
                self.assertEqual(output, expected)
                self.assertEqual(len(runtime.variants), variants)
                held.append((source, output))
            self.assertEqual(len({source.data_ptr() for source, _ in held}), 6)

    def test_repeated_output_preserves_same_tensor_object(self, device):
        with torch.cuda.device(device):
            runtime = self.runtime(duplicate_host, duplicate_output)
            for n in (512, 1024, 512):
                source = torch.randn(n, device=device)
                (output,) = runtime([n, source])
                self.assertEqual(output, torch.nn.functional.silu(source + 1) * 2)
            self.assertEqual(len(runtime.variants), 1)

    def test_output_allocation_after_last_local_launch(self, device):
        with torch.cuda.device(device):
            runtime = self.runtime(late_output_host, silu_and_empty)
            held = []
            for n in (512, 1024, 512):
                source = torch.randn(n, device=device)
                output, extra = runtime([n, source])
                self.assertEqual(output, torch.nn.functional.silu(source + 1) * 2)
                self.assertEqual(extra.shape, source.shape)
                held.append(extra)
            self.assertEqual(len({tensor.data_ptr() for tensor in held}), 3)
            self.assertEqual(len(runtime.variants), 1)

    def test_repeated_adapter_keeps_each_invocation(self, device):
        with torch.cuda.device(device):
            runtime = self.runtime(repeated_host, torch.nn.functional.silu)
            for n in (512, 1024, 512):
                source = torch.randn(n, device=device)
                (output,) = runtime([n, source])
                expected = (
                    torch.nn.functional.silu(torch.nn.functional.silu(source + 1)[2:])
                    * 2
                )
                self.assertEqual(output, expected)
            self.assertEqual(len(runtime.variants), 1)
            calls = [
                event
                for event in runtime.variants[0].program.events
                if type(event) is DirectPhysicalCall
            ]
            self.assertEqual(len(calls), 2)
            self.assertIsNot(calls[0].owner, calls[1].owner)
            self.assertIs(calls[0].owner.adapter, calls[1].owner.adapter)
            self.assertIsNot(calls[0].owner.tape, calls[1].owner.tape)

    def test_view_only_host_retains_local_guards(self, device):
        with torch.cuda.device(device):
            first, second = DirectTriton(before), DirectTriton(after)
            self.addCleanup(first.close)
            self.addCleanup(second.close)
            self.enterContext(
                mock.patch.dict(
                    globals(),
                    FIRST=first,
                    SECOND=second,
                    SILU=DirectCudaHost(view_dispatch),
                )
            )
            count = IntExpr("boxed", 0)
            contract = InputContract(
                ("integer", "tensor"),
                (TensorInput(1, torch.float32, (count,), (1,)),),
                (IntegerRange(0, 32, 16384),),
                torch.cuda.current_device(),
            )
            runtime = direct_host.DirectHost(view_host, contract)
            self.addCleanup(runtime.close)
            for n, variants in ((512, 1), (1024, 1), (4096, 2), (8192, 2), (512, 2)):
                source = torch.randn(n, device=device)
                (output,) = runtime([n, source])
                expected = view_dispatch(source + 1) * 2
                self.assertEqual(output, expected)
                self.assertEqual(len(runtime.variants), variants)

    def test_triton_cuda_triton_dynamic_native_replay(self, device):
        with torch.cuda.device(device):
            first, second = DirectTriton(before), DirectTriton(after)
            self.addCleanup(first.close)
            self.addCleanup(second.close)
            self.enterContext(
                mock.patch.dict(
                    globals(),
                    FIRST=first,
                    SECOND=second,
                    SILU=DirectCudaHost(torch.nn.functional.silu),
                )
            )
            count = IntExpr("boxed", 0)
            contract = InputContract(
                ("integer", "tensor"),
                (TensorInput(1, torch.float32, (count,), (1,)),),
                (IntegerRange(0, 32, 16384),),
                torch.cuda.current_device(),
            )
            runtime = direct_host.DirectHost(host, contract)
            self.addCleanup(runtime.close)
            captures = []
            original = replay._make_replay

            def record_capture(*args, **kwargs):
                # This is the complete shared capture, not one graph per frontend.
                launches = args[6] if len(args) > 6 else kwargs.get("launches")
                self.assertEqual(len(launches), 3)
                call = args[5][1]
                storage = args[2][0].source
                fields = [
                    field
                    for field in call.fields
                    if type(field.source) is PointerSource
                    and field.source.root == storage
                ]
                self.assertEqual(len(fields), 1)
                field = fields[0]
                payload = launches[1].argument_bytes[field.parameter]
                self.assertEqual(
                    payload[field.byte_offset : field.byte_offset + 8],
                    struct.pack("Q", args[7][storage].data_ptr() + 8),
                )
                result = original(*args, **kwargs)
                captures.append(len(launches) if launches is not None else None)
                return result

            self.enterContext(
                mock.patch.object(replay, "_make_replay", side_effect=record_capture)
            )
            held, variants, frames = [], [], []
            addresses = []
            for i, n in enumerate((512, 1024, 4096, 8192, 512)):
                storage = torch.randn(n + 16, device=device)
                source = storage[4 : 4 + n]
                addresses.append(source.data_ptr())
                box = [n, source]
                if i in (1, 3, 4):

                    def profile(frame, event, arg):
                        if event == "call":
                            frames.append(
                                (frame.f_code.co_filename, frame.f_code.co_name)
                            )

                    try:
                        sys.setprofile(profile)
                        (output,) = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                else:
                    (output,) = runtime(box)
                self.assertEqual(box, [])
                expected = torch.nn.functional.silu(source + 1) * 2
                self.assertEqual(output, expected)
                held.append((storage, output, expected))
                variants.append(len(runtime.variants))
            self.assertEqual(variants, [1, 1, 2, 2, 2])
            self.assertEqual(len(set(addresses)), 5)
            self.assertEqual(captures, [3, 3])
            self.assertEqual(len({output.data_ptr() for _, output, _ in held}), 5)
            for variant in runtime.variants:
                physical = [
                    event
                    for event in variant.program.events
                    if type(event) is DirectPhysicalCall
                ]
                self.assertEqual(len(physical), 1)
                fields = physical[0].bound.fields
                view_fields = [
                    field
                    for field in fields
                    if type(field.source) is PointerSource
                    and type(field.source.root) is BufferSource
                ]
                self.assertGreaterEqual(len(view_fields), 2)
            for _, output, expected in held:
                self.assertEqual(output, expected)


instantiate_device_type_tests(TestMixedCudaHost, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
