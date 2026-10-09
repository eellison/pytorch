# Owner(s): ["module: cuda graphs"]

import dataclasses
import unittest
import unittest.mock
from types import SimpleNamespace

import torch
from torch.cuda._host_trace import Declined
from torch.testing._internal.common_utils import (
    requires_cuda_python_bindings,
    run_tests,
    TEST_CUDA,
    TestCase,
)
from torch.utils._triton import has_triton


if has_triton():
    import triton
    import triton.language as tl
    from triton.compiler import ASTSource

    from torch.cuda._host_trace_triton import param_layout, triton_abi, TritonArg

    @triton.jit
    def _add(x_ptr, y_ptr, n, s, one, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        tl.store(y_ptr + i * one, tl.load(x_ptr + i, mask=i < n) + s, mask=i < n)


def _metadata(**overrides):
    fields = dict(
        num_warps=4,
        shared=0,
        num_ctas=1,
        global_scratch_size=0,
        profile_scratch_size=0,
        launch_cooperative_grid=False,
        launch_pdl=False,
        tensordesc_meta=[],
        instrumentation_mode="",
    )
    return SimpleNamespace(**(fields | overrides))


def _src(**types):
    # what Triton 3.8's JIT builds for _add(x, y, 1000, s, 1, B=128)
    signature = {"x_ptr": "*fp32", "y_ptr": "*fp32", "n": "i32", "s": "i32"}
    signature |= {"one": "constexpr", "B": "constexpr"} | types
    attrs = {(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]]}
    return ASTSource(_add, signature, {(4,): 1, (5,): 128}, attrs)


@unittest.skipIf(not has_triton(), "requires triton")
class TestTritonABI(TestCase):
    def test_slots_follow_the_function_arguments(self):
        abi = triton_abi(_src(), _metadata(num_warps=8, shared=1024))
        want = (
            TritonArg("x_ptr", "*fp32", 0, divisibility=16),
            TritonArg("y_ptr", "*fp32", 1, divisibility=16),
            TritonArg("n", "i32", 2),
            TritonArg("s", "i32", 3),
            TritonArg("one", "constexpr", None, 1),
            TritonArg("B", "constexpr", None, 128),
        )
        self.assertEqual(abi.args, want)
        # the launcher's global and profile scratch pointers come last
        self.assertEqual(abi.num_slots, 6)
        self.assertEqual(list(abi.scratch_slots), [4, 5])
        self.assertEqual([abi.slot_bytes(i) for i in range(6)], [8, 8, 4, 4, 8, 8])
        self.assertEqual((abi.num_warps, abi.shared), (8, 1024))

    def test_an_i64_scalar_is_an_eight_byte_slot(self):
        abi = triton_abi(_src(s="i64"), _metadata())
        self.assertEqual([abi.slot_bytes(i) for i in range(6)], [8, 8, 4, 8, 8, 8])

    def test_float_scalar_slot_widths(self):
        for ty, size in (("fp16", 2), ("bf16", 2), ("fp32", 4), ("fp64", 8)):
            abi = triton_abi(_src(s=ty), _metadata())
            self.assertEqual(abi.slot_bytes(3), size, ty)

    def test_pdl(self):
        self.assertFalse(triton_abi(_src(), _metadata()).pdl)
        self.assertTrue(triton_abi(_src(), _metadata(launch_pdl=True)).pdl)

    def test_pdl_declines_with_trace_pdl_off(self):
        with unittest.mock.patch.object(torch.cuda._host_trace, "trace_pdl", False):
            with self.assertRaisesRegex(Declined, "programmatic-dependent"):
                triton_abi(_src(), _metadata(launch_pdl=True))
            self.assertFalse(triton_abi(_src(), _metadata()).pdl)

    def test_declines(self):
        cases = {
            "clusters": (_src(), _metadata(num_ctas=2)),
            "cooperative": (_src(), _metadata(launch_cooperative_grid=True)),
            "global scratch": (_src(), _metadata(global_scratch_size=128)),
            "profile scratch": (_src(), _metadata(profile_scratch_size=128)),
            "tma": (_src(), _metadata(tensordesc_meta=[{}])),
            "gsan": (_src(), _metadata(instrumentation_mode="gsan")),
            "unsigned scalar": (_src(s="u64"), _metadata()),
            "tuple": (_src(s=("i32", "i32")), _metadata()),
        }
        for name, (src, metadata) in cases.items():
            with self.assertRaises(Declined, msg=name):
                triton_abi(src, metadata)
        src = _src()
        src.attrs[(2,)] = [["tt.pointer_range", 32]]
        with self.assertRaisesRegex(Declined, "tt.pointer_range"):
            triton_abi(src, _metadata())

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    @requires_cuda_python_bindings
    def test_layout_matches_the_loaded_function(self):
        x = torch.randn(1000, device="cuda")
        y = torch.empty_like(x)
        compiled = _add[(8,)](x, y, 1000, 3, 1, B=128)
        abi = triton_abi(compiled.src, compiled.metadata)
        self.assertEqual(abi.num_slots, 6)
        want = ((0, 8), (8, 8), (16, 4), (20, 4), (24, 8), (32, 8))
        self.assertEqual(param_layout(abi, compiled.function), want)
        short = dataclasses.replace(abi, num_slots=abi.num_slots - 1)
        with self.assertRaisesRegex(Declined, "past the ABI"):
            param_layout(short, compiled.function)


def setUpModule():
    from torch.cuda import _host_trace_hint_audit
    import torch.cuda._host_trace as host_trace

    _host_trace_hint_audit.enable_for_tests()
    host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
