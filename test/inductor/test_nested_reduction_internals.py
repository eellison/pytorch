# Owner(s): ["module: inductor"]

"""Generated-kernel structure checks for nested reduction.

These tests deliberately inspect captured Triton source. Keep them separate
from test_nested_reduction.py so kernel-form assertions do not drown out the
core numeric and fusion coverage.
"""

import re

import torch
import torch._inductor.config as inductor_config
from torch._inductor import metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.inductor_utils import (
    get_func_call,
    get_kernel_launch,
    GPU_TYPE,
    HAS_GPU,
)
from torch._inductor.choices import InductorChoices
from torch._inductor.virtualized import V


def _choices_context(force_persistent: bool | None):
    import contextlib

    if force_persistent is None:
        return contextlib.nullcontext()

    class _Choices(InductorChoices):
        @staticmethod
        def should_use_cooperative_reduction(*args, **kwargs):
            return False

        @staticmethod
        def should_use_persistent_reduction(*args, **kwargs):
            return force_persistent

    return V.set_choices_handler(_Choices())


TRITON_KERNEL_RE = re.compile(
    r"(?ms)^@triton_heuristics.*?(?=^@triton_heuristics|^async_compile\.wait|\Z)"
)


def _kernel_name(kernel_code: str) -> str:
    match = re.search(r"^def (triton_[^(]+)\(", kernel_code, re.M)
    assert match is not None
    return match.group(1)


def _nested_kernel_signature(force_persistent_outer_reduction: bool | None) -> str:
    return (
        "triton_red_fused"
        if force_persistent_outer_reduction is False
        else "triton_per_fused"
    )


def _is_wrapper_launched_kernel(wrapper_code: str, kernel_code: str) -> bool:
    return (
        re.search(rf"\b{re.escape(_kernel_name(kernel_code))}\b", wrapper_code)
        is not None
    )


def _run_and_capture_source_bundle(
    f,
    args,
    kernel_signature: str,
    *,
    dynamic: bool = False,
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, list[str]]:
    def capture():
        with (
            inductor_config.patch("triton.nested_reduction", True),
            _choices_context(force_persistent_outer_reduction),
        ):
            compiled = torch.compile(f, dynamic=dynamic)
            return compiled(*args)

    with fresh_inductor_cache():
        _, source_codes = run_and_get_code(capture)
    metrics.reset()
    torch._dynamo.reset()

    combined_code = "\n\n".join(source_codes)
    wrapper_code = next(code for code in source_codes if get_func_call() in code)
    kernel_codes = [
        kernel_code
        for kernel_code in TRITON_KERNEL_RE.findall(combined_code)
        if kernel_signature in kernel_code
        and _is_wrapper_launched_kernel(wrapper_code, kernel_code)
    ]
    return wrapper_code, kernel_codes


def _run_and_capture_sources(
    f,
    args,
    kernel_signature: str,
    *,
    dynamic: bool = False,
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, str]:
    wrapper_code, kernel_codes = _run_and_capture_source_bundle(
        f,
        args,
        kernel_signature,
        dynamic=dynamic,
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )
    if len(kernel_codes) != 1:
        nested_kernel_codes = [
            code
            for code in kernel_codes
            if "'min_xblock':" in code or "'min_rblock':" in code
        ]
        if len(nested_kernel_codes) == 1:
            kernel_codes = nested_kernel_codes
    assert len(kernel_codes) == 1, (
        f"expected exactly one fused kernel matching {kernel_signature!r}, "
        f"got {len(kernel_codes)}: "
        f"{[_kernel_name(kernel_code) for kernel_code in kernel_codes]}"
    )
    return wrapper_code, kernel_codes[0]


def _capture_pattern1_kernel_sources(
    batch_size: int,
    K: int,
    D: int,
    *,
    norm_kind: str = "rms",
    reduction: str = "sum",
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, str]:
    def f(x, w):
        BK, Dim = x.shape[0] * x.shape[1], x.shape[2]
        x_flat = x.reshape(BK, Dim)
        if norm_kind == "rms":
            rms = torch.sqrt(
                torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6
            )
            x_normed = (x_flat / rms).reshape(x.shape)
        else:
            mean = x_flat.mean(dim=-1, keepdim=True)
            var = x_flat.var(dim=-1, keepdim=True, correction=0)
            x_normed = ((x_flat - mean) / torch.sqrt(var + 1e-6)).reshape(x.shape)

        weighted = w[:, :, None] * x_normed
        if reduction == "sum":
            return weighted.sum(dim=1)
        if reduction == "amax":
            return weighted.amax(dim=1)
        if reduction == "amin":
            return weighted.amin(dim=1)
        raise AssertionError(f"unsupported reduction: {reduction}")

    x = torch.randn(batch_size, K, D, device=GPU_TYPE)
    w = torch.randn(batch_size, K, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_pattern2_kernel_sources(
    batch_size: int,
    D: int,
    G: int,
    *,
    norm_kind: str = "layernorm",
    reduction: str = "amax",
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, str]:
    def f(x, G):
        if norm_kind == "layernorm":
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, correction=0)
            x_normed = (x - mean) / torch.sqrt(var + 1e-6)
        else:
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_normed = x / rms

        grouped = x_normed.reshape(x.shape[0], x.shape[1] // G, G)
        if reduction == "amax":
            return grouped.abs().amax(dim=-1)
        if reduction == "sum":
            return grouped.sum(dim=-1)
        if reduction == "amin":
            return grouped.amin(dim=-1)
        raise AssertionError(f"unsupported reduction: {reduction}")

    x = torch.randn(batch_size, D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, G),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_dynamic_pattern1_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    K = 16

    def f(x, w):
        B, D = x.shape[0], x.shape[2]
        x_flat = x.reshape(B * K, D)
        rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
        x_normed = (x_flat / rms).reshape(B, K, D)
        return (w[:, :, None] * x_normed).sum(dim=1)

    x = torch.randn(batch_size, K, 4096, device=GPU_TYPE)
    w = torch.randn(batch_size, K, device=GPU_TYPE)
    torch._dynamo.mark_static(x, 1)
    torch._dynamo.mark_static(w, 1)
    # Dynamic D keeps the outer reduction looped, so this emits a red kernel
    # even when the persistent-forcing test class is active.
    return _run_and_capture_sources(
        f,
        (x, w),
        "triton_red_fused",
        dynamic=True,
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_dynamic_pattern2_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    def f(x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        x_normed = (x - mean) / torch.sqrt(var + 1e-6)
        return x_normed.reshape(x.shape[0], x.shape[1] // 16, 16).abs().amax(dim=-1)

    x = torch.randn(batch_size, 4096, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x,),
        "triton_red_fused",
        dynamic=True,
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_amax_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    B, D, G = batch_size, 4096, 16
    import torch.nn.functional as F

    def f(x, weight):
        x = F.rms_norm(x, (D,), weight)
        return x.view(B, D // G, G).abs().amax(dim=-1)

    x = torch.randn(B, D, device=GPU_TYPE)
    w = torch.randn(D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_producer_scale_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    B, D, G = batch_size, 4096, 16
    import torch.nn.functional as F

    def f(x, weight):
        x = F.rms_norm(x, (D,), weight)
        x = x.view(B, D // G, G)
        amax = x.abs().amax(dim=-1)
        scale = (amax / 448.0).clamp(min=1e-12).to(torch.float8_e4m3fn)
        return scale.float()

    x = torch.randn(B, D, device=GPU_TYPE)
    w = torch.randn(D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_fullres_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    B, D, G = batch_size, 4096, 128
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    import torch.nn.functional as F

    def f(x, weight):
        x = F.rms_norm(x, (D,), weight)
        x_groups = x.view(B, D // G, G)
        amax = x_groups.abs().amax(dim=-1)
        scale = (amax / fp8_max).clamp(min=1e-12)
        x_fp8 = (x_groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        return x_fp8.view(B, D).float(), scale

    x = torch.randn(B, D, device=GPU_TYPE)
    w = torch.randn(D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_no_fullres_epilogue_kernel_sources(
    batch_size: int,
    K: int,
    D: int,
    *,
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, list[str]]:
    def f(x, w):
        x_flat = x.reshape(batch_size * K, D)
        rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
        x_normed = (x_flat / rms).reshape(batch_size, K, D)
        s = (w[:, :, None] * x_normed).sum(dim=1)
        return x_normed + s[:, None, :]

    x = torch.randn(batch_size, K, D, device=GPU_TYPE)
    w = torch.randn(batch_size, K, device=GPU_TYPE)
    wrapper_code, kernel_codes = _run_and_capture_source_bundle(
        f,
        (x, w),
        "triton_",
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )
    kernel_codes = [code for code in kernel_codes if "tl.store(out_ptr" in code]
    kernel_codes.sort(
        key=lambda code: (
            code.count("tl.store(out_ptr"),
            code.count("tl.load(out_ptr"),
            code.count("tl.load(in_ptr"),
        ),
        reverse=True,
    )
    assert len(kernel_codes) == 2, f"expected 2 kernels, got {len(kernel_codes)}"
    return wrapper_code, kernel_codes


def _capture_bf16_epilogue_pattern1_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    def f(x, w):
        B, K, D = x.shape
        x_flat = x.reshape(B * K, D)
        rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
        x_normed = (x_flat / rms).reshape(B, K, D)
        return (w[:, :, None] * x_normed).sum(dim=1).to(torch.bfloat16)

    x = torch.randn(batch_size, 16, 4096, device=GPU_TYPE)
    w = torch.randn(batch_size, 16, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_bf16_epilogue_pattern2_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    def f(x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        x_normed = (x - mean) / torch.sqrt(var + 1e-6)
        return (
            x_normed.reshape(x.shape[0], x.shape[1] // 16, 16)
            .abs()
            .amax(dim=-1)
            .to(torch.bfloat16)
        )

    x = torch.randn(batch_size, 4096, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x,),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_pointwise_epilogue_pattern1_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    def f(x, w, scale, bias):
        B, K, D = x.shape
        x_flat = x.reshape(B * K, D)
        rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
        x_normed = (x_flat / rms).reshape(B, K, D)
        out = (w[:, :, None] * x_normed).sum(dim=1)
        return out * scale + bias

    x = torch.randn(batch_size, 16, 4096, device=GPU_TYPE)
    w = torch.randn(batch_size, 16, device=GPU_TYPE)
    scale = torch.randn(batch_size, 4096, device=GPU_TYPE)
    bias = torch.randn(batch_size, 4096, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w, scale, bias),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_pointwise_epilogue_pattern2_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    def f(x, scale, bias):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        x_normed = (x - mean) / torch.sqrt(var + 1e-6)
        out = x_normed.reshape(x.shape[0], x.shape[1] // 16, 16).abs().amax(dim=-1)
        return out * scale + bias

    x = torch.randn(batch_size, 4096, device=GPU_TYPE)
    scale = torch.randn(batch_size, 256, device=GPU_TYPE)
    bias = torch.randn(batch_size, 256, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, scale, bias),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


class _InternalsBase:
    force_persistent_outer_reduction: bool | None = None

    def setUp(self):
        super().setUp()
        metrics.reset()
        torch._dynamo.utils.clear_compilation_metrics()

    def check_code(
        self,
        code_str,
        num_kernels,
        num_allocs: int | None = None,
        num_deallocs: int | None = None,
    ):
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(),
            num_kernels,
            exactly=True,
        ).run(code_str)
        if num_allocs is not None:
            FileCheck().check(get_func_call()).check_count(
                "empty_strided", num_allocs, exactly=True
            ).run(code_str)
        if num_deallocs is not None and not inductor_config.cpp_wrapper:
            FileCheck().check(get_func_call()).check_count(
                "del ", num_deallocs, exactly=True
            ).run(code_str)

    def check_kernel_io(
        self, kernel_code: str, *, num_inputs: int, num_outputs: int
    ) -> None:
        load_ids = re.findall(r"tl\.load\(in_ptr(\d+)\b", kernel_code)
        output_load_ids = re.findall(r"tl\.load\(out_ptr(\d+)\b", kernel_code)
        store_ids = re.findall(r"tl\.store\(out_ptr(\d+)\b", kernel_code)
        self.assertEqual(len(load_ids), num_inputs)
        self.assertEqual(len(output_load_ids), 0)
        self.assertEqual(len(store_ids), num_outputs)
        self.assertEqual(len(set(load_ids)), num_inputs)
        self.assertEqual(len(set(store_ids)), num_outputs)

    def check_kernel_io_counts(
        self,
        kernel_code: str,
        *,
        input_counts: dict[int, int],
        num_outputs: int,
    ) -> None:
        load_ids = [int(i) for i in re.findall(r"tl\.load\(in_ptr(\d+)\b", kernel_code)]
        output_load_ids = re.findall(r"tl\.load\(out_ptr(\d+)\b", kernel_code)
        store_ids = re.findall(r"tl\.store\(out_ptr(\d+)\b", kernel_code)
        actual_input_counts = {
            idx: load_ids.count(idx) for idx in sorted(set(load_ids))
        }
        self.assertEqual(actual_input_counts, input_counts)
        self.assertEqual(len(output_load_ids), 0)
        self.assertEqual(len(store_ids), num_outputs)
        self.assertEqual(len(set(store_ids)), num_outputs)

    def check_kernel_meta(
        self, kernel_code: str, *, num_inputs: int, num_outputs: int
    ) -> None:
        FileCheck().check_count(
            f"'num_load': {num_inputs}", 1, exactly=True
        ).check_count(
            f"'num_store': {num_outputs}", 1, exactly=True
        ).run(kernel_code)

    def check_axis_classification_contract(
        self,
        kernel_code: str,
        *,
        min_xblock: int | None = None,
        min_rblock: int | None = None,
    ) -> None:
        if min_xblock is None:
            FileCheck().check_not("'min_xblock':").run(kernel_code)
        else:
            FileCheck().check_count(
                f"'min_xblock': {min_xblock}", 1, exactly=True
            ).run(kernel_code)
        if min_rblock is None:
            FileCheck().check_not("'min_rblock':").run(kernel_code)
        else:
            FileCheck().check_count(
                f"'min_rblock': {min_rblock}", 1, exactly=True
            ).run(kernel_code)

    def assert_single_kernel_form(
        self,
        capture,
        *capture_args,
        num_inputs: int | None = None,
        input_counts: dict[int, int] | None = None,
        num_outputs: int,
        meta_num_load: int | None = None,
        num_allocs: int | None = None,
        num_deallocs: int | None = None,
        min_xblock: int | None = None,
        min_rblock: int | None = None,
        extra_checks: FileCheck | None = None,
    ) -> None:
        wrapper_code, kernel_code = capture(
            *capture_args,
            force_persistent_outer_reduction=self.force_persistent_outer_reduction,
        )
        if input_counts is None:
            assert num_inputs is not None
            if num_deallocs is None:
                num_deallocs = num_inputs
            self.check_kernel_io(
                kernel_code, num_inputs=num_inputs, num_outputs=num_outputs
            )
            meta_load = meta_num_load if meta_num_load is not None else num_inputs
            self.check_kernel_meta(
                kernel_code, num_inputs=meta_load, num_outputs=num_outputs
            )
        else:
            if num_deallocs is None:
                num_deallocs = len(input_counts)
            self.check_kernel_io_counts(
                kernel_code, input_counts=input_counts, num_outputs=num_outputs
            )
            meta_load = meta_num_load if meta_num_load is not None else sum(input_counts.values())
            self.check_kernel_meta(
                kernel_code,
                num_inputs=meta_load,
                num_outputs=num_outputs,
            )
        if num_allocs is None:
            num_allocs = num_outputs
        self.check_code(
            wrapper_code,
            num_kernels=1,
            num_allocs=num_allocs,
            num_deallocs=num_deallocs,
        )
        self.check_axis_classification_contract(
            kernel_code,
            min_xblock=min_xblock,
            min_rblock=min_rblock,
        )
        if extra_checks is not None:
            extra_checks.run(kernel_code)

    def check_kernel_io_multiset(
        self,
        kernel_codes: list[str],
        *,
        expected: list[tuple[int, int]],
    ) -> None:
        actual = [
            (
                code.count("tl.load(in_ptr"),
                code.count("tl.store(out_ptr"),
            )
            for code in kernel_codes
        ]
        self.assertEqual(sorted(actual), sorted(expected))

    def check_kernel_load_source_sequence(
        self,
        kernel_codes: list[str],
        *,
        expected: list[tuple[int, int, int]],
    ) -> None:
        actual = []
        for code in kernel_codes:
            actual.append(
                (
                    len(re.findall(r"tl\.load\(in_ptr\d+\b", code)),
                    len(re.findall(r"tl\.load\(out_ptr\d+\b", code)),
                    len(re.findall(r"tl\.store\(out_ptr\d+\b", code)),
                )
            )
        self.assertEqual(actual, expected)

    def check_kernel_meta_multiset(
        self,
        kernel_codes: list[str],
        *,
        expected: list[tuple[int, int]],
    ) -> None:
        actual = []
        for code in kernel_codes:
            match = re.search(
                r"'num_load':\s*(\d+).*?'num_store':\s*(\d+)", code, re.S
            )
            self.assertIsNotNone(match)
            assert match is not None
            actual.append((int(match.group(1)), int(match.group(2))))
        self.assertEqual(sorted(actual), sorted(expected))

    def check_kernel_meta_sequence(
        self,
        kernel_codes: list[str],
        *,
        expected: list[tuple[int, int]],
    ) -> None:
        actual = []
        for code in kernel_codes:
            match = re.search(
                r"'num_load':\s*(\d+).*?'num_store':\s*(\d+)", code, re.S
            )
            self.assertIsNotNone(match)
            assert match is not None
            actual.append((int(match.group(1)), int(match.group(2))))
        self.assertEqual(actual, expected)

    def test_pattern1_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_pattern1_kernel_sources,
            32,
            16,
            4096,
            input_counts=(
                {0: 2, 1: 1}
                if self.force_persistent_outer_reduction is False
                else {0: 1, 1: 1}
            ),
            num_outputs=1,
            meta_num_load=(
                3 if self.force_persistent_outer_reduction is False else 2
            ),
            min_xblock=16,
        )

    def test_pattern1_B1_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_pattern1_kernel_sources,
            1,
            16,
            1024,
            input_counts=(
                {0: 2, 1: 1}
                if self.force_persistent_outer_reduction is False
                else {0: 1, 1: 1}
            ),
            num_outputs=1,
            min_xblock=16,
        )

    def test_pattern2_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_pattern2_kernel_sources,
            32,
            4096,
            16,
            input_counts=(
                {0: 2}
                if self.force_persistent_outer_reduction is False
                else {0: 1}
            ),
            num_outputs=1,
            meta_num_load=(
                2 if self.force_persistent_outer_reduction is False else 1
            ),
            min_rblock=16,
        )

    def test_dynamic_pattern1_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_dynamic_pattern1_kernel_sources,
            32,
            input_counts={0: 2, 1: 1},
            num_outputs=1,
            meta_num_load=3,
            min_xblock=16,
        )

    def test_dynamic_pattern2_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_dynamic_pattern2_kernel_sources,
            32,
            input_counts={0: 2},
            num_outputs=1,
            min_rblock=16,
        )

    def test_producer_consumer_amax_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_amax_kernel_sources,
            128,
            input_counts=(
                {0: 2, 1: 1}
                if self.force_persistent_outer_reduction is False
                else {0: 1, 1: 1}
            ),
            num_outputs=1,
            meta_num_load=(
                3 if self.force_persistent_outer_reduction is False else 2
            ),
            min_rblock=16,
            extra_checks=FileCheck().check_not("tl.split("),
        )

    def test_producer_consumer_scale_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_producer_scale_kernel_sources,
            128,
            input_counts=(
                {0: 2, 1: 1}
                if self.force_persistent_outer_reduction is False
                else {0: 1, 1: 1}
            ),
            num_outputs=1,
            meta_num_load=(
                3 if self.force_persistent_outer_reduction is False else 2
            ),
            min_rblock=16,
        )

    def test_fullres_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_fullres_kernel_sources,
            128,
            input_counts=(
                {0: 2, 1: 1}
                if self.force_persistent_outer_reduction is False
                else {0: 1, 1: 1}
            ),
            num_outputs=2,
            meta_num_load=(
                3 if self.force_persistent_outer_reduction is False else 2
            ),
            min_rblock=128,
            extra_checks=FileCheck().check_not("tl.split(").check(
                "tl.broadcast_to"
            ),
        )

    def test_bf16_epilogue_pattern1_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_bf16_epilogue_pattern1_sources,
            64,
            input_counts=(
                {0: 2, 1: 1}
                if self.force_persistent_outer_reduction is False
                else {0: 1, 1: 1}
            ),
            num_outputs=1,
            meta_num_load=(
                3 if self.force_persistent_outer_reduction is False else 2
            ),
            min_xblock=16,
        )

    def test_bf16_epilogue_pattern2_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_bf16_epilogue_pattern2_sources,
            64,
            input_counts=(
                {0: 2}
                if self.force_persistent_outer_reduction is False
                else {0: 1}
            ),
            num_outputs=1,
            meta_num_load=(
                2 if self.force_persistent_outer_reduction is False else 1
            ),
            min_rblock=16,
        )

    def test_pointwise_epilogue_pattern1_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_pointwise_epilogue_pattern1_sources,
            64,
            input_counts=(
                {0: 2, 1: 1, 2: 1, 3: 1}
                if self.force_persistent_outer_reduction is False
                else {0: 1, 1: 1, 2: 1, 3: 1}
            ),
            num_outputs=1,
            meta_num_load=(
                5 if self.force_persistent_outer_reduction is False else 4
            ),
            min_xblock=16,
        )

    def test_pointwise_epilogue_pattern2_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_pointwise_epilogue_pattern2_sources,
            64,
            input_counts=(
                {0: 2, 1: 1, 2: 1}
                if self.force_persistent_outer_reduction is False
                else {0: 1, 1: 1, 2: 1}
            ),
            num_outputs=1,
            meta_num_load=(
                4 if self.force_persistent_outer_reduction is False else 3
            ),
            min_rblock=16,
        )

    def test_no_fullres_epilogue_kernel_form(self):
        wrapper_code, kernel_codes = _capture_no_fullres_epilogue_kernel_sources(
            64,
            16,
            4096,
            force_persistent_outer_reduction=self.force_persistent_outer_reduction,
        )
        self.check_code(
            wrapper_code,
            num_kernels=2,
            num_allocs=3,
            num_deallocs=4,
        )
        if self.force_persistent_outer_reduction is False:
            self.check_kernel_load_source_sequence(
                kernel_codes,
                expected=[(3, 0, 2), (3, 0, 1)],
            )
            self.check_kernel_meta_sequence(
                kernel_codes, expected=[(3, 2), (3, 1)]
            )
        else:
            self.check_kernel_load_source_sequence(
                kernel_codes,
                expected=[(2, 0, 2), (3, 0, 1)],
            )
            self.check_kernel_meta_sequence(
                kernel_codes, expected=[(2, 2), (3, 1)]
            )


class NestedReductionInternalsPersistentTest(_InternalsBase, TestCase):
    __unittest_skip__ = not HAS_GPU
    force_persistent_outer_reduction = True


class NestedReductionInternalsNonPersistentTest(_InternalsBase, TestCase):
    __unittest_skip__ = not HAS_GPU
    force_persistent_outer_reduction = False


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
