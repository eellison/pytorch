# Owner(s): ["module: inductor"]
"""Tests for cat multi-consumer and pad-as-cat optimizations."""

import torch
from torch._dynamo.utils import counters
from torch._inductor import metrics
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


# required so that metrics.num_bytes_accessed is populated
torch._logging.set_logs(inductor_metrics=True)


class TestCatMultiConsumer(TestCase):
    @torch._inductor.config.patch(fx_graph_cache=False)
    @requires_gpu()
    def test_cat_to_fp16(self):
        """Multi-consumer cat avoids duplicate computation."""

        def fn(x):
            z = torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)
            y = x.to(torch.float16)
            return z, y

        x = torch.randn(1024, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        metrics.reset()
        result = compiled(x)
        ref = fn(x)

        self.assertEqual(result[0], ref[0])
        self.assertEqual(result[1], ref[1])

        # Without the optimization x would be read twice (once by cat, once
        # by to_fp16). With the optimization ConcatKernel shares x so it is
        # read only once.
        z = ref[0]
        y = ref[1]
        x_bytes = x.nelement() * x.element_size()
        z_bytes = z.nelement() * z.element_size()
        y_bytes = y.nelement() * y.element_size()
        unoptimized_bytes = 2 * x_bytes + z_bytes + y_bytes
        self.assertLess(
            metrics.num_bytes_accessed,
            unoptimized_bytes,
            "Optimization should avoid reading x twice.",
        )

    @torch._inductor.config.patch(fx_graph_cache=False)
    @requires_gpu()
    def test_single_consumer_cat_unchanged(self):
        """Single-consumer cat unchanged."""

        def fn(x):
            return torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)

        x = torch.randn(1024, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        metrics.reset()
        result = compiled(x)
        ref = fn(x)

        self.assertEqual(result, ref)

        # Single-consumer cat should use pointwise_cat which fuses the
        # zeros fill into the cat kernel.  Total bytes: read x + write z.
        x_bytes = x.nelement() * x.element_size()
        z_bytes = ref.nelement() * ref.element_size()
        expected_bytes = x_bytes + z_bytes
        self.assertEqual(
            metrics.num_bytes_accessed,
            expected_bytes,
            lambda msg: f"{msg}\nExpected {expected_bytes} bytes, got {metrics.num_bytes_accessed}.",
        )


class TestPadAsCat(TestCase):
    @requires_gpu()
    def test_mul_pad_addmm(self):
        """Multi-consumer F.pad uses ConcatKernel zero-copy."""
        counters.clear()

        def fn(x, scale, bias, weight):
            mul_result = x * scale
            padded = torch.nn.functional.pad(mul_result, [0, 192])
            mm_result = torch.addmm(bias, mul_result, weight)
            return padded, mm_result

        x = torch.randn(128, 2880, device=GPU_TYPE, dtype=torch.bfloat16)
        scale = torch.randn(128, 2880, device=GPU_TYPE, dtype=torch.bfloat16)
        bias = torch.randn(1024, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(2880, 1024, device=GPU_TYPE, dtype=torch.bfloat16)

        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x, scale, bias, weight)
        ref = fn(x, scale, bias, weight)

        self.assertEqual(result[0], ref[0])
        self.assertEqual(result[1], ref[1], atol=1e-2, rtol=1e-2)
        self.assertIn("reinterpret_tensor", code)
        self.assertGreater(counters["inductor"]["pad_rewritten_as_cat"], 0)

    @requires_gpu()
    def test_single_consumer_pad(self):
        """Single-consumer F.pad is decomposed into cat, which fuses via pointwise_cat."""
        counters.clear()

        def fn(x, scale):
            return torch.nn.functional.pad(x * scale, [0, 192])

        x = torch.randn(128, 2880, device=GPU_TYPE)
        scale = torch.randn(128, 2880, device=GPU_TYPE)

        compiled = torch.compile(fn)
        result = compiled(x, scale)
        ref = fn(x, scale)

        self.assertEqual(result, ref)
        self.assertGreater(counters["inductor"]["pad_rewritten_as_cat"], 0)


@torch._inductor.config.patch(fx_graph_cache=False)
class TestPadOfConcat(TestCase):
    """A right-pad along the cat dim that is the only user of a ConcatKernel
    cat is written by the cat directly instead of copying the concat again."""

    def _inputs(self, dtype=torch.float32):
        B, X, A, D = 16, 50, 20, 32
        x = torch.randn(B, X * D, device=GPU_TYPE, dtype=dtype)
        w = torch.randn(A, D, device=GPU_TYPE, dtype=dtype)
        y = torch.randn(A, B, D, device=GPU_TYPE, dtype=dtype)
        return x, w, y

    @staticmethod
    def _concat(x, w, y):
        # `a` has a second pointwise consumer, so the cat lowers to a ConcatKernel.
        a = (w.unsqueeze(1) + y).permute(1, 0, 2)
        return torch.cat([x.view(x.shape[0], -1, y.shape[-1]), a], 1), a.sigmoid()

    def _run(self, fn, *args, **compile_kwargs):
        counters.clear()
        metrics.reset()
        result, code = run_and_get_code(torch.compile(fn, **compile_kwargs), *args)
        self.assertEqual(result, fn(*args), atol=1e-4, rtol=1e-4)
        return result, "\n".join(code)

    @requires_gpu()
    def test_pad_output_written_once(self):
        def fn(x, w, y):
            c, a = self._concat(x, w, y)
            return torch.nn.functional.pad(c, (0, 0, 0, 5)), a

        def identity_expand(x, w, y):
            c, a = self._concat(x, w, y)
            return torch.nn.functional.pad(c.expand(c.shape), (0, 0, 0, 5)), a

        args = self._inputs()
        nbytes = lambda t: t.nelement() * t.element_size()  # noqa: E731
        for f in (fn, identity_expand):
            with self.subTest(f.__name__):
                torch._dynamo.reset()
                (padded, a), _ = self._run(f, *args)
                expected = sum(map(nbytes, args)) + nbytes(padded) + nbytes(a)
                self.assertEqual(metrics.num_bytes_accessed, expected)
                self.assertEqual(counters["inductor"]["pad_of_cat_preallocated"], 1)

    @requires_gpu()
    def test_pad_feeds_mm(self):
        def fn(x, w, y, m):
            c, a = self._concat(x, w, y)
            return torch.nn.functional.pad(c, (0, 0, 0, 2)) @ m, a

        x, w, y = self._inputs()
        self._run(fn, x, w, y, torch.randn(32, 8, device=GPU_TYPE))
        self.assertEqual(counters["inductor"]["pad_of_cat_preallocated"], 1)

    @requires_gpu()
    def test_reductions_computed_in_place(self):
        def fn(x, y):
            c = torch.cat([x.sum(-1), y.amax(-1)], 1)
            return torch.nn.functional.pad(c, (0, 3), value=-1.0)

        # The padded row (2006 elements) is long enough for comprehensive
        # padding to kick in, which must not apply to a graph output.
        x = torch.randn(8, 1000, 16, device=GPU_TYPE)
        y = torch.randn(8, 1003, 16, device=GPU_TYPE)
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                torch._dynamo.reset()
                padded, code = self._run(fn, x, y, dynamic=dynamic)
                self.assertEqual(padded.stride(), fn(x, y).stride())
                self.assertEqual(code.count("= empty_strided"), 1)
                self.assertEqual(counters["inductor"]["pad_of_cat_preallocated"], 1)

    @requires_gpu()
    @torch._inductor.config.patch(comprehensive_padding=True)
    def test_pad_view_feeds_mm(self):
        def fn(x, y, m):
            c = torch.cat([x.sum(-1), y.amax(-1)], 1)
            return torch.nn.functional.pad(c, (0, 3)).view(4, -1) @ m

        # The view merges the padded row (2006 elements), so the padded buffer
        # must not get a padded row stride, which would need a copy for the view.
        x = torch.randn(8, 1000, 16, device=GPU_TYPE)
        y = torch.randn(8, 1003, 16, device=GPU_TYPE)
        _, code = self._run(fn, x, y, torch.randn(4012, 8, device=GPU_TYPE))
        self.assertEqual(counters["inductor"]["pad_of_cat_preallocated"], 1)
        self.assertEqual(code.count("= empty_strided"), 2)

    @requires_gpu()
    def test_not_applied(self):
        x, w, y = self._inputs()
        pad = torch.nn.functional.pad

        def inline_consumer(x, w, y):
            c, a = self._concat(x, w, y)
            return pad(c, (0, 0, 0, 5)).relu(), pad(c, (0, 0, 0, 5)).sum(1), a

        def other_dim(x, w, y):
            c, a = self._concat(x, w, y)
            return pad(c, (0, 3)), a

        def left_pad(x, w, y):
            c, a = self._concat(x, w, y)
            return pad(c, (0, 0, 1, 0)), a

        def two_dims(x, w, y):
            c, a = self._concat(x, w, y)
            return pad(c, (0, 1, 0, 1)), a

        def cat_used_twice(x, w, y):
            c, a = self._concat(x, w, y)
            return pad(c, (0, 0, 0, 5)), c, a

        def index_select(x, w, y):
            c, a = self._concat(x, w, y)
            idx = torch.arange(0, 70, 3, device=x.device)
            return pad(c, (0, 0, 0, 5)).index_select(1, idx), a

        # The pad is read inline by a mutation, so there is no copy to save.
        def copy_into_input(x, w, y, out):
            c, a = self._concat(x, w, y)
            out.copy_(pad(c, (0, 0, 0, 5)))
            return a

        def index_put_values(x, w, y):
            c, a = self._concat(x, w, y)
            buf = torch.zeros(32, 75, 32, device=x.device)
            buf[torch.arange(0, 32, 2, device=x.device)] = pad(c, (0, 0, 0, 5))
            return buf, a

        # Channels-last inputs, alone or mixed with contiguous ones.
        def channels_last(x, w):
            a = w + 1
            return pad(torch.cat([x, a], 1), (0, 0, 0, 0, 0, 3)), a.sigmoid()

        def channels_last_second(x, w):
            a = w + 1
            return pad(torch.cat([a, x], 1), (0, 0, 0, 0, 0, 3)), a.sigmoid()

        out = torch.empty(16, 75, 32, device=GPU_TYPE)
        x4 = torch.randn(4, 8, 16, 16, device=GPU_TYPE)
        x4 = x4.to(memory_format=torch.channels_last)
        w4 = torch.randn(4, 6, 16, 16, device=GPU_TYPE)
        w4_cl = w4.to(memory_format=torch.channels_last)
        cases = [
            (fn, (x, w, y))
            for fn in (
                inline_consumer,
                other_dim,
                left_pad,
                two_dims,
                cat_used_twice,
                index_select,
                index_put_values,
            )
        ]
        cases += [
            (copy_into_input, (x, w, y, out)),
            (channels_last, (x4, w4)),
            (channels_last_second, (x4, w4)),
            (channels_last, (x4, w4_cl)),
        ]
        for fn, args in cases:
            with self.subTest(fn.__name__):
                self._run(fn, *args)
                self.assertEqual(counters["inductor"]["pad_of_cat_preallocated"], 0)

    @requires_gpu()
    @torch._inductor.config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON,ATEN"
    )
    def test_max_autotune_mm_inputs(self):
        a, b, c, d = (torch.randn(64, 64, device=GPU_TYPE) for _ in range(4))
        pad = torch.nn.functional.pad
        for cat_dim, padding in ((0, (0, 0, 0, 3)), (1, (0, 3))):
            with self.subTest(cat_dim=cat_dim):

                def fn(a, b, c, d):
                    return pad(torch.cat([a @ b, c @ d], cat_dim), padding)

                torch._dynamo.reset()
                self._run(fn, a, b, c, d)
                self.assertEqual(counters["inductor"]["pad_of_cat_preallocated"], 1)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
