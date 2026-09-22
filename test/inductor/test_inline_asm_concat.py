# Owner(s): ["module: inductor"]

import os
import re
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch
import torch._inductor.test_operators
from torch._dynamo.testing import CompileCounterWithBackend
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config, metrics
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.simd import _SubParentValueResolver
from torch._inductor.codegen.triton import TritonCSE, TritonCSEVariable
from torch._inductor.utils import run_and_get_code, run_and_get_kernels
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges


class TestLaneCSEScope(TestCase):
    def setUp(self):
        super().setUp()
        self.kernel = SimpleNamespace(
            _load_mask=None,
            create_cse_var=TritonCSEVariable,
            cse=TritonCSE(),
            emit_split_via_reshape=lambda *args: None,
        )
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(V.set_kernel_handler(self.kernel))
        self.parent = TritonCSEVariable(
            "source", ValueRanges.unknown(), torch.int32, (1, 8)
        )
        self.kernel.cse.put("source", self.parent)
        self.layout = SimpleNamespace(
            sub_parent_split_shapes=lambda *args, **kwargs: (
                ("1", "4", "2"),
                ("1", "4"),
            )
        )

    def resolver(self, mask):
        def set_masks(kernel, values):
            for value in values:
                value.mask_vars = OrderedSet([mask])

        return _SubParentValueResolver(
            None,
            self.kernel,
            self.layout,
            SimpleNamespace(set_value_masks=set_masks),
            access_relations=(),
            sub_parent_factor=2,
        )

    def test_equal_geometry_different_family_masks(self):
        left, right = self.resolver("left_mask"), self.resolver("right_mask")
        a = left._split_parent(self.parent, 2, 2)
        b = right._split_parent(self.parent, 2, 2)
        self.assertIsNot(a[0], b[0])
        self.assertEqual(a[0].mask_vars, OrderedSet(["left_mask"]))
        self.assertEqual(b[0].mask_vars, OrderedSet(["right_mask"]))
        self.assertIs(left._split_parent(self.parent, 2, 2)[0], a[0])

    def test_invalidation_and_live_parent(self):
        resolver = self.resolver("mask")
        first = resolver._split_parent(self.parent, 2, 2)
        self.kernel.cse.invalidate(OrderedSet([self.parent]))
        second = resolver._split_parent(self.parent, 2, 2)
        self.assertIsNot(first[0], second[0])
        self.kernel.cse.invalidate(OrderedSet())
        self.assertIsNone(resolver._split_parent(self.parent, 2, 2))

    def test_child_scope_does_not_escape(self):
        resolver = self.resolver("mask")
        original = self.kernel.cse
        self.kernel.cse = original.scoped_copy()
        child = resolver._split_parent(self.parent, 2, 2)
        self.kernel.cse = original
        parent = resolver._split_parent(self.parent, 2, 2)
        self.assertIsNot(child[0], parent[0])

    def test_load_masks_have_distinct_cache_entries(self):
        resolver = self.resolver("mask")
        outside = resolver._split_parent(self.parent, 2, 2)
        self.kernel._load_mask = SimpleNamespace(name="predicate")
        inside = resolver._split_parent(self.parent, 2, 2)
        self.kernel._load_mask = None
        self.assertIsNot(outside[0], inside[0])
        self.assertIs(outside[0], resolver._split_parent(self.parent, 2, 2)[0])


def combine_inputs_asm(outputs):
    """Make every input word contribute to one of the assembly outputs."""
    if torch.version.hip:
        instructions = []
        for i in range(outputs):
            instructions.append(f"v_xor_b32 ${i}, ${outputs + i}, ${outputs + 16}")
            for j in range(i + outputs, 16, outputs):
                instructions.append(f"v_xor_b32 ${i}, ${i}, ${outputs + j}")
        return "\n".join(instructions), ",".join(["=&v"] * outputs + ["v"] * 17)
    asm = f"{{ .reg .b32 a<16>; .reg .b32 m; mov.b32 m, ${outputs + 16}; "
    asm += " ".join(f"mov.b32 a{i}, ${outputs + i};" for i in range(16))
    for i in range(outputs):
        asm += f" xor.b32 ${i}, a{i}, m;"
        for j in range(i + outputs, 16, outputs):
            asm += f" xor.b32 ${i}, ${i}, a{j};"
    return asm + " }", ",".join(["=r"] * outputs + ["r"] * 17)


def xor_words(pairs, maximum, outputs):
    words = []
    for start in range(outputs):
        word = pairs[..., start] ^ maximum
        for lane in range(start + outputs, 16, outputs):
            word = word ^ pairs[..., lane]
        words.append(word)
    return words


def realized_rms_groups(x, group=32, dtype=torch.bfloat16):
    value = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)).to(dtype)
    return torch.ops._inductor_test.realize(value).view(x.shape[0], -1, group)


def lane_input(device, width=6144):
    columns = (torch.arange(width, device=device) % 32 - 16).float() / 8
    return torch.arange(17, device=device).float()[:, None] / 8 + columns


class NestedLaneTestCase(TestCase):
    def compare(self, fn, x, persistent, exact=True):
        class Choices(InductorChoices):
            @staticmethod
            def should_use_cooperative_reduction(*args, **kwargs):
                return False

            @staticmethod
            def should_use_persistent_reduction(features, cooperative_reduction):
                if features.reduction_numel == x.shape[-1]:
                    return persistent
                return InductorChoices.should_use_persistent_reduction(
                    features, cooperative_reduction
                )

        settings = {
            "triton.multi_kernel": 0,
            "loop_ordering_after_fusion": True,
            "split_reductions": False,
            "emulate_precision_casts": True,
            "force_disable_caches": True,
        }
        with config.patch({**settings, "triton.nested_reduction": False}):
            expected = torch.compile(fn, fullgraph=True)(x)
        torch._dynamo.reset()
        metrics.reset()
        with (
            V.set_choices_handler(Choices()),
            config.patch({**settings, "triton.nested_reduction": True}),
        ):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        if exact:
            self.assertEqual(actual, expected, atol=0, rtol=0)
        else:
            self.assertEqual(actual, expected)
        return "\n".join(code)


class TestInlineAsmConcat(TestCase):
    @parametrize("multiple_outputs", [False, True])
    @config.patch(
        {
            "force_pointwise_cat": False,
            "max_complex_pointwise_cat_inputs": 8,
            "triton.multi_kernel": 0,
            "force_disable_caches": True,
        }
    )
    def test_unrelated_asm_outputs(self, device, multiple_outputs):
        outputs = 2 if multiple_outputs else 1
        if torch.version.hip:
            asm = "\n".join(
                f"v_add_u32 ${i}, ${outputs}, {i + 1}" for i in range(outputs)
            )
            constraints = ",".join(["=&v"] * outputs + ["v"])
        else:
            asm = f"{{ .reg .b32 a; mov.b32 a, ${outputs}; "
            asm += " ".join(f"add.u32 ${i}, a, {i + 1};" for i in range(outputs))
            asm += " }"
            constraints = ",".join(["=r"] * outputs + ["r"])
        dtype = (torch.int32, torch.int32) if multiple_outputs else torch.int32

        def fn(x):
            a = inline_asm_elementwise(
                x, asm_str=asm, constraints=constraints, dtype=dtype
            )
            b = inline_asm_elementwise(
                x + 5, asm_str=asm, constraints=constraints, dtype=dtype
            )
            if multiple_outputs:
                a, b = a[0], b[1]
            return (torch.stack((a, b), -1) + 3).view(torch.uint8)

        x = torch.arange(1024, device=device, dtype=torch.int32).view(64, 16)
        expected = (torch.stack((x + 1, x + 5 + outputs), -1) + 3).view(torch.uint8)
        torch._dynamo.reset()
        metrics.reset()
        self.assertEqual(torch.compile(fn, fullgraph=True)(x), expected)
        self.assertEqual(metrics.generated_kernel_count, 1)

    @parametrize("outputs,dim", [(2, 0), (6, 1)])
    @config.patch(
        {
            "force_pointwise_cat": False,
            "max_complex_pointwise_cat_inputs": 8,
            "triton.multi_kernel": 0,
            "fx_graph_cache": False,
        }
    )
    def test_stack_outputs(self, device, outputs, dim):
        asm, constraints = combine_inputs_asm(outputs)

        def fn(x):
            maximum = x.amax(-1)
            values = inline_asm_elementwise(
                *x.unbind(-1),
                maximum,
                asm_str=asm,
                constraints=constraints,
                dtype=(torch.int32,) * outputs,
            )
            return torch.stack(values, dim).view(torch.uint8), maximum

        torch._dynamo.reset()
        metrics.reset()
        x = torch.arange(1024 * 16, device=device, dtype=torch.int32).view(1024, 16)
        maximum = x.amax(-1)
        expected_words = xor_words(x, maximum, outputs)
        expected = (
            torch.stack(expected_words, dim).view(torch.uint8),
            maximum,
        )
        self.assertEqual(torch.compile(fn, fullgraph=True)(x), expected)
        self.assertEqual(metrics.generated_kernel_count, 1)


class TestNestedAsmLaneInputs(NestedLaneTestCase):
    @parametrize("dynamic_batch", [False, True])
    @config.patch(
        {
            "triton.nested_reduction": True,
            "loop_ordering_after_fusion": True,
            "comprehensive_padding": True,
            "emulate_precision_casts": True,
            "triton.multi_kernel": 0,
            "force_disable_caches": True,
        }
    )
    def test_pitched_layer_norm_lanes(self, device, dynamic_batch):
        asm, constraints = combine_inputs_asm(3)
        width = 1056

        def fn(x, weight, bias, native=True):
            value = torch.nn.functional.layer_norm(
                x.float(), (width,), weight.float(), bias.float()
            ).to(torch.bfloat16)
            groups = value.view(x.shape[0], -1, 32)
            maximum = groups.abs().amax(-1)
            bits = groups.view(torch.int16).to(torch.int32) & 65535
            pairs = bits[..., ::2] | (bits[..., 1::2] << 16)
            if native:
                words = inline_asm_elementwise(
                    *pairs.unbind(-1),
                    maximum.to(torch.int32),
                    asm_str=asm,
                    constraints=constraints,
                    dtype=(torch.int32,) * 3,
                )
            else:
                words = []
                for i in range(3):
                    word = pairs[..., i] ^ maximum.to(torch.int32)
                    for j in range(i + 3, 16, 3):
                        word = word ^ pairs[..., j]
                    words.append(word)
            packed = torch.stack(words, -1).reshape(x.shape[0], -1)
            return torch.nn.functional.pad(packed, (0, 9)), maximum

        def make_input(batch):
            # Repeated signed rows keep BF16 rounding away from midpoints.
            row = (torch.arange(width, device=device) % 32 - 16).float() / 8
            signs = (torch.arange(batch, device=device) % 2 * 2 - 1).float()
            return (signs[:, None] * row[None, :]).half()

        weight = ((torch.arange(width, device=device) % 5 + 1).float() / 8).half()
        bias = ((torch.arange(width, device=device) % 3 - 1).float() / 16).half()
        x = make_input(128)
        if dynamic_batch:
            torch._dynamo.mark_dynamic(x, 0)
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        metrics.reset()
        actual, kernels = run_and_get_kernels(compiled, x, weight, bias)
        self.assertEqual(actual, fn(x, weight, bias, False), atol=0, rtol=0)
        for batch in (2, 17, 129) if dynamic_batch else (128,):
            value = make_input(batch)
            actual = compiled(value, weight, bias)
            expected = fn(value, weight, bias, False)
            self.assertEqual(actual, expected, atol=0, rtol=0)
            self.assertEqual(
                tuple(t.stride() for t in actual), tuple(t.stride() for t in expected)
            )
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        compute = [kernel for kernel in kernels if "tl.load(" in kernel]
        self.assertEqual(len(compute), 1)
        self.assertTrue("welford" in compute[0] or "tl.sum(" in compute[0])
        self.assertIn("triton_helpers.max2(", compute[0])
        self.assertIn("tl.inline_asm_elementwise(", compute[0])
        self.assertNotIn("in_ptr3", compute[0])  # No BF16 intermediate input.
        self.assertEqual(compute[0].count("tl.store("), 2)

    @parametrize(
        "mode", ["escape", "dynamic", "misaligned", "cross_row", "input_mutation"]
    )
    @config.patch(
        {
            "triton.nested_reduction": True,
            "loop_ordering_after_fusion": True,
            "comprehensive_padding": True,
            "emulate_precision_casts": True,
            "triton.multi_kernel": 0,
            "force_disable_caches": True,
        }
    )
    def test_pitched_group_lanes_and_aliases(self, device, mode):
        pitch = 14 if mode == "misaligned" else 16

        def fn(x):
            value = (x - x.mean(-1, keepdim=True)).to(torch.bfloat16)
            pitched = torch.empty_strided(
                x.shape, (pitch, 1), dtype=value.dtype, device=x.device
            )
            pitched.copy_(value)
            groups = pitched.view(x.shape[0], 3, 4)
            lane = groups[..., 1]
            if mode == "cross_row":
                lane = lane.roll(1, dims=0)
            result = groups.abs().amax(-1) + lane
            if mode == "input_mutation":
                x.add_(1)
                return result, x
            return (result,) if mode == "cross_row" else (result, pitched)

        def make_input(batch):
            row = (torch.arange(12, device=device) % 4 - 2).float() / 8
            signs = (torch.arange(batch, device=device) % 2 * 2 - 1).float()
            return signs[:, None] * row[None, :]

        x = make_input(37)
        if mode == "dynamic":
            torch._dynamo.mark_dynamic(x, 0)
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        expected = fn(x.clone())
        metrics.reset()
        actual, kernels = run_and_get_kernels(compiled, x)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(
            tuple(t.stride() for t in actual), tuple(t.stride() for t in expected)
        )
        if mode == "dynamic":
            for batch in (2, 17, 129):
                value = make_input(batch)
                result, ref = compiled(value), fn(value)
                self.assertEqual(result, ref, atol=0, rtol=0)
                self.assertEqual(
                    tuple(t.stride() for t in result), tuple(t.stride() for t in ref)
                )
        self.assertEqual(counter.frame_count, 1)
        if mode == "input_mutation":
            self.assertEqual(actual[1].data_ptr(), x.data_ptr())
        if len(actual) == 2:
            actual[0].fill_(17)
            expected[0].fill_(17)
            self.assertEqual(actual, expected, atol=0, rtol=0)
            actual[1].add_(5)
            expected[1].add_(5)
            self.assertEqual(actual, expected, atol=0, rtol=0)
        fused = mode not in ("misaligned", "cross_row")
        self.assertEqual(metrics.codegen_nested_reduction, int(fused))
        self.assertEqual(len(kernels), 1 if mode in ("escape", "dynamic") else 2)
        if mode in ("escape", "dynamic"):
            self.assertIn("tl.sum(", kernels[0])
            self.assertTrue(
                "triton_helpers.max2(" in kernels[0]
                or kernels[0].count("tl.maximum(") >= 3
            )
            self.assertNotIn("in_ptr1", kernels[0])
            self.assertEqual(kernels[0].count("tl.store("), 2)

    @parametrize("use_asm", [False, True])
    @parametrize(
        "width,multi_kernel,dynamic_batch",
        [(1024, 0, False), (1024, 1, True), (6144, 0, True), (6144, 1, False)],
    )
    def test_normal_choices_stack_reshape_pad(
        self, device, use_asm, width, multi_kernel, dynamic_batch
    ):
        asm, constraints = combine_inputs_asm(3)

        def fn(x, native=use_asm):
            value = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)).to(
                torch.bfloat16
            )
            groups = value.reshape(x.shape[0], -1, 32)
            maximum = groups.abs().amax(-1)
            bits = groups.view(torch.int16).to(torch.int32) & 65535
            pairs = bits[..., ::2] | (bits[..., 1::2] << 16)
            if native:
                words = inline_asm_elementwise(
                    *pairs.unbind(-1),
                    maximum.to(torch.int32),
                    asm_str=asm,
                    constraints=constraints,
                    dtype=(torch.int32,) * 3,
                )
            else:
                words = xor_words(pairs, maximum.to(torch.int32), 3)
            packed = torch.stack(words, -1).reshape(x.shape[0], -1)
            return torch.nn.functional.pad(packed, (0, 5)), maximum

        def make_input(batch):
            if not use_asm and not dynamic_batch:
                # Keep distinct row phases to detect incorrect row reads.
                return (
                    (torch.arange(batch * width, device=device) % 257 - 128).float()
                    / 64
                ).reshape(batch, width)
            # Avoid BF16 midpoints so packing comparisons can remain exact.
            row = (torch.arange(width, device=device) % 257 - 128).float() / 64
            signs = (torch.arange(batch, device=device) % 2 * 2 - 1).float()
            return signs[:, None] * row[None, :]

        x = make_input(64)
        expected = fn(x, False)
        if dynamic_batch:
            torch._dynamo.mark_dynamic(x, 0)
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        metrics.reset()
        with (
            mock.patch.dict(os.environ, TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE="1"),
            config.patch(
                {
                    "triton.multi_kernel": multi_kernel,
                    "coordinate_descent_tuning": bool(multi_kernel),
                    "max_autotune": bool(multi_kernel),
                    "triton.nested_reduction": True,
                    "loop_ordering_after_fusion": True,
                    "emulate_precision_casts": True,
                    "rebase_concat_copies": True,
                    "triton.coalesce_concat_stores": True,
                    "force_disable_caches": True,
                }
            ),
        ):
            actual, kernels = run_and_get_kernels(compiled, x)
            if dynamic_batch:
                for batch in (2, 17, 129):
                    value = make_input(batch)
                    self.assertEqual(compiled(value), fn(value, False), atol=0, rtol=0)
            self.assertEqual(counter.frame_count, 1)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(
            tuple(t.stride() for t in actual), tuple(t.stride() for t in expected)
        )
        self.assertEqual(actual[0][:, -5:], torch.zeros_like(actual[0][:, -5:]))
        kernels = [kernel for kernel in kernels if "@triton.jit" in kernel]
        if use_asm:
            self.assertEqual(metrics.codegen_nested_reduction, 1)
            compute = [k for k in kernels if "tl.inline_asm_elementwise(" in k]
            self.assertTrue(compute)
            for kernel in compute:
                self.assertIn("tl.sum(", kernel)
                self.assertIn("triton_helpers.max2(", kernel)
                self.assertEqual(kernel.count("tl.store("), 2)
            for kernel in kernels:
                if kernel not in compute:
                    self.assertNotIn("tl.load(", kernel)
        else:
            self.assertGreater(metrics.codegen_nested_reduction, 0)
            consumers = [k for k in kernels if "tl.load(" in k]
            reductions = [k for k in consumers if "tl.sum(" in k]
            pointwise = [k for k in consumers if "tl.sum(" not in k]
            self.assertTrue(reductions)
            for kernel in reductions:
                self.assertIn("triton_helpers.max2(", kernel)
            self.assertLessEqual(len(pointwise), 1)
            for kernel in pointwise:
                self.assertIn(" ^ ", kernel)  # Packing, not a payload copy.
            for kernel in consumers:
                self.assertNotIn("tl.inline_asm_elementwise(", kernel)
            fills = [k for k in kernels if "tl.load(" not in k]
            self.assertLessEqual(len(fills), 1)
            for kernel in fills:
                self.assertIn("tl.store(", kernel)

    @parametrize("persistent", [False, True])
    @parametrize("width,shift_groups", [(6144, False), (1024, True)])
    def test_parent_lane_inputs(self, device, persistent, width, shift_groups):
        asm, constraints = combine_inputs_asm(6)

        def fn(x):
            groups = realized_rms_groups(x)
            maximum = groups.abs().amax(-1)
            if shift_groups:
                groups = groups.roll(1, dims=1)
            bits = groups.view(torch.int16).to(torch.int32) & 65535
            pairs = bits[..., ::2] | (bits[..., 1::2] << 16)
            words = inline_asm_elementwise(
                *pairs.unbind(-1),
                maximum.to(torch.int32),
                asm_str=asm,
                constraints=constraints,
                dtype=(torch.int32,) * 6,
            )
            return torch.stack(words, -1).view(torch.uint8), maximum

        torch.manual_seed(1234)
        x = torch.randn(16, width, device=device)
        self.compare(fn, x, persistent)
        if shift_groups:
            self.assertGreater(metrics.generated_kernel_count, 1)
        else:
            self.assertEqual(metrics.codegen_nested_reduction, 1)
            self.assertEqual(metrics.generated_kernel_count, 1)


class TestNestedLaneForwarding(NestedLaneTestCase):
    @parametrize("persistent", [False, True])
    @parametrize("source_before_reduction", [False, True])
    def test_source_lifetime(self, device, persistent, source_before_reduction):
        width = 6144

        def fn(x):
            source = torch.ops._inductor_test.realize(x + 1)
            mean_square = source.square().mean(-1, keepdim=True)
            value = source * torch.rsqrt(mean_square + 1e-5)
            value = torch.ops._inductor_test.realize(value)
            maximum = value.view(x.shape[0], -1, 32).abs().amax(-1)
            selected = source if source_before_reduction else value
            lanes = selected.view(x.shape[0], -1, 32)
            return maximum + lanes[..., 0], maximum + lanes[..., 31]

        torch.manual_seed(1234)
        x = torch.randn(16, width, device=device)
        self.compare(fn, x, persistent, exact=False)
        if source_before_reduction:
            self.assertGreater(metrics.generated_kernel_count, 1)
        else:
            self.assertEqual(metrics.generated_kernel_count, 1)
            self.assertEqual(metrics.codegen_nested_reduction, 1)


class TestNestedLanePairs(NestedLaneTestCase):
    @parametrize("persistent", [False, True])
    @parametrize("operation", ["sub", "pack"])
    @parametrize(
        "group_size,pattern",
        [(2, "aligned"), (2, "reversed")]
        + [
            (32, pattern)
            for pattern in ("aligned", "reversed", "unaligned", "different_source")
        ],
    )
    def test_pair_before_lane_split(
        self, device, persistent, group_size, operation, pattern
    ):
        width = 6144

        def fn(x):
            groups = realized_rms_groups(x, group_size)
            maximum = groups.abs().amax(-1)
            other = groups
            if pattern == "different_source":
                other = torch.ops._inductor_test.realize(groups.view_as(x) + 1).view_as(
                    groups
                )
            outputs = []
            for base in (0, group_size - 2):
                left, right = base, base + 1
                if pattern == "reversed":
                    left, right = right, left
                elif pattern == "unaligned":
                    left, right = (left + 1) % group_size, (right + 1) % group_size
                a, b = groups[..., left], other[..., right]
                if operation == "pack":
                    a = a.view(torch.int16).to(torch.int32) & 65535
                    b = b.view(torch.int16).to(torch.int32) & 65535
                    outputs.append((a | (b << 16)) ^ maximum.to(torch.int32))
                else:
                    outputs.append(a.float() - b.float() + maximum)
            return tuple(outputs)

        x = lane_input(device, width)
        source = self.compare(fn, x, persistent)
        self.assertEqual(metrics.generated_kernel_count, 1)
        if pattern in ("aligned", "reversed"):
            # The size-two maximum also materializes its input lanes.
            extra_splits = 2 if group_size == 2 else 0
            # Packing splits both transformed parents: masked and shifted.
            # Subtraction shares one parent split between its operands.
            if operation == "pack":
                extra_splits += 1
            self.assertEqual(source.count("tl.split("), group_size // 2 + extra_splits)
            if operation == "pack":
                parents = []
                parent = r".*"
                for pattern in (
                    r"\.to\(tl\.int16, bitcast=True\)",
                    r"\.to\(tl\.int32\)",
                    r" & tmp\d+",
                    r" << tmp\d+",
                ):
                    match = re.search(r"(tmp\d+) = " + parent + pattern, source)
                    if match is None:
                        self.fail(f"expected parent operation {pattern}")
                    parent = re.escape(match.group(1))
                    parents.append(parent)
                for parent in parents[-2:]:
                    self.assertRegex(source, r"tl\.split\(tl\.reshape\(" + parent + ",")
        else:
            self.assertGreater(source.count("tl.split("), group_size // 2)

    @parametrize("persistent", [False, True])
    @parametrize("operation", ["recursive", "mixed_factor", "expensive"])
    def test_pair_composition(self, device, persistent, operation):
        width = 6144

        def fn(x):
            groups = realized_rms_groups(x, 8)
            maximum = groups.abs().amax(-1)
            a, b, c, d = (groups[..., lane].float() for lane in range(4))
            if operation == "recursive":
                return ((a - b) - (c - d)) + maximum
            if operation == "mixed_factor":
                pair = a - b
                return pair - c + maximum, pair + maximum
            return torch.sin(a) - torch.sin(b) + maximum

        x = lane_input(device, width)
        source = self.compare(fn, x, persistent)
        self.assertEqual(metrics.generated_kernel_count, 1)
        if operation == "recursive":
            # Split and subtract adjacent pairs, repeat on the pair results,
            # then materialize the two remaining lanes. Each split is reused.
            self.assertEqual(source.count("tl.split("), 3)
        elif operation == "mixed_factor":
            self.assertGreater(source.count("tl.split("), 4)
        else:
            self.assertEqual(source.count("tl.split("), 7)
            self.assertLess(source.index("tl.split("), source.index(".sin("))


class TestNestedLaneCasts(NestedLaneTestCase):
    @parametrize("persistent", [False, True])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("bitcast", [False, True])
    def test_cast_before_lane_split(self, device, persistent, dtype, bitcast):
        width = 6144

        def fn(x):
            groups = realized_rms_groups(x, dtype=dtype)
            maximum = groups.abs().amax(-1)
            lanes = tuple(groups[..., lane] for lane in (0, 1, 15, 31))
            if bitcast:
                lanes = tuple(lane.view(torch.int16) for lane in lanes)
            return tuple(lane.to(torch.int32) + maximum for lane in lanes)

        x = lane_input(device, width)
        source = self.compare(fn, x, persistent)
        self.assertEqual(metrics.generated_kernel_count, 1)
        self.assertEqual(source.count(".to(tl.int32)"), 1)
        self.assertLess(source.index(".to(tl.int32)"), source.index("tl.split("))
        if bitcast:
            self.assertEqual(source.count(".to(tl.int16, bitcast=True)"), 1)
            self.assertLess(
                source.index(".to(tl.int16, bitcast=True)"), source.index("tl.split(")
            )


class TestPointwisePackingChoices(TestCase):
    settings = {
        "triton.nested_reduction": True,
        "loop_ordering_after_fusion": True,
        "rebase_concat_copies": True,
        "triton.coalesce_concat_stores": True,
        "emulate_precision_casts": True,
        "force_disable_caches": True,
    }

    @parametrize("kind", ["two_word_pack", "float_interleave"])
    def test_cheap_inputs_keep_one_kernel(self, device, kind):
        if kind == "two_word_pack":

            def fn(x):
                first = x[:, 0] | (x[:, 1] << 4)
                second = x[:, 2] | (x[:, 3] << 4)
                packed = torch.stack((first, second), -1).reshape(-1)
                return torch.nn.functional.pad(packed, (0, 5))

            x = (torch.arange(1024, device=device, dtype=torch.int32) % 16).reshape(
                256, 4
            )
        else:

            def fn(x):
                packed = torch.stack((x + 1, x + 2), -1).flatten(1)
                return torch.nn.functional.pad(packed, (0, 5))

            x = torch.arange(77, device=device, dtype=torch.float32).reshape(7, 11)

        expected = fn(x)
        torch._dynamo.reset()
        metrics.reset()
        with (
            config.patch({**self.settings, "triton.multi_kernel": 0}),
        ):
            actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(metrics.generated_kernel_count, 1)


instantiate_device_type_tests(TestPointwisePackingChoices, globals(), only_for="cuda")

instantiate_device_type_tests(TestInlineAsmConcat, globals(), only_for="cuda")
instantiate_device_type_tests(TestNestedAsmLaneInputs, globals(), only_for="cuda")
instantiate_device_type_tests(TestNestedLaneForwarding, globals(), only_for="cuda")
instantiate_device_type_tests(TestNestedLaneCasts, globals(), only_for="cuda")
instantiate_device_type_tests(TestNestedLanePairs, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
