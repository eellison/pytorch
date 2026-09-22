# Owner(s): ["module: inductor"]

import ast
import collections
import dataclasses
from types import SimpleNamespace
from unittest import mock

import torch
import torch._inductor.test_operators
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config, metrics
from torch._inductor.choices import InductorChoices
from torch._inductor.codecache import PyCodeCache
from torch._inductor.codegen.triton import (
    _CoalescedConcatStoreLine,
    _ConcatStoreInfo,
    _ConcatStoreLine,
    TritonKernel,
)
from torch._inductor.utils import IndentedBuffer, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestConcatStoreRemoval(TestCase):
    @parametrize(
        "outputs,removed",
        [(outputs, removed) for outputs in (2, 3) for removed in range(1 << outputs)],
    )
    def test_independent_removal_and_reindent(self, outputs, removed):
        names = "abc"[:outputs]
        removed = {name for i, name in enumerate(names) if removed & (1 << i)}
        stores = tuple(
            _ConcatStoreLine(name, f"tl.store(p{name}, {name})", mock.Mock())
            for name in names
        )
        inner = IndentedBuffer(initial_indent=1)
        inner.writeline(_CoalescedConcatStoreLine("tl.store(joined)", *stores))
        outer = IndentedBuffer(initial_indent=2)
        outer.splice(inner)
        with (
            V.set_graph_handler(
                SimpleNamespace(removed_buffers=removed, inplaced_to_remove=set())
            ),
            V.set_kernel_handler(
                SimpleNamespace(removed_buffers=set(), inplaced_to_remove=set())
            ),
        ):
            expected = (
                "        tl.store(joined)\n"
                if not removed
                else "".join(
                    f"        tl.store(p{name}, {name})\n"
                    for name in names
                    if name not in removed
                )
            )
            self.assertEqual(outer.getvalue(), expected)

    @parametrize(
        "case",
        [
            None,
            "unmasked_1d",
            "unmasked_2d",
            "allocation",
            "dtype",
            "shape",
            "index",
            "mask",
            "offset",
            "indent",
            "barrier",
        ],
    )
    def test_three_store_compatibility(self, case):
        allocation = SimpleNamespace(get_name=lambda: "output", inputs=[])
        info = _ConcatStoreInfo(
            allocation, 0, "pa", "3*xindex", "xmask", "a", torch.int64, ("XBLOCK",)
        )
        unmasked_rank = {"unmasked_1d": 1, "unmasked_2d": 2}.get(case)
        if unmasked_rank is not None:
            info = dataclasses.replace(
                info, mask="None", shape=("XBLOCK", "R0_BLOCK")[:unmasked_rank]
            )
        stores = [
            _ConcatStoreLine(
                name,
                f"tl.store(p{name}, {name})",
                dataclasses.replace(info, offset=i, pointer=f"p{name}", value=name),
            )
            for i, name in enumerate(("a", "b", "c"))
        ]
        if case == "indent":
            stores[2] = stores[2].with_prefix("    ")
        elif case == "barrier":
            stores.insert(2, "tl.debug_barrier()")
        elif case is not None and unmasked_rank is None:
            other = {
                "allocation": SimpleNamespace(get_name=lambda: "other", inputs=[]),
                "dtype": torch.int32,
                "shape": ("XBLOCK", 1),
                "index": "6*xindex",
                "mask": "xmask & extra_mask",
                "offset": 3,
            }[case]
            stores[2].info = dataclasses.replace(stores[2].info, **{case: other})
        kernel = SimpleNamespace(
            stores=IndentedBuffer(),
            args=SimpleNamespace(input_buffers={}, inplace_buffers={}),
            inplace_update_buffers=set(),
            mutations=set(),
            _load_counts=collections.defaultdict(int),
            store_buffer_counts={},
            removed_buffers=set(),
            inplaced_to_remove=set(),
        )
        for store in stores:
            kernel.stores.writeline(store)
        graph = SimpleNamespace(
            sizevars=SimpleNamespace(statically_known_equals=lambda a, b: a == b),
            removed_buffers=set(),
            inplaced_to_remove=set(),
        )
        with V.set_graph_handler(graph), V.set_kernel_handler(kernel):
            TritonKernel._coalesce_concat_stores(kernel)
            source = kernel.stores.getvalue()
            # A later flush must reconsider hazards, including readback through
            # an output pointer that is absent from args.input_buffers.
            for counts in (kernel._load_counts, kernel.store_buffer_counts):
                counts["output"] = 2
                kernel.stores.clear()
                kernel.stores.writelines(stores)
                TritonKernel._coalesce_concat_stores(kernel)
                self.assertNotIn("tl.join(", kernel.stores.getvalue())
                counts.clear()
        if case is None or unmasked_rank is not None:
            self.assertEqual(source.count("tl.store("), 1)
            self.assertIn("tl.arange(0, 4) < 3", source)
        else:
            self.assertEqual(source.count("tl.store("), 2)
            self.assertIn("tl.arange(0, 2)", source)
            self.assertIn("tl.store(pc, c)", source)
            self.assertNotIn("tl.arange(0, 4)", source)
        if unmasked_rank is not None:
            # Triton stores require the mask and value to have equal rank.
            mask_expr = ast.Expression(ast.parse(source).body[0].value.args[-1])
            mask = eval(
                compile(mask_expr, "<store mask>", "eval"),
                {
                    "tl": SimpleNamespace(
                        arange=torch.arange,
                        expand_dims=torch.unsqueeze,
                        reshape=lambda value, shape: value.reshape(shape),
                    )
                },
            )
            self.assertEqual(mask.shape, (1,) * unmasked_rank + (4,))
            self.assertEqual(mask.flatten().tolist(), [True, True, True, False])
        if case == "barrier":
            self.assertIn("tl.debug_barrier()", source)


class ReductionChoices(InductorChoices):
    def __init__(self, persistent, width=None):
        self.persistent = persistent
        self.width = width

    @staticmethod
    def should_use_cooperative_reduction(*args, **kwargs):
        return False

    def should_use_persistent_reduction(self, features, cooperative_reduction):
        if self.width is None or features.reduction_numel == self.width:
            return self.persistent
        return super().should_use_persistent_reduction(features, cooperative_reduction)


class TestTritonConcatStores(TestCase):
    settings = {
        "max_complex_pointwise_cat_inputs": 0,
        "max_pointwise_cat_inputs": 0,
        "triton.coalesce_concat_stores": True,
        "triton.multi_kernel": 0,
        "force_disable_caches": True,
    }

    def check(
        self,
        fn,
        x,
        *,
        expected=None,
        stores=None,
        kernels=None,
        settings=None,
        **tolerances,
    ):
        if expected is None:
            expected = fn(x)
        metrics.reset()
        with config.patch({**self.settings, **(settings or {})}):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(actual, expected, **tolerances)
        if kernels is not None:
            self.assertEqual(metrics.generated_kernel_count, kernels)
        source = "\n".join(code)
        if stores is not None:
            self.assertEqual(source.count("tl.store("), stores)
        return source

    @parametrize("enabled", [False, True])
    def test_coalescing_config(self, device, enabled):
        def fn(x):
            return torch.stack((x + 1, x * 2), -1)

        x = torch.randn(17, 259, device=device)
        coalesce = TritonKernel._coalesce_concat_stores
        with mock.patch.object(
            TritonKernel,
            "_coalesce_concat_stores",
            autospec=True,
            side_effect=coalesce,
        ) as coalescer:
            self.check(
                fn,
                x,
                stores=1 if enabled else 2,
                atol=0,
                rtol=0,
                settings={"triton.coalesce_concat_stores": enabled},
            )
        if enabled:
            coalescer.assert_called()
        else:
            coalescer.assert_not_called()

    @parametrize("outputs", [2, 3, 4, 5, 6])
    @parametrize("dtype", [torch.float32, torch.int64, torch.float16, torch.bfloat16])
    def test_adjacent_stack_outputs(self, device, outputs, dtype):
        def fn(x):
            return torch.stack(tuple(x * (i + 1) + i for i in range(outputs)), -1)

        x = torch.randint(-16, 16, (17, 259), device=device).to(dtype)
        # A narrowing store cast keeps the original, separate stores.
        stores = outputs if dtype in (torch.float16, torch.bfloat16) else outputs // 2
        self.check(fn, x, stores=stores, kernels=1, atol=0, rtol=0)

    @parametrize("persistent", [False, True])
    @parametrize("outputs", [2, 3, 4, 5, 6])
    def test_reduction_store_flush(self, device, persistent, outputs):
        def fn(x):
            value = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
            return torch.stack(tuple(value * (i + 1) + i for i in range(outputs)), -1)

        x = torch.randn(17, 1280, device=device)
        with V.set_choices_handler(ReductionChoices(persistent)):
            source = self.check(
                fn,
                x,
                stores=outputs // 2,
                kernels=1,
                atol=1e-5,
                rtol=1e-5,
                settings={"split_reductions": False},
            )
        if not persistent:
            self.assertGreaterEqual(source.count("for r0_offset in"), 2)

    def test_nonadjacent_concat_slices(self, device):
        def fn(x):
            return torch.cat((x + 1, x + 2, x + 3), -1)

        x = torch.randn(17, 259, device=device)
        self.assertNotIn("tl.join(", self.check(fn, x))

    @parametrize("nested,persistent", [(False, False)])
    @parametrize("output_count", [2, 3])
    @parametrize("constraints_kind", ["fixed", "early_fixed", "generic"])
    def test_inline_asm_output_constraints(
        self, device, nested, persistent, output_count, constraints_kind
    ):
        if not torch.version.hip and constraints_kind != "generic":
            self.skipTest("uses AMDGPU physical register constraints")
        width = 6144 if nested else 259

        if not torch.version.hip:
            constraints = ",".join(["=&r"] * output_count + ["r"])
            asm = "\n".join(
                f"add.u32 ${i}, ${output_count}, {i + 1};" for i in range(output_count)
            )
        elif constraints_kind == "fixed":
            constraints = ",".join([f"={{v{i}}}" for i in range(output_count)] + ["v"])
            # Consume the input before writing either potentially aliased output.
            asm = f"v_mov_b32 ${output_count - 1}, ${output_count}\n" + "\n".join(
                f"v_add_u32 ${i}, ${output_count - 1}, {i + 1}"
                for i in range(output_count)
            )
        elif constraints_kind == "early_fixed":
            constraints = ",".join([f"=&{{v{i}}}" for i in range(output_count)] + ["v"])
            asm = "\n".join(
                f"v_add_u32 ${i}, ${output_count}, {i + 1}" for i in range(output_count)
            )
        else:
            constraints = ",".join(["=&v"] * output_count + [f"{{v{output_count}}}"])
            asm = "\n".join(
                f"v_add_u32 ${i}, ${output_count}, {i + 1}" for i in range(output_count)
            )

        def fn(x):
            if nested:
                value = torch.ops._inductor_test.realize(
                    (x - x.mean(-1, keepdim=True)).to(torch.bfloat16)
                )
                groups = value.view(x.shape[0], -1, 32)
                maximum = groups.abs().amax(-1)
                word = groups[..., 0].to(torch.int32) ^ maximum.to(torch.int32)
            else:
                word = x.to(torch.int32)
            outputs = inline_asm_elementwise(
                word,
                asm_str=asm,
                constraints=constraints,
                dtype=(torch.int32,) * output_count,
            )
            return torch.stack(outputs, -1)

        columns = torch.arange(width, device=device).float()[None, :]
        rows = torch.arange(17, device=device)[:, None] % 4 + 1
        x = columns * rows
        if nested:
            value = (x - x.mean(-1, keepdim=True)).to(torch.bfloat16).view(17, -1, 32)
            word = value[..., 0].to(torch.int32) ^ value.abs().amax(-1).to(torch.int32)
        else:
            word = x.to(torch.int32)
        expected = torch.stack(tuple(word + i + 1 for i in range(output_count)), -1)
        torch._dynamo.reset()
        with V.set_choices_handler(ReductionChoices(persistent, width)):
            source = self.check(
                fn,
                x,
                expected=expected,
                kernels=1,
                atol=0,
                rtol=0,
                stores=1,
                settings={
                    "triton.nested_reduction": nested,
                    "loop_ordering_after_fusion": True,
                    "split_reductions": False,
                    "emulate_precision_casts": True,
                },
            )
        self.assertEqual(metrics.codegen_nested_reduction, int(nested))
        self.assertEqual(source.count("tl.inline_asm_elementwise("), 1)

    def test_separate_concat_allocations(self, device):
        def fn(x):
            left = torch.stack((x + 1, x + 2, x + 3), -1)
            right = torch.stack((x + 4, x + 5, x + 6), -1)
            return left, right

        x = torch.randn(17, 259, device=device)
        # Each allocation has one store covering its three adjacent columns.
        self.check(fn, x, stores=2)

    @parametrize("numel", [257, 1031])
    def test_three_stores_tail_canary(self, device, numel):
        def fn(x):
            return torch.stack((x + 1, x + 2, x + 3), -1)

        x = torch.arange(numel, device=device, dtype=torch.int64) << 40
        source = self.check(fn, x, stores=1, atol=0, rtol=0)
        module = PyCodeCache.load(source)
        allocate = module.empty_strided_cuda
        guarded = []
        canary = 0x123456789ABCDEF
        guard = 32

        def guarded_allocate(size, stride, dtype):
            if dtype == torch.int64 and tuple(size) == (numel, 3):
                storage = torch.full(
                    (numel * 3 + 2 * guard,), canary, device=device, dtype=dtype
                )
                guarded.append(storage)
                return storage.as_strided(size, stride, guard)
            return allocate(size, stride, dtype)

        with mock.patch.object(module, "empty_strided_cuda", guarded_allocate):
            output = module.call([x])[0]
        self.assertEqual(output, fn(x), atol=0, rtol=0)
        self.assertEqual(len(guarded), 1)
        self.assertEqual(
            guarded[0][:guard], torch.full_like(guarded[0][:guard], canary)
        )
        self.assertEqual(
            guarded[0][-guard:], torch.full_like(guarded[0][-guard:], canary)
        )

    def test_three_stores_unmasked(self, device):
        """Power-of-two sizes drop the row mask; the joined store's lane mask
        must still be rank two or Triton cannot broadcast it."""

        def fn(x):
            return torch.stack((x + 1, x + 2, x + 3), -1)

        x = torch.arange(1 << 20, device=device, dtype=torch.int64).reshape(1024, 1024)
        source = self.check(fn, x, stores=1, atol=0, rtol=0)
        self.assertIn("tl.full([XBLOCK], True, tl.int1)", source)


instantiate_parametrized_tests(TestConcatStoreRemoval)
instantiate_device_type_tests(TestTritonConcatStores, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
