# Owner(s): ["module: inductor"]

from contextlib import contextmanager, ExitStack
from unittest import mock

import sympy

import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import concat_rebase, config, ir
from torch._inductor.graph import GraphLowering
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import ops, V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._ordered_set import OrderedSet


def stack_words(values):
    words = torch.stack(values, -1)
    return words.reshape(words.shape[0], -1)


def padded_words(values):
    return F.pad(stack_words(values), (0, 5))


def concat_tail(value):
    tail = torch.zeros((value.shape[0], 5), device=value.device, dtype=value.dtype)
    return torch.cat((value, tail), -1)


@contextmanager
def record_plans():
    plans = []
    original = concat_rebase._rebase_plan

    def plan(*args, **kwargs):
        result = original(*args, **kwargs)
        if result:
            plans.append(result)
        return result

    with mock.patch.object(concat_rebase, "_rebase_plan", plan):
        yield plans


class RebaseTestCase(TestCase):
    settings = {
        "rebase_concat_copies": True,
        "max_pointwise_cat_inputs": 0,
        "max_complex_pointwise_cat_inputs": 0,
        "comprehensive_padding": True,
        "compile_threads": 1,
        "force_disable_caches": True,
    }

    def check(
        self,
        fn,
        x,
        *,
        rebase,
        dynamic=False,
        expected=None,
        plan_size=None,
        copies=None,
        settings=None,
    ):
        replaced = []
        original_pass = concat_rebase.rebase_copies

        def run_pass(graph):
            candidates = {
                buffer.get_name(): buffer
                for buffer in graph.buffers
                if isinstance(buffer, ir.ComputedBuffer)
                and isinstance(buffer.data.inner_fn, concat_rebase._ConcatCopy)
            }
            original_pass(graph)
            for name, old in candidates.items():
                current = graph.name_to_buffer[name]
                if current is not old:
                    self.assertIsInstance(current, ir.ConcatKernel)
                    self.assertIs(graph.name_to_op[old.get_operation_name()], current)
                    replaced.append(name)

        if expected is None:
            expected = fn(x)
        torch._dynamo.reset()
        with (
            config.patch({**self.settings, **(settings or {})}),
            record_plans() as plans,
            mock.patch.object(concat_rebase, "rebase_copies", run_pass),
        ):
            actual = torch.compile(fn, fullgraph=True, dynamic=dynamic)(x)
        self.assert_outputs(actual, expected)
        self.assertEqual(bool(plans), rebase)
        if plan_size is not None:
            self.assertEqual(max(map(len, plans)), plan_size)
        if copies is not None:
            self.assertEqual(len(replaced), copies)
        return actual, expected

    def assert_outputs(self, actual, expected):
        self.assertEqual(actual, expected, atol=0, rtol=0)
        for a, e in zip(
            torch.utils._pytree.tree_leaves(actual),
            torch.utils._pytree.tree_leaves(expected),
        ):
            self.assertEqual(a.stride(), e.stride())

    def mutate(self, actual, expected, mutation):
        mutation(actual)
        mutation(expected)
        self.assertEqual(actual, expected, atol=0, rtol=0)


class TestConcatRebase(RebaseTestCase):
    @parametrize("dtype", [torch.float32, torch.int32, torch.int64])
    @parametrize("outputs", [2, 3, 6])
    def test_stack_reshape_pad(self, device, dtype, outputs):
        def fn(x):
            return padded_words(tuple(x * (i + 2) + i + 1 for i in range(outputs)))

        x = torch.arange(7 * 11, device=device).reshape(7, 11).to(dtype)
        self.check(fn, x, rebase=True, plan_size=outputs + 1, copies=1)

    @parametrize("prefix", [0, 2])
    def test_nested_concat_and_nonzero_offsets(self, device, prefix):
        def fn(x):
            left = stack_words((x + 1, x + 2))
            right = stack_words((x + 3, x + 4, x + 5))
            if prefix:
                head = torch.full(
                    (x.shape[0], prefix), 7, device=x.device, dtype=x.dtype
                )
                words = torch.cat((head, left, right), -1)
            else:
                words = torch.cat((left, right), -1)
            return F.pad(words, (0, 5))

        x = torch.arange(77, device=device).reshape(7, 11).float()
        self.check(fn, x, rebase=True, plan_size=11 if prefix else 10)

    @parametrize(
        "kind",
        [
            "partial",
            "transpose",
            "dtype",
            "expand",
            "input",
            "stack",
            "base",
            "identity",
            "reshape",
            "duplicate",
            "mutation",
        ],
    )
    def test_source_alias(self, device, kind):
        escaped = kind in (
            "stack",
            "base",
            "identity",
            "reshape",
            "duplicate",
            "mutation",
        )

        def fn(x):
            base = x + 1
            views = {
                "partial": lambda: base[:, :5],
                "transpose": base.t,
                "dtype": lambda: base.view(torch.int32),
                "expand": lambda: base[:1].expand_as(base),
                "input": lambda: x,
                "identity": lambda: base.view_as(base),
                "reshape": lambda: base.reshape(1, -1).reshape_as(base),
            }
            value = views.get(kind, lambda: base)()
            other = (
                (value if kind == "duplicate" else x + 2)
                if escaped
                else torch.full_like(value, 3)
            )
            words = torch.stack((value, other), -1)
            if kind == "mutation":
                base.add_(17)
            padded = F.pad(words.reshape(words.shape[0], -1), (0, 5))
            return (padded, words if kind == "stack" else base) if escaped else padded

        x = torch.arange(77, device=device).reshape(7, 11).float()
        # Materialized concat inputs are private even when their source escapes.
        actual, expected = self.check(fn, x, rebase=kind != "stack")
        if escaped:
            self.mutate(actual, expected, lambda result: result[0].fill_(19))
            self.mutate(actual, expected, lambda result: result[1].add_(100))

    @parametrize("enabled", [False, True])
    def test_whole_pointwise_root_escape(self, device, enabled):
        def fn(x):
            bits = x.reshape(x.shape[0], -1, 32)
            pairs = bits[..., ::2] | (bits[..., 1::2] << 16)
            words = []
            for start in range(3):
                word = pairs[..., start]
                for lane in range(start + 3, 16, 3):
                    word = word ^ pairs[..., lane]
                words.append(word)
            packed = torch.stack(words, -1).reshape(x.shape[0], -1)
            return packed, F.pad(packed, (0, 5))

        rejected = []
        original = concat_rebase._rebase_plan

        def plan(src, dst, allowed_copy_ops=(), **kwargs):
            result = original(src, dst, allowed_copy_ops, **kwargs)
            root = concat_rebase._storage_buffer(src)
            if (
                isinstance(root, ir.ComputedBuffer)
                and type(root.data) is ir.Pointwise
                and root.get_name() in kwargs["graph"].get_output_names()
            ):
                self.assertIsNone(result)
                rejected.append(root)
            return result

        x = (torch.arange(64 * 1024, device=device, dtype=torch.int32) % 65536).reshape(
            64, 1024
        )
        expected = fn(x)
        torch._dynamo.reset()
        with (
            config.patch(
                {
                    "rebase_concat_copies": enabled,
                    "triton.multi_kernel": 0,
                    "compile_threads": 1,
                    "force_disable_caches": True,
                }
            ),
            mock.patch.object(concat_rebase, "_rebase_plan", plan),
        ):
            # Normal cat policy must expose the escaped materialized Pointwise
            # root to the proof, rather than testing only a ConcatKernel root.
            actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(bool(rejected), enabled)
        self.assert_outputs(actual, expected)
        self.assertFalse(torch._C._is_alias_of(*actual))
        self.mutate(actual, expected, lambda result: result[0].add_(17))
        self.mutate(actual, expected, lambda result: result[1].fill_(43))

    @parametrize("case", ["symbolic", "partial", "non_affine", "duplicate", "empty"])
    def test_copy_geometry(self, device, case):
        def fn(x):
            value = x + 1
            if case == "non_affine":
                words = torch.cat((value, x + 2), 0).reshape(x.shape[0], -1)
            else:
                values = (value, value if case == "duplicate" else x + 2)
                words = torch.stack(values, -1).reshape(
                    x.shape[0], 0 if case == "empty" else -1
                )
            if case == "partial":
                words = words[:, 1:-1]
            return F.pad(words, (0, 5))

        x = (
            torch.empty(7, 0, device=device)
            if case == "empty"
            else torch.arange(77, device=device, dtype=torch.float32).reshape(7, 11)
        )
        # Symbolic width, partial source coverage, and non-affine destinations
        # decline. Repeated inputs are safe after private materialization.
        self.check(fn, x, rebase=case == "duplicate", dynamic=case == "symbolic")

    @parametrize("nested_escape", [False, True])
    def test_reader_index(self, device, nested_escape):
        def fn(x):
            if nested_escape:
                words = torch.stack((x + 1, x + 2), -1)
                value = words.flatten(1)
                for _ in range(4):
                    value = F.pad(value, (0, 3))
                return value, words
            return tuple(
                F.pad(torch.stack((x + i, x + i + 1), -1).flatten(1), (0, 3))
                for i in range(8)
            )

        original_plan = concat_rebase._rebase_plan
        original_pass = concat_rebase.rebase_copies
        original_reads = ir.ComputedBuffer.get_read_writes
        readers = None
        extractions = 0

        def plan(*args, **kwargs):
            nonlocal readers
            readers = kwargs["readers"]
            return original_plan(*args, **kwargs)

        def reads(buffer):
            nonlocal extractions
            extractions += 1
            return original_reads(buffer)

        def run_pass(graph):
            count = sum(isinstance(op, ir.ComputedBuffer) for op in graph.operations)
            with mock.patch.object(ir.ComputedBuffer, "get_read_writes", reads):
                original_pass(graph)
            self.assertLessEqual(extractions, 2 * count)
            self.assertIsNotNone(readers)
            expected = {}
            for op in graph.operations:
                for dep in op.get_read_writes().reads:
                    expected.setdefault(dep.name, set()).add(op.get_operation_name())
            self.assertEqual(
                {name: users for name, users in readers.items() if users}, expected
            )

        x = torch.arange(77, device=device).reshape(7, 11).float()
        with (
            mock.patch.object(concat_rebase, "_rebase_plan", plan),
            mock.patch.object(concat_rebase, "rebase_copies", run_pass),
        ):
            actual, expected = self.check(fn, x, rebase=True)
        if nested_escape:
            actual[0].add_(17)
            expected[0].add_(17)
            self.assertEqual(actual, expected)

    def test_no_fx_provenance(self, device):
        original = concat_rebase._rebase_plan

        def without_origins(src, dst, allowed_copy_ops=(), **kwargs):
            # Late ownership is an IR property. Hide FX metadata only during
            # the proof, preserving it for scheduling and diagnostics.
            with ExitStack() as stack:
                for buffer in V.graph.buffers:
                    stack.enter_context(mock.patch.object(buffer, "origin_node", None))
                    stack.enter_context(
                        mock.patch.object(buffer, "origins", OrderedSet())
                    )
                return original(src, dst, allowed_copy_ops, **kwargs)

        def fn(x):
            return padded_words((x + 1, x + 2))

        x = torch.arange(77, device=device).reshape(7, 11).float()
        with mock.patch.object(concat_rebase, "_rebase_plan", without_origins):
            self.check(fn, x, rebase=True, plan_size=3)

    @parametrize("stages", [3, 4])
    def test_repeated_copy_nesting(self, device, stages):
        def fn(x):
            value = stack_words((x + 1, x + 2))
            for i in range(stages):
                tail = torch.full(
                    (x.shape[0], 3 + 2 * i), i, device=x.device, dtype=x.dtype
                )
                value = torch.cat((value, tail), -1)
            return value

        x = torch.arange(77, device=device).reshape(7, 11).float()
        # Later plans must resolve replaced children by their registered name,
        # including children whose layout an earlier rebase already changed.
        self.check(fn, x, rebase=True, copies=stages)

    def test_rebased_output_view_mutation(self, device):
        def fn(x):
            words = torch.stack((x + 1, x + 2, x + 3), -1)
            output = F.pad(words.reshape(x.shape[0], -1), (0, 5))
            view = output[:, 1::2]
            view.add_(7)
            return output, view

        x = torch.arange(77, device=device, dtype=torch.int32).reshape(7, 11)
        actual, expected = self.check(fn, x, rebase=True, plan_size=4)
        self.mutate(actual, expected, lambda result: result[1].fill_(43))
        self.mutate(actual, expected, lambda result: result[0].add_(17))

    @parametrize(
        "escape,default_policy",
        [
            ("private", False),
            ("outputs", False),
            ("private", True),
            ("outputs", True),
            ("sibling", False),
        ],
    )
    def test_inline_asm_tuple(self, device, escape, default_policy):
        if torch.version.hip:
            asm = "v_add_u32 $0, $3, 1\nv_add_u32 $1, $3, 2\nv_add_u32 $2, $3, 3"
            constraints = "=&v,=&v,=&v,v"
        else:
            asm = "add.u32 $0, $3, 1; add.u32 $1, $3, 2; add.u32 $2, $3, 3;"
            constraints = "=&r,=&r,=&r,r"

        def result(outputs):
            padded = padded_words(outputs[:2] if escape == "sibling" else outputs)
            if escape == "sibling":
                return padded, outputs[2]
            return (padded, outputs) if escape == "outputs" else padded

        def fn(x):
            return result(
                inline_asm_elementwise(
                    x, asm_str=asm, constraints=constraints, dtype=(torch.int32,) * 3
                )
            )

        x = torch.arange(77, device=device, dtype=torch.int32).reshape(7, 11)
        settings = (
            {
                name: getattr(config, name)
                for name in (
                    "max_pointwise_cat_inputs",
                    "max_complex_pointwise_cat_inputs",
                )
            }
            if default_policy
            else None
        )
        expected = result((x + 1, x + 2, x + 3))
        actual, expected = self.check(
            fn, x, rebase=not default_policy, expected=expected, settings=settings
        )
        if escape == "sibling":
            # The third HOP output escapes; only private materialized inputs move.
            self.mutate(actual, expected, lambda value: value[0].fill_(19))
            self.mutate(actual, expected, lambda value: value[1].add_(17))
        elif escape == "outputs":
            self.mutate(actual, expected, lambda value: value[0].fill_(37))
            self.mutate(actual, expected, lambda value: value[1][0].add_(17))


class TestConcatRebaseCPU(RebaseTestCase):
    @parametrize("kind", ["simple", "nested", "escaped"])
    def test_concat_layout(self, kind):
        def fn(x):
            base = x + 1
            left = stack_words((base.view_as(base), x + 2))
            if kind == "nested":
                right = stack_words((x + 3, x + 4))
                words = torch.cat((left, right), -1)
            else:
                words = left
            # CPU padding uses a separate pointwise lowering. Exercise the
            # generic concat allocation path directly on this backend.
            tail = torch.full((x.shape[0], 5), 0.0, dtype=x.dtype)
            output = torch.cat((words, tail), -1)
            return (output, base) if kind == "escaped" else output

        x = torch.arange(77).reshape(7, 11).float()
        actual, expected = self.check(fn, x, rebase=True)
        if kind == "escaped":
            actual[0].fill_(31)
            expected[0].fill_(31)
            actual[1].add_(17)
            expected[1].add_(17)
            self.assertEqual(actual, expected)


class TestConcatRebaseRobustness(RebaseTestCase):
    @parametrize("enabled", [False, True])
    @parametrize("escaped", [False, True])
    def test_dynamic_batch_reuse_and_output_aliases(self, device, enabled, escaped):
        def fn(x):
            words = torch.stack((x + 1, x * 2), -1).flatten(1)
            padded = concat_tail(words)
            return (padded, words) if escaped else (padded,)

        replaced = []
        original = GraphLowering.replace_operation_buffer

        def replace(graph, old, new):
            replaced.append(old.get_name())
            return original(graph, old, new)

        torch._dynamo.reset()
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        with (
            config.patch({**self.settings, "rebase_concat_copies": enabled}),
            mock.patch.object(GraphLowering, "replace_operation_buffer", replace),
        ):
            for index, batch in enumerate((7, 17, 2, 129, 1, 0, 7)):
                x = torch.arange(batch * 11, device=device).reshape(batch, 11).float()
                if batch > 1:
                    torch._dynamo.mark_dynamic(x, 0, min=2, max=256)
                expected = fn(x)
                actual = compiled(x)
                self.assert_outputs(actual, expected)
                if index == 0:
                    self.assertEqual(bool(replaced), enabled and not escaped)
                if index < 4:
                    self.assertEqual(counter.frame_count, 1)
                actual[0].fill_(31)
                expected[0].fill_(31)
                if escaped:
                    actual[1].add_(17)
                    expected[1].add_(17)
                self.assertEqual(actual, expected, atol=0, rtol=0)

    def test_disabled(self, device):
        def fn(x):
            return concat_tail(torch.stack((x + 1, x * 2), -1).flatten(1))

        def unexpected_pass(*args):
            raise AssertionError("Disabled concat rebasing must not run")

        x = torch.arange(77, device=device, dtype=torch.float32).reshape(7, 11)
        with (
            mock.patch.object(
                concat_rebase, "copy_loader", wraps=concat_rebase.copy_loader
            ) as copy_loader,
            mock.patch.object(
                concat_rebase, "_snapshot_input", side_effect=AssertionError
            ),
            mock.patch.object(concat_rebase, "rebase_copies", unexpected_pass),
        ):
            self.check(fn, x, rebase=False, settings={"rebase_concat_copies": False})
        copy_loader.assert_called()

    def test_layout_symbol_cache_invalidation(self, device):
        def fn(x):
            return concat_tail(torch.stack((x + 1, x * 2), -1).flatten(1))

        original = concat_rebase.rebase_copies

        def run_pass(graph):
            cached = [
                (
                    buffer,
                    buffer.layout,
                    [buffer.get_free_symbol_uses(flag) for flag in (False, True)],
                )
                for buffer in graph.buffers
                if isinstance(buffer, (ir.ComputedBuffer, ir.ConcatKernel))
            ]
            original(graph)
            changed = set()
            for buffer, layout, symbols in cached:
                if buffer.layout is layout:
                    continue
                changed.add(type(buffer))
                # Static rebasing keeps the symbols, but both cached argument
                # variants (including the inherited concat cache) must refresh.
                for flag, previous in zip((False, True), symbols):
                    current = buffer.get_free_symbol_uses(flag)
                    self.assertIsNot(current, previous)
                    self.assertEqual(current, previous)
            self.assertEqual(changed, {ir.ComputedBuffer, ir.ConcatKernel})

        x = torch.arange(77, device=device, dtype=torch.float32).reshape(7, 11)
        with mock.patch.object(concat_rebase, "rebase_copies", run_pass):
            self.check(fn, x, rebase=True, copies=1)

    @parametrize("saved_activation", [False, True])
    def test_aot_autograd(self, device, saved_activation):
        def fn(x):
            first = x.softmax(-1) if saved_activation else x + 1
            return concat_tail(stack_words((first, x * 2)))

        torch._dynamo.reset()
        x = torch.linspace(-1, 1, 77, device=device).reshape(7, 11).requires_grad_()
        reference = x.detach().clone().requires_grad_()
        with (
            config.patch(self.settings),
            record_plans() as plans,
        ):
            actual = torch.compile(fn, fullgraph=True)(x)
            expected = fn(reference)
            self.assertEqual(actual, expected, atol=1e-6, rtol=1e-6)
            self.assertEqual(actual.stride(), expected.stride())
            if not saved_activation:
                self.assertTrue(plans)
            # AOT may save or recompute an activation. Mutating the returned
            # concat must not overwrite any activation needed by backward.
            with torch.no_grad():
                actual[:, 0].add_(0.125)
                expected[:, 0].add_(0.125)
            gradient = (
                torch.arange(
                    actual.numel(), device=device, dtype=actual.dtype
                ).reshape_as(actual)
                / actual.numel()
            )
            actual_grad = torch.autograd.grad(actual, x, gradient)
            expected_grad = torch.autograd.grad(expected, reference, gradient)
        self.assertEqual(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)

    @parametrize("dynamic", [None, True])
    def test_empty_and_dynamic_transitions(self, device, dynamic):
        def fn(x):
            words = torch.stack((x + 1, x * 2), -1)
            words = words.reshape(x.shape[0], x.shape[1] * 2)
            return concat_tail(words)

        torch._dynamo.reset()
        with (
            config.patch(self.settings),
            record_plans() as plans,
        ):
            compiled = torch.compile(fn, fullgraph=True, dynamic=dynamic)
            for index, (m, n) in enumerate([(7, 11), (7, 0), (0, 11), (3, 5), (7, 11)]):
                x = torch.arange(m * n, device=device, dtype=torch.float32).reshape(
                    m, n
                )
                plans.clear()
                actual, expected = compiled(x), fn(x)
                self.assert_outputs(actual, expected)
                if dynamic or not m or not n:
                    self.assertFalse(plans)
                elif index == 0:
                    self.assertTrue(plans)

    def test_returned_input_alias(self, device):
        def fn(x):
            return concat_tail(stack_words((x + 1, x * 2))), x[:, 1::2]

        x = torch.arange(77, device=device, dtype=torch.float32).reshape(7, 11)
        reference = x.clone()
        actual, expected = self.check(fn, x, rebase=True, expected=fn(reference))
        # A separate reference input prevents actual and expected views from
        # sharing storage and hiding an incorrect mutation.
        self.mutate(actual, expected, lambda result: result[0].fill_(37))
        self.assertEqual(x, reference)
        self.mutate(actual, expected, lambda result: result[1].add_(17))
        self.assertEqual(x, reference)

    @parametrize("force_pointwise", [False, True])
    def test_channels_last_policy(self, device, force_pointwise):
        def fn(x):
            words = torch.cat((x + 1, x + 2), 1)
            return torch.cat((words, x + 3), 1)

        x = torch.arange(210, device=device, dtype=torch.float32).reshape(2, 3, 5, 7)
        x = x.contiguous(memory_format=torch.channels_last)
        actual, _ = self.check(
            fn,
            x,
            rebase=not force_pointwise,
            settings={
                "force_pointwise_cat": force_pointwise,
                "comprehensive_padding": False,
            },
        )
        self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))


class TestConcatRebasePartitions(TestCase):
    def setUp(self):
        super().setUp()
        self.graph = mock.Mock(sizevars=SizeVarAllocator(), name_to_buffer={})
        self.enterContext(V.set_graph_handler(self.graph))

    @parametrize(
        "case",
        [
            "reshape",
            "singleton_row",
            "expand_row",
            "transpose_row",
            "mix_batch",
            "symbolic_inner",
            "unknown_batch",
        ],
    )
    def test_symbolic_batch_reshape_stride(self, case):
        batch = sympy.Symbol("batch", integer=True, positive=True) + 1
        inner = sympy.Symbol("inner", integer=True, positive=True) + 1
        unknown = sympy.Symbol("unknown", integer=True)
        size, stride, shape, expected = {
            "reshape": ((batch, 2, 3), (19, 3, 1), (batch, 6), (19, 1)),
            "singleton_row": ((batch,), (7,), (batch, 1), (7, 1)),
            "expand_row": ((batch,), (7,), (batch, 2), None),
            "transpose_row": ((batch, 2, 3), (19, 1, 2), (batch, 6), None),
            "mix_batch": ((batch, 6), (6, 1), (2 * batch, 3), None),
            "symbolic_inner": (
                (batch, inner, 2),
                (2 * inner, 2, 1),
                (batch, 2 * inner),
                None,
            ),
            "unknown_batch": ((unknown, 2, 3), (19, 3, 1), (unknown, 6), None),
        }[case]
        self.assertEqual(concat_rebase._reshape_stride(size, stride, shape), expected)

    def view_buffer(self, name, owner, size, stride, offset):
        return ir.Buffer(
            name=name,
            layout=ir.NonOwningLayout(
                ir.ReinterpretView(
                    data=ir.StorageBox(owner),
                    layout=ir.FixedLayout(
                        torch.device("cpu"), torch.int32, size, stride, offset
                    ),
                )
            ),
        )

    @parametrize("mismatch", ["offset", "recorded_range"])
    def test_recorded_slice_rejects_canonical_view_mismatch(self, mismatch):
        owner = ir.ConcatKernel(
            name="owner",
            layout=ir.FixedLayout(torch.device("cpu"), torch.int32, [2, 5], [5, 1]),
            inputs=[],
            dim=1,
            slices=((0, 2), (2, 5)),
        )
        original = self.view_buffer("first", owner, [2, 2], [5, 1], 0)
        second = self.view_buffer("second", owner, [2, 3], [5, 1], 2)
        owner.inputs = [original, second]
        canonical = self.view_buffer("first", owner, [2, 2], [5, 1], 0)
        self.graph.name_to_buffer = {
            "owner": owner,
            "first": canonical,
            "second": second,
        }
        partitions = concat_rebase.get_partitions(owner, self.graph)
        self.assertIsNotNone(partitions)
        self.assertIs(partitions[0][0], canonical)
        self.assertEqual([part[1:] for part in partitions], [(0, 2), (2, 5)])

        layout = canonical.layout.view.get_layout()
        if mismatch == "offset":
            target, attribute, value = layout, "_offset", 1
        else:
            target, attribute, value = owner, "slices", ((0, 3), (3, 5))
        # Recorded ranges still tile the owner but disagree with the current views.
        with mock.patch.object(target, attribute, value):
            self.assertIsNone(concat_rebase.get_partitions(owner, self.graph))
        self.assertIs(owner.inputs[0], original)
        self.assertIs(concat_rebase.get_partitions(owner, self.graph)[0][0], canonical)

    def test_recorded_slice_accepts_reshaped_producer(self):
        owner = ir.ConcatKernel(
            name="owner",
            layout=ir.FixedLayout(torch.device("cpu"), torch.int32, [2, 8], [8, 1]),
            inputs=[],
            dim=1,
            slices=((0, 4), (4, 8)),
        )
        first = self.view_buffer("first", owner, [2, 2, 2], [8, 2, 1], 0)
        second = self.view_buffer("second", owner, [2, 4], [8, 1], 4)
        # Slice ownership does not require the producer and slice ranks to match.
        owner.inputs = [first, second]
        self.graph.name_to_buffer = {
            "owner": owner,
            "first": first,
            "second": second,
        }
        partitions = concat_rebase.get_partitions(owner, self.graph)
        self.assertIsNotNone(partitions)
        self.assertIs(partitions[0][0], first)
        self.assertIs(partitions[1][0], second)
        self.assertEqual([part[1:] for part in partitions], [(0, 4), (4, 8)])


class TestConcatRebasePurity(TestCase):
    def setUp(self):
        super().setUp()
        graph = mock.Mock(sizevars=SizeVarAllocator())
        graph.get_dtype.return_value = torch.int32
        self.enterContext(V.set_graph_handler(graph))

    def buffer(self, inner_fn):
        device = torch.device("cpu")
        return ir.ComputedBuffer(
            name="output",
            layout=ir.FixedLayout(device, torch.int32, [8]),
            data=ir.Pointwise(
                device=device, dtype=torch.int32, ranges=[8], inner_fn=inner_fn
            ),
        )

    @parametrize("pure", [False, True])
    @parametrize("masked", [False, True])
    def test_inline_asm_purity(self, pure, masked):
        def inner(index):
            def assembly():
                # Trace only: no assembly is compiled or executed by this test.
                return ops.inline_asm_elementwise(
                    ops.load("input", index[0]),
                    asm="/* traced only */",
                    constraints="=r,r",
                    dtype=torch.int32,
                    is_pure=pure,
                    pack=1,
                )

            if not masked:
                return assembly()
            mask = ops.lt(
                ops.index_expr(index[0], torch.int64), ops.constant(4, torch.int64)
            )
            mask = ops.and_(
                mask,
                ops.ge(
                    ops.index_expr(index[0], torch.int64), ops.constant(0, torch.int64)
                ),
            )
            return ops.masked(
                mask,
                lambda: ops.masked(mask, assembly, ops.constant(0, torch.int32)),
                ops.constant(0, torch.int32),
            )

        buffer = self.buffer(inner)
        if masked:
            self.assertEqual(len(buffer.get_default_sizes_body()[1].subblocks), 2)
        self.assertEqual(concat_rebase._pure_pointwise_leaf(buffer), pure)

    @parametrize("effect", ["extra_store", "atomic_store", "device_assert", "unknown"])
    def test_side_effects_rejected(self, effect):
        def inner(index):
            value = ops.load("input", index[0])
            if effect == "extra_store":
                ops.store("external", index[0], value)
            elif effect == "atomic_store":
                ops.store("output", index[0], value, mode="atomic_add")
            elif effect == "device_assert":
                ops.device_assert_async(
                    ops.ge(value, ops.constant(0, torch.int32)), "effect"
                )
            return value

        buffer = self.buffer(inner)
        if effect == "unknown":
            graph = buffer.get_default_sizes_body()[1].root_block.graph
            handler = next(node for node in graph.nodes if node.op == "placeholder")
            output = next(node for node in graph.nodes if node.op == "output")
            with graph.inserting_before(output):
                graph.call_method("unknown_effect", (handler,), {})
        self.assertFalse(concat_rebase._pure_pointwise_leaf(buffer))

    @parametrize(
        "case", ["private", "escaped", "mutated", "shared", "alias", "offset", "dtype"]
    )
    def test_materialized_pointwise_root(self, case):
        root = self.buffer(lambda index: ops.load("input", index[0]))
        root.operation_name = "producer"
        device = torch.device("cpu")
        owner = ir.ConcatKernel(
            name="destination",
            layout=ir.FixedLayout(device, torch.int32, [11]),
            inputs=[],
            dim=0,
            slices=(),
        )
        owner.operation_name = "copy"
        source = ir.ReinterpretView(
            data=ir.StorageBox(root),
            layout=ir.FixedLayout(
                device,
                torch.float32 if case == "dtype" else torch.int32,
                [8],
                [1],
                1 if case == "offset" else 0,
            ),
        )
        target = ir.SliceView.create(ir.StorageBox(owner), 0, 0, 8, clamp=False)
        graph = V.graph
        graph.name_to_buffer = {"output": root}
        graph.name_to_op = {"producer": root}
        graph.buffers = [root]
        graph.operations = [root]
        graph.graph_inputs = {}
        graph.mutated_buffers = {"output"} if case == "mutated" else set()
        graph.get_output_names.return_value = ["output"] if case == "escaped" else []
        if case == "alias":
            alias = mock.Mock()
            alias.get_name.return_value = "escaped_view"
            alias.get_mutation_names.return_value = []
            alias.get_inputs_that_alias_output.return_value = ["output"]
            graph.buffers.append(alias)
        readers = {"output": {"copy", "reader"} if case == "shared" else {"copy"}}
        old_layout = root.layout
        plan = concat_rebase._rebase_plan(
            source, target, (owner,), graph=graph, readers=readers
        )
        self.assertIs(root.layout, old_layout)  # Planning must not mutate storage.
        if case != "private":
            self.assertIsNone(plan)
        else:
            self.assertIsNotNone(plan)
            self.assertEqual(len(plan), 1)
            self.assertIs(plan[0][0], root)
            self.assertIs(concat_rebase._storage_buffer(plan[0][1].view), owner)

    def test_pointwise_realize_into_keeps_concat_owner(self):
        root = self.buffer(lambda index: ops.load("input", index[0]))
        source = ir.TensorBox(ir.StorageBox(root))
        original = root.data.inner_fn
        with config.patch("rebase_concat_copies", True):
            self.assertIs(concat_rebase.copy_loader(original, (source,), 0), original)

    def test_math_and_casts(self):
        buffer = self.buffer(
            lambda index: ops.to_dtype_bitcast(
                ops.to_dtype(
                    ops.add(ops.load("input", index[0]), ops.constant(1, torch.int32)),
                    torch.int32,
                ),
                torch.int32,
                torch.int32,
            )
        )
        self.assertTrue(concat_rebase._pure_pointwise_leaf(buffer))


instantiate_parametrized_tests(TestConcatRebaseCPU)
instantiate_parametrized_tests(TestConcatRebasePartitions)
instantiate_parametrized_tests(TestConcatRebasePurity)
instantiate_device_type_tests(TestConcatRebase, globals(), only_for="cuda")
instantiate_device_type_tests(
    TestConcatRebaseRobustness, globals(), only_for=("cpu", "cuda")
)

if __name__ == "__main__":
    run_tests()
