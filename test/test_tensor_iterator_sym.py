# Owner(s): ["module: fakeTensor"]
import contextlib
from unittest import mock

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    optimization_hint,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


LAYOUTS = {
    "contiguous": lambda: (torch.randn(4, 6), torch.randn(4, 6)),
    "transposed": lambda: (torch.randn(6, 4).t(), torch.randn(4, 6)),
    "both_transposed": lambda: (torch.randn(6, 4).t(), torch.randn(6, 4).t()),
    "broadcast_row": lambda: (torch.randn(4, 6), torch.randn(6)),
    "broadcast_outer": lambda: (torch.randn(4, 1), torch.randn(1, 6)),
    "scalar": lambda: (torch.randn(()), torch.randn(4, 6)),
    "permuted_3d": lambda: (
        torch.randn(4, 6, 8).permute(2, 0, 1),
        torch.randn(8, 4, 6),
    ),
    "channels_last": lambda: (
        torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last),
        torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last),
    ),
    "channels_last_and_contiguous": lambda: (
        torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last),
        torch.randn(2, 3, 4, 5),
    ),
    "size_one_dims": lambda: (
        torch.empty_strided((3, 1, 5), (5, 100, 1)).normal_(),
        torch.randn(3, 1, 5),
    ),
    "sliced": lambda: (torch.randn(4, 12)[:, ::2], torch.randn(4, 6)),
    "padded": lambda: (
        torch.empty_strided((4, 5, 6), (70, 12, 1)).normal_(),
        torch.randn(4, 5, 6),
    ),
}

# Inputs built inside a FakeTensorMode from unbacked sizes u (and v) and a
# backed size s.
UNBACKED = {
    "contiguous": lambda u, v, s: (torch.randn(u, 4), torch.randn(u, 4)),
    "transposed": lambda u, v, s: (torch.randn(4, u).t(), torch.randn(u, 4)),
    "broadcast_row": lambda u, v, s: (torch.randn(u, 4), torch.randn(4)),
    "broadcast_row_reversed": lambda u, v, s: (torch.randn(4), torch.randn(u, 4)),
    "broadcast_col": lambda u, v, s: (torch.randn(u, 1), torch.randn(u, 4)),
    "outer": lambda u, v, s: (torch.randn(u, 1), torch.randn(1, v)),
    "outer_same_u": lambda u, v, s: (torch.randn(u, 1), torch.randn(1, u)),
    "outer_same_u_reversed": lambda u, v, s: (torch.randn(1, u), torch.randn(u, 1)),
    "permuted_3d": lambda u, v, s: (
        torch.randn(u, 4, 5).permute(2, 0, 1),
        torch.randn(5, u, 4),
    ),
    "channels_last": lambda u, v, s: (
        torch.randn(2, 3, u, 5).contiguous(memory_format=torch.channels_last),
        torch.randn(2, 3, u, 5).contiguous(memory_format=torch.channels_last),
    ),
    "channels_last_and_contiguous": lambda u, v, s: (
        torch.randn(2, 3, u, 5).contiguous(memory_format=torch.channels_last),
        torch.randn(2, 3, u, 5),
    ),
    "mixed_outer": lambda u, v, s: (torch.randn(u, 1), torch.randn(1, s)),
    "mixed_transposed": lambda u, v, s: (torch.randn(s, u).t(), torch.randn(u, s)),
}

# Size u broadcast against a size-1 dim, in both operand orders.
BROADCAST_U_AGAINST_ONE = {
    "u1_x_14": lambda u, v, s: (torch.randn(u, 1), torch.randn(1, 4)),
    "14_x_u1": lambda u, v, s: (torch.randn(1, 4), torch.randn(u, 1)),
    "u4_x_14": lambda u, v, s: (torch.randn(u, 4), torch.randn(1, 4)),
    "14_x_u4": lambda u, v, s: (torch.randn(1, 4), torch.randn(u, 4)),
    "1u_x_41": lambda u, v, s: (torch.randn(1, u), torch.randn(4, 1)),
    "41_x_1u": lambda u, v, s: (torch.randn(4, 1), torch.randn(1, u)),
    "u1_x_1u": lambda u, v, s: (torch.randn(u, 1), torch.randn(1, u)),
    "1u_x_u1": lambda u, v, s: (torch.randn(1, u), torch.randn(u, 1)),
}

OPS = {
    "add": lambda a, b: torch.add(a, b, alpha=2),
    "mul": torch.mul,
    "sigmoid": lambda a, b: torch.sigmoid(a),
}

# (op on a float tensor a and an int tensor i, whether it reaches the Meta
# kernel). Fake sees Python scalars where eager had wrapped-number tensors;
# torch._C._ti_meta parses them against the op's schema.
DISPATCH_CASES = {
    "add_int": (lambda a, i: torch.add(a, 1), True),
    "add_float_to_int": (lambda a, i: torch.add(i, 1.5), True),
    "add_float_to_half": (lambda a, i: a.half() + 1.5, True),
    "add_alpha": (lambda a, i: torch.add(a, a, alpha=0.5), True),
    "add_scalar_alpha": (lambda a, i: torch.add(i, 3, alpha=2), True),
    "rmul_int": (lambda a, i: torch.mul(2, a), True),
    "rmul_bool": (lambda a, i: True * a, True),
    "add_tensor_overload": (lambda a, i: torch.ops.aten.add.Tensor(a, 1), True),
    "add_scalar_overload": (lambda a, i: torch.ops.aten.add.Scalar(a, 1), False),
    "add_out": (lambda a, i: torch.add(a, i, out=torch.empty(0)), False),
}


@contextlib.contextmanager
def recorder():
    prev = torch._C._ht_set_recorder(True)
    try:
        yield
    finally:
        torch._C._ht_set_recorder(prev)


def run_fake(op, layout, use_symint_ti):
    shape_env = ShapeEnv()
    with torch._functorch.config.patch(
        fake_tensor_symint_tensor_iterator=use_symint_ti
    ):
        mode = FakeTensorMode(shape_env=shape_env)
    args = [
        mode.from_tensor(
            t,
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC] * t.dim()
            ),
        )
        for t in LAYOUTS[layout]()
    ]
    with mode:
        out = OPS[op](*args)
    return out, shape_env


def run_unbacked(op, case, use_symint_ti, cases=UNBACKED):
    shape_env = ShapeEnv()
    with torch._functorch.config.patch(
        fake_tensor_symint_tensor_iterator=use_symint_ti
    ):
        mode = FakeTensorMode(shape_env=shape_env)
    s = mode.from_tensor(
        torch.randn(6),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    ).size(0)
    with mode:
        u = mode.from_tensor(torch.tensor([1, 0, 1, 1])).nonzero().size(0)
        v = mode.from_tensor(torch.tensor([1, 1, 0])).nonzero().size(0)
        args = cases[case](u, v, s)
        before = runtime_asserts(shape_env)
        out = OPS[op](*args)
    return out, shape_env, sorted(set(runtime_asserts(shape_env)) - set(before))


def dispatch_inputs():
    return torch.randn(6, 4).t(), torch.ones(4, 6, dtype=torch.int32)


def run_dispatch(fn, use_symint_ti):
    with torch._functorch.config.patch(
        fake_tensor_symint_tensor_iterator=use_symint_ti
    ):
        mode = FakeTensorMode(shape_env=ShapeEnv())
    ctx = StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC] * 2)
    args = [mode.from_tensor(t, symbolic_context=ctx) for t in dispatch_inputs()]
    with mode:
        return fn(*args)


def runtime_asserts(shape_env):
    return [
        str(ra.expr) for rs in shape_env.deferred_runtime_asserts.values() for ra in rs
    ]


def hints(xs):
    return [optimization_hint(x) for x in xs]


class TestTensorIteratorSym(TestCase):
    @parametrize("op", list(OPS))
    @parametrize("layout", list(LAYOUTS))
    def test_fake_matches_eager(self, op, layout):
        eager = OPS[op](*LAYOUTS[layout]())
        out, _ = run_fake(op, layout, use_symint_ti=True)
        self.assertEqual(out.dtype, eager.dtype)
        self.assertEqual(out.device, eager.device)
        self.assertEqual(hints(out.shape), list(eager.shape))
        self.assertEqual(hints(out.stride()), list(eager.stride()))

    @parametrize("op", list(OPS))
    @parametrize("layout", list(LAYOUTS))
    def test_kernel_choice_guards(self, op, layout):
        # Fake returns before the kernel context, so the guards coalescing
        # raises with a recorder never reach ShapeEnv; the graph-level guards
        # are the same either way.
        with recorder():
            _, traced = run_fake(op, layout, use_symint_ti=True)
        _, fake = run_fake(op, layout, use_symint_ti=True)
        self.assertEqual(fake.kernel_choice_guards, [])
        graph_guards = [str(g.expr) for g in fake.guards]
        self.assertEqual([str(g.expr) for g in traced.guards], graph_guards)
        for g in traced.kernel_choice_guards:
            self.assertNotIn(str(g.expr), graph_guards)

    def test_kernel_choice_guards_recorded(self):
        # Coalescing the padded input's two outer dims compares its free
        # outer stride symbol against the others; nothing before the boundary
        # asks that.
        with recorder():
            _, shape_env = run_fake("add", "padded", use_symint_ti=True)
        self.assertNotEqual(shape_env.kernel_choice_guards, [])

    @parametrize("op", list(OPS))
    @parametrize("case", list(UNBACKED))
    def test_unbacked_matches_python_refs(self, op, case):
        ref, ref_env, ref_asserts = run_unbacked(op, case, use_symint_ti=False)
        out, out_env, out_asserts = run_unbacked(op, case, use_symint_ti=True)
        self.assertEqual(str(out.shape), str(ref.shape))
        self.assertEqual(str(out.stride()), str(ref.stride()))
        self.assertEqual(str(out_env.guards), str(ref_env.guards))
        self.assertEqual(out_asserts, ref_asserts)

    @parametrize("case", list(BROADCAST_U_AGAINST_ONE))
    def test_unbacked_broadcast_against_one_adds_no_asserts(self, case):
        for use_symint_ti in (False, True):
            _, shape_env, asserts = run_unbacked(
                "add", case, use_symint_ti, BROADCAST_U_AGAINST_ONE
            )
            self.assertEqual(shape_env.guards, [])
            self.assertEqual(asserts, [])

    @parametrize("case", list(DISPATCH_CASES))
    def test_dispatch_case_matches_eager(self, case):
        fn, routed = DISPATCH_CASES[case]
        eager = fn(*dispatch_inputs())
        with mock.patch.object(
            torch._C, "_ti_meta", wraps=torch._C._ti_meta
        ) as ti_meta:
            out = run_dispatch(fn, use_symint_ti=True)
        self.assertEqual(ti_meta.call_count, int(routed))
        # Overloads that are not routed keep flag-off behavior, which can
        # differ from eager (out= restrides contiguous).
        ref = eager if routed else run_dispatch(fn, use_symint_ti=False)
        self.assertEqual(out.dtype, ref.dtype)
        self.assertEqual(hints(out.shape), hints(ref.shape))
        self.assertEqual(hints(out.stride()), hints(ref.stride()))

    def test_alpha_check(self):
        # The Meta kernel runs alpha_check as eager does; the Python ref
        # (flag off) accepts a float alpha on integer inputs.
        def fn(a, i):
            return torch.add(i, i, alpha=0.5)

        msg = "alpha must not be a floating point number"
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(*dispatch_inputs())
        with self.assertRaisesRegex(RuntimeError, msg):
            run_dispatch(fn, use_symint_ti=True)
        self.assertEqual(run_dispatch(fn, use_symint_ti=False).dtype, torch.int32)

    def test_unbacked_size_one_takes_non_broadcast_path(self):
        # u may be 1 at runtime; like the Python refs, TensorIterator assumes
        # it is not and defers a runtime assert that u equals its partner.
        shape_env = ShapeEnv()
        with torch._functorch.config.patch(fake_tensor_symint_tensor_iterator=True):
            mode = FakeTensorMode(shape_env=shape_env)
        with mode:
            u = mode.from_tensor(torch.tensor([1, 0, 1, 1])).nonzero().size(0)
            out = torch.randn(u, 4) + torch.randn(3, 4)
        self.assertEqual(out.shape, (3, 4))
        ras = shape_env.deferred_runtime_asserts.values()
        self.assertIn("Eq(u0, 3)", [str(ra.expr) for rs in ras for ra in rs])


instantiate_parametrized_tests(TestTensorIteratorSym)

if __name__ == "__main__":
    run_tests()
