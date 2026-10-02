# Owner(s): ["module: cuda graphs"]

import unittest

import sympy

import torch
from torch.cuda._host_trace import Declined
from torch.cuda._host_trace_tape import (
    _hint,
    _IntOutputRec,
    _output_records,
    _symbolic_run,
    _Trace,
    _TracedTensor,
    current_trace,
    EagerCall,
    Memset,
    trace,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    requires_cuda_python_bindings,
    run_tests,
    TEST_CUDA,
    TestCase,
)
from torch.utils._triton import has_triton


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _copy(x_ptr, y_ptr, n, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        tl.store(y_ptr + i, tl.load(x_ptr + i, mask=i < n), mask=i < n)


# an operator with a CPU kernel and no meta
_lib = torch.library.Library("host_trace_test", "FRAGMENT")
_lib.define("no_meta(Tensor x) -> Tensor")
_lib.impl("no_meta", lambda x: x.clone(), "CPU")


# a CUDA kernel the trace does not follow: an eager call, whose metadata
# comes from its fake kernel
_lib.define("tape_transposed_fake(Tensor x) -> Tensor")
_lib.impl("tape_transposed_fake", lambda x: x.clone(), "CUDA")
transposed_fake = torch.ops.host_trace_test.tape_transposed_fake


@torch.library.register_fake("host_trace_test::tape_transposed_fake")
def _(x):
    return x.new_empty(x.shape[::-1]).t()


def _run(fn, *args):
    # the symbolic run on the CPU, without trace()'s capture
    tr = _Trace(torch.device("cpu"))
    positions = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
    ints = [i for i, a in enumerate(args) if type(a) is int]
    out, traced = _symbolic_run(tr, fn, args, positions, ints)
    return tr, out, traced


def _hints(values):
    return [_hint(v) for v in values]


def _holds(tr, values):
    # whether every guard holds with the symbols named in `values` (by source,
    # "arg0.size(0)") at those values and every other symbol at its hint
    env = tr.shape_env
    m = {}
    for s, hint in env.backed_var_to_val.items():
        m[s] = sympy.Integer(values.get(env.var_to_sources[s][0].name, hint))
    return all(g.expr.xreplace(m) is sympy.true for g in env.guards)


def _same_view(test, got, want, root):
    test.assertIsInstance(got, _TracedTensor)
    test.assertIs(got._root, root)
    test.assertEqual(_hints(got.shape), list(want.shape))
    test.assertEqual(_hints(got._sym_strides), list(want.stride()))
    test.assertEqual(_hint(got._sym_offset), want.storage_offset())
    test.assertEqual(got.dtype, want.dtype)


class TestSymbolicRun(TestCase):
    def test_allocations_lay_out_as_eager(self):
        x = torch.randn(4, 6)
        cases = {
            "empty": (lambda t: torch.empty(t.shape[0], 3, t.shape[1]), x),
            "channels_last": (
                lambda t: torch.empty(
                    2, 3, *t.shape, memory_format=torch.channels_last
                ),
                x,
            ),
            "empty_strided": (
                lambda t: torch.empty_strided((t.shape[0], 2), (1, t.shape[0])),
                x,
            ),
            "new_empty": (lambda t: t.new_empty((t.shape[1], 2), dtype=torch.half), x),
            "empty_like dense": (torch.empty_like, x.t()),
            # a strided input that is not dense keeps its dim order
            "empty_like not dense": (torch.empty_like, x.t()[::2]),
        }
        for name, (fn, arg) in cases.items():
            tr, out, _ = _run(fn, arg)
            want = fn(arg)
            self.assertEqual(_hints(out.shape), list(want.shape), name)
            self.assertEqual(_hints(out._sym_strides), list(want.stride()), name)
            self.assertEqual(out.dtype, want.dtype, name)
            (alloc,) = tr.allocs
            self.assertEqual(alloc.root.sym.node.expr, 256 * alloc.q.node.expr)
            self.assertEqual((alloc.name, out._root.name), ("alloc0", "a0"))

    def test_allocation_symbols_are_distinct_and_in_program_order(self):
        def fn(t):
            a = torch.empty_like(t)
            current_trace().record_launch("k0")
            b = torch.empty_like(t)
            return a, b

        tr, (a, b), _ = _run(fn, torch.randn(8))
        self.assertEqual([r.seq for r in tr.allocs], [0, 2])
        self.assertEqual(tr.launches, [(1, "k0")])
        self.assertNotEqual(a.data_ptr().node.hint, b.data_ptr().node.hint)
        self.assertEqual(a.data_ptr().node.hint % 256, 0)
        # empty_like's density test, recorded once
        (rec,) = tr.inputs
        (size,), (stride,) = [
            [v.node.expr for v in vs] for vs in (rec.sizes, rec.strides)
        ]
        self.assertEqual(
            [g.expr for g in tr.shape_env.guards],
            [sympy.Eq(size, 1) | sympy.Eq(stride, 1)],
        )

    def test_int_arguments_are_symbols(self):
        def fn(t, n):
            out = torch.empty(n, t.shape[0])
            return out if n > 2 else t

        tr, out, _ = _run(fn, torch.randn(5), 4)
        (rec,) = tr.int_inputs
        n = rec.sym.node.expr
        self.assertEqual((rec.position, rec.name), (1, "arg1"))
        self.assertEqual(out.shape[0].node.expr, n)
        # the allocation checks its size is nonnegative (and the wrapper
        # construction tests it for zero); the branch is a guard
        self.assertEqual(
            [g.expr for g in tr.shape_env.guards], [n >= 0, sympy.Ne(n, 0), n > 2]
        )
        # a bool is a constant
        self.assertEqual(
            _run(lambda t, flag: t, torch.randn(2), True)[0].int_inputs, []
        )

    def test_metadata_queries_are_guards(self):
        x = torch.randn(4, 6)

        def fn(t, tt):
            return t.is_contiguous(), tt.is_contiguous(), t.stride(), t.storage_offset()

        tr, (c, ct, strides, offset), _ = _run(fn, x, x.t())
        self.assertEqual((c, ct, strides), (True, False, (6, 1)))
        self.assertIsInstance(offset, torch.SymInt)
        self.assertTrue(tr.shape_env.guards)
        self.assertTrue(
            all(isinstance(g.expr, sympy.Basic) for g in tr.shape_env.guards)
        )

    def test_data_ptr_is_the_address_symbol(self):
        x = torch.randn(4, 8)[1:]

        def fn(t):
            return t.data_ptr() % 16 == 0

        tr, aligned, (t,) = _run(fn, x)
        self.assertEqual(bool(aligned), x.data_ptr() % 16 == 0)
        (rec,) = tr.inputs
        self.assertEqual(
            t.data_ptr().node.expr, rec.root.sym.node.expr + 4 * rec.offset.node.expr
        )
        # the placeholder keeps the address's low bits
        self.assertEqual(t.data_ptr().node.hint % (1 << 52), x.data_ptr() % (1 << 52))

    def test_host_reads_and_other_ops_decline(self):
        x = torch.randn(1)
        cases = {
            r"bool\(\)": lambda t: bool(t),
            "tolist": lambda t: t.tolist(),
            "_local_scalar_dense": lambda t: t.item(),
            "DLPack": lambda t: t.__dlpack__(),
            "aten.nonzero.default has no traced metadata": lambda t: t.nonzero(),
            "no_meta.default has no traced metadata": torch.ops.host_trace_test.no_meta,
            "aten.add.Tensor of a tensor the trace does not track": lambda t: t + x,
            "aten._to_copy.default returns a tensor on meta": lambda t: t.to("meta"),
            "aten.equal.default has no traced metadata": lambda t: torch.equal(t, t),
            "aten.is_same_size.default returns a bool": lambda t: t.is_same_size(t),
            "aten.resize_.default changes": lambda t: t.resize_(2),
            "an allocation on meta": lambda t: torch.empty(4, device="meta"),
            "a pinned allocation": lambda t: torch.empty(4, pin_memory=True),
        }
        for why, fn in cases.items():
            with self.assertRaisesRegex(Declined, why):
                _run(fn, x)
        with self.assertRaisesRegex(Declined, "format"):
            _run(lambda t: f"{t:.2f}", torch.tensor(1.0))

    def test_a_caught_decline_is_sticky(self):
        # PYREC-B1: host code that catches the decline and takes another path
        # would trace a path eager does not take
        def fn(t):
            try:
                t.nonzero()
            except RuntimeError:
                pass
            return torch.empty_like(t)

        with self.assertRaisesRegex(Declined, "aten.nonzero"):
            _run(fn, torch.randn(3, 3))

        def reraise(t):
            try:
                t.nonzero()
            except RuntimeError as e:
                raise ValueError("another error") from e

        with self.assertRaisesRegex(Declined, "aten.nonzero"):
            _run(reraise, torch.randn(3))

    def test_eager_calls(self):
        x, w = torch.randn(8, 4), torch.randn(4, 6)
        tr, out, (a, b) = _run(torch.mm, x, w)
        ((_, call),) = tr.launches
        self.assertEqual(call.target, torch.ops.aten.mm.default)
        self.assertEqual(len(call.args), 2)
        self.assertIs(call.args[0], a)
        self.assertIs(call.args[1], b)
        self.assertEqual(call.kwargs, {})
        self.assertIs(call.outputs[0], out)
        self.assertEqual((out._root.name, out._root.kind), ("e0", "eager"))
        # the output's metadata is mm's fake kernel's, over the inputs' symbols
        self.assertEqual(out.shape[0].node.expr, a.shape[0].node.expr)
        self.assertEqual(out.shape[1].node.expr, b.shape[1].node.expr)
        self.assertEqual(_hints(out._sym_strides), [6, 1])
        self.assertFalse(_holds(tr, {"arg1.size(0)": 5}))  # mm's inner dims

        # in-place and out= return the argument itself
        tr, out, (a,) = _run(lambda t: t.add_(1), x)
        ((_, call),) = tr.launches
        self.assertIs(out, a)
        self.assertIs(call.outputs[0], a)
        tr, out, (a, o) = _run(
            lambda t, u: torch.add(t, 1, out=u), x, torch.empty(8, 4)
        )
        ((_, call),) = tr.launches
        self.assertIs(out, o)
        with self.assertRaisesRegex(Declined, "aten.add.out resizes its out= argument"):
            _run(lambda t, u: torch.add(t, 1, out=u), x, torch.empty(32))
        # the eager outputs, two per call here, are fresh roots in program order
        tr, out, _ = _run(lambda t: torch.max(t * 2, 0), x)
        self.assertEqual([t._root.name for t in out], ["e1", "e2"])
        self.assertIsInstance(out, torch.return_types.max)

    def test_a_view_of_an_eager_output(self):
        x, w = torch.randn(8, 4), torch.randn(4, 6)
        tr, out, _ = _run(lambda a, b: (a @ b)[1:, 2:], x, w)
        (e0,) = tr.eager_outputs
        self.assertIs(out._root, e0._root)
        self.assertEqual(_hint(out._sym_offset), 8)
        base = e0._root.sym.node.expr
        self.assertEqual(out.data_ptr().node.expr, base + 4 * out._sym_offset.node.expr)
        # its storage is bounded by the output's end, as an input's
        _run(lambda a, b: (a @ b).as_strided((4, 12), (12, 1)), x, w)
        with self.assertRaisesRegex(Declined, "past the end of eager output e0"):
            _run(lambda a, b: (a @ b).as_strided((8, 6), (6, 1), 1), x, w)

    def test_deterministic_fill_declines(self):
        with DeterministicGuard(True, fill_uninitialized_memory=True):
            with self.assertRaisesRegex(Declined, "fill_uninitialized_memory"):
                _run(torch.empty_like, torch.randn(3))
        with DeterministicGuard(True, fill_uninitialized_memory=False):
            _run(torch.empty_like, torch.randn(3))

    def test_views_match_eager(self):
        x = torch.randn(6, 5, 8)[1:]
        z = torch.randn(6, 8, dtype=torch.complex64)[:, 2:]
        heads = torch.randn(4, 1, 48)
        cases = {
            "view": (lambda t: t[:, 1:3].view(5, 2, 2, 4), x),
            # computeStride's strides on size-1 dims and empty views
            "view size-1 dim": (lambda t: t.narrow(2, 16, 16).view(t.shape[0], 1, 4, 4), heads),
            "view of size-1 dims": (lambda t: t.as_strided((1, 1, 1, 4), (7, 7, 1, 1), 5).view(-1, 4), x),
            "empty view": (lambda t: t[:, :0].view(t.shape[0], 0, 8), x),
            "reshape": (lambda t: t.reshape(-1, 8), x),
            "flatten": (lambda t: t.flatten(1), x),
            "slice": (lambda t: t[:, 1:4, ::3], x),
            "select": (lambda t: t.select(2, -1), x),
            "transpose": (lambda t: t.transpose(0, 2), x),
            "permute": (lambda t: t.permute(2, 0, 1), x),
            "movedim": (lambda t: t.movedim(0, -1), x),
            "expand": (lambda t: t[:, :1].expand(5, 3, 8), x),
            "expand leading": (lambda t: t[:1].expand(1, 1, 5, 8), x),
            "unsqueeze": (lambda t: t.unsqueeze(1), x),
            "squeeze": (lambda t: t[:, :1].squeeze(1), x),
            "split": (lambda t: t.split(3, 2), x),
            "split_with_sizes": (lambda t: t.split([1, 4], 1), x),
            "chunk": (lambda t: t.chunk(2, 0), x),
            "unbind": (lambda t: t.unbind(1), x),
            "dtype": (lambda t: t.view(torch.int16), x),
            "as_strided": (lambda t: t.as_strided((4, 4), (2, 1), 3), x),
            "diagonal": (lambda t: t.diagonal(0, 1, 2), x),
            "unfold": (lambda t: t.unfold(2, 3, 2), x),
            "detach": (lambda t: t.detach(), x),
            "_unsafe_view": (lambda t: torch.ops.aten._unsafe_view(t, [25, 8]), x),
            "view_as_real": (torch.view_as_real, z),
            "view_as_complex": (
                lambda t: torch.view_as_complex(t.view(5, 5, 4, 2)),
                x,
            ),
        }
        for name, (fn, arg) in cases.items():
            with self.subTest(name):
                tr, out, _ = _run(fn, arg)
                want = fn(arg)
                outs = out if isinstance(out, (tuple, list)) else (out,)
                wants = want if isinstance(want, (tuple, list)) else (want,)
                self.assertEqual(len(outs), len(wants))
                for got, w in zip(outs, wants):
                    _same_view(self, got, w, tr.inputs[0].root)

    def test_a_view_of_an_allocation_and_its_output_record(self):
        def fn(t):
            return torch.empty(t.shape[0], 16)[1:, 4:].t()

        tr, out, traced = _run(fn, torch.randn(8))
        _same_view(self, out, torch.empty(8, 16)[1:, 4:].t(), tr.allocs[0].root)
        _, (rec,) = _output_records(out, traced, [0])
        self.assertEqual((rec.root.name, rec.offset), ("a0", 20))
        self.assertEqual(out.data_ptr(), tr.allocs[0].root.sym + 80)

    def test_size_one_branches_are_guarded_on_the_traced_value(self):
        def fn(t):
            return t.squeeze(0)

        for traced_at, other in ((1, 8), (8, 1)):
            tr, out, _ = _run(fn, torch.randn(traced_at, 4))
            self.assertEqual(out.dim(), 1 if traced_at == 1 else 2)
            self.assertTrue(_holds(tr, {"arg0.size(0)": traced_at}))
            self.assertFalse(_holds(tr, {"arg0.size(0)": other}))

    def test_symbolic_slices_and_selects_do_not_pin(self):
        cases = {
            "slice": lambda t: t[1 : t.shape[0] - 1],
            "open slice": lambda t: t[:, t.shape[1] // 2 :],
            "select": lambda t: t.select(0, t.shape[0] - 1),
            "narrow": lambda t: t.narrow(0, 1, t.shape[0] - 2),
            "view": lambda t: t.view(t.shape[0] // 2, -1),
        }
        x = torch.randn(16, 8)
        for name, fn in cases.items():
            with self.subTest(name):
                tr, out, _ = _run(fn, x)
                _same_view(self, out, fn(x), tr.inputs[0].root)
                other = {"arg0.size(0)": 6, "arg0.size(1)": 4, "arg0.stride(0)": 4}
                self.assertTrue(_holds(tr, other))

    def test_symbolic_integer_indices_do_not_pin(self):
        x = torch.randn(16, 8)
        tr, _, _ = _run(lambda t: t[t.shape[0] - 1], x)
        if not _holds(tr, {"arg0.size(0)": 6}):
            # TODO: drop once every binary this runs on has the fix
            self.skipTest("Tensor.__getitem__ turns a SymInt index into an int")
        cases = {
            "getitem": lambda t: t[t.shape[0] - 1],
            "negative": lambda t: t[1 - t.shape[0]],
            "in a tuple": lambda t: t[:, t.shape[1] - 1],
        }
        for name, fn in cases.items():
            with self.subTest(name):
                tr, out, _ = _run(fn, x)
                _same_view(self, out, fn(x), tr.inputs[0].root)
                other = {"arg0.size(0)": 6, "arg0.size(1)": 4, "arg0.stride(0)": 4}
                self.assertTrue(_holds(tr, other))

    def test_failing_views(self):
        x = torch.randn(8, 4)
        # a check of a composite's C++ body raises eager's error
        for fn in (
            lambda t: t.narrow(0, 0, -1),
            lambda t: t.narrow(0, 0, t.shape[0] - 9),
        ):
            with self.assertRaisesRegex(RuntimeError, "must be non-negative"):
                fn(x)
            with self.assertRaisesRegex(RuntimeError, "must be non-negative") as cm:
                _run(fn, x)
            self.assertNotIsInstance(cm.exception, Declined)
        # an error of a fake kernel declines: its type need not be eager's
        cases = [
            ("aten.view.default raised", lambda t: t.t().view(32)),
            ("aten.view.default raised", lambda t: t.view(0, -1)),
            ("aten.select.int raised", lambda t: t.select(0, t.shape[0])),
        ]
        for why, fn in cases:
            with self.assertRaises(Exception):
                fn(x)
            with self.assertRaisesRegex(Declined, why) as cm:
                _run(fn, x)
            self.assertNotIsInstance(cm.exception.__cause__, ZeroDivisionError)

    def test_an_empty_view_has_a_null_address_like_eager(self):
        x = torch.randn(8, 4)
        self.assertEqual(x[:0].data_ptr(), 0)

        def fn(t):
            views = (t[:0], t[:, :0], t[3:3], t[8:], t[:1])
            return [v.data_ptr() for v in views]

        tr, addresses, _ = _run(fn, x)
        self.assertEqual(addresses[:4], [0] * 4)
        self.assertIsInstance(addresses[4], torch.SymInt)
        # t[8:] is empty at the traced size only
        self.assertFalse(_holds(tr, {"arg0.size(0)": 9}))

    def test_as_strided_checks_its_storage(self):
        x = torch.randn(8, 4)
        tr, out, _ = _run(lambda t: torch.empty(t.shape).as_strided((4, 8), (8, 1)), x)
        self.assertIs(out._root, tr.allocs[0].root)
        empty = _run(lambda t: torch.empty(t.shape).as_strided((0, 4), (100, 1), 64), x)
        self.assertEqual(empty[1].shape, (0, 4))
        # eager's setStrided checks, which the fake kernel skips for symbolic
        # values; t.shape[1] - 5 is -1
        cases = {
            "out of bounds": lambda t: torch.empty(t.shape).as_strided(
                (4, 8), (8, 1), 1
            ),
            "negative stride": lambda t: torch.empty(t.shape).as_strided(
                (4,), (t.shape[1] - 5,), 4
            ),
            "negative offset": lambda t: torch.empty(t.shape).as_strided(
                (4,), (1,), t.shape[1] - 5
            ),
        }
        for name, fn in cases.items():
            with self.subTest(name):
                with self.assertRaises(RuntimeError):
                    fn(x)
                with self.assertRaises(RuntimeError) as cm:
                    _run(fn, x)
                self.assertNotIsInstance(cm.exception, Declined)
        # an input's storage size is not traced: within the bytes it spans
        # traces, past them declines
        _run(lambda t: t.as_strided((2, 4), (4, 1), 8), x[:4])
        x[:4].as_strided((2, 4), (4, 1), 16)
        with self.assertRaisesRegex(Declined, "past the end of arg0"):
            _run(lambda t: t.as_strided((2, 4), (4, 1), 16), x[:4])

    def test_views_that_decline(self):
        x = torch.randn(4, 4)
        w = torch.randn(4, 1)
        cases = {
            "transpose_": lambda t: t.transpose_(0, 1),
            "as_strided_": lambda t: t.as_strided_((4,), (1,)),
            "untraced tensor with symbolic": lambda t: w.expand(4, t.shape[0]),
            "not a plain strided view": lambda t: t.to(torch.complex64).conj(),
        }
        for why, fn in cases.items():
            with self.assertRaisesRegex(Declined, why):
                _run(
                    fn,
                    x if why != "not a plain strided view" else x.to(torch.complex64),
                )
        # a view of a tensor the trace does not own, with constant arguments,
        # is that tensor's eager view
        _, out, _ = _run(lambda t: (t, w.t()), x)
        self.assertEqual(out[1], w.t())

    def test_output_records(self):
        x, y = torch.randn(3), torch.randn(3)

        def fn(a, b):
            out = torch.empty_like(a)
            return b, out, out

        tr, out, traced = _run(fn, x, y)
        kind, records = _output_records(out, traced, [0, 1])
        self.assertEqual(kind, "tuple")
        self.assertEqual(
            [r.identity for r in records], [("argument", 1), None, ("output", 1)]
        )
        self.assertEqual([r.root.name for r in records], ["p1", "a0", "a0"])
        self.assertEqual(_output_records(None, traced, [0, 1]), ("none", []))
        kind, records = _output_records([out[1], 3], traced, [0, 1])
        self.assertEqual(kind, "list")
        self.assertIsInstance(records[1], _IntOutputRec)
        self.assertEqual(records[1].value, 3)
        with self.assertRaisesRegex(Declined, "returned int"):
            _output_records(3, traced, [0, 1])
        with self.assertRaisesRegex(Declined, "not a tensor of the trace"):
            _output_records([x], traced, [0, 1])


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
class TestTrace(TestCase):
    def test_declines(self):
        x = torch.randn(4, device="cuda")
        fn = torch.empty_like
        cases = {
            "no tensor arguments": (3,),
            "only CUDA tensors": (x.cpu(),),
            "negative or conjugate": (torch._neg_view(x),),
        }
        for why, args in cases.items():
            with self.assertRaisesRegex(Declined, why):
                trace(fn, args)
        # a trace inside a trace sees traced tensors
        with self.assertRaisesRegex(Declined, "another trace"):
            trace(lambda t: trace(fn, (t,)), (x,), warm_up=False)
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            with self.assertRaisesRegex(Declined, "capturing"):
                trace(fn, (x,), warm_up=False)
            x.add_(0)  # the graph is not empty

    def test_tape(self):
        x = torch.randn(8, 16, device="cuda")

        def fn(t, n, eps, flag):
            out = torch.empty(n, t.shape[1], device=t.device)
            return t, out

        tape = trace(fn, (x, 4, 1e-5, True))
        self.assertEqual([i.position for i in tape.inputs], [0])
        self.assertEqual(len(tape.allocs), 1)
        self.assertEqual(tape.result_kind, "tuple")
        self.assertEqual([o.identity for o in tape.outputs], [("argument", 0), None])
        self.assertIs(tape.warm_up_result[0], x)
        self.assertEqual(tape.contract[1], torch._C._host_trace_global_state())
        sources = {s[0].name for s in tape.shape_env.var_to_sources.values()}
        self.assertTrue({"arg0.size(0)", "arg0.base", "arg1", "alloc0.base/256"} <= sources)
        # -0.0 is another constant
        self.assertNotEqual(trace(fn, (x, 4, 0.0, True)).contract, trace(fn, (x, 4, -0.0, True)).contract)

    def test_warm_up(self):
        x = torch.randn(4, device="cuda")
        calls = []

        def fn(t):
            calls.append(isinstance(t, _TracedTensor))
            return torch.empty_like(t)

        trace(fn, (x,))
        self.assertEqual(calls, [False, True])
        calls.clear()
        tape = trace(fn, (x,), warm_up=False)
        self.assertEqual(calls, [True])
        self.assertIsNone(tape.warm_up_result)

        def resizes(t):
            t.resize_(8)
            return t

        with self.assertRaisesRegex(
            Declined, "warm-up changed the metadata of arg0"
        ) as cm:
            trace(resizes, (torch.randn(4, device="cuda"),))
        self.assertTrue(cm.exception.warm_up_ran)
        self.assertEqual(cm.exception.warm_up_result.shape, (8,))
        with self.assertRaises(Declined) as cm:
            trace(lambda t: t.nonzero(), (x,))
        self.assertTrue(cm.exception.warm_up_ran)
        self.assertEqual(cm.exception.warm_up_result, x.nonzero())

    def test_the_warm_up_checks_eager_metadata(self):
        x = torch.randn(8, 16, device="cuda")
        with self.assertRaisesRegex(
            Declined,
            r"tape_transposed_fake.default output 0 .*\(16, 1\).* at the warm-up; its fake kernel predicted .*\(1, 8\)",
        ) as cm:
            trace(transposed_fake, (x,))
        op = torch.ops.host_trace_test.tape_transposed_fake.default
        self.assertIs(cm.exception.meta_op, op)
        self.assertTrue(cm.exception.warm_up_ran)
        self.assertEqual(cm.exception.warm_up_result, x)
        # without a warm-up there is nothing to check against
        trace(transposed_fake, (x,), warm_up=False)

    def test_eager_call_in_the_tape(self):
        x, w = torch.randn(8, 4, device="cuda"), torch.randn(4, 6, device="cuda")
        tape = trace(torch.mm, (x, w))
        ((_, call),) = tape.launches
        self.assertIs(call.target, torch.ops.aten.mm.default)
        self.assertEqual([a._root.name for a in call.args], ["p0", "p1"])
        self.assertEqual(call.outputs[0]._root.name, "e0")
        sources = {s[0].name for s in tape.shape_env.var_to_sources.values()}
        self.assertIn("eager0.base/256", sources)
        self.assertEqual([o.root.name for o in tape.outputs], ["e0"])

        def side_stream(t):
            with torch.cuda.stream(torch.cuda.Stream()):
                return t + 1

        with self.assertRaisesRegex(Declined, "on a stream other than the trace's"):
            trace(side_stream, (x,))

    def test_zeroing_an_allocation_is_a_memset(self):
        x = torch.randn(8, 4, device="cuda")

        def fn(t):
            w = torch.empty(t.shape[0], 3, device=t.device).zero_()
            v = torch.empty_like(t)
            v[1:].zero_()
            return w, v

        tape = trace(fn, (x,))
        (memset, view) = [rec for _, rec in tape.launches]
        self.assertIsInstance(memset, Memset)
        self.assertEqual(memset.roots, (tape.allocs[0].root,))
        self.assertEqual((memset.value, memset.element_size, memset.height), (0, 1, 1))
        self.assertEqual(int(memset.width), 8 * 3 * 4)
        # a contiguous view's zero_ is a memset from its offset
        self.assertIsInstance(view, Memset)
        self.assertEqual(view.roots, (tape.allocs[1].root,))
        self.assertEqual(int(view.width), 7 * 4 * 4)

    def test_synchronizing_declines(self):
        x = torch.randn(4, device="cuda")

        def syncs(t):
            torch.cuda.synchronize()
            return t

        def swallows(t):
            try:
                torch.cuda.current_stream().synchronize()
            except RuntimeError:
                pass
            return t

        for fn in (syncs, swallows):
            with self.assertRaisesRegex(Declined, "stream capture does not permit"):
                trace(fn, (x,), warm_up=False)
            self.assertFalse(torch.cuda.is_current_stream_capturing())

    def test_unrecorded_work_declines(self):
        # a kernel launched without dispatch, which no interceptor records
        def fn(t):
            torch.cuda._sleep(1)
            return t

        with self.assertRaisesRegex(Declined, "enqueued 1 operations"):
            trace(fn, (torch.randn(4, device="cuda"),), warm_up=False)

    def test_an_exception_closes_the_capture(self):
        x = torch.randn(4, device="cuda")

        def fn(t):
            torch.empty_like(t)
            raise ValueError("after an allocation")

        with self.assertRaisesRegex(ValueError, "after an allocation"):
            trace(fn, (x,), warm_up=False)
        self.assertIsNone(current_trace())
        self.assertFalse(torch.cuda.is_current_stream_capturing())
        self.assertEqual(len(trace(torch.empty_like, (x,)).allocs), 1)

    @unittest.skipIf(not has_triton(), "requires triton")
    def test_a_launch_on_views_records_root_and_offset(self):
        x = torch.randn(64, 16, device="cuda")

        def fn(t):
            y = torch.empty_like(t)
            _copy[(1,)](t[4:], y[:, 2:], 32, B=32)
            return y

        tape = trace(fn, (x,))
        ((_, launch),) = tape.launches
        self.assertEqual([r.name for r in launch.roots], ["p0", "a0"])
        rec, alloc = tape.inputs[0], tape.allocs[0]
        x_ptr = rec.root.sym + 4 * (rec.offset + 4 * rec.strides[0])
        self.assertEqual(launch.slots[0].node.expr, x_ptr.node.expr)
        y_ptr = alloc.root.sym + 4 * 2 * alloc.strides[1]
        self.assertEqual(launch.slots[1].node.expr, y_ptr.node.expr)

    def test_a_copy_on_write_input_stays_lazy(self):
        y = torch._lazy_clone(torch.randn(4, device="cuda"))
        trace(torch.empty_like, (y,), warm_up=False)
        self.assertTrue(torch._C._is_cow_tensor(y))

    def test_is_cow_tensor_of_a_traced_tensor(self):
        # what torch._native's conditions ask: an argument answers its input's
        # state at the trace, a tensor the trace allocates False
        answers = []

        def fn(t):
            answers.append((torch._C._is_cow_tensor(t), torch._C._is_cow_tensor(torch.empty_like(t))))
            return t

        x = torch.randn(4)
        _run(fn, x)
        _run(fn, torch._lazy_clone(x))
        self.assertEqual(answers, [(False, False), (True, False)])


def setUpModule():
    import torch.cuda._host_trace as host_trace

    host_trace.raise_unexpected = True


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not hasattr(torch._C, "_cuda_hostTraceMul"), "needs traced hosts")
class TestOpGuards(TestCase):
    def test_a_guard_of_one_op_is_its_own(self):
        x = torch.randn(64, 600, device="cuda")
        tape = trace(lambda x: torch.softmax(x, -1), (x,))
        (op,) = tape.ops
        self.assertEqual(op.kind, "traced")
        self.assertEqual(len(op.launches), len(tape.launches))
        self.assertTrue(op.guards)
        self.assertEqual(op.guards, tuple(i for i, o in enumerate(tape.owners) if o == 0))

    def test_a_guard_two_ops_read_is_the_graphs(self):
        x = torch.randn(64, 600, device="cuda")
        tape = trace(lambda x: (torch.softmax(x, -1), torch.softmax(x, -1)), (x,))
        self.assertEqual([op.guards for op in tape.ops], [(), ()])
        self.assertEqual(set(tape.owners), {None})

    def test_an_eager_call_is_one_op(self):
        x = torch.randn(64, 600, device="cuda")
        tape = trace(lambda x: torch.rsqrt(x.sum(-1, dtype=torch.float32)), (x,))
        self.assertEqual([op.kind for op in tape.ops], ["eager", "traced"])
        self.assertIsInstance(tape.launches[tape.ops[0].launches.start][1], EagerCall)


if __name__ == "__main__":
    run_tests()
