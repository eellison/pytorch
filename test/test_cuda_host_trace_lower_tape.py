# Owner(s): ["module: cuda graphs"]

import unittest

import torch
from torch.cuda._host_trace import Declined
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_lower_tape import (
    lower_tape,
    LoweredView,
    PointerSlot,
    ScalarSlot,
)
from torch.cuda._host_trace_tape import (
    _output_records,
    _OutputRec,
    _symbolic_run,
    _Trace,
    current_trace,
    Tape,
    trace,
)
from torch.cuda._host_trace_triton import TritonABI, TritonArg
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

    @triton.jit
    def _add(x_ptr, y_ptr, n, s, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) + s, mask=m)


# x, y, n, m, then the launcher's two scratch pointers
_ABI = TritonABI(
    (
        TritonArg("x", "*fp32", 0),
        TritonArg("y", "*fp32", 1),
        TritonArg("n", "i32", 2),
        TritonArg("m", "i32", 3),
    ),
    6,
    4,
    0,
)
_LAYOUT = ((0, 8), (8, 8), (16, 4), (20, 4), (24, 8), (32, 8))


def _launch(x, y, n, m, grid):
    slots = (x.data_ptr(), y.data_ptr(), n, m, 0, 0)
    roots = tuple({id(t._root): t._root for t in (x, y)}.values())
    launch = KernelLaunch("k", None, _ABI, _LAYOUT, grid, (128, 1, 1), 0, slots, roots)
    current_trace().record_launch(launch)


def _tape(fn, *args):
    # a tape of the symbolic run on the CPU, without trace()'s capture
    tr = _Trace(torch.device("cpu"))
    positions = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
    ints = [i for i, a in enumerate(args) if type(a) is int]
    out, traced = _symbolic_run(tr, fn, args, positions, ints)
    kind, outputs = _output_records(out, traced, positions)
    return Tape(tr, args, outputs, kind, None)


def _scale(a, n):
    # an n-row allocation and, past n > 4, a (recorded, never run) launch over it
    out, numel = torch.empty(n, a.shape[1]), n * a.shape[1]
    if n > 4:
        _launch(a, out, numel, a.shape[1], ((numel + 127) // 128, 1, 1))
    return out, a


def _far(a):
    # a's metadata at address 2**63; never dereferenced
    storage = torch._C._construct_storage_from_data_pointer(
        -(2**63), a.device, a.nbytes
    )
    return torch.empty(0, dtype=a.dtype).set_(storage, 0, a.shape, a.stride())


class TestLowerTape(TestCase):
    def _slots(self, lowered, values, launch=0):
        out = []
        for slot in lowered.launches[launch].slots:
            if isinstance(slot, ScalarSlot):
                out.append(values[slot.row])
            else:
                address = None if slot.address is None else values[slot.address]
                out.append((slot.root, values[slot.displacement], address))
        return out

    def test_rows_at_the_trace_and_at_another_call(self):
        base = torch.randn(40, 12)
        a = base[3:, ::2]  # storage offset 36 floats, strides (12, 2)
        lowered = lower_tape(_tape(_scale, a, 8))
        self.assertEqual(lowered.outputs, (("allocation", 0), ("argument", 0)))
        for args in ((a, 8), (base[1:30, :6], 20), (torch.randn(6, 5), 6)):
            x, n = args
            values = lowered.evaluate(args)
            self.assertIsNotNone(values)
            want = [
                (("argument", 0), x.storage_offset() * 4, x.data_ptr()),
                (("allocation", 0), 0, None),
                n * x.shape[1],
                x.shape[1],
                0,
                0,
            ]
            self.assertEqual(self._slots(lowered, values), want)
            grid = [values[r] for r in lowered.launches[0].grid]
            self.assertEqual(grid, [(n * x.shape[1] + 127) // 128, 1, 1])
            (alloc,) = lowered.allocations
            self.assertEqual([values[r] for r in alloc.sizes], [n, x.shape[1]])
            self.assertEqual([values[r] for r in alloc.strides], [x.shape[1], 1])
            self.assertEqual(values[alloc.nbytes], n * x.shape[1] * 4)

    def test_misses(self):
        a = torch.randn(10, 6)
        lowered = lower_tape(_tape(_scale, a, 8))
        cases = {
            "the host branched on n > 4": (a, 3),
            "a size of 0 against a positive size symbol": (a[:, :0], 8),
            "a tensor where an int was": (a, a),
            "an int where a tensor was": (3, 8),
            "a pointer at or above 2**63": (_far(a), 8),
            "the wrong arity": (a,),
        }
        for why, args in cases.items():
            self.assertIsNone(lowered.evaluate(args), msg=why)

    def test_allocation_requirements(self):
        def fn(a, n, stride):
            return torch.empty_strided((n,), (stride,)), a

        a = torch.randn(4)
        # the trace guards the size's emptiness, so each tape serves one side
        empty = lower_tape(_tape(fn, a, 0, 1))
        lowered = lower_tape(_tape(fn, a, 5, 1))
        cases = {(3, 2): (lowered, 20), (0, -1): (empty, 0), (0, 7): (empty, 0)}
        for (n, stride), (lo, nbytes) in cases.items():
            values = lo.evaluate((a, n, stride))
            self.assertEqual(values[lo.allocations[0].nbytes], nbytes)
        # empty_strided refuses a negative stride of a nonempty allocation
        self.assertIsNone(lowered.evaluate((a, 3, -1)))
        # the storage bytes overflow int64
        self.assertIsNone(lowered.evaluate((a, 2**40, 2**40)))

    def test_declines(self):
        def guard_on_allocation(a):
            out = torch.empty_like(a)
            if out.data_ptr() % 512 == 0:
                return out
            return a

        def two_roots(a):
            out = torch.empty_like(a)
            _launch(a, out, 1, 1, (1, 1, 1))
            tr = current_trace()
            seq, launch = tr.launches[-1]
            slots = (a.data_ptr() + out.data_ptr(), *launch.slots[1:])
            launch = KernelLaunch(
                "k", None, _ABI, _LAYOUT, (1, 1, 1), (128, 1, 1), 0, slots, launch.roots
            )
            tr.launches[-1] = (seq, launch)
            return out

        a = torch.randn(4)
        with self.assertRaisesRegex(Declined, "not read from the call's inputs"):
            lower_tape(_tape(guard_on_allocation, a))
        with self.assertRaisesRegex(Declined, "not one root plus an offset"):
            lower_tape(_tape(two_roots, a))
        tape = _tape(lambda t: (t + 1)[1:], a)
        with self.assertRaisesRegex(
            Declined,
            "no traced launches: every operation runs eagerly \\(aten.add.Tensor\\)",
        ):
            lower_tape(tape)
        tape = _tape(lambda t: torch.empty_like(t), a)
        rec = tape.outputs[0]
        tape.outputs = [_OutputRec("out0", rec.root, [1], [1], 0, torch.int32)]
        (out,) = lower_tape(tape).outputs
        self.assertEqual((out.base, out.dtype), (("allocation", 0), torch.int32))

    def test_output_views(self):
        def fn(a):
            out = torch.empty(a.shape[0], 4)
            return out[1:], a[:, 1:], out.view(-1), out

        base = torch.randn(10, 5)
        lowered = lower_tape(_tape(fn, base[2:]))
        v0, v1, v2, whole = lowered.outputs
        self.assertEqual(whole, ("allocation", 0))
        bases = ("allocation", 0), ("argument", 0), ("allocation", 0)
        self.assertEqual((v0.base, v1.base, v2.base), bases)
        for a in (base[2:], torch.randn(10, 5)[1:], torch.randn(3, 7)):
            values = lowered.evaluate((a,))
            self.assertIsNotNone(values)
            n, want = a.shape[0], fn(a)
            for view, t in zip((v0, v1, v2), want):
                self.assertIsInstance(view, LoweredView)
                self.assertEqual([values[r] for r in view.sizes], list(t.shape))
                self.assertEqual([values[r] for r in view.strides], list(t.stride()))
                self.assertEqual(values[view.offset], t.storage_offset())
            self.assertEqual(values[v0.offset], 4)
            self.assertEqual(values[v2.sizes[0]], 4 * n)


@unittest.skipIf(not TEST_CUDA or not has_triton(), "requires CUDA and Triton")
@requires_cuda_python_bindings
class TestLowerTritonTape(TestCase):
    def test_a_traced_launch(self):
        def fn(x, n):
            y = torch.empty_like(x)
            _add[(triton.cdiv(n, 128),)](x, y, n, 3, B=128)
            return y

        x = torch.randn(1000, device="cuda")
        lowered = lower_tape(trace(fn, (x, 1000)))
        (launch,) = lowered.launches
        x2 = torch.randn(3000, device="cuda")
        values = lowered.evaluate((x2, 2500))
        self.assertIsNotNone(values)
        x_slot, y_slot = launch.slots[:2]
        self.assertIsInstance(x_slot, PointerSlot)
        self.assertEqual(values[x_slot.address], x2.data_ptr())
        self.assertEqual(y_slot.root, ("allocation", 0))
        self.assertEqual(values[y_slot.displacement], 0)
        self.assertEqual(values[launch.slots[2].row], 2500)
        self.assertEqual([values[r] for r in launch.grid], [20, 1, 1])
        self.assertEqual(values[lowered.allocations[0].nbytes], 3000 * 4)
        # Triton specialized x's address on 16-byte alignment
        self.assertIsNone(lowered.evaluate((x2[1:], 2500)))


def setUpModule():
    from torch.cuda import _host_trace_hint_audit
    import torch.cuda._host_trace as host_trace

    _host_trace_hint_audit.enable_for_tests()
    host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
