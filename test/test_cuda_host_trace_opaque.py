# Owner(s): ["module: cuda graphs"]

import dataclasses
import struct
import unittest
from unittest import mock

import torch
from torch.cuda import _host_trace_replay
from torch.cuda._host_trace_harvest import _launch, HarvestProvider
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_lower_tape import (
    lower_tape,
    LoweredMemset,
    LoweredView,
    PointerSlot,
    PredictedOutput,
)
from torch.cuda._host_trace_opaque import (
    OpaqueBinding,
    OpaqueKernel,
    OpaqueMemset,
    Slot,
)
from torch.cuda._host_trace_memory import plan_memory, split_runs
from torch.cuda._host_trace_tape import bind_opaque, EagerCall, Memset, OpaqueCall, trace
from torch.cuda._host_trace_triton import param_layout, triton_abi
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    TEST_CUDA,
    TestCase,
)
from torch.utils._triton import has_triton


class HostTraceReplay(_host_trace_replay.HostTraceReplay):
    # traces at its first call: these are tests of the trace; an entry's first
    # call runs eagerly (test_the_first_call_runs_eagerly in
    # test_cuda_host_trace_replay)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._called = True


aten = torch.ops.aten

if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _add(x_ptr, y_ptr, n, s, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) + s, mask=m)

    @triton.jit
    def _mul(x_ptr, y_ptr, out_ptr, n, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        x = tl.load(x_ptr + i, mask=m)
        tl.store(out_ptr + i, x * tl.load(y_ptr + i, mask=m), mask=m)


def add(x, s=3):
    y = torch.empty_like(x)
    n = x.numel()
    _add[(triton.cdiv(n, 128),)](x, y, n, s, B=128)
    return y


def mul_chain(x, y):
    return add(torch.mul(add(x), y), 1)


def mm_chain(x, w):
    return add(torch.mm(add(x), w))


class TableProvider:
    """Stands in for a library: `ops` of float32 tensors; aten.mul of
    contiguous operands binds to the Triton kernel _mul, anything else is
    refused. With style "scratch" (or "mixed" and n % 256 == 0) the binding
    is a memset of 1.0 into scratch s0, s1 = s0 * x, out = s1 * y. Each
    kernel carries `attributes`."""

    def __init__(self, ops=(aten.mul.Tensor,), style="plain", attributes=(), rng_sizes=()):
        self.ops = ops
        self.rng_sizes = rng_sizes  # a binding at these sizes claims 4 philox offsets
        self.style = style
        self.attributes = attributes
        self.table = {}  # key -> binding, or None when refused
        self.binds = []
        self.learned = []
        self.kernels = []  # keeps each bound function loaded

    def accepts(self, op, args, kwargs):
        if op not in self.ops:
            return f"TableProvider: {op} is not in its table"
        if any(isinstance(a, torch.Tensor) and a.dtype != torch.float32 for a in args):
            return "TableProvider: float32 only"
        return None

    def bind(self, key):
        self.binds.append(key)
        return self.table.get(key)

    def refusal(self, key):
        return "TableProvider: refused" if key in self.table and self.table[key] is None else None

    def learn(self, key, args, kwargs, operands):
        if key in self.table:
            return None
        self.learned.append((key, operands))
        ok = key.op in (aten.mul.Tensor, aten.mul.out) and len(set(key.sizes)) == 1
        ok = ok and all(t.is_contiguous() for t in operands)
        self.table[key] = self._binding(*operands) if ok else None
        return self.table[key]

    def _binding(self, x, y, out):
        n = out.numel()
        compiled = _mul.warmup(x, y, out, n, B=128, grid=(1,))
        compiled._init_handles()
        self.kernels.append(compiled)
        abi = triton_abi(compiled.src, compiled.metadata)
        params = param_layout(abi, int(compiled.function))
        images = [bytes(size) for _, size in params]
        images[3] = struct.pack("<i", n)

        def kernel(*buffers):
            slots = tuple(Slot(i, 0, *b) for i, b in enumerate(buffers))
            grid, block = (triton.cdiv(n, 128), 1, 1), (32 * abi.num_warps, 1, 1)
            call = (int(compiled.function), grid, block, abi.shared, self.attributes)
            return OpaqueKernel(*call, params, tuple(images), slots)

        x, y, out = (("operand", i, 0) for i in range(3))
        if self.style == "plain" or (self.style == "mixed" and n % 256):
            return OpaqueBinding((kernel(x, y, out),), (), 4 if n in self.rng_sizes else 0)
        s0, s1 = ("scratch", 0, 0), ("scratch", 0, 4 * n)
        ones = OpaqueMemset(Slot(0, 0, *s0), 0x3F800000, 4, n, 1, 4 * n)
        return OpaqueBinding((ones, kernel(s0, x, s1), kernel(s1, y, out)), (8 * n,))


def plan_errors(v):
    """The variant's memory plan against an independent lifetime simulation."""
    lowered, plan = v.captured.lowered, v.memory
    n_alloc = len(lowered.allocations)

    def uses(step):
        if isinstance(step, range):
            slots = [s for lo in lowered.launches[step.start : step.stop] for s in lo.slots]
            return {s.base for s in slots if isinstance(s, PointerSlot) and s.base is not None}
        out = {n_alloc + p.root for p in step.outputs if isinstance(p, PredictedOutput)}
        for leaf in step.leaves:
            if isinstance(leaf, LoweredView) and leaf.base[0] != "argument":
                kind, j = leaf.base
                out.add(j if kind == "allocation" else n_alloc + j)
        return out

    escapes = set()
    for o in lowered.outputs:
        kind, k = o.base if isinstance(o, LoweredView) else o
        if kind != "argument":
            escapes.add(k if kind == "allocation" else n_alloc + k)
    held, dropped, made, errors = set(), set(), set(), []
    for i, (step, mem) in enumerate(zip((*lowered.steps, None), plan.steps)):
        held |= set(mem.tensors)
        temps = {k for k, _, _ in mem.temporaries}
        made |= set(mem.tensors) | temps
        produced = set()
        if step is not None and not isinstance(step, range):
            produced = {n_alloc + p.root for p in step.outputs if isinstance(p, PredictedOutput)}
            held |= produced
        for b in uses(step) if step is not None else ():
            if b in dropped:
                errors.append(f"step {i} uses base {b} after its drop")
            elif b not in held | temps | produced:
                errors.append(f"step {i} uses base {b}, which is not held")
            if b in temps and not isinstance(step, range):
                errors.append(f"eager step {i} uses scratch temporary {b}")
        for b in mem.drops:
            if b in escapes:
                errors.append(f"step {i} drops escaping base {b}")
            held.discard(b)
            dropped.add(b)
        dropped |= temps
    errors += [f"allocation {k} is never made" for k in range(n_alloc) if k not in made]
    if held - escapes:
        errors.append(f"bases held at the end but not escaping: {sorted(held - escapes)}")
    return errors


def lowered_rows(lowered):
    """A lowered tape's rows and records, without its symbols' hints (an
    allocation's or eager output's placeholder address)."""

    def launch(lo):
        r = lo.launch
        if isinstance(lo, LoweredMemset):
            return (lo.seq, r.name, lo.slots, lo.width, lo.height, lo.pitch, r.value, r.element_size)
        fixed = (r.function, r.layout, r.fields, r.pointers, r.images, r.attributes, r.programmatic, r.rng, r.rng_increment)
        return (lo.seq, r.name, lo.slots, lo.grid, lo.block, lo.smem, *fixed)

    def step(s):
        if isinstance(s, range):
            return s
        return (s.seq, s.call.name, s.spec, s.flat, s.leaves, s.outputs, s.grid)

    def site(s):
        scratch = {j: n for j, (_, n) in s.site.scratch.items()}
        return (s.site.op, s.dtypes, s.ranks, s.rows, s.operands, s.scratch, s.nodes, s.site.topology, s.site.rng, scratch)

    opaque = {i: (o.op, o.dtypes, o.ranks, o.rows, o.scalars) for i, o in lowered.opaque.items()}
    return (
        lowered.program.instructions,
        lowered.valid,
        lowered.allocations,
        [launch(lo) for lo in lowered.launches],
        [step(s) for s in lowered.steps],
        len(lowered.eager_roots),
        lowered.outputs,
        opaque,
        [site(s) for s in lowered.sites],
    )


def bound(f):
    """The variants that launch an opaque call's binding."""
    return [v for v in f.variants if not v.captured.lowered.opaque]


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not has_triton(), "requires triton")
class TestOpaqueCalls(TestCase):
    def test_keys(self):
        p = TableProvider()
        f = HostTraceReplay(mul_chain, opaque=(p,))
        base = torch.randn(1001, device="cuda")
        for n, offset in [(512, 0), (512, 0), (512, 1), (700, 3)]:
            x = torch.randn(n, device="cuda")
            y = base[offset : offset + n]
            self.assertEqual(f(x, y), mul_chain(x, y))
        keys = list(dict.fromkeys(p.binds))
        self.assertEqual(len(keys), 3)
        for key, (n, offset) in zip(keys, [(512, 0), (512, 1), (700, 3)]):
            self.assertIs(key.op, aten.mul.Tensor)
            self.assertEqual(key.dtypes, (torch.float32,) * 3)
            self.assertEqual(key.sizes, ((n,),) * 3)
            self.assertEqual(key.strides, ((1,),) * 3)
            align = (base.data_ptr() + 4 * offset) % 256
            self.assertEqual(key.align, (0, align, 0))
            self.assertEqual(key.scalars, ())

    def test_scalar_and_symint_leaves(self):
        p = TableProvider(ops=(aten.add.Tensor, aten.full.default))

        def fn(x):
            y = add(x)
            twos = torch.full((x.numel(),), 2.0, device=x.device)
            return add(torch.add(y, twos, alpha=3))

        f = HostTraceReplay(fn, opaque=(p,))
        for n in (256, 256, 300):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), fn(x))
        binds = list(dict.fromkeys(p.binds))
        full, add_ = binds[:2]
        self.assertIs(full.op, aten.full.default)
        self.assertEqual(full.scalars[:2], (256, 2.0))
        self.assertEqual(full.sizes, ((256,),))
        self.assertIs(add_.op, aten.add.Tensor)
        self.assertEqual(add_.scalars, (3,))
        self.assertEqual(binds[2].scalars[:2], (300, 2.0))

    def test_miss_learn_hit(self):
        p = TableProvider()
        f = HostTraceReplay(mul_chain, opaque=(p,))
        x, y = torch.randn(512, device="cuda"), torch.randn(512, device="cuda")
        self.assertEqual(f(x, y), mul_chain(x, y))  # the trace's warm-up
        # the trace's bind, and the variant's check of the keys it left eager
        self.assertEqual((len(p.binds), len(p.learned)), (2, 0))
        for want_learned in (1, 1, 1):
            self.assertEqual(f(x, y), mul_chain(x, y))
            self.assertEqual(len(p.learned), want_learned)
        key, operands = p.learned[0]
        self.assertIsNotNone(p.table[key])
        # the fresh output holds the eager result the provider learned from
        self.assertEqual(operands[2], add(x) * y)
        # the key bound at the third call, which relowered the tape with it as launches
        self.assertEqual((f.traces, f.relowers, len(bound(f))), (1, 1, 1))
        records = [r for _, r in bound(f)[0].tape.launches]
        self.assertFalse(any(isinstance(r, EagerCall) for r in records))
        self.assertIn("aten.mul.Tensor node 0", [r.name for r in records])
        x2, y2 = torch.randn(640, device="cuda"), torch.randn(640, device="cuda")
        self.assertEqual(f(x2, y2), mul_chain(x2, y2))
        self.assertEqual(len(p.learned), 2)
        self.assertEqual((f.traces, f.relowers), (1, 1))

    def test_a_refused_key_stays_eager(self):
        p = TableProvider()
        f = HostTraceReplay(mul_chain, opaque=(p,))
        x = torch.randn(512, device="cuda")
        y = torch.randn(1024, device="cuda")[::2]
        for _ in range(4):
            self.assertEqual(f(x, y), mul_chain(x, y))
        self.assertEqual(len(p.learned), 1)
        self.assertIsNone(p.table[p.learned[0][0]])
        # the refusal traced it again as a plain eager step: that variant
        # does not learn, so its calls run natively
        self.assertEqual(f.traces, 2)
        refused = [v for v in f.variants if not v.learns]
        self.assertEqual(len(refused), 1)
        records = [r for _, r in refused[0].tape.launches]
        eager = [r for r in records if type(r) is EagerCall]
        self.assertEqual(len(eager), 1)
        self.assertTrue(eager[0].reason.endswith(": TableProvider: refused"))

    def test_programmatic_binding(self):
        # a kernel launched with PSS: its node's edge in is programmatic
        from cuda.bindings import driver

        from torch.cuda._host_trace_capture import graph_nodes

        pss = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        p = TableProvider(attributes=((pss, 1),))
        f = HostTraceReplay(mul_chain, opaque=(p,))
        x, y = torch.randn(512, device="cuda"), torch.randn(512, device="cuda")
        for _ in range(6):
            self.assertEqual(f(x, y), mul_chain(x, y), atol=0, rtol=0)
        self.assertEqual((f.traces, f.relowers, len(bound(f))), (1, 1, 1))
        (variant,) = bound(f)
        (launch,) = [r for _, r in variant.tape.launches if r.owner is p]
        self.assertTrue(launch.programmatic)
        (segment,) = variant.captured.segments
        nodes = graph_nodes(segment.graph.raw_cuda_graph())
        self.assertEqual([n.attribute("PROGRAMMATIC_STREAM_SERIALIZATION") for n in nodes], [0, 1, 0])

    def test_a_cooperative_binding_launches_cooperative(self):
        from cuda.bindings import driver

        from torch.cuda._host_trace_capture import graph_nodes

        coop = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
        p = TableProvider(attributes=((coop, 1),))
        f = HostTraceReplay(mul_chain, opaque=(p,))
        x, y = torch.randn(512, device="cuda"), torch.randn(512, device="cuda")
        for _ in range(6):
            self.assertEqual(f(x, y), mul_chain(x, y), atol=0, rtol=0)
        self.assertEqual((f.traces, f.relowers, len(bound(f))), (1, 1, 1))
        (segment,) = bound(f)[0].captured.segments
        nodes = graph_nodes(segment.graph.raw_cuda_graph())
        self.assertEqual([n.attribute("COOPERATIVE") for n in nodes], [0, 1, 0])

    def test_a_constant_raw_attribute_is_launched(self):
        # an attribute _ATTRS does not decode (an access policy window over a
        # buffer the call does not move) is launched as its raw bytes
        from cuda.bindings import driver

        from torch.cuda._host_trace_capture import graph_nodes

        window = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW
        persistent = torch.empty(4096, dtype=torch.uint8, device="cuda")
        v = driver.CUlaunchAttributeValue()
        w = v.accessPolicyWindow
        w.base_ptr, w.num_bytes, w.hitRatio = persistent.data_ptr(), persistent.numel(), 1.0
        w.hitProp = driver.CUaccessProperty.CU_ACCESS_PROPERTY_PERSISTING
        w.missProp = driver.CUaccessProperty.CU_ACCESS_PROPERTY_STREAMING
        raw = bytes(v.pad)
        p = TableProvider(attributes=((window, raw),))
        f = HostTraceReplay(mul_chain, opaque=(p,))
        x, y = torch.randn(512, device="cuda"), torch.randn(512, device="cuda")
        for _ in range(6):
            self.assertEqual(f(x, y), mul_chain(x, y), atol=0, rtol=0)
        self.assertEqual((f.traces, f.relowers, len(bound(f))), (1, 1, 1))
        (segment,) = bound(f)[0].captured.segments
        nodes = graph_nodes(segment.graph.raw_cuda_graph())
        self.assertEqual([n.attribute("ACCESS_POLICY_WINDOW") == raw for n in nodes], [False, True, False])

    def test_an_unrecordable_binding_is_an_eager_step(self):
        # the key binds, but the trace cannot record the binding (a
        # device-updatable node): traced once more, as a plain eager step
        from cuda.bindings import driver

        updatable = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE
        v = driver.CUlaunchAttributeValue()
        v.deviceUpdatableKernelNode.deviceUpdatable = 1
        p = TableProvider(attributes=((updatable, bytes(v.pad)),))
        f = HostTraceReplay(mul_chain, opaque=(p,))
        x, y = torch.randn(512, device="cuda"), torch.randn(512, device="cuda")
        for _ in range(8):
            self.assertEqual(f(x, y), mul_chain(x, y))
        self.assertEqual((f.traces, len(f.variants), len(bound(f))), (2, 2, 1))
        eager = [r for _, r in bound(f)[0].tape.launches if type(r) is EagerCall]
        self.assertEqual(len(eager), 1)
        self.assertTrue(eager[0].reason.endswith("device-updatable"))

    def test_a_binding_launches_like_eager(self):
        p = TableProvider()
        f = HostTraceReplay(mul_chain, opaque=(p,))
        x, y = torch.randn(700, device="cuda"), torch.randn(700, device="cuda")
        f(x, y)
        f(x, y)
        (binding,) = [b for b in p.table.values() if b is not None]
        a, b = torch.randn(700, device="cuda"), torch.randn(700, device="cuda")
        out = torch.empty_like(a)
        stream = torch.cuda.current_stream().cuda_stream
        _launch(binding, [t.data_ptr() for t in (a, b, out)], [], stream)
        self.assertEqual(out, a * b, atol=0, rtol=0)
        self.assertEqual(binding.topology, (("kernel", ()),))
        self.assertFalse(binding.rng)

    def test_an_rng_binding_fits_rng_sites(self):
        # a binding with philox slots or an increment is a row of RNG sites
        # only, whose replays pack each row's philox offsets
        p = TableProvider()
        f = HostTraceReplay(mul_chain, opaque=(p,))
        x, y = torch.randn(700, device="cuda"), torch.randn(700, device="cuda")
        f(x, y)
        f(x, y)
        (binding,) = [b for b in p.table.values() if b is not None]
        (kernel,) = binding.nodes
        philox = kernel.slots[:2] + (Slot(0, 8, "philox_seed", 0, 0),)
        seeded = dataclasses.replace(binding, nodes=(dataclasses.replace(kernel, slots=philox),))
        incremented = dataclasses.replace(binding, rng_increment=4)
        self.assertTrue(seeded.rng)
        self.assertTrue(incremented.rng)
        f(x, y)
        (v,) = bound(f)
        (site,) = v.captured.lowered.sites
        self.assertTrue(site.site.fits(binding))
        self.assertFalse(site.site.fits(seeded))
        self.assertFalse(site.site.fits(incremented))

    def test_an_rng_key_is_a_row(self):
        # a new key of an RNG site is a row of the variant, and each replay
        # advances the generator by what its rows take (the table's mul, run
        # eagerly while its key is learned, takes none); a non-RNG key does
        # not fit the site and traces again
        p = TableProvider(rng_sizes=(700, 900, 1100))
        f = HostTraceReplay(mul_chain, opaque=(p,))
        gen = torch.cuda.default_generators[0]
        for n, traces in ((700, 1), (900, 1), (1100, 1), (1300, 2)):
            x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            for _ in range(3):
                offset = gen.get_offset()
                self.assertEqual(f(x, y), mul_chain(x, y), atol=0, rtol=0)
            self.assertEqual(gen.get_offset(), offset + 4 * (n in p.rng_sizes))
            self.assertEqual(f.traces, traces)
        self.assertEqual(len(bound(f)), 2)

    def test_decline_reasons(self):
        args = (torch.randn(32, 16, device="cuda"), torch.randn(16, 8, device="cuda"), torch.randn(8, device="cuda"))

        def fn(x, w, v):
            h = add(x)
            return add(torch.mv(torch.mm(h.half(), w.half()).float(), v))

        tape = trace(fn, args, opaque=(TableProvider(ops=(aten.mm.default,)),))
        eager = [rec for _, rec in tape.launches if isinstance(rec, EagerCall)]
        calls = {str(rec.target): rec for rec in eager}
        self.assertFalse(any(isinstance(c, OpaqueCall) for c in calls.values()))
        self.assertEqual(calls["aten.mm.default"].reason, "TableProvider: float32 only")
        mv = "TableProvider: aten.mv.default is not in its table; aten.mv.default's body declines: host_trace: its part aten.addmv_.default runs eagerly (TableProvider: aten.addmv_.default is not in its table) (declined)"  # noqa: B950
        self.assertEqual(calls["aten.mv.default"].reason, mv)
        self.assertEqual(lower_tape(tape).opaque, {})
        plain = trace(fn, args)
        calls = [rec for _, rec in plain.launches if isinstance(rec, EagerCall)]
        self.assertTrue(all(type(c) is EagerCall and c.reason is None for c in calls))

    def test_an_out_overlapping_an_operand_is_not_bound(self):
        x = torch.randn(512, device="cuda")

        def fn(x):
            h = add(x)
            torch.mul(h, h, out=h)
            y = add(h)
            return torch.mul(y, h, out=torch.empty_like(y))

        tape = trace(fn, (x,), opaque=(TableProvider(ops=(aten.mul.out,)),))
        # the pointwise host traces the overlapping mul; only the other binds
        self.assertFalse(any(type(rec) is EagerCall for _, rec in tape.launches))
        self.assertEqual(len(lower_tape(tape).opaque), 1)

    def test_an_out_argument_overlapping_at_replay_runs_eagerly(self):
        # the binding writes the argument out: a call where it overlaps x runs
        # eagerly
        def fn(x, y, out):
            torch.mul(x, y, out=out)
            return add(out)

        def args(shared):
            buf = torch.randn(576, device="cuda")
            out = buf[64:] if shared else torch.empty(512, device="cuda")
            return buf[:512], torch.randn(512, device="cuda"), out

        f = HostTraceReplay(fn, opaque=(TableProvider(ops=(aten.mul.out,)),))
        for _ in range(3):
            a = args(False)
            self.assertEqual(f(*a), fn(*a), atol=0, rtol=0)
        self.assertEqual(len(bound(f)), 1)
        traces, eager = f.traces, f.eager
        overlap = "refer to a single memory location"
        with self.assertRaisesRegex(RuntimeError, overlap):
            fn(*args(True))
        with self.assertRaisesRegex(RuntimeError, overlap):
            f(*args(True))
        self.assertEqual((f.traces, f.eager), (traces, eager + 1))

    def test_an_out_into_an_eager_output_is_its_argument(self):
        # Inductor reuses a freed eager call's output as an extern mm's out=:
        # the out= return is the argument, not a fourth operand, so the key
        # harvests and binds
        def fn(a, b):
            out = torch.cumsum(add(a), 1)
            torch.mm(a, b, out=out)
            return add(out)

        p = HarvestProvider()
        f = HostTraceReplay(fn, opaque=(p,))
        a, b = (torch.randn(64, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        for _ in range(4):
            self.assertEqual(f(a, b), fn(a, b), atol=0, rtol=0)
        self.assertEqual(p.refused, {})
        (v,) = bound(f)
        (site,) = v.captured.lowered.sites
        self.assertEqual(len(site.site.operands), 3)

    def mm_rows(self, ms):
        # the keyed site of an extern mm over M, each M a key: the learner's
        # retrace, then every new key a row of the bound variant's table
        f = HostTraceReplay(mm_chain, opaque=(HarvestProvider(),))
        w = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        for m in ms:
            for _ in range(2):
                x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
                self.assertEqual(f(x, w), mm_chain(x, w), atol=0, rtol=0)
        return f, w

    def assertNativeRow(self, f, w, m):
        # a new key: the first call's fill harvests it and adds its row,
        # then the native entry serves it with no retrace
        traces, harvests = f.traces, f.opaque[0].harvests
        slow_path = HostTraceReplay._call_slow
        with mock.patch.object(HostTraceReplay, "_call_slow", autospec=True, side_effect=slow_path) as slow:
            for _ in range(5):
                x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
                self.assertEqual(f(x, w), mm_chain(x, w), atol=0, rtol=0)
        self.assertEqual((slow.call_count, f.traces, f.opaque[0].harvests), (1, traces, harvests + 1))

    def test_a_new_key_at_a_keyed_site_is_a_native_row(self):
        f, w = self.mm_rows((64, 96))
        (v,) = bound(f)
        (site,) = v.captured.lowered.sites
        self.assertEqual(site.device, w.device.index)
        self.assertNativeRow(f, w, 128)

    def test_keys_past_64_harvest(self):
        # the default budget holds more keys than a dynamic M sweep reaches
        f, w = self.mm_rows(range(8, 8 * 66, 8))
        self.assertEqual(f.opaque[0].harvests, 65)
        self.assertNativeRow(f, w, 8 * 66)

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires two GPUs")
    def test_a_binding_on_another_device(self):
        # the mm of cuda:1 operands under cuda:0 harvests and launches on
        # cuda:1; cuda:0's key of the same layouts is another key
        def fn(x, w):
            with torch.cuda.device(x.device):
                h = add(x)
            y = torch.mm(h, w)
            with torch.cuda.device(x.device):
                return add(y)

        p = HarvestProvider()
        for device in ("cuda:1", "cuda:0"):
            f = HostTraceReplay(fn, opaque=(p,))
            w = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
            for _ in range(4):
                x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
                self.assertEqual(f(x, w), fn(x, w), atol=0, rtol=0)
            (v,) = bound(f)
            (site,) = v.captured.lowered.sites
            self.assertEqual(site.device, w.device.index)
        self.assertEqual((p.harvests, p.refused), (2, {}))

    def test_a_2d_memset_topology_holds_its_width(self):
        # a graph exec fixes a 2-D memset's width, not a 1-D one's
        def binding(width, height):
            memset = OpaqueMemset(Slot(0, 0, "scratch", 0, 0), 0, 4, width, height, 256)
            return OpaqueBinding((memset,), (512,))

        self.assertEqual(binding(16, 1).topology, binding(32, 1).topology)
        self.assertNotEqual(binding(16, 2).topology, binding(32, 2).topology)

    def test_steps(self):
        x, y = torch.randn(512, device="cuda"), torch.randn(512, device="cuda")

        def fn(x, y):
            return add(torch.cumsum(add(torch.mul(add(x), y)), 0))

        lowered = lower_tape(trace(fn, (x, y), opaque=(TableProvider(),)))
        kinds = [type(s).__name__ for s in lowered.steps]
        self.assertEqual(
            kinds, ["range", "LoweredEagerCall", "range", "LoweredEagerCall", "range"]
        )
        self.assertEqual(list(lowered.opaque), [1])

    @parametrize("style", ["plain", "scratch"])
    def test_bitwise_over_sizes(self, style):
        p = TableProvider(style=style)
        f = HostTraceReplay(mul_chain, opaque=(p,))
        sizes = (512, 640, 1024, 4096)
        for n in sizes:
            for _ in range(4):
                x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
                self.assertEqual(f(x, y), mul_chain(x, y), atol=0, rtol=0)
        # a binding of every size is a row of the bound variant's table, its
        # scratch buffer allocated at the key's bytes
        self.assertEqual(len(bound(f)), 1)
        self.assertEqual((f.traces, f.relowers), (1, 1))
        for v in bound(f):
            kinds = {type(lo).__name__ for lo in v.captured.lowered.launches}
            self.assertEqual("LoweredMemset" in kinds, style == "scratch")

    @parametrize("first", [512, 640])
    def test_binding_switches(self, first):
        # bindings of both topologies, and two alignments of y, alternate; the
        # first call's topology is the segment's, the other a piece of a clone
        p = TableProvider(style="mixed")
        f = HostTraceReplay(mul_chain, opaque=(p,))
        base = torch.randn(4096, device="cuda")
        cases = [(first, 0), (1152 - first, 0), (first, 4), (1152 - first, 4)]
        for _ in range(3):
            for n, offset in cases:
                f(torch.randn(n, device="cuda"), base[offset : offset + n])
        # one variant: the other topology is a piece of its site's segment,
        # each alignment a row of its table
        self.assertEqual(len(bound(f)), 1)
        traces, replays = f.traces, f.replays
        for i in range(24):
            # a form each call, then each form twice
            n, offset = cases[i % 4 if i < 12 else i // 2 % 4]
            x, y = torch.randn(n, device="cuda"), base[offset : offset + n]
            base.normal_()
            sets = torch._C._host_trace_memory_node_sets()
            self.assertEqual(f(x, y), mul_chain(x, y), atol=0, rtol=0)
            # the scratch binding's memset node is set every replay, in either form
            self.assertEqual(torch._C._host_trace_memory_node_sets() - sets, int(n % 256 == 0))
        self.assertEqual((f.traces, f.replays), (traces, replays + 24))

    def test_bound_key_relowers(self):
        # a key that binds after the trace relowers the tape with its binding
        p = TableProvider(style="scratch")
        f = HostTraceReplay(mul_chain, opaque=(p,))
        for n in (512, 640):
            for _ in range(4):
                x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
                self.assertEqual(f(x, y), mul_chain(x, y), atol=0, rtol=0)
        self.assertEqual((f.traces, f.relowers, len(f.variants), len(bound(f))), (1, 1, 2, 1))
        self.assertEqual(sum(b is not None for b in p.table.values()), 2)

    @parametrize("provider", ["table", "harvest"])
    @parametrize("learned", ["both", "first"])
    def test_relowered_tape_is_the_bound_trace(self, provider, learned):
        # a learner's tape with its now-bound keys bound (bind_opaque) lowers
        # and plans as the trace at the same call with them bound; the first
        # call's output feeds the second, a launch and the call's result
        if provider == "table":
            p = TableProvider(style="scratch")
            op, k = torch.mul, 512
            args = (torch.randn(1024, device="cuda"), torch.randn(1024, device="cuda"), torch.randn(k, device="cuda"))
        else:
            p = HarvestProvider()
            op, k = torch.mm, 128
            args = tuple(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) for _ in range(3))

        def fn(x, y, z):
            m = op(add(x), y)
            return m, add(op(m[:k], z))

        learner = trace(fn, args, opaque=(p,))
        f = HostTraceReplay(fn if learned == "both" else lambda x, y, z: op(add(x), y), opaque=(p,))
        for _ in range(3):
            f(*args)  # the eager steps learn the keys
        learner.release_args()
        calls = [r for _, r in learner.launches if isinstance(r, OpaqueCall)]
        self.assertEqual(len(calls), 2)
        wanted = calls if learned == "both" else calls[:1]
        relowered, sites = bind_opaque(learner, {id(c) for c in wanted})
        fresh = trace(fn, args, opaque=(p,))
        self.assertEqual(len(sites), len(wanted))
        self.assertEqual(sum(isinstance(r, OpaqueCall) for _, r in fresh.launches), 2 - len(wanted))
        self.assertEqual([str(g) for g in relowered.guards], [str(g) for g in fresh.guards])
        a, b = lower_tape(relowered), lower_tape(fresh)
        self.assertEqual(lowered_rows(a), lowered_rows(b))
        self.assertEqual(plan_memory(a), plan_memory(b))
        self.assertEqual(split_runs(a, ())[1], split_runs(b, ())[1])

    def test_memory_plan(self):
        p = TableProvider(style="scratch")

        def fn(x, y):
            h = add(x)
            m = torch.mul(h, y)  # its output escapes and a later launch reads it
            return m, add(m.sin(), 1)

        f = HostTraceReplay(fn, opaque=(p,))
        for n in (1024, 1024, 1024, 2048, 2048, 2048, 1024):
            x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            self.assertEqual(f(x, y), fn(x, y), atol=0, rtol=0)
        self.assertEqual(len(bound(f)), 1)
        for v in f.variants:
            self.assertEqual(plan_errors(v), [])
        tape = bound(f)[0].tape
        (memset,) = [r for _, r in tape.launches if isinstance(r, Memset)]
        # the binding's kernels; sin is traced by the pointwise host
        kernels = [r for _, r in tape.launches if isinstance(r, KernelLaunch) and r.name.startswith("aten.mul")]
        self.assertEqual([k.name for k in kernels], ["aten.mul.Tensor node 1", "aten.mul.Tensor node 2"])
        self.assertEqual(kernels[0].fields, ((0, 0, 8), (1, 0, 8), (2, 0, 8)))
        # the scratch buffer is a temporary of the run, the output a tensor
        memory = bound(f)[0].memory
        temporaries = {k for m in memory.steps for k, _, _ in m.temporaries}
        scratch = [k for k, a in enumerate(tape.allocs) if a.dtype == torch.uint8]
        self.assertEqual(len(scratch), 1)
        self.assertLessEqual(set(scratch), temporaries)
        self.assertEqual(len(memory.outputs), 2)  # m and the last add
        lowered = bound(f)[0].captured.lowered
        self.assertEqual(sum(isinstance(lo, LoweredMemset) for lo in lowered.launches), 1)


instantiate_parametrized_tests(TestOpaqueCalls)


def setUpModule():
    import torch.cuda._host_trace_capture as capture

    capture.raise_trace_disagreements = True
    torch.cuda._host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
