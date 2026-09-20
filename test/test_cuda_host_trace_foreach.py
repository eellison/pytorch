# Owner(s): ["module: cuda"]

import re
import unittest

from host_trace_testing import assert_eager_function_handles

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfRocm,
    TEST_CUDA_PYTHON_BINDINGS,
    TestCase,
)


if torch.cuda.is_available():
    import host_trace_two_hint as two_hint

    from torch.cuda import _host_trace as ht

C = torch._C

# MultiTensorApply.cuh: the chunk size and the per-launch capacities of the
# metadata block (the 32 KB kernel-argument layout on CUDA 13, 4 KB before)
_CHUNK = 65536
_CUDA13 = torch.version.cuda is not None and int(torch.version.cuda.split(".")[0]) >= 13
_MAX_TENSORS = (770, 448, 336, 252, 210) if _CUDA13 else (110, 64, 48, 36, 30)
_MAX_BLOCKS = 2240 if _CUDA13 else 320
_BLOCK_INDEX_SIZE = 2 if _CUDA13 else 1

# the mangled kernel name with the translation unit's anonymous-namespace
# token (`<len>_GLOBAL__N__<hash>_<len>_<file>_cu_<hash>`) removed: the sibling
# instantiates the same template with the same arguments from its own file
_ANON = re.compile(r"\d+_GLOBAL__N__[0-9a-f]{8}_\d+_\w+?_cu_[0-9a-f]{8}")


def _family(name):
    return _ANON.sub("", name)


def _bits(t):
    return t.contiguous().view(
        {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
            t.element_size()
        ]
    )


def _launch_plan(numels, depth):
    # the chunking loop's launches: [(tensors, blocks)] per launch
    max_tensors, max_blocks = _MAX_TENSORS[depth - 1], _MAX_BLOCKS
    plan, tensors, blocks = [], 0, 0
    for numel in numels:
        if numel == 0:
            continue
        tensors += 1
        chunks = -(-numel // _CHUNK)
        for chunk in range(chunks):
            blocks += 1
            last = chunk == chunks - 1
            if (tensors == max_tensors and last) or blocks == max_blocks:
                plan.append((tensors, blocks))
                blocks = 0
                tensors = 0 if last else 1
    if blocks:
        plan.append((tensors, blocks))
    return plan


def _layout(depth, fused):
    # byte offsets of the metadata block's members and the block's size
    mt, mb = _MAX_TENSORS[depth - 1], _MAX_BLOCKS
    off = {"addresses": 0}
    pos = depth * mt * 8
    off["numel"] = pos
    pos += mt * 8
    if fused:
        off["state_steps"] = pos
        pos += mt * 8
    off["block_to_tensor"] = pos
    pos += mb * _BLOCK_INDEX_SIZE
    pos = (pos + 3) // 4 * 4
    off["block_to_chunk"] = pos
    pos += mb * 4
    off["start"] = pos
    pos += 4
    off["size"] = (pos + 7) // 8 * 8
    off["stride"] = mt * 8
    return off


def _written_ranges(depth, fused, tensors, blocks):
    # the slots the host writes for one launch: the pointer slots of each
    # list, the numels, the step pointers and the two block tables
    lay = _layout(depth, fused)
    ranges = []
    for d in range(depth):
        base = lay["addresses"] + d * lay["stride"]
        ranges.append((base, base + tensors * 8))
    ranges.append((lay["numel"], lay["numel"] + tensors * 8))
    if fused:
        ranges.append((lay["state_steps"], lay["state_steps"] + tensors * 8))
    ranges.append(
        (lay["block_to_tensor"], lay["block_to_tensor"] + blocks * _BLOCK_INDEX_SIZE)
    )
    ranges.append((lay["block_to_chunk"], lay["block_to_chunk"] + blocks * 4))
    return ranges


def _align(x, a):
    return (x + a - 1) // a * a


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceForeach(TestCase):
    # ---- helpers

    def _capture(self, fn):
        # the kernel nodes one call produces: (name, grid, block, smem, image)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(g, stream=stream, capture_error_mode="relaxed"):
                fn()
        stream.synchronize()
        e = C._HostTraceExec(g, torch.cuda.current_device())
        return [
            (
                e.kernel_name(j),
                tuple(e.grid(j)),
                tuple(e.block(j)),
                e.smem(j),
                e.image(j),
            )
            for j in range(e.num_nodes)
        ]

    def _same_launches(self, real, ours, plan, depth, fused, tail):
        # the same kernel instantiation and launch configuration per launch;
        # the image compared on the slots the host writes and on the kernel's
        # scalar arguments after the block (`tail`: (offset, size) pairs from
        # the block's size); the rest of a block is uninitialized in eager
        self.assertEqual([len(real), len(ours)], [len(plan), len(plan)])
        for j, ((tensors, blocks), r, o) in enumerate(zip(plan, real, ours)):
            self.assertEqual(
                _family(o[0]), _family(r[0]), f"launch {j}: {o[0]} vs {r[0]}"
            )
            self.assertEqual(o[1:4], r[1:4], f"launch {j}")
            self.assertEqual(o[1], (blocks, 1, 1), f"launch {j}")
            self.assertEqual(len(o[4]), len(r[4]))
            size = _layout(depth, fused)["size"]
            ranges = _written_ranges(depth, fused, tensors, blocks) + [
                (size + off, size + off + n) for off, n in tail
            ]
            for lo, hi in ranges:
                self.assertEqual(
                    o[4][lo:hi], r[4][lo:hi], f"launch {j}: bytes {lo}:{hi}"
                )

    def _adamw_lists(self, device, shapes, dtype, amsgrad, state_dtype=None):
        state_dtype = state_dtype or dtype
        torch.manual_seed(0)
        params = [torch.randn(s, device=device, dtype=dtype) for s in shapes]
        grads = [torch.randn_like(p) * 1e-2 for p in params]
        exp_avgs = [torch.randn_like(p, dtype=state_dtype) * 1e-3 for p in params]
        exp_avg_sqs = [torch.rand_like(p, dtype=state_dtype) * 1e-4 for p in params]
        max_sqs = (
            [torch.rand_like(p, dtype=state_dtype) * 1e-4 for p in params]
            if amsgrad
            else []
        )
        steps = [torch.zeros((), device=device, dtype=torch.float32) for _ in params]
        return [params, grads, exp_avgs, exp_avg_sqs, max_sqs, steps]

    def _clone_lists(self, lists):
        return [[t.clone() for t in lst] for lst in lists]

    def _flat(self, lists):
        return tuple(t for lst in lists for t in lst)

    def _step(self, n, amsgrad, kw, lr_tensor=False):
        # the training step's optimizer piece, in place on the flat arguments:
        # the counters advance, then one fused launch over every list
        def step(*flat):
            m = 5 if amsgrad else 4
            lists = [list(flat[i * n : (i + 1) * n]) for i in range(m)]
            steps = list(flat[m * n : (m + 1) * n])
            lr = flat[(m + 1) * n] if lr_tensor else kw["lr"]
            params, grads, exp_avgs, exp_avg_sqs = lists[:4]
            max_sqs = lists[4] if amsgrad else []
            torch._foreach_add_(steps, 1)
            torch._fused_adamw_(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_sqs,
                steps,
                lr=lr,
                beta1=kw["beta1"],
                beta2=kw["beta2"],
                weight_decay=kw["weight_decay"],
                eps=kw["eps"],
                amsgrad=amsgrad,
                maximize=kw.get("maximize", False),
            )
            return params

        return step

    def _flat_args(self, lists, lr_tensor=None):
        lists = [lst for lst in lists if lst]
        args = self._flat(lists)
        return args + ((lr_tensor,) if lr_tensor is not None else ())

    def _check_steps(self, variant, step, lists, n_steps, lr_tensor=None, what=""):
        # replay n steps on one copy of the state, eager on another; every
        # tensor bitwise equal after every step (the grads untouched)
        ours, ref = self._clone_lists(lists), self._clone_lists(lists)
        lr_ours = lr_tensor.clone() if lr_tensor is not None else None
        lr_ref = lr_tensor.clone() if lr_tensor is not None else None
        for k in range(n_steps):
            variant.replay(self._flat_args(ours, lr_ours))
            step(*self._flat_args(ref, lr_ref))
            torch.cuda.synchronize()
            for name, a, b in zip(
                ("params", "grads", "exp_avgs", "exp_avg_sqs", "max", "steps"),
                ours,
                ref,
            ):
                for i, (x, y) in enumerate(zip(a, b)):
                    self.assertTrue(
                        torch.equal(_bits(x), _bits(y)), f"{what} step {k} {name}[{i}]"
                    )
            self.assertEqual(ours[-1][0].item(), float(k + 1), what)

    _KW = {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.01, "eps": 1e-8}
    _SHAPES_3 = [(768, 768), (768,), (3072, 768)]
    _SHAPES_50 = [
        (3,),
        (17, 5),
        (1, 65536),
        (65537,),
        (2, 2, 2),
        (256, 256),
        (),
        (100, 3),
    ] * 6 + [(7,), (9,)]

    # ---- _fused_adamw_

    def test_fused_adamw_parity_with_the_real_op(self, device):
        for shapes, dtype in (
            (self._SHAPES_3, torch.float32),
            (self._SHAPES_50, torch.bfloat16),
            (self._SHAPES_3, torch.float16),
        ):
            for amsgrad in (False, True):
                lists = self._adamw_lists(device, shapes, dtype, amsgrad)
                params, grads, exp_avgs, exp_avg_sqs, max_sqs, steps = lists
                torch._foreach_add_(steps, 1)
                kw = dict(self._KW, amsgrad=amsgrad, maximize=False)
                want = self._clone_lists(lists)
                torch._fused_adamw_(
                    want[0], want[1], want[2], want[3], want[4], want[5], **kw
                )
                got = self._clone_lists(lists)
                C._host_trace_fused_adamw_(
                    got[0],
                    got[1],
                    got[2],
                    got[3],
                    got[4],
                    got[5],
                    None,
                    kw["lr"],
                    kw["beta1"],
                    kw["beta2"],
                    kw["weight_decay"],
                    kw["eps"],
                    amsgrad,
                    False,
                    None,
                    None,
                )
                torch.cuda.synchronize()
                for a, b in zip(want, got):
                    for x, y in zip(a, b):
                        self.assertTrue(
                            torch.equal(_bits(x), _bits(y)),
                            f"{shapes[:3]} {dtype} amsgrad={amsgrad}",
                        )
                depth = 5 if amsgrad else 4
                real = self._capture(
                    lambda: torch._fused_adamw_(
                        params, grads, exp_avgs, exp_avg_sqs, max_sqs, steps, **kw
                    )
                )
                ours = self._capture(
                    lambda: C._host_trace_fused_adamw_(
                        params,
                        grads,
                        exp_avgs,
                        exp_avg_sqs,
                        max_sqs,
                        steps,
                        None,
                        kw["lr"],
                        kw["beta1"],
                        kw["beta2"],
                        kw["weight_decay"],
                        kw["eps"],
                        amsgrad,
                        False,
                        None,
                        None,
                    )
                )
                plan = _launch_plan([p.numel() for p in params], depth)
                # after the block: the functor (1 byte), then lr_ptr, lr, beta1,
                # beta2, weight_decay, eps (8 each), maximize (1), grad_scale_ptr,
                # found_inf_ptr
                base = _align(1, 8)
                tail = [(base + 8 * k, 8) for k in range(6)] + [
                    (base + 48, 1),
                    (base + 56, 8),
                    (base + 64, 8),
                ]
                self.assertEqual(
                    len(ours[0][4]), _layout(depth, True)["size"] + base + 72
                )
                self._same_launches(real, ours, plan, depth, True, tail)

    def test_fused_adamw_replays_bitwise_over_five_steps(self, device):
        for shapes, dtype in (
            (self._SHAPES_3, torch.float32),
            (self._SHAPES_50, torch.bfloat16),
        ):
            lists = self._adamw_lists(device, shapes, dtype, False)
            n = len(shapes)
            step = self._step(n, False, self._KW)
            scratch = self._clone_lists(lists)
            tape = ht.trace(step, self._flat_args(scratch))
            self.assertEqual(tape.num_launches, 2)
            self.assertEqual(tape.num_memcpys, 0)
            variant = ht.build(tape, step, self._flat_args(scratch))
            self._check_steps(variant, step, lists, 5, what=f"{n} tensors {dtype}")

    def test_fused_adamw_amsgrad_weight_decay_and_tensor_lr(self, device):
        for amsgrad in (False, True):
            for lr_tensor in (False, True):
                for weight_decay in (0.0, 0.05):
                    kw = dict(self._KW, weight_decay=weight_decay)
                    lists = self._adamw_lists(
                        device, self._SHAPES_3, torch.float32, amsgrad
                    )
                    lr = torch.full((), kw["lr"], device=device) if lr_tensor else None
                    step = self._step(3, amsgrad, kw, lr_tensor=lr_tensor)
                    scratch = self._clone_lists(lists)
                    lr_scratch = lr.clone() if lr is not None else None
                    args = self._flat_args(scratch, lr_scratch)
                    tape = ht.trace(step, args)
                    variant = ht.build(tape, step, args)
                    self._check_steps(
                        variant,
                        step,
                        lists,
                        5,
                        lr,
                        what=f"amsgrad={amsgrad} lr_tensor={lr_tensor} wd={weight_decay}",
                    )

    def test_fused_adamw_maximize_grad_scale_and_found_inf(self, device):
        # the AMP form: the kernel unscales the gradient in place and skips
        # the update when found_inf is set; the grads are written, so their
        # addresses are taken through the mutable accessor
        lists = self._adamw_lists(device, self._SHAPES_3, torch.float32, False)
        grad_scale = torch.full((), 4.0, device=device)
        for found in (0.0, 1.0):
            found_inf = torch.full((), found, device=device)

            def step(*flat):
                p, g, m, v, s = (list(flat[i * 3 : (i + 1) * 3]) for i in range(5))
                torch._foreach_add_(s, 1)
                torch._fused_adamw_(
                    p,
                    g,
                    m,
                    v,
                    [],
                    s,
                    lr=1e-3,
                    beta1=0.9,
                    beta2=0.999,
                    weight_decay=0.01,
                    eps=1e-8,
                    amsgrad=False,
                    maximize=True,
                    grad_scale=grad_scale,
                    found_inf=found_inf,
                )
                return p

            scratch = self._clone_lists(lists)
            tape = ht.trace(step, self._flat_args(scratch))
            variant = ht.build(tape, step, self._flat_args(scratch))
            self._check_steps(variant, step, lists, 3, what=f"found_inf={found}")

    def test_fused_adamw_mixed_precision_states(self, device):
        # fp32 params with bf16 moments take the mixed-precision functor
        lists = self._adamw_lists(
            device, self._SHAPES_3, torch.float32, False, state_dtype=torch.bfloat16
        )
        step = self._step(3, False, self._KW)
        scratch = self._clone_lists(lists)
        tape = ht.trace(step, self._flat_args(scratch))
        variant = ht.build(tape, step, self._flat_args(scratch))
        self.assertIn("FusedAdamMathFunctorMP", tape.launches[1]["kernel"])
        self._check_steps(variant, step, lists, 3, what="mixed precision")

    def test_fused_adamw_two_hundred_tensors_cross_the_block_capacity(self, device):
        # 200 tensors whose chunks exceed one launch's block capacity: the
        # fused step is several launches, each with its own slots and zeros
        # elsewhere; the counters' add stays one launch
        big = _MAX_BLOCKS // 100 + 1
        shapes = [
            (big * _CHUNK,) if i % 2 else (i % 7 + 1, i % 5 + 1) for i in range(200)
        ]
        plan = _launch_plan([torch.Size(s).numel() for s in shapes], 4)
        self.assertGreater(len(plan), 1)
        lists = self._adamw_lists(device, shapes, torch.bfloat16, False)
        step = self._step(200, False, self._KW)
        scratch = self._clone_lists(lists)
        tape = ht.trace(step, self._flat_args(scratch))
        self.assertEqual(tape.num_launches, 1 + len(plan))
        for j, (tensors, blocks) in enumerate(plan):
            launch = tape.launches[1 + j]
            self.assertEqual([int(g) for g in launch["grid"]], [blocks, 1, 1])
            self.assertEqual(
                sum(
                    1
                    for p in launch["params"]
                    if p["name"].startswith("numel_for_tensor")
                ),
                tensors,
            )
        variant = ht.build(tape, step, self._flat_args(scratch))
        self._check_steps(variant, step, lists, 2, what="200 tensors")
        del lists, scratch, variant, tape

    def test_fused_adamw_many_small_tensors_cross_the_tensor_capacity(self, device):
        n = _MAX_TENSORS[3] + 8
        shapes = [(i % 3 + 1,) for i in range(n)]
        lists = self._adamw_lists(device, shapes, torch.float32, False)
        step = self._step(n, False, self._KW)
        scratch = self._clone_lists(lists)
        tape = ht.trace(step, self._flat_args(scratch))
        self.assertEqual(tape.num_launches, 1 + len(_launch_plan([1] * n, 4)))
        self.assertEqual(tape.num_launches, 3)
        variant = ht.build(tape, step, self._flat_args(scratch))
        self._check_steps(variant, step, lists, 2, what=f"{n} tensors")

    def test_fused_adamw_serves_the_chunk_class_and_misses_across_it(self, device):
        # the block tables are constants of the chunk decomposition: another
        # numel within a tensor's chunk count serves (the numel is a field),
        # one that adds a chunk misses on the chunk guard, and another list
        # length misses on the arity
        def lists_for(n_rows):
            return self._adamw_lists(
                device, [(n_rows, 256), (768,), (16, 16)], torch.float32, False
            )

        base = lists_for(200)  # 51200 elements: one chunk
        step = self._step(3, False, self._KW)
        scratch = self._clone_lists(base)
        tape = ht.trace(step, self._flat_args(scratch))
        variant = ht.build(tape, step, self._flat_args(scratch))
        self._check_steps(
            variant, step, lists_for(255), 2, what="255 rows"
        )  # 65280: one chunk
        self._check_steps(variant, step, lists_for(2), 2, what="2 rows")
        with self.assertRaisesRegex(ht.Miss, r"6553[56]|guard"):
            variant.replay(self._flat_args(lists_for(257)))  # 65792: two chunks
        with self.assertRaises(ht.Miss):
            variant.replay(self._flat_args(lists_for(2048)))
        with self.assertRaisesRegex(ht.Miss, "arguments"):
            variant.replay(
                self._flat_args(
                    self._adamw_lists(device, [(4, 4), (8,)], torch.float32, False)
                )
            )

    def test_fused_adamw_declines_and_refuses_by_name(self, device):
        lists = self._adamw_lists(device, self._SHAPES_3, torch.float32, False)
        step = self._step(3, False, self._KW)
        # a CPU tensor lr is read on the host at the trace: declined by name
        lr_cpu = torch.full((), 1e-3)

        def step_cpu_lr(*flat):
            p, g, m, v, s = (list(flat[i * 3 : (i + 1) * 3]) for i in range(5))
            torch._fused_adamw_(
                p,
                g,
                m,
                v,
                [],
                s,
                lr=lr_cpu,
                beta1=0.9,
                beta2=0.999,
                weight_decay=0.0,
                eps=1e-8,
                amsgrad=False,
                maximize=False,
            )
            return p

        with self.assertRaisesRegex(ht.Declined, "tensor lr"):
            ht.trace(step_cpu_lr, self._flat_args(self._clone_lists(lists)))
        # eager's own checks with eager's texts: a state list of another dtype
        bad = self._clone_lists(lists)
        bad[1][0] = bad[1][0][:-1].clone()
        with self.assertRaisesRegex(
            RuntimeError, "must have same dtype, device, and layout"
        ):
            step(*self._flat_args(bad))
        # the warm-up call raises eager's error before the symbolic run; the
        # symbolic run alone raises the entry's identical check as a decline
        with self.assertRaisesRegex(
            RuntimeError, "must have same dtype, device, and layout"
        ):
            ht.trace(step, self._flat_args(bad))
        with self.assertRaisesRegex(
            ht.Declined, "must have same dtype, device, and layout"
        ):
            ht.trace(step, self._flat_args(bad), warm_up=False)

    def test_fused_adamw_grads_stay_lazy_without_a_grad_scale(self, device):
        # without a grad scale the kernel only reads the gradients, so their
        # addresses go through the const accessor: a copy-on-write gradient
        # stays lazy through the ordinary op, the warm-up, the build and the
        # replays; the params and moments are written and materialize
        lists = self._adamw_lists(device, self._SHAPES_3, torch.float32, False)
        lazy = [torch._lazy_clone(g) for g in lists[1]]
        self.assertTrue(all(C._is_cow_tensor(g) for g in lazy))
        lists[1] = lazy
        step = self._step(3, False, self._KW)
        args = self._flat_args(lists)
        variant = ht.build(ht.trace(step, args), step, args)
        self.assertTrue(all(C._is_cow_tensor(g) for g in lazy))
        variant.replay(args)
        torch.cuda.synchronize()
        self.assertTrue(all(C._is_cow_tensor(g) for g in lazy))

    # ---- _foreach_add_

    def test_foreach_add_scalar_replays_the_step_counters(self, device):
        n = 196
        steps = [torch.zeros((), device=device) for _ in range(n)]

        def advance(*s):
            torch._foreach_add_(list(s), 1)
            return s

        real = self._capture(lambda: torch._foreach_add_(steps, 1))
        ours = self._capture(lambda: C._host_trace_foreach_add_scalar_(steps, 1))
        # the block, the functor (1 byte), the plus functor (1 byte), the float scalar
        self._same_launches(real, ours, _launch_plan([1] * n, 1), 1, False, [(4, 4)])
        args = tuple(t.clone() for t in steps)
        tape = ht.trace(advance, args)
        self.assertEqual(tape.num_launches, 1)
        variant = ht.build(tape, advance, args)
        fresh = tuple(torch.zeros((), device=device) for _ in range(n))
        for k in range(3):
            variant.replay(fresh)
            torch.cuda.synchronize()
            self.assertEqual([t.item() for t in fresh], [float(k + 1)] * n)

    def test_foreach_add_list_with_alpha_replays(self, device):
        def axpy(a0, a1, a2, b0, b1, b2):
            torch._foreach_add_([a0, a1, a2], [b0, b1, b2], alpha=0.5)
            return a0, a1, a2

        def make(rows):
            torch.manual_seed(1)
            xs = [
                torch.randn(rows, 64, device=device, dtype=torch.bfloat16),
                torch.randn(5, device=device, dtype=torch.bfloat16),
                torch.randn(3, 7, 9, device=device, dtype=torch.bfloat16),
            ]
            return tuple(xs) + tuple(torch.randn_like(x) for x in xs)

        base = make(16)
        real = self._capture(
            lambda: torch._foreach_add_(list(base[:3]), list(base[3:]), alpha=0.5)
        )
        ours = self._capture(
            lambda: C._host_trace_foreach_add_list_(list(base[:3]), list(base[3:]), 0.5)
        )
        self._same_launches(
            real,
            ours,
            _launch_plan([x.numel() for x in base[:3]], 2),
            2,
            False,
            [(4, 4)],
        )
        tape = ht.trace(axpy, base)
        self.assertEqual(tape.num_launches, 1)
        variant = ht.build(tape, axpy, base)
        for rows in (16, 2, 1000):
            args = make(rows)
            want = [a + 0.5 * b for a, b in zip(args[:3], args[3:])]
            variant.replay(args)
            torch.cuda.synchronize()
            for x, w in zip(args[:3], want):
                self.assertTrue(torch.equal(_bits(x), _bits(w)), f"rows {rows}")
        with self.assertRaises(ht.Miss):
            variant.replay(make(2048))  # 131072 elements: two chunks
        # a size-1 dim is the fast route's stride-skip question, a branch the
        # trace took the other way (as for every sibling-iterator op)
        with self.assertRaises(ht.Miss):
            variant.replay(make(1))

    def test_foreach_add_slow_path_is_the_per_tensor_add(self, device):
        # strides that differ between the lists fail the fast route: eager
        # falls back to a per-tensor add_, and so does the sibling (each add_
        # its own traced launch)
        def axpy(a0, a1, b0, b1):
            torch._foreach_add_([a0, a1], [b0, b1], alpha=2)
            return a0, a1

        def make():
            torch.manual_seed(2)
            a = [torch.randn(8, 8, device=device), torch.randn(4, 6, device=device)]
            b = [
                torch.randn(8, 8, device=device).t(),
                torch.randn(6, 4, device=device).t(),
            ]
            return (*a, *b)

        base = make()
        tape = ht.trace(axpy, base)
        self.assertEqual(tape.num_launches, 2)
        self.assertFalse(
            any("multi_tensor_apply" in L["kernel"] for L in tape.launches)
        )
        variant = ht.build(tape, axpy, base)
        args = make()
        want = [a + 2 * b for a, b in zip(args[:2], args[2:])]
        variant.replay(args)
        torch.cuda.synchronize()
        for x, w in zip(args[:2], want):
            self.assertTrue(torch.equal(x, w))

    def test_foreach_add_declines_by_name(self, device):
        x = torch.randn(4, device=device)
        cpu = torch.tensor(1.0)
        with self.assertRaisesRegex(ht.Declined, "cpu tensor operand"):
            ht.trace(lambda a: torch._foreach_add_([a], [cpu]), (x,))
        # the overloads without a sibling are not traceable hosts
        with self.assertRaisesRegex(ht.Declined, "not a traceable CUDA host"):
            ht.trace(
                lambda a, b: torch._foreach_add_([a], b),
                (x, torch.ones((), device=device)),
            )

    def test_ordinary_ops_are_untouched(self, device):
        # the eager hosts launch as before: one multi_tensor_apply_kernel from
        # their own translation units
        lists = self._adamw_lists(device, self._SHAPES_3, torch.float32, False)
        params, grads, exp_avgs, exp_avg_sqs, _, steps = lists
        nodes = self._capture(
            lambda: torch._fused_adamw_(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                [],
                steps,
                **dict(self._KW, amsgrad=False, maximize=False),
            )
        )
        self.assertEqual(len(nodes), 1)
        self.assertIn("multi_tensor_apply_kernel", nodes[0][0])
        self.assertNotIn("Foreach", nodes[0][0])
        nodes = self._capture(lambda: torch._foreach_add_(steps, 1))
        self.assertEqual(len(nodes), 1)
        self.assertIn("multi_tensor_apply_kernel", nodes[0][0])

    @unittest.skipIf(not TEST_CUDA_PYTHON_BINDINGS, "cuda.bindings reads the nodes")
    def test_foreach_hosts_replay_eager_function_handles(self, device):
        # the entries are compiled into the eager hosts' translation units
        # (ForeachBinaryOpScalar.cu, ForeachBinaryOpList.cu, fused_adamw_impl.cu,
        # fused_adamw_amsgrad_impl.cu), where multi_tensor_apply_kernel's
        # anonymous-namespace instantiation lives: the tape's launch and the
        # replay's node hold the function handle eager's capture holds (E36),
        # not a name-identical second instantiation; one launch per case
        n = len(self._SHAPES_3)

        def add_scalar(*steps):
            torch._foreach_add_(list(steps), 1)
            return steps

        def add_list(*flat):
            torch._foreach_add_(list(flat[:n]), list(flat[n:]), alpha=0.5)
            return flat[:n]

        def adamw(amsgrad, lr_tensor):
            def step(*flat):
                m = 5 if amsgrad else 4
                lists = [list(flat[i * n : (i + 1) * n]) for i in range(m)]
                steps = list(flat[m * n : (m + 1) * n])
                lr = flat[(m + 1) * n] if lr_tensor else self._KW["lr"]
                torch._fused_adamw_(
                    *lists[:4],
                    lists[4] if amsgrad else [],
                    steps,
                    lr=lr,
                    beta1=self._KW["beta1"],
                    beta2=self._KW["beta2"],
                    weight_decay=self._KW["weight_decay"],
                    eps=self._KW["eps"],
                    amsgrad=amsgrad,
                    maximize=False,
                )
                return lists[0]

            return step

        f32 = self._adamw_lists(device, self._SHAPES_3, torch.float32, False)
        bf16_ams = self._adamw_lists(device, self._SHAPES_3, torch.bfloat16, True)
        mixed = self._adamw_lists(
            device, self._SHAPES_3, torch.float32, False, state_dtype=torch.bfloat16
        )
        lr = torch.full((), 1e-3, device=device)
        cases = {
            "_foreach_add_.Scalar (the step counters)": (add_scalar, tuple(f32[5])),
            "_foreach_add_.List with alpha": (add_list, tuple(f32[2] + f32[1])),
            "_fused_adamw_ f32": (adamw(False, False), self._flat_args(f32)),
            "_fused_adamw_ bf16 amsgrad": (
                adamw(True, False),
                self._flat_args(bf16_ams),
            ),
            "_fused_adamw_ mixed precision": (
                adamw(False, False),
                self._flat_args(mixed),
            ),
            "_fused_adamw_ tensor lr": (adamw(False, True), self._flat_args(f32, lr)),
        }
        for name, (fn, args) in cases.items():
            with self.subTest(case=name):
                eager = assert_eager_function_handles(self, fn, args, launches=1)
                self.assertEqual(len(eager), 1)

    def test_every_case_traces_the_same_program_under_other_hints(self, device):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint);
        # the block tables and the zeroed slots are the same bytes under both
        two_hint.assert_family(
            self,
            exclude={"test_exclusions": "a device-type class attribute, not a test"},
        )


instantiate_device_type_tests(TestCudaHostTraceForeach, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
