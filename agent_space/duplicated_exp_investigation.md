# The duplicated `exp`: IR, why CSE cannot help, and where the fusion dies

> **Historical measurement.** The experiment ran at the old tip
> `51304c0a0b8`, not the current seven-commit stack. The topology and conclusion
> remain useful; rerun before using the exact timings as current performance.

MXFP6 with a silu upstream emits **five `libdevice.exp`** where one would do.
This is the investigation: the actual scheduler IR for three variants, why this
is not a CSE bug, and the current best hypothesis for the fix.

Measured at stack tip `51304c0a0b8` on B200. Shapes shrunk to `B=4, D=64, G=32`
so the indices are readable; the real case is `8192 x 4096`.

## The computation

```python
z  = silu(x) * 1.125
xg = z.view(B, D//G, G).float()
s  = xg.abs().amax(-1) / 7.5        # reduction: wants z at FULL resolution
v  = (xg / s).round() & 0x3F        # epilogue:  wants z at LANE resolution
out = pack43(v)                      # combines 4 lanes -> 3 bytes
```

Two consumers of `z`. The amax wants all of it; the packing wants it four
elements at a time.

## Variant A — inlined prefix (what ships today)

Six nodes, all fusing into one kernel.

```
op0    group=(8, 32)      reduction=True
    R arg0_1  64*d0 + 32*d1 + d2           size=(4, 2, 32)   <- full resolution
    W buf0    2*d0 + d1                    size=(4, 2)
op1    group=(8, 1)       reduction=False
    R buf0 ; W buf1                                           <- the scale
op2    group=(64, 1)      reduction=False
    R arg0_1  64*d0 + 32*d1 + 4*d2         size=(4, 2, 8)    <- lane 0
    R arg0_1  64*d0 + 32*d1 + 4*d2 + 1     size=(4, 2, 8)    <- lane 1
    R buf1 ; W buf2                                           <- "low" byte
op3    group=(64, 1)
    R arg0_1  ... 4*d2 + 1 ; ... 4*d2 + 2  ; R buf1 ; W buf3  <- "mid" byte
op4    group=(64, 1)
    R arg0_1  ... 4*d2 + 2 ; ... 4*d2 + 3  ; R buf1 ; W buf4  <- "high" byte
op5    group=(192, 1)
    R buf2, buf3, buf4 ; W buf5            size=(4, 2, 8, 3)  <- the stack

AFTER fusion: [op0_op1_op2_op3_op4_op5]      1 kernel, exp=5
```

`silu` never appears as a node. Inductor **inlined** it into every consumer, so
each of `op0`, `op2`, `op3`, `op4` recomputes it at its own index. Four distinct
lane reads (`4*d2 + 0..3`) plus one full read (`32*d1 + d2`) = **5 silus**.

CSE is already working: `op2` and `op3` both read `4*d2 + 1`, and `op3`/`op4`
both read `4*d2 + 2`, so those collapse. Five is what remains after CSE, not
before it.

## Why CSE cannot fix the rest

The generated code is:

```python
tmp7  = tmp0.to(tl.float32)     # tmp0 = the full parent tile
tmp8  = -tmp7
tmp9  = libdevice.exp(tmp8)     # reduction's silu

tmp26 = tmp1.to(tl.float32)     # tmp1 = lane 0, a tl.split of tmp0
tmp27 = -tmp26
tmp28 = libdevice.exp(tmp27)    # epilogue's silu
```

Inductor's CSE caches *expression string -> variable*. `exp(tmp8)` and
`exp(tmp27)` are different keys because `tmp8` and `tmp27` hold **different
values** -- the whole tile versus one lane of it. There is no duplicate
expression to find.

The property CSE cannot express is *"this value is a sub-slice of that value."*
Once `tl.split` manufactures `tmp1`, everything downstream of it is legitimately
new work. Nothing is missing from CSE; the duplication is created before CSE
ever sees it, by splitting too early in the chain:

```
load ──split──> 4 lanes ──silu x4──> pack        5 exp   (today)
load ──silu──> ──split──> 4 lanes ──pack         1 exp   (wanted)
```

Both are valid because `split(silu(x)) == silu(split(x))` for elementwise
`silu`. The lanes only genuinely diverge at the first *lane-combining* op,
`low = v0 | ((v1 & 3) << 6)`.

**Why the planner splits at the load:** it reasons about `MemoryDep`s, so it can
only name **buffers**. `silu(x) * 1.125` is a fused intermediate with no buffer
name, so the deepest nameable thing is `arg0_1`, the kernel input.

## Variant B — name the prefix

Force `z` into a buffer. (Not with `torch.ops._inductor_test.realize`, which is
`x.realize(); return clone(x)` -- the clone is a separate node and confounds
everything. This uses a patched lowering that drops the clone.)

```
op0    group=(256, 1)     reduction=False
    R arg0_1  64*d0 + d1                   size=(4, 64)
    W buf0    64*d0 + d1                   size=(4, 64)      <- the silu, named
op1    group=(8, 32)      reduction=True
    R buf0    64*d0 + 32*d1 + d2           size=(4, 2, 32)
    W buf1
op2    group=(8, 1)  -> the scale
op3    group=(192, 1)     reduction=False
    R buf0    64*d0 + 32*d1 + 4*d2         size=(4, 2, 8)
    R buf0    64*d0 + 32*d1 + 4*d2 + 1     size=(4, 2, 8)
    R buf0    64*d0 + 32*d1 + 4*d2 + 2     size=(4, 2, 8)
    R buf0    64*d0 + 32*d1 + 4*d2 + 3     size=(4, 2, 8)
    R buf2 ; W buf3       size=(4, 2, 8, 3)
op4    group=(192, 1)     -> the uint8 cast

AFTER fusion: [op0]  [op1_op2_op3_op4]       2 kernels, exp=1
```

Two things changed. The packing collapsed from three nodes into one `op3`
reading all four lane offsets, and **`exp` dropped to 1** -- the split now
happens on `buf0`, which already holds the silu result. The mechanism works.

But `op0` did not join the group, so `buf0` is a real 67 MB round-trip at
production size, which costs far more than four `exp`.

## Variant C — named prefix, no epilogue

```
op0    group=(256, 1)   R arg0_1 ; W buf0
op1    group=(8, 32)    R buf0 (64*d0 + 32*d1 + d2) ; W buf1

AFTER fusion: [op0_op1]                      1 kernel, exp=1
```

The **same** `op0` and `op1`, with the same deps, fuse fine. So a named prefix
fusing into a reduction is not the problem. The only difference in B is `op3`.

## Where the fusion actually dies

`TORCH_LOGS=fusion` on variant B:

```
cannot fuse op0 with op1: no shared data
cannot fuse op0 with op1_op2: no shared data
cannot fuse op0 with op1_op2_op3_op4: no shared data
```

`"no shared data"` is the first check in `V.choices.can_fuse` -- it means
`score_fusion_memory(op0, ...)` returned **0**. Not a sub-parent guard.

Three sub-parent guards were suspected and all three cleared by experiment
(each neutralised in turn, with the clone confound removed): the
"internally produced source" rejection in `sub_parent_epilogue_plan`,
`_is_sub_parent_shaped` in the fusion decision, and the coalesce guard.
None of them changes the outcome.

The same log on variant C is the tell:

```
cannot fuse op0 with op1: no shared data     <- first pass fails
fusing op0 with op1                          <- succeeds later
```

C fails the same score check on the first pass and is **rescued by the reorder
round**, which re-normalises loop orders and recomputes the score. The score is
zero initially because the write dep `buf0 @ 64*d0 + d1, size (4,64)` and the
read dep `buf0 @ 64*d0 + 32*d1 + d2, size (4,2,32)` are not equal until the
read's `(2,32)` dims merge to `64`.

In B that rescue never lands.

## RESOLVED: the ideal codegen already exists, gated behind a realization heuristic

Variants B and C above are misleading, and so was the reorder-round hypothesis
that used to be in this section. Both were artifacts of *how* the prefix was
named -- a patched `_inductor_test.realize` without its clone, and returning `z`
as a graph output. Neither is how Inductor normally realizes a value.

Through the normal path (`mark_reuse` during lowering) it just works:

```
baseline (today)                  kernels=1  exp=5
realize_reads_threshold=1         kernels=1  exp=1     <- ideal
z also returned as an output      kernels=2  exp=1
realize_opcount_threshold=1       kernels=2  exp=1
```

With `realize_reads_threshold=1`, `z` becomes a `ComputedBuffer`, the producer
fuses into the group, the buffer is kernel-local (one kernel, so
`remove_kernel_local_buffers` drops it and it never reaches memory), and the
split lands on the silu *result*. One kernel, one `exp`, no round-trip.

So there is no missing capability, no split-placement work, and no deferred
ops-handler layer needed. The sub-parent machinery already does the right thing
when the shared prefix has a name.

## Cost: measured, and small

Same program, same single kernel, bit-identical outputs, differing only in 5
`exp` versus 1 (cudagraph-captured, `do_bench`, B200):

| shape | 5 exp | 1 exp | delta |
|---|---|---|---|
| 512 x 1024 | 8.11 us | 8.12 us | none |
| 8192 x 4096 | 63.65 us | 62.12 us | **2.4%** |

The kernel is bandwidth-bound enough that four extra transcendentals hide under
memory latency. This is the clean A/B that three earlier attempts failed to
produce -- each of those perturbed kernel count or memory traffic as well.

## Recommendation: leave it

`realize_reads_threshold=1` is a global knob and cannot ship -- it would realize
buffers across every graph in the compiler.

A targeted version hits a layering problem. Realization happens during
**lowering**; the sub-parent fusion decision happens in the **scheduler**. By
the time the epilogue is known, `silu` is already inlined into both consumers
and cannot be un-inlined. A lowering-time rule would need a local signal, and
the honest one is "this pointwise value is read at two different index strides"
(stride 1 by the amax, stride 4 by the packing) -- but that means editing
`realize_hint`/`mark_reuse`, which governs realization for every graph.

Worth 2.4% on one large shape and nothing on a small one, that is not a trade
worth making. Revisit only if a workload appears that is compute-bound in this
epilogue -- a longer or more expensive shared prefix would move the number.
