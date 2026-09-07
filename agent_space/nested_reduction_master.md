# Nested Reduction: Master Document

A single comprehensive synthesis of the design space, current state, and
future directions for PyTorch Inductor's dependent cross-axis reduction
fusion. Built from the contents of 12 separate design docs in
`agent_space/`. Self-contained for an unfamiliar reader.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background](#2-background)
3. [Why This Is Hard](#3-why-this-is-hard)
4. [The Inductor Stack](#4-the-inductor-stack)
5. [Anatomy of a Nested-Reduction Kernel](#5-anatomy-of-a-nested-reduction-kernel)
6. [The Iteration-Space Model](#6-the-iteration-space-model)
7. [Legality Invariants](#7-legality-invariants)
8. [Key Abstractions](#8-key-abstractions)
9. [Historical Iterations](#9-historical-iterations)
10. [Current Landing Candidates](#10-current-landing-candidates)
11. [Future Architectural Directions](#11-future-architectural-directions)
12. [Comparison Matrix](#12-comparison-matrix)
13. [Decision Tree and Recommendations](#13-decision-tree-and-recommendations)
14. [What's Preserved](#14-whats-preserved)
15. [Glossary](#15-glossary)
16. [Reference](#16-reference)

---

## 1. Executive Summary

**Feature**: fuse two dependent reductions over the same large input
(e.g. RMSNorm + per-block amax for FP8 quantization) into a single Triton
kernel. The shared input is read once, reused through later phases in
registers when legal.

**Performance impact**: 2–3× faster than the hand-written CUDA kernel for
RMSNorm + FP8 quantize on vLLM. NVFP4 packing within ~10% of hand-written.

**Status**: working implementation across multiple worktrees. 80/80 tests
passing on the full feature, 76/76 on the trimmed (no-NVFP4) variant.

**The challenge**: the implementation puts ~1500 LOC of feature-specific
machinery into `simd.py`. The feature is genuinely complex (multiple
iteration extents in one kernel, register-resident value flow across
stages, multiple consumer patterns), and Inductor wasn't built for this
shape of fusion. The PR is large.

**Recommended landing**: `nested_reduction_core_candidate` — trimmed
scope (no NVFP4), 2 reviewable commits, 76/76 tests, no architectural
argument needed in review. NVFP4 follows as a separate PR. The
architectural improvement (Path B+) follows as a third PR.

**Future architecture** (separate Inductor project, not blocking): unify
the iteration-family abstraction to handle nested reduction, NVFP4, and
future cat/scatter/MLA-style fusion patterns under one model. Captured
as design doc; awaits a second customer to amortize the cost.

---

## 2. Background

### 2.1 What "nested reduction" means here

A common pattern in modern numerical kernels is two reductions stacked
back-to-back over the same input, with the second reduction operating
on a reshape of the first reduction's working tile.

#### Pattern A — RMSNorm + per-block amax (FP8 quantization prep)

```python
def f(x, weight):
    x = F.rms_norm(x, (D,), weight)              # outer reduction (norm)
    x_groups = x.view(B, D // G, G)
    amax = x_groups.abs().amax(dim=-1)           # grouped reduction
    scale = (amax / fp8_max).clamp(min=1e-12)
    x_fp8 = (x_groups / scale.unsqueeze(-1))     # full-resolution use of x
                .to(torch.float8_e4m3fn)
    return x_fp8.view(B, D), scale
```

The semantics: normalize the row, then for every group of G elements
compute the max-of-abs (one `scale` per group), then quantize the row
using those per-group scales.

#### Pattern B — LayerNorm + per-block amax

Similar shape, different norm. `mean` and `var` reductions inform the
normalize step; per-group amax follows.

#### Pattern C — NVFP4 packing

After amax/scale, the row is split into even/odd lanes and packed
two-at-a-time into bytes. This is the "half-resolution" case: consumers
operate at `parent_extent / 2` with constant-lane access patterns.

### 2.2 The performance motivation

What's special about these patterns is that **the second reduction
reads exactly the same data the first reduction already brought into
registers**. Without fusion, you write the post-RMSNorm tile to memory,
read it back, do the amax, write the scale, read both `x` and scale
back, quantize. With fusion: read `x` once, do everything in registers,
write only the final outputs.

The performance impact is real:
- vLLM RMSNorm + FP8 quantize fusion: 2–3× faster than hand-written CUDA
- NVFP4 pass 3 (half-res packing): 0.034ms vs 0.031ms hand-written, vs
  0.094ms unfused — within ~10% of hand-written
- 100% exact match vs eager with `emulate_precision_casts=True`

---

## 3. Why This Is Hard

PyTorch Inductor's standard fusion machinery does a great job of fusing
pointwise + reduction (epilogue fusion) and pointwise + pointwise (loop
fusion). It does **not** natively express "two reductions over the same
data with intermediate ops between them in a single kernel."

Three structural reasons:

### 3.1 Different iteration extents

The outer reduction's body iterates over `[B, D]` producing `[B]` outputs.
The grouped reduction's body iterates over `[B, D//G, G]` producing
`[B, D//G]` outputs. Standard fusion picks one iteration extent per
kernel. Nested reduction needs four:

- Outer reduction: `[XBLOCK, RBLOCK]`
- Grouped reduction: `[XBLOCK, RBLOCK/G, G]` (or `[XBLOCK/G, G, RBLOCK]`)
- Reduced-output: `[XBLOCK, RBLOCK/G]`
- Full-resolution: `[XBLOCK, RBLOCK]`
- Half-resolution: `[XBLOCK, RBLOCK/2]`

### 3.2 Multi-stage value flow

The post-RMSNorm tile has to live in registers across the kernel, then
get reshaped (logically) for the grouped reduction, then potentially
used at full-resolution AND reduced-resolution by epilogue ops.
Standard codegen has a single-stage register-flow model.

### 3.3 Standard reduction codegen reduces over the trailing range tree

It can't natively express "reshape this tile, then reduce over an
interior axis." That's the operation a grouped reduction needs. To do
it, the codegen has to either:
- Emit an explicit `tl.reshape` then reduce (what's done today), or
- Add a derived range tree for the inner axis (a possible future design).

---

## 4. The Inductor Stack

Inductor has roughly four layers where fusion-pattern work *can* live:

| Layer | File | Role |
|---|---|---|
| Inductor IR | `torch/_inductor/ir.py` | `Buffer`, `Reduction`, `Pointwise`, `Loops` — the operations and their data |
| Scheduler | `torch/_inductor/scheduler.py` | Fusion decisions, ordering, scheduler nodes |
| Backend-shared codegen | `torch/_inductor/codegen/simd.py` | Range trees, indexing, kernel-level abstractions usable across Triton/CPP |
| Backend emission | `torch/_inductor/codegen/triton.py`, `cpp.py` | Actual Triton or C++ code strings |

Standard pipeline: lowering produces Inductor IR → scheduler groups
nodes into fused groups → backend codegen emits one kernel per fused
group.

For nested reduction, the question that runs through this whole
document is: **at which of these layers should the special handling
live?**

The current implementation puts most of the work in layer 3 (`simd.py`)
with slivers in layer 2 (`scheduler.py`) and layer 4 (`triton.py`).

---

## 5. Anatomy of a Nested-Reduction Kernel

A fused nested-reduction Triton kernel runs in this order:

### Stage 1: Outer reduction body
Execute `node1` (e.g. RMSNorm). Keep `node1` outputs in registers / CSE
when possible. The outer reduction produces a normalized tile that
stays in registers for downstream stages.

### Stage 2: Group reduction body
Reinterpret the outer-reduction tile as a grouped structure. For
example, `[XBLOCK, RBLOCK]` becomes logically `[XBLOCK, RBLOCK/G, G]`.
Reduce over `G`. Produce one value per group. The reduction is
emitted as `tl.sum(tl.reshape(value, [...]), axis=N)` or equivalent.

### Stage 3a: Reduced-resolution epilogues
Pointwise ops on the per-group output (e.g. divide amax by 448 to
compute scale). Run in iteration extent `[XBLOCK, RBLOCK/G]`.

### Stage 3b: Full-resolution epilogues
Pointwise ops that consume the per-group reduced value broadcast back
to the original tile shape (e.g. divide `x` by scale to quantize).
Run in iteration extent `[XBLOCK, RBLOCK]`. Reads the post-RMSNorm
tile from registers AND the reduced-resolution output broadcast back
to full extent.

### Stage 3c: Half-resolution consumers
Pointwise ops that operate at `parent_extent / 2` (e.g. NVFP4 even/odd
packing). Run in iteration extent `[XBLOCK, RBLOCK/2]`. Reads the
post-RMSNorm tile by splitting it into two lanes (even, odd).

### Where the complication lives

The complication: each of these stages runs in a *different effective
iteration extent*. Standard codegen assumes one iteration extent per
kernel. Nested reduction needs four, with explicit value flow between
them.

The core question becomes: **can each later phase be expressed as a
legal derived iteration space of the original tile?** That's what the
abstraction work is about.

---

## 5.bis Example Generated Triton Code

Here's what the codegen actually produces for each pattern. These are
real captured outputs from the working implementation, abbreviated to
the kernel body itself.

### 5.bis.1 Pattern B — RMSNorm + per-block amax (`small_dim_in_r`, no NVFP4)

Workload: `rms_norm(x).view(B, D//G, G).abs().amax(dim=-1)` with B=128,
D=4096, G=16.

```python
@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32',
                               'out_ptr0': '*fp32', ...}},
    inductor_meta={'kernel_name': 'triton_per_fused__fused_rms_norm_0',
                   'num_load': 2, 'num_store': 0, 'num_reduction': 1,
                   'min_rblock': 16, 'max_xblock': 256},
)
@triton.jit
def triton_per_fused__fused_rms_norm_0(in_ptr0, in_ptr1, out_ptr0,
                                       xnumel, r0_numel,
                                       XBLOCK: tl.constexpr):
    xnumel = 128
    r0_numel = 4096
    R0_BLOCK: tl.constexpr = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_1 = r0_index
    x0 = xindex

    # Stage 1: outer reduction (RMSNorm sum-of-squares)
    tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), xmask, other=0.0)   # x
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)                  # sum(x*x)

    # RMSNorm normalize
    tmp12 = tl.load(in_ptr1 + (r0_1), None,
                    eviction_policy='evict_last')                    # weight
    tmp7 = (tmp5 / 4096.0)
    tmp9 = tmp7 + 1.1920928955078125e-07
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp0 * tmp10            # post-RMSNorm tile, in registers
    tmp13 = tmp11 * tmp12

    # Stage 2: grouped reduction — reshape + reduce over inner G=16
    tmp14 = tl_math.abs(tmp13)
    tmp15 = tl.reshape(tmp14, [XBLOCK, R0_BLOCK // 16, 16])
    tmp16 = triton_helpers.max2(tmp15, 2)            # one value per group

    # Stage 3a: store reduced output at [XBLOCK, RBLOCK/16] extent
    pass2_x = xoffset + tl.arange(0, XBLOCK)[:, None]
    pass2_r = tl.arange(0, R0_BLOCK // 16)[None, :]
    pass2_idx = pass2_r + 256*pass2_x
    pass2_mask = (pass2_x < xnumel) & (pass2_r < 256)
    tl.store(out_ptr0 + pass2_idx, tmp16, pass2_mask)
```

Key observations:
- Single `tl.load` of in_ptr0 (the x input). Read once, reused.
- `tmp11` is the post-RMSNorm tile, lives in registers, never stored.
- The grouped reduction is `tl.reshape(...) → max2(..., axis=2)`.
- `pass2_*` variables are the reduced-output extent indices: half the
  number of dims, 16× smaller R extent.
- One `tl.store` for the final amax output.

### 5.bis.2 Pattern C — RMSNorm + FP8 quantize (full-resolution epilogue)

Workload: same as above but G=128, plus a full-resolution epilogue that
divides x by the per-group scale and quantizes to FP8.

```python
@triton.jit
def triton_per_fused__fused_rms_norm_0(in_ptr0, in_ptr1,
                                       out_ptr0, out_ptr1,
                                       xnumel, r0_numel,
                                       XBLOCK: tl.constexpr):
    # ... (Stage 1 + Stage 2 identical to above, with G=128) ...

    # Stage 2: grouped amax → scale
    tmp14 = tl_math.abs(tmp13)
    tmp15 = tl.reshape(tmp14, [XBLOCK, R0_BLOCK // 128, 128])
    tmp16 = triton_helpers.max2(tmp15, 2)

    # Stage 3a: reduced-output epilogue (compute scale)
    tmp17 = tl.full([1, 1], 0.002232142857142857, tl.float32)  # 1 / fp8_max
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full([1, 1], 1e-12, tl.float32)
    tmp20 = triton_helpers.maximum(tmp18, tmp19)               # scale per group

    # Stage 3b: full-resolution epilogue — broadcast scale back to full extent
    #   broadcast_to(reshape(scale, [..., 1]), [..., 128]) → reshape to flat
    tmp21 = tl.reshape(
        tl.broadcast_to(
            tl.reshape(tmp20, [XBLOCK, R0_BLOCK // 128, 1]),
            [XBLOCK, R0_BLOCK // 128, 128]
        ),
        [XBLOCK, R0_BLOCK]
    )
    tmp22 = (tmp13 / tmp21)                          # quantize at full extent
    tmp23 = tmp22.to(tl.float8e4nv)
    tmp24 = tmp23.to(tl.float32)

    # Two stores: scale at reduced extent, fp8 at full extent
    tl.store(out_ptr0 + (pass2_r + 32*pass2_x), tmp20, pass2_mask)
    tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp24, None)
```

Key observations:
- `tmp13` (post-RMSNorm) and `tmp20` (per-group scale) both live in
  registers across the stages.
- Full-resolution epilogue uses `tl.broadcast_to` to lift the per-group
  scale back to the full tile shape.
- Two stores: one at reduced extent (scale), one at full extent (fp8 output).

### 5.bis.3 NVFP4 — half-resolution split + pack

Workload: RMSNorm + per-block amax + NVFP4 packing (B=1, D=4096, G=16).
Even/odd consumers pack two FP4 values into one byte using inline asm.

```python
@triton.jit
def triton_per_fused__fused_rms_norm_0(in_ptr0, in_ptr1,
                                       out_ptr0, out_ptr1, out_ptr2,
                                       xnumel, r0_numel,
                                       XBLOCK: tl.constexpr):
    # ... Stage 1: RMSNorm ...
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tmp0 * tmp0
    tmp4 = tl.sum(tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK]), 1)[:, None]
    tl.store(out_ptr0 + ..., tmp4, None)              # mean(x^2) output
    tmp12 = tl.load(in_ptr1 + (r0_0 + 4096*x3), None)  # weight
    tmp10 = libdevice.rsqrt(tmp4 / 4096.0 + 1.19e-7)
    tmp11 = tmp0 * tmp10
    tmp13 = tmp11 * tmp12                              # post-RMSNorm

    # Stage 2: grouped amax (G=16)
    tmp14 = tl_math.abs(tmp13)
    tmp15 = tl.reshape(tmp14, [XBLOCK, R0_BLOCK // 16, 16])
    tmp16 = triton_helpers.max2(tmp15, 2)
    tmp20 = triton_helpers.maximum(tmp16 * 0.00223..., 1e-12)
    tmp21 = tmp20.to(tl.float8e4nv)              # scale (FP8)

    # Stage 3c: half-resolution split — even/odd lanes
    _rm_arg0_1_0, _rm_arg0_1_1 = tl.split(
        tl.reshape(tmp0, [1, R0_BLOCK // 2, 2])
    )
    tmp22 = _rm_arg0_1_0       # even lane of x
    tmp23 = _rm_arg0_1_1       # odd lane of x

    _rm_arg1_1_0, _rm_arg1_1_1 = tl.split(
        tl.reshape(tmp12, [XBLOCK, R0_BLOCK // 2, 2])
    )
    tmp24 = _rm_arg1_1_0       # even lane of weight
    tmp25 = _rm_arg1_1_1       # odd lane of weight

    # Broadcast scale to half-resolution extent
    tmp26 = tl.reshape(
        tl.broadcast_to(
            tl.reshape(tmp21, [XBLOCK, R0_BLOCK // 16, 1]),
            [XBLOCK, R0_BLOCK // 16, 16 // 2]   # half the inner dim
        ),
        [XBLOCK, R0_BLOCK // 2]
    )

    # Quantize even/odd separately
    tmp30 = (tmp22 * tmp10 * tmp24) / tmp26.to(tl.float32)   # even
    tmp33 = (tmp23 * tmp10 * tmp25) / tmp26.to(tl.float32)   # odd

    # NVFP4 pack: convert pair of FP32 to packed FP4 byte via inline asm
    tmp34 = tl.inline_asm_elementwise(
        '{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; '
        ' cvt.u32.u8 $0, t;}',
        '=r,f,f', [tmp30, tmp33],
        dtype=tl.int32, is_pure=True, pack=1
    )
    tmp35 = tmp34.to(tl.uint8)

    tl.store(out_ptr1 + ..., tmp21, pass2_mask)              # scale (FP8)
    tl.store(out_ptr2 + ..., tmp35, None)                    # packed FP4 bytes
```

Key observations:
- `tl.split(tl.reshape(value, [..., 2]))` is how the half-resolution
  family expresses the even/odd split. Each half-res consumer reads
  one of the resulting CSE variables (`_rm_arg0_1_0` or `_rm_arg0_1_1`).
- All three stages (outer reduction, grouped reduction, half-res
  consumers) share the same kernel.
- The post-RMSNorm value is reconstructed for even and odd lanes
  separately (`tmp22 * tmp10` for even, `tmp23 * tmp10` for odd) — this
  is what the half-resolution family's `Split` value source provides.
- The inline asm `cvt.rn.satfinite.e2m1x2.f32` is the NVFP4 packing
  instruction: pack two FP32 values into one E2M1 byte.

### 5.bis.4 What this tells you about the abstractions

Reading these three kernels side-by-side, you can see the iteration
families and their value transformations:

- **Stage 1 (outer reduction)**: `[XBLOCK, R0_BLOCK]` extent, standard
  reduction codegen.
- **Stage 2 (grouped reduction)**: emits `tl.reshape + max2` to do the
  reduce-after-reshape. This is `_GroupedReductionOpsHandler.reduction()`
  in code.
- **Stage 3a (reduced-output epilogue)**: indices like `pass2_r + 256 *
  pass2_x` — the reduced extent. `_DerivedIterationFamily` (kind
  `reduced_output`) is active.
- **Stage 3b (full-res epilogue)**: `tl.reshape(tl.broadcast_to(...))`
  lifts the reduced value back to full extent. `_DerivedIterationFamily`
  (kind `full_resolution`) with `Broadcast` value source.
- **Stage 3c (half-res NVFP4)**: `tl.split(tl.reshape(...))` produces
  two lane CSEs. `_DerivedIterationFamily` (kind `half_resolution`)
  with `Split` value source.

The same input (`tmp0`, `tmp12`) is loaded **once** at the top of the
kernel and reused across all stages. That's the perf win.

---

## 6. The Iteration-Space Model

There are two supported ways the grouped reduction can be embedded in
the outer-reduction tile.

### 6.1 `small_dim_in_r`

`group_size` lives in the outer reduction's *reduction* axis.

Conceptually:
- outer tile: `[XBLOCK, RBLOCK]`
- grouped view: `[XBLOCK, RBLOCK / G, G]`
- reduction axis: the last axis (axis=2)

This is the common `layernorm → per-group amax` / quantization shape.

### 6.2 `small_dim_in_x`

`group_size` lives in the outer reduction's *non-reduction* axis.

Conceptually:
- outer tile: `[XBLOCK, RBLOCK]`
- grouped view: `[XBLOCK / G, G, RBLOCK]`
- reduction axis: the middle axis (axis=1)

This is the cross-axis case where the smaller grouped dimension is
embedded in `X`, not `R`.

### 6.3 Why classification matters

The scheduler computes this classification once at fusion time and
codegen reuses it. Re-deriving it independently is what caused earlier
scheduler/codegen mismatches. The current code (`small_dim_in_r` field
on `FusedNestedReductions`) carries this decision through the pipeline.

---

## 7. Legality Invariants

The fusion is only legal when all of the following hold:

1. **`node2` is a grouped reinterpretation.** It can be expressed as a
   grouped view of one axis of `node1`'s tile.
2. **`group_size` is statically known.** The codegen path rejects
   symbolic group sizes.
3. **The grouped output is a derived iteration space.** Its shape and
   stride must be expressible as a derived range tree.
4. **Full-resolution epilogues correspond to the original tile.** The
   lifted value (broadcast of reduced result) must truly correspond to
   the original full tile, not a transformation of it.
5. **Half-resolution lane access is statically constant modulo split
   factor.** For the NVFP4 case (factor=2), this means even/odd-style
   access patterns only.

Violations of any of these reject the fusion at scheduler time.

---

## 8. Key Abstractions

These are the load-bearing types and helpers in the current
implementation. Locations refer to `single_commit_wip` baseline at
commit `03baa04b984`.

### 8.1 `DerivedIterationRangesRoot` (`simd.py:381`)

A subclass of `IterationRangesRoot` whose geometry is derived from a
parent tree. Used for the grouped output space, where the grouped
axis has:
- Smaller logical `numel`
- Smaller effective block size
- Smaller block offset

Lets reduced-resolution codegen use the standard indexing machinery.
Critical detail: `is_loop=parent.is_loop` — derived view inherits
looped-ness, otherwise its block offset closes over a stale outer
offset under non-persistent reductions.

### 8.2 `_DerivedIterationFamily` (`simd.py:1450`)

The consumer-side abstraction. A family is parameterized by:

- `range_trees`: the iteration variables and their extents
- `index_subs`: substitutions from logical body iter_vars to
  family-local vars (used by reduced-output)
- `remapped_values: dict[str, RemappedRangeValue]`: per-buffer-name
  CSE variables that are at this family's extent (broadcasted,
  split, or direct)
- `flat_index_expr`: optional flattened index expression for
  full-resolution

Usage:

```python
with reduced_output_family.activate(kernel):
    for ep_sn in reduced_output_epilogues:
        ep_sn._body(...)
```

While the family is active, all loads/stores in the body resolve
through the family's range trees, not the kernel's primary range
trees.

### 8.3 `_GroupReductionLayout` (`simd.py:1553`)

Frozen dataclass with the structural description of the grouped
reduction. Centralizes:
- Which tree is the grouped one (`group_tree`)
- Which tree is the "other" one (`other_tree`)
- Whether the grouped axis is X or R (`small_dim_in_r`)
- `reshape_shape`, `reduce_axis`, `output_shape`
- `broadcast_shapes`
- Child shapes for half-resolution consumers
- Flat-index reconstruction helpers

Factory methods produce iteration families:
- `make_reduced_output_family(x_var, r_var)` — `[XBLOCK, RBLOCK/G]`
- `make_full_resolution_family(kernel, names)` — `[XBLOCK, RBLOCK]`
  with broadcast values precomputed
- `make_half_resolution_family(factor=2)` — `[XBLOCK, RBLOCK/2]`

### 8.4 `_GroupedReductionOpsHandler` (`simd.py:1919`)

`WrapperHandler` subclass that runs during the reduction body's emit.
Three responsibilities:

1. **Reshape + reduce**: the `reduction()` method emits
   `kernel.emit_reshape` then `kernel.emit_reduce`, producing the
   grouped output tile.
2. **Capture register values**: the `load()` method captures CSE
   values during the reduction body for downstream half-res use
   (`_stage_load_values`).
3. **Route store**: the `store_reduction()` method writes to
   `cse.store_cache` (mirroring `KernelHandler.store_reduction`'s
   bookkeeping) and routes the physical store through the
   reduced-output family.

### 8.5 `_PointwiseRemapHandler` (`simd.py:2017`)

`WrapperHandler` for pointwise consumer epilogues. Used for:
- Reduced-resolution epilogues
- Full-resolution epilogues
- Half-resolution consumers

Load resolution order:
1. `family.remapped_values` (per-buffer CSE)
2. `kernel.cse.store_cache` (canonical store cache)
3. Fall-back to standard `kernel.load` with remapped index

### 8.6 `NestedReduction` and `FusedNestedReductions` (`scheduler.py`)

`NestedReduction.can_fuse` (`scheduler.py:445`) recognizes the pattern
at fusion time. It uses `MemoryDep` analysis to:
- Detect shared input reads between node1 and node2
- Classify `small_dim_in_r` based on stride patterns
- Validate that the grouped reduction is a clean reinterpretation

`FusedNestedReductions(FusedSchedulerNode)` (`scheduler.py:2535`) is
the scheduler node wrapping node1 + node2. Carries `small_dim_in_r`
classification computed once at fusion time. May also carry
half-resolution consumer membership (path-3) or a full plan dataclass
(Path B+).

### 8.7 `codegen_nested_reduction` (`simd.py:2625`)

The ~200-line orchestrator entry point. Drives the multi-stage emit:
1. Recognize node2 reduction vs epilogues; classify axis
2. Build kernel + set kernel knobs
3. Mark internal node1 outputs + run combined schedule
4. Set up node2 outputs + verify node1 in store_cache
5. Inside `with kernel:` — emit grouped reduction body + epilogue +
   half-res consumers + finalize

This is the orchestrator that all paths in this document either
preserve, simplify, or eliminate.

### 8.8 What's generic vs feature-specific

**Generic infrastructure** (could be reused for other patterns):
- `DerivedIterationRangesRoot`
- `use_range_trees` context manager
- `_DerivedIterationFamily`
- `_PointwiseRemapHandler`

**Feature-specific logic** (specific to nested reduction today):
- `_GroupReductionLayout`
- `_GroupedReductionOpsHandler`
- `codegen_nested_reduction` orchestrator
- `NestedReduction.can_fuse` (the recognition pattern)
- Full-resolution lifting
- Half-resolution split/broadcast
- B=1 flattened handling

---

## 9. Historical Iterations

These are concrete branches that lived at some point. They informed
the current shape but were superseded.

### A1. Original spike — `nested_reduction_backup`

Where everything started. Pattern detection, layout, handlers all
inline in codegen. Multiple specialized handlers per pass (`pass1`,
`pass2`, `pass3` corresponding to outer reduction, grouped reduction,
half-resolution). No abstraction over what they shared.

This was the working prototype but was stylistically unsustainable —
every new pattern (full-res epilogue, half-res NVFP4) added another
handler.

### A2. Derived-range probes — `nested_reduction_derived_range_attempt`, `derived_try_from_range`, `derived_try_from_range_v2`

First attempts at lifting iteration ranges into a shared abstraction.
Tried various ways to teach the kernel about derived ranges: a kernel
flag indicating "we're now in stage X," explicit range-tree swap APIs,
hierarchical range trees.

Most were abandoned because the kernel-internal indexing assumed range
trees were flat; making them hierarchical broke things downstream.
The descendant of these attempts is `DerivedIterationRangesRoot` plus
the `use_range_trees()` context manager — a flat swap rather than
hierarchical extension.

### A3. Half-res derived try — `nested_reduction_halfres_derived_try`

Attempt to unify half-resolution with the rest via derived ranges.
Showed that the half-res case has different constraints
(lane-validity mod 2 — i.e. consumer reads must land cleanly on lane 0
or lane 1, not crossing them) that didn't fit the simple
"scaled-down range tree" model. Half-res stayed specialized but the
range tree machinery used to express it became more general.

### A4. Semantic stack — `nested_reduction_semantic_stack`, `_v2`, `_v3`

Tried structuring the codegen as an explicit stack of stages with
explicit transition rules. The motivation: the imperative orchestrator
in `codegen_nested_reduction` is hard to reason about; a declarative
stack would make the stages and their dependencies explicit.

More verbose than the imperative orchestrator that won out. Abandoned
for being heavy without clear payoff.

### A5. Unify probes — `nested_reduction_unify_v2`, `unify_derived`, `unify_probe`

Tried to collapse `_ReducedOutputSpace` and `_HalfResolutionSpace`
(earlier separate dataclasses) into one type. This eventually
succeeded — the current `_DerivedIterationFamily` is the descendant.
But intermediate attempts mixed in other refactors and got abandoned.

### A6. Restack attempts — `nested_reduction_restack`, `restack_work`, `restack_commitwise`

Attempts to commit-split the work for review. Each tried different
splits (by file, by feature, by phase). The current `core_candidate`'s
2-commit split (plumbing vs feature) is the descendant of these
experiments.

### A7. Merged attempt — `nested_reduction_merged_attempt`

A consolidation attempt before `single_commit_wip` happened. Showed
that consolidation was achievable but was itself superseded.

### A8. Path 3 prototype — `nested_reduction_path3`

The first scheduler-ownership probe. Moved only half-resolution
discovery to the scheduler. Built on a stale baseline; per
`agent_space/explore_path_b.md`, has regressions vs current main and
~14h of cleanup is needed to reach baseline parity. Its design
insight (scheduler-owned half-res discovery) lives on in B3 below
(Path B+).

### What we learned from the iterations

Three durable insights anchor the current shape:

**1. The consumer side wants one abstraction.** Reduced-output,
full-resolution, half-resolution all execute pointwise bodies in
remapped iteration extents. The differences are *which extent* and
*how values from upstream stages are reshaped to fit*. One
`_DerivedIterationFamily` parameterized by these is cleaner than
three handlers.

**2. The reduction side is harder to fit a clean abstraction.** The
grouped reduction is genuinely different from a standard reduction
(reshape-then-reduce, not just reduce). Various attempts to express it
through standard codegen primitives failed; the current implementation
specializes it via `_GroupedReductionOpsHandler`.

**3. The orchestration is what makes the feature feel heavy.**
`codegen_nested_reduction` is ~200 lines of explicit multi-stage
emit. It's the function that knows "node1 first, then group reduction
body, then reduced-output epilogue, then full-resolution, then
half-resolution." Each stage has its own handler swap and family
activation. This orchestrator is the load-bearing thing — and it's the
hardest thing to dissolve into standard codegen.

---

## 10. Current Landing Candidates

These are live worktrees with working tests. Each represents a
different answer to "what should we ship?"

### B1. Baseline / `single_commit_wip` (= `nested_reduction_with_nvfp4_saved`)

**Codegen-owned, full feature.**

The all-in version. Everything described in Sections 5–8 lives in
codegen.

- `NestedReduction.can_fuse` recognizes the pattern in scheduler.
- `FusedNestedReductions` is just a fused-node wrapper with
  `small_dim_in_r` axis classification.
- `codegen_nested_reduction` re-derives layout, axis classification,
  internal-buffer decisions, half-res discovery (twice — early/late).
- `_GroupedReductionOpsHandler` does the reduction's reshape+reduce.
- `_PointwiseRemapHandler` runs consumer epilogues with family activation.
- `_DerivedIterationFamily` is the consumer-side abstraction.

**Tests**: 80/80. **LOC**: ~+2793 vs main, ~+1469 in `simd.py`.

**Pros**: working, full feature, well-tested. Clean
`_DerivedIterationFamily` abstraction. NVFP4 perf result included.

**Cons**: big PR, codegen carries layer-1/layer-2 work, early/late
dual pass is a code smell (two BFS passes for half-res discovery
guarded by a runtime `RuntimeError` if they disagree), `simd.py`
footprint hard to review.

**Use when**: want to ship the full feature with no architectural
argument needed (it's just bigger).

### B2. `core_candidate` / `core_landable`

**Codegen-owned, trimmed (no half-res / NVFP4).**

Same architecture as B1 minus the half-resolution path. The half-res /
NVFP4 codegen is removed; the saved branch
`nested_reduction_with_nvfp4_saved` preserves it for follow-up.

Two reviewable commits:
- Commit 1: backend plumbing (triton.py, runtime, config, metrics).
- Commit 2: scheduler + simd.py + tests.

**Tests**: 76/76 (4 NVFP4 tests removed). **LOC**: ~+2118 vs main,
~+1008 in `simd.py`.

**Pros**: smaller PR, cleaner commit split, NVFP4 follows separately.
Reviewer doesn't have to swallow the half-res complexity in the first
PR.

**Cons**: loses the marquee NVFP4 perf result on this PR; same
architectural shape as B1.

**Use when**: "ship soon" dominates and NVFP4 can wait.

### B3. `path_b_plus_candidate` (in `ir_explore` worktree)

**Scheduler-owned grouped-reduction plan, full feature.**

The architectural improvement that's actually achievable on top of B1.
Adds three dataclasses to `scheduler.py`:

```python
@dataclass(frozen=True)
class BlockLocalReductionSpec:
    reduction_node: SchedulerNode
    output_name: str
    group_size: sympy.Expr
    small_dim_in_r: bool
    requires_persistent_reduction: bool

@dataclass(frozen=True)
class NestedReductionPlan:
    group_reduction: BlockLocalReductionSpec
    reduced_output_epilogues: tuple[SchedulerNode, ...]
    full_resolution_epilogues: tuple[SchedulerNode, ...]
    half_resolution_consumer_names: tuple[str, ...] = ()
    internal_node1_outputs: tuple[str, ...] = ()
    late_internal_outputs: tuple[str, ...] = ()
```

`FusedNestedReductions._build_plan()` computes the plan at fusion
time. `codegen_nested_reduction` reads from `node.plan` instead of
re-deriving.

**What dissolves from `simd.py` vs B1**:
- Pattern recognition (which buffers are internal, fullres vs reduced)
- Axis classification re-derivation
- Shared-reads detection re-derivation
- Persistent-reduction prediction
- Internal-buffer ownership checks
- Early/late half-res discovery passes (and the `RuntimeError`
  agreement check)

**What stays in codegen**: range tree manipulation, family activation,
body emit. The reduction handler stays
(`_GroupedReductionOpsHandler`). The consumer machinery stays
(`_PointwiseRemapHandler`, `_DerivedIterationFamily`).

**Tests**: 80/80 (verified). **LOC**: ~+2683 vs main, ~+1234 in
`simd.py` (`simd.py` is **−235 vs Baseline** for the same scope).

**Pros**: real architectural improvement, full feature kept, clean
layering between scheduler semantics and codegen execution, no
early/late dual pass. Sets up future Path A (Inductor IR primitive)
without committing to it.

**Cons**: bigger PR than B2 (+655 LOC), introduces new scheduler
abstractions that reviewers must accept, two commits split by *time*
(squashed feature + refactor) rather than by *scope* (plumbing vs
feature).

**Use when**: want to ship the full feature with the architectural
improvement, willing to argue for the new scheduler abstractions in
review.

### B4. Trimmed B3 (would-be `path_b_plus_core_candidate`)

**Scheduler-owned plan, no half-res.**

Doesn't exist yet. Would be: take B2 (core_candidate scope) and apply
the B3 refactor (scheduler-owned plan) on top. Estimated ~1 hour to
produce; would have ~+2028 LOC total but `simd.py` around ~+773
(−235 vs B2).

Could be 3 commits: plumbing / feature / refactor. Each
independently reviewable.

**Pros**: smallest PR with architectural improvement, half-res
follows on top, clean commit split.

**Cons**: doesn't exist yet; needs to be built. NVFP4 follow-up has
to land on top of the refactored shape, slightly more work than
landing NVFP4 on B2.

**Use when**: want both the smaller scope of B2 and the architectural
improvement of B3. Probably the best of both worlds *if* the 1 hour
of work to produce it is worth it.

---

## 11. Future Architectural Directions

These are coherent design points that are not built but have been
worked through in detail. They form a roadmap for follow-up
architecture work.

### C1. Minimal Path A — IR primitive for reduction only

Add `BlockLocalReduction(Reduction)` IR class. Scheduler rewrites
node2's reduction IR to it at fusion time. `TritonKernel.reduction()`
dispatches on IR node type to do reshape-then-reduce.

```python
@ir_dataclass
class BlockLocalReduction(Reduction):
    group_size: Expr
    small_dim_in_r: bool

class TritonKernel:
    def reduction(self, dtype, src_dtype, reduction_type, value):
        if isinstance(self.current_node.data, BlockLocalReduction):
            return self._reduction_block_local(...)  # reshape-then-reduce
        return self._reduction_standard(...)
```

**Dissolves**: `_GroupedReductionOpsHandler` (~98 LOC).
**Adds**: IR class (~80), scheduler rewrite (~30), dispatch + lifted
helper (~80), kernel load-capture context (~20).
**Net**: roughly +100 LOC.

**Architectural shape**: mostly Path B+ plus a typed reduction
primitive. Doesn't change the orchestrator, doesn't change
consumer-side code.

**Why it doesn't pay off in isolation**: the complexity isn't on the
reduction side. Moves boxes without unlocking anything. The reduction
handler is the *smallest* piece of complexity; lifting only that
piece doesn't help much.

**Worth it only if**: a follow-up Path A (C2) is planned, in which
case this is the first installment.

### C2. Full Path A — no `codegen_nested_reduction` entry

Standard codegen iterates the fused group. `BlockLocalReduction` IR
primitive handled by `TritonKernel.reduction()`. Iteration families
activated automatically when standard codegen sees a tagged consumer.

Conceptually, `codegen_node_schedule_with_kernel` handles everything:

```python
for sn in fused_nodes:
    if sn.iteration_family is not None:
        family = kernel.get_or_build_family(sn.iteration_family)
        with family.activate(kernel):
            sn.codegen()
    else:
        sn.codegen()
```

No `codegen_nested_reduction`. No `_GroupedReductionOpsHandler`.
No `_PointwiseRemapHandler`.

**Dissolves**: the orchestrator (~200 LOC), the reduction handler
(~98), the consumer handler (~50), epilogue codegen functions
(~200), half-res discovery and codegen (~150), plan dataclasses
(~50).

**Adds**: IR primitive, scheduler IR rewrite, family construction
hooks, possibly an `iteration_family` field on `SchedulerNode`,
backend dispatch.

**Hard parts**:
1. Cross-extent fused group membership: half-res consumers run at a
   different iteration extent. Today they're discovered separately
   and emitted in their own pass; in C2 they have to be members of
   the fused group with their family tag. Standard scheduler fusion
   doesn't naturally accept this.
2. Index decomposition through standard codegen: today's
   `_decompose_flat_index` is called explicitly; in C2 it has to
   happen transparently when family-activated.
3. Half-resolution capture lifecycle: today's `_stage_load_values`
   captures register values during the reduction body for downstream
   half-res use. In C2 this needs another home.
4. Dynamic shapes: scheduler's existing `NestedReduction.can_fuse`
   rejects non-static cases. Path A would either inherit this
   rejection or have to handle symbolic group_size.

**Estimated**: 2–4 weeks (Option B — special-case in
`store_reduction`) or 4–8 weeks (Option A — real backend extension to
reduce over interior range tree). Per `explore_path_a.md`.

**Architectural shape**: what the IR-uplift design doc calls Path A.
The "non-local optimum" for nested reduction.

**Worth it only if**: Inductor team has a multi-week budget for this,
OR a second similar fusion pattern (cat/scatter/MLA) is on the
roadmap to amortize the cost.

### C6. Range-tree-driven grouped reduction

Express the grouped reduction as a standard `Reduction` whose body's
indexing decomposes `R = R_outer × G`, with `G` introduced as a
derived range tree during the grouped stage. Standard reduction
codegen reduces over the trailing (G) tree. The "reshape" lives in
the index decomposition, not as an explicit op.

**What dissolves**: `_GroupedReductionOpsHandler.reduction()`, the
explicit `emit_reshape` call, `_GroupReductionLayout.reshape_shape`
(the structure becomes range tree config).

**What's required**: rigorous handling of `r = r_outer*G + g`
decomposition in indexing, plus verification that standard reduction
codegen works with a derived trailing range tree.

**Worth pursuing if**: the codegen-side mechanics (range tree swap,
indexing simplification) work out cleanly in a prototype. Could be a
1–2 day prototype to find out.

**Limitation**: only handles the factorized-split-as-reduction case.
Doesn't unify with NVFP4 (half-res) or affine subregions.

### C7. Unified iteration family with explicit value sources

The most ambitious architectural direction. Generalize
`_DerivedIterationFamily` to be the abstraction for *every* stage,
including the reduction itself.

```python
@dataclass
class IterationFamily:
    range_trees: tuple[IterationRangesRoot, ...]
    mode: Literal["pointwise", "reduction"]
    value_sources: dict[str, ValueSource]
    mask: Optional[Expr] = None    # for affine subregions

# ValueSource is a sum type:
#   Direct(cse_var)                 — value already at this extent
#   Broadcast(cse_var, target_shape)— smaller-extent CSE, broadcast
#   Split(parent_cse, lane)          — bigger-extent CSE, split, pick
#   MemoryLoad(name, index_fn)       — load with custom index expression
#   MaskedMemoryLoad(name, idx, mask)— masked load (affine subregions)
```

Each stage activates a family:

```python
families = [
    outer_reduction_family,         # mode=reduction, range_trees=[X, R]
    grouped_reduction_family,       # mode=reduction, range_trees=[X, R/G, G]
    reduced_output_family,          # mode=pointwise, range_trees=[X, R/G]
    full_resolution_family,         # mode=pointwise, range_trees=[X, R], broadcast
    half_resolution_family,         # mode=pointwise, range_trees=[X, R/2], split
    affine_subregion_family,        # mode=pointwise, mask, future
]

for stage_family, stage_body in zip(families, bodies):
    with stage_family.activate(kernel):
        stage_body(...)
```

**Why this is structurally important**: all five iteration-extent
cases (grouped reduction, reduced-output, full-resolution,
half-resolution, future affine subregions) become instances of the
same abstraction. The family expresses *iteration extent +
per-source value transformations*. Path A's `BlockLocalReduction`
only addresses the reduction; C7 addresses the whole picture.

**Forward compatibility**: NVFP4 already fits (Split variant). Affine
subregions become a new ValueSource variant + mask field. The family
is forward-compatible with future fusion patterns.

**Cost**: 1–2 weeks. Bigger than Path B+, smaller than full Path A.
Higher reward than either: unifies all the iteration-extent cases
under one abstraction.

### C-roadmap (paths from `nested_reduction_non_local_optimum.md`)

The non-local-optimum design doc proposes a 5-step migration:

1. **Path 1**: Land the current local optimum. ← what B-series is doing.
2. **Path 2**: Unify reduced-output and half-resolution into one
   generic derived-family object. ← done (current
   `_DerivedIterationFamily`).
3. **Path 3**: Move legality and family construction onto
   `FusedNestedReductions`. ← Path B+ (B3) partially does this.
4. **Path 4**: Teach the generic family model affine masked
   subregions (`small_col = big_col - 512`, valid only when
   `512 ≤ big_col < 576`).
5. **Path 5**: Reuse the family abstraction for vertical pointwise
   fusion beyond nested reduction.

Steps 4 and 5 are blocked on a second customer to amortize the
abstraction cost. They become viable when cat/scatter/MLA-style
fusion patterns materialize.

### Why future paths are blocked

Path A / C2 / C7 require either:
- **(a)** A second customer beyond nested reduction to amortize the
  abstraction cost, or
- **(b)** A separate Inductor architecture project funded on its own
  merits.

The current branch can't carry that work. These are the right moves
*eventually*, but only when the second customer is concrete enough
to inform the abstraction. With only nested reduction, the
abstraction would be overfit to one pattern's needs.

---

## 12. Comparison Matrix

| | B1 Baseline | B2 core_candidate | B3 path_b_plus | B4 trimmed B3 | C1 minimal Path A | C2 full Path A | C7 unified family |
|---|---|---|---|---|---|---|---|
| **Architecture** | Codegen-owned | Codegen-owned | Scheduler-owned plan | Scheduler-owned plan | + IR primitive | IR-owned | IR + family-owned |
| **Scope** | Full feature | No NVFP4 | Full feature | No NVFP4 | Full feature | Full feature | Full feature |
| **Tests** | 80/80 ✅ | 76/76 ✅ | 80/80 ✅ | n/a | n/a | n/a | n/a |
| **Total LOC vs main** | +2793 | +2118 | +2683 | ~+2028 | ~+2783 | varies | varies |
| `simd.py` LOC | +1469 | +1008 | +1234 | ~+773 | ~+1140 | ~+800 | ~+700 |
| `scheduler.py` LOC | +350 | +350 | +565 | ~+550 | ~+560 | ~+450 | ~+450 |
| **Built / verified** | ✅ | ✅ | ✅ | ❌ (~1h) | ❌ | ❌ | ❌ |
| **Time to land** | 3–5d + review | 2–4d + review | 1–2d + review | ~2d + review | n/a (multi-PR) | weeks–months | weeks |
| **Reviewability** | Hard | Medium | Medium | Medium-low | Hardest | Hardest | Hard |
| **Early/late dual pass** | Present | Present | Removed ✅ | Removed ✅ | Removed | Removed | Removed |
| `_GroupedReductionOpsHandler` | Present | Present | Present | Present | Removed | Removed | Removed |
| `_DerivedIterationFamily` | Present | Present | Present | Present | Present | Present | Generalized |
| **Future generality** (cat/scatter/MLA) | Low | Low | Medium | Medium-low | Medium | High | Highest |
| **Risk** | None (done) | Low | Low | Low | Medium | High | Medium-high |

---

## 13. Decision Tree and Recommendations

The right answer depends on what dominates:

| Constraint dominates → | Pick |
|---|---|
| Ship soon, smaller scope OK | **B2 core_candidate** |
| Ship soon, full feature | **B1 Baseline** |
| Ship architectural improvement, full feature | **B3 path_b_plus** |
| Ship architectural improvement, smaller scope, willing to spend ~1h | **B4 trimmed B3** |
| Right architecture, willing to spend weeks | **C2 Full Path A** or **C7** |
| Long-term Inductor evolution | **C7** then **path-4 + path-5** |

### 13.1 Primary recommendation

**Land `core_candidate` (B2).**

- Smallest review surface that ships a real piece of the feature
- 2 reviewable commits split cleanly (plumbing vs feature)
- 76/76 tests passing
- No architectural argument needed in review
- NVFP4 / half-res preserved in `nested_reduction_with_nvfp4_saved` for follow-up PR
- Path B+ scheduler refactor preserved in `ir_explore` for follow-up PR

### 13.2 Alternative recommendation

If you want the architectural improvement in the same PR:
**Land `path_b_plus_candidate` (B3)**.

- Real architectural win (scheduler owns the plan)
- 80/80 tests, full feature
- Bigger PR but the architecture argument is bounded (it's
  scheduler-side dataclasses, not new abstractions across layers)

### 13.3 Long-term roadmap

After landing B2 (or B3), the future-architecture work is **C7**: a
unified iteration-family abstraction. This is the abstraction that
generalizes to NVFP4, affine subregions, and future cat/scatter/MLA
patterns. It's the architecturally correct destination but requires
either a second customer or its own funded Inductor project.

**Don't pursue C7 in isolation**. With only nested reduction as a
customer, the abstraction overfits to one pattern. Wait for a second
pattern to inform the design.

### 13.4 What NOT to pursue

- **Path B / path3** is obsolete. Stale baseline, regressions vs
  current main, ~14h cleanup just to reach baseline parity. Its good
  idea (scheduler-owned half-res discovery) is independently absorbed
  by Path B+. Archive.

- **Minimal Path A (C1) in isolation**. Mostly cosmetic — moves boxes
  without unlocking anything. Only worth doing if full Path A
  follows.

- **Range-tree-driven approach (C6) in isolation**. Only handles the
  factorized-reduction case; doesn't unify with NVFP4 or affine
  subregions. Useful as a piece of C7, not standalone.

---

## 14. What's Preserved

Listed once because every option keeps these:

- **`_DerivedIterationFamily`**: every option keeps it. It's the right
  shape for "run a body in a remapped iteration extent." C7 generalizes
  it; the others use it as-is.
- **`DerivedIterationRangesRoot`**: same. Range-tree subclass that
  backs derived families.
- **`NestedReduction.can_fuse` recognition logic**: lives in scheduler
  in every option. FX-level recognition has been ruled out as not
  viable without `MemoryDep` access.
- **Half-resolution discovery BFS**: lives in scheduler (path-3 onward).
- **Tests, form-checks, the `triton.nested_reduction` config flag**.
- **Saved snapshots**: B1 preserved as
  `nested_reduction_with_nvfp4_saved` and as a 3461-line patch
  snapshot at
  `agent_space/snapshots/nested_reduction_with_nvfp4_03baa04.patch`.

---

## 15. Glossary

- **Codegen-owned**: pattern recognition, layout, family construction,
  consumer dispatch all live in `simd.py` / `triton.py`.
- **Scheduler-owned**: pattern recognition + family/plan decisions
  live on the `FusedNestedReductions` scheduler node; codegen reads
  the plan.
- **IR-owned**: the operation itself (`BlockLocalReduction`) is an
  Inductor IR primitive; standard reduction codegen handles it.
- **Iteration family**: `_DerivedIterationFamily` in codegen — runs a
  pointwise body in a remapped extent (reduced-output,
  full-resolution, half-resolution).
- **Special handler**: a `WrapperHandler` subclass
  (`_GroupedReductionOpsHandler`, `_PointwiseRemapHandler`) that
  intercepts load/store/reduction during a stage's body emit.
- **Path B+**: scheduler-owned grouped-reduction plan. The scheduler's
  `FusedNestedReductions` carries a `NestedReductionPlan` with all
  pattern facts; codegen reads it instead of re-deriving.
- **Path A**: lift the reduction itself to an Inductor IR primitive
  (`BlockLocalReduction`). Standard reduction codegen learns to
  handle it.
- **`small_dim_in_r`**: `group_size` lives in the outer reduction's
  reduction axis. Common case (layernorm + per-group amax).
- **`small_dim_in_x`**: `group_size` lives in the outer reduction's
  non-reduction axis. Cross-axis case.
- **NVFP4**: a 4-bit floating-point format where two FP4 values are
  packed into one byte. Drives the half-resolution path because
  packing requires even/odd lane consumers.
- **Producer-consumer**: a fusion shape where node2 reads node1's
  *output* (not the shared input). Different from the more common
  shared-input case.

---

## 16. Reference

### 16.1 Where each option lives

- **B1 Baseline**: `pytorch_nested_reduction_single_commit_wip` (HEAD `03baa04`)
- **B2 core_candidate**: `pytorch_nested_reduction_core_candidate` /
  `pytorch_nested_reduction_core_landable`
- **B3 path_b_plus**: `pytorch_nested_reduction_ir_explore` (branch
  `nested_reduction_path_b_plus_candidate`, HEAD `f257057`)
- **B4 trimmed B3**: would build off B2 + B3's scheduler refactor.
  Doesn't exist yet.
- **C-series**: design only. See `nested_reduction_ir_uplift.md` and
  `nested_reduction_non_local_optimum.md`.

### 16.2 Saved snapshots

Defensive backups in case any worktree state is lost:

- `nested_reduction_with_nvfp4_saved` branch — full B1 state
- `agent_space/snapshots/nested_reduction_with_nvfp4_03baa04.patch` —
  3461-line patch of the full B1 state

### 16.3 File:line index of load-bearing code

In `single_commit_wip` baseline:

| Symbol | Location |
|---|---|
| `DerivedIterationRangesRoot` | `simd.py:381` |
| `_DerivedIterationFamily` | `simd.py:1450` |
| `_GroupReductionLayout` | `simd.py:1553` |
| `_GroupedReductionOpsHandler` | `simd.py:1919` |
| `_PointwiseRemapHandler` | `simd.py:2017` |
| `codegen_nested_reduction` | `simd.py:2625` |
| `NestedReduction.can_fuse` | `scheduler.py:445` |
| `FusedNestedReductions` | `scheduler.py:2535` |
| `TritonKernel.reduction` | `triton.py:4211` |
| `KernelHandler.store_reduction` | `common.py:2884` |

### 16.4 Source documents this synthesizes

This document combines content from:

- `nested_reduction_design_overview.md` — iteration-space model
- `nested_reduction_full_design.md` — long-form design notes
- `nested_reduction_fullres_epilogue.md` — full-res epilogue specifics
- `nested_reduction_non_local_optimum.md` — paths 1–5 of consumer-side
  abstraction generalization
- `nested_reduction_ir_uplift.md` — paths A–D of IR uplift
- `nested_reduction_ir_explore_comparison.md` — Path B+ documentation
- `nested_reduction_design_choices.md` — earlier exploration
- `nested_reduction_design_space.md` — comprehensive design space
- `path_comparison.md` — focused B-series comparison
- `MORNING_READING_GUIDE.md` — overnight orientation
- `explore_path_a.md` — agent's Path A feasibility probe
- `explore_path_b.md` — agent's path3 prototype assessment

The originals remain in `agent_space/` for deeper reading on any
specific topic; this document is the canonical entry point.

### 16.5 Key tests

`test/inductor/test_nested_reduction.py`:

- 25 test methods (parametrized to ~80 individual test runs)
- Covers numerics (vs eager, vs unfused-compiled)
- Covers kernel form (FileCheck on emitted Triton)
- Three pattern-form checkers: nvfp4, amax, fullres
- Tests for both `small_dim_in_r` and `small_dim_in_x`
- Dynamic-shape variants
- B=1 edge cases
- Producer-consumer variants
- Internal-buffer-removal verification

### 16.6 Config flag

`torch._inductor.config.triton.nested_reduction` — gates the
nested-reduction fusion path. Default `True` in the working branches.

---

*End of master document. Last updated 2026-04-29.*
