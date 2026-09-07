# TorchAO #4743 comparison and Inductor follow-up

Date: 2026-08-18

## Scope

TorchAO #4743 implements a specialized CuTe DSL kernel for:

```text
SwiGLU -> MXFP8 rowwise 1x32 quantization
       -> MXFP8 colwise 32x1 quantization
```

It can emit either direction or both from one pass over the packed
`[gate | up]` input. This is separate from the nested/sub-parent reduction
stack: the rowwise and colwise reductions are independent siblings, not a
producer-consumer reduction pair or a derived-domain packing epilogue.

## Local measurements

Same-session measurements on an NVIDIA B200, PyTorch
`2.15.0a0+git7c78410`, CUDA 12.8, and CuTe DSL 4.6.1. The PR reports DSL
4.7.0, CUDA 13.4, and pinned clocks, so use these local measurements for the
relative comparison rather than comparing either side to the PR's published
absolute times.

### Forward, rowwise only

| Shape | Inductor | CuTe #4743 | Result |
|---|---:|---:|---:|
| 4096x2048 | 16.28 us | 66.60 us | Inductor 4.09x faster |
| 4096x7168 | 39.14 us | 70.26 us | Inductor 1.80x faster |
| 16384x7168 | 133.14 us | 126.47 us | CuTe 1.05x faster |

Inductor already emits one ordinary reduction kernel for this case with
`triton.nested_reduction` either enabled or disabled. The staged-reduction
stack is not involved (`codegen_nested_reduction == 0`).

### Forward, rowwise and colwise

| Shape | Inductor | CuTe #4743 | Result |
|---|---:|---:|---:|
| 4096x2048 | 46.18 us | 67.06 us | Inductor 1.45x faster |
| 4096x7168 | 120.75 us | 71.86 us | CuTe 1.68x faster |
| 16384x7168 | 433.34 us | 174.51 us | CuTe 2.48x faster |

Inductor emits four kernels for this case, again unaffected by
`triton.nested_reduction`. The specialized CuTe kernel loads a 2D tile once,
computes SwiGLU once, performs both orthogonal block reductions, and writes
both quantized layouts and scale layouts.

At 4096x7168, colwise-only takes 87.51 us and emits three kernels:

1. Colwise block maximum and E8M0 scale computation, including SwiGLU.
2. Full-resolution FP8 quantization, recomputing SwiGLU and reading the scale.
3. Blocked scale-layout permutation.

Rowwise-only is one kernel, so the four-kernel both-direction result is exactly
the one rowwise kernel plus these three colwise kernels.

A forced post-fusion prototype proves existing generic reduction codegen can
emit the complete colwise branch as one kernel:

| Shape | Normal, 3 kernels | Forced, 1 kernel | Speedup |
|---|---:|---:|---:|
| 4096x2048 | 79.50 us | 22.69 us | 3.50x |
| 4096x7168 | 90.43 us | 59.88 us | 1.51x |
| 16384x7168 | 347.06 us | 324.59 us | 1.07x |

The outputs were bitwise identical. The reproducer is
`agent_space/prototype_colwise_mxfp8_fusion.py`.

The diminishing large-shape gain is important: scheduler fusion removes launch
and intermediate traffic, but does not fix the generic kernel's colwise memory
layout. Large-shape competitiveness still depends on the transposed-store and
tile-layout experiment described below.

Raw measurements are in `agent_space/torchao_4743_comparison_20260818.json`.
The scratch benchmark is `agent_space/bench_swiglu_mxfp8_inductor.py`.

## Why current fusion misses the case

The two reductions are both block-size 32 and flatten to similar
`(M*K/32, 32)` scheduler groups. Their distinction is in their index maps:

- rowwise reduces contiguous 1x32 K blocks;
- colwise reduces 32x1 M blocks and writes a transposed payload layout.

This does not match the existing nested-reduction topology. It also does not
match current `MixOrderReduction` admission, which expects reversed reduction
groups and only supports sum/product on the secondary reduction path. Merely
enabling `triton.mix_order_reduction` is therefore insufficient.

## Proposed follow-up

Use `FusedMixOrderReductions` as the scheduler identity and legality home, but
add a blockwise 2D mode rather than extending the sub-parent planner.

The planner would need to prove:

1. Both reductions consume the same logical producer.
2. Their normalized indices describe orthogonal fixed-size blocks of the same
   2D domain.
3. The block dimensions and output layouts are compatible with one shared
   tile.
4. The shared pointwise producer can be evaluated once without materializing
   an intermediate.
5. Each reduction epilogue consumes only its corresponding reduced result and
   the shared tile values.

Codegen would then:

1. Load a 2D tile of `gate` and `up` once.
2. Compute the SwiGLU activation once.
3. Reduce 1x32 blocks across K and 32x1 blocks across M.
4. Quantize the tile using each scale.
5. Store row-major and transposed FP8 payloads.
6. Store both E8M0 scale tensors directly in their blocked layouts.

The first target should be the fixed 32x32 MXFP8 case. Generalizing arbitrary
orthogonal reductions before measuring this target would add unnecessary
complexity. A Triton implementation can establish whether a shared-register
tile is sufficient or whether competitive large-shape performance requires a
TMA/shared-memory pipeline similar to the CuTe kernel.

### First milestone: two kernels

Before building shared 2D codegen, reduce the colwise branch from three kernels
to one. The combined operation would then be two kernels: one rowwise and one
colwise. Each kernel would independently evaluate SwiGLU, deliberately trading
duplicated arithmetic and input traffic for a much smaller compiler change.

The colwise reduction kernel already evaluates SwiGLU directly from `gate` and
`up`. The missing fusion is between that reduction, its full-resolution
quantization epilogue, and the reduced scale-layout permutation. The first
approach should be to canonicalize or reindex the colwise consumers into the
reduction's loop order, reusing the consumer-loop reindexing and index-inversion
ideas already exercised by the staged stack. Investigate this before adding a
new mix-order codegen mode.

The forced prototype establishes the required canonicalizations:

1. Reorder the quantization domain from
   `[row_block, row_lane, feature]` to
   `[row_block, feature] x row_lane`, matching the reduction's parent domain.
2. Apply the same mapping to the flattened FP8-cast consumer so its internal
   load/store relation remains exact.
3. Invert the blocked-scale permutation back to the reduced parent domain.
4. Fuse the reduction, both quantization nodes, and scale swizzle atomically.

The normal pairwise scheduler cannot discover this incrementally. The reduced
scale fans out to both the full-parent quantizer and the reduced-domain swizzle,
and their accesses are dense permutations rather than exact `MemoryDep`
matches. The production implementation therefore needs a small planner that
proves both reindexings and returns the complete group; it does not need a new
codegen path.

This milestone answers two questions cheaply:

1. How much of the four-to-one CuTe advantage comes from kernel boundaries and
   materialized scale traffic?
2. How much remains attributable to reading and evaluating the producer twice
   instead of sharing one 2D tile?

If two kernels recover most of the gap, stop there. If the large both-direction
case remains bandwidth-bound from reading the producer twice, proceed to the
shared 2D mix-order kernel.

### Shared-tile endgame

The first prototype must measure the cost of producing the transposed FP8
payload from a shared register tile. This is not necessarily an uncoalesced
store: Triton may use a register-layout conversion and still issue coalesced
stores. If that conversion cannot sustain bandwidth, the design will need a
shared-memory/TMA transpose pipeline similar to CuTe, materially increasing the
implementation size.

Shared activation identity is by construction in this design: one codegen
context evaluates one SwiGLU tile once, and both reductions and quantizers
consume that same value at the same 2D resolution. No cross-domain projection
or name-based forwarding contract is involved.

## Numerical caveat

The scratch generic Inductor spelling is not yet bitwise-equivalent to the
CuTe contract. At 4096x2048 rowwise it differed in 293,012 of 8,388,608 FP8
payload bytes and 836 of 262,144 scale bytes; some payload differences exceeded
one FP8 code. The CuTe output was independently verified bitwise against the
eager TorchAO reference at 128x128.

Resolve the RCEIL/FP8 conversion and activation evaluation differences before
using the latency table as a production-quality performance claim. This is
separate from the missing both-direction fusion, but both issues must be closed
for an apples-to-apples benchmark.

Use the eager TorchAO implementation as the numerical oracle and check CuTe
independently against it. Agreement with CuTe alone is not sufficient. Separate
scale-byte mismatches from payload-only rounding mismatches because one wrong
scale changes all 32 payload codes in its block.

## Initial capability scope

- Fixed 32x32 MXFP8 blocks.
- Static block size; symbolic M/K only when divisibility is proven.
- Forward quantization only.
- Backward/training fusion is a separate follow-up after the forward topology
  and performance are established.

## Recommendation

Do not expand the current sub-parent stack for this workload. Land that stack
for NVFP4/MXFP4/MXFP6, then pursue orthogonal block reductions as a separate
mix-order reduction follow-up. The expected payoff is primarily the large
rowwise-plus-colwise case; rowwise-only already performs well with ordinary
Inductor reduction codegen.
