# MLA RoPE Fusion Findings

## Current experimental patch

The current scheduler-side patch handles the three isolated
`mla_fusion.py` blockers with ordinary fused pointwise/scatter kernels:

1. Q/K RoPE sibling fusion:
   - lifts the smaller `[T, D]` / `[T, 1, D]` pointwise body into the larger
     `[T, H, D]` loop domain by inserting a broadcast dimension;
   - keeps the original smaller body/ranges as a subdomain;
   - codegen emits the smaller body under a program-level guard whenever that
     subdomain body is scheduled:

     ```python
     if tl.program_id(0) < tl.cdiv(T * D, XBLOCK):
         ...
     ```

   - this avoids doing K work for every Q head. The guarded section is CSE
     scoped and, if codegen continues afterward, the main `xnumel/xindex/xmask`
     headers are re-emitted. Without both pieces, a temporary or range variable
     defined inside the K guard can be reused by later Q code.
   - it gives up some cross-domain CSE of RoPE table loads, but a quick isolated
     CUDA-event timing was directionally faster than the earlier store-only
     masked version.

2. RoPE producer into cache scatter:
   - reindexes a compact producer domain like `[T, D]` into the tail slice of a
     scatter consumer domain like `[T, KV + D]`;
   - masks the whole producer body to the valid tail slice because indices
     outside the slice would be invalid;
   - normalizes vertical dependency matching by substituting write loop vars
     into read loop vars and expanding `Identity(...)` wrappers before comparing
     indices.

3. Concat slice writer fusion:
   - detects two pointwise `NonOwningLayout` writers into the same
     `ConcatKernel` storage;
   - reindexes both slice writers into the full concat iteration space and
     masks each body to its own slice.

Tests added in `test/inductor/test_loop_ordering.py` cover all three cases:

- `test_horizontal_fusion_with_inserted_broadcast_dim`
- `test_horizontal_fusion_with_inserted_broadcast_dim_not_last`
- `test_horizontal_fusion_with_concat_slices`
- `test_vertical_fusion_with_scatter_tail_slice`

## Measurements

With `/home/eellison/local/pytorch/mla_fusion.py`:

```text
Isolated blockers:
  blocker 1: 1 kernel
  blocker 2: 1 kernel
  blocker 3: 1 kernel

Combined patterns:
  Full vLLM: 2 kernels, ideal script target is 1
  Full TRT-LLM generation: 2 kernels, ideal script target is 1
  RMSNorm + RoPE: 3 kernels, ideal script target is 2

Total: 10 kernels, ideal script target is 7
```

Original total was 17 kernels. Before the vertical scatter fix, it was 12.

## Remaining combined-pattern issue

The remaining gap is not another simple pairwise mismatch. In the full vLLM
pattern, the scheduler can fuse:

- Q RoPE with K RoPE, sharing the RoPE table loads; or
- K RoPE with the cache scatter, keeping K in registers for the scatter.

To get one kernel, K needs to participate in both relationships while Q and the
scatter use incompatible loop domains:

```text
Q RoPE output:       [T, H, 64]
cache scatter body:  [T, KV + 64]
```

The current `FusedSchedulerNode` model picks one iteration domain for the fused
kernel. Forcing both into a single rectangular domain such as `[T, H, KV + 64]`
would be a large amount of masked work and is not the same as the hand-written
kernel structure.

Combo kernels are also not a full answer for this case. They can package
separate kernels into one launch, but they do not naturally share the RoPE table
loads between Q and K or keep K in registers across the K/ scatter boundary.

The program-guard subdomain prototype solves the worst per-head redundant K
work for Q/K sibling fusion. It does not solve the full combined vLLM target,
because K still has to participate in both the Q/K RoPE relationship and the
K/scatter relationship while those consumers want different natural domains.

## Likely next design

The clean next step is a real multi-domain fusion representation:

- one fused launch can contain multiple loop domains;
- a producer value can be materialized once at the domain where it is naturally
  computed and consumed by a compatible subdomain;
- shared loads remain visible to CSE when domains overlap;
- masks are represented as separate load-valid and store-valid predicates, not
  as a single "mask the whole body" bit.

That is bigger than the pairwise scheduler reindexing patch here, but it is the
shape needed to close the combined MLA one-kernel target without hiding extra
work behind masks.
