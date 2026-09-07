# Nested Reduction: Full-Resolution Epilogue Fusion

## Motivation

The RMSNorm → FP8 group quantize pattern is critical for MLA (Multi-head Latent
Attention) inference in models like DeepSeek-V2. The decomposed pattern is:

```python
x = F.rms_norm(x, (D,), weight)           # reduction over D
xg = x.view(B, D // G, G)
amax = xg.abs().amax(dim=-1)              # reduction over groups of G
scale = (amax / 448.0).clamp(min=1e-12)
x_fp8 = (xg / scale.unsqueeze(-1)).to(fp8)  # full-res: reads x AND scale
```

The nested reduction feature already fuses the RMSNorm (pass 1, big reduction
over D) with the amax (pass 2, small reduction over groups of G) into a single
kernel. The codegen infrastructure for a "full-resolution epilogue" — where a
downstream pointwise reads both the pass 2 output (scale) and the pass 1 output
(normalized x, still in registers) — also existed.

**However, the full-res epilogue never actually fused.** The `x / scale → fp8`
quantization step always ended up in a separate kernel, re-reading x from global
memory.

## Root Causes (8 bugs)

### Bug 1: `init_group_node` ancestor pollution (scheduler.py)

`FusedSchedulerNode.ancestors` was computed as the union of sub-node ancestors
but never subtracted the fused node's own internal operation names:

```python
# Before
group_snode.ancestors = OrderedSet.union(
    *[x.ancestors for x in snodes ...]
)

# After
group_snode.ancestors = OrderedSet.union(
    *[x.ancestors for x in snodes ...]
) - group_snode.get_operation_names()
```

This polluted ancestors for ALL fused node types (`FusedSchedulerNode`,
`FusedNestedReductions`, `FusedMixOrderReductions`, `GroupedSchedulerNode`),
causing false "intermediate nodes between node1 & node2" rejections in
`can_fuse` reorder checks and potentially false cycle detection in
`will_fusion_create_cycle`.

### Bug 2: `can_fuse_with` used `scheduler.can_fuse` (scheduler.py)

`FusedNestedReductions.can_fuse_with()` delegated to `scheduler.can_fuse()` to
check if a downstream pointwise could fuse in. This failed for two reasons:

1. The ancestors pollution from Bug 1 (now fixed but not sufficient alone)
2. The memory-dep matching check (`remaining_deps & node1_buf_names`) rejects
   the fusion because the epilogue reads the reduction output with different
   indexing than it was written (e.g., broadcasted scale)

Fix: use `backend.can_fuse_vertical()` directly, which only checks
numel/rnumel compatibility — sufficient since the outer fusion loop already
handles reordering.

```python
# Before
if not self.scheduler.can_fuse(self.node2, other, ...):

# After
device = self.node2.get_device()
backend = self.scheduler.get_backend(device)
if not backend.can_fuse_vertical(self.node2, other):
```

### Bug 3: `inline_reduction_buffers` gated by `is_producer_consumer` (simd.py)

The logic to keep node1's intermediate outputs in registers (avoiding dead
stores to global memory) was gated by `if is_producer_consumer:`. In the
shared-input pattern (where both nodes read the original input), this gate
was False, so `buf0` (sum-of-squares from RMSNorm) was always stored to
global memory even when all its consumers were internal.

Fix: remove the gate. The per-buffer user check (`all users in fused_names`)
is sufficient regardless of pattern type.

### Bug 4: `store_reduction` didn't check `inline_reduction_buffers` (triton.py)

The `store()` method had an intercept for `inline_reduction_buffers` (keeping
values in registers instead of writing to global memory), but
`store_reduction()` did not. Since the RMSNorm's reduction accumulator output
is written via `store_reduction`, the intercept never triggered.

```python
def store_reduction(self, name, index, value):
    if name in self.inline_reduction_buffers:
        self.inline_reduction_buffers[name] = value
        return
    # ... original code
```

### Bug 5: `_RemappedOpsHandler.store` skipped `simplify_indexing` (simd.py)

The full-resolution epilogue handler emitted store indices by calling
`k.index_to_str(index)` directly, bypassing `simplify_indexing()`. This
produced complex index expressions like
`128*(((r0_1 % 4096)) // 128) + 4096*x0 + ((r0_1 % 4096) % 128)` instead of
the equivalent `r0_1 + 4096*x0`. The existing `simplify_with_ranges` logic
already knows how to simplify these (it uses var_ranges to prove
`ModularIndexing(r, 1, 4096) → r` when `r < 4096`), but was never called.

```python
# Before
if self._subs_map is not None:
    idx_str = k.index_to_str(index.subs(self._subs_map))
else:
    idx_str = k.index_to_str(index)

# After
if self._subs_map is not None:
    index = index.subs(self._subs_map)
index = k.simplify_indexing(index)
idx_str = k.index_to_str(index)
```

### Bug 6: Missing `small_dim_in_x` guard in `can_fuse_with` (scheduler.py)

`can_fuse_with` admitted full-resolution epilogues for both `small_dim_in_r`
and `small_dim_in_x` patterns, but the codegen at `_codegen_pass2_epilogue`
only implements full-res for `small_dim_in_r`. For `small_dim_in_x`, the
epilogue stores are silently skipped, producing wrong results (max diff ~25).

Fix: detect full-res epilogues (other's numel > node2's numel) and verify
the pattern is `small_dim_in_r` by checking node2's reduction subnode's first
iteration range equals node1's numel.

### Bug 7: Inlined buffers not added to `removed_buffers` (simd.py)

When `inline_reduction_buffers` suppresses the store of `buf0`, the buffer
is never written by the kernel. But the wrapper still allocated it (and
immediately deleted it) because `buf0` was not added to
`V.graph.removed_buffers`. Fix: add inlined buffer names to
`removed_buffers` alongside the `inline_reduction_buffers` setup.

### Bug 8: Broadcast via multiply-by-ones (simd.py)

The full-resolution epilogue broadcast of pass 2 values (e.g., scale) to
full resolution used `tl.reshape(val, [X, G, 1]) * tl.full([1, 1, gs], 1, f32)`.
This wasteful multiply-by-ones is replaced with `tl.broadcast_to`.

## Generated Kernel (after fixes)

For `RMSNorm([128, 4096]) → FP8 group quant (groups of 128)`:

**Before:** 2 kernels
- Kernel 1: RMSNorm + amax + scale (nested reduction, stores buf0 to global memory)
- Kernel 2: re-reads x + buf0 from global memory, recomputes normalize, divides by scale, casts to fp8

**After:** 1 kernel
```
in_ptr0 (x, bf16), in_ptr1 (weight, bf16)
  → out_ptr0 (scale, bf16[128,32]), out_ptr1 (fp8[128,4096])

Pass 1:  load x → x² → sum (reduction over 4096) → rsqrt → normalize
Pass 2:  abs → reshape [XBLOCK, 32, 128] → max2 (reduce axis 2) → scale
Epilogue: broadcast scale → x_norm / scale → cast fp8 → store
```

No dead stores, clean indexing (`r0_1 + 4096*x0`).

## Correctness

- Fused output is **bit-identical** to unfused compiled output (verified across
  B=1..1024, D=4096..8192)
- Both match eager at ~96% for fp8 values (expected: different intermediate
  precision from keeping values in f32 registers vs round-tripping through bf16
  in global memory)
- Scales are identical between fused and unfused

## Performance

With CUDA graphs (removing launch overhead): identical compute time (~4μs).
The fusion benefit is kernel launch savings (~7μs per call without graphs).

## Test Coverage

New test: `test_producer_consumer_rmsnorm_fp8_quant` — verifies the full
pattern (RMSNorm → amax → scale → x/scale → fp8) fuses into 1 kernel with
nested reduction, including numeric correctness.

Existing tests (66) all pass. Also verified: `test_loop_ordering` (83),
`test_fused_attention` (118), `test_combo_kernels` (90).

## Audit

Reviewed all related code paths for similar bugs:

- `_RemappedOpsHandler.load` — path 3 (global memory) correctly goes through
  `prepare_indexing` → `simplify_indexing` via `index_to_str`. ✓
- `_Pass2OpsHandler.store_reduction` — correctly uses `k.index_to_str` which
  goes through `prepare_indexing`. ✓
- `inline_reduction_buffers` intercept points — all 3 covered: `load`,
  `store`, `store_reduction`. ✓
- `will_fusion_create_cycle` — uses `node.ancestors` which benefits from the
  Bug 1 fix. ✓

## Structural Refactor: Shared Axis Classification

`NestedReduction.is_small_dim_in_r()` is now the single source of truth for
axis classification.  Both the scheduler (`FusedNestedReductions._is_small_dim_in_r`)
and codegen (`codegen_nested_reduction`) call it.  The old `_detect_group_size_axis`
in simd.py is deleted.  A `RuntimeError` guard in `_codegen_pass2_epilogue` catches
any future disagreement between scheduler and codegen.

## Known Pre-existing Issue

Dynamic shapes + nested reduction crashes with `Unsupported ptr type
triton.language.float64 in tl.load`. This is pre-existing (reproduces on
main with `nested_reduction=True` before these changes). Not addressed here.

## Files Changed

```
scheduler.py  (+8, -8)   # init_group_node ancestors fix + can_fuse_with
simd.py       (+25, -24) # inline_reduction_buffers ungated + removed_buffers +
                          #   simplify_indexing + broadcast_to
triton.py     (+3, -0)   # store_reduction inline check
test_nested_reduction.py (+26, -0)  # new test
```
