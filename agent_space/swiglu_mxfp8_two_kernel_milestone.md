# SwiGLU -> MXFP8 rowwise+colwise: the two-kernel milestone

Date: 2026-08-18

Follow-up to `agent_space/torchao_4743_inductor_followup.md`, "First milestone:
two kernels". Workload definition is `agent_space/bench_swiglu_mxfp8_inductor.py`
(copied to `swiglu_workload.py` in the worktree). Prototype diff:
`agent_space/swiglu_2kernel_proto.patch`.

Environment: NVIDIA B200, conda env `pytorch-3.12`, Triton 3.8.0, worktree
`/tmp/swiglu2k_wt` at `5c77934b4c5` overlaid onto the installed torch by
`agent_space/run_wt.py` (a variant of `run_stochastic_worktree_test.py` that
overlays the whole `torch._inductor` package -- the original harness left
`choices.py`, `dependencies.py` and other `torch._inductor` submodules resolving
to the actively-edited main checkout, which is worth fixing in the shared
harness).

## 1. Kernel inventory of the four-kernel baseline

`scales="both"`, 4096x7168, no coordinate-descent tuning. Times are CUDA-side
per-iteration averages over 20 iterations from `torch.profiler`.

| # | Scheduler nodes | Role | Reads | Writes | us |
|---|---|---|---|---|---:|
| 0 | op0,op1,op8,op2,op3 | rowwise: SwiGLU + 1x32 amax + E8M0 scale + quantize + swizzled scale store | `arg0_1` | row payload, row scale (already swizzled) | 32.3 |
| 1 | op4,op5 | colwise: SwiGLU + 32x1 amax + E8M0 scale | `arg0_1` | `buf5` (colwise scale, unswizzled) | 37.9 |
| 2 | op6,op7 | colwise: SwiGLU **recomputed** + quantize, transposed payload store | `arg0_1`, `buf5` | col payload | 40.0 |
| 3 | op9 | colwise scale swizzle | `buf5` | col scale | 2.7 |

Two facts contradict the natural first hypothesis:

- **The activation is never materialized.** No kernel reads or writes a SwiGLU
  buffer. Kernels 1 and 2 each recompute `silu(gate)*up` from `arg0_1`. The
  round-tripped buffer is only `buf5`, the 0.92 MB unswizzled colwise scale.
  So there is no "realize the activation" lever to pull; recompute already
  happens.
- **Rowwise survives intact when colwise is present.** Kernel 0 is the same
  single kernel as in the rowwise-only case, including the blocked scale
  permutation fused into its epilogue.

The whole deficit is that the colwise direction costs three kernels and reads
`arg0_1` twice.

Scheduler node list (from `TORCH_LOGS=fusion`):

```
op0 Reduction [4096,224]x[32]   rowwise amax
op1 Pointwise [4096,224]        rowwise E8M0 scale
op2 Pointwise [4096,224,32]     rowwise quantize multiply
op3 Pointwise [4096,224,32]     rowwise clamp + to_fp8
op4 Reduction [128,7168]x[32]   colwise amax
op5 Pointwise [7168,128]        colwise E8M0 scale
op6 Pointwise [7168,128,32]     colwise quantize multiply
op7 Pointwise [7168,128,32]     colwise clamp + to_fp8
op8 Pointwise [1792,32,4,4]     rowwise scale swizzle
op9 Pointwise [1792,32,4,4]     colwise scale swizzle
```

## 2. Why each colwise fusion was rejected

### 2a. The structural asymmetry

The rowwise and colwise reductions are mirror images, but their *consumers* are
not. Node deps (dumped by instrumenting `Scheduler.can_fuse`):

```
op0 sizes=((4096, 224), (32,))     reads arg0_1 at 14336*d0 + 32*d1 + d2
op2 sizes=((4096, 224, 32),)       reads arg0_1 at 14336*d0 + 32*d1 + d2
op4 sizes=((128, 7168), (32,))     reads arg0_1 at 458752*d0 + d1 + 14336*d2
op6 sizes=((128, 32, 7168),)       reads arg0_1 at 458752*d0 + 14336*d1 + d2
```

For rowwise, the reduction dim (`d2`, the 32 contiguous K values) is last in
both the reduction and its pointwise consumer, so `op2`'s dep on `op1`'s output
is literally the same `MemoryDep` and set-intersection scoring finds it.

For colwise, `op4` is forced to put the reduction dim (`d2`, 32 rows) last,
while `op6` picks its own stride-descending order and puts that dim in the
*middle*. Same linear index function, different loop order, so no dep matches.

### 2b. The rejection chain, in order

Every one of these is an actual log line, not a guess.

1. `cannot fuse op5 with op6: no shared data` -- `score_fusion_memory` is a
   set intersection over `MemoryDep`s. `op5` writes `buf5` over 2 vars,
   `op6` reads it over 3 (with the 32 broadcast); no intersection, score 0.
   `choices.CollectiveDecisions.can_fuse` rejects on `shared_data_score == 0`
   before legality ever runs.
2. `cannot fuse op4 with op6: intermediate nodes between node1 & node2` --
   `op5` sits between them, so the pair can never be considered directly.
3. Consequently the only colwise fusions found are `op4+op5` and `op6+op7`,
   both of which only become possible in the single `is_reorder_round=True`
   pass that `Scheduler.fuse_nodes` runs after the ordinary rounds. Because
   that pass runs exactly once, the pair `(op4_op5, op6_op7)` it produces is
   never re-examined.
4. Forcing that pair to be considered exposes three further blockers, each of
   which had to be fixed for the fusion to happen:

   a. `Scheduler._try_reorder_loops_for_candidates` found a perfectly good
      reorder candidate on the shared `arg0_1` read
      (`equal_nwso = True`, `num_vars 3 vs 3`) but then hit `buf5`, whose
      write/read pair cannot be aligned by reordering, and did
      `return -1` -- **discarding the valid candidate**. The early return was
      meant to hand off to the reindexing path, but it also throws away
      reordering opportunities on other buffers.

   b. `FusedSchedulerNode.reorder_loops_by_dep_pair` requires every member to
      have identical `_sizes[0]`. Here `op6` is `(128, 32, 7168)` and `op7` has
      already been merged to `(29360128,)`, so it bailed with
      `Can not reorder fused node due to different sizes`.

   c. After reordering and after
      `_try_reindex_pointwise_for_reduction` mapped `op6_op7` onto
      `(917504, 32)`, the deps became **identical up to variable naming**:

      ```
      op4_op5 writes MemoryDep('buf5', (c0//7168) + 128*ModularIndexing(c0, 1, 7168), {c0: 917504})
      op6_op7 reads  MemoryDep('buf5', (d0//7168) + 128*ModularIndexing(d0, 1, 7168), {d0: 917504})
      ```

      `Scheduler.fusable_read_and_write` only canonicalizes with `normalize()`
      when `read.num_vars != write.num_vars`, so with both at 1 var it compared
      raw sympy expressions in different symbols and reported
      `memory deps did not match`.

   d. With legality passing, the re-validation inside `_try_fusion_pairs`
      recomputed the score from scratch and got 0 again, because
      `_score_fusion_memory_for_can_fuse` only falls back to
      `_score_fusion_memory_by_fusable_read_write` when
      `index_equivalent_dep_names` is non-empty (a nested-reduction-stack
      concept that does not apply here). Result: `no shared data`.

### 2c. Why the colwise scale swizzle (op9) still does not fuse

`op8` (rowwise swizzle) fuses because `op1` writes `buf1` contiguously in its
own iteration order, so `shared_data_after_inverting_indexing` can invert the
swizzle. `op5` writes `buf5` *transposed* relative to its iteration order
(`m + 128*k` while iterating `m`-major), so the inversion machinery returns 0
(`Shared memory after inversion: 0`) and `op9` stays a separate kernel. This is
a distinct lever and was not pursued.

## 3. Prototype

`agent_space/swiglu_2kernel_proto.patch`, four scheduler changes, no new
codegen mode and no new config flag:

1. `_try_reorder_loops_for_candidates`: replace the early `return -1` on an
   unalignable write->read buffer with a `prefer_reindex` flag; still defer to
   reindexing, but only if the reorder that was actually attempted scored 0.
2. `FusedSchedulerNode.reorder_loops_by_dep_pair`: allow members whose loops are
   fully merged into one var by splitting them back to the shape being permuted
   (`apply_loop_reindexing`) before applying the permutation to all members.
3. `Scheduler._member_deps_consistent` (new): after that split, verify every
   intra-node producer write still satisfies its reader, and roll back
   otherwise. **This check is load-bearing.** Without it the prototype silently
   corrupted the colwise payload: permuting `op6` while leaving the flat `op7`
   alone crossed their shared `buf6`, and 29,118,165 of 29,360,128 payload bytes
   differed from the baseline while the scales stayed correct.
4. `fusable_read_and_write` normalizes whenever `loop_ordering_after_fusion` is
   on rather than only on `num_vars` mismatch, and
   `_score_fusion_memory_for_can_fuse` always falls back to the vertical
   read/write scorer when exact scoring yields 0.

An earlier candidate lever -- iterating the reorder round to a fixpoint via a
new `loop_ordering_fusion_rounds` config -- turned out to be **unnecessary**
once the three blockers above were fixed (the fusion lands inside the existing
single reorder round), and it produced a worse tiling. It was dropped.

Result: colwise-only goes 3 -> 2 kernels, both-direction goes 4 -> 3 kernels
(rowwise, fused colwise, colwise scale swizzle).

The fused colwise kernel keeps the good `{'y': 128, 'x': 8192, 'r0_': 32}`
tiling and stores both outputs from one pass:

```python
tl.store(out_ptr1 + (y0 + 128*x1), tmp22, xmask & ymask)          # colwise scale
tl.store(out_ptr3 + (r0_2 + 32*y0 + 4096*x1), tmp47, xmask & ymask)  # colwise payload
```

## 4. Correctness

Compared against the pre-prototype schedule (reproduced in-process with
`loop_reindexing_after_fusion=False`), at 256x256 and 4096x7168, for
`colwise` and `both`: **all four outputs bitwise identical**, 0 mismatched
bytes. The pre-existing Inductor-vs-eager `cvt_e8m0_rceil` differences noted in
the follow-up doc are still there and are unaffected by this change (they show
up identically on the untouched rowwise path).

Test suites, run through the overlay harness:

```
python agent_space/run_wt.py test/inductor/test_loop_ordering.py   # Ran 124 tests, OK
python agent_space/run_wt.py test/inductor/test_perf.py            # Ran 69 tests, OK
```

`test_torchinductor.py` and `test_torchinductor_strided_blocks.py` could not be
run: they fail at import with `RuntimeError: operator torchvision::nms does not
exist` in this environment, unrelated to the patch. **This is a real gap in
validation** -- the `fusable_read_and_write` relaxation touches fusion legality
for every model, and it needs the full suite before it is trustworthy.

As a partial substitute, a differential test over eight fusion-sensitive graphs
(layernorm+transpose, row/col amax pairs, split-gate, blocked scale, softmax
pair, cat+reduce, permute chains, double reduce with transpose) was run against
an unpatched worktree at the same commit. Kernel counts were identical for all
eight. One output differed: `double_reduce_transpose` by 1.9e-6 in fp32, a
reduction-reassociation difference; the patched result is within 1e-4 of eager.

## 5. Measurements

`torch.compile` wall clock via `triton.testing.do_bench` and the CUDA-side sum
of per-kernel profiler times, both-direction, B200. Wall clock on this box was
noisy across runs (up to 15% spread on the same configuration), so the kernel
sum is the more reliable comparison; three full runs were taken and the median
is reported. CuTe numbers are the local same-session values from
`agent_space/torchao_4743_comparison_20260818.json`.

Kernel-time sum (us):

| Shape | baseline 4k | baseline 4k +cd | proto 3k | proto 3k +cd |
|---|---:|---:|---:|---:|
| 4096x2048  | 36.96 | 34.95 | 34.64 | **26.90** |
| 4096x7168  | 121.21 | 114.02 | 106.67 | **85.86** |
| 16384x7168 | 447.22 | 484.19 | 467.07 | **316.48** |

Wall clock (us), median of three runs, against CuTe:

| Shape | baseline+cd | proto+cd | CuTe #4743 | before | after |
|---|---:|---:|---:|---|---|
| 4096x2048  | 46.18 | 35.85 | 67.06 | Inductor 1.45x | **Inductor 1.87x** |
| 4096x7168  | 134.14 | 102.59 | 71.86 | CuTe 1.68x | **CuTe 1.43x** |
| 16384x7168 | 494.84 | 362.04 | 174.51 | CuTe 2.48x | **CuTe 2.07x** |

("before" uses the follow-up doc's published 4-kernel numbers, "after" the
prototype.)

Colwise-only, kernel sum (us), showing the isolated effect:

| Shape | baseline 3k +cd | proto 2k +cd | speedup |
|---|---:|---:|---:|
| 4096x2048  | 23.71 | 14.91 | 1.59x |
| 4096x7168  | 79.10 | 51.79 | 1.53x |
| 16384x7168 | 307.49 | 191.50 | 1.61x |

Rowwise-only is unchanged: 1 kernel, 11.3 / 33.4 / 143.9 us, within noise of
baseline. No regression.

**Coordinate-descent tuning is required.** Without it the prototype is roughly
neutral (and at 16384x7168 the fused colwise kernel is 320 us, slightly worse
than the 163+151 us pair it replaces). The default tiling heuristic picks a poor
configuration for the merged reduction+epilogue kernel at large shapes. If this
lands, that heuristic gap should be closed, or the fusion gated on the
coalescing analysis actually predicting a win.

Fraction of the CuTe gap recovered, using same-session baselines:
4096x7168 51%, 16384x7168 41%.

## 6. Does two kernels suffice?

**No, not at large shapes.** The bytes make this unambiguous. At 16384x7168
(input 469.8 MB, two 117.4 MB payloads, two 3.7 MB scale tensors):

| Schedule | HBM traffic | vs ideal |
|---|---:|---:|
| One shared-tile kernel (CuTe) | 712 MB | 1.00x |
| Two kernels, one per direction | 1182 MB | 1.66x |
| Current four kernels | 1662 MB | 2.33x |

The prototype moves traffic from 2.33x to 1.66x of ideal and measured time from
2.80x to 2.07x of CuTe. Those track. The residual 1.66x is *structural*: each
direction reads the full 469.8 MB producer independently, and no scheduler-level
change can remove that -- only one codegen context evaluating one SwiGLU tile
consumed by both reductions can.

Achieved bandwidths at 16384x7168 confirm there is little else to win:

- rowwise kernel: 590.9 MB / 133.4 us = 4.43 TB/s
- fused colwise kernel: 590.9 MB / 176.0 us = 3.36 TB/s
- CuTe single kernel: 712.0 MB / 174.5 us = 4.08 TB/s

Even if the colwise kernel reached the rowwise kernel's 4.43 TB/s, two kernels
would cost 2 x 590.9 / 4.43 = 267 us, still 1.53x CuTe. Notably CuTe sustains
4.08 TB/s *including* the transposed FP8 store, essentially matching Inductor's
untransposed rowwise kernel -- evidence that the shared-memory transpose
pipeline works and that a Triton shared-tile kernel would need something similar
rather than a pure register-layout conversion.

At 4096x2048 the picture is different: Inductor is already 1.87x faster than
CuTe, whose fixed overhead dominates at that size.

### Recommendation

1. **Land the two-kernel work anyway.** It is a self-contained scheduler fix,
   it is bitwise-neutral on this workload, it gives 1.3-1.5x on both-direction
   and 1.5-1.6x on colwise-only, and it removes a whole class of "reduction and
   its epilogue disagree about loop order" misses that is not specific to MXFP8.
   Blockers 4a-4d are all general bugs, not workload-specific hacks.
2. **Do not treat it as a substitute for the blockwise-2D shared-tile mode.**
   The follow-up doc's stop condition ("if two kernels recover most of the gap,
   stop there") is not met at large shapes: 41% recovered at 16384x7168, with a
   1.66x traffic floor proving the rest is unreachable. Proceed with the shared
   2D mix-order kernel for the large both-direction case.
3. Before landing, (a) run the full inductor suite -- blocked here by a
   torchvision import error, (b) fix the default tiling for the merged kernel so
   the win does not depend on `coordinate_descent_tuning`, and (c) decide
   whether the `op9` colwise scale-swizzle fusion (a transposed-producer index
   inversion, ~1.4-2.5% of runtime) is worth a separate change.
