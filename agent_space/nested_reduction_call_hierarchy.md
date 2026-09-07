# Nested Reduction: Detailed Call Hierarchy and Review Guide

This note is for the **core** nested-reduction PR, not the larger
NVFP4/half-resolution branch.

It matches the branch and stack currently meant for review:

- worktree: `/data/users/eellison/pytorch_nested_reduction_core_review_v3`
- branch: `nested_reduction_core_review_v3`
- commits:
  1. `a2219b39b81` `[inductor] Refactor Triton reshape helpers`
  2. `8746b05048f` `[inductor] Fuse dependent cross-axis reductions`

It does **not** cover the saved half-resolution / NVFP4 branch. That work is
preserved separately on:

- worktree: `/data/users/eellison/pytorch_nested_reduction_single_commit_wip`
- branch: `nested_reduction_with_nvfp4_saved`

The goal of this doc is not just "what class exists where", but:

1. what order things happen in
2. which objects own which state
3. why `codegen_nested_reduction()` still exists even after the derived-family
   refactor
4. what to focus on when reviewing

---

## 1. Scope of the core PR

This branch implements:

- fusion of **two dependent reductions** over the same large input
- reduced-output pointwise epilogues
- full-resolution pointwise epilogues
- dynamic-shape correctness for the derived reduction-space path
- `B=1` / singleton-extent indexing canonicalization

This branch intentionally does **not** implement:

- half-resolution consumers
- NVFP4 even/odd split-lane path
- scheduler-owned derived families
- generic staged composite-kernel infrastructure
- affine masked subregions / MLA / cat-scatter generalization

So when reviewing this branch, think:

> "Can Inductor fuse `outer reduction -> grouped reduction -> reduced/full-res
> epilogue` correctly and cleanly?"

Do **not** judge it by whether it already removes `codegen_nested_reduction()`
or solves the more general architecture.

---

## 2. Review order

If you want the shortest useful review path, read in this order:

1. `test/inductor/test_nested_reduction.py`
2. `torch/_inductor/scheduler.py`
3. `torch/_inductor/codegen/simd.py`
4. `torch/_inductor/codegen/triton.py`

Reason:

- the test file defines the intended capability surface
- the scheduler defines what patterns are legal
- `simd.py` is the actual implementation
- `triton.py` is support code for the implementation

More concretely:

### Commit 1: `a2219b39b81`

Read only:

- `torch/_inductor/codegen/triton.py`

This is a small helper refactor around reshape formatting. It is intentionally
kept separate because it is generic and easy to validate on its own.

### Commit 2: `8746b05048f`

Read in this order:

1. `test/inductor/test_nested_reduction.py`
2. `torch/_inductor/scheduler.py`
3. `torch/_inductor/codegen/simd.py`
4. `torch/_inductor/codegen/triton.py`
5. small support files only after the above:
   - `torch/_inductor/runtime/triton_heuristics.py`
   - `torch/_inductor/codegen/cuda_combined_scheduling.py`
   - `torch/_inductor/config.py`
   - `torch/_inductor/metrics.py`

---

## 3. High-level end-to-end call flow

This is the most important picture to keep in mind.

```text
Scheduler fusion phase
│
├─ NestedReduction.can_fuse(node1, node2)            scheduler.py:445
│  ├─ check devices/backend/config
│  ├─ check node1 is reduction, node2 depends on node1
│  ├─ compute shared_reads or producer-consumer relation
│  ├─ check node2 is exactly one small reduction
│  ├─ check total element counts match
│  └─ profitability / coalescing guard
│
├─ if accepted:
│  └─ FusedNestedReductions(node1, node2)            scheduler.py:2535
│     ├─ strips internal ancestors
│     └─ computes small_dim_in_r once
│
├─ downstream pointwise fusion:
│  └─ FusedNestedReductions.can_fuse_with(other)     scheduler.py:2572
│     ├─ accepts reduced-output consumers
│     └─ accepts full-resolution consumers only when small_dim_in_r
│
Codegen dispatch
│
└─ SIMDScheduling.codegen_node(node)                 simd.py:2772
   └─ if isinstance(node, FusedNestedReductions):
      └─ codegen_nested_reduction(node)              simd.py:2394
         ├─ build one kernel over node1's outer iteration space
         ├─ emit outer reduction stage
         ├─ emit grouped second reduction stage
         ├─ emit reduced/full-resolution epilogues
         └─ finalize and launch the fused Triton kernel
```

The key point is:

- the scheduler recognizes a special composite pattern
- codegen then emits a **single staged kernel**
- derived families unify the **consumer-side** remapped stages
- but orchestration is still custom

---

## 4. What pattern is actually being fused?

The pattern is:

1. `node1` does a large reduction over some dimension `D`
2. `node2` depends on `node1` and reduces again over a small `group_size`
3. both traverse the same total logical element count
4. the second reduction is a grouped reinterpretation of one axis of the
   first stage's working tile

Two supported layouts:

### A. `small_dim_in_x`

Example shape idea:

- `node1`: outer reduction over `R`
- `group_size` is embedded in the non-reduction / `X` direction
- grouped stage reshapes `XBLOCK -> [XBLOCK / G, G]`

Example pattern:

- RMSNorm-like outer reduction
- then weighted sum / max across a small `K`

### B. `small_dim_in_r`

Example shape idea:

- `group_size` is embedded in the reduction / `R` direction
- grouped stage reshapes `RBLOCK -> [RBLOCK / G, G]`

Example pattern:

- layernorm or rmsnorm over `[B, D]`
- then per-block `amax` or sum over groups of 8/16/32/128

This second case is the one that also supports full-resolution epilogues in the
core branch.

---

## 5. Scheduler-side flow in detail

### 5.1 `NestedReduction` — `scheduler.py:426`

This is the pattern recognizer. It is a namespace of classmethods; it is never
instantiated.

Main entrypoint:

- `NestedReduction.can_fuse(node1, node2)` — `scheduler.py:445`

What it checks, in order:

1. feature flag:
   - `torch._inductor.config.triton.nested_reduction`
2. reject `cpp_wrapper`
3. require GPU Triton backend
4. require `node1.is_reduction()`
5. require `node2` depends on `node1`
6. require either:
   - shared input reads, or
   - producer-consumer through node1's output
7. require `node2` is exactly one reduction
8. require `node2` reduction size is:
   - static
   - power of two
   - <= `MAX_SMALL_REDUCTION`
9. require `numel1 * rnumel1 == numel2 * rnumel2`
10. apply profitability guard for the `small_dim_in_x` case

Supporting helpers:

- `get_shared_input_reads(node1, node2)`
  - finds buffers both sides read
  - excludes node1's own outputs
- `is_small_dim_in_r(...)`
  - single source of truth for axis classification
  - used by both scheduler and codegen
- `get_fusion_score(node1, node2)`
  - byte-size based fusion score

The important review question here is:

> Are the legality and profitability conditions narrow enough that the
> codegen path only sees cases it actually supports?

### 5.2 `FusedNestedReductions` — `scheduler.py:2535`

This is the fused scheduler node representing the composite pattern.

Important things it does:

1. stores `node1` and `node2`
2. strips node1's operation names back out of `self.ancestors`
   - because after fusion node1 is internal, not an external ancestor
3. computes `self.small_dim_in_r` once in `__init__`

That last piece matters:

- scheduler and codegen both read the same precomputed classification
- they do not re-derive it independently

`can_fuse_with(other)` is the scheduler-side rule for downstream pointwise
consumers:

- reduced-output consumers are allowed
- full-resolution consumers are allowed only if:
  - their `numel` matches full resolution
  - `small_dim_in_r == True`

The practical effect is:

- node2 can absorb some pointwise consumers before codegen
- codegen then splits those consumers into reduced-output vs full-resolution
  based on extent

---

## 6. Why `codegen_nested_reduction()` still exists

This is the core architecture question.

The derived-family refactor did **not** eliminate `codegen_nested_reduction()`
because the feature still needs custom orchestration.

The family abstraction solves:

- how a **pointwise** body runs in a remapped iteration space
- how a pointwise body resolves loads/stores through that remapped space

It does **not** solve:

1. how to recognize the nested reduction pattern
2. how to plan a staged kernel
3. how to run a grouped second reduction
4. how to order:
   - outer reduction
   - grouped reduction
   - epilogues

So `codegen_nested_reduction()` exists because this feature is still:

> one special composite kernel with multiple stages

The long-term architecture could eliminate that entry point, but only if
Inductor grows:

- generic staged composite kernels
- reduction stages that can run inside derived iteration families
- earlier composite recognition before codegen

That is future work. It is not what this core PR is trying to finish.

---

## 7. The main codegen concepts in `simd.py`

This is the heart of the implementation.

### 7.1 `DerivedIterationRangesRoot` — `simd.py:381`

This is a subclass of `IterationRangesRoot`.

Purpose:

- represent a **derived view** of an existing range tree
- keep normal indexing/mask generation working against that derived view

It carries:

- `parent`
- derived `numel`
- derived `block_size`
- derived `block_offset`

Critical invariant:

- `is_loop = parent.is_loop`

That was the real dynamic-shapes bug fix.

Why it matters:

- if the parent reduction tree is looped, the derived tree must also be looped
- otherwise its block offset gets computed once in the wrong scope
- dynamic-shape pattern 2 used that stale offset and produced misaligned
  addresses

Important overrides:

- `block_size()`
- `block_offset()`
- `index_sym()`
- `supports_constant_mask()`
- `mask_name()`

### 7.2 `_DerivedIterationFamily` — `simd.py:1424`

This is the key abstraction introduced by the refactor.

It bundles:

- which range trees are active
- optional index substitutions
- optional precomputed remapped values
- optional flat index expression

The core branch uses **two** configurations:

#### Reduced-output family

- `range_trees` = two derived trees
- `index_subs` populated
- `remapped_values` empty
- `flat_index_expr` absent

Used for:

- grouped reduction output
- reduced-output pointwise epilogues

#### Full-resolution family

- `range_trees` = outer trees
- `index_subs` empty
- `remapped_values` populated with lifted/broadcast values
- `flat_index_expr` populated

Used for:

- full-resolution pointwise epilogues

Important methods:

- `remap_index(index)`
- `is_active_on(kernel)`
- `load(kernel, name, index)`
- `store(kernel, name, index, value)`
- `ensure_headers(kernel)`
- `activate(kernel)`

The load resolution order is important:

1. `family.remapped_values`
2. `kernel.cse.store_cache`
3. real load through the active family

That is the design center of the consumer-side path.

### 7.3 `_GroupReductionLayout` — `simd.py:1531`

This is the structural description of the grouped second reduction.

Fields:

- `x_tree`
- `r_tree`
- `group_size`
- `group_size_str`
- `small_dim_in_r`

From that it derives:

- which tree is the grouped one
- which tree is the "other" one
- which axis is parent/grouped
- `reshape_shape`
- `reduce_axis`
- `output_shape`
- group counts
- broadcast shapes
- flat-index reconstruction

This class is dense, but coherent. It is the one place where the grouped
reduction geometry is defined.

Key methods:

- `from_kernel(...)`
- `construct_group_reduction_vars(...)`
- `make_reduced_output_family(...)`
- `make_full_resolution_family(...)`
- `emit_broadcast_value(...)`
- `full_resolution_flat_index()`

### 7.4 `_GroupedReductionOpsHandler` — `simd.py:1752`

This handler owns the grouped reduction stage itself.

It is active only around `node2_reduction_body(...)`.

It does two real things:

1. `reduction(...)`
   - reshape the full-resolution tile into the grouped layout
   - reduce over the group axis
2. `store_reduction(...)`
   - mirror the standard reduction-store bookkeeping
   - route the physical store through the reduced-output family

Why the bookkeeping mirror exists:

- this stage bypasses the normal `KernelHandler.store_reduction` path
- but it still needs:
  - `store_buffer_names.add`
  - `store_cache` update
  - mutation propagation
  - `num_store` update

So the code mirrors the standard handler bookkeeping and then uses
`family.store(...)` for the actual remapped store.

### 7.5 `_PointwiseRemapHandler` — `simd.py:1823`

This is intentionally thin now.

It exists so a pointwise body can run with:

- remapped loads through a family
- remapped stores through a family

It is used for:

- reduced-output epilogues
- full-resolution epilogues

The important simplification from earlier versions is:

- this handler does not own separate special-case logic anymore
- all the real load/store routing lives on `_DerivedIterationFamily`

---

## 8. `codegen_nested_reduction()` phase by phase

`SIMDScheduling.codegen_nested_reduction()` is at `simd.py:2394`.

This is the function to read carefully. It is large, but it is doing
real orchestration.

### Phase 1: split `node2`

It starts by:

- unpacking `node1`, `node2`
- reading group sizes from their scheduler groups
- splitting `node2.get_nodes()` into:
  - `node2_reduction`
  - `node2_epilogues`

That split matters because:

- only the reduction subnode runs under `_GroupedReductionOpsHandler`
- the epilogues run later under `_PointwiseRemapHandler`

### Phase 2: choose kernel shape

It then computes:

- `shared_reads`
- `is_producer_consumer`
- `small_dim_in_r = node.small_dim_in_r`

Then it builds the **outer** schedule from `node1` only:

- `combined_schedule = generate_node_schedule(list(node1.get_nodes()), numel1, rnumel1)`

Then it computes:

- coalescing analysis
- `SIMDKernelFeatures`
- tiling / tiling scores

Then it creates the kernel and sets nested-reduction-specific tiling knobs:

- `nested_reduction_min_rblock`
- `nested_reduction_min_xblock`
- `nested_reduction_max_xblock`

Those knobs are not global abstractions; they are feature-local constraints for
the grouped second reduction.

### Phase 3: pre-mark internal node1 outputs

For each buffer written by node1:

- if all users are internal to the fused node
- mark it in `V.graph.removed_buffers`
- remember it in `internal_node1_outputs`

Why early:

- so when normal node codegen runs, those stores can already be treated as
  internal-only

### Phase 4: run normal node schedule codegen

Then:

- `self.codegen_node_schedule_with_kernel(combined_schedule, kernel)`

This is subtle.

What it does **not** mean:

- it is not doing the grouped second reduction yet

What it **does** mean:

- run the normal codegen for node1's existing scheduled nodes
- populate CSE state, arguments, and buffering for the outer stage

After that:

- call `kernel.remove_buffer(name)` for internal node1 outputs
- verify every node1 output is present in `kernel.cse.store_cache`

That `store_cache` verification is important:

- the grouped reduction stage reads node1 outputs from CSE/register state
- if they are missing, the grouped second reduction cannot proceed

### Phase 5: prepare node2 reduction output / epilogue outputs

It computes `node2_reduction_output_name` and:

- if epilogues exist:
  - add epilogue outputs as kernel outputs
  - possibly mark the reduction output itself as internal-only
- otherwise:
  - make the reduction output a real kernel output

### Phase 6: emit staged kernel body

Inside `with kernel:` the staged emission happens.

#### Stage 6A: flush outer reduction

`kernel.codegen_body()` emits the outer stage.

#### Stage 6B: build layout and reduced-output family

It builds:

- `layout = _GroupReductionLayout.from_kernel(...)`
- grouped reduction iter-var remap via `construct_group_reduction_vars(...)`
- `reduced_output_family = layout.make_reduced_output_family(...)`

#### Stage 6C: emit grouped reduction

It creates `_GroupedReductionOpsHandler(...)` and runs:

```python
with V.set_ops_handler(handler), kernel.set_current_node(node2_reduction):
    node2_reduction_body(iter_remapped, reduce_remapped, ...)
```

This is the grouped second reduction proper.

At this point:

- reduction ops are intercepted by `_GroupedReductionOpsHandler.reduction`
- stores are intercepted by `_GroupedReductionOpsHandler.store_reduction`

#### Stage 6D: emit epilogues

`_codegen_group_reduction_epilogue(...)` splits epilogues into:

- reduced-output epilogues
- full-resolution epilogues

and dispatches them to separate helpers.

#### Stage 6E: clear unused post-loop reduction machinery

Because the grouped reduction already stored its result explicitly, it clears:

- `kernel.post_loop_combine`
- `kernel.post_loop_store`

#### Stage 6F: finalize internal epilogue candidates

This is the late internal-buffer cleanup for node2's reduction output when that
output ended up being fully internal after epilogue fusion.

#### Stage 6G: final `kernel.codegen_body()`

This flushes the remaining staged code.

### Phase 7: finalize the kernel

`_finalize_nested_reduction_kernel(...)` does the standard endgame:

- collect config patches
- `kernel.codegen_kernel()`
- `define_kernel(...)`
- `mark_run()` on all fused nodes
- wrapper comments / profiling guards
- `kernel.call_kernel(...)`
- optional nan checks / mix-layout warnings
- merge `kernel.removed_buffers` back into `V.graph.removed_buffers`
- free scheduler buffers

---

## 9. Epilogue paths

### 9.1 Reduced-output epilogues

Helper:

- `_codegen_reduced_resolution_epilogue(...)`

Flow:

1. flatten the grouped reduction iteration vars
2. activate the reduced-output family
3. for each epilogue:
   - decompose the flat grouped index into that body's iter vars
   - create `_PointwiseRemapHandler`
   - run the epilogue body

This is the cleanest family use case:

- the body logically runs in the grouped output space

### 9.2 Full-resolution epilogues

Helper:

- `_codegen_fullres_epilogue(...)`

Flow:

1. add epilogue outputs to `kernel.args`
2. collect all read names across full-resolution epilogues
3. build a full-resolution family with:
   - outer trees
   - `flat_index_expr`
   - pre-broadcast `remapped_values`
4. activate the family
5. for each epilogue:
   - decompose full-resolution flat index into body iter vars
   - run the body under `_PointwiseRemapHandler`

This is the core reason the family abstraction is a win:

- full-resolution epilogues look like the same problem as reduced-output
  epilogues, just with a different family

---

## 10. How indexing and masks work

This is the part that tends to feel magical if you read it too late.

### 10.1 `use_range_trees(...)`

`SIMDKernel.use_range_trees(...)` temporarily swaps `kernel.range_trees`.

That is what makes:

- derived reduced-output trees
- outer full-resolution trees

look like the active iteration space for load/store/indexing.

It also clears `simplify_indexing`'s cache on entry and exit because the cache
does not otherwise know the active family changed.

### 10.2 `active_range_trees()`

`SIMDKernel.active_range_trees()` is:

```python
[t for t in self.range_trees if not t.is_reduction or self.inside_reduction]
```

This is **not** a no-op.

Outside reduction loops:

- reduction trees are hidden from mask/index reasoning

Inside reduction loops:

- reduction trees are active

That distinction matters for correct mask generation and simplification.

### 10.3 Singleton-numel canonicalization

In `SIMDKernel.prepare_indexing()`:

- any active range-tree variable whose `numel` is statically `1`
- gets replaced with `0`

This is global, not nested-specific.

Why it exists:

- if an extent is singleton, its loop/index variable can only be `0`
- canonicalizing it to `0` is mathematically exact
- it lets degenerate `B=1` tiles CSE identical loads/stores

### 10.4 Derived-tree mask names

In `triton.py`, mask-name handling was generalized so derived trees can carry
their own mask names instead of being inferred from prefix strings.

That matters because reduced-output trees are not just plain `xmask` / `rmask`
any more.

---

## 11. CSE, stores, and buffer ownership

This is another area worth reviewing carefully.

### 11.1 Standardized mechanism

The branch intentionally removed the older `inline_reduction_buffers` path.

Now the mechanisms are:

- `kernel.cse.store_cache`
- `V.graph.removed_buffers`
- `kernel.remove_buffer(...)`

That is the same family of mechanisms the rest of Inductor already uses.

### 11.2 Why `store_cache` is important here

The grouped reduction stage needs to consume node1 outputs without materializing
an intermediate kernel boundary.

So after the outer stage codegen, `codegen_nested_reduction()` asserts:

- every node1 output must be present in `kernel.cse.store_cache`

If not, the grouped second reduction cannot read the prior stage correctly.

### 11.3 Why `_GroupedReductionOpsHandler.store_reduction()` mirrors bookkeeping

Because it does not route through the normal `KernelHandler.store_reduction()`,
it explicitly mirrors the standard bookkeeping:

- `store_buffer_names.add(name)`
- `store_cache[name] = value`
- mutation propagation
- `num_store += 1`

Then it routes the physical store through `family.store(...)`.

This is intentional, not accidental duplication.

### 11.4 Why removed buffers are checked in both places

`_DerivedIterationFamily.store(...)` checks both:

- `V.graph.removed_buffers`
- `kernel.removed_buffers`

because during codegen the kernel-local removed set can get ahead of the global
one. The final union happens later.

---

## 12. What good generated kernels should look like

The tests now check structure, not just numerics.

### Reduced-output `amax` path

The expected kernel form is:

- one `tl.load(in_ptr0 +`
- one `tl.load(in_ptr1 +`
- no `tl.split(`
- one `tl.store(out_ptr`

### Full-resolution FP8-ish path

The expected kernel form is:

- one load from each input
- no `tl.split(`
- `tl.broadcast_to` present
- two stores

Those are pinned in:

- `assert_amax_kernel_form(...)`
- `assert_fullres_kernel_form(...)`

in `test/inductor/test_nested_reduction.py`.

Those tests use:

- `run_and_get_code`
- `fresh_inductor_cache`
- `FileCheck`

So they validate the actual emitted fused Triton source, not just numerics.

---

## 13. What to look for in review

If you want a checklist:

### Scheduler

- Is `NestedReduction.can_fuse()` narrow enough?
- Is `is_small_dim_in_r()` the right single source of truth?
- Is `FusedNestedReductions.can_fuse_with()` consistent with codegen support?

### `simd.py`

- Does `_DerivedIterationFamily` actually simplify consumer-side codegen?
- Does `_GroupReductionLayout` keep the geometry in one place?
- Does `_GroupedReductionOpsHandler` do only the grouped reduction stage?
- Does `_PointwiseRemapHandler` stay thin?
- Is `codegen_nested_reduction()` large because of real orchestration, or
  because of avoidable helper churn?

### `triton.py`

- Are the backend hooks minimal and justified?
- Does the mask-name generalization make sense for derived trees?

### Tests

- Do the tests cover:
  - numerics
  - dynamic shapes
  - `B=1`
  - kernel form
- Are they testing the core branch's actual promises, not the future NVFP4
  branch?

---

## 14. Remaining debt that is real, but not blocking

These are the things I would keep in mind while reviewing, without expecting
this PR to solve them.

1. `codegen_nested_reduction()` is still a dedicated orchestrator.
   - correct for current architecture
   - not the long-term end state

2. There is still feature-local kernel state.
   - nested-reduction block hints
   - some late internal-buffer bookkeeping

3. The next simplification step is architectural.
   - staged composite kernels
   - scheduler-owned derived families
   - reduction stages inside derived families

That is a follow-up design discussion, not a reason this core branch is wrong.

---

## 15. If you want to trace it live

Useful commands while reading:

```bash
rg -n "class NestedReduction|class FusedNestedReductions|def can_fuse_with" \
  /data/users/eellison/pytorch_nested_reduction_core_review_v3/torch/_inductor/scheduler.py

rg -n "class DerivedIterationRangesRoot|class _DerivedIterationFamily|class _GroupReductionLayout|class _GroupedReductionOpsHandler|class _PointwiseRemapHandler|def codegen_nested_reduction" \
  /data/users/eellison/pytorch_nested_reduction_core_review_v3/torch/_inductor/codegen/simd.py

rg -n "assert_amax_kernel_form|assert_fullres_kernel_form|test_dynamic_shapes_pattern2|test_producer_consumer_rmsnorm_amax_B1|test_producer_consumer_rmsnorm_fp8_quant_B1" \
  /data/users/eellison/pytorch_nested_reduction_core_review_v3/test/inductor/test_nested_reduction.py
```

If you want to add temporary tracing, the most useful places are:

- top of `NestedReduction.can_fuse`
- top of `FusedNestedReductions.can_fuse_with`
- top of `codegen_nested_reduction`
- `_GroupedReductionOpsHandler.reduction`
- `_GroupedReductionOpsHandler.store_reduction`
- `_PointwiseRemapHandler.load`
- `_PointwiseRemapHandler.store`

But for review, I would first read the code statically with this document in
hand. The current branch is understandable without dynamic tracing once the
stage structure is clear.
