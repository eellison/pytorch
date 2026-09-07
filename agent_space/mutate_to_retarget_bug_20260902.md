# Silent wrong answers: `mutate_to` retargets an earlier in-place mutation

Found on 2026-09-02 by the randomized `index_put` fuzzer written for the padded-scatter
work (`agent_space/pr191974_failure_investigation/tmp_scatter_fuzz.py`, seed 188).

**This is a pre-existing mainline bug. It has nothing to do with nested reductions**
(reproduces with `triton.nested_reduction` off, and on a pristine `cdd22ade294`
worktree with no stack patches). It affects **CPU and CUDA** equally.

## Minimal repro

```python
def f(x):
    y = x.clone()
    y[torch.tensor([1, 3])] = torch.tensor([9.0, 9.0])
    y[x > 0] = -5.0
    return y

x        [-0.925, -0.425, -2.644,  0.145, -0.121, -0.58, -0.623, -0.328]
eager    [-0.925,  9.0,   -2.644, -5.0,   -0.121, -0.58, -0.623, -0.328]
compiled [-0.925,  9.0,   -2.644,  9.0,   -0.121, -0.58, -0.623, -0.328]
                                   ^^^ masked fill lost; the earlier index assign wins
```

## Root cause

`index_put_` with a tensor index builds a `Scatter` whose layout is
`ir.MutationLayoutSHOULDREMOVE(self)`, where `self` is the destination **TensorBox**
(`lowering.py:5094`). `MutationLayoutSHOULDREMOVE.get_buffer()` (`ir.py:5213`) resolves
`self.target` by walking `MutableBox.data` **at call time**, not at construction time.

The following masked `index_put_` dispatches to `index_put_as_masked_fill` ->
`mutate_to(self, where(mask, value, self))`. `mutate_to`'s fast path
(`lowering.py:7711`) does:

```python
val.realize()
changed_data.data = val.data   # swing the StorageBox pointer
```

That swing repoints the very box the earlier scatter resolves through. The scatter's
mutation target therefore becomes the `where` kernel's output buffer. Generated code
for the fuzz case:

```
buf0 = empty(7)                    # the clone; now never written
buf2 = empty(7)
kernel0(mask, fill, buf0, buf2)    # buf2 = where(mask, fill, buf0)  <- reads unwritten buf0
kernel1(base, buf2)                # scatter writes into buf2        <- clobbers the where
return buf2
```

Two errors from one cause: the `where` reads a buffer the scatter no longer writes, and
the scatter runs last and overwrites the `where` result.

`squeeze_` / `unsqueeze_` / `as_strided_` also swing a box pointer, but only onto a view
of the same storage, which `get_buffer()`'s view unwrapping still resolves to the same
`Buffer`. `mutate_to` is the only site that swings onto a *different* buffer.

## Trigger surface

Any `MutationLayoutSHOULDREMOVE`-based mutation of an intermediate, followed by a
`mutate_to`-based mutation of the same intermediate.

| case | mainline | with padded-scatter patch | with fix |
| --- | --- | --- | --- |
| index assign then masked assign | WRONG | ok | ok |
| index assign then masked accumulate | WRONG | **WRONG** | ok |
| index assign then `copy_` | ok | ok | ok |
| index assign then `fill_` | ok | ok | ok |
| partial index assign then masked assign | WRONG | ok | ok |
| masked assign then masked assign | ok | ok | ok |
| `scatter_` then masked assign | WRONG | ok | ok |

The padded-scatter lowering in `agent_space/pr191974_padded_wt` incidentally fixes 4 of
the 5 because it stops routing masked fills through `mutate_to`. It does **not** fix
`accumulate=True`, which still falls back to `mutate_to`. The two changes are
independent; this one belongs in its own PR against main.

## Fix

`agent_space/mutate_to_retarget_wt` (detached at `cdd22ade294`), uncommitted,
`+51/-5` across two files:

```diff
-    if isinstance(changed_data, ir.StorageBox) and not (
-        changed_data.is_input_buffer()
-        # In AOTI, module parameters and buffers are not lifted as graph inputs
-        or changed_data.is_module_buffer()
-        or isinstance(changed_data.data, ir.NopKernel)
+    changed_name = ir.try_get_name(changed_data)
+    if (
+        isinstance(changed_data, ir.StorageBox)
+        and not (
+            changed_data.is_input_buffer()
+            # In AOTI, module parameters and buffers are not lifted as graph inputs
+            or changed_data.is_module_buffer()
+            or isinstance(changed_data.data, ir.NopKernel)
+        )
+        # An earlier MutationLayoutSHOULDREMOVE resolves its target lazily through
+        # this box, so swinging the pointer would retarget that mutation at `val`.
+        and changed_name not in V.graph.mutated_buffers
     ):
```

Falling through to `MutationLayoutSHOULDREMOVE.realize_into` is correct here: both
mutations then target the same buffer and the scheduler orders them, which is already
what happens for input buffers today. Cost is one `Pointwise` copy node that the
scheduler normally fuses in place.

### Alternative considered and rejected

Snapshotting the resolved `Buffer` inside `MutationLayoutSHOULDREMOVE.__init__` is the
more root-cause fix, but `self.target` is also used by `make_indexer()` and `get_size()`,
where the *view* wrapper matters, so it cannot simply be replaced by the unwrapped
buffer. The narrow `mutate_to` guard changes behaviour only in the case that is
currently wrong.

## Randomized campaign

`agent_space/pr191974_failure_investigation/tmp_mutseq_fuzz.py` generates random
sequences of 2-4 in-place mutations (`index_assign`, `masked_assign`,
`masked_accum`, `scatter_`, `copy_`, `fill_`, `add_`, `slice_assign`,
`index_copy_`, `masked_fill_`, `index_add_`, `_unsafe_index_put`) applied to one
intermediate, with eager as the oracle. 300 seeds:

| tree | static shapes (300 seeds) | dynamic shapes (250 seeds) |
| --- | --- | --- |
| plain main (no fix) | 285 PASS, **15 MISMATCH** | 237 PASS, **13 MISMATCH** |
| with the fix | **300 PASS** | **250 PASS** |

Every one of the 15 failures is the same shape: a scatter-style mutation
(`scatter_` / `index_copy_` / `index_add_` / `index_assign` / `_unsafe_index_put`)
followed later by `masked_assign` or `masked_accum`. All are large tensor errors
(max abs 0.43 - 11.1), not float noise. Sample sequences:

```
scatter_ -> masked_assign                                  (x2)
index_copy_ -> masked_accum
index_copy_ -> index_assign -> scatter_ -> masked_assign
add_ -> unsafe_index_put -> masked_assign
unsafe_index_put -> index_copy_ -> masked_accum -> slice_assign
```

Coverage control: of the 300 seeds, 41 contain scatter-like-then-masked (15 of
which produce an observable difference) and 112 contain scatter-like followed by
`copy_` / `fill_` / `add_` / `slice_assign` / `masked_fill_`. None of that second
group failed, which matches the hand-written table above: only the two masked
`index_put_` paths reach the unguarded swing in a way that changes the result.

### Where each tree stands (same 300 seeds)

| tree | result |
| --- | --- |
| plain main | 285 PASS, 15 MISMATCH |
| `pr191974_padded_wt` (padded scatter only) | 293 PASS, **7 MISMATCH** |
| `mutate_to_retarget_wt` (this fix only) | **300 PASS** |

All 7 that survive the padded change involve `masked_accum`, i.e. exactly the
`accumulate=True` branch that still falls back to `mutate_to`. So the padded
lowering is not a substitute for this fix. It is also not a regression: those 7
are already wrong on main.

The original scatter fuzzer (`tmp_scatter_fuzz.py`, 250 static seeds) run against
this fix gives 246 PASS / 4 MISMATCH, and all 4 are the previously-characterised
benign class: `out1 nbad=1/1 max<=3.05e-05` on the `.sum()` consumer, i.e. float
reduction-order noise. Seed 188 -- the seed that originally exposed this bug --
now reports only that `out1` noise (`max=9.54e-07`); its `out0` tensor error
(`nbad=3/7 max=5.1`) is gone.

## Verification

```
# targeted slices, with the fix
python /tmp/run_wt_novision.py test/inductor/test_torchinductor.py \
  -k index_put -k unsafe_index -k masked_fill -k scatter -k index_ \
  -k copy_ -k fill_ -k mutat -k inplace
-> Ran 250 tests, OK (skipped=13)

# new regression tests, with the fix
python /tmp/run_wt_novision.py test/inductor/test_torchinductor.py -k after_scatter_mutation
-> Ran 4 tests, OK

# same two test bodies on plain main (no fix): non-vacuity check
-> test_index_put_after_scatter_mutation_cpu             *** MISMATCH ***
   test_index_put_after_scatter_mutation_cuda            *** MISMATCH ***
   test_index_put_accumulate_after_scatter_mutation_cpu  *** MISMATCH ***
   test_index_put_accumulate_after_scatter_mutation_cuda *** MISMATCH ***

# same slices under dynamic shapes
python /tmp/run_wt_novision.py test/inductor/test_torchinductor_dynamic_shapes.py \
  -k index_put -k unsafe_index -k masked_fill -k scatter -k index_ \
  -k copy_ -k fill_ -k mutat -k inplace
-> Ran 250 tests, OK (skipped=14)

# opinfo
python /tmp/run_wt_novision.py test/inductor/test_torchinductor_opinfo.py \
  -k index_put -k index_copy -k index_add -k scatter -k masked_fill
-> Ran 233 tests, OK (skipped=34)

# AOT inductor
python /tmp/run_wt_novision.py test/inductor/test_aot_inductor.py \
  -k index_put -k mutat -k scatter -k buffer -k inplace
-> Ran 100 tests, FAILED (errors=1, skipped=33)
   the single error is test_buffer_mutation_and_force_mmap_weights_cpu, which
   fails identically on the baseline worktree: onednn::qlinear_prepack has no
   Meta kernel in this build. Not related to the fix.

lintrunner torch/_inductor/lowering.py test/inductor/test_torchinductor.py  -> clean
```

A clean A/B of the full `test/inductor/test_torchinductor.py` is running:
`agent_space/mutate_to_baseline_wt` is the same commit with the identical test
file (including the two new tests) but without the lowering fix, so failures can
be attributed instead of guessed.

Note on running these: `test_aot_inductor.py` and `test_torchinductor_opinfo.py`
import sibling test modules, so the test file's directory has to be on
`sys.path`. `conda run` drops `PYTHONPATH`, so `/tmp/run_wt_novision.py` inserts
it itself. Without that the suites exit 0 having run nothing.

## Env note

`torchvision` in the `pytorch-3.12` conda env is built against a different libtorch and
raises `RuntimeError` (not `ImportError`) on import, which escapes the `try/except
ImportError` in `torch/testing/_internal/common_quantization.py:496`. Any inductor test
importing that module dies at collection. Workaround used here:
`/tmp/run_wt_novision.py`, which stubs `sys.modules["torchvision"] = None` before
delegating to `agent_space/run_wt.py`.

## Not done

Not checked whether this is already filed upstream (`gh` returns HTTP 401 in this
session). Nothing committed, amended, pushed, or published.

## Full `test_torchinductor.py` A/B (2026-09-02)

Same commit (`cdd22ade294`), same test file (both trees carry the two new
regression tests); the only difference is the `mutate_to` guard.

| | failures | errors |
|---|---|---|
| `mutate_to_baseline_wt` (no fix) | 4 | 12 |
| `mutate_to_retarget_wt` (fix) | **0** | 12 |

`comm -13 base_fails fix_fails` is empty: **no fix-only failures**.

The 4 baseline failures are exactly the two new tests x {cpu, cuda}, which
confirms both are non-vacuous and both are fixed by the guard:

```
FAIL: test_index_put_accumulate_after_scatter_mutation_{cpu,cuda}
FAIL: test_index_put_after_scatter_mutation_{cpu,cuda}
```

The 12 errors are byte-identical between the two runs and unrelated
(`sdpa_unaligned_mask*`, `sdpa_prefer_nd_tiling*`,
`scaled_dot_product_efficient_attention`, `conv_transpose_zero_size_output`,
`linalg_eig_stride_consistency`) - this build has no cuDNN/mem-efficient
attention support.

Commands:

```bash
# per tree, with $WT set to the worktree
PYTORCH_WORKTREE=$WT TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
  LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib \
  CUDA_VISIBLE_DEVICES=N conda run --no-capture-output -n pytorch-3.12 \
  python /tmp/run_wt_novision.py $WT/test/inductor/test_torchinductor.py
```
