# AI-Assisted Commit Review: `c0787d4c3985e99ff8efc9d145cc1be454598f6b`

This is local AI-generated review material. It has not been posted to GitHub. If
used in a GitHub review, keep the review text quoted and add the reviewer's own
commentary as required by `AI_POLICY.md`.

## Summary

This commit adds factor-2 interleaved sub-parent epilogue planning and staged
Triton codegen for standalone reduction kernels, enabling the standalone NVFP4
packing pattern. Request changes: three existing Inductor configuration paths
bypass or discard the staged codegen, two of them abort compilation, and the
recorded Test Plan does not run any of the new tests.

## Infrastructure

### 1. Blocking: `benchmark_fusion=True` aborts compilation

`torch/_inductor/codegen/simd.py:5538-5547` generates benchmark candidates with
the generic `generate_node_schedule`. That bypasses the sub-parent dispatch in
`_codegen_nodes` at `simd.py:3546-3551`. The factor-2 epilogue has group
`(numel * rnumel / 2, 1)`, which the generic scheduler cannot represent, so it
raises the `unexpected group` `NotImplementedError` at `simd.py:2820-2823`.
The exception escapes fusion benchmarking and aborts the compile.

Minimal configuration reproducer around the graph in
`test/inductor/test_nested_reduction.py:851-860`:

```python
with torch._inductor.config.patch(
    {
        "triton.nested_reduction": True,
        "loop_ordering_after_fusion": True,
        "fx_graph_cache": False,
        "benchmark_fusion": True,
    }
):
    torch.compile(f, fullgraph=True)(x)
```

Observed failure:

```text
InductorError: NotImplementedError:
unexpected group: (2048, 16) != (16384, 1)
```

Route benchmark source generation through the sub-parent staged emitter, or
explicitly decline benchmark-based fusion for a group with a sub-parent plan.
Add a `benchmark_fusion=True` regression.

### 2. Blocking: reduction combo kernels abort final codegen

`SIMDScheduling.generate_combo_kernel_code` also expands the fused sub-parent
node through generic scheduling at `simd.py:4561-4565`. An ordinary
`FusedSchedulerNode` with a standalone sub-parent plan is not excluded by the
existing combo filtering, so combining it with another reduction reaches the
same `unexpected group` exception during final codegen.

This reproduces with the standalone factor-2 graph plus an independent
same-shaped reduction and:

```python
with torch._inductor.config.patch(
    {
        "triton.nested_reduction": True,
        "loop_ordering_after_fusion": True,
        "fx_graph_cache": False,
        "combo_kernels": True,
        "combo_kernels_pointwise_only": False,
    }
):
    torch.compile(f, fullgraph=True)(x, z)
```

Either filter nodes for which `_find_sub_parent_epilogue_plan` succeeds out of
combo grouping, or teach combo generation to preserve the staged schedule. Add
a regression with an independent reduction so the combo path is exercised.

### 3. `triton.multi_kernel` silently discards kernel choices

`_codegen_reduction_with_sub_parent_epilogue` takes `[0]` from
`create_kernel_choices` at `simd.py:3600-3603`. Triton may return persistent and
non-persistent choices and sorts the non-persistent choice first at
`torch/_inductor/codegen/triton.py:8385-8391`. Unlike the generic path at
`simd.py:3795-3821`, the special path neither emits all choices nor constructs a
`MultiKernel`.

For this graph, `triton.multi_kernel=1/2/3` creates two choices but emits only
the looped kernel and no multi-kernel wrapper. Mode 1 therefore loses tuning,
and mode 2 violates its documented force-persistent contract. Generate and
wrap all choices using the normal multi-kernel pattern, or explicitly disable
multi-kernel construction while still honoring modes 2 and 3. Cover all four
config values.

## Code Quality

### 4. Remove staged helpers that have no caller in this commit

- `_GroupedReductionLayout._broadcast_value_to_axis_resolution` at
  `simd.py:2031-2053` is unused. If called with the sub-parent family, it emits
  `nested_R0_REDUCED_BLOCK`, but `make_sub_parent_family` at `simd.py:1917-1930`
  does not attach the named constant that defines that symbol. Delete it here
  and add it in the first commit that supplies the required constants.
- `_materialize_sub_parent_source_load` at `simd.py:2231-2257` is unused and
  duplicates the active `_SubParentSourceLoadMaterializer.load` path.
- `TritonKernel._emit_recursive_split` at
  `torch/_inductor/codegen/triton.py:6154-6190` has no entry-point caller in
  this commit; its only references are its recursive self-calls. Move the
  factor-generic machinery to the later factor-general commit.

### 5. `reduction_nodes` does not mean reduction nodes

`NestedReduction.SubParentEpiloguePlan.reduction_nodes` at
`torch/_inductor/scheduler.py:603` is populated at `scheduler.py:681` with every
non-epilogue node, including pointwise prologues and reduced-output siblings.
Codegen intentionally sends the whole tuple to `generate_node_schedule`, so
rename it to `parent_nodes` or `non_epilogue_nodes` before a caller starts
assuming every member is a reduction.

## Testing

### 6. The recorded Test Plan is false-green

Two listed selectors do not exist at this commit and exit successfully after
running zero tests:

```text
nested_reduction_parent_half_domain
producer_consumer_rmsnorm_interleaved_pair_epilogue
```

The other three listed selectors predate this commit. Consequently, the Test
Plan records no command that selects the new standalone tests at
`test_nested_reduction.py:762`, `:851`, `:1771`, or `:1787`. Replace it with the
literal commands that were run for the tests introduced by this commit.

### 7. The forced-looped tests do not pin multiple R-loop iterations

The new capture helpers hard-code `G=16` at
`test_nested_reduction.py:1372` and `:1407`. The non-persistent class forces a
looped kernel form, but the assertions only require `min_rblock=2`; they do not
choose or assert `R0_BLOCK < G`. Thus a one-iteration loop satisfies the test
while the second, looped epilogue traversal remains unpinned.

Add a large-group forced-looped case with a deterministic small `R0_BLOCK`,
then assert numerics and the two actual R loops: one for the reduction and one
for the sub-parent epilogue.

### 8. New legality and dtype branches lack direct coverage

- The float8-specific split and broadcast branches at
  `triton.py:6211-6224` and `:6240-6247` are not exercised by the NVFP4 tests:
  those split BF16 input and convert the FP8 scale to float before use.
- The planner's alias/mutation rejection at `scheduler.py:620-621` has no
  fallback regression; the added rejection tests cover readers and ambiguous
  loads instead.
- Existing dynamic tests cover older nested-reduction paths. The standalone
  sub-parent test at `test_nested_reduction.py:851-864` is static, leaving
  dynamic X behavior unpinned.

Add focused FP8 split/broadcast, mutation fallback, and dynamic-batch tests.

### 9. The B=3 comment names a nonexistent XBLOCK floor

`test_nested_reduction.py:848-849` says B=3 tests an XBLOCK floor, while the
implementation explicitly imposes no `min_xblock` at `simd.py:3608-3617` and
the kernel-form tests assert that absence. Describe it as non-power-of-two X
coverage instead.

## Recommendation

**Request Changes**

The benchmark-fusion and combo-kernel failures are compile aborts in supported
Inductor configurations. The multi-kernel path violates its selection contract,
and the current Test Plan can report success without running any new feature
test.
