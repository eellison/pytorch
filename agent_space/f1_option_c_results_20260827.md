# F1 Option C results

## Decision

Use the planner's exhaustive `SubParentAccessRelation` records to build a
small per-buffer codegen contract. Keep exact dependency authorization in the
fusion planner, but do not reconstruct `MemoryDep` objects from the active FX
interpreter during codegen.

The specialized replay wrapper:

1. captures optional external sources from loads and required internal sources
   from stores;
2. keeps equivalent external shapes as alternatives, while internal replay
   writes use the latest value;
3. derives a lane from the actual replay load index and checks that it belongs
   to the planner-approved lane set;
4. reloads an optional missing source through a local physical-load path that
   deliberately bypasses name-only `store_cache` forwarding; and
5. raises if a required source is missing or no longer live.

This is still name-keyed at codegen. Its soundness relies on the planner proving
every tracked access and on codegen rebuilding the final relation plan. Exact
fusion authorization remains per access; the name-keyed collapse is only a
codegen consequence of that exhaustive proof.

## Containment

The production delta from the current F1 snapshot touches only
`torch/_inductor/codegen/simd.py`. It adds no state or behavior to generic CSE,
`LoopBody`, dependency extraction, scheduler APIs, or ordinary Triton codegen.
The resolver is constructed only for a sub-parent stage. A neutral full-
resolution nested-reduction test asserts that ordinary nested codegen does not
construct it.

## Defect found during validation

The first prototype retained every internal store value for a name. A `(4, 3)`
packing node is replayed three times, so later replays incorrectly selected the
first replay's packed byte. This produced a 66.7% mismatch in the preshuffled
MXFP6 test.

Internal sources now use last-writer-wins per replay and evict the prior
materialization. External loads still retain multiple equivalent live shapes.
Both persistent and looped preshuffled tests pass after this change.

## Evidence

### Full suites

```text
test/inductor/test_inductor_scheduler.py: 124 tests, OK, 6 skipped
test/inductor/test_nested_reduction.py:   399 tests, OK, 8 skipped
```

### Differential fuzz

Option A and Option C were compiled and executed independently with caches
disabled:

```text
62/62 generated cases
98 runtime invocations
70 exact tensors and 175 floating tensors compared
maximum absolute error: 0.0
staged kernels: 54 vs 54
generated kernels: 92 vs 92
no per-case differences in conversions, splits, or broadcasts
```

Coverage includes persistent and looped reductions, factor 2 and factor 4,
bf16/fp16/fp32, FP8 and integer casts, tails, masked fallback, and dynamic
reduction and batch shapes.

### Protected kernel forms

All ten generated-kernel hashes match Option A exactly:

```text
factor2 persistent       98b111d6f7798393c16439e9ed2c770ad98ddb8dee6cb731cb2d1640a75bc59c
mxfp6 persistent         b0fe88ce8d8fb83adc55d9d9d9e7c4c16bc360160bd14db3afe6343b42b57a79
internal persistent      2efc561292dd1ba54ae8576515638189fcf7671122fcafbb64c0f987f3e9c887
factor2 looped           6edb0230b628013056895dcbddd3643228608bb4be95d40250de15d7ec906961
mxfp6 looped             fcd5bda8ac08084965dfc92ad7142b0dadde8f3615469f3c4a8b28a44f6a976e
internal looped          fb769f5dca3089dce0684cef2b3f4fcc6a5d9c232d736570d89aedd434be52ba
reduced broadcast        a907b3fbb5d83fdf8a61c7f292a4729d26aa01d4c8f9c9ed61f8562bb83b8258
scale swizzle            7f72ab3924703863f8e92edcf6849f6c70f9270e31f696cde87fd115571b48e9
preshuffle               710d6b80635a0add97076be321f2c19dd82301319f8cc85224f6421c99871497
dcn preshuffle           f3a02e5c076058f991b61a2ebb5bb8a54fb5d7c0e8f08f110018c2f99a869d19
```

### Size and complexity versus Option A

```text
simd runtime delta:       123 insertions, 139 deletions (-16 lines)
scheduler documentation: 5 insertions, 4 deletions (+1 line)
total production delta:  128 insertions, 143 deletions (-15 lines)
aggregate AST LOC:         16525 -> 16512
aggregate cyclomatic CC:  3568 -> 3567
changed-function AST LOC: 229 -> 216
changed-function CC:      61 -> 60
maximum nesting:          unchanged at 8 globally / 3 in changed functions
```

### Lint

`spin lint` reports no source lint findings. Its overall exit is nonzero because
the isolated scratch worktree has no `build/` directory for
`CLANGTIDY_EXECUTORCH_COMPATIBILITY`. `py_compile` and `git diff --check` pass.

## F2a compatibility

A separate Option C + F2a worktree validates the handoff. The F2a production
delta is `+69/-3`, entirely in `simd.py`; it preserves an unmasked, direct
group-width source through `resolve_sources`, performs the narrowing cast, then
materializes before an ordinary operation or store. It changes no scheduler,
generic CSE, `LoopBody`, or Triton codegen API.

Focused resolver and kernel-form tests pass. The same 62-case, 98-invocation
differential corpus has exact numerics and unchanged fusion, kernel, graph, and
conversion counts. This satisfies the compatibility gate without changing the
F1 design.
