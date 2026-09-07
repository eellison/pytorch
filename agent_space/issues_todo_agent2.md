# Nested Reduction Notes (Agent 2)

Scope for this pass:

- Repo: `/data/users/eellison/pytorch`
- Branch context: current core nested-reduction work on main repo
- Explicitly **not** reviewing NVFP4 / half-resolution / pass-3 here
- Explicitly **not** reviewing unrelated `graph.py` / `ir.py` realization work

## Current branch reality

- `torch/_inductor/codegen/simd.py` is already on the newer family-based core design:
  - `_DerivedIterationFamily`
  - `_GroupReductionLayout`
  - `_GroupedReductionOpsHandler`
  - `_PointwiseRemapHandler`
- It is **not** the older pass2/pass3 shape.
- There are no current `nvfp4` / `pass3` / `half_resolution` code paths in `simd.py`.
- The current test file explicitly checks that the core kernels do **not** use `tl.split(`.
- Every fused core test now has an exact structural assertion for:
  - kernel count
  - allocation/deallocation count
  - input load count
  - output store count

That means older concerns about `_Pass2Coords`, `_Pass2OpsHandler`, `_RemappedOpsHandler`,
or half-resolution legality are stale for this branch and should not drive review.

## High-confidence current observations

### 1. No clear core nested-reduction correctness blocker from source review

Files:

- `torch/_inductor/codegen/simd.py`
- `torch/_inductor/scheduler.py`
- `test/inductor/test_nested_reduction.py`

Current read:

- The scheduler computes `small_dim_in_r` once on `FusedNestedReductions`.
- Codegen consumes that classification rather than re-deriving it.
- Full-resolution epilogues are explicitly blocked for `small_dim_in_x`, and the tests
  cover those negative cases.
- `B=1` reduced-output `small_dim_in_x` is intended to work; the current tests cover it.

Assessment:

- I do not currently see an obvious source-level correctness bug in the core-only path.
- The remaining concerns are mostly test hygiene and complexity concentration, not an
  identified miscompile.

### 2. Singleton-numel canonicalization is global, but mathematically safe

File:

- `torch/_inductor/codegen/simd.py`

Current behavior:

- `SIMDKernel.indexing()` substitutes any active range-tree symbol with `0` when that
  tree's `numel` is statically known to be `1`.

Why it matters:

- This is broader than nested reduction.
- It is the kind of change that reviewers will notice because it affects all SIMD kernels.

Assessment:

- The transformation is mathematically exact.
- I would treat it as acceptable in this PR, but it is worth calling out explicitly in
  review as a generic canonicalization, not a nested-reduction-only hack.

## Test / validation concerns

### 3. The structural kernel-capture helpers are still heavier than they need to be

File:

- `test/inductor/test_nested_reduction.py`

Current behavior:

- The file now contains many pattern-specific capture helpers:
  - `_capture_amax_kernel_sources`
  - `_capture_fullres_kernel_sources`
  - `_capture_pattern1_kernel_sources`
  - `_capture_pattern2_kernel_sources`
  - dynamic variants
- These are useful, but they add a lot of internals-oriented testing code to the main
  feature test file.

Why it matters:

- This makes the main nested-reduction test file larger and more review-heavy than needed.
- It is harder to distinguish feature behavior tests from emitted-kernel structure tests.

Suggested follow-up:

- Move the source-capture / `FileCheck` structure assertions into an internals-oriented
  file such as `test/inductor/test_nested_reduction_internals.py`, or
- at least consolidate the helper surface to reduce duplication.

### 4. Kernel selection in `run_and_get_code()` helpers is still somewhat brittle

File:

- `test/inductor/test_nested_reduction.py`

Current behavior:

- `_run_and_capture_source_bundle()` filters Triton kernels by a generic
  `kernel_signature` substring, often `"triton_per_fused"`.
- Single-kernel helpers now assert that there is exactly one matching fused
  kernel; the intentionally non-fused full-resolution-negative helper asserts
  there are exactly two.

Assessment:

- The earlier ambiguity is fixed for the current core patterns.
- A future broadening of coverage should keep the same exact-match discipline.

### 5. Structural form checks compile patterns more than once

File:

- `test/inductor/test_nested_reduction.py`

Current behavior:

- Numeric/fusion tests compile once for actual execution.
- Kernel-form assertions compile again through the capture helper.

Why it matters:

- This is not a correctness bug.
- It does make the test file slower and noisier than necessary.

Suggested follow-up:

- If test cost becomes a problem, cache the captured code per test method or move the
  structural checks into a separate internals test class.

## Design / code-shape notes

### 6. `_GroupReductionLayout` is still carrying a lot of semantics in codegen

File:

- `torch/_inductor/codegen/simd.py`

Current role:

- grouped axis selection
- reshape geometry
- reduction axis
- family construction
- broadcast lifting rules
- flattened index reconstruction

Assessment:

- This is the main remaining reason `simd.py` still feels large.
- It is not a local bug.
- It is the clearest sign that grouped/block-local reduction semantics still live too low
  in the stack.

Suggested follow-up:

- Do not churn this branch further trying to split `_GroupReductionLayout` into more local
  helpers.
- If the team wants a cleaner architecture, the next step is a scheduler/IR-level
  `BlockLocalReduction` or `NestedReductionPlan` design, not more local refactoring here.

### 7. Full-resolution epilogues are intentionally limited to `small_dim_in_r`

Files:

- `torch/_inductor/scheduler.py`
- `torch/_inductor/codegen/simd.py`
- `test/inductor/test_nested_reduction.py`

Current behavior:

- Scheduler only admits full-resolution epilogues when `small_dim_in_r`.
- Codegen hard-errors if that invariant is violated.
- Tests include negative cases for `small_dim_in_x`.

Assessment:

- This is the correct shape for the current core branch.
- It is worth highlighting in review because it is a deliberate capability boundary, not
  a missing optimization.

## Suggested next validations

### 8. Do one emitted-kernel pass for the current core branch

Patterns worth checking:

- grouped `amax`
- full-resolution fp8/blockwise quant
- `B=1` reduced-output `small_dim_in_x`

Why:

- The earlier detailed output-code reviews were often done on saved side branches.
- For this current main-repo branch, one targeted emitted-kernel sanity pass would be the
  most useful extra confidence check.

What to verify:

- grouped `amax`: one main input load, one weight load, one store
- full-resolution epilogue: grouped reduction plus broadcast lift and two stores
- `B=1` reduced-output case: no obviously degenerate duplicate-load form

Status:

- The source-level test assertions now encode this contract.
- Dynamic tests are now explicitly split by outer-reduction mode:
  - `NestedReductionTest` forces persistent outer reduction
  - `NestedReductionNonPersistentTest` forces non-persistent outer reduction
  This is intentional. The outer reduction can validly run in either mode;
  the grouped second stage stays local either way.
- A fresh emitted-kernel sanity pass is still worthwhile, but it is now
  validating runtime output against an already-explicit contract rather than
  discovering the contract from scratch.

## Recommendation

Current recommendation for review:

1. Treat this as a core-only nested-reduction branch.
2. Ignore stale NVFP4 / pass-3 discussions when reviewing this tree.
3. Focus review on:
   - `torch/_inductor/scheduler.py`
   - `torch/_inductor/codegen/simd.py`
   - `test/inductor/test_nested_reduction.py`
4. The likely review comments are about:
   - complexity concentration in `_GroupReductionLayout`
   - global singleton canonicalization scope
   - structural test-helper size / brittleness
5. I do **not** currently have a source-level core correctness blocker to flag.
