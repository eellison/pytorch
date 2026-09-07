# Stack re-review: sub-parent reduction epilogues (2026-08-19)

Delta review of the chain cut 2026-08-19 07:53-07:56, against the previously
reviewed and fixed state (`5c77934b4c5` chain; see `stack_review_20260818.md`
and its two addenda for the full history). This is a fidelity-plus-delta
review, not a from-scratch audit: commit 1 is byte-identical to the reviewed
state, so only the amended material was re-examined.

Chain (base `41d7be3f48d`):

| # | commit | PR | delta vs reviewed chain |
|---|---|---|---|
| 1 | `86038d6e1cd` | #190594 | byte-identical (`range-diff: =`) |
| 2 | `9e1c3dce860` | #190595 | +237/-1: the group-projection change |
| 3 | `8ce0cf8de5c` | #191775 | mechanical ripple only |
| 4 | `deb04e6f59a` | #190596 | zero content change (context drift) |
| 5 | `ba5bc861716` | #191974 | +1/-3: dedup extended to new sites |

## Amend fidelity (static, verified by me)

- **Commit 2 delta is byte-identical to the validated
  `nvfp4_group_projection_draft_20260818/pr190595_isolated.patch`** (`diff`
  of `git diff 9f54c3b9731 9e1c3dce860` against the patch: empty). That patch
  was independently validated on 2026-08-18 (342/2 isolated suite, 18/18
  focused, perf 32.64us vs FlashInfer 36.67 at 4096x8192, bitwise-equal
  outputs).
- **Tip content equals old tip + the validated tip patch**, modulo one
  neutral difference: the amend places the `group_broadcast` keyword
  parameter last in three signatures instead of mid-list (all call sites pass
  it by keyword; content-level tree diff shows only these 5 moved lines).
- **The #191775 ripple is exactly the required mechanical adaptation**:
  isolated commit 2 constructs `_GroupInvariantBroadcast` inline at its
  single call site; commit 3 hoists it to a variable when the
  `output_groups` loop multiplies call sites. Each commit is
  self-consistent; bisectability preserved.
- **The #190596 ripple has zero +/- content changes** (hunk-position drift
  only).
- **The #191974 extension dedupes the two isolation-artifact sites** (local
  `import torch.nn.functional as F`, inline PTX literal ->
  `E2M1X2_PACK_ASM`). The original review's "leftover local import" nit is
  now fully resolved: the tip test file has exactly one module-level import.
- **All prior-review fix mechanisms confirmed present at the tip** by grep:
  the looped-CONTIGUOUS persistence gate (C1), the strict-reduction
  contiguous decline (C2), the reads-only classification tiebreak (C3),
  `lane_source_sizes` planner/codegen symmetry (C4), the Identity strips
  (C5), `disable_multi_kernel` (P1), indexing-schedule metrics (P2),
  `store_buffer_counts`, and `_PROJECTION_BARRIERS`.
- Handoff docs kept pace: `READ_ME_FIRST.md` carries the new hashes and
  records `with-proxy lintrunner -a` passing; `FOLLOWUPS.md` gained the
  "give inlined values explicit cross-domain identity" section, which
  faithfully states the projection design's residual trade (identity by
  replay + CSE) and names the single-FP8-conversion kernel-form tests as the
  tripwire.

## Dynamic verification (all green, exact expected counts)

Run through the full-package overlay harness (`agent_space/run_wt.py`),
Triton 3.8.0 pin, caches disabled, worktrees per commit:

| Run | Commit | Result |
|---|---|---|
| test_nested_reduction.py full | tip | 450 passed / 12 skipped |
| test_inductor_scheduler.py | tip | 88 passed / 6 skipped |
| -k nvfp4 | tip, c2, c3, c4 | 18 passed at every commit |
| test_nested_reduction.py full | isolated #190595 | 342 passed / 2 skipped |
| C1 regression repro (looped external CONTIGUOUS + internal source) | tip | staged=1, bitwise match (bug stays fixed) |
| MXFP6 stack dim=-2 repro | tip | both dims staged=1, pack bitwise-equal |
| py_compile on touched files | tip | clean |

Mid-stack bisectability is directly confirmed by the per-commit nvfp4 runs.

## Findings

1. **[action required before/at resubmit] PR #190595's description does not
   cover the projection change.** The amended commit message is the original
   #190595 text; ghstack message edits are no-ops, so the GitHub description
   must be updated by hand (`ghstack -u` after editing, or edit on GitHub).
   It should mention: the group-invariant projection (deferred broadcast,
   scalar math at group resolution, projection barriers), the visible golden
   change to `test_nvfp4_inline_asm_kernel_form` (uint8-bitcast forwarding
   -> broadcast of the converted value; strictly less emitted work), the
   perf table (32.64us vs FlashInfer 36.67 at 4096x8192, both NVFP4 and
   MXFP4), and the config-dependent-numerics disposition (cross-variant
   nibble differences on unmodified code are reduction-order effects under
   different tuned configs; scales bitwise-equal; matched configs bitwise).
2. **[non-blocking, carried from 2026-08-18] `_PROJECTION_BARRIERS` has no
   drift tripwire.** A unit test asserting every barrier name is a real
   `OpsHandler` method, plus a one-line pointer in `OpsHandler`'s docstring
   for authors of new positional/stateful/subgraph ops, remains the only
   enforcement the fail-open barrier set lacks. Third and final carry; also
   suitable as a FOLLOWUPS entry if deliberately declined.
3. **[out of chain] The main working tree still holds uncommitted WIP**
   (`torch/_higher_order_ops/flex_gemm.py`, a scheduler
   `_sub_parent_has_duplicated_cross_domain_origins` guard, test edits) on
   the stale `94367fffd77` checkout. Not part of this chain and not
   reviewed here; decide its fate (stack it or drop it) so it does not rot.
4. Cosmetic only: the kwarg-ordering difference vs the reviewed patch text
   (see fidelity section); no action needed.

## Verdict

**Approve. The stack is land-ready.** Every correctness finding from the
2026-08-18 review is fixed and regression-pinned at this chain; the
projection change was amended byte-faithfully to its validated form with
correct per-commit adaptations; all suites pass at the tip and at every
intermediate commit; lint is recorded passing. The only pre-resubmit action
is the #190595 PR-description update (finding 1); findings 2-3 are
non-blocking hygiene.
