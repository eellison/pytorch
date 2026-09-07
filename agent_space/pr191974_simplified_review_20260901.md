# Review: simplified PR #191974 (R-grouped-only nested reduction enablement)

Date: 2026-09-01

This is an AI-assisted local review document. It is not intended to be pasted
into GitHub without human review and the disclosure required by `AI_POLICY.md`.

## Snapshot

- Worktree: `/data/users/eellison/pytorch/agent_space/pr191974_simplified_wt`
  (uncommitted four-file diff: `scheduler.py` +190/-~40, `config.py` +4/-4,
  two test files)
- Commit message draft:
  `agent_space/pr191974_failure_investigation/PR191974_COMMIT_MESSAGE.md`
  (ghstack trailers preserved; AI disclosure present; test plan has literal
  commands)
- Supersedes the full-solver fix in `agent_space/pr191974_fix` (+815/-165),
  which remains parked as reference for a future X-grouped profitability
  follow-up.

## Verdict

APPROVE for push. All conditions from the prior review rounds are closed or
structurally dissolved. Independent reruns match the test plan's claims; the
only failures on this box are a known local openssl environment issue
(fingerprint below), not the change.

## What the PR now does

1. Enables `nested_reduction` by default (Config/JustKnob, env force retained).
2. Restricts nested fusion to the R-grouped topology at the candidacy gate
   (`grouped_axis is not GroupedAxis.R -> return None`, with the perf
   rationale in a comment). X-grouped candidates keep the ordinary unfused
   schedule. Rationale: X-grouping forces `min_xblock`, lost to the two-kernel
   schedule on every measured shape, and its persistent variants can exceed
   shared memory (the BERT-class CI failures).
3. Fixes the R-grouped silent-wrong-result bug (reduction-only transpose,
   max err ~414 in the minimized repro) with a narrow exact guard,
   `_r_grouped_stage_accesses_match` (scheduler.py:1013).
4. Replaces the `reads_reduction_source` name heuristic with fail-closed
   rejection: a node compatible with both REDUCED and SUB_PARENT declines the
   plan. Kills the stack's last name-based coordinate heuristic; cost is one
   conservative lost fusion in the ambiguous `group_size == factor` shape.

## Review findings

### The guard is a faithful extraction of the proven solver, not a rewrite

`_r_grouped_stage_accesses_match` expresses the fixed `[X, R]` and
`[X, R/G, G]` geometries in one symbolic frame and evaluates every node
through codegen's own `map_kernel_groups_to_node_sizes` +
`indexing_from_args` — the same machinery the deleted solver used, so the
proof is about what codegen will actually emit. Verified properties:

- Every grouped-stage consumer's loads of parent-stage outputs must exactly
  equal a parent store index (`_index_exprs_equal`, sympy-proven, fail-closed
  on unprovable).
- Non-`MemoryDep` dependencies on internal buffers decline
  (`internal_names != memory_internal_names`) — covers index_expr /
  bounds / bucketize semantic reads.
- Multi-writer buffers decline; missing reads/writes decline; `CantSplit`
  declines; unknown domains decline; X-grouping declines.
- SUB_PARENT consumers are delegated to their existing exact lane-access
  proofs, which is structurally safe in an R-only world (parent-full and
  local-full share x-major flattened order).
- Cross-frame hazards exist only parent->grouped under R-grouping (grouped-
  internal exchanges are same-frame and ordinal-preserving), which is exactly
  the edge set the guard checks. The reduction-only counterexample is caught
  whichever stage the broadcast node lands in.

### Structural wins beyond the diff size

- The saved-plans machinery is gone entirely: classification and the guard
  are recomputed fresh on every `plan_from_topology` call, including at
  codegen. This dissolves the dual audit's #1 hardening item (fusion-time
  plans consumed at codegen without re-proof) rather than guarding it.
  It is possible because domain classification is numel-based and therefore
  stable across `merge_loops`; the same-numel PARENT_FULL-vs-LOCAL ambiguity
  that forced plan caching left with the solver.
- The audit's #2 item (duplicated frame formulas) shrinks from many branches
  across two files to one frame in one function. Residual: the guard's
  1-iter-var flatten (`parent_x * group_count + group_r`) still hand-mirrors
  `construct_group_reduction_vars`; see "Residual items."

### Tests are repurposed, not deleted

- X-topology numeric tests retained with `check_fusion` flipped to
  `check_no_fusion` — a future re-enablement must consciously flip explicit
  assertions and inherits numerics coverage for the historical wrong-result
  shapes.
- X-grouped kernel-form tests (nested-kernel golden checks) deleted, which is
  correct — there is no nested kernel to pin.
- New reject tests for both transposed counterexamples:
  `test_reject_transposed_parent_reduction_broadcast` and
  `test_reject_transposed_grouped_pointwise_producer` (the realized
  grouped-producer variant that exercises the guard's consumer-side check).

### Commit message

Accurate against the diff: root cause explained (equal element counts plus
reshape compatibility let a transposed consumer read by position), the chosen
path justified (X-grouping restricted with perf evidence), byte-identical
retained kernels claimed with A/B timings (2.180 vs 2.181 us block quant,
2.302 vs 2.284 us swizzle), trailers preserved, AI assistance disclosed.

## Independent verification (this box)

- Nested suite: 401 ran, 8 skipped, 1 error =
  `NestedReductionAOTITest.test_rmsnorm_block_amax`, the known environment
  casualty (see fingerprint). Effectively 400/400.
- Scheduler suite: 142 ran, OK, 6 skipped.
- Config suite: 3 errors, all the same environment casualty (confirmed
  `_get_file_checksum` frames in each traceback); these tests exercise
  cpp-wrapper paths that hit precompiled-header hashing.

Environment casualty fingerprint (recurring on this box, NOT the change):
`torch._inductor.exc.InductorError: IndexError: list index out of range`
raised from `codecache._get_file_checksum`, because under `conda run`
`/usr/bin/openssl` shadows the conda env's openssl and fails with
`symbol lookup error: BIO_new_dgram_sctp` against the conda `libcrypto`
picked up via `LD_LIBRARY_PATH`, yielding empty stdout. Any suite run using
the worktree-overlay runner on this machine will show these; suspect the
environment before the code.

## Residual items (none block the push)

1. Add a cross-reference comment tying the guard's frame construction
   (especially the 1-iter-var flatten branch) to
   `construct_group_reduction_vars` in simd.py, so a one-sided edit is
   findable in review. This is the last residue of the audit's
   frame-duplication finding.
2. X-grouped codegen machinery in the earlier stack commits is now
   gated-dead from this PR's perspective. Deleting it belongs to a stack
   cleanup, not this commit; track it.
3. The parked full solver (worktree `pr191974_fix`, design in
   `FINDINGS.md` "Root cause 2" and the coordinate-provenance audit section)
   is reviewed and fuzzed IP for any future X-grouped re-enablement with
   profitability gating (e.g., outer-reduction parents only). Do not rewrite
   it from scratch later.
4. FINDINGS.md's flattened grouped-axis coefficient-matching seam remains a
   disclosed follow-up; unchanged by this PR.
5. Standing invariant, twice paid for, worth keeping in review muscle memory:
   total-size/shape/name evidence may nominate candidates or decline fusion;
   it may never select coordinate semantics. Both #191974 bugs and both
   review-found bugs were violations of exactly this rule.
