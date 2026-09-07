---
name: nested-reduction-stack
description: "Current state of the sub-parent/nested-reduction ghstack (structure, config default, review outcome, where docs live)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 38bdc619-dab6-4792-83b9-2600e9efd2fe
  modified: 2026-09-01T18:43:13.121Z
---

The norm->quantization fusion work is a 5-commit ghstack (PRs #190594, #190595,
#191775, #190596, #191974), NOT the old 2-commit `nested_reduction_backup`
layout. Authoritative per-stack docs live in `agent_space/READ_ME_FIRST.md`
plus per-PR guides (`agent_space/pr19XXXX_*.md`) and `agent_space/FOLLOWUPS.md`
— read those instead of trusting remembered structure; the stack gets rebased
and hashes change.

`config.triton.nested_reduction` is LANDED in origin/main (verified 2026-09-03) with
`default=True`, `justknob="pytorch/inductor:nested_reduction"`,
`env_name_force="TORCHINDUCTOR_NESTED_REDUCTION"`. The OSS default is ON; internal
rollout is a JK ramp, not a code change. (Earlier note said "defaults to OFF" --
that was true only while the stack was in flight.)

When testing a worktree's inductor changes against the installed editable
torch, use `agent_space/run_wt.py` (whole-package overlay), NOT
`run_stochastic_worktree_test.py` — the older harness overlays only an
explicit module list, so `torch._inductor.choices`, `dependencies`, etc.
silently resolve to the main checkout (which is often being actively edited).

The local `wip_quant` checkout has at times been a stale replay of the stack
(different hashes, same subjects, neither an ancestor of the other). Always
compare `git log` subjects+dates against READ_ME_FIRST's table before assuming
the working tree matches the submitted stack.

Dim1 (columnwise) MX/NVFP4 cast investigation (2026-08-21):
`agent_space/dim1_mx_nvfp4_cast_investigation_20260821.md`. Key durable facts:
the staged stack never engages for dim1 (standalone lane proof requires lanes
on the trailing R extent; dim1 lanes sit at stride K); MXFP8 dim1 fuses to one
generic kernel at the current tip and BEATS torchao's dedicated dim1 Triton
kernel at all shapes, but only with coordinate_descent_tuning at large shapes
(default config is 2.4x slower); NVFP4 dim1 stays 2 kernels (double input
read) and would need strided-lane planner+codegen support, a follow-up outside
the landing stack.

Combined dim0+dim1 NVFP4 kernel (2026-08-21): handwritten Triton prototype in
agent_space/nvfp4_dual_dim_kernel.py (writeup
agent_space/nvfp4_combined_dim0_dim1_20260821.md, worktree
agent_space/nvfp4_dual_dim_wt at 3f62ae27baa). One kernel/one load, four
outputs, bitwise-equal to compiled references and torchao pure-torch;
1.8-1.9x vs best status quo at all shapes (95.1us vs 169-179us at
16384x7168); NOT memory-bound (L1/compute mixed, ~80-90us ceiling; TMA wrong
lever). Best config 16x256 w2 (64-128 elems/thread — confirms the
elements-per-thread band rule). dim1 transposed layout costs 1.4x vs
row-major; ship rm if wgrad can consume it. Per-group reciprocal+multiply
worth 20%, bitwise-identical. Author decision (2026-08-22): build Inductor CODEGEN support as GENERAL
capabilities, not a one-kernel template (triton_op HOP route rejected).
The combined dual-dim kernel must emerge from ordinary fusion so producers
fuse in and other group sizes/dtypes (MX 32-groups, MXFP6 rates) are
covered. Decomposes into: (a) amax
support in mix-order reduction (allowlist at scheduler.py can_fuse is
sum/prod only), (b) mix-order composing with FusedStagedReduction siblings,
(c) strided-lane sub-parent proof+codegen (the #190594-sized item, required
for any single-kernel dim1 pack), (d) blockwise-2D tile-local schedule
(matches the proven-fast kernel shape; mix-order's cross-CTA split
accumulation is pure overhead for 16-local groups). Probe results
(2026-08-22, addendum in nvfp4_combined_dim0_dim1_20260821.md, probe
agent_space/probe_mix_order_decline.py): for the dual-quant amax pair the
FIRST decline is has_mix_reduction_orders (both groups are (M*K/16, 16) -
equal, never reversed), NOT the allowlist; FusedStagedReduction nodes DO
reach can_fuse and die at the same shape check, so (b) is moot for this
goal. (a) is genuinely SMALL: a one-line allowlist edit (uncommitted in the
nvfp4_dual_dim_wt worktree, marked EXPERIMENT) fused the canonical
row+col amax pair with ZERO codegen changes - 3 kernels 152.6us -> 1 kernel
69.7us (2.19x), bitwise-exact; kernel = RSPLIT_SIZE=16 rows/program,
29.4MB workspace, host-side final amax. Mix-order is structurally
inapplicable to group-local dim1 (its contract computes ncol-sized outputs
by cross-chunk combine; dim1 group members are strided K/16 in the
flattened x domain). Sequencing: land (a) opportunistically, skip (b),
build (d) directly with the handwritten kernel as blueprint.
General-capability reframe (2026-08-22, addendum 2 in the writeup): with an
RMSNorm producer, nested-ON dual quant = 3 kernels 267us; Inductor never
materializes normed x (recompute-from-stats, 64KB stats buffer) - the waste
is purely the 3x raw-x read. Producer-fusion critical path is (d2): the
BAND-PERSISTENT form of the tile-local contract (band = lcm of group
extents x full K, looped; = the staged stack with row-band widened from 1
to 16/32 and grouped axis allowed on Y - the case
test_producer_consumer_rejects_sub_parent_grouped_axis_x pins as rejected).
(d2) subsumes (c)'s in-band lanes; (c)/(d1) top out at 2 kernels via
recompute; (a)/(b) never touch the norm schedule. MXFP8 needs no lane
machinery (full-res); NVFP4/MXFP6 reuse staged rate machinery on Y.
Design proposal rev 2 (2026-08-22, centered on band-persistent d2):
agent_space/blockwise2d_design_proposal.md (awaiting author review). KEY
CODE FACT found while drafting: the nested stack ALREADY emits X-grouped
reduction STAGES (grouped-axis discovery classifies [X/G, R] reduce G as
GroupedAxis.X ~scheduler.py:1846, and
test_producer_consumer_rejects_sub_parent_grouped_axis_x asserts
codegen_nested_reduction==1 with kernels==2) - only the X-lane packing
EPILOGUE is rejected, by the single guard at _nested_sub_parent_rate
scheduler.py:946 (grouped_axis is not GroupedAxis.R). The planner change =
delete that guard + transposed-view lane proof + XBLOCK % group_size
legality; review anchor = flipping that test to kernels==1. Primitives: N1
non-R grouped family (exists on X, new on Y), N2 non-R lane-split stores
(transposed-view split, subsumes strided-lane item, standalone-usable so
single-dim dim1 is the degenerate case). Identity:
FusedBlockwiseReductions(FusedStagedReduction) for dual-family only (2D
factoring not rediscoverable post-merge_loops); single-family keeps existing
identities. PRs: B1 guard relaxation+N2 (test flip + standalone dim1 2->1
kernels <=~70us), B2 dual-family recognition+identity+tiled form (MXFP8
slice first; NVFP4 <=~105us kernel-only vs handwritten 83.7), B3 producer
banded dual (needs F1; rms_dual_quant 267.2 -> 120-150us band), B4
t-layouts/dynamic; PR-S mix-order amax on the side.

F1/F2 + replay review (2026-08-22): consolidated in
agent_space/projection_layers_review_20260822.md; track artifacts in
agent_space/rev_8bit|rev_f1|rev_replay|rev_evidence/. Verdicts: F1 and F2
correctness-clean under attack; the 8-bit frontier is placement-only (gate
misprediction = perf-only by construction; pow2-autotune arm rests on
asserted R0_BLOCK caps incl. CD tuner). ONE P0: the #190596 replay's
internal_source_names migration dropped the lane-layout filter (replay
simd.py:4303) -> skipped pass-boundary flush -> looped epilogue references
the post-loop accumulator; 6 suite tests fail; one-line fix proven in
agent_space/rev_replay/replay_fixed_tree; masked by the doc's narrow -k
validation. P1s: fail-open _PROJECTION_BARRIERS is an unmet explicit
ARCH_REVIEW requirement (needs OP_NAMES completeness test; concrete gap:
store_reduction is not a barrier and fail-open there = wrong store, not
perf); standalone-path
reduced outputs still ride name-keyed store_cache (README overclaims; next
layer must not assume index-checked loads); record_store lacks an
atomic-mode gate. The two kernel_num_gb "known failures" are a preload-
runner artifact (pass under run_wt.py full overlay). Bitwise-fuzz caveat:
reference fusion boundaries realize bf16 intermediates, so boundary-shifting
fuzz cases can see benign one-step fp8 divergence.

#190595 round-2 review (2026-08-24, at b55adf52d4d + lazy-materialization
delta): agent_space/pr190595_round2_review_20260824.md. ONE PROVEN bug:
emit_split_via_reshape/emit_broadcast_via_reshape drop mask_vars ->
unmasked indirect indexing on forwarded lane/broadcast values (OOB gather
from tail-lane fill garbage; repro rev195_cg/attack7_oob.py; class
PRE-EXISTS in landed #190594 standalone, attack8). Land fix = plan-time
reject indirect_indexing epilogues in the shared proof path; proper fix =
propagate DERIVED-tree mask vars; F2's _copy_masks copies SOURCE masks onto
lane-shaped results - same class one layer up, needs audit. Lazy
materialization delta (must_forward from store_buffer_names, sweep deleted)
reviewed + APPROVED, suites green. Other pre-resubmit items: commit-message
materialization sentence, nested cpp_wrapper test (behavior verified
working), NVFP4 conversion-count pin. Stack follow-ups: loop-mutation
tracker misses wrapper node1/node2 rollback; nested tiling-gate mismatch
(standalone force-2D fix not mirrored); ~5 planner invocations per fusion.

Mask-propagation fix + #191775 rebase (2026-08-24): the indirect-indexing
blocker was fixed the RIGHT way (Option B, 220 lines): mask_vars_for_shape /
set_value_masks assign the derived family's shape-compatible masks at
materialization; approved in-conversation, patch
agent_space/pr190595_mask_propagation_review.patch (bundles the approved
lazy-materialization delta + the NVFP4 conversion-count pin). #191775
rebased onto it (85d18ee9e06 on a2f664ecf57): approved,
agent_space/pr191775_rebased_review_20260824.md — interdiff clean (half->
lane rename, GroupInvariant threading gone, materialize API migration, P0-
shaped flush gate correctly filtered at simd.py:3672); four-line
ensure_active integration fix correct but wants a why-safe comment. Suites
independently green in both worktrees; kernel_num_gb "known failures" are a
narrow-runner artifact (3rd confirmation) — use run_wt.py. Still open at
the #190595 level: commit-message materialization sentence, nested
cpp_wrapper test. F2's _copy_masks must adopt mask_vars_for_shape on rebase
(open I4).

F1+F2 redesigned rebase (2026-08-24, by this session):
agent_space/f1f2_redesign_wt at 931151d8c53, all uncommitted;
agent_space/f1f2_redesign/README.md + f1f2_redesigned_full.patch
(+1895/-247). Redesign deltas vs canonical F1/F2: must_forward derived from
kernel.store_buffer_names (plan field deleted); F2 _copy_masks deleted
(masks assigned at materialization via set_value_masks; structured views
mask-free, barriers protect gathers); fail-closed OpsHandler classification
(133 safe / 19 barriers / 3 handled) + OP_NAMES completeness test (closes
the 5x-carried ARCH_REVIEW debt); loud stale view/split caches. P1.2 and
P1.3 were already closed in the canonical layers. All suites green
(396/108/59) + 33-case semantic fuzz matches reference + quicklint clean.
CAVEAT: semantic_fuzz.py self-overlays only test helpers - run it through
run_wt.py or staged-kernel assertions fail spuriously. COMPLETE (2026-08-24):
62-case differential fuzz vs fresh BASE-tree captures = bitwise (0.0 across
142 invocations, 104 exact/260 numeric); adversarial fresh-eyes review of
all four redesign deltas held with zero findings (generated-code proof for
identity-forwarded gather guards; deferral chain instrumented; stale-cache
raise structurally unreachable); completeness test hardened with
HANDLED-set disjointness. Final patch sha 27db67941fa8. Ready to become the
F1/F2 PRs.

Stack RESUBMITTED on new trunk base e812f6fc74c (2026-08-24): orig commits
70d9ccee025 (#190595) -> b225e3b078b (#191775); rebase artifacts in
agent_space/stack_rebase_20260824/ (per-PR before/rebased patches + wt).
#191775 exact-dependency redesign (uncommitted in that wt, patch sha
82df89c0e6) APPROVED: agent_space/pr191775_exact_dependency_review_20260824.md.
Replaces name-keyed index_proven_names with MemoryDepMatch/
ProjectedSourceAccess records (= indexed-forwarding vocabulary built at
fusion time); resolves stale-routing + score-bridge round-2 findings by
construction; has_planned_matches suppresses loop rewrites. IMPORTANT ENV
FACT: installed editable torch (git7c78410) predates the new base -
run_wt.py full overlay is UNUSABLE against stack_rebase worktrees (op +
kernel-launch skew; shims insufficient); use agent_space/run_narrow_plus.py
(narrow preload + triton_utils) DIFFERENTIALLY (base-vs-patched) instead;
flop_counter float32 x2 fail environmentally either way. Full suites via
that protocol: zero patch regressions (nested 387 OK). Real fix: rebuild
editable torch onto the new base.

#191775 negative-space audit (2026-08-24, after the redesign was AMENDED into
HEAD 140296df2441 of stack_rebase_20260824/wt): a NEW uncommitted 36-line
refactor there (StagedReductionPlan.projected_access_pairs helper) BREAKS 8
dependency_matches unit tests — the tests Mock the plan, and
Mock.projected_access_pairs() is not iterable (scheduler.py:9204). Fix tests
to use real plans/configure the Mock (or keep inline stage iteration) before
amending. Mutation analysis (runner
agent_space/adv_round_191775/tests/run_mutant.py, SCHEDULER_OVERRIDE env)
showed the killed name-permission bug class is pinned by hard NUMERICS in 3
shifted rejection tests (rejects_shifted_parent_output/_shifted_reduced_source/
_shifted_reduction_output), and empty-match non-suppression is pinned by ~29
e2e tests. UNPINNED holes (mutants survive both suites): multi-write-producer
decline (len(writes)!=1), mutation renames on planned matches, sync-mode and
TMP gates inside _memory_dep_supports_index_equivalence, and the append-path
None-matches decline (shadowed by ordinary vertical legality). All scheduler.py
line pins in the three pr191775_* docs drifted (+13..+32) after the amend; the
docs still call the redesign "uncommitted" (stale). mxfp6 -k count is 27
passed + 3 skipped (docs say 30 passed).

#191775 adversarial round 2 (2026-08-24):
agent_space/pr191775_adversarial_round2_20260824.md. ONE PROVEN silent
wrong-fusion (P1, BLOCKS #191775): a frame-mismatched transposed consumer
whose read linearizes to a lane-0 read in its own loop order is admitted
(per-node lane proof + position-blind _fusable_read_after_broadcast on the
scale read + reindex bypassed), codegen CSEs the bodies -> ~96% wrong
(repro adv_round_191775/sched/attack6, case transposed_8_512; also nested
variant). SCOPED by me: NOT in landed trunk, NOT in #190595 alone -- 
arrives with #191775's relaxation, identical in name-based and redesign
spellings (survives via the kept generic-equivalence branch). Fix: frame-
consistent residual validation in the (x, child_r) frame, OR plan-records-
for-broadcast/identity + delete the generic branch (P1 makes the eventual
model a correctness fix, not cleanliness). Also: uncommitted
projected_access_pairs() refactor breaks 8 Mock-fixture unit tests
(fixture-only; e2e passes with it); 4 mutation-tested coverage holes
(multi-write, renames-through-match, atomic, TMP); reindex True-path
unreachable+uncovered (S1, folds into P1 fix); codegen lens fully clean
(byte-identical codegen files, plan recomputed + loud on skew).

Split re-review (2026-08-25, later): SWIZZLE BLOCKER RETRACTED and
countersigned -- fusion was intact, only output-pointer numbering changed
(my root-cause conflated the consumer's swizzled STORE with its scale READ,
which is all the frame proof inspects; the traceback had failed PAST the
form assertions). Updated split state: layered wt staged==prereq diff
byte-identical; FULL suites green on both layers (nested OK both, scheduler
env-only errors); P1 attacks 4/4+5/5 on both; guide
pr191775_split_step_by_step.md line-pins all accurate; ownership-scoping,
unsafe-write, empty-plan-suppression, and branch-local-recognition tests
all present with real fixtures. Split is review-ready.

#191775 split review (2026-08-25):
agent_space/pr191775_split_review_verdict_20260825.md. Split = prerequisite
(scheduler-only exact-dependency mechanism + P1 fix via broadcast plan
records + ownership precedence: sub-parent reads MUST be plan records,
legacy equivalence only for grouped-stage-owned reads) + upper MXFP6 layer;
worktrees pr191775_prereq_split_wt / pr191775_layered_split_wt on #190595
6b8ef64bd37 (stack respun again). VERDICT: architecture approved, P1
confirmed killed (attacks 4/4+5/5 under my harness), ONE must-fix
regression: the raw-positional broadcast frame proof un-fuses the
swizzled-scale epilogue (test_rmsnorm_block_scale_swizzle_kernel_form fails
both modes, both layers; base passes; missed AGAIN by focused -k
verification -- 4th recurrence). Fix: normalize source+consumer into the
SOURCE-anchored shared frame via normalize_with_ranges instead of raw var
positions. Note: records dropped per-consumer lane -- F1 rebase must
re-derive lanes at codegen (resolver does, loud assert) or re-add.

MXFP6 layer review + convergence (2026-08-25/26, this session + parallel
author session): stepped guide agent_space/pr191775_mxfp6_split_step_by_step.md
steps 1-6 reviewed. Step 5 deferral challenged ("why not
invalidated_stores/debug_barrier?") -> author MEASURED it (deferred vs
materialize/reload: 2.82x / 4-8% / 29% wins, doc
agent_space/pr191775_looped_internal_source_placement.md) and reintegrated as
generate_node_schedule(required_post_reduction_index=...) — write-only state on
the default path, unit-tested; still open: no-reduction-past-boundary
assertion, "must follow a reduction loop" leg test, co-residency docstring,
two dead planner rejection legs. Slice-store experiment
(bench_mxfp6_slice_stores.py): (4,1) concat-slice writers fuse under a narrow
complete-ConcatKernel partition check, FASTER on synthetics (ratio 0.94; 4 vs
5 stores — the (4,3) path double-stores byte lanes), but only ~45 NET lines
after transparency machinery. TOOLING TRAP: editable-install
ScikitBuildRedirectingFinder silently imports the INSTALLED worktree's
modules; sys.path is not enough — repoint finder.known_source_files (see
agent_space/run_slice_proto.py). PRODUCTION REALITY (options doc
agent_space/mxfp6_dcn_preshuffle_fusion_options.md): the DCN+preshuffle
(2048x3072, G=32, B200) graph declines ALL reviewed machinery (outer-axis
frame permutation on the code buffer; two kernels 28-30us; hand kernel 24.1us
and has a tail-mask bug at this shape). Proposal C (frame canonicalization)
measured 18.2us -> REJECTED. FINAL DESIGN (2026-08-26): keep codes+packed
bytes ROW-MAJOR; swizzle only the small scale tensor; retain the reviewed
(4,3) codegen (slice route recorded as net-45-lines, not taken); admit the
swizzled-scale node — measured 16.38us (scale-only + narrow allowance) vs
16.5us (local scatter modeled on PR #193599); my guidance: prefer REINDEXING
the swizzle node into the producer frame (exact reads, permuted stores, no new
legality category); the padding contract of the real block-scale layout
decides permute-form vs 193599-style auxiliary-write scatter; #193599 itself
is precedent, not a dependency (alone it does NOT fix the pre-permuted graph);
if any allowance survives, it must be plan-classified and memory-resolved
(barrier), never name-keyed-forwarded. Process rules paid for twice: the
production graph is the ONLY acceptance test (synthetics misled on both the
slice deletion and the (4,3) reprieve); experiments with pre-declared gates
settle design fights that argument cannot. CLOSED 2026-08-26: user confirmed
good perf on all target kernels and moved on; the "still open" items above are
unlanded review asks to check if this stack is touched again, not active work.

F1 prototyping round (2026-08-26/27, this session orchestrating subagents; user
opted in): disposable fork agent_space/pr191775_followups_wt = 6b8ef64bd37 +
the layered diff (snapshot /tmp/layered_combined_1707.patch, scheduler diff md5
0945d36d7faafcd72b474ee5d1379e3e); production worktrees untouched. Plan:
agent_space/f1_rebase_plan_20260826.md (reviewed+amended: Phase 2 split into 2a
shadow-mode/2b authority-flip, phases 1-3 ship as one ghstack stack, end-state
~+95 LOC net-add ACCEPTED as mechanism-inventory reduction — do not relitigate).
P0 guard battery LANDED on the fork (test files only, verified): mutation-kill
matrix all YES; found the original bare-pairs chain-reject e2e test was
VACUOUS (never formed a staged candidate; now a genuine declined candidate) and
that killing the extra-axis mutation requires a colliding-source case (cross-
node symbol collision fools frame equality — the guard's unique value). Suites
120 OK scheduler / 401 OK nested on B200 via run_wt.py. Phase 1 (access
identity: ProjectedConsumerAccess+lane, _logical_memory_access Approach A,
agreement sweep) launched as agent f1-phase1-impl; parked agent
f1-rebase-analysis holds plan context for reviewing the implementation.
Phase 1 LANDED on the fork (2026-08-27, test+scheduler+simd, uncommitted;
my spot-check passed): zero plan-vs-body drift across all 8 sweep legs
(Approach A holds, Approach B not needed); 32 kernel files byte-identical
(pristine-vs-pristine determinism proven first; only backend_hash/tmpdir
normalize); suites 126/409 OK; the 1 nested "error" is a PRE-EXISTING AOTI
toolchain IndexError in test_rmsnorm_block_amax at pristine baseline —
environment, not stack code. Phase 2a inherits: port the record/materialized-
invalidation half of OLD test_projected_access_records_exact_index (Phase 1
ported only the _SourceAccessKey keying half). Backups+logs:
agent_space/p1_backup/, captures agent_space/p1_kernels_before|after.
Phase 1 conformance review by the plan author: APPROVE, zero issues (complete
6-site .consumers reader inventory; sweep verified test-only + two-directional;
byte-identity re-checked from captures); plan doc marks Phase 1 DONE. Phase 2a
(shadow-mode _IndexedProjectedValueStore, name path authoritative, kernels
byte-identical by construction, 62-case fuzz + shadow-agreement acceptance)
dispatched 2026-08-27 to f1-phase1-impl, including the reviewer's note that
shadow observation must extend to parent-stage recording.
Phase 2a LANDED + APPROVED (2026-08-27): zero shadow disagreements across all
legs, both FULL suites with live asserts, and the 62-case fuzz (104/260/0.0 =
OLD F1's exact numbers); kernels byte-identical; scheduler.py untouched. Two
rulings: (R1) plan's "declines -> keyed must miss" was WRONG for standalone
BROADCAST (store_cache forwards below the handler); outcome-level contract
adopted into the plan — NOTE 2a agreement is source-value level, 2b
byte-identity is what validates projected-value equality. (R2) emission leg
legitimately 2b; exhaustive "2b INHERITED FROM 2a" list now in the plan doc.
No double-recording (store is resolver's inner; single traversal even in the
doubly-wrapped nested region); reload_checks counts store-cache forwards AND
genuine reloads (looped legs 2/0/4 = loop-flush reload story confirmed).
Phase 2b (authority flip) dispatched to f1-phase1-impl with binding gate:
2b is NOT done until the record-removal mutation battery (battery agent,
post-flip) passes; name machinery stays present until Phase 3.
Phase 2b LANDED + conformance-APPROVED (2026-08-27, still NOT-done pending the
battery): authority flipped, all 32 kernels byte-identical (= projected-value
equality validated), fuzz bitwise again, common.py load_without_store_forwarding
extraction verified behavior-identical case-by-case (the highest-blast-radius
edit). Ruling outcomes: stale-memo raise removal ACCEPTED (raise was invalid
for newvar lanes, never in OLD; standalone BROADCAST memo IS the source object;
nested memos safe via the verified one-region structure between flushes) with a
STANDING FAIL-CLOSED REQUIREMENT in the plan: any future inter-stage flush (F2
deferral) must clear _materialized alongside cse.invalidate — nested memo
validity does NOT follow from source liveness. Lazy nested placement supersedes
OLD eager sweep. Two OLD test ports confirmed F2-only, moved to the plan's F2
interface (now carries three rebase preconditions). Comment-only follow-up
(flush coupling at the memo site) dispatched. Remaining for 2b-done: the
record-removal mutation battery (running).
Battery PASSED + Phase 2b marked DONE (2026-08-27): 6 permanent tests/12
cells, no cell silently name-forwards; suites 144/431. Gate rulings: (A) F1
looped whole-record drop degrades to the STANDARD barrier-guarded reload —
verified by re-deriving the mutated kernel (6 loads/6 tl.debug_barrier, one
before each lane re-read; chain = invalidate -> invalidated_stores -> the
#1615 guard; NOT luck); (B) the loud backstop for desync is two-layer:
width-mismatched forwards trip shape_propagation.py:45-51, width-compatible
ones are benign by exactness/broadcast semantics; "raise on unplanned access
of planned name" hardening REJECTED (false-positives on legitimate exact
reads); (C) battery tested CODEGEN desync only (fusion saw unmutated plans) —
IDENTITY records' authoritative role is FUSION LEGALITY; "redundant at
codegen" != deletable. Phase 3 (net-delete sweep) dispatched with the risk-4
do-not-touch note (generic CSEProxy machinery stays; the degradation path
depends on it). Phase 4 (parent-to-grouped records + legacy branch deletion)
remains after that.
Phase 3 LANDED + APPROVED + DONE (2026-08-27): -189 net realized (reviewer
re-measured ~-195; gap vs -225 estimate = builder/hook scaffolding, accepted);
resolver class, family name machinery, forwarded/masked params, three name
views all deleted; shared builder _codegen_sub_parent_stage takes the STAGE
(accepted deviation); name-partition TODO re-scoped (partition = input
classification, proofs unification is the removal path); AttributeError
tripwire added. Deleted extent assert ruled CLEAN dead-validation (pre-F1 it
guarded live codegen lane derivation; post-F1 equivalent guards = plan-time
lane proof + loud rebuild failure + _select_lane invalid-lane raise; residual
backstop is the loud compile-time class). Phase 4 dispatched (verbatim-move
of the nested equivalence proof into grouped_relations records with a
DEDICATED kind fail-closed at the codegen store; delete legacy branch +
_fusable_read_after_index_equivalence + _fusable_read_after_broadcast;
strictness upgrade OUT of round; battery will arm at FUSION time for these).
Phase 4 LANDED + APPROVED pending fusion-time battery (2026-08-27). SPEC GAP
found+fixed: legacy branch also accepted GROUPED-INTERNAL relations (grouped-
written scale broadcast-read by later grouped consumers); relation builder's
write table spans outer+grouped writers (8 tests regressed under parent-only,
pass after; faithfulness = union over fusion steps, per-step scoping restored
by producer_output_names filter). COLLAPSE SAFETY resolved: epilogue exclusion
exists BY CONSTRUCTION (grouped_stage_nodes = grouped minus sub_parent_nodes
at the sole NestedReductionStage construction site ~2058-2080); three-legged
argument recorded (exclusion + classification exhaustiveness + hard plan gate
at 2054-2055). KNOWN NUANCE: under artificial desync a value-colliding
sub-parent drop is accepted at fusion via the grouped pair and caught at
CODEGEN (normalization drops trailing broadcast axes -> value collisions are
real); battery cells must be collision-free or expect the codegen-loud
backstop. PARENT_TO_GROUPED as 4th enum member ruled OK (allowlist guard in
store ctor + structural separation). Net +37 measured (mechanism win intact:
one prover, membership-only fusion, legacy branch + both helpers gone/moved).
Fusion-time battery dispatched incl. the REQUIRED plan-gate pin test
(plan_from_topology None when SUB_PARENT nodes exist and stage planning
fails). Kernels byte-identical throughout all phases so far.
F1 REBASE ROUND COMPLETE (2026-08-27): Phase 4 DONE, fusion-time battery green
(suites 150/441; AOTI error confirmed environmental — no repro on battery box).
Closing rulings: (R1) reshape-equal grouped relations are e2e-INERT under
LOAF=True (post-merge re-extraction + gated normalize make reads raw-exact)
but CONTRACT-REQUIRED at the prove step under LOAF=False (fbcode default) —
risk 8 in the plan, double-pinned (unit test under LOAF=False patch at
test_inductor_scheduler.py:1302 + e2e shift-flag at test_nested_reduction.py:
4380); DO NOT simplify the reshape family away from OSS-regime observations.
(R2) METHODOLOGY: bitwise between two INDEPENDENT compiles is unsound (~1/6
1-ulp f32 flake; autotune config flips change contraction) — battery helper
keeps ints + sub-16-bit floats (packed bytes, fp8/e8m0) bitwise, floats at the
1e-2 convention; kernel-source byte-compare remains valid; future "bitwise"
fuzz claims must be same-compile-reference. END STATE on the fork
(agent_space/pr191775_followups_wt, all uncommitted): one prover,
membership-only fusion, per-read keyed resolution (per-key mask+fill, planner
lanes), all name-based machinery deleted; ~+190 net LOC accepted under the
mechanism-inventory ruling; 32 kernels byte-identical through every phase;
3 batteries + 6 conformance reviews all green. CARRIED FOLLOW-UPS (also in
the plan doc's ROUND COMPLETE block): ship shape = phases 1-3 one ghstack
stack + Phase 4 independent, USER-driven per AI policy; optional explicit-
frame hardening commit; F2 rebase preconditions (memo-flush clearing, delta-4
raise belongs to F2 caches, two F2-only tests port with F2, keep the
resolve/materialize seam); proof-unification TODO (scheduler.py:1665); doc
sync of FOLLOWUPS.md item 1 + contract doc at shipping time (both satisfied).
PARALLEL LINE reviewed (2026-08-27): the author session built its OWN F1+F2a as
stacked worktrees (agent_space/followup_indexed_rebase_wt = layered staged +
F1 unstaged; followup_lazy_projection_wt adds F2a unstaged). Their F1: no
layout enum (shape+lane-witness dispatch; num_groups==child_block ambiguity
ordered+tested), one record type SubParentAccessRelation (one consumer/record,
planner-derived requires_live_source), FINER guard identity (mask + Python
type + constant spelling; 0/0.0/-0.0 distinct — adopt in any merge), +144 prod
net; their Phase-4 trial LOST 20+ fusions and was REVERTED (legacy branch
retained with TODO) — my fork's verbatim-move Phase 4 is the existence proof
+ transplant path (~130 lines onto their record type). Their F2a: narrow
cast-before-broadcast rule (+67 net), exact outputs, NVFP4 geomean 1.30x
(pathological persistent 512->173us), MXFP4 neutral control — banked win.
CRITICAL SIDE-PRODUCT: masked_group_source_diagnosis_20260826.md = REAL
inherited fail-to-fallback bug in shipped #191775 (raw-frame broadcast proof
accepts, merge_loops normalizes producer/consumer differently, codegen replan
aborts "plan was lost" instead of 2-kernel fallback); 10-line fail-closed fix
+ 2-arm regression validated in masked_group_source_diag_wt — MUST land in
the base, applies to BOTH lines and my fork. HARNESS: their
run_worktree_tests.py is a NARROW overlay omitting common.py (which the base
modifies — store_buffer_counts); I re-ran BOTH their suites under run_wt.py
full overlay: scheduler 130 ran OK/6 skip, nested 397 ran OK/8 skip — their
numbers CONFIRMED (skew existed, outcome-neutral); their newer
run_test_from_worktree_20260826.py is the proper redirect runner. Evidence
gap: their adversarial results live in DOCS not batteries — port my battery
suite. Recommended merged line: their F1+F2a base + my Phase 4 + my batteries
+ their guard identity + the masked-group-source fix in the base.
Phase 1 IMPLEMENTED on the fork (2026-08-26, uncommitted): scheduler 126 OK /
nested 409 with only the pre-existing env error (AOTI test_rmsnorm_block_amax
toolchain IndexError, present at pristine baseline); 16-form kernel capture
byte-identical; Approach A validated by the sweep on all 8 legs (3-7 records
per leg), zero plan-vs-body dep drift (risk 6 clear). Backups/logs in
agent_space/p1_backup, captures in p1_kernels_before|after. OPERATIONAL
LESSONS: (1) never edit worktree sources while a run_wt.py suite runs --
Inductor compile-worker subprocesses re-import torch._inductor mid-run and
poison results with mixed-state AttributeErrors; (2) generated-source byte
captures are deterministic across fresh processes EXCEPT the
fresh_inductor_cache tmpdir in the wrapper's "# kernel path:" comment and
inductor_meta backend_hash (fingerprints _inductor sources) -- normalize
exactly those two; (3) OLD test_projected_access_records_exact_index needs
phase-2's _IndexedProjectedValueStore, so its phase-1 port pins
_SourceAccessKey dict keying instead.
Phase 4 (plan-authoritative grouped relations) IMPLEMENTED on the fork
(2026-08-27, uncommitted): PARENT_TO_GROUPED enum member (scheduler.py:608),
_parent_to_grouped_relations classmethod (1440, THE fusion-battery patch
point), relocated _fusable_read_after_broadcast verbatim to NestedReduction
(1490), grouped_relations field on NestedReductionStage (2220, built in
plan_from_topology), projected_access_pairs extended, prove step collapsed to
strict-match/plan-membership/decline (legacy branch + composite helper +
read-classification sets deleted; mode_requires_synchronization and
_memory_dep_supports_index_equivalence became staticmethods so planning can
call them), fail-closed non-codegen-layout guard in the keyed store ctor
(simd.py:2440). KEY SPEC GAP FOUND: the plan said "reads of parent-written
buffers" but the legacy branch ALSO accepted GROUPED-INTERNAL relations (scale
written by the grouped stage, broadcast-read by the quant/swizzle node) -- 8
suite tests regressed until the write table included grouped-node writes too.
Net +36 production (vs plan's ~-25; the relations builder outweighs the
deleted branch). 16 forms byte-identical; sched 144 OK (-2 = ownership test
went 3->2 parametrizations, renamed
test_nested_dependency_matches_require_plan_membership); nested 431 identical.
Phase 3 (net-delete sweep) IMPLEMENTED on the fork (2026-08-27, uncommitted):
deleted _SubParentSourceLoadResolver, family remapped_values/resolve_load/
lane fields (+make_sub_parent_family population incl. its extent assert),
handler load_resolver/forwarded/masked params, the three stage name-view
properties; standalone internal_source_names re-derived from INTERLEAVED
projection sources ∩ parent-written (simd.py:3950); ONE shared
_codegen_sub_parent_stage builder (simd.py:3890) with emit_parent_stages
hooks in both paths (nested hook 3561 also handles the no-stage case by
calling hook(None) outside the builder); grouped-schedule/grouped-reduction
params retyped to the store; name-partition TODO re-scoped as
planner-internal (scheduler.py:1668). Net production delta -189 (simd -163,
scheduler -26) vs plan estimate -225 (hook scaffolding/reindent absorbs the
gap). 16 forms byte-identical; sched 146 OK (+2 tripwire
test_sub_parent_stage_has_no_name_views); nested 431 identical incl. battery.
Kept: planner-local broadcast_source_names name classification (proof
selection, not a codegen view); RemappedRangeValue alias (store memo type);
_select_lane (materialize_source).
Phase 2b (authority flip) IMPLEMENTED on the fork (2026-08-27, uncommitted,
PENDING record-removal mutation battery): handler consumer path is
get_projected_load -> resolve_source (must-forward raise live) -> IDENTITY
fast-path / materialize_source(planner lane); planned misses use the SHARED
common.py load_without_store_forwarding (CSEProxy.load refactored
behavior-identically around it, risk 3); name machinery present but unread;
name-free materialize_value_at_sub_parent_resolution + split-out
broadcast_group_value_to_lanes. All 16 kernel forms BYTE-IDENTICAL post-flip;
sched 144 OK / nested 419 (same AOTI env error); fuzz bitwise (104/260/0.0).
TWO STRUCTURAL FINDINGS: (1) the "expired within its stage" stale-memo raise
is unimplementable -- split lanes are cse.newvar variables invisible to
cse.contains_value (raise fired on every INTERLEAVED memo reuse); the
live-SOURCE gate before the memo read is the real staleness guard (matches
the applied OLD tree, which also has no raise). (2) The nested
full-INTERLEAVED eager materialize_projections sweep from OLD moves/reorders
tl.split emission vs today's lazy first-consumer-load point -> byte diffs;
nested stays lazy (standalone internal-source eager sweep kept at today's
spot). (3) test_projection_materializes_store_reduction and
_masked_callback_body are F2-ONLY tests (_DerivedDomainProjection /
_DeferredDerivedValue) -- unportable to F1; substitutes: 2a record tests +
guarded-projection matrix. Standalone BROADCAST scale is [XBLOCK, 1] shaped
(num_groups on X) so materialize passes it through unprojected = today's
store-cache behavior; nested BROADCAST needs the eager axis broadcast (both
byte-verified).
Phase 2a (shadow keyed store) IMPLEMENTED on the fork (2026-08-27,
uncommitted): _IndexedProjectedValueStore (simd.py:2628, records loads/stores
under normalized access+guard, resolve_source, must_forward; _materialize =
memo+liveness+stale raise, emission leg deferred to 2b) wired as the name
resolver's INNER in both paths plus a region wrap over the nested parent-full
pointwise + grouped schedule; _PointwiseRemapHandler shadow checks (2442/2501)
compare every epilogue load decision. KEY DISCOVERY: standalone BROADCAST
"forwarding" happens inside CSEProxy's name-keyed store_cache, NOT at the
handler (handler-level name path declines) -- the shadow reload check must
compare against the store-cache outcome (value is cached), and unplanned
reads are exempt via the risk-4 exact-index coupling. Results: kernels
byte-identical (16 forms), sched 134 OK, nested 417 with only the
pre-existing AOTI env error, 62-case fuzz bitwise (104 exact/260 numeric/max
0.0) with ZERO shadow asserts anywhere; grouped store_reduction records
arrive via _GroupedReductionOpsHandler.store_reduction forwarding as
inner.store under a store_reduction FX node (why record_store accepts both
targets).

Planned (2026-08-21, not started): wrap the quant-cast workloads (MX/NVFP4 x
dim0/dim1 x layouts x fused producers, from agent_space/bench_dim1_quant_casts.py
and the gemm_ready/swiglu bench scripts) into a quant benchmark for the
benchmark suite ("better benchmark"; benchmarks/dynamo/genai_layers is the
likely in-tree home), then iterate on the default launch-config heuristic
using the 5000+ kernel corpus as evidence. Key finding to encode: tiling
scores already drive score-proportional block distribution
(_match_target_block_product), but the persistent seed ladder caps the block
budget at 64*rnumel and scales warps up with product; the fix is bigger budget
seeds for score-skewed kernels + warps derived from an elements-per-thread
band (~32-128). Untested prediction: Y256xX2 at w4 on the mxfp8_dim1 kernel.

Full-stack review (2026-08-18, at d4d6f910e9f) found two CONTIGUOUS-layout
correctness holes in #190596 (external-contiguous split from partial R0_BLOCK
in looped kernels when an internal source skips the CSE flush — proven by
repro `agent_space/repro_looped_external_contiguous.py`; and the pow2 gate not
covering strict-reduction R0_BLOCK > rnumel), plus a G=2 tiebreak bug in
#190595 (`used_buffer_names()` includes the reduction's own output).

Upper-stack replay verification (2026-08-22, upper_projection_latest_replay_wt
= 3f62ae27baa + upper_stack_replay_190596_191974.patch): found ONE P0
replay-introduced miscompile — the #190596 `internal_source_names` migration at
replay simd.py:4303 dropped the lane-layout filter (must_forward over ALL
layouts vs original INTERLEAVED/CONTIGUOUS ∩ parent-written), so a parent-written
reduction output consumed as IDENTITY/BROADCAST suppresses the looped
pass-boundary `codegen_body()` flush at simd.py:4376 → epilogue emitted inside
the reduction loop referencing the not-yet-finalized accumulator (NameError
tmp6, wrong-numerics class). 6 full-suite tests fail (chunk_many_chunks_8/16,
interleaved_pair_D_1024_G_2 x2, chunk8/16_kernel_form, all NonPersistent);
UPPER_STACK_REPLAY.md's focused -k selection deselects all 6, so its 35/3 and
20-passed claims reproduce while the full suite is red (442/6/12 of 460).
One-line fix proven in agent_space/rev_replay/replay_fixed_tree (restore
`projection.layout in (INTERLEAVED, CONTIGUOUS)`); original #190596 and
composed stack both pass the same cases. All other migrated #190596 sites
verified semantics-preserving; CONTIGUOUS provably cannot reach the lazy
projection engine (nested planner allow_contiguous=False + handler defers only
IDENTITY/BROADCAST/INTERLEAVED + standalone path never constructs
_DerivedDomainProjection). Repros/evidence in agent_space/rev_replay/.

8-bit frontier adversarial review (2026-08-22, relayer_projection_composed_wt
unstaged diff): 20+ fresh differential cases in `agent_space/rev_8bit/` all
held (fixed divisible/tail/oversized R0, autotuned pow2/non-pow2, CD tuning,
dynamic pow2, factor-4 tails, u8/i8/fp8 triggers, fp8 reload, group-where,
multi-output-group). No miscompile. Pow2-arm soundness rests on
`_get_nd_reduction_numels` capping R0_BLOCK at size hints + CD `get_config_max`
doing the same; sub-parent stages only exist for GroupedAxis.R
(scheduler.py `_nested_sub_parent_rate` guard), so the R-loop gate reasoning
holds. Known benign class: differential refs vs nested-OFF can flip fp8 scale
bytes when fusion boundaries change bf16 intermediate rounding (staged keeps
fp32) — proven internally consistent, not a bug. Note: e8m0 maps to tl.uint8
in codegen; `has_r0_mask` in the tail audit JSON is textual (weak metric).
Addendum (same day): instrumented CD (check_all_directions, radius 2) — all
R0_BLOCK candidates pow2, hard-capped at the extent for pow2 rnumel (435
candidates, max tried == 8192 == extent); group_tree.numel empirically = full
rnumel. multi_kernel=1 never emits MultiKernelCall for staged kernels.
frexp differential failures are a BASELINE recompute hazard (nested-off mixes
bf16/fp32 amax paths across kernels, 2x errors near pow2 boundaries; staged
kernel is consistent). Robustness recs filed: _view/_split cache staleness
fall-through should raise (simd.py ~2434/2459), and store_reduction should be
a projection barrier (unreachable-with-deferred today).

F1/F2a REVIEW ROUND, current state (2026-08-27, this session as independent
reviewer; user walking agent_space/f1_detailed_review_guide_20260827.md):
review doc agent_space/f1_f2a_stack_review_20260827.md (verdicts: base fix +
F1 + F2a all APPROVED; section 8 = implementation session's disposition,
section 9 = reviewer reply). MID-REVIEW GUARD REMOVAL: user's probing
questions (record-vs-inferable, dedup of same-input reads) triggered
instrumentation that (a) proved source_accesses plurality REAL (flat [B,R] +
grouped [B,G,L] raw witnesses of one input; FOLLOWUPS item 7 corrected from
"maybe singular" to witness-vs-runtime-identity distinction), then (b) F1
DELETED _AccessGuard entirely: guarded loads never recorded, forwarding
declines on concrete _load_other, ops.masked's outer where owns predicate+
fill; fill-aliasing now structurally impossible (elimination beats keying);
F1 net +148 -> +58; delta review APPROVED (my earlier praise of guard keying
superseded for stated reason). F1 commit now 6d545ab0fb9 with cleanup
unstaged, pinned by re-attestation (all 10 hashes, corpus in
agent_space/f1_no_guard_final_corpus_20260827/). Guide verified refreshed.
OPEN ITEMS: (1) F2a mechanical rebase TRAP: its narrow group-width exit must
re-key on kernel._load_mask is None, NOT _load_other is None (latter would
wrongly admit masked-callback consumers to the raw group-width path); (2)
mutant-vs-in-tree-oracle matrix (run 7 scratch mutants with only checked-in
tests as oracles) still pending. LESSON reinforced: user's "I don't
understand this" questions outperform formal review passes (behemoth, now
guard removal) — treat over-machined-feeling code as a review signal.
F1 FINAL FORM = OPTION C (2026-08-27, endorsed after gate verification):
user's "icky" objection to _logical_memory_access led to a design-options doc
(agent_space/f1_access_resolution_design_options_20260827.md) and Option C —
planner keeps exact per-access fusion authorization (P1-class protection
UNCHANGED); codegen collapses records to a per-buffer contract (name-keyed,
sound via the twice-enforced exhaustiveness invariant: builders decline
incomplete coverage + codegen plan-rebuild fails loud). Interpreter bridge
DELETED (zero V.interpreter refs in simd.py); lane derived from the replay
index and validated against the planner lane SET; containment total (simd-only
delta + neutral no-construction test). Net -15 more lines, CC down, all 10
hashes + suites + 62-case fuzz hold. KEY DEFECT caught in validation: (4,3)
replay stores need LAST-WRITER-per-replay while external sources keep
ALTERNATIVES (role-dependent retention; old F1 handled implicitly via
re-record eviction) — pinned by test_sub_parent_source_capture_is_role_aware;
new walls pinned by rejects_inconsistent_name_contract / uses_planned_lane_set
/ required_source_must_remain_live. Results:
agent_space/f1_option_c_results_20260827.md; guide:
f1_option_c_review_guide_20260827.md. REMAINING at close: amend C into F1
commit + re-attest snapshot; refresh detailed guide once then FREEZE F1; F2a
promotion from f2a_option_c_wt (one focused test improvement + verify narrow
exit keys on MASK-absence not fill-absence); mutant-oracle matrix re-aimed at
the four new C tests; ship sequence = base fix -> F1(C) -> F2a, user reads
diffs and submits per AI policy, evidence tables go in commit messages.
Fuzz caveat stands: independent-compile bitwise on f32 can flake 1-ulp
(autotune contraction) — suspect the flake before the code on reruns.
F2A FINAL FORM = REGISTRY-BASED GROUP-WIDTH NARROWING (2026-08-27, reviewed
APPROVE in agent_space/f2a_final_wt on committed F1-C base 859385382d5).
Purpose: the natural NVFP4 quant+swizzle graph computed the group scale in
TWO fused epilogue bodies; eager broadcast defeated CSE -> two FP8
conversions. Fix: forwarded lane-less unmasked group-width sources stay raw
(resolve_load early return keyed on _load_mask is None — the mask-absence
trap check PASSES); the scale chain stays narrow through ops that are (a) in
registered_pointwise_ops AND (b) ShapePropagationOpsHandler-proven
group-width, with a loud post-assert; boundaries widen at store, non-listed
_default (pytree over args+kwargs), and a masked() override (mask/other/body
result, .graph forwarded). The user replaced an initial 6-op frozenset with
registered_pointwise_ops populated by register_pointwise() in lowering.py
(+ explicit to_dtype/mul/truediv at their make_pointwise sites) — invariant
"deterministic scalar op commuting with broadcast" is structurally
maintained by the registration path; I audited all 100 names (pure, no
RNG/memory/position ops; rand exclusion test-pinned). materialize_group_width
routes through _materialize (liveness check -> loud fail, cache). Lane-less
materialize_source never consulted the index, so raw forwarding skips ONLY
the broadcast; exhaustiveness invariant unchanged. Silent-leak channel
closed: var names cross bodies only via resolver forwarding, so no
resolver-less body can CSE-hit a narrow var. Evidence: I independently reran
both suites on the final tree (403 OK/8 skip nested, 136 OK/6 skip
scheduler); guide claims 24 NVFP4/MXFP4 sources byte-stable across the
6-op->registry change (keeps prior mutation battery valid), all 10 protected
hashes unchanged (corpus has no NVFP4 quant entry; new form pinned
structurally by test_nvfp4_scale_swizzle_reuses_group_scale: exactly ONE
.to(tl.float8e4nv) + division-before-broadcast). Perf vs FlashInfer (B200,
CD on): NVFP4 geomean 1.036x, MXFP4 1.101x; no-CD config gap (7-13%) is a
separate issue. OPEN FLAG: perf table NVFP4-8192 and MXFP4-8192 rows
identical to 0.01us on both columns — verify against artifact JSON before
citing. Guide: agent_space/f2a_final_review_guide_20260827.md (anchors
verified). Production delta: simd.py + 8-line ops_handler registry + 3
register_pointwise_op calls in lowering.py + tests.
PR191974 DEFAULT-ON CI FAILURES + FIX (2026-09-01, reviewed APPROVE with
conditions). Write-up agent_space/pr191974_failure_investigation/FINDINGS.md;
fix worktree agent_space/pr191974_fix (uncommitted 4-file diff on 4ede30ec8aa).
TWO root causes: (1) X-grouped nested sets min_xblock=group_size AND selects
persistent reduction -> [128,1024] tile -> 131-262KB smem > A10 101KB (BERT/
DistilBERT/M2M100/MobileBERT/Pegasus/XGLM + distributed-BERT timeout). Fix:
override_persistent_reduction=False when not local_reduction_in_r; sort
topologies rejected (sort needs persistent). (2) WRONG-RESULT BUG (MT5 RMSE
5.4, CLIP grad): full-res pointwise replayed from grouped/transposed source;
generic range remapper proves only total-size match -> derives reshape where
truth is transpose -> weight[x+128*(r%4)] instead of weight[r]. ROOT DEFECT:
stage placement conflated with iteration order. Fix: PointwiseStage x
PointwiseIterationDomain split (PointwisePlan, consistency assert);
_resolve_full_resolution_iteration_domains solver = all domains in ONE
canonical symbolic frame, node LoopBody evaluated via codegen's own
map_kernel_groups_to_node_sizes/indexing_from_args, producer-store==
consumer-load index equality in shared frame == positional forwarding
correctness, arc-consistency to fixpoint, ambiguity accepted only if both
domains give identical accesses AND plain non-mutating Pointwise (permutation
safety), fail-closed everywhere (multi-writer, bucketize/index-expr with
in-fusion writer, CantSplit); plan preserved across merge_loops (provenance
unrecoverable after collapse). Shape-only rule measurably insufficient
(prototype instrumentation: loses B=1-parent-epilogue + flat-local families;
37/75 ambiguous decisions need constraints); unconditional parent-first is
WRONG (counterexample 2016/2048 corrupted). MY VERIFICATION: solver soundness
traced; scheduler suite 140 OK; nested 418 = 417 green + 1 env error
(NestedReductionAOTITest: /usr/bin/openssl shadows conda's under conda run,
breaks on LD_LIBRARY_PATH libcrypto -> empty stdout -> IndexError in mainline
codecache _get_file_checksum — NOT the fix; recurring on this box).
CONDITIONS BEFORE RELAND: (a) validation gap — CI failed
PegasusForConditionalGeneration but CSV validated PegasusForCausalLM (different
model!) — rerun the right one; (b) perf recheck of X-grouped looped-vs-
persistent policy on dashboards (accuracy validated, perf deltas not);
(c) answer/test whether sub-parent consumer can read a LOCAL-domain-assigned
producer (sub-parent nodes excluded from solve; expected fail-loud). FOLLOW-UP:
convert remaining reads_reduction_source name heuristic (REDUCED vs one-lane
SUB_PARENT) to same two-candidate exact validation — recovers a lost fusion,
kills the stack's last coordinate heuristic. META-LESSON (user's framing
validated): the bug was in the ONE remaining structural heuristic, not in the
F1/F2a exact-access layers; every coordinate decision must be index-proven.
2026-09-01 DUAL AUDIT (planner + codegen agents, post-fix pr191974_fix): NO
live silent-wrong-result site remains; both audits independently ranked the
SAME #1 fragility: fusion-time plans consumed at codegen with NO domain
re-proof (plan_from_topology with saved pointwise_plans skips the solver,
checks only node-set equality; scheduler.py:2321-2352) — sound today only
because merge_loops is order-preserving and reorder/reindex are frozen for
staged nodes (call-graph accident, no assert). HARDEN: re-run
_resolve_full_resolution_iteration_domains at codegen plan rebuild, mismatch
-> existing "plan was lost" error. #2: planner frame formulas
(_nested_iteration_sources scheduler.py:1055-1113) and codegen frame formulas
(construct_group_reduction_vars/parent_full_iteration_values/
sub_parent_iteration_values simd.py) are hand-maintained DUPLICATES with no
cross-assert — one-sided edit = silent cross-frame forwarding corruption
(the #191974 class). HARDEN: shared frame constructor or codegen assert.
Load-bearing implicit invariants to document in code: P1 splits are
ordinal-preserving (reshape never transpose; wrong-but-bijective frame is
self-consistent for pure replay, only POSITIONAL exchange can corrupt);
P2 every positional exchange must be exactly proven. [XBLOCK,1] bug
(their session, fix in flight): loud Triton reshape error for R0_BLOCK>1,
accidentally CORRECT for R0_BLOCK==1 — compile-failure class, never silent;
root channel = shape-STRING dispatch in _GroupedReductionLayout.parent_dim/
ensure_parent_tile_resolution/_broadcast_value_to_parent_resolution.
Benign-but-notable: get_grouped_axis square-collision picks wrong axis
(perf-only, benign by global consistency); reduced-shaped (x,1) epilogue
(mean=sum*inv_n) fits neither full domain -> silently declines WHOLE nested
fusion (perf, missing fusions on real models); mutation/aliasing checks
weaker on nested path than standalone (unify); nested codegen path doesn't
filter removed_ops (loud). reads_reduction_source tie-break re-verified safe
from 5 adversarial angles. Perf sequencing agreed with user: correctness
first; then gate X-grouped nesting (maybe outer-reduction parents only;
persistent dedupes loads and can win when it fits).
2026-09-01 FINAL FIX REVIEW (APPROVE, all conditions closed): singleton fix =
materialize_singleton param on _broadcast_value_to_parent_resolution
(consumer-declared: ensure_parent_tile_resolution=True for grouped-reshape
feeder, maybe_broadcast=False for pointwise); passthrough singleton broadcast
now precedes the parent-dim early return that hid it; positive+reject test
pair on minimized shapes. SECOND BUG FOUND IN THEIR ADVERSARIAL PASS,
SILENT-WRONG-RESULT CLASS: resolver early return `if not
full_resolution_nodes: return {}` skipped reduction-store<->reduction-load
constraints; reduction-only counterexample (column-reduce->broadcast->
reshape->reduce, both orders same total size) silently accepted transposed
fusion, max err ~414; fix removes early return so store/load equations
reject. MY MISS ON RECORD: I reviewed that early return and called the
solver fail-closed; worse, I suggested restricting constraints to edges
touching unresolved nodes as an optimization — that would REINTRODUCE this
bug; pinned-pinned constraints are SEMANTIC, never skip them. Conditions
closed: Pegasus ForConditionalGeneration validated; perf = R-grouped quant
neutral (-0.06%, kernel counts unchanged), X-grouped profitability data says
two-kernel form beats nested X-grouped across the synthetic matrix (separate
follow-up; load-dedup caveat; generic fusion bench can't measure staged);
LOCAL->SUB_PARENT structurally safe (SUB_PARENT is R-grouping-only, where
parent-full/local-full share x-major flat order). Nested suite 422 = 421
green + known openssl env casualty (verified same test); scheduler 140 OK.
OPEN LEDGER post-land: harden #1 codegen re-proof of saved plans, #2 shared
frame constructor, their flattened grouped-axis coefficient seam,
REDUCED/SUB_PARENT tie-break plan-based replacement, X-grouped profitability
gating.
2026-09-01 SIMPLIFICATION PIVOT = FINAL SHIPPED FORM (reviewed APPROVE:
agent_space/pr191974_simplified_review_20260901.md). User questioned the
stage/domain "behemoth"; perf data showed nested X-grouped LOST to two-kernel
form on every measured shape -> PR rewritten in
agent_space/pr191974_simplified_wt (+244/-135, 4 files, NO simd.py change):
(1) R-grouped-only at candidacy gate (X returns None, perf comment);
(2) 600-line solver replaced by narrow exact guard
_r_grouped_stage_accesses_match (scheduler.py:1013) — extraction of solver
constraint check into the ONE R-grouped frame, same codegen-owned index
mapping, all fail-closed (non-MemoryDep internal deps, multi-writer,
CantSplit, unknown domain); (3) reads_reduction_source name heuristic
DELETED — REDUCED+SUB_PARENT both-compatible now rejects plan (fail-closed,
loses one ambiguous fusion); (4) saved-plans machinery GONE — plan recomputed
fresh at codegen (audit hardening #1 DISSOLVED; possible because numel-based
classification is merge_loops-stable). X-topology tests repurposed:
check_numeric kept + check_fusion -> check_no_fusion. Retained R-grouped
quant/swizzle kernels byte-identical, timings neutral (2.180 vs 2.181us).
Commit msg (PR191974_COMMIT_MESSAGE.md) preserves trailers + AI disclosure.
MY VERIFICATION on simplified wt: nested 401 = 400 green + openssl AOTI env
casualty; scheduler 142 OK; config 3 errors ALL the openssl issue (cpp-wrapper
precompiled-header hashing; fingerprint: InductorError IndexError from
codecache._get_file_checksum, /usr/bin/openssl shadows conda's under conda
run + LD_LIBRARY_PATH libcrypto -> BIO_new_dgram_sctp symbol error -> empty
stdout; ALWAYS suspect env first on this box). RESIDUALS (non-blocking):
cross-ref comment guard frame <-> construct_group_reduction_vars (last
frame-dup residue); gated-dead X codegen machinery in earlier stack commits
(stack cleanup later); parked solver in pr191974_fix = reviewed+fuzzed IP for
future X re-enablement, do not rewrite; flattened grouped-axis coefficient
seam still open. STANDING RULE (paid for 4x): size/shape/name evidence may
nominate or decline, NEVER select coordinate semantics.
2026-09-01 ADVERSARIAL ROUND 2 on simplified wt (guard v3 at scheduler.py:1013
after their cycles fixed: grouped-internal edges, per-writer planned frames,
dep-INSTANCE purity, SUB_PARENT-writer fail-closed): FIFTH HOLE FOUND, REAL,
I CONFIRMED BOTH REPROS. NEW EVIDENCE CLASS = STAGE ORDERING (not index
equality): classification displaces outer pointwise (ancestors include outer
reduction) into grouped stage as LRI, but outer REDUCTIONS stay in
parent_nodes; a second outer reduction reading a displaced LRI buffer is
emitted BEFORE the write (parent stage first) -> same-kernel read-before-
write with PERFECT index equality. Repro: a=x.amax(-1,kd); e=(x-a).exp();
s=(e*x).sum(-1); z=e.view(64,16,32).amax(-1); return e,s,z. Persistent: s
max_err 2011 (e,z exact). Looped: FIRST CALL PASSES (autotune warmup leaves
correct data in reused alloc), second call stale, err 2070 — defeats
compile-once tests; run repros in separate processes (fusion greediness
nondeterministic in-process). Scripts:
pr191974_failure_investigation/repro_parent_reads_lri{,_looped}.py.
FIX SHAPE: reject/reorder plans where a parent_nodes member reads a buffer
whose writer is emitted in a later stage; nested path lacks the counterpart
of _order_sub_parent_parent_nodes/required_post_reduction_index; same check
should cover the WAR direction (currently unguarded, survives by accident
stack: functionalization/reinplacer/body-section order/load CSE).
SECOND (loud) GAP: template SchedulerNode passes isinstance gate with
_body=None -> AttributeError in can_fuse; add is_template()/_body-None gate.
CLOSED with verified args: mutation renaming fail-closed (dep renamed to
mutation output name, body memory_usage keeps original -> reads empty ->
False); read_writes covers body loads (same-body extraction; prunes only
weak/unmet); outer-LRI READ direction safe (parent==local logical coords in
R-grouping incl 1-iter flatten; intra-outer edges pre-proven at ordinary
fusion; acyclicity); welford (grouped side single-reduction whitelist, outer
verified); scatter unreachable (3 layers, note CSEProxy.store DOES cache
mode=None scatters — layers load-bearing). EMPIRICAL BATTERY (2nd agent): 94
cases, ZERO wrong results, 2 false criticals exonerated via
nested-vs-unnested BYTE comparison (the robust oracle; eager-tolerance
fragile at fp16 ULP/quant boundaries — adopt in quant tests). Conservative
perf gaps: dynamic-D disables nesting; indirect reads of realized internals
unfuse; column-axis reductions in chain unfuse. LESSON 5x: each hole was an
INCOMPLETE EDGE/PROPERTY SET (pinned-pinned edges, grouped-internal edges,
dep-type purity, stage ORDERING) — reviews must enumerate the full property
set {index equality x liveness x emission order x dep purity} over the FULL
edge set, not verify checked edges.
2026-09-01 PASS 4 (temporal axis) on COMMITTED simplified fix 61b333901fa
(ordering fix + template gate + 2 regression tests landed). Doc:
agent_space/pr191974_pass4_temporal_review_20260901.md. Ordering fix VERIFIED
sound: op-name spaces agree; ancestry includes mutation WeakDep/StarDeps
(unmet_dependencies = full reads; DCE prunes only removed-op WeakDeps;
compute_ancestors after DCE, never recomputed) -> RAW/WAR/WAW covered; static
attacker (45 topologies) found no temporal hole; battery ~600 runs 0 silent.
BUT TWO LOUD COMPILE CRASHES = DEFAULT-ON BLOCKERS (both I reproduced):
(A) dynamic batch (mark_dynamic OR automatic dynamic) + materialized normalized
output + block reduce -> "nested reduction plan was lost before codegen".
Mechanism: fusion-time plan on unmerged bodies accepts; codegen re-plan after
merge_loops sees one merged symbolic range, guard's ModularIndexing won't
simplify -> None -> assert. VERIFIED FIX: skip merge_loops for
FusedStagedReduction members (monkeypatch -> nested=1 correct); recheck
protected hashes after. Standalone F1 path unaffected (merge-invariant proofs).
(B) LOOPED parent + ANY prologue buffer (output or kernel-local) read by
sub-parent stage -> "lost required sub-parent source 'buf0'". Hits residual-
stream pattern h=x+res; return h, quant(rms(h)*w) at D=2048/4096/8192 bf16 and
fp32 — THE production shape family. Mechanism: prologue source marked
requires_live_source, emitted inside loop, dead after loop close; nested
sub-parent planner (_plan_nested_sub_parent_stage :1763) lacks standalone
path's _order_sub_parent_parent_nodes leading/final check (standalone DECLINES
same topology correctly, 2 kernels). FIX: (1) fail closed now — decline
SUB_PARENT when required internal source writer is an ancestor of the outer
reduction; (2) follow-up: prologue sources reloadable (stores complete before
stage; input_buffers registration blocks removal) to recover persistent+looped
fusion — F1 contract change, own review. Tripwire asserts did their job both
times (loud not silent) — keep them. bf16/fp16 nested-vs-unnested 1-ulp diffs
unless emulate_precision_casts (cast folding; say so in PR desc; never compare
low precision to eager). Scratch: tmp_dyn_basic_repro.py,
tmp_subparent_lost_repro.py, tmp_merge_freeze_check.py,
tmp_resid_bf16_check.py, tmp_standalone_staged_check.py, battery
tmp_temporal_* (see [[nested-reduction-temporal-battery]]).
Pass-4 addendum (agents' full reports): static attack A-G all SAFE with two
fragilities to document in code: looped-parent prologue buffers survive only
via must_keep_buffers (KernelArgs.input returns out_ptr for own outputs, so
NOT args.input_buffers — corrects my earlier reload-safety rationale);
_pre_fusion_custom_pass runs AFTER compute_ancestors (scheduler.py:5746) so
a dep-adding custom pass weakens the ordering guard. Outer group frozen after
first nested fusion (fuse_with reuses node1; _plan_fusion_with admits only
grouped-side consumers, rejects reductions and would-be-LRI appends).
Battery final: 205 cases x P/L/LF = 615 runs, 0 silent, 28 loud = only A+B.
Perf-pass cheap win: PARENT_FULL->REDUCED consumer inside grouped stage
splits off into its own kernel.
2026-09-02 REVIEW of two uncommitted diffs (doc:
agent_space/pr191974_followup_and_padded_review_20260902.md). (1) Correctness
follow-up on simplified wt: crash A fix = merge_loops skips
FusedNestedReductions (APPROVE, verified dyn mark->1 kernel correct). Crash B
fix = decline required-live sources written by outer-reduction ANCESTORS
(interim APPROVE with condition): it declines for ALL policies (persistent
residual case regressed 1->2 kernels at D=512; tests pin that; flip back
after reload fix) and the predicate is TOO NARROW — planner instrumentation
showed a non-ancestor parent-stage source (realized sigmoid(h) sibling) is
ACCEPTED 6/6; no crash only because the pack had already fused via the
standalone staged path and "staged reduction plan would be lost" refused to
nest an existing FSR (fusion-order race). Broaden to the displacement
predicate (outer node, not reduction, no outer-reduction ancestor) + add a
planner-level mock unit test. "nested=1 kernels=2" in these fallbacks =
nested[prologue,rms,block amax] + pack pointwise kernel (x=B*D/2, r0=1).
(2) Padded-scatter diff (padded wt): index_put masked-fill -> ir.Scatter with
store_mask (predicated tl.store, no destination read), feature-gated
MASKED_SCATTER_WITH_INDEX (Triton only), scheduler refuses consumer fusion
into groups with a predicated scatter because CSEProxy.store populates
store_cache unconditionally. APPROVE; verified can_fuse control flow reaches
the rule for staged/nested plans; suggest CSEProxy.store skip store_cache
under active _load_mask as the precise invariant; register_users_of gap is
hygiene only; rebase onto follow-up. Harness note: test_torchinductor needs
tmp_run_with_nms_stub.py (torchvision::nms) under run_wt.py; unittest -k
takes multiple flags (OR), not "a or b" expressions.
2026-09-02 ROUND: WIDENED CRASH-B GUARD IMPLEMENTED BY ME in
pr191974_simplified_wt (uncommitted, alongside the other session's A/B
follow-up). Predicate = required-live source written by an outer node that is
neither a reduction nor LOCAL_REDUCTION_INPUT (= emitted inside the parent
loop). Pinned by test_nested_sub_parent_rejects_parent_stage_live_source
(mock, 3 writer roles x cpu/cuda) and
test_nested_sub_parent_rejects_parent_stage_sibling_source (e2e, both
configs; realized sigmoid(prologue) gate read by the pack). Verified: real
plan now returns None on the sibling topology (was accepted 6/6); all crash
probes cured; nested 413 = 412 green + openssl env; scheduler 150 OK; spin
quicklint clean (fixlint applied); battery rerun of the 11 formerly-loud cases
x P/L/LF = 31 PASS + 2 PASS* (bf16 cast placement; emu siblings byte-eq).
Trade-off retained: persistent residual case split into 2 kernels until the
reload fix. Battery driver: tmp_temporal_driver.py --modes P,L,LF --jobs 8
--out <json> <case...>; reclassify with tmp_temporal_table.py <json>.
NEXT: reload prototype (fork), design constraint: physical reload of an
internal source ONLY when name in kernel.cse.invalidated_stores (store in a
closed earlier loop scope); same-flush miss stays a loud assert (loads
precede stores in a flush, so a same-flush reload would read stale memory).
2026-09-02 LANDING CHECKS on simplified wt (HEAD 61b333901fa + uncommitted
A/B follow-up incl. widened guard): kernel hash corpus
(tmp_kernel_hash_corpus.py drives every test_nested_reduction
_capture_*_sources helper x persistent True/False, normalized sha) = 30/30
IDENTICAL vs clean baseline worktree pr191974_base_wt @61b333901fa. Perf A/B
(bench_nested_freeze.py --mode current --coordinate-descent, CUDA graphs):
r_block_quant_128x4096 2.180->2.181us (+0.07%), r_swizzle_128x4096
2.325->2.344us (+0.81%, noise), x_parent/x_local cases unfused on both.
Amended commit message draft: pr191974_failure_investigation/
PR191974_COMMIT_MESSAGE_v2.md (adds A/B follow-up paragraphs, reload
follow-up disclosure, bf16 cast-placement note; trailers preserved). gh token
on this box is INVALID (HTTP 401) — user must `gh auth login` before CI can
be pulled. Reload prototype agent running in pr191974_reload_wt.

2026-09-01 CRASH-B GUARD SCOPE (decided): the widened decline is
unconditional, not looped-only, even though the ask was to preserve the
persistent one-kernel case. Reason, measured on prologue_realize_noout via
run_wt.py: base @61b333901fa crashes at D=512/persist=0 AND at D=4096 with
triton.persistent_reductions=True (codegen still picks a looped kernel above
the persistent RBLOCK limit). So the config flag is NOT the persistence
choice; a planner-side prediction would have to replicate
should_use_persistent_reduction (features + tiling scores + cooperative +
RBLOCK) and any divergence re-raises "lost required sub-parent source" as a
user-visible compile error. Cost of the unconditional decline is exactly one
schedule: D=512/persist=1 goes 1 kernel -> 2. Reload follow-up removes the
decline entirely and needs no prediction.
2026-09-01 TEST VACUITY CHECK (do this for every new reject test): disable
the guard (`if False and ...`) and re-run. Result: the 4 prologue instances
all change (looped -> ERROR "lost required sub-parent source", persistent ->
1 kernel so the count assert fires), but the sibling-source test passes
either way -- it is declined by the pre-existing staged fusion gate, not the
planner guard. Renamed to test_nested_sub_parent_parent_stage_sibling_source
with a comment saying so; real guard coverage is the parametrized writer_role
unit test in test_inductor_scheduler.py.
2026-09-01 CSE.clone() drops invalidated_stores and scoped_copy() does not
restore it, so inside a swap_buffers scope the set is empty, disabling the
must_keep_buffers pin in CSEProxy.load and the tl.debug_barrier() in
TritonKernel.load for loop-invalidated names. I first called this a Triton
hazard for ops.masked bodies; that was WRONG and verified so: swap_buffers
has callers only in cpp.py (1056/1836/1864) and mps.py (248), and Triton's
TritonKernelOverrides.masked runs the body under mask_loads, which only sets
_load_mask/_load_other and never replaces kernel.cse. So the Triton path
always reads the real set. The mismatch is latent in cpp/MPS only. Do not
fold the clone()-sharing fix into a nested-reduction PR: its only live target
is cpp/MPS, which no nested test covers.

2026-09-01 later: team-lead accepted the premise correction. CSE.clone()
sharing is parked as tmp_reload_logs/03_cse_invalidated_stores_sharing.diff (common.py +
test_codegen_triton.py unit test) and reverted from pr191974_reload_wt, which
now holds follow-up + 01_reload_path (4 files) + 02_nested_fusion_gate. The
masked pad-then-slice regression test passes 4/4 without 03. Finding to keep
prominent: cat/stack masked epilogues never reach the reload path (declined
at _nested_sub_parent_rate or by the lane proof), so pointwise_cat forms are
unreachable by construction. Write-up: RELOAD_PROTOTYPE.md.

## 2026-09-04: multi-kernel support (uncommitted in the main tree)

Nested reduction had hard-disabled multi-kernel at two codegen sites
(`simd.py` `_codegen_nested_reduction` and
`_codegen_reduction_with_sub_parent_epilogue`), each passing
`"disable_multi_kernel": True` and taking `create_kernel_choices(...)[0]`. Both
lines came from #190595. Removing them and looping the per-kernel body over
`create_kernel_choices(...)` just works -- the body codegen is already
re-runnable per kernel choice, exactly like `codegen_node_schedule`. The
finalizer became `_finalize_nested_reduction_kernels` (merge workspaces, define
each, wrap in `MultiKernel` when >1). The `disable_multi_kernel` kwarg in
`triton.py:create_kernel_choices` had no other user and was deleted.

Facts worth keeping:

- Only **two** forms are ever produced (looped + persistent), and only when the
  base kernel is persistent -- `add_multi_kernel_choices` offers a
  non-persistent alternative to a persistent base, never the reverse. Nested
  reduction forces `override_cooperative_reduction: False`, so the cooperative
  axis is dead by construction. Tests must expect 1 kernel under
  `force_persistent_outer_reduction is False` and 2 under True.
- **cpp_wrapper + multi_kernel needs `triton.autotune_at_compile_time=True`**
  or you get `AssertionError: expected multi_kernel_name multi_kernel_0 to be
  recorded during cpp-wrapper codegen`. This is generic inductor behaviour, NOT
  a nested-reduction bug -- reproduced identically with
  `triton.nested_reduction=False`. `test_multi_kernel.py`'s
  `make_cpp_wrapper_test` patches both. Do not re-debug this.
- Test-harness trap: `TRITON_KERNEL_RE` chunks run from one `@triton_heuristics`
  to the next, so a chunk **trails the following kernel's definition header**
  and its name. The old substring filter therefore matched the looped kernel
  against the `triton_per_fused` signature once two forms existed. Fixed by
  filtering on `_kernel_name(chunk).startswith(signatures)`.

Validated: test_nested_reduction 428 OK (skipped=8), test_multi_kernel 21 OK,
test_loop_ordering 126 OK, lintrunner clean. New tests: parametrized
`test_multi_kernel` numerics in `_NestedReductionBase`, wrapper-level
`multi_kernel_wrapper_checks` on the two `*_multi_kernel_form` tests (which
previously asserted single-form), and `test_rmsnorm_block_amax_multi_kernel` in
the AOTI class.

## 2026-09-05: four independent changes, four branches (all from origin/main 071dd4d98ee)

- `nested-reduction-lane-fold` @ agent_space/lanefold_pr, commit 6d243f4900c -- validated (nested 419, lo 125, ti identical to baseline). See [[sub-parent-lane-fold]].
- `nested-reduction-multi-kernel` @ agent_space/multikernel_pr, commit a30f04c119d -- validated (nested 422, multi_kernel 21, lo 125).
- `scheduler-reorder-fixpoint` @ agent_space/fixpoint_pr, commit adb23a4d464 (nested 419 OK, lo 125 OK, ti pending at commit time); regression test `test_standalone_flat_reshape_pack_fuses` passes on branch, fails on main. NOTE: the test formulation needs the torchao `where(isnan(amax), 255, ...)` on the e8m0 scale; without it the scale inlines into the codes node and the standalone planner declines ("sub-parent epilogue planning failed") -- a different, pre-existing limitation. Low priority per the user: only matters for torchao-style code.
- `nested-reduction-mutation-hoist` @ agent_space/mutfix_pr, commit e5196889c55 (older origin/main base).
Commit messages: agent_space/peak_cmp/{lanefold,multikernel,fixpoint}_commit_msg.txt.
The main tree still carries all four uncommitted and mixed; #195874 (crash fixes) is NOT on origin/main.
Worktree lint: pyrefly reports F.pad/_VF stub errors in unrelated files (unbuilt .pyi); ignore, main-tree lint is clean.

2026-09-06 refreshed dual-dim baseline on the main tree (lane fold + mutation
fix + fixpoint + multi-kernel present; cd; probe
agent_space/peak_cmp/dual_dim_status.py): 16384x7168 rms+rowwise 1 kernel
73.7us; rms+colwise 3 kernels 175.9 (nested=0 -- colwise alone never engages);
rms+dual 3 kernels 226.8 (was 267 in the 2026-08-22 proposal; handwritten
dual 83.7 kernel-only, band 120-150); dual no-producer 3 kernels 170.5 (vs
handwritten 95.1 wall). 8192x4096: rowwise 16.5, colwise 42.7, dual 51.5,
dual-no-rms 39.0. Largest unrealized gap in the whole effort (1.7-2x); the
B1-B4 plan in blockwise2d_design_proposal.md is still draft/unreviewed and
unimplemented. User's framing 2026-09-06: "the nvfp4 combined dim0/dim1 cast
in training". Heuristics/configs explicitly out of scope.
