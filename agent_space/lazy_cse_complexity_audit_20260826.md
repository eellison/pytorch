# F2a narrow group-width cast complexity audit

Date: 2026-08-26

Worktree:
`/data/users/eellison/pytorch/agent_space/followup_lazy_projection_wt`

Final snapshot:

- HEAD: `6b8ef64bd373b221c6ed1ba4f4b2f1edef2643c3`
- unstaged diff SHA-256:
  `7ab016dc72657354c894e0252d74bc3d0bd8a9c5f4ab35423bf36785236e82b0`
- production `simd.py` diff SHA-256:
  `a165db8f8e54db06476b9a2828e0df4aed41afd661364e2fc4a2012b81dcc3ab`

## Verdict

Approve. The adopted F2a is a single cast-before-widening rule, not a general
lazy-expression framework. Its production delta is `+71/-4`, net `+67`, all in
`simd.py`. It adds no type, wrapper, cache, enum, module helper, scheduler
policy, or public API.

The implementation adds six small methods to existing classes. Its new
dispatcher `_default` is six lines at CC 2/nesting 1. The only algebraic
exception is the typed `to_dtype` override at 15 lines, CC 3/nesting 1. The
resolver remains the most sensitive method at 21 lines, CC 7/nesting 2, because
it retains the existing source-alternative, liveness, and required-miss logic.

The final design is substantially smaller than the superseded general F2a
prototype while generating the same code for all 24 measured NVFP4/MXFP4
shape-mode cases.

## Method

Measurements use `agent_space/complexity_audit_ast.py` with the same rules as
the F1 audit: control-flow branches include boolean terms and comprehension
branches, CC is one plus branch points, and nested local functions are excluded
from their parent's method inventory.

Raw result:
`agent_space/f2a_narrow_complexity_20260826.json`.

## F2a-only delta versus staged F1

| Metric | Staged F1 | Narrow F2a | Delta |
| --- | ---: | ---: | ---: |
| Physical production lines | - | - | `+71/-4`, net `+67` |
| Functions/methods | 1,230 | 1,236 | +6 |
| Class definitions | 117 | 117 | 0 |
| Function-body LOC | 26,658 | 26,718 | +60 |
| Branch points | 4,431 | 4,444 | +13 |
| Cyclomatic complexity | 5,661 | 5,680 | +19 |
| Maximum nesting | 8 | 8 | 0 |

On the changed-function surface, four F1 methods become ten F2a methods:
function LOC moves from 92 to 152, branches from 17 to 30, aggregate CC from
21 to 40, and maximum nesting remains 2.

Mechanism inventory:

| Surface | Narrow F2a delta |
| --- | ---: |
| Private types | 0 |
| Module helpers | 0 |
| Methods | +6 |
| Policy sets | 0 |
| Caches or deferred-value wrappers | 0 |
| Public API | 0 |

## Per-function metrics

`B/F` means staged F1 baseline / final narrow F2a. A dash denotes a new method.

| Symbol | LOC B/F | Branches B/F | CC B/F | Nest B/F |
| --- | ---: | ---: | ---: | ---: |
| `_GroupedReductionLayout.parent_dim` | 11/10 | 4/4 | 5/5 | 1/1 |
| `_GroupedReductionLayout.materialize_value_at_sub_parent_resolution` | 56/56 | 10/10 | 11/11 | 1/1 |
| `_PointwiseRemapHandler.store` | 11/12 | 0/0 | 1/1 | 1/1 |
| `_PointwiseRemapHandler._is_group_width_value` | -/7 | -/2 | -/3 | -/0 |
| `_PointwiseRemapHandler._materialize_group_width` | -/5 | -/2 | -/3 | -/1 |
| `_PointwiseRemapHandler.to_dtype` | -/15 | -/2 | -/3 | -/1 |
| `_PointwiseRemapHandler._default` | -/6 | -/1 | -/2 | -/1 |
| `_SubParentValueResolver.is_group_width_shape` | -/7 | -/1 | -/2 | -/0 |
| `_SubParentValueResolver.materialize_group_width` | -/13 | -/2 | -/3 | -/1 |
| `_SubParentValueResolver.resolve_load` | 14/21 | 3/6 | 4/7 | 2/2 |

## Why the remaining code is necessary

### Exact raw-source escape

`resolve_load` may return a live source without materialization only when the
planned relation is lane-free, the consumer has no `ops.masked` guard, and the
ordinary CSE shape is group-width rather than direct child-width. The check is
intrinsic to the resolver; there is no caller-supplied callback or flag.

Source ordering, CSE liveness, later alternatives, and loud failure for a lost
required source remain in the existing loop. Removing any of those would
weaken F1 correctness rather than simplify F2a.

### One typed algebraic exception

`_PointwiseRemapHandler.to_dtype` is the only operation allowed to consume a
group-width value. It delegates with the full OpsHandler signature, verifies
that the result stayed group-width, then immediately uses the existing
sub-parent materializer. This is the exact operation ordering needed by NVFP4:
FP8-to-FP32 conversion before broadcast.

Every other operation flows through `_default`, which recursively materializes
operands before ordinary codegen. `store` independently enforces the same
boundary. This makes the policy visible in code instead of encoding it in a
string allowlist.

### Shared shape and mask logic

The resolver reuses `_GroupedReductionLayout.parent_dim(shape)` and
`materialize_value_at_sub_parent_resolution`. The latter already assigns masks
from the target child shape. F2a contains no copied broadcast or mask
calculation, and the `G == factor` direct-child case retains precedence over
group-width classification.

## Superseded general prototype

The first F2a implementation supported a positive allowlist of 16 arithmetic,
comparison, and selection operations, pure pack-1 inline asm, canonical shape
preflight, and a masked callback wrapper. It was correct and passed the same
full suites, but cost `+119/-4`, net `+115`, with aggregate F2 CC `+26`.

That version is retained only as design history. Across the measured corpus:

- only NVFP4 changed generated code;
- every NVFP4 change was exactly `to_dtype` before broadcast;
- MXFP4 and all protected sources were unchanged;
- every delayed target source was unguarded and lane-free.

The narrow implementation reproduces all 24 generated kernels and their
performance while deleting 48 net production lines, the policy set, the shape
preflight, and the masked callback protocol. General group-width arithmetic is
therefore not justified in this change.

## Follow-ups only

- Add another explicit group-width operation only when a measured workload
  requires it and a kernel-form test proves the intended ordering.
- Add masked-source deferral only with a concrete guarded workload; the current
  rule intentionally materializes guarded sources eagerly.
- If sub-parent reduction nodes become legal, add an explicit remapped
  `store_reduction` implementation then. Current planning rejects that case.
- Keep parent-width/F2b deferral separate and subject to its own performance
  gate.

Final assessment: the narrow implementation meets the requested behavior with
an explicit, reviewable rule and no unused general policy surface.
