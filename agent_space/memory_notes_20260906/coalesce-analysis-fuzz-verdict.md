---
name: coalesce-analysis-fuzz-verdict
description: "2026-09-03 triage of the five coalesce/tiling_utils fuzz findings: only #1 was a real crash (landed in #195874), #2-#5 are advisory-scoring only -- do not adopt them as a ride-along"
metadata:
  node_type: memory
  type: project
  modified: 2026-09-03
---

`agent_space/COALESCE_ANALYSIS_FUZZ_FINDINGS_20260903.md` reports five defects
in `torch/_inductor/tiling_utils.py`, with fixes and regression tests sitting
in the clean worktree `agent_space/dcn_mxfp6_5bd51` at commit
`5bd51f56358846780bd3d0f6ab2faa6543be474d` (uncommitted, +69/-29).

Verdict under the user's stated filter -- "I just want to fix crashes, not
necessarily every edge case of coalescing" -- is **nothing further worth
doing**, decided 2026-09-03. Finding #1 (`FloorDiv.is_constant()` compile
crash) is the only crash class, and it already landed byte-identical in
PR #195874; the worktree's crash-class regression test duplicates the one that
landed with it. Findings #2-#5 (unsound `ModularIndexing` zero solution,
unsound direct-term shortcut in `find_coalesced_var`, wrong collapsed-coordinate
remap in `apply_var_mapping`, buffer metadata lost on normalized-expression
collision) all terminate in `tiling_scores[v][tiling_factor] += addr_score`
behind three guards. That path is purely advisory ranking, so none of them can
produce wrong numerics or a crash -- only a differently-ranked tiling.

Corroborating evidence from the doc's own validation: 4,800 expressions,
40,000 factorizations, and 60 CUDA graphs all matched eager, 127/127
`test_loop_ordering`, and the DCN checks moved only within noise (unpadded
8.81 -> 8.90us, padded 23.85 -> 23.96us).

Active reason NOT to land #2-#5 opportunistically: `coalesce_tiling_analysis`
is `default=True` (justknob `pytorch/inductor:coalesce_tiling_analysis`), and
the `apply_var_mapping` change alters the *form* of every normalized address
expression. It would therefore shift tiling choices across unrelated kernels
repo-wide. That needs its own benchmark sweep and its own PR, not a ride-along
on the padded-blocked stack -- see [[to-padded-blocked-prim]].

The analysis and its three fuzzers are only preserved in that agent_space doc.
Filing a GitHub issue to preserve them was offered to the user on 2026-09-03
and explicitly NOT posted; nothing has been drafted.
