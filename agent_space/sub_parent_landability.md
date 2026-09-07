# Per-Commit Landability

Question: is each commit landable on its own without exposing a correctness or
perf problem that a later commit fixes?

Answer: **yes**, for the committed stack `C1 packing+fix -> C2 generalize ->
C3 contiguous`, with one mechanical prerequisite (squash the duplicated packing
commit; see end).

Base: `96ac987dbfb`. Committed stack:
- C1 `996f03` "Fuse NVFP4 nested-reduction packing" (= 96ac + the leaf fix)
- C2 `8e6ab`  "Generalize half-resolution epilogues to sub-parent"
- C3 `d59fc87` "Fuse contiguous sub-parent epilogues"

## Verification performed

- Tests green at each stage (applied incrementally on a fresh 96ac worktree):
  C1 = 214 passed, C1+C2 = 216, C1+C2+C3 = 246.
- Real-workload perf at HEAD (min of 3 do_bench, bf16): standalone-NVFP4 and
  producer-consumer-NVFP4 across B/D are all wins (1.27x-2.27x), `nested=1`,
  including the cases that run looped (still 1.8-2x). No regression.
- Correctness: ~50 adversarial cases over the session, zero miscompiles in the
  current code; the read-before-write fix lives in C1.

Note on a false alarm: a synthetic pattern (norm + ungrouped stride-2 even/odd)
appeared to regress 10-100x, but it does **not** take the nested path
(`codegen_nested_reduction == 0`) -- it is ordinary strided-slice codegen,
unrelated to this stack. Real interleaved workloads (grouped) are all wins.

## C1 -- packing + leaf fix

- Correctness: the fix is *in* this commit, so the half-output read-before-write
  miscompile is never exposed. The fix only adds guards (rejects unsafe
  fusions); it does not change codegen for the safe/win cases.
- Perf: NVFP4 grouped packing is the headline win. At sizes where the heuristic
  does not pick persistent, C1 simply does not fuse (the original gate falls back
  to unfused) -- baseline, not a regression. There is no looped path in C1.
- Landable: yes. A graph that previously fused-and-miscompiled now falls back to
  a correct kernel; "the wrong kernel was faster" is not a regression.

## C2 -- generalize to sub-parent (interleaved) + looped

- Correctness: green (216). Generalizes factor-2 -> factor-N for INTERLEAVED and
  adds the looped codegen path; all interleaved/looped adversarial cases correct.
- Perf: the only perf-sensitive new path here is looped, and it is INTERLEAVED-
  only in C2 (contiguous arrives in C3). Interleaved carries the
  `min_xblock = 128` amortization floor, so looped-interleaved stays profitable
  -- confirmed by the looped real-workload wins at HEAD (1.8-2x). C2 turns the
  "don't fuse at large D" cases from C1 into looped wins; that is an improvement,
  not a regression.
- Why HEAD perf is a valid proxy for C2 here: C3 does not touch the interleaved
  path (its guard returns early for non-contiguous, and `min_xblock = 128`
  remains set for interleaved). So interleaved perf at C2 == interleaved perf at
  HEAD = wins.
- Landable: yes. No contiguous cliff is present in C2, and the looped path it
  introduces is amortization-protected for interleaved.

## C3 -- contiguous chunk/SwiGLU + profitability guard

- Correctness: green (246). Adds the contiguous layout, proof, materializer
  branch, and lane selection; all chunk/SwiGLU and rejection cases correct.
- Perf: contiguous lacks interleaved's `min_xblock` floor, so it could regress at
  large D / small batch -- but the **profitability guard ships in this same
  commit** (`_sub_parent_epilogue_layouts_profitable`: fuse all if rnumel<=4096,
  else require numel>=128). So the regression-prone shapes never fuse; the guard
  and the path it guards land together.
- Landable: yes. The contiguous perf risk is self-contained: there is no window
  where contiguous exists without its guard.

## Conclusion

Each commit is independently green and self-contained: no commit exposes a bug
or perf regression that a later commit cleans up. The fix is in C1, the looped
path is amortization-safe for interleaved in C2, and the contiguous guard is in
C3 with the contiguous path.

## Mechanical prerequisite before submitting

The packing PR commit currently appears twice in history: 96ac and C1 (`996f03`)
share the same `ghstack-source-id` and `Pull-Request: #183638`, with C1 stacked
on top of 96ac instead of replacing it. #183638 is open, not on `main`. Squash
96ac and C1 into one commit (keep C1's message + the `#183638` trailers) so the
stack is one commit per PR:

```
base -> [packing + leaf fix, #183638] -> [generalize, new PR] -> [contiguous, new PR]
```

Also: the `agent_space/sub_parent_split_patches` series and `split_plan.md` are
stale (they predate the final edits); regenerate from the commits or delete so
there is one source of truth.
