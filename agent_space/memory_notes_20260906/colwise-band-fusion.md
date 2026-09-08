---
name: colwise-band-fusion
description: 2026-09-06 EXPERIMENT (worktree agent_space/colwise_pr, uncommitted, flag nested_reduction_allow_x): X-grouped band fusion for RMSNorm->column-wise MXFP8 needs only two planner changes; numerics fine; -28% at two shapes, +13-18% at two others with cd; default config bad
metadata:
  type: project
---

User framing 2026-09-06: "the nvfp4 combined dim0/dim1 cast in training";
MXFP8 first as the simpler version (no lane packing). Baseline today:
RMSNorm->col MXFP8 = 3 kernels, nested never engages; RMSNorm->dual NVFP4 = 3
kernels 227us at 16384x7168 vs handwritten 83.7 kernel-only.

**Why it never engaged:** `NestedReduction.plan` returns None for any grouped
axis other than R ("splitting X forces a minimum XBLOCK and has consistently
lost", introduced by abaacefd143 #191974), and even past that gate the two
divisibility checks after it and `_r_grouped_stage_accesses_match` are
R-geometry only. The emitter side (`local_reduction_in_r=False`, min_xblock)
already works.

**Changes made (uncommitted in agent_space/colwise_pr, branch
nested-reduction-colwise-mxfp8, origin/main 071dd4d98ee):** config flag
`triton.nested_reduction_allow_x` (default False); gate relaxed under the
flag; axis-aware divisibility check (X: iter_ranges == (numel/G, rnumel));
X coordinate frame in the access proof (group_x, local_x, parent_r; grouped
values (group_x, parent_r, local_x); parent (group_x*G+local_x, parent_r)).
Result: 1 nested kernel = the (d2) band form (XBLOCK>=32 rows, K looped
twice, 32-row max in-tile). Scales byte-identical, 14/33.5M payload bytes
differ by one e4m3 step (rstd reduction order), same dequant error.

**Perf (rms->col MXFP8, us; 3-kernel cd vs band cd vs band default):**
16384x7168 208/146/349; 32768x4096 228/161/319; 65536x2048 228/257/266;
65536x8192 922/1090/1236; 8192x4096 47/63/118 (grid = M/32 programs starves
small M). Wins where the 32-row band stays L2-resident for pass 2 and the
grid is large; loses at K=8192 (512KB bands) and at K=2048; default config
is bad everywhere (no heuristic knows the band geometry).

**Decision pending:** recommended to the user to keep it flag-off and land
the four PRs first; the band heuristic/config work is what would make it a
default win, and the user scoped heuristics out. Not in any test suite yet.
Probes: agent_space/peak_cmp/colwise_probe.py, colwise_trace{,2,3}.py,
colwise_x_probe.py, colwise_x_check.py, colwise_x_shapes.py,
dual_dim_status.py (refreshed dual baseline).

**Gluon band prototype (2026-09-06, agent_space/gluon_proto/rms_colwise_mxfp8_gluon.py):**
best config 8x8 per thread, 8 warps, KC=512. rms->col MXFP8 us (3-kernel cd
/ Triton band cd / Gluon): 8192x4096 47/63/25.7; 16384x7168 208/146/126;
32768x4096 228/161/127; 65536x2048 228/257/113; 65536x8192 922/1090/550.
Correct to rounding. Gluon band wins everywhere (1.6-2x vs status quo), incl.
where the Triton band loses -> geometry right, Triton codegen (row-major
layout makes the 32-row max cross warps) + config is what loses. Row-wise
Gluon prototype showed NO gain (see sub-parent-lane-fold.md). Conclusion: a
Gluon emitter is justified for the band/dual (training) geometry only.
