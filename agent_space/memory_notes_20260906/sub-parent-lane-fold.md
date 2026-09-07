---
name: sub-parent-lane-fold
description: 2026-09-05 lane-fold codegen fix in _SubParentValueResolver; RMSNorm->NVFP4/MXFP4 20-27% faster and ahead of flashinfer's fused CuTe kernel; how it was found (ncu + TTGIR), the two required parts, and the branch/worktree state
metadata:
  type: project
---

2026-09-05, commit 6d243f4900c on branch `nested-reduction-lane-fold` in worktree
`agent_space/lanefold_pr` (based on origin/main 071dd4d98ee; fold-only hunks of
simd.py + two test edits; also present uncommitted in the main tree mixed with
the multi-kernel hunks). Commit message draft:
`agent_space/peak_cmp/lanefold_commit_msg.txt`. VALIDATED 2026-09-05 on the
branch via run_wt overlay: test_nested_reduction 419 OK, test_loop_ordering
125 OK, test_torchinductor failure set identical to the clean origin/main
baseline (104 env CppCompileErrors). Not pushed; awaiting the user's read +
ghstack. Worktree pyrefly reports unrelated F.pad typing errors (unbuilt
stubs), main-tree lint is clean.

**What was wrong.** Fused RMSNorm->FP4 kernels moved the same 67MB as
flashinfer's fused `rmsnorm_fp4quant` but ran 47% more instructions, 52 regs
vs 32, 30x more shared-memory stores (ncu, 8192x4096). TTGIR showed two causes:
(1) the sub-parent lane replay split the RAW x and w loads and recomputed
x*rstd*w per lane (lowering had inlined the normalized value into the pack
node), (2) the per-group scale reached lane width via its own reshape and
landed in a different Triton layout than the split data, so every multiply
paid a full-tile `convert_layout` through shared memory.

**Both parts are required.** Hand kernels proved it: split-normalized-value
alone (V2) 27.6us, grouped layout with recompute (V1b) 28.5us, both together
(V1/V1d) 19-21us vs generated 27.5us and flashinfer 21.2us.

**The fix** (`_SubParentValueResolver`): `_default` tries
`_try_fold_lane_op` -- if every CSE arg is a lane projection (same lane) or
lane-invariant and `parent_handler.<op>(parent_args)` is already in
`kernel.cse` (string key) and live, return a *pending* placeholder
(parent_value, lane); `_resolve_pending` splits on first non-foldable
consumer or store. `materialize_source` defers external-load lanes the same
way, and `materialize_sources` no longer eagerly splits external sources.
`materialize_value_at_sub_parent_resolution` lifts group-width values with
`_broadcast_value_to_parent_resolution` then splits (lane 0 kept in
`materialize_group_width`, marked lane-invariant). `masked` bodies must keep
their `.graph` attribute when wrapped (first suite run: 26 errors from that).

**Results** (ours best vs fi fused): NVFP4 3.77/4.87/18.8/81.4 vs
4.43/5.75/21.2/109 (0.75-0.89x); MXFP4 3.32/4.79/17.7/79.4 vs 5.07/6.96/27.1/127
(0.62-0.69x). All 12 pre-change cells byte-identical (quant+scale sha). DCN
MXFP6 (4,3) unchanged: 9.60 -> 9.63us, identical outputs. Standalone quant
unchanged (user: "never standalone nvfp4").

**Why:** this is the first time nested reduction beats the hand-written fused
kernel on NVFP4 at every shape; the persistent-vs-looped question matters less
for FP4 now (looped+cd 18.8 vs persistent+cd 20.4 at 8192x4096).

**How to apply:** tests pin `tl.split(` counts; the new form is exactly two
splits for rms->fp4 (normalized value + scale broadcast) and the weight split
`[1, (R0_BLOCK//2), 2]` must not appear. run_wt.py now also overlays
`torch.testing._internal.inductor_utils` (main tree carries
`_device_is_available` which upstream reverted; without the overlay every
worktree test dies on import). Never `pkill -f` a string that is in your own
bash command line.

Related: [[quack-quantizers-eval]], [[nested-reduction-stack]],
[[worktree-overlay-harness]].

## 2026-09-05 later: MXFP8 fused kernel (user: "then look at the mxfp8 quant")

Lane fold does not apply (no pair split). ncu at 8192x4096: fused MXFP8 ran
25.5 instr/element vs 10.8 standalone quant and 12.3 flashinfer NVFP4 --
issue-bound, cd had picked 16 warps/row. Two levers found:

1. **Formulation: divide -> multiply by reciprocal.** `g / scale_f32` costs a
   full fp32 division per element; `g * recip_ue8m0(scale)` is bit-identical
   for e8m0 (power-of-two) scales. Fused persistent_cd 25.0 -> 21.0us,
   instr/element 25.5 -> 16.6, standalone 16.3 -> 15.4. peak_cell.py's
   `mxfp8_quant` now uses it. QuACK/torchao `to_mx` divide the same way.
2. **fp32 register residency** (user's observation): loads upcast to fp32 at
   the load, so a persistent row costs 2x registers (96 regs at 2 warps, 60 at
   4), forcing >=4-8 warps/row and cross-warp reductions; flashinfer holds 32
   regs with bf16 in smem. `triton.codegen_upcast_to_fp32=False` is NOT the fix
   (slower, 25.6 vs 22.2, and changes numerics because intermediate math goes
   bf16). The real change is codegen: keep the bf16 load live, upcast per use.
   Not implemented.

Config sweep of the generated fused MXFP8 (recip) persistent kernel: best
XBLOCK=1 nw=2 20.9us (96 regs), nw=8 22.7 (32 regs); cd's pick (~21.0) is
near-best, so config is not the lever. Remaining gap to roofline (~15us for
97MB) is fp32 residency + per-row w reload + cross-warp reductions.
Tested 2026-09-05: removing the load-time `.to(tl.float32)` from the generated
fused MXFP8 kernel leaves registers unchanged (96/60/32 at 2/4/8 warps) --
ptxas keeps the CSE'd fp32 value live. bf16 residency needs per-group
streaming or smem staging, i.e. a structural codegen change, not a load tweak.
Also: generated MXFP8 kernel at num_warps=2 runs 19.5us vs cd's 21.2 pick.
bf16-residency hand test (agent_space/peak_cmp/bf16_recompute.py, MXFP8 fused
8192x4096, non-pure asm cvt to defeat CSE): fp32-resident 21.0/22.5/26.0us at
2/4/8 warps (114/62/39 regs); bf16-resident with y shared 21.6/22.2/22.8
(98/60/32 regs); bf16-resident with y recomputed per pass 24.3/25.7/27.7 (128/60/39).
Registers are not the limiter (fastest config has the most); recompute costs
more than it saves. Do not pursue bf16 residency inside Triton's tile model;
the remaining MXFP8 gap to roofline is issue-bound (reductions, per-row w
load, reshapes) and would need a streaming/smem-staged design.
MXFP8 opcode histogram (2-warp hand kernel): 11.6 instr/elem = FMNMX 2.44
(~2 of it the redundant clamp before the satfinite e4m3 cast), IMAD 2.2 +
PRMT 1.0 (layout plumbing), FMUL 2.5, F2FP 0.56. Clamp removal is
bit-identical incl. inf/NaN under Inductor (PropagateNan clamp + satfinite
cast) and gives -4..-9% fused at fixed config, but the standalone
default-config kernel got slower (15.4->19.7, autotune config flip); left the
bench formulation with the clamp. Remaining gap to roofline is layout
plumbing + reductions + per-row w reload -> not an Inductor-side fix.

## 2026-09-06: Gluon row-wise prototype -- layout control is NOT the lever

agent_space/gluon_proto/rms_mxfp8_gluon.py (Triton 3.8 has Gluon; no Inductor
gluon support exists). Thread-owns-32-group layout: 24.3us graph / 11.6
instr/elem but 2x L1 load sectors (coalescing tax); coalesced+convert_layout
25.5us (smem round trip +2.4M instr); 16/thread 8 warps 22.5. Best Triton 2
warps 19.5-21.0. All correct. Row-wise kernels are at their practical floor;
the Gluon lever only applies to the column-group band/dual geometry. Do not
re-run this experiment; the numbers are in REPORT.md.
