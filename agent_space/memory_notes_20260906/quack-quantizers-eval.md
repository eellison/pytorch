---
name: quack-quantizers-eval
description: 2026-09-04 QuACK 0.6.4 pure-PyTorch quantizers as a nested-reduction acceptance test; harness paths, perf table, the four Inductor gaps found, and the fixpoint reorder-round fix
metadata:
  type: project
---

2026-09-04: the user asked to "try quack's quantizers" as one last acceptance
test. `quack.blockscaled.quantize` (QuACK 0.6.4, installed in pytorch-3.12
with nvidia-cutlass-dsl 4.6.2; 0.2.10 had no quantizers and did not import) is
a torchao port: pure PyTorch with `torch.compile` handles. Nine registry
entries (mxfp8 e4m3/e5m2, mxfp4, mxfp4_byte, mxfp6 e2m3/e3m2 plain+packed,
nvfp4) plus `to_mx_dim0` and the 128x4 `to_blocked` swizzle.

Harness: `agent_space/quack_quant/` -- `probe.py M K` (fusion + byte-exactness
on/off/eager, in-process), `cell.py` + `run.py` (timing matrix, one cell per
process, GPUs 2-7, CUDA graph of 100 calls), `analyze.py`, `results.jsonl`
(189 cells). Modes: eager/off/on/on_mk/on_looped/on_persistent (+_dynamic).
`out_hash` was fixed to sha1 mid-way; the first matrix's eq(on,off) column is
garbage (salted str hash), exactness comes from the probes instead.

Results (B200, median us, RMSNorm -> quantizer, 8192x4096 bf16):
mxfp8 off 30.2 / on 32.8 / persistent 24.1; mxfp4 60.7/47.6/39.5; nvfp4
63.7/49.1/41.6; mxfp6 byte 46.8/49.3/37.8; mxfp6 packed 100/105/94.
Byte-exact on==off in all 90 probe cases across three shapes; no crashes,
including the dynamic=True MXFP6 paths. eq_eager=False on RMSNorm rows is
65/6.3M bf16 elements at 1 ulp (5 MXFP8 codes), same with nested off.

Findings, in priority order:

1. **Persistent threshold miss.** For K >= 2048 the nested kernel is looped
   (INNER threshold 1024), and looped nested is slower than not fusing for
   MXFP8/MXFP6. Forcing persistent reproduces the multi-kernel pick exactly
   (multi-kernel log: looped 37.7 vs persistent 26.5us). Either enable
   multi-kernel for nested reductions or raise the threshold for them.
2. **Reorder post-pass runs one round.** `Scheduler.fuse_nodes` iterates plain
   rounds to a fixpoint but the `is_reorder_round=True` pass once. QuACK's
   `to_mxfp4` computes codes from the flat `(M, K)` reshape, so the scale load
   carries `FloorDiv(d1, 32)`, `score_fusion_memory` sees "no shared data",
   and only the reorder round fuses reduction+codes; the pack then never gets
   retried. Fixpoint version (applied to scheduler.py, uncommitted): standalone
   to_mxfp4 2 -> 1 kernel, 10.2 -> 7.4 / 44.1 -> 32.3 / 177.9 -> 123.2us,
   byte-exact. Validated 2026-09-05: test_torchinductor failure set byte-identical
   to a same-tree baseline (104 known CPU/AOTI CppCompileErrors both arms),
   test_loop_ordering 126 OK, test_nested_reduction 428 OK. It sits in the
   same uncommitted scheduler.py as the mutation-hoisting hunk; split before
   committing.
3. **dynamic=True makes block_size symbolic.** Dynamo turns the `block_size=32`
   default into a SymInt (`l_block_size_` placeholder) under dynamic=True, so
   `r0_numel` is a runtime arg, the reduction can never be persistent, and
   the `(M, K)`-shaped codes cannot fuse. QuACK ships MXFP6/mxfp4_byte handles
   with dynamic=True: 131us vs 5.9us static (22-25x) -- their own "24x"
   comment. With a literal block size but symbolic M,K it is still 2 kernels
   (82us): Inductor cannot prove `s27 % (s27//32) == 0`. QuACK-side fix is
   dynamic=False; Inductor-side is a divisibility-replacement improvement.
4. **NVFP4 standalone pack.** `_sub_parent_broadcast_access_relations`
   compares raw loop frames; after reordering the pack has one merged axis vs
   the scale writer's two, so `num_vars` mismatches (and the odd lane reads
   `(2*d0+1)//16`). Needs normalization into the (x, child_r) domain like the
   source-relation path. 3 kernels today, one of which is `lift_fresh` for
   QuACK's `torch.tensor(1.0)` per-tensor scale.

Also: packed MXFP6's `torch.stack` 6-bit pack never fuses (known pointwise_cat
limitation) and costs as much as the quantize; `to_mx_dim0` never engages
nested (2-3 kernels); `to_blocked` after any quantizer adds exactly one kernel.

**Why:** this is the first external, real-world quantizer codebase run through
the nested-reduction stack end to end, and it found a heuristic miss that the
DCN graph never showed (K=3072 there is also looped, 7.5 vs 6.0us).

**How to apply:** rerun `python agent_space/quack_quant/run.py` after any
scheduler or heuristic change; compare with `analyze.py`. Use `on_persistent`
to see the ceiling without multi-kernel. Do not time eager NVFP4 in a CUDA
graph (the fresh scalar tensor breaks capture; the cell falls back to events).

Related: [[nested-reduction-stack]], [[bench-variant-isolation]],
[[padblocked-layout-bench]].

## 2026-09-05: peak vs peak (agent_space/peak_cmp/, REPORT.md there)

The user reframed: compare OUR optimal torch.compile formulation against THEIR
hand-written kernels, and record fusion misses. QuACK has no CuTe quantizer
outside GEMM epilogues (its `blockscaled.quantize` is torch.compile'd torchao;
the docstring's fused-RMSNorm-quant forward exists neither in 0.6.4 nor
upstream main 2026-08-30). So "theirs" = flashinfer fused CuTe
`rmsnorm_fp4quant`, flashinfer TRT-LLM `fp4/mxfp4/mxfp8_quantize`, QuACK CuTe
`rmsnorm_fwd`, QuACK `gemm(out_dtype=nvfp4|mxfp8)` fused SFD epilogue.

Results (ours best / theirs): MXFP8 and MXFP4 we lead everywhere (fused
MXFP4 vs fi fused 0.80-0.87x; MXFP8 0.51-0.60x vs composed). NVFP4 is the gap:
fused 1.16x at 8192x4096 (24.6 vs 21.2), standalone 1.35x/1.21x at
8192x4096/65536x2048 (18.6 vs 13.8; 82.3 vs 67.9), tie elsewhere. RMSNorm alone
within +-7% of QuACK/flashinfer. GEMM quant-out 9-15% behind QuACK's fused
epilogue (cuBLAS mm + separate quant kernel; QuACK's fused nvfp4 GEMM is as
fast as bf16 mm). Standalone quantizers are byte-identical to flashinfer.

Recorded misses: (1) GEMM + block-scaled quant epilogue unfusable (reduction
epilogue on mm); (2) FP4 kernels at ~4.5 TB/s vs 6 TB/s for our MXFP8 and fi
NVFP4 -- x loaded once, so suspect `tl.reshape`+`tl.split` layout conversion,
needs ncu; (3) form choice is shape/format dependent and multi-kernel's pick
can be worse than both single forms (nvfp4 8192x4096 mk 30.8 vs 27.2/28.8);
coordinate descent is the bigger lever (default vs best 1.4x at 65536x2048).

**How to apply:** `python agent_space/peak_cmp/peak_run.py` then
`peak_analyze.py`; always quote ours-best (cd or persistent_cd), never default,
when claiming parity. Do not use event timing for their kernels: smoke-test
event numbers were 2-3x the graph numbers (launch overhead).
