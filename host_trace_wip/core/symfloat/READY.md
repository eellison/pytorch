# symfloat lane: READY (symbolic float arithmetic in traced host code; design DESIGN.md, sections 0-9 plus 10 = as built)

## Patches (patches/)
| md5 | file | base | lines |
| --- | --- | --- | --- |
| 961c0798b21c | S1_symbolic_float.step288.int6cpp.patch | int6 snap/step288 (candidate 4 even top); byte-identical to the .step286 one | core +280 -18, tests +381 |
| 08ef98489136 | S1_symbolic_float.step281.int6.patch | int6 snap/step281 (odd) | same hunks |
| 0299fdc13020 | S1_symbolic_float.int6.patch | int6 snap/step246 (where it was built and tested) | same hunks |
| 9978616c405d | S1_symbolic_float_csrc.int6cpp.patch | torch/csrc/cuda/host_trace/Program.{cpp,h}; identical in build_cpp8/src, build_cpp9/src, snap/step288 (0 fuzz) | +59 -2 |
| 5413505c888c | F3_fork_fused_cost_order.patch | the attn lane's fork trtllm_trace.py md5 661e28f3e32b (F1 decline + F2 parity) | fork +23 -27, GPU test +152 |

S1 Python needs the csrc rows: Program.cpp rejects the new row names ("unknown op") at compile_program. The odd line's
older Program.cpp (snap/step281, md5 9046688059a2) would hit that only for a trace that uses the float API.

## What it is
- New rows `fadd fsub fmul f32i f32r f32add f32sub f32mul f32quot f32fma`, IEEE as the C++ host computes them. A
  float32 value is held as its double. No float domain checks.
- Python reference `float_value`; `f32_bits` now rounds an int once (the 2^53+2^29+1 bug).
- API `torch.cuda._host_trace.f32 / f32_add / f32_sub / f32_mul / f32_quot / f32_fma / argmin / FLT_MAX`. The names
  are `f32_quot` and the `f32quot` row because `f32_div` / `f32div` (int quotient bits) already exist.
- Comparisons are ordinary guards, so inside an op's host they are the op's. `argmin` records k-1 comparisons against
  the winner, plus the bound: strict before it, <= after it.
- The sympy export wraps each float operation in the existing `Identity`. The cover solver evaluates float IR nodes by
  the rows, not by the export.
- Fork F3: `_best_tile_q` is traced (no decline, no `choice`, no pin) in fmha_gen.so's order:
  `fmaf(R*128, kv, (factor*M)*seq) * waves`.

## Tests (land/core/symfloat/python_sf.sh = py tree on c/install, the isolated build_cpp8 + csrc rows, CPUs 108-143)
- test_cuda_host_trace_float.py 12/12 (GPU 1).
  - Rows vs numpy float32 and an exact-rational fma, Python vs C++ on 21k cases.
  - Guard program exact over max_kv x batch boxes around 4 hints (holds iff the same winner).
  - The 2049 exact tie decided by the fused order.
  - Ints past 2^24.
  - Identity survives simplify/expand/factor/cancel/together/doit/subs and numeric subs; rows evaluate in op order.
  - Cover by rows.
  - The dispatch_unit cost-model op: 1 trace, 0 eager, redispatches == choices first seen, bitwise.
- test_cuda_host_trace_program.py 114 OK, with table rows for every new op and f32div above 2^53. The int64 random
  fuzz keeps its old op set, because a NaN payload is the hardware's.
- Regression OK on this patch: triton 63, lower_tape 6, tape 44, cover 20, lower 18, guards 12. Opaque and replay
  were cut by the reboot and are rerunning (logs/g0f_*, logs/g1s_replay.log).
- GPU sweep against the real launcher (tests/test_trtllm_tile_sweep.py, `T.parity` = kernel, grid, block, smem,
  launch attributes, every KernelParams byte), GB300 with 152 SMs:
  - 40 classes (bf16 h128 p16 at q 3/4/8/16, the same at 6 q heads per KV head at q 2/4, bf16 h128 p64 and fp8 h256
    p32 at q 4/16; each at batch 1/2/4/8) x 3805 max_kv (1..3072 dense, then 256-step boundaries +-1 to 65536).
  - 152,200 points, 0 skipped, 0 mismatches. logs/sweep_state.txt.
  - The cubins came from FlashInfer's artifact repository (480 files, sha-checked) into land/core/symfloat/cubins.
    The box had only the q8 generation kernels.
- T5b, the 2049 point: Q/KV 32/8, max_q 16, batch 1, max_kv 2030..2580.
  - The port equals the launcher at all 551 points; the launcher takes tile 16.
  - With the old unfused order, `parity` reports a kernel mismatch at exactly max_kv 2049..2560 (512 points): port
    Q32 vs launcher Q16, grid (10,8,1) vs (16,8,1).
- Traced decode at q 16 (HostTraceReplay over 12 max_kv values through the tile regions): bitwise, 0 eager steps.
  - Every retrace is the op's own guard (class "dispatch").
  - Traces == distinct (tile, split) pairs (4). A new kernel or split is another launch topology, so the native
    replay needs a variant per kernel, which is the redispatch2 lane's item.
- CPU: the traced port over symbolic (max_kv, batch) is sound at 0 of 16k box points violated. Its guard regions
  are narrower than "same tile", because the port's cta() uses Python min/branches. Those are the split regions the
  launch topology pins anyway.

## Downloads (not approved in advance: recorded here per the coordinator, 2026-10-09)
- 471 trtllm-gen fmha cubins were fetched from FlashInfer's public artifact repository
  (https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/, through fwdproxy) with
  FlashInfer's own `get_artifact` (sha256-checked against flashInferMetaInfo.h).
  - When: 2026-10-08 22:38 (one trial file), then 22:41:04 .. 22:47:06 (470 more).
  - Total: 56440904 bytes (53.8 MiB).
  - Selection (scratch/fetch_cubins.py): the 480 grouped tokens+heads generation kernels for SM100f, bf16/bf16/bf16 and
    e4m3/e4m3/bf16, head dim 128/256, page 16/32/64, causal; 9 of
    the 480 were already present.
  - Where: land/core/symfloat/cubins/2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/fmha/trtllm-gen/. The directory started
    as a copy of land/core/attn/cubins, so the q8 kernels it already held were not re-downloaded.
  - Log: logs/fetch_cubins.log. Only the GPU sweep and tile tests read them (python_sf.sh's FLASHINFER_CUBIN_DIR).
  - Nothing else was downloaded, and nothing more will be without asking.
