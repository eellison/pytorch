# 27B prefill-64 replay mismatch (GPU 2) -- status 2026-10-12 15:10, NOT root-caused

The name is historical: the first bisection pointed at memory="run_buffer"; the later runs rule that out.

## Symptom
Qwen3.8-27B NVFP4, V-stock SOUND (c5 + ARMV_TRTLLM_CPP + armV_r27, ports on or off), drive_fn --check, prefill of a single
64-token request repeated 8 times (one variant: 2396 kernels, 48 eager steps = the GDN CuTe chunked-prefill custom op,
49 segments). The trace call is bitwise; every replay of that variant is not:
- run_buffer: the replay's output is 100% NaN (327680 of 327680), every GDN state tensor differs (96 of 96), layer 0's
  conv state rows already hold 61440 NaN; identical counts in every failing run (v7, v9nopdl, vc2);
- memory=eager (v8e2): finite garbage instead (max_abs 37-66), state wrong from layer 0;
- eager vs eager (ARMV_CHECK_TWICE, v4): bitwise. Decode rows (one segment, no eager steps): bitwise in every run.

## Where it happens (device + CPU set)
| run | GPU | CPUs | memory | other | prefill-64 replays |
|---|---|---|---|---|---|
| v2a, v4, v5, v6m, v7, vc2 | 2 | 108-125 | run_buffer | ports on/off, host args pinned/meta, GDN_WS | 0/7-8 |
| v8e2 | 2 | 108-125 | eager | | 0/8 |
| v9nopdl | 2 | 108-125 | run_buffer | PDL stripped from every captured launch | 0/8 |
| va2 | 2 | 126-143 | run_buffer | | 8/8 |
| va1 | 1 | 108-125 | run_buffer | | 8/8 |
| v3b5, v5b, v6e, v8rb13 | 1 or 3 | 72-107 / 126-143 | run_buffer or eager | KV 8 or 16 | 7/7-8/8 |
| q35rb1 / q35rb2 (Qwen3.5-9B) | 1 / 2 | 72-107 / 108-125 | run_buffer | | 17/17 prefill, all mixed |
GPU 2 failed whether otherwise idle (v5, v7, v9nopdl) or shared with another user's job (vc2). GPU 1/2/3 report the same
SM count (152), clocks, L2; no ECC errors. GPU 2 and 3 are on NUMA node 1 (CPUs 72-143), GPU 0/1 on node 0.

## Ruled out
- the memory plan mode (run_buffer vs eager): fails in both on GPU 2, passes in both on GPU 1/3;
- PDL / programmatic edges (v9nopdl);
- FlashInfer's GDN chunked prefill reading uninitialized memory: standalone gdn_poison.py (NaN-filled free blocks,
  poisoned workspace, T 17..1000) bitwise;
- the GDN CuTe call itself: with it traced (ARMV_GDN_WS=1, v5) the replay still fails (the eager steps are then
  aten.index / index_put_ on the state pool);
- host-argument mode (pinned vs meta);
- tactic variation: FlashInfer's autotuner saves 0 configs in every run (fallback tactic).
- A standalone torch shape (traced runs -> custom-op eager step reading the run's output and writing a state pool ->
  traced runs, T=64; scripts/rb_repro.py) passes in run_buffer / eager / held with allocator perturbation.

## Open hypotheses (next runs, blocked on GPU memory at 15:10: another user holds 255-262 GB on every GPU)
1. A stale device address baked into a harvested closed region (cuBLAS at M=64: 6 learned mm keys), valid or stale
   depending on process layout (the K21 / patch 51 class). Test: vnb2b = ARMV_DIAG_NO_BLAS=1 (nothing harvested) on the
   failing GPU 2 + CPUs 108-125. If it passes: check the learn's captures vary the workspace / op-scoped allocations
   and the binding's slots cover every 8-byte device-address field.
2. va2's pass is luck (1 of 9 GPU-2 runs): repeat GPU 2 + 126-143.
3. v9sync (device sync before each replay GDN eager step) for a remaining GPU-side ordering issue.

## Diagnostics in armV_r27 (all off by default)
ARMV_DIAG_NO_PDL, ARMV_DIAG_SYNC_GDN, ARMV_DIAG_NO_BLAS, ARMV_CHECK_STATE_CPU; check rows carry path, tokens, nan_v,
state_bad (first differing layer / tensor), state_first_bad_detail.

## CORRECTION 16:20: NaN-aware compares hid NaN references
The check's row compare treats NaN == NaN as equal. Recounted with only rows whose eager reference has no NaN:
- processes with NaN references in every row from the first (the warm-up's first, untraced call, before any replay):
  v3b5, v6e, v6r, v8rb13, v9sync, vprobe2 (GPUs 1, 2, 3). Their "passes" (memory=eager, GPU 3, sync, probe) are not
  evidence; vprobe2's probe shows layer 0's GDN output all NaN from finite inputs in vLLM's eager forward itself;
- clean processes: decode bitwise in every clean row (bs 1/4/8); prefill-64 replays pass on GPU 1 (va1, v5b) and fail on
  GPU 2 in 7 of 8 (v5, v6m, v7, v8e2, v9nopdl, va2b, vc2; va2 passed), replay output 100% NaN vs a clean reference;
- bs 64 and long-KV decode have no clean evidence yet.
Two problems now: (A) vLLM's own eager 27B forward goes NaN in about half the processes from start-up (GDN prefill, likely
the stock FlashInfer GDN NaN / state-pool garbage class); (B) GPU-2 prefill replays NaN against a clean reference. The
sync / memory-mode / cache discriminators must be rerun in clean processes.

## 18:50 snapshot evidence (vsnap1, GPU 1, ARMV_DIAG_GDN_SNAP: stream-ordered clones, no sync)
Layer-0 GDN core op calls on 64 tokens (pairs: replay eager step, then the check's eager forward from restored state):
- the target state rows before the op hold uninitialized pool garbage (|x| up to 2e10, and inf / 3.4e38 in later calls);
  with has_initial_state False the op should ignore them;
- pair (2,3): identical inputs (mixed_qkvz, ba, all metadata tensors) and identical pre-state rows; call 2 output clean,
  call 3 output 100% NaN and post ssm rows NaN. Pair (10,11): replay NaN, eager clean. Calls 3-10 NaN, 11+ clean.
=> the output depends on something outside the op's inputs and target state rows, in vLLM's eager path as well as in the
replay's eager step. Candidates: FlashInfer's GDN chunked-prefill workspace (module global, torch.empty, persists across
calls and layers) or other global scratch. Not a replay-only bug; problem B (GPU 2 replay NaN vs clean eager) is likely
the same dependence with a deterministic layout on GPU 2.
Test running (18:44): vt1/vt2/vt3 = gdn_prefill_backend=triton on GPUs 1/2/3 (diagnosis config; stock default is FlashInfer).

## 19:00 Triton GDN prefill removes both problems (vt1/vt2/vt3, GPUs 1/2/3)
gdn_prefill_backend=triton (diagnosis config), ports, eager-twice, KV 40 GiB: VERDICT PASS on all three, identical counts;
every eager reference NaN-free; decode clean replays bs 1/4/8/64 incl. KV 6-8K all bitwise (0 eager / 1 segment); prefill-64
replays 8/8 bitwise on GPU 2 too; mixed 2056 / 6858 replays bitwise. => problems A and B both trace to FlashInfer's CuTe GDN
chunked prefill (vLLM's default on sm_10x), not to the replay. Next: vfill1/vfill2 (state rows filled zero/1e30/inf/nan).

## 19:15 ROOT CAUSE of problem A: FlashInfer GDN prefill, standalone repro (README_repro_fi_gdn_nan.md)
scripts/gdn_min.py, no vLLM / no tracing: in 5 of 9 fresh processes (all GPUs) chunk_gated_delta_rule's LAST 64-token chunk of
every sequence is NaN and the final state is NaN, deterministically for the process; the other processes are clean. Not
workspace- or free-memory-content dependent. vfig1/2 (vLLM processes): vLLM's GDN warm-up call (4x128) already NaN from clean
inputs; vfig3 clean. Problem B (replay NaN vs clean eager in some GPU-2 processes) is in the same FlashInfer kernel; with
Triton GDN prefill it does not occur (vt2). Decision: checks and timing use gdn_prefill_backend=triton (labelled diagnosis
config) until FlashInfer's kernel is fixed.
