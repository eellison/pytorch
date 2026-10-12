# B200 runbook: host-trace replay vs stock vLLM on InferenceX shapes

You are an agent on a 4x B200 box (sm_100) with no access to the machine this branch was made on. Goal: InferenceX-style
serving numbers for Qwen3.8-27B NVFP4 (first) and Qwen3.8-Flash-Next NVFP4 (second), three arms each, using
this branch's PyTorch. Everything you need is in this branch; paths inside the copied scripts are from the original
box (`/data/eellison/...`) and must be rewritten for this one.

## TODAY: fast path (deadline 15:30 PDT)

The numbers are for a talk tonight. Do only this, in order, and report each result as soon as it exists:

1. Build torch (section 1) and vLLM (section 2), install FlashInfer and the fork (section 3). Download `Qwen/Qwen3-8B`
   only. Smoke: `python test/test_cuda_host_trace_replay.py`.
2. Correctness first: one ours-arm run with the check on (the `--check` path of `vllm/bench/drive_ix2.py`, or
   `vllm/ix_run_check.py`): every checked output must be bitwise vs stock. If not, stop and report.
3. SOUND CONFIG ONLY (hard rule): only cuBLAS may be harvested; every open op (vLLM _C ops, Triton, FlashInfer JIT,
   ATen) is traced or runs eagerly. The ours arm uses `ARMV_SOUND=1` with adapter `vllm/armV_stock_fast_m`
   (`INTEG_ARMV`); the run raises if any non-cuBLAS key is learned. Never use the older adapters' default extern lists.
   Correctness check first: `vllm/bench/drive_ix2.py --arm V --check` with the same env as `VS=(...)` in
   `vllm/round_g4s_line.sh`, covering the 8 mixed shapes at KV 1-2K and 6-8K plus decode at long KV. Every check must
   be bitwise; report eager steps and graph segments per variant.
4. The controlled mixed-step benchmark: `vllm/round_g4s_line.sh` (via `vllm/timing_lib.sh` and
   `vllm/bench/drive_ix2.py`), three arms default / fullnc (no compile) / ours (V, sound), arms rotated per round,
   3 rounds, one GPU, nothing else on it. Skip the ARMV_PORTS arm: in vLLM 0.29 the ports never engage
   (the _C ops are called inside vllm_ir library-op bodies), so it equals the sound arm.
5. Report per shape: median ms per arm and ratios ours vs default / vs fullnc, for both ours arms; then eager steps
   and segments per variant, model forwards at init, init seconds, bitwise counts. Expect the sound arm to be much
   closer to fullnc than earlier numbers; report whatever it is.

The InferenceX sweep (sections 5-6) is for later, not today.

## Rules

- At most 2 GPUs at a time unless the user says the box is yours alone. Wrap every GPU job in `flock -x <lockfile>`
  per GPU and check the GPU is idle first (`nvidia-smi --query-compute-apps`). Never kill or signal another user's
  processes; kill only PIDs you started and checked.
- Build PyTorch only with `pip install -e . -v --no-build-isolation`. Use your own venv; no installs into shared envs.
- Long jobs: `setsid nohup ... &` with a log file; background shells may be killed after about an hour.
- Report faithfully, including losses and failures. Report order: counts (bitwise checks, traces, eager steps, learns),
  then speed, then memory.
- Nothing from this box goes to GitHub or any external host. Results go back as a tarball the user copies.

## 1. PyTorch (this branch)

The build is the branch head minus `host_trace_wip/` (not part of the build; ignore it). Commit
`[cuda] Host-trace replay: candidate 4 even top (step294 + build_cpp10)` is the code under test.

```
python -m venv ~/ht-env && source ~/ht-env/bin/activate
pip install -r requirements.txt           # torch's own build deps
export TORCH_CUDA_ARCH_LIST=10.0 CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=<cores/2> USE_CUDA=1
pip install -e . -v --no-build-isolation
```

CUDA 13.0 was used originally. Smoke (expected on the original GB300: all pass; on B200 a few `harvest` tests that
assume GB300 cuBLAS kernel picks may fail, which is fine; anything in replay/pinned/float failing is not):

```
python test/test_cuda_host_trace_replay.py     # 279 pass on GB300
python test/test_cuda_host_trace_pinned.py     # 38
python test/test_cuda_host_trace_float.py      # 12
```

## 2. vLLM v0.29.0 from source against this torch

Unmodified upstream vLLM, commit `98dff2a81` (`git clone https://github.com/vllm-project/vllm && git checkout 98dff2a81`).

```
cd vllm && python use_existing_torch.py
TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=16 pip install --no-deps --no-build-isolation -v .
pip install -r <branch>/host_trace_wip/b200/requirements_pinned.txt --no-deps
```

`requirements_pinned.txt` is the exact package set of the original runs (torch and vllm removed). It was taken from an
aarch64 box: if a pinned version has no x86 wheel, take the nearest and record it. Do not let pip replace torch
(`pip check` warnings about torch's version string are expected). If vLLM's `_C` fails to load with missing cuBLAS
symbols, `LD_PRELOAD=$CUDA_HOME/lib64/libcublas.so.13` (the original launcher does this).

## 3. FlashInfer trtllm-gen fork (ours arm only)

The "ours" arm traces FlashInfer's trtllm-gen attention launcher with symbolic sizes; that needs a small fork of
flashinfer-python 0.6.18 (`flashinfer-0.6.18-trtllm-trace.patch`: decode.py, prefill.py, utils.py, new
trtllm_trace.py / trtllm_witness.py). Keep stock FlashInfer installed for the default arms; make a fork overlay:

```
mkdir -p ~/fi_fork && cp -r $(python -c 'import flashinfer,os;print(os.path.dirname(flashinfer.__file__))') ~/fi_fork/
cd ~/fi_fork && patch -p1 < <branch>/host_trace_wip/b200/flashinfer-0.6.18-trtllm-trace.patch
```

and put `~/fi_fork` first on PYTHONPATH only for the ours arm (that is what `CAND_FORK=1` did). FlashInfer downloads
its cubins on first use; that is fine. Give the fork arm its own `FLASHINFER_WORKSPACE_BASE` and `TVM_FFI_CACHE_DIR`.

## 4. Harness

`host_trace_wip/b200/integration/` keeps the original layout:

- `python_vllm_cand2.sh`: the original launcher. Replace it with a short launcher for this box that: activates your
  venv; sets PYTHONPATH to `[~/fi_fork if CAND_FORK=1]:$VLLM_EXTRA_PYTHONPATH`; sets `INTEG_ARMV` / caches
  (`VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`), `CUDA_HOME`,
  `TORCHINDUCTOR_FORCE_DISABLE_CACHES=${HOSTTRACE_TORCH_DISABLE_CACHES:-1}`, `VLLM_NO_USAGE_STATS=1`; then
  `exec python "$@"`. Drop the `CAND_LINE` cases and `site_land`: there is one torch here.
- `serve/ix/ix_run.py`: runs one point (fresh server, InferenceX client, profiler burst, counters). Usage:
  `bash <launcher> serve/ix/ix_run.py --sku <sku> --scenario 1k1k --conc 4 --gpu 0 --tag <tag>`.
- `serve/ix/ix_nv_report.py --master vllm/ix_master_attnlc.yaml <tag_r1> <tag_r2> <tag_r3>`: the NVFP4 table, repeats
  merged as median [min-max]. `serve/ix/nv_sweep.sh` is the original sweep driver (order, repeats, idle gating).
- `serve/plugin/armv_serve.py`: the worker extension (`--worker-extension-cls armv_serve.ArmVWorkerExt`) and the
  `/armv/install` endpoint. Default arms load it in observe mode (counters only).
- `serve/armV_stock/`, `vllm/armV_stock_fast/`: the "ours" adapters (`ARMV_SERVE_ADAPTER`).
- `serve/ix/recipes/*.yaml`: per-model `vllm serve` args (InferenceX srt-slurm recipe shape).
- InferenceX client: clone `https://github.com/SemiAnalysisAI/InferenceX` at `e0315d2a`; ix_run.py expects it
  at `serve/inferencex` (it imports `infx` from `inferencex-e2e`).

## 5. SKUs and arms

Copy the GB300 SKUs in `vllm/ix_master_attnlc.yaml` to `-b200-` names, fix `model:` paths and `gpu_type`, and size
`x-local.kv-cache-memory-bytes` for 180 GB (the GB300 values assumed 288 GB; leave about 25 GB headroom after
weights). Three arms per model, all with the same kernels; they differ only in who graphs what:

| arm | Qwen3.8-27B SKU to copy | what it is |
|---|---|---|
| default | `qwen3.827bp4-fp4-gb300-vllm` | the InferenceX recipe: torch.compile + FULL_AND_PIECEWISE graphs |
| default-nc | `qwen3.827bnc4-fp4-gb300-vllm` | `-cc.mode none -cc.cudagraph_mode full_decode_only`: no compile, full graphs for decode only |
| ours | `qwen3.827bvs4-fp4-gb300-vllm` | `--enforce-eager` + host-trace replay of the whole step (adapter `serve/armV_stock`, `ARMV_*` env as in the SKU) |

Flash-Next: `qwen3.8flashnextp4-fp4-gb300-vllm` (the recipe's single-GPU override, already no-compile, so default =
default-nc and there are two arms) and `qwen3.8flashnextvs4-fp4-gb300-vllm`. Replace every `CAND_LINE` value with
nothing (one torch) and every absolute `ARMV_SERVE_ADAPTER` / `FLASHINFER_WORKSPACE_BASE` path with this box's.

Important: the ours SKUs must keep `ARMV_HOST_ARGS: pinned`. Without it every mixed/prefill trace declines on a CPU
argument and runs eagerly after paying a full trace (an earlier run lost 10-26% that way). Check the counts: mixed
steps should replay, not show up as eager. `ARMV_STEPLOG=1` (on in these SKUs) saves a per-step timeline
(`*_steps.json`); `serve/ix/ix_tails.py <tag>` breaks the tail requests down into trace / learn / redispatch /
eager / replay time.

The 27B ours arm sends attention through the adapter's trtllm shim (the fork refuses head dim 256); that is expected.

Models (Hugging Face): `Inferact/Qwen3.8-27B-NVFP4`, `Inferact/Qwen3.8-Flash-Next-NVFP4`; `Qwen/Qwen3-8B` for a smoke.
Check `df` before downloading.

## 6. Order of work

1. Smoke: Qwen3-8B, SKUs `qwen38bp4-bf16-gb300-vllm` (default) and `qwen38bvs4-bf16-gb300-vllm` (ours) copied to b200, 1k1k conc 4. Expected
   from the original box: ours bitwise on every checked request (the point JSON's `check.summary`: `bitwise == n`),
   2-10 traces, 2-3 eager steps, `replays` about equal to `steps`.
2. Qwen3.8-27B NVFP4: 1k1k and 8k1k at conc 1/4/16/64, default and default-nc only, 3 repeats at conc 4 and 64.
   Skip the ours arm on this branch: every decode trace declines with "sN is not read from the call's inputs"
   (the CuTe DSL `mm_fp4` address scalar, a known core gap being fixed), and prefill declines with "an allocation on
   meta in a trace" (adapter bug, fixed in a later adapter). It gets 0 replays, so it measures nothing. Compare the
   default arm against InferenceX's published B200 numbers for this model if they exist.
3. Flash-Next NVFP4: same matrix (conc up to 32 per the SKU), both arms. Before timing the ours arm, run one point and
   read `/armv/summary`: if traces are not far below steps, or the point JSON says MISCONFIGURED / RETRACE-BOUND, stop
   and report the decline reasons instead of sweeping.

Per point, report: eager steps (split mm_fp4 vs other: `breakdown_after.eager_path_calls`, `entry_eager`), learns
and learn share of the run, traces, bitwise checks; then total and output tok/s/GPU, TPOT/ITL p50/p90, TTFT p50/p90;
then server ready time and reserved/peak memory (`counters`).

## 7. Returning results

```
tar czf b200_results_<date>.tgz serve/out/ix/<tag>/*_point.json serve/out/ix/<tag>/*.json REPORT_<tag>.md
```

plus a short NOTES.md: build deviations (versions changed, patches that did not apply), anything skipped, and every
failure with its log excerpt. The user carries the tarball back.
