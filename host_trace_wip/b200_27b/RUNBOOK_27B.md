# Qwen3.8-27B NVFP4 on B200: host-trace replay ("V-stock") correctness check and decode timing

You are running a self-contained experiment on a machine with NVIDIA B200 GPUs (sm_100). Nothing here needs network access
except installing packages, downloading the model, and FlashInfer's one-time download of its trtllm-gen cubins (step 5).
Return a results tarball and NOTES.md (step 9).

## What this measures

The bundle runs stock vLLM 0.29.0 with `enforce_eager=True` and wraps the model forward in a CUDA-graph host-trace replay. The
replay is built on a PyTorch fork, and the wrapper is the adapter in `armV_r27/`. We call this arm "V-stock". Its rules:
- Every GPU launch in a model step is traced with symbolic sizes and replayed as a CUDA graph.
- Only closed-source cuBLAS kernels may be *harvested* (learned from a capture). This is "SOUND" mode, `ARMV_SOUND=1`. The run
  fails if any other kernel is learned.
- Everything open-source is traced from its real host code or runs eagerly. That covers vLLM's _C ops, Triton, FlashInfer JIT
  and CuTe DSL kernels, and ATen.
- vLLM's _C ops are traced through `vllm_ports/`: Python ports of their host launchers, each byte-checked against a capture of
  the real op.
- FlashInfer's trtllm-gen attention is traced through FlashInfer's own C++ launcher. That launcher is compiled with symbolic
  sizes into a small extension (`trtllm_cpp/`), and a 3-file FlashInfer overlay calls into it.
- The kernels are the same as stock vLLM's. There are no kernel swaps.

Arms for timing:
- **default**: vLLM's own recipe configuration (torch.compile + FULL_AND_PIECEWISE CUDA graphs).
- **default-nc**: compile off, vLLM FULL_DECODE_ONLY graphs.
- **V-stock**: as above. Prefill and mixed steps run vLLM's eager forward (`ARMV_EAGER_KINDS=prefill,mixed`; see "Known open
  issue").

## 0. Requirements
- B200 GPUs, Linux x86_64 or aarch64, CUDA toolkit 13.0 (the reference runs used nvcc 13.0.88), Python 3.12, gcc >= 11.
- About 200 GB disk (two source builds, model 25 GB, caches).
- Reference package versions (from the GB300 runs): vllm 0.29.0, flashinfer-python 0.6.18, nvidia-cutlass-dsl 4.6.2,
  cuda-python / cuda-bindings 13.3.1, apache-tvm-ffi 0.1.11, transformers 5.12.1, numpy 2.5.2.
- One idle GPU per job: the check and the timing each use one GPU (`CUDA_VISIBLE_DEVICES=N`). Several GPUs let you run
  checks in parallel, but timing should run alone on the machine.

## 1. PyTorch (the host-trace fork)
```
git clone https://github.com/eellison/pytorch.git torch-ht && cd torch-ht
git checkout host-trace-wip-20261012-c5
git submodule update --init --recursive
python -m venv ~/venv-ht && source ~/venv-ht/bin/activate     # or conda; use this env for every step below
pip install -r requirements.txt
export TORCH_CUDA_ARCH_LIST=10.0 USE_CUDA=1 CUDA_HOME=/usr/local/cuda-13.0
pip install -e . -v --no-build-isolation
python -c "import torch, torch.cuda._host_trace as ht; print(torch.__version__, torch.cuda.get_device_capability())"
```
`export TORCH_SRC=$PWD` (recorded in the results).

## 2. vLLM v0.29.0 from source, on the existing torch
```
git clone https://github.com/vllm-project/vllm.git && cd vllm && git checkout 98dff2a81   # = v0.29.0
python use_existing_torch.py
pip install -r requirements/build.txt
export TORCH_CUDA_ARCH_LIST=10.0 VLLM_TARGET_DEVICE=cuda MAX_JOBS=32
pip install -e . --no-build-isolation -v
python -c "import vllm, vllm._custom_ops; print(vllm.__version__)"
```
`export VLLM_SRC=$PWD`. If pip tries to replace torch, stop. The fork's torch must stay installed (`pip list | grep torch`).

## 3. FlashInfer 0.6.18 and friends
```
pip install flashinfer-python==0.6.18 nvidia-cutlass-dsl==4.6.2 apache-tvm-ffi==0.1.11 cuda-python==13.3.1
```
Use exactly 0.6.18. The traced launcher is a patch of that version's files, checked by hash in steps 4 and 6.

## 4. Bundle setup
```
cd <this bundle>; export BUNDLE=$PWD
# optional overrides (defaults in env.sh): WORK (caches/outputs, default $BUNDLE/work), MODEL_27B, CUDA_HOME,
# FLASHINFER_WORKSPACE_BASE, FI_SITE_PKG (installed flashinfer dir), FI_ARCH (100a on B200; auto-detected from the GPU)
export CUDA_HOME=/usr/local/cuda-13.0
bash bin/make_overlay.sh      # overlay of the installed flashinfer + the 3 hooked/glue files; checks the stock file hashes
```

## 5. Model and one-time warm-up (downloads)
```
huggingface-cli download Inferact/Qwen3.8-27B-NVFP4 --local-dir $WORK/models/Qwen3.8-27B-NVFP4
export MODEL_27B=$WORK/models/Qwen3.8-27B-NVFP4
CUDA_VISIBLE_DEVICES=0 bash bin/warm.sh
```
The warm-up runs vLLM eagerly over the check's shapes. That has three effects:
- FlashInfer JIT-builds its modules for sm_100 under `$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/0.6.18/100a/` (the
  directory name follows the arch; `fmha_gen` is the one we need).
- FlashInfer downloads every trtllm-gen FMHA cubin those shapes select into `$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/cubins/`.
  The 27B attention class is fp8 e4m3 Q/KV, bf16 out, head 256, page 32, 24 q heads / 4 kv heads.
- The CuTe DSL mm_fp4 kernels compile under the host trace's observation (`ARMV_CUTE_OBSERVE=1`). Never compile them into
  this workspace with observation off: a persisted kernel compiled unobserved becomes an opaque foreign call that the trace
  cannot follow. If that happens, delete `.../cached_ops/mm_fp4_*_cute_dsl` and rerun `bin/warm.sh`.

After this, every run sets `FLASHINFER_NO_DOWNLOAD=1`. A replay that needs a cubin missing on disk declines with "kernel X was
not loaded at the warm-up". If you see that, rerun `bin/warm.sh` with the shape that needs it, then repeat the check.

## 6. The traced trtllm-gen extension for sm_100
```
bash bin/build_ext.sh
```
It runs four steps:
1. Checks `trtllm_cpp/upstream/src` and the installed FlashInfer's `data/` files against `trtllm_cpp/src/UPSTREAM.sha256`.
2. Writes `trtllm_cpp/src` from upstream with `ht/make_patch.py` (each change is an asserted substitution).
3. Builds `$FI_HT_BUILD/fi_ht_trtllm.so` against the installed torch with `ht/build.py`. That is about 2 minutes, CPU only.
4. Prints our nvcc flags against FlashInfer's own `fmha_gen/build.ninja` with `bin/check_flags.py`.

What changes between GB300 (sm_103a) and B200 (sm_100) is all in `trtllm_cpp/ht/build_flags.py`, and it is now env-driven:
- The arch: `ARCH` is `100a` on B200, from `torch.cuda.get_device_capability()` or `FI_ARCH`. It gives
  `-gencode=arch=compute_100a,code=sm_100a`. Before this change the value was hard-coded as `103a`.
- The stock build path: `STOCK_BUILD = $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/0.6.18/<ARCH>/cached_ops/fmha_gen`. It was
  hard-coded to the GB300 `103a` directory. Override it with `FI_STOCK_FMHA_BUILD`.
- The cubin dir: `CUBIN_DIR = $FLASHINFER_CUBIN_DIR`, default `$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/cubins`.
- Values that are arch-independent and unchanged:
  - `CUBIN_SUB = 2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/fmha/trtllm-gen/` (flashinfer.artifacts.ArtifactPath.TRTLLM_GEN_FMHA);
  - `METAINFO_HASH = 892b5a75...` (FlashInfer derives it from that directory's checksums.txt).
  - Check both against `fmha_gen/build.ninja` (`-DTLLM_GEN_FMHA_CUBIN_PATH`, `-DTLLM_GEN_FMHA_METAINFO_HASH`). `check_flags.py`
    prints any difference.
- Machine paths: the Python include dir (sysconfig), tvm_ffi include (import), FlashInfer `data/` dir (`FI_SITE_PKG`), and
  `CUDA_HOME`.
- If `check_flags.py` reports a flag only in stock other than include paths, add it to `CUDA_ONLY` / `DEFINES` in
  build_flags.py and rebuild. Report it in NOTES.md.

`trtllm_cpp/ht/kernel_params_fields.h` holds the KernelParams field offsets, generated from DWARF on the GB300 (aarch64) build.
The struct layout is the same on x86_64 (LP64, same compiler ABI). The runtime check below verifies it anyway: every recorded
trtllm-gen launch is compared byte for byte with a capture of the stock launcher's launch, and any difference declines the
call to an eager step, never a wrong replay. If you see "the traced build's launches are not the binding's", regenerate
the offsets with `python trtllm_cpp/ht/gen_fields.py` (needs the stock fmha_gen object with debug info; see its docstring),
then rebuild.

Optional, CPU only: `bash bin/python_27b.sh trtllm_cpp/tests/test_drift.py`. It checks the patched launcher compiles to the
same host code as the stock one. Its instruction-masking regexes were written for aarch64 objdump output, so on x86_64 a
failure of `test_same_code` alone is not a blocker; note it.

## 7. Smoke tests and the correctness check
```
CUDA_VISIBLE_DEVICES=0 bash bin/smoke.sh
CUDA_VISIBLE_DEVICES=0 bash bin/run_check.sh c1
```
The smoke tests should show the following:
- Every `PORT ...` line reads `bitwise 8/8 ... eager [] ... declines {}`.
- The trtllm trace tests print OK.

The check (`bin/run_check.sh`, about 15-30 minutes) runs `bench/drive_fn.py --arm V --check` over these shapes:
- decode bs 1/8/64 at short KV, bs 8 at KV 6000-8000, bs 64 at KV 1000-3000;
- prefill 64 and 512;
- mixed steps, 1024-token chunk + 8 decoders at KV 1-2K and 2048 + 8 at KV 6-8K.

It uses fp8 KV, max_model_len 9216, KV pool `KV_GIB` (default 24 GiB). Every traced or replayed step is compared with vLLM's
eager forward on the same metadata, from the same GDN recurrent state, including the state rows after the step. With
`ARMV_CHECK_TWICE=1`, eager also runs twice as a control. Outputs go to `$WORK/out/c1.{json,summary.txt,verdict.txt}` and
`$WORK/logs/c1.log`.

**PASS** (`tools/verdict.py`) requires all of the following:
- every decode check row is bitwise, output and state;
- every decode variant has **0 eager steps and 1 segment** (one CUDA graph per step);
- harvested keys are cuBLAS only (`unsound_learned` is empty; `learned_by_op` shows only aten::mm / addmm / bmm);
- at least one decode replay.

Reference on GB300: decode variants are 2540-2684 kernels, 0 eager, 1 segment. Learned keys were aten::mm only. Every decode
row was bitwise, including bs 64 and KV up to 8K.

If the check FAILS, do not time. Report the verdict and summary, and the first 200 lines of the log after the first traceback
or `VLLM_PORT_DECLINE` line.

## 8. Decode-only timing (only after a PASS)
```
CUDA_VISIBLE_DEVICES=0 bash bin/timing_all.sh
```
It runs one untimed stock warm-up run, so the stock arms' JIT builds are not timed. Then come 3 rounds of `bin/timing_round.sh`:
- arms default / default-nc / V-stock, rotated per round;
- one GPU, each run waiting for the GPU to have no other process;
- shapes: decode bs 1/8/64 at short KV, bs 8 and 64 at KV 6000-8000;
- KV pool `KV_GIB_TIMING` (default 60 GiB), max_model_len 16384, fp8 KV;
- V-stock with `ARMV_EAGER_KINDS=prefill,mixed`.

The default arms run stock FlashInfer (no overlay) in their own JIT workspace and share the downloaded cubins.
`tools/timing_table.py` writes `$WORK/timing/TABLE.txt`: median [min-max] step ms per arm, plus V-stock vs default and vs
default-nc.
Do not run anything else on the machine during timing, and note the load average in NOTES.md (it is in `timing.log`).

## Known open issue (do not time prefill / mixed)
On the GB300 reference box, prefill-64 replays were not bitwise on one specific GPU. The replay output was all NaN or finite
garbage, and the GDN state was wrong from layer 0, while eager vs eager was bitwise. The same configuration was bitwise on the
other GPUs. It is not root-caused. We have ruled out the memory plan mode, PDL, host-argument mode, and the GDN CuTe call
itself.

So, on the B200:
- report the prefill and mixed rows of the check (`verdict.txt` lists them by path / tokens / bitwise, with the eager-twice
  control). If any prefill row is not bitwise, also run `bin/run_check.sh c1_gpuK` on two other GPUs and report which
  GPUs show it;
- do not time prefill or mixed, and do not report them as replay results.

A second known item, not a correctness problem: mixed steps of 1032 tokens may decline to trace ("sN is not read from the
call's inputs") and run eagerly. Report the count.

## 9. Results
Write `$WORK/NOTES.md` with:
- the machine (GPU count, CPU arch);
- deviations from this runbook;
- the check verdict and summary;
- the prefill/mixed rows;
- the timing table;
- anything that declined or failed.

Then:
```
bash bin/collect_results.sh        # -> $WORK/r27_b200_results.tgz (outputs, verdicts, timing, log tails, ENV.txt)
```
Return the tarball and NOTES.md.

## Files
- `env.sh`: every path and cache location, all overridable.
- `bin/`:
  - `python_27b.sh`: the launcher (overlay first on PYTHONPATH unless NO_OVERLAY=1);
  - `make_overlay.sh`, `build_ext.sh`, `check_flags.py`, `warm.sh`, `smoke.sh`;
  - `run_check.sh` + `check_shapes.txt`, `vstock_env.sh` (the V-stock flags);
  - `timing_round.sh`, `timing_all.sh`, `collect_results.sh`.
- `overlay_src/flashinfer/`:
  - `decode.py`, `prefill.py`: FlashInfer 0.6.18's files plus a 2-line hook at each trtllm-gen binding call;
  - `trtllm_cpp.py`: the glue that records the traced launches and byte-checks them against the stock binding.
- `trtllm_cpp/`:
  - `upstream/src`: FlashInfer 0.6.18's launcher files;
  - `src/`: the patched copies, regenerated by `ht/make_patch.py`;
  - `ht/`: the extension sources, build_flags, build, make_patch, gen_fields;
  - `tests/`: test_drift, test_trtllm_cpp_trace, traced_shapes.cu.
- `armV_r27/`: the vLLM adapter (adapter.py, adapter_cpp.py, hostcuts.py, trtllm_shim.py).
- `vllm_ports/`: `vllm_ops_trace.py` (the vLLM _C launchers) and `sgl_ops_trace.py` (helpers it imports).
- `bench/drive_fn.py`: the step driver and checker.
- `tools/`: `summ.py`, `verdict.py`, `timing_table.py`, `ports_test.py`.
