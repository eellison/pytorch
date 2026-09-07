# Exact-current F1 performance baseline plan

## Pinned source

- Worktree: `/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt`
- HEAD: `6b8ef64bd373b221c6ed1ba4f4b2f1edef2643c3`
- Expected unstaged diff SHA-256 before the sweep:
  `ce766480d4bed7972584fbbc146608dc4a3d6c50c3e5a075f6b98c8e65f1c16e`

The editable install currently points at
`agent_space/pr191775_layered_split_wt`, not the F1 worktree. Every command must
therefore use `domain_projection_work/run_worktree_script.py` and set
`PYTORCH_WORKTREE`; running the benchmark directly would silently measure the
wrong source tree.

## Environment

- Conda environment: `pytorch-3.12`
- GPU: NVIDIA B200, SM100, CUDA 13.0, driver 580.82.07
- Selected device: physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`
- Coordinate descent: disabled
- Triton multi-kernel: disabled
- Compilation and resource collection: one fresh Inductor cache per case
- Authoritative timing: emitted wrappers reloaded in a fresh process, shared
  inputs within each format/shape, CUDA graphs, rotating mode order, and 12
  CUDA-event rounds of 100 graph replays
- Fresh Inductor cache for every compile

## Matrix

For NVFP4 and MXFP4, run shapes `128x4096`, `4096x4096`, `4096x4608`, and
`4096x8192` in three modes:

- `default`: ordinary Inductor reduction heuristic;
- `persistent`: persistent forced with `XBLOCK=8`, four warps, one stage;
- `looped`: looped forced with `XBLOCK=8`, `R0_BLOCK=1024`, four warps, one
  stage.

Record median/min/p10/p90, every timing sample, compile time, kernel and staged
kernel counts, selected launcher config, registers, spills, shared memory,
source hash, loads/stores, conversions, reciprocals, splits, broadcasts,
reshapes, and loop count. Packed bytes and scales must be exact across the three
modes only when their reduction orders match; otherwise record byte mismatch
counts and maximum scale error. The later F1/F2 paired run uses identical
configs and requires exact output.

Add one MXFP6/DCN `(2048, 3072)` preshuffle row comparing the software
conversion with native `cvt.rn.satfinite.e2m3x2.f32` pack=2 in the same process.
The outputs must be exact, and both forms must remain one nested kernel. Record
the same launcher registers/spills/config already exposed by the existing DCN
harness.

## Commands

Preflight, which must print F1 paths for both `scheduler.py` and `simd.py`:

```bash
cd /data/users/eellison/pytorch
CUDA_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
PYTHONPATH=/data/users/eellison/pytorch \
PYTORCH_WORKTREE=/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt \
conda run --no-capture-output -n pytorch-3.12 python \
  agent_space/domain_projection_work/run_worktree_script.py \
  agent_space/f1_perf_baseline_20260826.py \
  --preflight \
  --output agent_space/f1_perf_baseline_20260826/preflight.json \
  --generated-dir agent_space/f1_perf_baseline_20260826/generated
```

Main NVFP4/MXFP4 matrix:

```bash
CUDA_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
PYTHONPATH=/data/users/eellison/pytorch \
PYTORCH_WORKTREE=/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt \
conda run --no-capture-output -n pytorch-3.12 python \
  agent_space/domain_projection_work/run_worktree_script.py \
  agent_space/f1_perf_baseline_20260826.py \
  --formats nvfp4 mxfp4 \
  --shapes 128x4096 4096x4096 4096x4608 4096x8192 \
  --modes default persistent looped \
  --label f1 --skip-live-timing \
  --output agent_space/f1_perf_baseline_20260826/results.json \
  --generated-dir agent_space/f1_perf_baseline_20260826/generated
```

MXFP6/DCN non-regression:

```bash
CUDA_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
PYTHONPATH=/data/users/eellison/pytorch \
PYTORCH_WORKTREE=/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt \
MXFP6_PROTOTYPE=/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt \
conda run --no-capture-output -n pytorch-3.12 python \
  agent_space/domain_projection_work/run_worktree_script.py \
  agent_space/f1_mxfp6_dcn_baseline_20260826.py \
  --rounds 15 --inner-reps 100 \
  --output agent_space/f1_perf_baseline_20260826/mxfp6_dcn.json \
  --generated-dir agent_space/f1_perf_baseline_20260826/generated
```

The timing fields written by the compile command are not authoritative: keeping
multiple `torch.compile` wrappers for the same Python function alive did not
preserve their distinct forced-mode dispatch. The emitted wrappers and all
compile/resource/source metadata are valid. Replay the wrappers directly:

```bash
CUDA_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
PYTHONPATH=/data/users/eellison/pytorch \
conda run --no-capture-output -n pytorch-3.12 python \
  agent_space/f1_perf_replay_20260826.py \
  agent_space/f1_perf_baseline_20260826/results.json \
  --rounds 12 --inner-reps 100 \
  --output agent_space/f1_perf_baseline_20260826/replay_results.json
```

These generated wrappers are also retained so eventual F2a wrappers can be
loaded beside them and timed in one process with alternating order.

For an exact pair of F1/F2 wrappers, use the generic comparator
`agent_space/archived_wrapper_replay_20260826.py`. It shares inputs, requires
exact outputs by default, captures the wrappers independently, and rotates
timing order. The validated baseline table is in
`agent_space/f1_perf_baseline_results_20260826.md`.
