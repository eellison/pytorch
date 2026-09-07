# Exact-current F1 baseline results

## Source and method

- Worktree: `/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt`
- HEAD: `6b8ef64bd373b221c6ed1ba4f4b2f1edef2643c3`
- Unstaged diff SHA-256:
  `ce766480d4bed7972584fbbc146608dc4a3d6c50c3e5a075f6b98c8e65f1c16e`
- Device: idle NVIDIA B200 on physical GPU 1, CUDA 13.0
- Timing: generated wrappers loaded in a fresh process, common inputs within
  each format/shape, one CUDA graph per mode, rotating mode order, 12 rounds of
  100 replays timed with CUDA events.
- Raw results: `agent_space/f1_perf_baseline_20260826/replay_results.json`
- Generated wrappers and compile metadata:
  `agent_space/f1_perf_baseline_20260826/generated/` and
  `agent_space/f1_perf_baseline_20260826/results.json`

The timing fields in `results.json` are invalid because the three live
`torch.compile` callables did not retain distinct forced-mode dispatch. Only its
compile, source, kernel-count, and launcher-resource fields are used. Direct
replay of the archived wrappers reproduces the expected persistent spill cliff.

## NVFP4 and MXFP4

All rows generated one staged kernel. Times are median microseconds.

| Format | Shape | Default | Persistent | Looped |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 | 128x4096 | 4.13 | 14.40 | 10.28 |
| NVFP4 | 4096x4096 | 20.50 | 43.24 | 20.56 |
| NVFP4 | 4096x4608 | 28.70 | 514.28 | 28.68 |
| NVFP4 | 4096x8192 | 40.98 | 611.79 | 39.24 |
| MXFP4 | 128x4096 | 4.13 | 10.36 | 10.25 |
| MXFP4 | 4096x4096 | 18.17 | 26.63 | 18.46 |
| MXFP4 | 4096x4608 | 26.64 | 53.64 | 25.21 |
| MXFP4 | 4096x8192 | 30.19 | 453.82 | 32.98 |

Default used `XBLOCK=1`, `R0_BLOCK=1024`, eight warps, and no spills. Looped
used `XBLOCK=8`, `R0_BLOCK=1024`, four warps, and no spills. Forced persistent
used `XBLOCK=8` and four warps. Its tail rows had the expected spill cliffs:

| Format | Shape | Registers | Spills |
| --- | ---: | ---: | ---: |
| NVFP4 | 4096x4608 | 32 | 1404 |
| NVFP4 | 4096x8192 | 32 | 1920 |
| MXFP4 | 4096x4608 | 255 | 354 |
| MXFP4 | 4096x8192 | 32 | 1360 |

With shared inputs, modes differing only in reduction schedule had at most
three packed-byte mismatches and maximum scale difference 0.0625. This is not
the F1/F2 correctness criterion: paired F1/F2 wrappers will use the same config
and must match exactly.

## MXFP6/DCN preshuffle

Shape `(2048, 3072)`, 15 rotated rounds of 100 replays. Both variants generated
one staged kernel and matched exactly for all 196,608 scale bytes and 4,718,592
packed bytes.

| Conversion | Median us | Registers | Spills | Inline asm |
| --- | ---: | ---: | ---: | ---: |
| Software E2M3 | 16.43 | 47 | 0 | 0 |
| Native pack-2 E2M3 | 9.75 | 32 | 0 | 1 |

Raw replay: `agent_space/f1_perf_baseline_20260826/mxfp6_dcn_replay.json`.
Compile/resource metadata:
`agent_space/f1_perf_baseline_20260826/mxfp6_dcn.json`.

## Reusable F1/F2 replay

`agent_space/archived_wrapper_replay_20260826.py` loads arbitrary generated
wrappers with shared inputs, requires exact outputs by default, captures each
wrapper independently, and rotates timing order. Once F2 is stable, compile its
matching wrappers with `f1_perf_baseline_20260826.py --label f2
--skip-live-timing`, then run, for example:

```bash
CUDA_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
PYTHONPATH=/data/users/eellison/pytorch \
conda run --no-capture-output -n pytorch-3.12 python \
  agent_space/archived_wrapper_replay_20260826.py \
  path/to/f1_wrapper.py path/to/f2_wrapper.py \
  --labels f1 f2 --rounds 15 --inner-reps 100 \
  --output agent_space/f1_f2_paired_result.json
```
