# Padded scale-layout benchmark

This suite isolates the choice of padding implementation from quantization and
upstream fusion. Each benchmark cell runs in a fresh process and is timed by
CUDA-event measurement of a CUDA graph containing 100 calls. Inductor's own
CUDA-graph wrapper is disabled. FlashInfer PDL is disabled.

The matrix covers:

- standalone 128x4 blocked-scale swizzle and padding;
- standalone 96-logical-row to 128-physical-row XDL swizzle and padding;
- MXFP4 and MXFP8 quantization with padded 128x4 scales;
- RMSNorm to MXFP4, NVFP4, and MXFP8 with padded 128x4 scales;
- MXFP6 quantization, RMSNorm to MXFP6, and DCN `addcmul` to MXFP6 with
  padded XDL scales.

For each full Inductor path, default and coordinate-descent tuning are separate
cells. The ordinary `F.pad` formulation is compared with prefill plus scatter
and, for the 128x4 layout, explicit padding-region scatter. Optional worktrees
can add the predicated masked-fill and auxiliary-write implementations.

The core shapes include a no-padding 128x4 control, heavy row padding,
simultaneous row/column padding, and a larger production-like case. XDL cases
cover the 95/96/97 row boundary and the 2048x3072 DCN shape.

Run the current-main core matrix:

```bash
python agent_space/padding_matrix_20260903/run_matrix.py \
  --worktree agent_space/padding_matrix_main_20260903 \
  --output agent_space/padding_matrix_20260903/results_main_f854_core_v2.jsonl \
  --preset core --gpus 2,3,4,5,6,7 \
  --warmup 20 --samples 50 --calls-per-graph 100
```

Generate a report:

```bash
python agent_space/padding_matrix_20260903/analyze.py \
  agent_space/padding_matrix_20260903/results_main_f854_core_v2.jsonl \
  --output agent_space/padding_matrix_20260903/REPORT.md
```

The runner is resumable. Workload/shape groups remain on one physical GPU so
ratios between padding implementations and tuning modes do not include
cross-device variation.
