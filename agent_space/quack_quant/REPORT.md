# QuACK quantizers under nested reduction (2026-09-04)

QuACK 0.6.4 `quack.blockscaled.quantize`: pure-PyTorch quantizers ported from
torchao, shipped with `torch.compile` handles. Nine formats plus `to_mx_dim0`
and the 128x4 `to_blocked` scale swizzle. B200, bf16 inputs, CUDA graph of 100
calls, one cell per process (`cell.py`/`run.py`), production Inductor config
except `triton.cudagraphs=False`.

## Correctness

`probe.py` at 2048x3072, 8192x4096, 65536x2048: 30 cases each (every format
standalone, after `F.rms_norm`, and with `to_blocked`). No compile failures.
Nested-on is byte-identical to nested-off in all 90 cases. RMSNorm rows differ
from eager by 65/6.3M bf16 elements at 1 ulp (5 MXFP8 codes), identically with
nested off.

## Perf: RMSNorm -> quantizer (median us)

off = nested disabled, on = default, persistent = forced persistent nested
form (identical to what `triton.multi_kernel=1` picks at runtime).

| case | shape | eager | off | on | persistent | vs off |
|---|---|---|---|---|---|---|
| mxfp8_e4m3 | 2048x3072 | 111 | 7.0 | 7.5 | 6.0 | -14% |
| mxfp8_e4m3 | 8192x4096 | 504 | 30.2 | 32.8 | 24.1 | -20% |
| mxfp8_e4m3 | 65536x2048 | 1947 | 138 | 160 | 127 | -8% |
| mxfp4 | 2048x3072 | 241 | 14.2 | 10.8 | 10.8 | -24% |
| mxfp4 | 8192x4096 | 1322 | 60.7 | 47.6 | 39.5 | -35% |
| mxfp4 | 65536x2048 | 5267 | 256 | 206 | 173 | -33% |
| nvfp4 | 2048x3072 | 479 | 16.0 | 11.6 | 11.6 | -28% |
| nvfp4 | 8192x4096 | 1447 | 63.7 | 49.1 | 41.6 | -35% |
| nvfp4 | 65536x2048 | 5610 | 271 | 213 | 180 | -34% |
| mxfp6_e2m3 | 8192x4096 | 1267 | 46.8 | 49.3 | 37.8 | -19% |
| mxfp6_e2m3_packed | 8192x4096 | 1398 | 100 | 105 | 94 | -6% |
| mxfp8_dim0 | 8192x4096 | 303 | 66.7 | 66.6 | - | nested never engages |

Full table: `python analyze.py` over `results.jsonl`.

## Findings

1. **Persistent threshold miss (heuristic).** For K >= 2048 the nested kernel
   is looped (INNER threshold 1024). Looped nested is slower than not fusing
   for MXFP8/MXFP6. Multi-kernel measured looped 37.7 vs persistent 26.5us and
   picked persistent; forcing persistent reproduces those numbers everywhere.
   Options: enable multi-kernel for nested reductions, or raise the persistent
   threshold for them.
2. **Loop-reordering post-pass runs one round.** `Scheduler.fuse_nodes` runs
   plain rounds to a fixpoint, the reorder round once. QuACK's `to_mxfp4`
   builds codes from the flat `(M, K)` reshape, so its scale load carries
   `FloorDiv(d1, 32)` and the initial pass sees "no shared data"; only the
   reorder round fuses reduction+codes, after which the nibble pack is never
   retried. Iterating the reorder round (applied to scheduler.py, uncommitted)
   fuses standalone `to_mxfp4` to one kernel: 10.2->7.4, 44.1->32.3,
   177.9->123.2us, byte-exact.
3. **`dynamic=True` makes `block_size` symbolic.** Dynamo lifts the
   `block_size=32` default to a SymInt, so the reduction takes `r0_numel` at
   runtime, cannot be persistent, and the `(M, K)`-shaped code conversion
   cannot fuse. QuACK ships MXFP6 and MXFP4-byte with `dynamic=True`:
   131us vs 5.9us static at 2048x3072 (22x), 682 vs 27.4us at 8192x4096 (25x).
   A literal block size with symbolic M, K still leaves 2 kernels (82us)
   because Inductor cannot prove `K % (K // 32) == 0`. After RMSNorm, nested
   reduction absorbs the whole thing regardless (10.4us).
4. **NVFP4 standalone pack.** `_sub_parent_broadcast_access_relations`
   compares raw loop frames; after reordering the pack has one merged axis
   while the scale writer has two, so it declines before comparing indices
   (and the odd lane reads `(2*d0+1)//16`). Normalizing into the
   `(x, child_r)` domain like the source-relation path would fix it.
5. Packed MXFP6's `torch.stack` 6-bit pack never fuses (pointwise_cat) and
   costs as much as the quantize itself. `to_mx_dim0` never engages nested.

## Files

`probe.py`, `cell.py`, `run.py`, `analyze.py`, `results.jsonl`,
`probe_*.json|log`, `standalone_variants.py`, `planner_trace.py`,
`fusion_why.py`, `reorder_fixpoint.py`, `nvfp4_trace.py`, `dyn_dump.py`,
`dyn_placeholders.py`, `mxfp4_fixpoint_bench.py`, `mxfp6_static_block.py`,
`rmsnorm_mismatch.py`.
