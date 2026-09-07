# Combined dim0+dim1 NVFP4 quantization kernel

Date: 2026-08-21

## Scope and environment

One handwritten Triton kernel, one load of a bf16 `[M, K]` input, four
outputs: dim0 (rowwise) packed fp4 + e4m3 scales and dim1 (columnwise)
packed fp4 + e4m3 scales. Compared against today's 3-kernel / 3-read
Inductor schedule.

- Worktree `agent_space/nvfp4_dual_dim_wt` at stack tip `3f62ae27baa`,
  run via `agent_space/run_wt.py`, installed torch `2.15.0a0+git7c78410`,
  Triton 3.8.0, NVIDIA B200, `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`.
- Semantics/layouts exactly match `agent_space/bench_dim1_quant_casts.py`:
  16-element groups, scale `e4m3((amax/6).clamp(1e-12, 448))`, value
  `x / scale.float()`, pairs packed per byte via
  `cvt.rn.satfinite.e2m1x2.f32` (even element in the low nibble). No
  per-tensor scale, row-major (non-swizzled) scale layouts.
- dim0: payload `[M, K/2]` u8, scale `[M, K/16]`. dim1 row-major (rm):
  payload `[M/2, K]`, scale `[M/16, K]`. dim1 transposed (t): payload
  `[K, M/2]`, scale `[K, M/16]`, both contiguous.
- torchao 0.13.0: `mslk_quantize_nvfp4` requires the MSLK extension (not
  importable here) and there is no dedicated NVFP4 dim1 kernel; the only
  runnable torchao path is the pure-torch `NVFP4Tensor.to_nvfp4`
  (dim0-only, eager, ~50x slower than compiled - included as a row for
  completeness, not a kernel yardstick).

## Files

- `agent_space/nvfp4_dual_dim_kernel.py` - the kernel + host wrapper.
  Constexprs: `DIM1_TRANSPOSED` (dim1 output layout), `PRECISE`
  (per-element `x / s` vs per-group `1/s` then multiply), `EVEN`
  (mask-free fast path when the tile divides the shape).
- `agent_space/bench_nvfp4_dual_dim.py` - correctness + benchmark
  harness (run through `run_wt.py`).
- `agent_space/probe_nvfp4_dual_dim.py`, `probe_nvfp4_trans_variant.py` -
  spill/kernel-time probes and the dim1-formulation A/B.
- Logs: `agent_space/nvfp4_dual_dim_bench_nocd.log`,
  `agent_space/nvfp4_dual_dim_bench_cd.log`. Generated status-quo kernel
  dumps in `agent_space/nvfp4_dual_dumps/`.

## Kernel shape

Each program owns a `BLOCK_M x BLOCK_K` tile (both multiples of 16, so
every 16-group in either direction is tile-local; no cross-CTA
communication):

1. Load the tile once, convert to fp32.
2. dim0: reshape `(BLOCK_M, g0, 16)`, amax over the trailing 16, e4m3
   scale, divide, `tl.split` adjacent column pairs, pack via the e2m1x2
   inline asm, store payload + scale (all stores coalesced along K).
3. dim1: `tl.trans(x)` then the structurally identical code on the
   transposed register tile (groups again trailing). rm layout stores
   `tl.trans` back (coalesced along K); t layout stores the transposed
   tile directly.

What was tried for the dim1 side, in order:

- `tl.reshape(y, (M/2, 2, K)) + tl.permute + tl.split` on the row-major
  tile: works, 92-110 us at 16384x7168.
- explicit `tl.trans(x)` then dim0-style code: consistently ~8-10% faster
  (one layout conversion for the whole dim1 side instead of separate
  conversions for the axis-0 reduction and the row-pair split). Folded
  into the kernel. Bitwise identical outputs.
- A shared-memory staging variant was not needed: Triton's own layout
  conversion for `tl.trans` already goes through shared memory; the NCU
  profile (below) shows the kernel is not DRAM-bound, so hand-written
  smem staging would attack the wrong limit.

## Correctness

Oracle: the compiled reference paths from `bench_dim1_quant_casts.py`
(`nvfp4_dim0`, `nvfp4_dim1_rm`, `nvfp4_dim1_t`, nested=True so dim0 is
the staged single kernel).

- All four outputs bitwise equal (`torch.equal` on u8 views) at
  1024x1024, 4096x4096, 16384x7168 and the non-multiple-of-tile edge
  shapes 528x1040 and 2064x4112 (masked partial tiles), for both dim1
  layouts, multiple tile configs, and BOTH `PRECISE=True` and
  `PRECISE=False`.
- Independent near-oracle: torchao's pure-torch `nvfp4_quantize(x, 16)`
  (dim0) and `nvfp4_quantize(x.t().contiguous(), 16)` (equals the dim1_t
  layout): ZERO mismatched payload or scale bytes at every shape tested,
  up to 58.7M payload bytes at 16384x7168.
- `PRECISE=False` (one reciprocal per 16-element group, then multiply) is
  not formally guaranteed to round identically to per-element division,
  but produced zero differing bytes across ~470 MB of compared outputs
  (torchao's own reference also uses the reciprocal form).

## Results (do_bench wall medians, us)

Floor = one input read + all four outputs = 3.125 B/elem at 8 TB/s.
eff TB/s = single-load ideal bytes / wall time. Status quo = compiled
staged dim0 kernel + compiled 2-kernel dim1 back-to-back (3 kernels, x
read 3x = 2.3x the single-load traffic). dual_graph = one compiled graph
returning all four outputs (still 3 kernels; mix-order reduction ON by
default did not change this).

### No coordinate-descent tuning

| 16384x7168 (floor 45.9) | us | eff TB/s |
|---|---:|---:|
| combined_rm_fast (16x256w2) | 93.2 | 3.94 |
| combined_rm precise (16x256w2) | 116.7 | 3.14 |
| combined_t_fast (32x128w2) | 129.2 | 2.84 |
| combined_t precise (32x128w2) | 142.4 | 2.58 |
| inductor dim0 staged | 70.7 | 5.19 |
| inductor dim1_rm (2 kernels) | 208.1 | 1.76 |
| inductor dim1_t (2 kernels) | 250.9 | 1.46 |
| status_quo_rm (dim0+dim1_rm) | 277.4 | 1.32 |
| status_quo_t | 324.6 | 1.13 |
| dual_graph_rm (1 graph, 3 kernels) | 280.6 | 1.31 |
| dual_graph_t | 312.3 | 1.18 |
| torchao to_nvfp4 eager (dim0 only) | 5263 | 0.07 |

| 4096x4096 (floor 6.55) | us | eff TB/s |
|---|---:|---:|
| combined_rm_fast (16x512w4) | 18.5 | 2.84 |
| combined_rm precise (16x128w2) | 22.5 | 2.33 |
| combined_t_fast (32x128w2) | 26.7 | 1.97 |
| inductor dim0 staged | 16.4 | 3.20 |
| inductor dim1_rm | 37.0 | 1.42 |
| status_quo_rm | 50.1 | 1.05 |
| dual_graph_rm | 48.4 | 1.08 |

| 1024x1024 (floor 0.41) | us |
|---|---:|
| combined (any layout/config) | 8.2 |
| status_quo_rm | 27.7 |
| dual_graph_rm | 16.4 |

1024x1024 is launch/host-overhead bound for everything; the combined
kernel wins purely by having one launch + one wrapper.

### With coordinate-descent tuning (status quo baselines)

cd only affects the compiled baselines (the handwritten kernel is
identical; its rows are repeat measurements).

| 16384x7168 | us | eff TB/s |
|---|---:|---:|
| combined_rm_fast (16x256w2) | 95.1 | 3.86 |
| inductor dim0 staged +cd | 68.6 | 5.35 |
| inductor dim1_rm +cd | 103.4 | 3.55 |
| inductor dim1_t +cd | 224.1 | 1.64 |
| status_quo_rm +cd | 179.2 | 2.05 |
| status_quo_t +cd | 292.0 | 1.26 |
| dual_graph_rm +cd | 169.0 | 2.17 |
| dual_graph_t +cd | 273.6 | 1.34 |

| 4096x4096 | us |
|---|---:|
| combined_rm_fast | 18.6 |
| status_quo_rm +cd | 35.8 |
| dual_graph_rm +cd | 35.8 |

At 1024x1024 cd does not help the baselines (dual_graph 15.4 vs
combined 8.2 us).

### Headline

Combined kernel vs the best status quo (dual_graph_rm + cd):
1024x1024 8.2 vs 15.4 us (1.9x), 4096x4096 18.6 vs 35.8 us (1.9x),
16384x7168 95.1 vs 169.0 us (1.8x; 2.95x vs untuned). Kernel-only time
at 16384x7168 is 83.7 us = 4.39 TB/s on single-load traffic, 1.8x above
the 45.9 us / 8 TB/s floor; ~10 us of the wall is host wrapper +
4-output allocation overhead.

## Config findings

- Best band matches the prior cd finding: 16-row tiles, K-extent 256-512,
  warps chosen for 64-128 elements/thread (16x256w2 = 64/th,
  16x512w4 = 64/th). Scaling warps up with block size loses
  (16x512w8: 111.7 us; 16x1024w8: 108.6 us vs 93.2 for 16x256w2).
- Taller tiles lose: 32-row ~ +12%, 64-row ~ +45% (more dim1 layout
  conversion traffic per element).
- No register spills in any measured config (max 229 regs at 16x512w2
  precise; the winning fast configs sit at ~96 regs).
- `PRECISE=False` (reciprocal-multiply) is worth ~20% at large shapes
  (116.7 -> 93.2 us): per-element fp32 division is one of the two big
  compute costs. The e2m1x2 pack asm (one byte per call, 2 elems) is the
  other and is irreducible at the language level.
- dim1_t (transposed outputs) costs ~1.4x vs dim1_rm: the `[K, M/2]` u8
  payload store has only BLOCK_M/2-byte contiguous runs. If the wgrad
  consumer can take the rm layout, rm is the one to ship.

## What limits performance

NCU on combined_rm_fast 16x512w4 at 16384x7168 (84.3 us; the winning
16x256w2 profiles at 83.7 us kernel-only):

- DRAM throughput 52% (~4.2 TB/s), SM 63.5%, L1/TEX 71.7%.
- The kernel is L1/compute-mixed-bound, NOT DRAM-bound: two quantize
  passes per element (2 divisions or multiplies, 2 abs+max trees, one
  pack-asm call per element across the two dims) plus the shared-memory
  layout conversion for the dim1 side.
- Compare: the staged dim0-only Inductor kernel does half the quantize
  work and reaches 5.19 TB/s; our combined kernel does 2x the quantize
  work of that kernel per loaded byte and lands at ~2x its time minus
  the saved re-reads.
- Path to floor (45.9 us) would need the quantize compute itself to
  shrink: packed 2-elems-per-asm-call is already used; a warp-specialized
  or TMA/persistent design mostly helps the memory side, which is not
  the limit. The realistic ceiling for this register-level design is
  ~80-90 us at this shape.

## Inductor design note (what codegen would need)

Today's dual-quant graph compiles to 3 kernels reading x 3x: the staged
sub-parent kernel handles dim0 (amax + pack in one kernel), and dim1
falls apart into amax-kernel + pack-kernel because
`_try_get_sub_parent_source_layouts` proves lanes only on the trailing R
extent while dim1 lanes sit at stride K
(`agent_space/dim1_mx_nvfp4_cast_investigation_20260821.md` Q2).

To GENERATE the combined kernel, Inductor needs three composed
capabilities:

1. **Sibling mix-order fusion including staged nodes.** The two amax
   reductions are orthogonal siblings over one read - exactly what
   `FusedMixOrderReductions` (config `triton.mix_order_reduction`, ON by
   default in OSS) fuses. It did not engage here (the dual graph still
   compiled to 3 kernels with the config ON; the dim0 amax is consumed
   into a `FusedStagedReduction` during fusion, and the exact decline
   point was not probed). Even if it engaged, mix-order fuses the
   reduction stage only - the packing epilogues would still be homeless.
2. **Strided-lane sub-parent proof + codegen ((b) from the dim1 note).**
   The dim1 packing epilogue needs lanes proven on a non-trailing axis
   (normalize to `(x_hi, lane, x_lo)` with the lane at stride K) and
   lane-split stores along an X-derived axis. This is required for ANY
   single-kernel dim1 NVFP4, fused or not.
3. **A tile-local grouping contract.** What makes the handwritten kernel
   trivial is 16 | BLOCK in both directions, making every scale group
   tile-local. A generic "2D-tiled pointwise with per-tile group
   reductions" schedule (the blockwise-2D mode sketched as (c) in the
   dim1 note) expresses this directly: tile constraint = lcm of the two
   group sizes, then both quantize sides are pure per-tile epilogues.
   The generic tiled path already proves it can run the dim1 MXFP8 cast
   at 4.15 TB/s under cd; the missing pieces are only (a) dual outputs
   with different group axes from one tile and (b) half-resolution
   (packed) stores, which the staged stack's lane machinery already does
   for trailing axes.

Pragmatic recommendation, matching the combined-MXFP8 conclusion
(`agent_space/combined_mxfp8_kernel/README.md`): do not extend the
sub-parent planner for this; a `torch.library.triton_op` around the
handwritten kernel is the smallest correct integration and Inductor
already embeds such kernels into the generated wrapper. The scheduler
path ((c) + (b)) only pays off if blockwise-2D dual-quant becomes a
family of workloads rather than one kernel.

## Addendum 2026-08-22: mix-order decline chain + amax experiment

Probe: `agent_space/probe_mix_order_decline.py` (instrumented verbatim
replica of `MixOrderReduction.can_fuse`, monkeypatched; worktree at tip
plus one uncommitted allowlist edit, see below). Shape 16384x7168.

### Decline chain on the dual-quant graph (empirical)

For every amax-pair combination - raw nodes, scale-fused nodes
(`op0_op1 x op3_op4`), and with nested ON the staged node
(`op0_op1_op2[FusedStagedReduction] x op3_op4`) - the FIRST failing
check is `has_mix_reduction_orders`: both sides have group
`(7340032, 16)` (numel M*K/16, rnumel 16), and the check requires
`g1 == reversed(g2)`. Groups are EQUAL, not reversed. The reduction-type
allowlist is never reached for this graph. A `FusedStagedReduction`
node DOES reach `can_fuse` (passes is_reduction/sibling checks) and dies
at the same shape check, so "staged nodes excluded from mix-order" is
NOT the blocker. The packing nodes (group `(58720256, 1)`) fail
`is_reduction` as expected. Root cause: dim1's 16-group amax is a
GROUP-LOCAL reduction, so it presents the same (x, r=16) shape as dim0
instead of the reversed (ncol, nrow) shape mix-order's contract wants
(full-column reduction over all of node1's rows).

### Allowlist +max experiment (canonical shape)

Worktree edit (uncommitted, marked EXPERIMENT): add "max"/"min" to the
`{sum, prod}` allowlist at the end of `MixOrderReduction.can_fuse`
(`torch/_inductor/scheduler.py` ~458). NO codegen changes were needed:
the accumulator path is fully generic (`default_accumulator` -> -inf,
`get_reduction_combine_fn` -> maximum, `triton_helpers.max2`, and the
host-side final reduce already maps max -> `amax`).

On the canonical-shape pair `x.abs().amax(dim=1)` + `x.abs().amax(dim=0)`
(groups `(16384, 7168)` x `(7168, 16384)`, reversed - passes):

| dual_full_amax 16384x7168 | kernels | us |
|---|---:|---:|
| allowlist {sum,prod} (today) | 3 | 152.6 |
| allowlist +max | 1 triton + host amax | 69.7 |

Fused version is bitwise-equal to eager (max is exact). 2.19x. Kernel
shape: `MixOrderReductionGrid`, RSPLIT_SIZE=16 rows/program, per-row
persistent r0=7168 amax inline, column partials accumulated in
registers and stored to a 29.4 MB fp32 workspace
(`(pid + idx*num_programs)*r0_numel + r0_index`), final combine
HOST-side: `ws.view(1024, 7168).amax(dim=0)` (an extra ATen kernel, not
a grid barrier, in this configuration). Control: `dual_full_sum` fuses
today (mix_order=1, 71.8 us; compiled != eager bitwise only because
split accumulation reorders float sums - expected).

With the extended allowlist, dual_group_amax and the full dual-quant
graph still decline at `has_mix_reduction_orders` - the allowlist was
never their blocker.

### Is mix-order's split accumulation sane for rnumel=16 groups?

Structurally inapplicable, not just overhead. Mix-order's contract
computes node2 outputs of size ncol (= node1's rnumel = 16) by
combining partials across ALL row chunks - for dual-quant that would be
a 16-element global amax, not the `[M/16, K]` per-group scales. Also,
in node1's flattened x domain (m*K/16 + kgroup), the 16 members of a
dim1 group are strided by K/16, so "aligned row chunk contains whole
groups" only holds in the 2D (m, k) view, which mix-order never forms.
Quantified: today's 2-kernel dual_group_amax alone is 189.4 us; the
handwritten 2D-tile kernel does both amaxes PLUS both quantize/pack
passes and all four stores in 83.7 us with zero workspace bytes,
vs the ~59 MB workspace round-trip mix-order machinery spends in its
own regime. Tile-local grouping needs no cross-CTA combine at all.

### Effort estimates (relative)

- (a) amax in mix-order: SMALL (a one-line allowlist change + tests;
  codegen already generic; 2.19x win on canonical shapes). Worth
  landing on its own, but does NOTHING for dual-quant.
- (b) mix-order x FusedStagedReduction composition: #190595-sized
  (codegen must emit the staged kernel body inside the RSPLIT loop);
  ALSO does nothing for dual-quant, whose shapes never reverse-match.
  Skip for this goal.
- (c) strided-lane sub-parent proof + codegen: #190594-sized; required
  for any single-kernel dim1 NVFP4 pack in the flattened-domain world.
- (d) blockwise-2D tile-local schedule: #190594-sized, but it is the
  route that matches the proven-fast kernel (2D (m, k) tile, both group
  reductions tile-local, half-res packed stores on both axes). In the
  2D view the dim1 lanes are an ordinary y-axis pair split, so (d)
  absorbs the hard part of (c) in a simpler form.

Sequencing if the goal is Inductor generating the single-load combined
kernel: land (a) opportunistically (small, independent), skip (b), and
build (d) directly - generalizing mix-order's 1D (x, r) contract cannot
reach group-local dual quantization; the 2D tile schedule can, and the
handwritten kernel is its codegen blueprint (16-row tiles,
trans-then-rowwise dim1 formulation, optional reciprocal-multiply).

## Addendum 2: general-capability framing + producer fusion (2026-08-22)

Empirical anchor (probe `rms_dual_quant`, RMSNorm feeding both quant dims,
16384x7168, nested ON): 3 kernels, 267.2 us. K0 = the flagship staged
kernel (norm + dim0 amax + pack, reads raw x, writes 64 KB stats + dim0
outputs). K1/K2 = dim1 amax / dim1 pack, each re-reading RAW x and
RECOMPUTING the normalization inline from the stats buffer. The normed
intermediate is never materialized - Inductor already uses
recompute-from-stats. The entire waste is the 3x raw-x read
(~700 MB read vs 234 MB ideal). `rms_dim0_only` nested ON = 1 kernel,
110.6 us (flagship intact).

What happens to the norm producer in each route:

- (a) reduction-type widening in mix-order: no interaction. The quant
  family never presents mix-order's reversed-group shape, so the norm
  keeps fusing into the staged dim0 kernel and dim1 stays 2 kernels.
  Generic value is real but elsewhere (true row+column sibling
  reductions over one buffer).
- (b) staged x mix-order sibling composition: generalizing it would put
  the staged body (norm included) inside the RSPLIT row loop, preserving
  producer fusion BY CONSTRUCTION - but no member of the quant family
  (MX 32, NVFP4 16, MXFP6, either dim) has the full-column reversed
  shape the contract needs. Dead end for this family; skip.
- (c) generic non-trailing-lane sub-parent (lanes at stride K, any
  rate): dim1 becomes ONE kernel for every format/rate/layout. The norm
  stays fused in the staged dim0 kernel unchanged; the fused dim1 kernel
  inherits the existing recompute-from-stats pattern (reads raw x +
  64 KB stats). Result: 2 kernels, raw x read 2x (~470 MB). Standalone
  value for dim1-only (wgrad) workloads independent of dual-dim.
- (d) tile-local grouping contract (tile/band = lcm of group extents;
  16 NVFP4, 32 MX, static per dtype). Two sub-forms with different
  producer behavior:
  - (d1) one-shot 2D tile: single-kernel dual-quant for a materialized
    input, but a norm CANNOT fuse in - a 2D tile never sees a whole row,
    so stats are not computable tile-locally. Best schedule with a norm:
    [stats kernel] + [dual-quant tile kernel recomputing norm from
    broadcast stats] = 2 kernels, x read 2x. Same traffic class as (c).
  - (d2) band-persistent form: band = lcm rows x full K, looped over K,
    stats pass then quantize pass. This is the existing staged/nested
    kernel with its row-band widened from 1 to lcm rows and the grouped
    axis allowed on Y (the exact case `_nested_sub_parent_rate` rejects
    today, pinned by
    test_producer_consumer_rejects_sub_parent_grouped_axis_x). Norm +
    dim0 + dim1, all rates, ONE kernel, x read once (366 MB ideal,
    ~45.9 us floor; realistic ~120-150 us vs today's 267 given the
    flagship's 110.6 us for the norm+dim0 half). In the band view the
    dim1 lanes are ordinary in-band Y splits, so (d2) subsumes (c) for
    the fused case.

CRITICAL PATH for producer fusion: (d2). It is the only route where the
norm ends up inside the dual-quant kernel; (a)/(b) never touch the
schedule, and (c)/(d1) top out at 2 kernels with one redundant raw-x
read (still ~1.8x better than today's 3-read schedule).

Family coverage notes for (d2): rm outputs stay coalesced along K;
transposed outputs reuse the generic transposed-store codegen (perf
gated on the launch-config heuristic, per the dim1 note); dynamic K is
what the looped staged kernels already handle, dynamic M needs
M % band == 0 guards or masked bands; MXFP8 needs no lane machinery at
all (full-res payload), NVFP4/MXFP6 reuse the staged rate/lane planner
generalized to the Y axis.

Sequencing (general capabilities): land (a) now (small, independent,
proven). Build (d2) as the next major stack increment - it reuses the
landed producer contract, staged identity, and lane machinery, and (c)'s
in-band case falls out of it. Build standalone (c) first only if
dim1-only workloads need a win before (d2) lands. Skip (b).
