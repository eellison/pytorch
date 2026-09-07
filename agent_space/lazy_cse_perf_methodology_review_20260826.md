# F2a performance methodology review

Date: 2026-08-26

Reviewed:

- `agent_space/f1_perf_baseline_20260826.py`
- `agent_space/archived_wrapper_replay_20260826.py`
- `agent_space/f1_perf_baseline_20260826/{results,replay_results}.json`
- `agent_space/f1_perf_baseline_results_20260826.md`
- `agent_space/f2a_perf_20260826/results.json`
- `agent_space/f2a_perf_matched_20260826/{f1,f2,paired}/`

## Findings

### Initial blocker, repaired: the first F1/F2 wrappers used different cache policies

Every archived F1 wrapper has `force_disable_caches=False`; every current F2
wrapper has `force_disable_caches=True`. The current capture script was also
modified after the F1 manifest was produced:

- F1 manifest timestamp: 22:05
- current capture-script timestamp: 22:14
- current capture-script SHA-256:
  `fd90d1e88f706aeb792f80cdf79875f74f9105dde2a38692ee1fd242c4c1d48d`

This is not a runtime-timing overhead issue, because compilation is outside the
timed region. It is nevertheless a comparison-invalidating configuration
difference: cache policy participates in Triton autotune/dynamic-RBLOCK
selection. In particular, default mode is not a fixed configuration. The old
F1 results must remain historical only.

This was repaired in `agent_space/f2a_perf_matched_20260826`: all 24 F1 and all
24 F2 wrappers embed `force_disable_caches=True`, and both sides were regenerated
from the same current capture script. Each case still used a fresh cache.

### Initial resource gap, repaired in paired replay

All 24 rows in `agent_space/f2a_perf_20260826/results.json` have
`launchers=[]`. Therefore that manifest alone cannot establish matching launch
configuration, register count, spill count, or shared-memory use.

The revised `archived_wrapper_replay_20260826.py` correctly extracts resources
from each loaded wrapper after compilation. All 24 matched results contain
exactly one launcher per side, and an independent audit confirmed equal config,
`num_warps`, and `num_stages` for every pair. Future aggregation should turn
those audit checks into assertions rather than merely reporting the fields.

### The manifest's worktree provenance hash is incomplete, but this run is recoverable

`f1_perf_baseline_20260826.py::diff_hash()` hashes `git diff --binary`, which
omits staged changes. Both follow-up worktrees contain the reviewed base in the
index, so HEAD plus this hash does not identify the source tree that generated
the wrapper.

The rerun artifacts can be tied to these independently recorded hashes:

- HEAD for both: `6b8ef64bd373b221c6ed1ba4f4b2f1edef2643c3`;
- F1 staged diff: `8c0b3a3bf060571e6f90ecca4cd371af29c68575ce48e964690a28cf5ed1fc78`;
- F1 unstaged diff: `ce766480d4bed7972584fbbc146608dc4a3d6c50c3e5a075f6b98c8e65f1c16e`;
- F1 full `git diff HEAD`: `195484cf6db6ebb7f4d3e9f649ce361ff4fc80d9e1de6027d3a5237649b658e1`;
- F2 staged diff: `195484cf6db6ebb7f4d3e9f649ce361ff4fc80d9e1de6027d3a5237649b658e1`;
- F2 unstaged diff: `c186184aeb29b4c0dc1ffc0aa8f65b1e5879138fcce9a9aba81889b64eace84b`;
- F2 full `git diff HEAD`: `b8a3b068b663da2d7f091e2642edf93cdcc63f2b31e92399982770e52aa1c011`;
- capture script: `fd90d1e88f706aeb792f80cdf79875f74f9105dde2a38692ee1fd242c4c1d48d`;
- replay script: `5493be195f7a4c2014c67efd3fa78f74ffdeddbff3f3adecfc59bf80bf510592`.

The archived wrapper itself is sufficient to identify the timed GPU program,
but the full-tree hash is needed to tie that program back to F1 or F2 source.

### Paired statistics support the two-percent gate, but should be emitted directly

The replay loop is well designed: both graphs use shared inputs, every sample
contains a fixed replay count, and order rotates by round. However, the result
only reports per-wrapper medians. It does not report the within-round F2/F1
ratios that make the run paired, and the default of 15 rounds gives the first
label eight first-position samples and the second label seven.

The repaired sweep used 20 rounds, so each side ran first ten times. I computed
the per-round ratios from the saved samples. No row regressed by 2%: the largest
median slowdown was 0.24%. The unchanged-source MXFP4 controls were essentially
flat; eleven were within 0.24%, and the remaining default 4096x8192 row measured
0.99% faster. That control distribution is consistent with ordinary benchmark
drift and supports treating the much larger NVFP4 changes as real.

The aggregation script should still emit the median and spread of the
within-round ratios directly. The old standalone baseline measured the same F1
MXFP4 128x4096 looped wrapper at about 10.25 us, while fresh paired runs measured
about 8.25 us. This demonstrates why standalone historical medians must not be
used for the F1/F2 delta.

### The 24 rows cover F2a's affected factor-2 surface, not every generalized path

The matrix is appropriately broad for F2a:

- NVFP4 and MXFP4;
- small and large row counts;
- exact-block, loop-tail, and large-reduction widths;
- default, fixed persistent, and fixed looped schedules.

That is enough to accept or reject factor-2 performance once the blockers above
are fixed. It does not by itself establish a no-regression claim for factor-4
MXFP6, internal forwarding, scale swizzle, preshuffle, or DCN preshuffle. Those
paths may be covered by a normalized generated-source identity check, because
F2a is expected not to change them. At least one representative MXFP6/DCN
wrapper/resource cross-check should remain in the final evidence if the summary
makes a stack-wide performance claim.

## What is already sound

- The F1 and F2 manifests have the same 24 `(format, shape, mode)` keys.
- Every row has exactly one archived generated source and one staged generated
  kernel.
- Static inspection found identical `get_args`, wrapper `call`, and launch
  metadata for every corresponding pair after normalizing only the temporary
  kernel path.
- Both manifests name the same HEAD, B200 device/capability, Torch build, CUDA
  version, and requested mode definitions.
- The comparator loads wrappers under unique module names, uses the first
  wrapper's inputs for both, captures each wrapper independently, and rotates
  execution order.
- Exact output comparison covers every returned tensor. The special packed
  float4 dtype is compared through its uint8 representation.
- The revised comparator obtains register/spill/shared-memory data from the
  exact compiled wrapper used for timing.

## Repaired sweep results

All 24 pairs passed these structural checks:

- exact outputs for every returned tensor;
- one generated staged kernel and one compiled launcher per side;
- identical F1/F2 launcher config, warp count, and stage count;
- 20 timing samples of 100 graph replays, with balanced rotating order;
- no new spills.

MXFP4 is a useful negative control: all 12 F1/F2 normalized sources are
identical, all launcher resources are identical, and median deltas range from
-0.99% to +0.24%.

NVFP4 has the intended source change in all 12 rows. Each F1 source contains the
old uint8 bitcast-broadcast/FP8-bitcast round trip; no F2 source does, while both
sides still contain exactly one FP8 conversion. Median F2 changes were:

| Shape | Default | Looped | Persistent |
| --- | ---: | ---: | ---: |
| 128x4096 | 0.00% | -0.07% | -14.43% |
| 4096x4096 | -19.73% | -10.23% | -36.08% |
| 4096x4608 | -14.22% | -9.98% | -66.78% |
| 4096x8192 | -29.55% | -10.64% | -22.83% |

Negative percentages are faster. The large improvements also appear in the
within-round paired ratios with narrow p10-p90 spreads. The 128-row default and
looped cases are latency-bound and remain flat; they do not regress.

## Requirements for future reruns

1. Freeze the F2a worktree and record its full diff hash.
2. Generate both F1 and F2 manifests from the same capture-script revision,
   in separate fresh processes, with the same explicit cache policy and all
   other environment variables identical.
3. Pair exactly one F1 and one F2 wrapper for each of the 24 matching manifest
   keys, with one fresh process per pair.
4. Seed input generation before `get_args()` for reproducibility; continue
   sharing that one input set between F1 and F2.
5. Use an even number of rotating rounds, at least 12, with 100 graph replays
   per sample.
6. Require one staged kernel, one launcher, identical launch configuration,
   and exact F1/F2 outputs for each matched-config pair.
7. Record registers, spills, shared memory, medians, all raw samples, and
   per-round ratios. Repeat noisy or greater-than-2% regression rows.
8. Keep default-mode results as the user-facing performance result. Treat the
   forced modes as diagnostics, especially the intentionally pathological
   persistent spill-cliff rows.
9. Retain normalized-source identity evidence for unaffected factor-4/DCN
   cases, with one representative resource cross-check.

## Verdict

The repaired wrapper-replay sweep is suitable evidence for F2a. It shows exact
same-config outputs, no performance regression, unchanged MXFP4 controls, and
large NVFP4 gains where duplicated full-width scale-side work was removed. The
older `f1_perf_baseline_20260826` timings should be retained only as historical
context, not mixed into the final F1/F2 delta.
