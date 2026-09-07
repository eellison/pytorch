---
name: bench-variant-isolation
description: Benching several compiled variants in one process silently inflates all but the last; also insert a 3s cooldown between samples or B200 throttling swamps sub-5% effects
metadata:
  type: feedback
---

When comparing multiple `torch.compile`d variants of the same workload, bench
**one variant per process**. Compiling N variants in a single process and then
timing them in a loop inflates every variant except the last one in the dict by
20-70%. Sweeping forward then reversed and keeping the min does NOT fix it --
the contamination tracks position in the variant dict (compile / cudagraph
capture order), not bench order.

**Why:** it silently manufactures a winner. Two separate padded-swizzle result
matrices (v4, v5) each "proved" whichever variant happened to be last was
1.5-2x faster than the rest; isolated runs showed all variants within 2%.
The tell is that the last variant's interleaved number matches its isolated
number to ~1% while every other variant is way off.

**How to apply:** drive the sweep from bash with one subprocess per variant
(`--variants $v --output results/iso_$v.json`), then join the JSON. Cross-check
any interleaved result against at least one isolated run before reporting it.

Second, independent hazard (found 2026-09-03 on B200 while A/B-ing generated
Triton kernels): back-to-back `graph_bench` calls thermally throttle. A single
kernel measured repeatedly drifts 104 -> 123 us, with only the very first
sample at boost clocks. Min-across-sweeps then reports whichever variant
happened to run first, and medians converge to the throttled rate, hiding
real differences. **Insert `time.sleep(3)` before every measurement.** With the
cooldown, medians become reproducible to ~0.05 us and 0.6-1% effects are
resolvable; this is what distinguished a 0.6% cast-elision win from a
1.1% zero-init win from ~12% of pure ordering noise.

Third refinement (2026-09-03): **the 3 s cooldown is only correct for kernels
of ~100 us.** For few-microsecond kernels it backfires -- after 3 s idle the GPU
has dropped to idle clocks and a 20 x 5 us graph replay cannot ramp them back,
so the measurement lands at idle clocks. The same `99x4096` FP4 shape reads
5.08 us with a 3 s cooldown and 2.96 us with a 0.02 s cooldown plus
`calls_per_graph=100`; FlashInfer is inflated the same way, so even the ratio
shifts (0.94 -> 0.99). Rule: long kernels need the cooldown to avoid throttling,
short kernels need a short cooldown and more calls per graph to stay at boost
clocks. Pick the cooldown from the kernel duration, and never compare absolute
numbers across the two regimes.

Related: [[worktree-overlay-harness]], [[nested-reduction-stack]]
