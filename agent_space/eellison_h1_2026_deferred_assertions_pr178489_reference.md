# PR #178489 deferred-copy performance reference

## Source PR

- PR: https://github.com/pytorch/pytorch/pull/178489
- Title: `[inductor] Defer copy_misaligned_inputs to first use`
- Author: `tianrengao`
- PR state from GitHub API: closed. Labels include `Merged`, `Reverted`,
  `Stale`, `module: inductor`, and `release notes: inductor`.
- Git history in this checkout shows #178489 landed briefly as
  `808c9bf25f5d` (`[inductor] Defer copy_misaligned_inputs to first use
  (#178489)`) and was autoreverted by `c154b7fc8fd3` (`Revert
  "[inductor] Defer copy_misaligned_inputs to first use (#178489)"`).
- Use #178489 as the PR-reference source for the dashboard performance claim,
  not as the durable landed H1 credit. The durable landed copy-deferral path is
  #179039, followed by the #180599 rename from `copy_misaligned` to
  `copy_if_misaligned`.

## Claimed performance impact

The PR body describes the mechanism:

> Instead of checking all input alignments in a wrapper before the compiled
> call() function, defer each alignment check + clone to just before the first
> kernel that reads that input. This hides the alignment check cost behind GPU
> execution of earlier kernels.

The PR body also ties this to DeepSeek-R1:

> On DeepSeek-R1 (TP=8, 8xH100), codegen analysis shows redundant alignment
> checks ... that were previously executed serially in the wrapper before the
> first GPU kernel launch. With this change, they are distributed across kernel
> boundaries in the generated code, allowing GPU execution of earlier kernels
> to overlap with later alignment checks.

The exact textual dashboard claim in #178489 is:

> Performance improved on all e2e huggingface models for 6-10%. Slight
> improvements on timm model and torchbench.

The attached dashboard screenshot is `Geometric mean speedup (threshold =
0.95x)` for H100 CUDA inference bfloat16, comparing main against
`gh/tianrengao/46/head`. The visible suite/backend rows support the quoted
claim as changes in geomean speedup:

| Backend | HuggingFace geomean | Relative change | timm geomean | Relative change | TorchBench geomean | Relative change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `default` | `1.90x -> 2.04x` | `+7.4%` | `1.77x -> 1.79x` | `+1.1%` | `1.80x -> 1.84x` | `+2.2%` |
| `cudagraphs_freezing` | `2.50x -> 2.68x` | `+7.2%` | `1.94x -> 1.97x` | `+1.5%` | `2.82x -> 2.83x` | `+0.4%` |
| `cudagraphs_dynamic` | `2.38x -> 2.53x` | `+6.3%` | `1.50x -> 1.55x` | `+3.3%` | `2.54x -> 2.55x` | `+0.4%` |
| `aot_inductor` | `2.33x -> 2.56x` | `+9.9%` | `1.80x -> 1.85x` | `+2.8%` | `3.09x -> 3.27x` | `+5.8%` |
| `cudagraphs` | `2.39x -> 2.60x` | `+8.8%` | `1.81x -> 1.85x` | `+2.2%` | `2.80x -> 2.81x` | `+0.4%` |
| `cpp_wrapper` | `2.24x -> 2.39x` | `+6.7%` | `1.78x -> 1.83x` | `+2.8%` | `2.58x -> 2.70x` | `+4.7%` |

Rows with missing data in the screenshot (`inductor_max_autotune`,
`inductor_deterministic_perf`) are not useful for before/after percentage
claims.

## Connection to landed H1 2026 PRs

- #177783: https://github.com/pytorch/pytorch/pull/177783
  - Landed commits in the local H1 inventory:
    `f0c3582ef2d`, `779c78b3d7d`.
  - This is the size/stride assertion half of the pattern: defer
    `assert_size_stride()` calls from the top of the generated `call()` wrapper
    until just before the first kernel that reads the input.
  - Its PR body explicitly claims reduced CPU overhead before the first kernel,
    with later checks more likely to be hidden by GPU work.

- #178489: https://github.com/pytorch/pytorch/pull/178489
  - Reference PR for the H100 dashboard performance claim above.
  - It applied the same first-use deferral idea to `copy_misaligned_inputs`.
  - It was reverted for an implementation issue: the autorevert diagnosis says
    the deferred alignment code emitted raw Python strings into wrapper lines,
    which broke FXIR backend conversion. That caveat affects #178489 as a
    landed artifact, but it does not directly negate the dashboard speedup
    mechanism.

- #179039: https://github.com/pytorch/pytorch/pull/179039
  - Landed commits in the local H1 inventory:
    `ddaac926c33`, `55fc17f8653`.
  - The PR body says it is a "resumbit of
    https://github.com/pytorch/pytorch/pull/178489" and repeats the same
    copy-deferral mechanism: defer each alignment check plus clone to just
    before the first kernel that reads that input; keep the wrapper path for
    mutated inputs that require writeback.
  - This is the durable landed PR to credit for the #178489 copy-deferral
    mechanism.

- #180599: https://github.com/pytorch/pytorch/pull/180599
  - Landed commit in the local H1 inventory: `13a6d0ddbc7`.
  - This is a naming follow-up: "update copy_misaligned to copy_if_misaligned"
    / "Renaming to make output code more clear."
  - It is relevant when connecting the PR history to current generated-code
    names and the local microbench report, which measures `copy_if_misaligned`.

## Applicability and caveats

- #178489 should be cited as external dashboard evidence for the performance
  claim, while shipped-impact credit should go to #179039/#180599 and #177783.
- The #178489 dashboard was H100, CUDA, inference, bfloat16, and model-suite
  level. It is not the same as the local B200 microbench evidence below.
- The exact individual model names and per-model values are not present in the
  PR body text or issue/review comments. The readable attachment gives
  suite/backend geomean speedup rows, while the PR body summarizes this as
  "all e2e huggingface models for 6-10%."
- CudaGraph replay is a caveat for attribution. #178489 says CudaGraph replay
  does not invoke the generated `call()` function, so replay does not hit the
  deferred alignment checks. The benefit is primarily in direct/generated-call
  host placement before or between kernel launches, not in steady graph replay.
- The local H1 summary already marks wrapper overhead and first-use deferral as
  needing parent-vs-landed checkouts for strict landed-impact claims. The
  microbench is an isolation of the mechanism, not a replacement for a full
  parent/landed model benchmark.

## Local microbench support for the mechanism

Local report: `agent_space/eellison_h1_2026_deferred_assertions_benchmarks.md`.

The local B200 report uses scratch-only monkey-patches because current HEAD has
no stable config toggle for eager-vs-deferred placement. It measures time from
entering the compiled function to the first external matmul, which isolates the
host-wrapper path before the first GPU kernel.

For #177783-style size/stride assertion deferral:

- Generated wrapper placement dropped pre-first-kernel `assert_size_stride`
  calls from `6` in the eager scratch variant to `2` in current/deferred code.
- With `64` tail inputs first used after the first matmul, median
  `first_mm_us` improved from `65.32 us` eager-wrapper to `33.61 us`
  deferred (`1.94x`).
- With `256` tail inputs, median `first_mm_us` improved from `129.01 us` to
  `69.16 us` (`1.87x`).

For #179039/#180599-style misaligned-input copy deferral:

- Generated wrapper placement dropped pre-first-kernel `copy_if_misaligned`
  calls from `6` in the eager scratch variant to `2` in current/deferred code.
- With `8` misaligned tail inputs, median `first_mm_us` improved from
  `78.89 us` eager-wrapper to `29.87 us` deferred (`2.64x`).
- With `32` misaligned tail inputs, median `first_mm_us` improved from
  `218.18 us` to `36.41 us` (`5.99x`).
- Synchronized wall time for the copy rows was mostly flat, which is consistent
  with the intended mechanism: the host placement win is before the first
  kernel and later copy/check costs can be hidden by GPU work.

The graph-partition probe in the same report also shows current generated
partition code places `copy_if_misaligned` inside `partition_0`/`partition_1`
before kernels that read those inputs, rather than unconditionally in the outer
wrapper before all partitions.

Overall, #178489 provides H100 model-suite dashboard evidence for the
performance claim; #177783 and #179039/#180599 are the H1 landed mechanisms;
and the local B200 microbench supports the causal explanation by directly
showing that first-use deferral reduces pre-first-kernel host latency when many
inputs are not consumed by the first kernel.
