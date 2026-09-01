# torch.compile Time-to-First-Kernel Experiment

This branch preserves experimental work from July 2026 to reduce the warm
runtime overhead between invoking a compiled callable and launching its first
Triton kernel. The primary workload was FlexAttention through the normal
`torch.compile` path without CUDA graphs or the C++ wrapper.

This is a research snapshot, not a polished upstream pull request. It combines
several related optimizations so that the results and implementation ideas are
not lost.

## What changed

- Flatten the AOTAutograd-to-Inductor handoff with a boxed runtime callable.
- Use a lower-overhead AOTAutograd save-for-backward path.
- Remove redundant Inductor input size/stride assertions when Dynamo and
  AOTAutograd already guard those inputs.
- Fuse CUDA device guarding and raw-stream lookup into a C++ helper.
- Batch adjacent CUDA output allocations.
- Populate static Triton launcher state before the first runtime launch.
- Guard storage offsets for eligible non-view inputs instead of performing
  runtime alignment copies. Views retain the runtime safety check.

## Measurements

The final recorded FlexAttention run used:

```text
agent_space/bench_flex_ttfk_variants.py --variants current --iters 300
```

It reported:

```text
time to first kernel median: 45.45 us
time to return median:       61.43 us
p90 first kernel:            58.48 us
p90 return:                  77.43 us
```

These are local measurements from the original development environment, not a
portable performance guarantee. The benchmark was noisy enough that deltas
below about 2 us required paired runs or direct microbenchmarks.

The generated Inductor wrapper was no longer the dominant measured component.
A profiled run attributed roughly 3.7 us of self time to that wrapper, 1.1 us
to the Inductor launcher, and 0.8 us to the output-code handoff. The larger
remaining opportunity appeared above Inductor in Dynamo's steady-state
eval-frame/guard machinery and the AOTAutograd runtime wrapper. Profiler
overhead inflates those absolute values, so they are useful mainly for ranking.

## Suggested review order

1. `torch/_functorch/_aot_autograd/runtime_wrappers.py` and
   `torch/_inductor/output_code.py` for the boxed handoff.
2. `torch/_inductor/compile_fx.py`, `torch/_inductor/graph.py`, and
   `torch/_inductor/codegen/wrapper.py` for the generated-wrapper changes.
3. `torch/csrc/dynamo/guards.cpp`, the CUDA device overrides, and
   `torch/_inductor/runtime/triton_heuristics.py` for the runtime fast paths.
4. The accompanying Dynamo, AOTAutograd, Inductor, Triton, and user-stream
   tests.

## Validation and limitations

The original session reported that Python compilation checks and the targeted
AOTAutograd, Inductor, Triton, CUDA, and user-stream tests passed. `lintrunner
-a` did not complete because Pyrefly could not fetch `sympy` from PyPI; it did
not report a code lint failure before stopping.

Before upstreaming, split or simplify the combined patch, reproduce the
benchmark on a current base, and revalidate cache behavior, multi-device and
user-stream semantics, view alignment, and launcher initialization.

Prepared with assistance from an AI coding tool.
