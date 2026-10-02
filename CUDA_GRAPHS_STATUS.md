_Drafted with an AI assistant (Claude); numbers are copied from our eval tables, not rerun for this note._

Branches: `hosttrace/eager-aten-wip` holds only validated snapshots (currently int4). `hosttrace/candidate` holds int5a/int5b; it is merged (fast-forward) into the validated branch once the suite A/B against int4c passes.

# Dynamic-shape CUDA graphs: status, 2026-10-02

GB300 (aarch64), bf16 inference / AMP training, HF + timm + TorchBench (61 rows). "Ours" = this branch;
"trees" = cudagraph trees (reduce-overhead); "default" = torch.compile without graphs.

## How it works

- **Trace once, replay at any shape.** On the first call we run the host code with symbolic sizes and
  addresses. Every kernel launch is recorded with its grid and arguments as integer programs over those symbols,
  and the guards are whatever the host code branched on or the kernel compiled in. We then capture one CUDA graph.
- **Replay is all C++.** A later call evaluates the programs, checks the guards, patches only the kernel nodes
  whose parameters changed (`cuGraphExecKernelNodeSetParams`), and replays. There is no Python on the hit path.
- **Memory is eager's.** Allocations come from the caching allocator in eager order. There are no private pools,
  so turning graphs on costs no extra memory.
- **What gets traced:**
  - Triton: Inductor kernels and user `@triton.jit`.
  - Eager ATen: TensorIterator pointwise and reductions, through the SymInt TensorIterator.
  - CuTe DSL.
- **Harvested libraries.** We can't compute the parameters of cuBLAS, cuDNN, SDPA or RNG calls. Instead we
  capture each call a few times, classify every parameter qword (address, size, constant, scratch), verify with a
  seeded relaunch, and keep one launch row per shape key. A hit is a hash lookup plus SetParams.
- **Misses stay local.** A new shape is new parameters in the same graph. A guard miss affects one kernel's
  selector, not the whole graph. When something can't be traced, it becomes an eager step between graph
  segments, not a whole-graph fallback.

## Measured: compile line (Inductor kernels; ours vs trees on the same compiled graph)

Merge11, 3 repeats. Speedups are geomeans.

| | HF | timm | TB |
|---|---|---|---|
| inference, vs eager | 4.14x | 2.51x | 4.66x |
| inference, vs default | 1.92x | 1.21x | 2.58x |
| inference, vs trees | 1.03x | 1.02x | 1.11x |
| training, vs eager | 2.87x | 2.52x | 2.90x |
| training, vs default | 1.58x | 1.32x | 2.03x |
| training, vs trees | 0.96x | 0.99x | 0.98x |

**Coverage (fully graphed, no eager steps):**
- Inference: 26/27, 17/17, 16/17.
- Training: 23/27, 13/17, 9/17. The training gap is mostly `convolution_backward`.

**Graph memory over default (same kernels)**, median per row:
- Inference: ours +0 / +4 / +42 MB. Trees +176 / +208 / +202 MB, flagged in 23/13/13 rows.
- Training: ours +78 / +294 / +80 MB. Trees +425 / +208 / +258 MB.
- Process max reserved in training, median: ours about +70 MB vs trees +1.0 to 6.5 GB.

## Measured: eager line (no torch.compile; the plain eager model is traced and replayed)

int4/int4c, 61 rows, 59 ran.

- **Speedup vs eager:** 1.75x geomean, or 1.88x with the cuDNN conv family harvested.
  - HF 2.04x, timm 1.22x, TB 1.99x (2.66x with conv).
- **Reference points:**
  - Ours is 0.85x of default compile.
  - A static `torch.cuda.graph` capture of the same call gets 2.18x; ours reaches 0.83x of it, 0.90x with conv.
    That capture has no guards and no shape handling.
- **Correctness:** every row's output is bitwise equal to eager.
- **Memory vs eager:** reserved median -15 MB, max +203 MB.
- **What's left eager:** convolution is 1270 of the 1551 remaining eager steps. Harvesting the conv family brings
  the total down to 281.
- **Whole-call declines (4):** `.item()`, a CPU-to-CUDA copy during capture, a CPU buffer of symbolic size, and
  `as_strided_`.

## Measured: cost with the feature off

- **Plain eager:** user-space instructions per op are within +1.1% of base, and GPT-2 / resnet18 forwards within
  +0.14%. Device kernels are unchanged.
- **One narrow regression** (int4): contiguous mixed-dtype elementwise ops cost about +0.7 to 1.5 us per call.
- **int5b:** the eager guard is 4 host instructions, and the microbenchmark is within noise. libtorch_cuda
  grows by 6.2 MB.

## What improved recently

- **int4.** Fixed a host-step memory-plan bug: M2M100 was +1.8 GB over eager and is now about 0. The eager line
  got 1.11x faster than the previous tree (TB 1.25x).
- **int5a** (on `hosttrace/candidate`, validation running). An independent red team ran against int4c and found 8 S0 and 2 S1 issues. Most were global state
  missing from the replay key: TF32/matmul precision, SDPA backend, autocast, grad mode, cuDNN flags.
  - int5a keys on that global state and turns static reads into guards.
  - The red team's final rerun on int5a found no new silent-wrong results.
  - int5a also introduced a memcpy capture regression; int5b fixes it.
- **int5b** (on `hosttrace/candidate`, validation running):
  - Every byte of an eager pointwise launch is now typed: launch sites report the parameter layout, and functors
    declare their fields.
  - op_db: 3111/3111 ops bitwise, 0 witness declines.
  - Pointwise decline census: 1031 -> 128 rows.
  - A failed capture now falls back per segment instead of for the whole call.
- **SymInt TensorIterator** (standalone, 6 commits on the fork branch `symint-tensoriterator`, no PR yet): fake
  tensor's C++ route is 1.25 to 5.2x faster than the symbolic route.

## Not measured yet (next)

- **Full suite re-time on int5b.** GPU 0 was held by another job, so int5b has only spot checks.
- **Serving cold/warm start against SGLang and vLLM.** What we have so far is SGLang's own baseline on Qwen3-8B
  (cc012abd):
  - It captures 126 graphs: 74 piecewise prefill (split around attention) over 4..16384 tokens, plus 52 decode.
  - Graphs add 22 s to a warm start. A cold start takes 194 s to ready, 114 s of it first-bucket JIT.
  - Graph memory is 2.7 GB.
  - Ours would be one capture with no padding, and prefill including attention in one graph. That needs varlen
    attention (cu_seqlens) checked first.
- **Bucket-stride sweep:** warm-up time and graph memory vs padding cost, with ours as a single point.
- **Not yet supported:** NCCL (designed, about 1k LOC, not built), and pinned H2D copies inside a graph.

## Reference numbers (GB300)

- **Kernel cost on device:** eager 4.1 us vs 1.2 to 1.5 us inside a graph.
- **A graph break** (graph, one eager kernel, graph) adds about 6.4 us of device time.
