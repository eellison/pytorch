# Fused AllReduce + Bias + RMSNorm: Benchmark & Analysis

**Hardware:** 8x NVIDIA B200 (NVLink interconnect)  
**Dtype:** bf16  
**Benchmark:** CUDA graphs, median of 200 iterations, L2 cache flush between iterations  
**Hidden dims:** 4096, 5120, 7168, 8192, 16384 (covering Llama-8B through 405B, DeepSeek-V3)  
**Batch sizes:** 1, 4, 8, 16, 32, 64, 128, 512, 1024, 2048  
**World sizes:** 2, 4, 8  

## Background

In transformer inference, every layer performs two allreduce operations (after attention `o_proj` and MLP `down_proj`), each followed by a residual add (bias) and RMSNorm. Fusing these into a single kernel eliminates extra memory passes and kernel launch overhead — critical at small batch sizes where decode latency is dominated by kernel launch costs, not compute.

FlashInfer provides hand-tuned CUDA C++ kernels (from TRT-LLM) for this fusion, with one-shot and two-shot allreduce variants. We explore whether Triton-based and in-tree PyTorch approaches can match or exceed this performance.

## Kernels Benchmarked

### Custom Fused Triton Kernels (this work)

| Kernel | Strategy | Barriers | Best regime |
|---|---|---|---|
| **lamport** | Push to all peers, poll with neg-zero sentinels, zero barriers | 0 | Decode, D≤8192 |
| **push_sync** | Push to all peers, signal-pad barrier, read local | 2 | Small batch, all D |
| **kraken** | Write local, barrier, read all peers (from kraken library) | 2 | D=16384 small batch |
| **inplace** | Like kraken but input already in symm_mem (no copy) | 2 | D=16384 small batch |
| **butterfly** | Recursive-doubling in log2(W) rounds, zero remote writes | 2+log2(W) | Medium batch, large D, high ws |
| **lamport 2shot** | Lamport with reduce-scatter + all-gather | 0 | Not competitive |
| **push_sync 2shot** | Barrier-based two-shot | 3 | Not competitive |

### In-Tree PyTorch (no custom kernels)

| Kernel | Strategy | Best regime |
|---|---|---|
| **symm_1s** | `torch.ops.symm_mem.one_shot_all_reduce` + `torch.compile(bias+rmsnorm)` | Small batch |
| **symm_2s** | `torch.ops.symm_mem.two_shot_all_reduce_` + `torch.compile(bias+rmsnorm)` | Large batch, ws=8 |
| **nccl** | `dist.all_reduce` + `torch.compile(bias+rmsnorm)` | Baseline |

### FlashInfer (TRT-LLM CUDA C++)

| Kernel | Strategy |
|---|---|
| **fi_1shot** | Hand-tuned Lamport one-shot with 128-bit vectorized loads |
| **fi_2shot** | Barrier-based reduce-scatter + all-gather |

## Results Summary

### Decode (b=1..128) — Latency-Critical Path

At small batch sizes, kernel launch overhead and synchronization costs dominate. Fused kernels provide 2-3x speedup over NCCL.

| Config | Best non-FI | FlashInfer | Speedup vs FI | Speedup vs NCCL |
|---|---|---|---|---|
| ws=2, D=4096, b=1 | **10us** (lamport) | 14us (fi1) | 1.4x faster | 2.3x faster |
| ws=2, D=8192, b=1 | **12us** (lamport) | 14us (fi1) | 1.2x faster | 2.3x faster |
| ws=4, D=5120, b=1 | **12us** (lamport) | 14us (fi1) | 1.2x faster | 2.6x faster |
| ws=8, D=4096, b=1 | **16us** (lamport) | 16us (fi1) | ~tied | 2.4x faster |
| ws=8, D=8192, b=1 | 20us (push_sync) | **14us** (fi1) | 1.4x slower | 2.1x faster |
| ws=8, D=16384, b=1 | 24us (kraken) | **16us** (fi1) | 1.5x slower | 2.0x faster |
| ws=8, D=16384, b=128 | **55us** (symm_2s) | 74us (fi2) | 1.3x faster | 1.3x faster |

**Takeaway:** Lamport one-shot dominates at ws=2-4 and ws=8 D≤8192. FlashInfer's fi_1shot wins at ws=8 D=16384 small batch due to lower fixed overhead from hand-tuned CUDA.

### Prefill (b=128..2048) — Throughput-Oriented

At large batch sizes, bandwidth efficiency matters more than launch overhead. Two-shot patterns (reduce-scatter + all-gather) reduce remote memory traffic.

| Config | Best non-FI | FlashInfer | Speedup vs FI |
|---|---|---|---|
| ws=2, D=8192, b=2048 | **69us** (lamport) | 121us (fi1) | 1.7x faster |
| ws=4, D=8192, b=2048 | **160us** (symm_2s) | 158us (fi2) | ~tied |
| ws=8, D=4096, b=2048 | **80us** (symm_2s) | 110us (fi2) | 1.4x faster |
| ws=8, D=8192, b=2048 | **192us** (symm_2s) | 225us (fi2) | 1.2x faster |
| ws=8, D=16384, b=2048 | **299us** (symm_2s) | 592us (fi2) | 2.0x faster |

**Takeaway:** `symm_mem.two_shot_all_reduce_` + compiled rmsnorm (fully in-tree, no custom kernels) beats FlashInfer at large batch across all configs.

### Practical Size Limits

vLLM's fused allreduce cutoffs on B200 (SM 10.0):

| D | ws=2 max tokens | ws=4 max tokens | ws=8 max tokens |
|---|---|---|---|
| 4096 | 8192 | 4096 | **128** |
| 8192 | 4096 | 2048 | **64** |
| 16384 | 2048 | 1024 | **32** |

Above these limits, vLLM falls back to NCCL. So the fused kernels only matter in the decode regime (small batch). The prefill numbers are useful context but wouldn't be exercised in production at ws=8.

## Design Insights

### Why Lamport wins at small D, loses at large D

Lamport's zero-barrier polling checks every element for a neg-zero sentinel. At D=4096 (BLOCK_D=4096), this is cheap — 4096 elements polled per peer. At D=16384, it's 16384 elements with a `tl.sum()` reduction per poll iteration, which becomes expensive. Barrier-based kernels (kraken, push_sync) pay a fixed ~4us per barrier regardless of D.

### Why two-shot is hard in Triton

FlashInfer's two-shot works because every thread block participates in both phases (reduce-scatter and all-gather), with work split by elements. In Triton with 1-CTA-per-row, two-shot means (W-1)/W of CTAs are idle during reduce-scatter — pure overhead. Our push_sync two-shot and lamport two-shot both suffered from this.

### Why symm_mem two-shot is so fast

`torch.ops.symm_mem.two_shot_all_reduce_` is a C++ kernel that uses the same reduce-scatter + all-gather pattern as FlashInfer but with PyTorch's optimized symmetric memory infrastructure. It's not fused with rmsnorm, but the separate compiled rmsnorm kernel is cheap enough that the total is competitive.

### Butterfly: a novel approach

The butterfly (recursive-doubling) kernel does log2(W) rounds of pairwise exchange. At ws=8: 3 remote reads total vs 8 for one-shot. No remote writes at all (peers read from our buffer). But 5 barriers (1 initial + 3 rounds + 1 final) add ~20us fixed cost, making it only competitive at medium batch, large D.

## Files

| File | Description |
|---|---|
| `push_poll_allreduce_rmsnorm.py` | All kernel implementations |
| `bench_sweep.py` | Benchmark script (`torchrun --nproc-per-node=N bench_sweep.py`) |
| `plot_results.py` | Chart generation |
| `best_decode.png` | Summary chart: decode regime (b=1..128) |
| `best_prefill.png` | Summary chart: prefill regime (b=128..2048) |
| `bench_decode.png` | All methods: decode regime |
| `bench_prefill.png` | All methods: prefill regime |
| `check_numerics.py` | Correctness verification vs kraken |

## Reproducing

```bash
# Run full benchmark at world_size=8
torchrun --nproc-per-node=8 bench_sweep.py

# Run all world sizes
python bench_sweep.py --sweep-world-sizes

# Generate charts
python plot_results.py

# Verify numerics
torchrun --nproc-per-node=8 check_numerics.py
```
