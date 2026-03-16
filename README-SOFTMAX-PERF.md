# Softmax Performance Optimization Branch

This branch contains **3 independent commits** that optimize PyTorch Inductor's softmax performance, achieving **~42% speedup** and **Helion parity** on key sizes.

## Quick Summary

| Change | Files | Impact | Risk |
|--------|-------|--------|------|
| 1. Persistent threshold 1024→8192 | `choices.py` | ~35% | Low |
| 2. fast_max in twopass softmax | `simd.py`, `triton.py`, `ir.py` | ~19% | Low |
| 3. Generalized fast_max analysis | `simd.py`, `triton.py` | ~2% | Medium |

## Key Results (B200, fp16)

- **(4096, 4096): 21.1µs → 12.3µs** (vs Helion 17.4µs) — **29% faster than Helion**
- **(4096, 8192): 42.3µs → 28.8µs** (vs Helion 28.7µs) — **Helion parity**

## Commits

```
073ac1ecc7c Add comprehensive softmax performance optimization documentation
c078b0d731f [inductor] Generalize fast_max with dataflow-based NaN analysis
ccf13586ede [inductor] Add fast_max reduction type for softmax twopass fallback
8160712d8c8 [inductor] Increase persistent reduction threshold for INNER reductions
```

## Files

- **`SOFTMAX_PERF.md`** — Comprehensive technical documentation
- **Code changes** — 3 focused commits, ready for upstreaming
- **Validation** — 178 fuzz tests, ncu profiling, correctness verification

## Landing Strategy

Each commit is independent and can land separately:
1. Persistent threshold (simple, high-impact)
2. Targeted fast_max (proven safe for twopass)
3. General dataflow analysis (extends to any pattern)

See `SOFTMAX_PERF.md` for detailed technical analysis, performance measurements, and implementation notes.
