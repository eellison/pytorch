"""Swizzle-only scaling sweep: does the padding strategy matter once the scale
buffer is big enough to be bandwidth-bound rather than launch-bound?

zeros_scatter writes the whole padded buffer twice (zero fill + scatter);
predicated / pad_scatter write it roughly once. At the scale-tensor sizes the
rmsnorm benchmarks use, everything is launch-bound and they tie. This sweeps up
to multi-MB buffers to find the crossover.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bench_main_cuda import graph_bench
from blocked_variants import VARIANTS

import torch._inductor.config as inductor_config

SHAPES = ((989, 256), (8000, 250), (32700, 512), (100_000, 512), (32768, 512))
PICK = ("fpad", "index_put", "predicated", "zeros_scatter", "pad_scatter")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    config = parser.parse_args()

    fn = VARIANTS[config.variant]
    results = []
    for rows, cols in SHAPES:
        torch.manual_seed(0)
        scale = torch.randint(0, 255, (rows, cols), dtype=torch.uint8, device="cuda")
        with inductor_config.patch({"fx_graph_cache": False, "triton.cudagraphs": False}):
            compiled = torch.compile(fn, fullgraph=True, dynamic=False)
            out = compiled(scale)
        torch.cuda.synchronize()
        padded = out.numel()
        got = graph_bench(lambda: compiled(scale), warmup=config.warmup,
                          samples=config.samples, calls_per_graph=config.calls_per_graph)
        row = {"variant": config.variant, "rows": rows, "cols": cols,
               "padded_bytes": padded, "pad_frac": 1 - rows * cols / padded,
               "equal_to_fpad": bool(torch.equal(out, VARIANTS["fpad"](scale)))}
        row.update(got)
        results.append(row)
        print(f"{config.variant:<14}{rows}x{cols:<6} {padded/1e6:6.2f}MB pad={row['pad_frac']*100:5.1f}%"
              f"  {got['median_us']:8.2f}us  eq={row['equal_to_fpad']}", flush=True)
        torch._dynamo.reset()
    config.output.write_text(json.dumps(results, indent=2) + "\n")
    print("wrote", config.output, flush=True)


if __name__ == "__main__":
    main()
