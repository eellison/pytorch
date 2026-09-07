# Production DCN MXFP6 preshuffle analysis

Date: 2026-08-26

This measures the existing `(2048, 3072)`, group-size-32 DCN-style graph on
NVIDIA B200 using the reviewed `#191775` worktree.

## Result

The production preshuffle graph emits two kernels and does not reach the new
sub-parent `(4,3)` path (`codegen_nested_reduction == 0`).

| Kernel | Work | Mean CUDA time |
| --- | --- | ---: |
| Quantize | fused `addcmul`, group max/scale, E2M3 encode; writes 6,291,456 `int32` codes and 196,608 scales | 17.11 us |
| Pack/copy | reads the code buffer, packs 4 codes into 3 bytes, applies the inverse packed-data permutation, writes 4,718,592 bytes | 13.09 us |

The per-kernel sum is 30.20 us. The independent end-to-end benchmark median was
about 28.36 us.

Generated code is in `agent_space/dcn_preshuffle_43_dump.txt`. The profiling
harness is `agent_space/profile_dcn_mxfp6_paths.py`.

## One-kernel estimate

Measured nearby paths:

- row-major direct packing: 16.45 us, one kernel;
- scale-only preshuffle main kernel: 16.11 us;
- separate scale swizzle: 2.40 us.

A kernel that directly writes row-major packed bytes and uses the preshuffled
scale address should therefore be about 16-18 us. It removes the 25.2 MB
`int32` temporary write and roughly 37.7 MB of code-buffer reads. The remaining
cost is dominated by input traffic and FP32 scale/E2M3 conversion, not the three
packed-byte stores.

The supplied hand-written DCN kernel is not a clean oracle at this shape: its
`GROUPS_PER_THREAD=79` tail stores are missing a per-program bound and race into
the next program's packed output. With that mask corrected it takes about
24.1 us.
