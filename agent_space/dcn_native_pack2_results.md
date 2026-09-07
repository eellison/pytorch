# DCN MXFP6 native pack=2 results

Date: 2026-08-26. GPU: NVIDIA B200. All timings are CUDA-graph replay
medians. No reviewed worktree was edited.

## Production shape

Graph: fused fp16 DCN, MXFP6 group size 32, row-major packed bytes, and the
aligned scale-only preshuffle. Shape: `2048x3072`.

| Variant | Median | Kernels / nested | Stores | Inline asm | Registers / spills |
|---|---:|---:|---:|---:|---:|
| Software conversion | 16.376 us | 1 / 1 | 4 | 0 | 47 / 0 |
| Native pack=2 | 8.698 us | 1 / 1 | 4 | 1 | 32 / 0 |
| Native pack=2 + zero canonicalization | 9.015 us | 1 / 1 | 4 | 1 | 32 / 0 |
| Native pack=2 + UE8M0 reciprocal | 9.005 us | 1 / 1 | 4 | 2 | 32 / 0 |
| Corrected D112902015 | 24.216 us | 1 / 0 | 4 | n/a | n/a |

All Inductor variants selected a 6144-CTA launch with `XBLOCK=32`,
`R0_BLOCK=32`, two warps, and one stage. Corrected D uses grid 2509,
`GROUP_LOAD=64`, 79 groups per program, two loop iterations, and four warps. The
canonical native path is 1.82x faster than software and 2.69x faster than the
corrected handwritten kernel. The raw intrinsic is 1.88x and 2.78x faster,
respectively, but has the signed-zero caveat below.

The current production graph already CSEs scale-side work: software emits one
`log2`, zero `floor`, one `ceil`, and two `exp2` calls. Native pack=2 division
emits `1/0/1/1`; the UE8M0 reciprocal version emits `1/0/1/0`. Therefore the
reciprocal removes the last `exp2`, but does not reduce `log2` further in this
graph. Its measured performance is effectively tied with canonical pack=2
division.

## Exactness

For the seeded production input, raw, canonical, and reciprocal native output
are bitwise identical to the compiled software output. Native and software have
the same pre-existing difference from eager/corrected D because fused DCN omits
the fp16 `addcmul` materialization: 67 of 196608 scale bytes and 16359 of
4718592 packed bytes differ.

The raw PTX conversion preserves negative zero as FP6 code `0x20`, while the
software encoder canonicalizes zero. An all-negative-zero input therefore
differs in every packed byte. `torch.where(scaled == 0, 0, codes)` restores
bitwise equality; the canonical and reciprocal variants both pass that case.

## Odd tail and padded scale

At `rows=7`, `width=96` (three non-power-of-two groups per row), the exact
append-guard prototype emits one nested kernel with four masked stores and one
native-conversion asm site. It is bitwise equal to the eager software encoder.

Ordinary pad/reshape scale swizzling remains two kernels. In a separate scratch
composition of #193599, the staged stack, and the exact append guard, the full
dynamic XDL padded scale output is one nested kernel after narrowing the staged
planner's alias/mutation rejection to epilogue nodes. The result is bitwise
exact and has five masked stores: one scale scatter, three packed-byte stores,
and one auxiliary padding store. The two observed `rows=7` medians were 1.536
us and 1.837 us.

## Artifacts

- `agent_space/bench_dcn_scale_only_native_pack2.py`
- `agent_space/probe_dcn_native_pack2_tail.py`
- `agent_space/dumps/dcn_scale_only_native_pack2_0.py`
- `agent_space/dumps/dcn_scale_only_native_pack2_recip_0.py`
- `agent_space/dumps/dcn_native_pack2_tail_xdl_0.py`
- `agent_space/mxfp6_native_tail_pr193599`
