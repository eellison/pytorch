# Template Reduction Epilogue Findings

This tracks the matmul epilogue block-quantization path:

`matmul -> local grouped reduction -> reduced scale -> full-resolution quant`

## What Works

The full template epilogue chain now works for the basic block-quant pattern.

Required pieces:

- Allow template epilogue fusion to see reduction consumers.
- In `TritonTemplateKernel`, lower a single-value reduction epilogue by
  reshaping the template accumulator from `[BLOCK_M, BLOCK_N]` to
  `[BLOCK_M, BLOCK_N // group_size, group_size]`, then reducing the group axis.
- Run reduced-resolution epilogues under a reduced store context, with
  `template_out_shape = [BLOCK_M, BLOCK_N // group_size]`.
- Run later full-resolution epilogues under the normal template tile context.
- Resolve reduced values lazily when a full-resolution epilogue loads them:
  `[BLOCK_M, groups] -> [BLOCK_M, groups, 1] -> [BLOCK_M, groups, group] ->
  [BLOCK_M, BLOCK_N]`.
- For ragged tiles, derive the reduced store mask from the template mask:
  `[BLOCK_M, BLOCK_N] -> [BLOCK_M, BLOCK_N // group_size, group_size]`, then
  reduce the group axis and cast back to bool.

This passed numerics for:

- `M=17, N=384, group=128`

This case is important because it exercises the reduced mask path and multiple
N tiles.

## Current Status

The implementation is still intentionally scoped, but no longer stops at
`matmul -> grouped amax`.

- the reshape/reduce emission is shared with standalone nested reduction via
  `emit_grouped_reduction`
- template code owns only the template-specific shape, mask, store context, and
  reduced-to-full load resolution
- the scheduler only admits simple single-value reductions, matching the
  standalone nested-reduction allowlist
- reduced stores now derive a reduced pointer tile, so they store once per
  output group instead of redundantly storing from every lane in the group
- scheduler vertical fusion has one constrained extra rule: a later pointwise
  template epilogue may read a reduced-output epilogue value as a
  repeat-interleave style broadcast, e.g. write `[M, N // G]`, read `[M, N]`
  with index `d1 // G`
- a focused test covers:
  - amax-only output
  - returning both the original matmul output and the grouped amax
  - the full quant-like chain fusing into one template kernel
  - unsupported grouped reductions (`argmax`, `var`) failing open as two
    kernels

The quant pattern is:

1. template output at full resolution: `[M, N]`
2. grouped reduction output: `[M, N // group]`
3. scale pointwise at reduced resolution: `[M, N // group]`
4. quantizing pointwise back at full resolution: `[M, N]`

That chain now codegens as one template kernel for the test shape.

## Remaining Limits

This is not a general arbitrary-resolution template fusion framework.

- The grouped reduction must be the first epilogue after the template.
- Exactly one grouped reduction is supported.
- The grouped reduction must be a simple single-value Triton reduction from the
  nested-reduction allowlist.
- Template output is assumed to be a 2D `[BLOCK_M, BLOCK_N]` tile.
- The grouped axis is the N/block axis.
- Reduced-to-full scheduler matching only accepts the direct repeat-interleave
  projection; it does not try to prove arbitrary semantic equivalence.

## Autotune Interaction

The GEMM tile must contain whole local groups. For `group_size=128`, choices
with `BLOCK_N=32` or `BLOCK_N=64` cannot fuse the local amax. The prototype
rejected those choices.

The focused test forces a valid `BLOCK_N=128` template choice. In real tuning,
the epilogue fusion search must see a compatible choice; incompatible choices
should fail open during template rendering.

A production version needs either:

- choice filtering before epilogue benchmarking, or
- enough choice search to skip incompatible tiles without disabling the whole
  fusion opportunity.

## Possible Follow-Up

This is enough for the inline-asm NVFP4-style template epilogue direction, but
there are still obvious generalizations:

- support grouped axes other than N
- share more of the nested-reduction resolution/family machinery instead of
  keeping a template-local handler
- filter invalid template choices earlier in epilogue benchmarking
- add coverage for the real inline-asm pack once the benchmark pattern is ready
