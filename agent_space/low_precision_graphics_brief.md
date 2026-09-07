# Low Precision Kernel Presentation: Excalidraw Brief

This doc is a handoff for graphics agents. The goal is to create simple,
technical Excalidraw diagrams that make the low-precision kernel story visual.
Use a consistent visual language across diagrams:

- Blue boxes: original high-precision tensors / compute.
- Green boxes: low-precision tensors or packed values.
- Orange boxes: scale tensors.
- Purple boxes: compiler transformations / fusion decisions.
- Red arrows: HBM reads/writes or distributed communication.
- Black arrows: in-register or in-kernel dataflow.

Keep the diagrams dense and technical, not marketing-style. Prefer 16:9
canvas layout. Use concise labels that can be read in a slide deck.

## Diagram 1: Why Quant Fusion Matters

Purpose: motivate the whole talk.

Show two side-by-side pipelines:

Left, unfused:

```text
RMSNorm kernel -> HBM -> Quant scale kernel -> HBM -> Cast/pack kernel -> HBM -> GEMM/collective
```

Right, fused:

```text
One fused kernel: RMSNorm + scale + cast/pack -> GEMM/collective
```

Emphasize:

- Fewer HBM round trips.
- Fewer kernel launches.
- Less data communicated if quantization happens before a collective.
- The fused upstream kernel can itself become faster because its epilogue or
  consumer work changes memory traffic.

Suggested title: `Quant Fusion Turns Memory Traffic Into Register Dataflow`

Prompt for graphics agent:

```text
Create an Excalidraw diagram comparing unfused and fused low-precision quantization pipelines.
Left side should show RMSNorm, scale computation, cast/pack, and GEMM/collective as separate kernels with red HBM arrows between each stage.
Right side should show one fused kernel containing RMSNorm + scale + cast/pack, with black in-register arrows inside the kernel and one output arrow to GEMM/collective.
Use blue for high precision, orange for scales, green for low precision outputs, purple for fusion/compiler.
Keep labels short and technical.
```

## Diagram 2: Hardware Format Matrix

Purpose: show why compiler-generated kernels need to compose with many formats.

Draw a small matrix:

```text
Hardware | Common low precision formats | Compiler implications
H100     | FP8                           | scale + cast + GEMM
B200     | MXFP8, NVFP4, MXFP4           | blocked scales, swizzles, inline asm pack
```

Below the matrix, draw three format-specific branches:

- FP8: value tensor + scalar/per-tensor/per-channel scale.
- MXFP8: value tensor + E8M0 scale tile.
- NVFP4/MXFP4: packed 4-bit values + scale tile.

Suggested title: `Different Hardware Wants Different Quantization Kernels`

Prompt for graphics agent:

```text
Create an Excalidraw slide showing a hardware/format matrix for H100 FP8 and B200 MXFP8/NVFP4/MXFP4.
Under it, draw three branches showing FP8, MXFP8, and NVFP4/MXFP4 data layouts.
Make clear that B200 formats introduce blocked scale tensors, swizzled layouts, and packing instructions.
Use simple rectangles and arrows, no decorative imagery.
```

## Diagram 3: MXFP8 Scale Swizzle / Index Inversion

Purpose: explain `to_blocked` / scale layout conversion and why index
inversion matters.

Show:

1. Input tile `[M, K]`.
2. Block amax over groups of 32 -> scale tensor `[M, K/32]`.
3. E8M0 conversion -> `uint8` / `float8_e8m0fnu` scale.
4. Reshape/permute/reshape into hardware blocked layout.

Then show the compiler transformation:

Unfused:

```text
reduction writes natural scale layout -> HBM -> swizzle kernel reads natural layout -> writes blocked layout
```

Fused with index inversion:

```text
reduction computes scale in registers -> final store uses swizzled address
```

Important labels:

- `load side: simple`
- `store side: blocked/swizzled`
- `cvt.rp.satfinite.ue8m0x2.f32`
- `loop_index_inversion_in_fusion`

Suggested title: `MXFP8: Move the Swizzle to the Store`

Prompt for graphics agent:

```text
Create an Excalidraw diagram explaining MXFP8 scale swizzle and index inversion.
Left half: unfused two-kernel path. Kernel 1 computes block amax and E8M0 scale in natural layout, writes HBM. Kernel 2 reads natural layout and writes blocked/swizzled layout.
Right half: fused path. A single reduction kernel computes scale in registers and stores directly to the blocked/swizzled address.
Include a small shape transformation chain: [M,K] -> [M,K/32] -> [M/128,K/128,128,4] or similar blocked scale layout.
Label the PTX conversion `cvt.rp.satfinite.ue8m0x2.f32`.
```

## Diagram 4: Nested Reduction for RMSNorm + Quant

Purpose: show why nested reduction is central for low-precision pipelines.

Show an input row of `D=8192` values loaded once into a persistent reduction
tile. Inside one kernel:

1. Pass 1: `sum(x*x)` -> RMS statistic.
2. Normalize full-resolution tile.
3. Reshape register tile into groups of `G=16` or `G=128`.
4. Pass 2: block `amax`.
5. Scale and cast/pack.

Contrast with unfused path that reloads `x` in the second kernel.

Important labels:

- `FusedNestedReductions`
- `tl.reshape` in registers
- `x loaded once`
- `variance / amax never materialize`

Suggested title: `Nested Reduction: One Load, Two Reduction Granularities`

Prompt for graphics agent:

```text
Create an Excalidraw diagram for a fused RMSNorm + quantization kernel using nested reduction.
Show a row of D=8192 values loaded once. Inside a single kernel box, show pass 1 reducing across D for RMS, then normalized values staying in registers, then an in-register reshape into groups G=16 or G=128, then pass 2 block amax, then scale/cast/pack.
Add a small unfused comparison showing variance written to HBM and x reloaded.
Use labels `FusedNestedReductions`, `tl.reshape`, and `x loaded once`.
```

## Diagram 5: Inline ASM HOP as a Fusible Escape Hatch

Purpose: explain why inline asm belongs in the compiler graph, not behind a
custom op boundary.

Show two paths:

Custom op boundary:

```text
pointwise ops -> HBM -> custom CUDA/PTX op -> HBM -> pointwise ops
```

Inline ASM HOP:

```text
pointwise ops -> tl.inline_asm_elementwise -> pointwise ops
```

Use concrete instructions:

- MXFP8 scale conversion: `cvt.rp.satfinite.ue8m0x2.f32`
- NVFP4 pack: `cvt.rn.satfinite.e2m1x2.f32`

Important labels:

- `fusible elementwise op`
- `is_pure=True`
- `no kernel boundary`

Suggested title: `Inline ASM: Hardware Instructions Without Losing Fusion`

Prompt for graphics agent:

```text
Create an Excalidraw diagram comparing a custom-op boundary with Inline ASM HOP.
Custom-op path should show pointwise work writing HBM, a custom PTX/CUDA op, then another HBM write/read.
Inline ASM path should show one fused kernel with pointwise ops before and after `tl.inline_asm_elementwise`.
Include instruction labels `cvt.rp.satfinite.ue8m0x2.f32` and `cvt.rn.satfinite.e2m1x2.f32`.
Emphasize `fusible elementwise op`, `is_pure=True`, and `no kernel boundary`.
```

## Diagram 6: Quantization Is Layer- and Communication-Aware

Purpose: connect compiler mechanics to model-serving choices.

Draw a simplified transformer block with labels:

- first/last GEMM may stay higher precision.
- attention / MLP GEMMs use low precision.
- KV cache may use a separate quantization policy.
- activation quantization before collective reduces communicated bytes.

Use callouts for:

- eval quality tradeoffs.
- perf/memory tradeoffs.
- serving precision may differ per deployment.

Suggested title: `Quantization Is a Per-Layer System Decision`

Prompt for graphics agent:

```text
Create an Excalidraw diagram of a simplified transformer block annotated with quantization decisions.
Show first/last GEMM optionally excluded, attention/MLP GEMMs low precision, KV cache with separate quant policy, and quant before collective reducing communicated bytes.
Add callouts for eval quality, perf, memory, and serving precision.
Keep it technical and compact.
```

## Optional Final Overview Graphic

Purpose: one slide that ties Inductor investments together.

Draw a pipeline:

```text
Quant pattern in graph
  -> fusion analysis
  -> nested reductions
  -> index inversion / layout swizzle
  -> inline asm HOP
  -> generated low-precision kernel
```

Under each stage, include one concrete artifact:

- `FusedNestedReductions`
- `loop_index_inversion_in_fusion`
- `tl.inline_asm_elementwise`
- `float8_e8m0fnu / NVFP4 pack`

Suggested title: `Compiler Pieces Needed for Low Precision Kernels`

Prompt for graphics agent:

```text
Create an Excalidraw overview diagram showing the Inductor compiler pieces needed for low precision kernels.
Pipeline: quant pattern in graph -> fusion analysis -> nested reductions -> index inversion/layout swizzle -> inline asm HOP -> generated low-precision kernel.
Annotate with `FusedNestedReductions`, `loop_index_inversion_in_fusion`, `tl.inline_asm_elementwise`, and `float8_e8m0fnu / NVFP4 pack`.
Use the same color scheme as the other diagrams.
```
