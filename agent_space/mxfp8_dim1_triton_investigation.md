# MXFP8 Dim1 / Combined Dim0+Dim1 Perf Investigation

## Goal

Track work toward:

1. Getting a Triton MXFP8 dim1 cast kernel close to the TorchAO CUDA kernel.
2. Getting the decomposed `torch.compile` kernel close to that Triton/AO path.
3. Handling the more important combined dim0+dim1 MXFP8 cast case, where one input read should feed both outputs.

## Current Environment

- PyTorch repo: `/data/users/eellison/pytorch`
- TorchAO repo: `/data/users/eellison/ao`
- Triton repo: `/data/users/eellison/triton`
- GPU: NVIDIA B200
- PyTorch: `2.13.0a0+gitcc07682`
- Conda Triton used by PyTorch today: `3.7.0`
- Local Triton branch: `improve_bank_conflict`
- TorchAO CUDA extension is not usable in this environment:
  - `_C.abi3.so`, `_C_cutlass_90a.abi3.so`, `_C_cutlass_100a.abi3.so` fail to load.
  - `torchao::mxfp8_quantize` only has fake/meta registration, so CUDA calls fail.

## Local Artifacts

- Scratch benchmark: `agent_space/mxfp8_combined_probe.py`
- Compile code dump helper: `agent_space/mxfp8_dump_compile.py`
- Dim1 compile output code log: `agent_space/mxfp8_dim1_output_code.log`
- Relevant TorchAO Triton kernel: `/data/users/eellison/ao/torchao/prototype/mx_formats/kernels.py`
- Relevant TorchAO CUDA kernel:
  - `/data/users/eellison/ao/torchao/csrc/cuda/mx_kernels/mxfp8_extension.cpp`
  - `/data/users/eellison/ao/torchao/csrc/cuda/mx_kernels/mxfp8_quantize.cuh`
- Relevant Triton branch commit:
  - `/data/users/eellison/triton`
  - `b9d0e821e [Swizzle] Fix bank conflicts when tile has no bank-bit coverage`

## Current Numbers

Shape: `16384 x 16384`, input `bf16`, block size `32`, scaling mode `rceil`, B200.

| Case | Time | Notes |
| --- | ---: | --- |
| AO Triton dim1 | `344 us` | `torchao.prototype.mx_formats.kernels.triton_to_mxfp8_dim1` |
| `torch.compile` decomposed dim1 | `544-577 us` | two generated kernels, no nested reduction |
| AO Triton dim0 + dim1, separate kernels | `491 us` | two AO Triton kernels; reads input twice |
| `torch.compile` decomposed dim0+dim1 | `795 us` | three generated kernels, no nested reduction |

Older scratch result, not apples-to-apples because it targets transposed-input layout:

| Case | Time |
| --- | ---: |
| strided compile-style dim1 reduction | `~286 us` |
| 2D load + `tl.trans` optimized scratch kernel | `~209-213 us` |
| AO Triton dim1 in that older run | `~366 us` |

## TorchAO Triton Dim1 Path

`triton_to_mxfp8_dim1` expects contiguous row-major input and returns column-major FP8 output plus column scales.

Core structure:

1. Load a contiguous 2D tile with shape `[ROW_TILE_SIZE, COL_TILE_SIZE]`.
2. `tl.trans(x_block)`.
3. Reshape to `[COL_TILE_SIZE * (ROW_TILE_SIZE // 32), 32]`.
4. Reduce over the 32-row local group.
5. Quantize.
6. Transpose back and store output in column-major layout.
7. Store e8m0 scales in a column-major-ish scale layout.

The `tl.trans`/reshape path is exactly where the local Triton branch is relevant.

Autotune space today:

- `ROW_TILE_SIZE`: `128, 256, 512`
- `COL_TILE_SIZE`: `128, 256`
- `num_warps`: `4, 8`
- `num_stages`: `2, 3`

## Triton Branch Finding

Branch: `/data/users/eellison/triton`, `improve_bank_conflict`.

Commit message says:

> Fixes a degenerate case in mxfp8 kernels where lane bases like `[512, 1024, 2048, 4096, 1]` cause 32-way bank conflicts. The fix improves `to_mxfp8_dim1` from ~45% to ~60% of peak H100 bandwidth.

Code change:

- In `lib/Tools/GenericSwizzling.cpp`, after computing the shared-memory swizzle, it checks `bankConflictsLdSt`.
- If one side has no coverage of bank bits, it injects bank bits from the other side via XOR and retries the swizzle.
- Unit test added: `Test64x128Float8NoBankCoverage`.

Current blocker:

- `PYTHONPATH=/data/users/eellison/triton/python` imports the local Python package, but fails against the conda `_C.libtriton` ABI.
- Need a built/installed local Triton from this branch before measuring B200 perf.

## Decomposed `torch.compile` Dim1 Kernel

Source pattern:

```python
xt = x.t().contiguous()
scale, data = to_mx(xt, torch.float8_e4m3fn, 32, scaling_mode=RCEIL)
return data.t(), scale
```

Generated output code has two kernels.

Kernel 1: persistent reduction for amax and e8m0 scale.

Key load:

```python
tl.load(in_ptr0 + (y0 + 16384*r0_2 + 524288*x1), ...)
```

This is the transposed logical view expressed as a strided read from the original contiguous input. It reduces `r0_ = 32`, stores:

- bf16 amax temp: `(16384, 512)`
- e8m0 scale output: `(16384, 512)`

Kernel 2: pointwise quantization.

Key loads/stores:

```python
tl.load(in_ptr0 + (y0 + 16384*x3), ...)
tl.load(amax + (x2 + 512*y0), ...)
tl.store(out + (x3 + 16384*y0), ...)
```

This rereads the input, rereads the bf16 amax temp, recomputes the e8m0 conversion/scale inverse, and writes FP8 output. The e8m0 scale computed by kernel 1 is returned but not used by kernel 2.

Immediate compile-side perf issue:

- It is not one fused blockwise quantization kernel.
- It reads the large input twice.
- It writes and reads a large amax temporary.
- It recomputes scale conversion work in the quantization kernel.
- It does not use the AO Triton-style contiguous tile load + `tl.trans` path.

## Combined Dim0+Dim1 Status

TorchAO Triton has separate dim0 and dim1 kernels. There is no fused Triton combined path found.

TorchAO CUDA lower-level kernel already supports the combined concept:

- `mxfp8_quantize.cuh` has `USE_ROWWISE_SCALING = SCALE_DIM_X > 1`.
- `USE_COLWISE_SCALING = SCALE_DIM_Y > 1`.
- Launch table includes `scale_dim_x == 32 && scale_dim_y == 32`.
- One TMA global-to-shared input tile feeds both rowwise and colwise sections.
- Rowwise output and colwise output are both written via TMA shared-to-global.

But the public C++ wrapper blocks it:

```cpp
STD_TORCH_CHECK(!rowwise, "rowwise scaling is not supported yet");
```

So combined dim0+dim1 is present in the CUDA template but not exposed/validated through the current public op.

Current Inductor nested-reduction branch does not model the combined case:

- It requires a dependent outer reduction -> grouped reduction pair.
- It rejects grouped nodes containing multiple reductions.
- Combined dim0+dim1 is sibling grouped reductions from the same input tile, not a producer-consumer pair.

## Working Hypotheses

Dim1 Triton parity:

1. The biggest immediate Triton-specific lead is the `improve_bank_conflict` branch, because AO dim1 uses `tl.trans` on 8-bit tiles.
2. Need to rebuild/use the local Triton branch and rerun AO dim1.
3. Verify with profiler counters, especially shared-memory bank conflicts and effective memory throughput.

`torch.compile` dim1:

1. Decomposed compile is currently structurally worse than AO Triton because it generates two kernels and rereads input.
2. The right target is a single blockwise quantization kernel with local reduction plus full-resolution quantization epilogue.
3. For dim1, that kernel likely needs an AO-style 2D tile load + `tl.trans` lowering, not the current strided logical transpose lowering.
4. A smaller interim improvement could use the returned e8m0 scale in the quantization kernel instead of recomputing conversion from bf16 amax, but that does not solve the extra input read.

Combined dim0+dim1:

1. Best target is a single tiled kernel that reads input once and produces both rowwise and colwise outputs/scales.
2. TorchAO CUDA template is the blueprint.
3. Triton needs either a hand-written combined kernel or an Inductor representation for sibling block reductions sharing the same parent tile.
4. Existing nested-reduction support is not enough because it handles dependent reductions, not sibling reductions.

## Next Experiments

1. Build or otherwise activate `/data/users/eellison/triton` at `improve_bank_conflict`.
2. Rerun:

```bash
python /data/users/eellison/ao/benchmarks/mx_formats/cast_bench.py \
  --M 16384 --K 16384 --BLOCK_SIZE 32 --mode dim1_mxfp8_triton_rceil
```

3. Collect profiler counters before/after the Triton swizzle change:
   - shared-memory bank conflicts
   - global load/store throughput
   - instruction count
4. Dump TTIR/TTGIR/PTX for AO `to_mxfp8_dim1_kernel` before/after the swizzle branch.
5. Prototype a fused Triton dim0+dim1 kernel in `agent_space/`, initially copying the AO Triton dim0/dim1 math into one tile.
6. Compare fused Triton combined against two separate AO Triton kernels.
7. Decide integration path:
   - upstream Triton swizzle fix first;
   - then add an Inductor lowering/pattern for blockwise MXFP8 dim1;
   - separately design sibling grouped reductions or a custom op/HOP path for combined dim0+dim1.

## Commands Run

```bash
python /data/users/eellison/ao/benchmarks/mx_formats/cast_bench.py \
  --M 16384 --K 16384 --BLOCK_SIZE 32 --mode dim1_mxfp8_triton_rceil

TORCHINDUCTOR_NESTED_REDUCTION=1 python /data/users/eellison/ao/benchmarks/mx_formats/cast_bench.py \
  --M 16384 --K 16384 --BLOCK_SIZE 32 --mode dim1_mxfp8_rceil

TORCHINDUCTOR_NESTED_REDUCTION=1 python agent_space/mxfp8_combined_probe.py \
  --m 16384 --k 16384

TORCHINDUCTOR_NESTED_REDUCTION=1 TORCH_LOGS=output_code TORCHINDUCTOR_COMPILE_THREADS=1 \
  python agent_space/mxfp8_dump_compile.py --m 16384 --k 16384 --case dim1 \
  > agent_space/mxfp8_dim1_output_code.log 2>&1
```
