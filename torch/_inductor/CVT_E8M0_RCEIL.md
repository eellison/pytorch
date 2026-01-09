# CVT E8M0 RCEIL - PTX Lowering for MX Format Scaling

## Original Issue

GitHub Issue: https://github.com/pytorch/pytorch/issues/170635

The goal is to enable custom PTX assembly instructions in Inductor lowerings, specifically for SM100+ (Blackwell) hardware optimizations. This PR implements support for the `cvt.rp.satfinite.ue8m0x2.f32` PTX instruction which converts float32 to e8m0 format with ceiling rounding.

## Background

### What is E8M0?

E8M0 is an 8-bit exponent-only format (no mantissa) used in MX (Microscaling) quantization formats. It stores just the biased exponent with bias=127, representing powers of 2.

For example:
- `1.0 = 2^0` → e8m0 biased = 127
- `2.0 = 2^1` → e8m0 biased = 128
- `0.5 = 2^-1` → e8m0 biased = 126

### Why This Matters

MX format quantization (used in FP8 training/inference) requires extracting scale values as e8m0. The naive implementation uses bit manipulation:

```python
inp_bits = inp.view(torch.int32)
biased_exp = (inp_bits >> 23) & 0xFF
mantissa = inp_bits & 0x7FFFFF
needs_round_up = mantissa != 0
e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
e8m0_biased = torch.clamp(e8m0_biased, 0, 255)
return e8m0_biased.to(torch.uint8)
```

On SM100+ (Blackwell), this can be replaced with a single PTX instruction:
```
cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;
```

## Implementation

### 1. Inductor Prim (`torch/_inductor/inductor_prims.py`)

Added `cvt_e8m0_rceil` prim with eager fallback implementation.

### 2. Lowering (`torch/_inductor/lowering.py`)

- **SM100+**: Uses `ops.inline_asm_elementwise` with PTX instruction
- **Older hardware**: Falls back to fusible bit manipulation ops

### 3. Pattern Matcher (`torch/_inductor/fx_passes/misc_patterns.py`)

Detects the bit manipulation pattern and replaces with `inductor_prims.cvt_e8m0_rceil`. Only enabled on SM100+ since that's when the optimization is beneficial.

## Usage

### Direct Prim Call
```python
from torch._inductor import inductor_prims

result = inductor_prims.cvt_e8m0_rceil(float32_tensor)
```

### Automatic Pattern Matching (SM100+ only)
```python
# This code will be automatically optimized on SM100+
inp_bits = inp.view(torch.int32)
biased_exp = (inp_bits >> 23) & 0xFF
mantissa = inp_bits & 0x7FFFFF
needs_round_up = mantissa != 0
e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
e8m0_biased = torch.clamp(e8m0_biased, 0, 255)
result = e8m0_biased.to(torch.uint8)
```

## Future Work

### TODO: Add `cvt.rn.bf16x2.ue8m0x2` support

This is the inverse operation - converting e8m0 scale values back to bf16 for dequantization:

```
cvt.rn.bf16x2.ue8m0x2 $0, $1;
```

This would complete the MX format scaling round-trip:
1. **Quantize**: `cvt.rp.satfinite.ue8m0x2.f32` (float32 → e8m0)
2. **Dequantize**: `cvt.rn.bf16x2.ue8m0x2` (e8m0 → bf16)

### Pattern for inverse operation

```python
# Pattern to match for e8m0 -> bf16 conversion
scale_fp = (scale_e8m0.to(torch.int32) << 23).view(torch.float32)
# or similar bit manipulation
```

## Related Work

- PyTorch AO PR: https://github.com/pytorch/ao/pull/3498 (uses `tl.inline_asm_elementwise` in Triton kernels)
- Test file: `test/inductor/test_cvt_e8m0_rceil.py`

## How to Add More PTX Instructions

This PR establishes the pattern for adding custom PTX lowerings:

1. **Define a prim** in `inductor_prims.py`:
   ```python
   my_prim = make_prim(
       "inductor_my_prim(Tensor input) -> Tensor",
       eager_implementation,
       doc="Description",
   )
   ```

2. **Register lowering** in `lowering.py`:
   ```python
   @register_lowering(inductor_prims.my_prim, type_promotion_kind=None)
   def my_prim_lowering(inp):
       fn = functools.partial(
           ops.inline_asm_elementwise,
           asm="my.ptx.instruction $0, $1;",
           constraints="=r,r",
           dtype=torch.float32,
           is_pure=True,
           pack=1,
       )
       return make_pointwise(fn)(inp)
   ```

3. **Optionally add pattern matcher** in `fx_passes/misc_patterns.py` to auto-detect and replace existing code patterns.
