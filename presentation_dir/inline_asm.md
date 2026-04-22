# Inline ASM Integration

**Op:** `torch._higher_order_ops.inline_asm_elementwise.inline_asm_elementwise`
(FX: `torch.ops.higher_order.inline_asm_elementwise`)
**Test:** `test/higher_order_ops/test_inline_asm_elementwise.py`
**Real-world use:** `agent_space/test_reindex_deps.py` — E8M0 exponent encoding
(part of the MX-fp8 scale computation)
**Repro:** `agent_space/capture_inline_asm.py` (log in `inline_asm.log`)

---

## The idea

`inline_asm_elementwise` is a higher-order op that lets user code drop a
snippet of PTX (NVIDIA) or AMDGCN (AMD) straight into a compiled graph.
From Inductor's perspective it's just another elementwise op: it shows up
in the loop body via `ops.inline_asm_elementwise(...)`, flows through the
scheduler like `ops.add` or `ops.mul`, and at codegen becomes a
`tl.inline_asm_elementwise(...)` call **inside** the Triton kernel body.

That means inline asm **fuses** with surrounding pointwise ops — no
separate kernel, no round-trip through HBM.

## Minimal example

```python
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise

@torch.compile(fullgraph=True)
def f(x, y):
    z = x * 2
    w = inline_asm_elementwise(
        z, y,
        asm_str="add.f32 $0, $1, $2;",   # PTX — AMD uses v_add_f32 ...
        constraints="=f,f,f",
        dtype=torch.float32,
    )
    return w + 1.0
```

Three operations: aten mul, inline-asm add, aten add.

## Pre-fusion IR

```
op0: SchedulerNode(ComputedBuffer)
  op0.group.iteration = (128, 1)
  var_ranges = {p0: 128}
  index0 = p0

  body:
    load   = ops.load('arg0_1', p0)
    mul    = ops.mul(load, 2.0)
    load_1 = ops.load('arg1_1', p0)
    asm    = ops.inline_asm_elementwise(mul, load_1,
                                        asm='add.f32 $0, $1, $2;',
                                        constraints='=f,f,f',
                                        dtype=torch.float32,
                                        is_pure=True, pack=1)
    add    = ops.add(asm, 1.0)
    store  = ops.store('buf0', p0, add, None)
```

The inline-asm op lives in the same loop body as the mul and add — it's a
first-class node in Inductor's ops IR. Inductor knows its inputs, its
output dtype, and that it's `is_pure=True` (no side effects, no ordering
constraints beyond data deps).

## Post-fusion IR

Same single SchedulerNode — the whole chain was already a single
elementwise loop, nothing to fuse or rewrite. (In more interesting cases
the inline-asm op can sit **between** ops that would have been separate
kernels, and the normal fusion logic pulls them together around it.)

## Output code (1 fused kernel)

```python
@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)         # x
    tmp3 = tl.load(in_ptr1 + (x0), xmask)         # y
    tmp1 = tl.full([1], 2.0, tl.float32)
    tmp2 = tmp0 * tmp1                            # x * 2
    tmp4 = tl.inline_asm_elementwise(             # ← inline PTX inlined here
        'add.f32 $0, $1, $2;',
        '=f,f,f',
        [tmp2, tmp3],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    tmp5 = tl.full([1], 1.0, tl.float32)
    tmp6 = tmp4 + tmp5                            # + 1.0
    tl.store(out_ptr0 + (x0), tmp6, xmask)
```

One kernel, one load of each input, one store. The asm fragment sits
between the mul and the add, exactly where the Python source put it.

## Real-world use: E8M0 exponent encoding for MX-fp8

In `agent_space/test_reindex_deps.py`, the MX scaling preamble uses
inline asm to do the f32 → e8m0 conversion in one PTX instruction:

```python
ias = torch.ops.higher_order.inline_asm_elementwise(
    div,
    asm_str='cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;',
    constraints='=h,r',
    dtype=torch.uint16,
    is_pure=True,
    pack=1,
)
```

This `cvt.rp.satfinite.ue8m0x2.f32` is a PTX instruction available only
on sm_100+ — one PTX opcode replaces a ~10-op fallback
(bit-shift, mask, sub, clamp, isnan, where, reencode). Because
`inline_asm_elementwise` is just another op in Inductor's IR, the whole
MX preamble still fuses into one kernel — see the
[index inversion writeup](./index_inversion.md) for that fusion.

## What it enables

1. **Codegen escape hatch:** access to intrinsics Triton doesn't expose
   yet (new PTX instructions, hardware-specific conversions, fast paths).
2. **Stays fusible:** because it lands in the ops IR, the fusion machinery
   treats it like any other elementwise. No kernel-boundary tax.
3. **Per-backend dispatch:** callers pick PTX vs AMDGCN asm strings
   themselves. Inductor just passes the string through to Triton.

## Narrative for slides

- **Before (without inline_asm_elementwise):** to access a one-op PTX
  intrinsic, users either (a) wrote a custom C++ op (kernel boundary +
  HBM round-trip) or (b) emulated the instruction in portable code
  (many more ops, slower).
- **After:** drop a PTX snippet into the compiled graph, Inductor fuses
  around it. Zero overhead, full fusion.
- **Canonical use cases:** E8M0 exponent pack/unpack (MX-fp8), NVFP4
  encoding, `redux` / warp-reduction intrinsics, fast approximations
  (`ex2.approx`, `rcp.approx`).
- **Caveats:** `is_pure=True` required for the scheduler to reorder/fuse
  safely. Caller owns correctness of the asm string across backends.

## Runnable repro

```python
import torch
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config as inductor_config

torch._logging.set_logs(
    ir_pre_fusion=True,
    ir_post_fusion=True,
    output_code=True,
)

@torch.compile(fullgraph=True)
def f(x, y):
    z = x * 2
    w = inline_asm_elementwise(
        z, y,
        asm_str="add.f32 $0, $1, $2;",   # use v_add_f32 ... on AMD
        constraints="=f,f,f",
        dtype=torch.float32,
    )
    return w + 1.0

x = torch.randn(128, device="cuda", dtype=torch.float32)
y = torch.randn(128, device="cuda", dtype=torch.float32)

with inductor_config.patch({"triton.unique_kernel_names": True}):
    f(x, y)
    torch.cuda.synchronize()
```

```bash
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python repro.py 2> out.log
```
