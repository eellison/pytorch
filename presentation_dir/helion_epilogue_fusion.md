# Helion (External Template) Epilogue Fusion

**Motivating test:** `test/inductor/test_select_algorithm.py::TestTemplateRender::test_external_template_prologue_epilogue_fusion`
**Helion PR:** https://github.com/pytorch/helion/pull/1324
**Related issue:** https://github.com/pytorch/helion/issues/1346
**Config:** `torch._inductor.config.epilogue_fusion` (default `True`)
**Repro:** `agent_space/capture_template_epilogue.py` (logs in `epilogue_{off,on}.log`)

---

## The idea

Inductor's matmul / attention / custom *templates* produce outputs that
often feed directly into a pointwise chain (bias, activation, residual,
etc). Without epilogue fusion, that chain runs as a separate kernel —
meaning the matmul output round-trips through HBM.

Inductor's template-render path exposes placeholder hooks (`<STORE_OUTPUT_0>`,
`<LOAD_INPUT_B>`, etc) that let Inductor rewrite **into the template**:
- **Prologue fusion:** inline a load-side transformation (e.g. `sigmoid(B)`)
  at the template's B-load site.
- **Epilogue fusion:** inline the pointwise chain at the template's output
  store site — so the matmul accumulator goes straight through
  bias + activation + etc. before it ever leaves the kernel.

**Why Helion matters here:** Helion ([helion repo](https://github.com/pytorch/helion))
is a high-level Python-embedded DSL for writing Triton kernels. Helion
kernels compile to Triton templates, and through Inductor's
`ExternalTritonTemplateKernel` hook they can participate in the same
prologue/epilogue fusion as Inductor's own templates. Helion PR #1324 is
what hooks the `<STORE_OUTPUT_0>` / `<LOAD_INPUT_B>` machinery up.

## Minimal example

```python
def f(a, b, bias):
    return torch.relu(a @ torch.sigmoid(b)) * bias
```

- `sigmoid(b)`: prologue candidate (rewrites the B load).
- `relu(...) * bias`: epilogue candidate (rewrites the output store).

## Pre-fusion IR (identical for both configs)

Shapes `M=K=N=1024`, bf16.

```
op0: SchedulerNode                        # sigmoid(b) → buf0
op1: SchedulerNode(MultiTemplateBuffer)   # mm(a, buf0) → buf1   ← template
op2: SchedulerNode                        # relu(buf1) * bias → buf2
```

`op1` is a `MultiTemplateBuffer` — Inductor's wrapper around a Triton
template with render hooks. Its output `buf1` is consumed by `op2`.

## Post-fusion — epilogue_fusion OFF

```
op0: SchedulerNode                        # sigmoid
op1: SchedulerNode(MultiTemplateBuffer)   # mm   ← stays alone
op2: SchedulerNode                        # relu * bias   ← stays alone
```

**3 kernels:**

```python
triton_poi_fused_sigmoid_0.run(arg0_1, buf0, 1048576, ...)             # sigmoid
triton_tem_fused_mm_sigmoid_1.run(arg1_1, buf0, buf1, 128, 1, 1, ...)  # mm template
triton_poi_fused_mul_relu_2.run(buf2, arg2_1, 1048576, ...)            # relu * bias
```

Note `buf1` is materialized out of the matmul template, then re-read by
kernel 3.

## Post-fusion — epilogue_fusion ON

```
op0: SchedulerNode                         # sigmoid (prologue in this config is still separate)
op1_op2: FusedSchedulerNode(               ← template + epilogue fused
    SchedulerNode(MultiTemplateBuffer),    # mm template
    SchedulerNode,                         # relu * bias (baked into template)
)
```

**2 kernels:**

```python
triton_poi_fused_sigmoid_0.run(arg0_1, buf0, 1048576, ...)                                 # sigmoid
triton_tem_fused_mm_mul_relu_sigmoid_1.run(arg1_1, buf0, arg2_1, buf2, 128, 1, 1, ...)     # mm + epilogue
```

Notice the template kernel name now includes `mul_relu` and it takes
`arg2_1` (bias) as an extra input — the scheduler rewrote the template's
`<STORE_OUTPUT_0>` hook to inline the `relu(acc) * bias` pointwise chain.

## Output code — OFF (matmul template, then separate epilogue)

Matmul template's store path:
```python
# kernel 2 tail: plain output
tl.store(out_ptr0 + (idx_n + 1024*idx_m), acc.to(tl.bfloat16), mask)
```

Separate epilogue kernel:
```python
@triton.jit
def triton_poi_fused_mul_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)   # re-read mm output
    tmp3 = tl.load(in_ptr0 + (x0), None)                       # bias
    tmp2 = triton_helpers.maximum(0, tmp0)                     # relu
    tmp4 = tmp2 * tmp3                                         # * bias
    tl.store(in_out_ptr0 + (x2), tmp4, None)                   # write back
```

## Output code — ON (template with inlined epilogue)

Matmul template's store path (same kernel):
```python
def triton_tem_fused_mm_mul_relu_sigmoid_1(arg_A, arg_B, in_ptr2, out_ptr1):
    # ... matmul loop producing `acc` ...
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A + ...)
        b = tl.load(B + ...)
        acc += tl.dot(a, b, ...)

    # Inductor's <STORE_OUTPUT_0> hook was rendered with the epilogue:
    tmp2 = tl.load(in_ptr2 + (idx_n_broadcast), mask).to(tl.float32)   # bias
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = triton_helpers.maximum(tmp0, acc)                           # relu(acc)
    tmp3 = tmp1 * tmp2                                                  # * bias
    tl.store(out_ptr1 + (idx_n + 1024*idx_m), tmp3, mask)               # final store
```

The accumulator never leaves the matmul kernel. The `bias` load happens
inside the template's store tile, then `relu * bias` runs in registers
before a single store.

## What the transformation did

1. Inductor's scheduler sees a `MultiTemplateBuffer` consumed by a
   pointwise chain. Fusion check asks: *can this consumer be rendered into
   the template's output hook?*
2. If yes, the consumer's `loop_body` is serialized to a Triton snippet
   and wired into the template's `<STORE_OUTPUT_0>` placeholder, with any
   extra reads (like `bias`) registered as `_extra_inputs` on the kernel.
3. Render time: the placeholder is substituted → one Triton kernel with
   matmul + epilogue baked in.
4. Prologue side (`<LOAD_INPUT_B>`) uses the same machinery but at the
   template's load site. Only inputs listed in `allowed_prologue_inps` can
   be fused.

## What Helion adds

Helion contributes an `ExternalTritonTemplateKernel` subclass (per test
`test_external_template_prologue_epilogue_fusion`) that:

- Exposes its own source template with `<LOAD_INPUT_B>` / `<STORE_OUTPUT_0>`
  placeholders,
- Implements `_render()` to consume Inductor's fusion-hook state
  (`_prologue_source_buffers`, `_extra_store_targets`, `_extra_inputs`),
- Sets `allowed_prologue_inps` and `epilogue_fusable_outputs`.

The result: a Helion-authored kernel gets Inductor prologue/epilogue
fusion "for free", without Inductor having to know the kernel body.

## Narrative for slides

- **Before (3 kernels):** sigmoid + mm + epilogue. MM output materialized
  to HBM, re-read by the epilogue. ~8MB round-trip for 1024² bf16.
- **After (2 kernels):** mm + epilogue fused. MM accumulator → bias load
  (cached) → `relu * bias` → single store. HBM round-trip avoided.
- **Bigger picture:** this is the same mechanism Inductor uses for its own
  matmul / FlexAttention templates. Helion's contribution is plumbing
  *external* kernels into the same fusion hooks — so user-authored
  kernels compose with Inductor graph-level fusion.
- **Follow-ups:** issue [#1346](https://github.com/pytorch/helion/issues/1346)
  tracks the next step (broader support for prologue-side fusion patterns).

## Runnable repro

```python
# Toggle EPILOGUE between False and True to see both cases.
import torch
from torch._inductor import config as inductor_config

EPILOGUE = True

torch._logging.set_logs(
    ir_pre_fusion=True,
    ir_post_fusion=True,
    output_code=True,
)

@torch.compile(fullgraph=True)
def f(a, b, bias):
    return torch.relu(a @ torch.sigmoid(b)) * bias

M, K, N = 1024, 1024, 1024
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)

with inductor_config.patch({
    "max_autotune_gemm": True,
    "max_autotune_gemm_backends": "TRITON",   # force Triton template
    "epilogue_fusion": EPILOGUE,
    "triton.unique_kernel_names": True,
}):
    out = f(a, b, bias)
    torch.cuda.synchronize()
```

```bash
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python repro.py 2> out.log
```

For the Helion-specific wire-up, run:
```bash
pytest test/inductor/test_select_algorithm.py \
  -k test_external_template_prologue_epilogue_fusion -v
```
and read the `_MOCK_ADD_KERNEL_TEMPLATE` / `_render()` in that test — it's
the reference pattern Helion follows.
