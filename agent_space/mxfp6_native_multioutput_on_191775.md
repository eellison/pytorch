# Native MXFP6 inline asm on the reviewed `(4,3)` planner

Date: 2026-08-26

This was tested in the scratch worktree
`/data/users/eellison/pytorch/agent_space/mxfp6_native_multioutput_survey`.
It is based on reviewed commit `6b8ef64bd37` plus the current reviewed
worktree changes and the production-code changes from `0adc27a2563`. No
reviewed worktree was modified and no commit was created.

## Conclusion

The native three-output E2M3 pack can be one kernel with the current layered
planner. It does not need a scheduler exception. The minimum graph change is to
give the AITER field permutation a full-resolution scheduler value before the
quarter-resolution inline asm:

```python
fields = torch.stack((scaled[..., :16], scaled[..., 16:]), dim=-1)
fields = fields.reshape(rows, width // 32, 32)
fields = torch.ops._inductor_test.realize(fields)
groups = fields.reshape(rows, width // 32, 8, 4)
low, middle, high = inline_asm_elementwise(
    groups[..., 0], groups[..., 1], groups[..., 2], groups[..., 3],
    asm_str=E2M3X4_PACK_ASM,
    constraints="=r,=r,=r,f,f,f,f",
    dtype=(torch.int32, torch.int32, torch.int32),
)
packed = torch.stack((low, middle, high), dim=-1).to(torch.uint8)
```

The realization is not a global-memory materialization after fusion. Generated
code has one nested Triton kernel and exactly four stores: one scale store and
three packed-byte stores.

`_inductor_test.realize` is only a diagnostic stand-in. A production direct
helper would need its lowering/decomposition to preserve this full-resolution
field-order value as a planner-visible boundary. The `pack=2` form obtains the
same useful boundary naturally from its full-resolution HOP output.

The simpler and slightly faster form is still a single full-resolution
`pack=2` E2M3 conversion followed by the reviewed no-realize `(4,3)` pack. That
form emits one inline-asm call site and already works on the reviewed tree
without `0adc27a2563`. Therefore the tuple-return feature is not required for
this AITER graph.

## Results

All packed bytes and scales were bitwise equal to AITER. Times are B200 CUDA
Graph medians in microseconds.

| Shape | AITER | Software `(4,3)` | Full-res `pack=2` | Direct 3-output + field realize |
|---|---:|---:|---:|---:|
| 8192x128 | 4.711 | 3.768 | 2.498 | 2.508 |
| 32768x128 | 12.892 | 10.792 | 5.386 | 5.529 |
| 1024x1024 | 4.382 | 3.768 | 2.498 | 2.518 |
| 128x12288 | 5.899 | 5.161 | 3.031 | 3.094 |

At `8192x128`:

| Form | Kernels | Nested | Triton asm call sites | Stores | Time |
|---|---:|---:|---:|---:|---:|
| Direct 3-output, no field realization | 3 | 0 | n/a | n/a | 5.65 us |
| Direct 3-output, realized fields | 1 | 1 | 9 | 4 | 2.51 us |
| Reduced pair outputs | 3 | 0 | n/a | n/a | 6.43 us |
| Full-resolution `pack=2` | 1 | 1 | 1 | 4 | 2.50 us |
| Single packed-u32 asm prototype | 1 | 1 | 1 | 4 | 2.58 us |

Adding the three output realizations used by the `0adc27a2563` test helper to
the successful direct form remains one kernel at 2.52 us. It does not reduce
the nine generated asm call sites.

The direct form also matched AITER exactly for all-zero, all-negative-zero, and
wide finite FP16 inputs. The full-resolution `pack=2` conversion matched all
63,488 finite FP16 bit patterns; the expected signed-zero difference from the
old Python converter was accounted for, and full quantization retained AITER's
signed-zero behavior.

## Exact planner diagnosis

For the planner, the outer grouped reduction has logical domain
`(parent_x, parent_r)` with `parent_rnumel=32`; its input access normalizes to:

```text
32 * parent_x + parent_r
```

### Reduced pair-output form

Each tuple result is correctly classified as rate `(2,1)`. For child index
`j in [0,16)`, however, the two AITER-ordered inputs normalize to:

```text
32 * child_x + j
32 * child_x + j + 16
```

The supported factor-2 projection requires each access to equal:

```text
32 * child_x + 2 * child_r + lane, lane in {0,1}
```

`Mod(j, 2)` is not constant, so
`_try_get_sub_parent_source_projections` rejects the reduction-to-pair fusion.
This is a contiguous-halves projection, not the interleaved projection that the
current code generator implements. Reconstructing a full-resolution tensor
after the two reduced outputs also creates a downstream consumer of
sub-parent-resolution values, which is not the planner's supported staging
direction.

### Direct three-output form

The three outputs are correctly classified and grouped as three rate `(4,1)`
nodes. For quartet index `j in [0,8)`, their four raw source reads normalize to:

```text
32 * child_x + 2 * j
32 * child_x + 2 * j + 16
32 * child_x + 2 * j + 1
32 * child_x + 2 * j + 17
```

The factor-4 source proof requires `32*child_x + 4*j + lane`. The lane of
`2*j` modulo four depends on `j`, so the proof correctly returns `None`.
This is the AITER cross-half field order, not an unsupported output rate.

After realizing the full-resolution field-order value, it writes a logical
row-major buffer at `32*parent_x + parent_r`. The direct pack reads that value
at `32*child_x + 4*j + {0,1,2,3}`. The existing factor-4 projection succeeds,
and the graph becomes one nested kernel.

Thus the non-fusion is primarily the wrong graph representation: the
quarter-resolution HOP erased the full-resolution AITER field-order boundary
that the planner needs. A new contiguous/cross-half projection would broaden
the compiler, but it is unnecessary for this graph.

## Multi-output codegen caveat

`0adc27a2563` represents each tuple result as a separate Pointwise and relies on
kernel CSE to share the asm invocation. In the `(4,3)` staged output, the stack
is specialized into three stores, and each store evaluates all three tuple
choices. The generated Triton currently contains nine syntactically identical
three-result asm calls. This is the coupled-output IR limitation already noted
by the commit's TODO; it is separate from fusion legality.

The full-resolution `pack=2` form avoids that issue. It exposes one logical
full-resolution tensor, emits one `tl.inline_asm_elementwise`, then lets the
existing `(4,3)` epilogue emit the three byte stores. It was also verified on
the unmodified reviewed worktree with identical one-kernel behavior and timing,
so the tuple-output commit itself is not needed for the recommended graph.

## Commands and artifacts

```bash
source agent_space/env.sh
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 "$PY" \
  agent_space/run_mxfp6_native_multioutput.py \
  agent_space/bench_aiter_vs_inductor_grouped_native_subagent.py

MXFP6_NATIVE_VARIANT=postpermute_realize_fields \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 "$PY" \
  agent_space/run_mxfp6_native_multioutput.py \
  agent_space/bench_mxfp6_native_multioutput_variants.py

MXFP6_PLAN_VARIANT=direct TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 "$PY" \
  agent_space/run_mxfp6_native_multioutput.py \
  agent_space/diagnose_mxfp6_native_multioutput_plan.py
```

Relevant scratch files:

- `agent_space/bench_aiter_vs_inductor_grouped_native_subagent.py`
- `agent_space/bench_mxfp6_native_multioutput_variants.py`
- `agent_space/bench_mxfp6_native_pair_output_variants.py`
- `agent_space/diagnose_mxfp6_native_multioutput_plan.py`
- `agent_space/capture_mxfp6_native_forms.py`
- `agent_space/check_mxfp6_native_direct_edges.py`
- `agent_space/mxfp6_plan_pair.log`
- `agent_space/mxfp6_plan_direct.log`
- `agent_space/mxfp6_plan_direct_realize_fields.log`
- `agent_space/mxfp6_plan_pack2.log`

No production file in the reviewed worktree was edited.
