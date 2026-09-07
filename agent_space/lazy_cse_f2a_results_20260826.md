# F2a narrow group-width cast results

Date: 2026-08-26

## Snapshot

- Canonical worktree:
  `/data/users/eellison/pytorch/agent_space/followup_lazy_projection_wt`
- HEAD: `6b8ef64bd373b221c6ed1ba4f4b2f1edef2643c3`
- Production diff: `torch/_inductor/codegen/simd.py`, `+71/-4`
  (net `+67`)
- Production diff SHA-256:
  `a165db8f8e54db06476b9a2828e0df4aed41afd661364e2fc4a2012b81dcc3ab`
- Full unstaged diff SHA-256:
  `7ab016dc72657354c894e0252d74bc3d0bd8a9c5f4ab35423bf36785236e82b0`
- Full tree diff from HEAD SHA-256:
  `e5ecf48c90bdea1a68b9c54d29b5f69f6db2ea69b78a2da5bbd9c07883b8c0c8`

The staged PR+F1 baseline is unchanged. All F2a code and tests remain unstaged.

## Final rule

F2a now has one explicit algebraic exception:

```text
unguarded, lane-free, exact group-width source
    -> to_dtype at group width
    -> immediate materialization to child width
```

`_PointwiseRemapHandler.to_dtype` implements that rule with the full typed
OpsHandler signature. Generic `_default` recursively widens all group-width
operands before delegation, and `store` widens its value before remapping and
recording it.

The resolver permits a raw group-width source only when its exact planned
relation is lane-free, its consumer has no `ops.masked` guard, and its ordinary
`CSEVariable.shape` is group-width rather than direct child-width. The existing
sub-parent materializer performs the broadcast and assigns masks from the
target shape.

There is no operation allowlist, shape-propagation preflight, masked callback
wrapper, new value type, domain enum, or cache. Compared with the broader
prototype, this reduces production from net `+115` to net `+67` and scheduler
test additions from `+163` to `+76`.

## Verification

- Focused exact-access/direct-width/cast-policy tests: `6 passed`.
- NVFP4/MXFP4/MXFP6 kernel-form tests: `8 passed`.
- Full scheduler file: `126 passed, 6 skipped`.
- Full nested-reduction file: `389 passed, 8 skipped`.
- `spin quicklint`: clean.
- `py_compile`: clean.
- `git diff --check`: clean.
- Independent adversarial review: approve, no semantic blocker.

The canonical files byte-match the fully tested narrow prototype for:

- `torch/_inductor/codegen/simd.py`
- `test/inductor/test_inductor_scheduler.py`
- `test/inductor/test_nested_reduction.py`

The only explicit scope limitation is intentional: masked-source deferral and
multi-operation group-width chains remain eager. They can be added later only
with a concrete source-form and performance justification.

## Generated source

- All 12 NVFP4 shape/mode wrappers are exactly equal to the broader F2a
  implementation after normalizing temporary wrapper metadata.
- All 12 MXFP4 wrappers are also exactly equal to the broader F2a form.
- Ten protected factor-2, MXFP6, internal-source, reduced-broadcast, swizzle,
  preshuffle, and DCN sources are exactly equal to F1.
- Every capture remains one staged kernel.

For NVFP4, F1 emitted an FP8-to-uint8 bitcast, broadcast, and uint8-to-FP8
bitcast before converting to FP32. F2a converts the group-width FP8 value to
FP32 first and then broadcasts that FP32 value. The FP8 conversion itself is
still emitted exactly once. MXFP4 generated kernel code is unchanged.

Protected-source hashes:

- `agent_space/f2a_source_compare/f1.txt`
- `agent_space/f2a_narrow_source_compare/results.txt`

## Performance protocol

- Device: NVIDIA B200, physical GPU 1.
- F1 and final narrow F2a wrappers were generated with identical
  `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`, FX graph cache off, coordinate descent
  off, and multi-kernel off settings.
- Each matching pair ran in a fresh process with shared inputs.
- Timing used CUDA graphs, 20 rotating-order rounds, and 100 replays per round.
- Every row produced one staged kernel and one launcher.
- Every F1/F2a pair had identical launch configuration and exact outputs.
- No row regressed by more than 2%, so no noise-triggered rerun was required.

Launch configurations:

- default: `XBLOCK=1`, `R0_BLOCK=1024`, 8 warps, 1 stage
- looped: `XBLOCK=8`, `R0_BLOCK=1024`, 4 warps, 1 stage
- persistent: `XBLOCK=8`, 4 warps, 1 stage

Capture script SHA-256:
`fd90d1e88f706aeb792f80cdf79875f74f9105dde2a38692ee1fd242c4c1d48d`

Replay script SHA-256:
`5493be195f7a4c2014c67efd3fa78f74ffdeddbff3f3adecfc59bf80bf510592`

Raw artifacts:

- F1 wrappers: `agent_space/f2a_perf_matched_20260826/f1/`
- Final F2a wrappers: `agent_space/f2a_narrow_perf_20260826/generated/`
- Final paired results: `agent_space/f2a_narrow_perf_20260826/paired_final/`

## NVFP4 paired results

| Shape | Mode | F1 us | F2a us | Delta | F1 regs/spills | F2a regs/spills |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128x4096 | default | 4.129 | 4.129 | -0.00% | 30/0 | 28/0 |
| 128x4096 | looped | 10.275 | 10.275 | +0.00% | 128/0 | 122/0 |
| 128x4096 | persistent | 14.391 | 10.285 | -28.53% | 255/314 | 255/184 |
| 4096x4096 | default | 20.498 | 16.428 | -19.85% | 30/0 | 28/0 |
| 4096x4096 | looped | 20.529 | 18.467 | -10.04% | 128/0 | 122/0 |
| 4096x4096 | persistent | 44.312 | 28.728 | -35.17% | 255/314 | 255/184 |
| 4096x4608 | default | 28.706 | 24.622 | -14.23% | 30/0 | 30/0 |
| 4096x4608 | looped | 27.252 | 24.615 | -9.68% | 128/0 | 126/0 |
| 4096x4608 | persistent | 511.999 | 173.544 | -66.10% | 32/1404 | 168/724 |
| 4096x8192 | default | 40.954 | 28.759 | -29.78% | 30/0 | 28/0 |
| 4096x8192 | looped | 38.988 | 34.851 | -10.61% | 128/0 | 122/0 |
| 4096x8192 | persistent | 611.312 | 472.213 | -22.75% | 32/1920 | 32/1602 |

NVFP4 geometric-mean speedup across the 12 rows is `1.301x` (`23.13%`
lower median time).

## MXFP4 paired results

| Shape | Mode | F1 us | F2a us | Delta | F1 regs/spills | F2a regs/spills |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128x4096 | default | 4.130 | 4.130 | -0.01% | 26/0 | 26/0 |
| 128x4096 | looped | 8.288 | 8.277 | -0.14% | 127/0 | 127/0 |
| 128x4096 | persistent | 10.275 | 10.275 | -0.00% | 255/136 | 255/136 |
| 4096x4096 | default | 16.429 | 16.429 | +0.00% | 26/0 | 26/0 |
| 4096x4096 | looped | 18.451 | 18.453 | +0.01% | 127/0 | 127/0 |
| 4096x4096 | persistent | 26.918 | 26.907 | -0.04% | 255/136 | 255/136 |
| 4096x4608 | default | 26.637 | 26.638 | +0.00% | 28/0 | 28/0 |
| 4096x4608 | looped | 24.626 | 24.622 | -0.01% | 130/0 | 130/0 |
| 4096x4608 | persistent | 53.336 | 53.332 | -0.01% | 255/354 | 255/354 |
| 4096x8192 | default | 29.026 | 28.765 | -0.90% | 30/0 | 30/0 |
| 4096x8192 | looped | 32.810 | 32.805 | -0.02% | 127/0 | 127/0 |
| 4096x8192 | persistent | 455.508 | 455.638 | +0.03% | 32/1360 | 32/1360 |

MXFP4 is neutral: geometric-mean speedup `1.001x` (`0.09%` lower median
time), with identical register and spill counts in every row.

## Conclusion

The final F2a removes the duplicated NVFP4 FP8 broadcast round trip with a
single explicit cast-before-widening rule. It preserves exact output bytes,
fusion, launch configuration, protected source forms, and the full measured
performance gain while keeping the implementation and tests narrowly scoped.
