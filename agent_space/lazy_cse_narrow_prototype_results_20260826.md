# F2a narrow cast-only prototype

Date: 2026-08-26

## Snapshot

- Worktree:
  `/data/users/eellison/pytorch/agent_space/f2a_narrow_prototype_wt`
- Production diff: `torch/_inductor/codegen/simd.py`, `+71/-4`
  (net `+67`)
- Production diff SHA-256:
  `a165db8f8e54db06476b9a2828e0df4aed41afd661364e2fc4a2012b81dcc3ab`
- Full unstaged diff SHA-256:
  `7ab016dc72657354c894e0252d74bc3d0bd8a9c5f4ab35423bf36785236e82b0`

This is a separate prototype. The canonical F2a worktree was not modified.

## Rule

The prototype has one algebraic exception:

```text
unguarded, lane-free, exact group-width source
    -> to_dtype at group width
    -> immediate materialization to child width
```

`_PointwiseRemapHandler.to_dtype` exposes that exception with the full typed
OpsHandler signature. Generic `_default` recursively materializes every
group-width operand before delegation. `store` also materializes first.

The resolver permits a raw group-width source only when the planned relation is
lane-free and `consumer_guard.mask is None`. It does not defer masked sources.
The general operation allowlist, shape-propagation preflight, pure pack-1 asm
exception, and masked callback wrapper are deleted.

## Size and complexity

Compared with the general F2a implementation:

| Metric | General F2a | Narrow prototype |
| --- | ---: | ---: |
| Production net LOC | +115 | +67 |
| Scheduler-test additions | +163 | +76 |
| Added aggregate cyclomatic complexity | +26 | +19 |
| `_default` complexity | 10 | 2 |

The narrow form adds no type, wrapper, enum, or cache.

## Verification

- Focused exact-access/direct-width/cast-policy tests: `6 passed`.
- Kernel-form tests: `8 passed`.
- Full scheduler file: `126 passed, 6 skipped`.
- Full nested-reduction file: `389 passed, 8 skipped`.
- `spin quicklint`, `py_compile`, and `git diff --check`: clean.
- Independent adversarial review: approve, no semantic blocker.

Post-refactor generated-source checks:

- All 12 NVFP4 shape/mode wrappers are exactly equal to the general F2a
  wrappers after removing the temporary kernel-path line.
- The ten protected factor-2, MXFP6, internal-source, reduced-broadcast,
  swizzle, preshuffle, and DCN sources are exactly equal to F1.
- Every captured target remains one staged kernel.

## Representative paired performance

Fresh F1/narrow wrapper pairs used the same matched protocol as the full F2a
matrix: exact outputs, one launcher, identical launch configuration, 20
rotating rounds, and 100 CUDA-graph replays per round on an NVIDIA B200.

| Case | F1 us | Narrow us | Delta | F1 regs/spills | Narrow regs/spills |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVFP4 4096x4096 default | 20.503 | 16.428 | -19.87% | 30/0 | 28/0 |
| NVFP4 4096x4096 looped | 20.525 | 18.468 | -10.02% | 128/0 | 122/0 |
| NVFP4 4096x4608 persistent | 512.276 | 173.698 | -66.09% | 32/1404 | 168/724 |
| NVFP4 4096x8192 default | 40.952 | 28.764 | -29.76% | 30/0 | 28/0 |

Raw artifacts:

- `agent_space/f2a_narrow_perf_20260826/`
- `agent_space/f2a_narrow_source_compare/`
- `agent_space/lazy_cse_narrow_cast_review_20260826.md`

## Recommendation

Replace the general F2a implementation with this narrow version. It produces
the same code and performance for every measured target, preserves every
protected source, and removes 48 net production lines plus most policy-test
surface.

The tradeoff is explicit: masked-source deferral and multi-operation
group-width chains are not supported until a concrete workload needs them.
Those extensions can be added independently to this smaller mechanism with a
source-form and performance test that justifies each one.
