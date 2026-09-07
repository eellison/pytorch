# Masked group-source fallback repro

Date: 2026-08-26

Runner:
`/data/users/eellison/pytorch/agent_space/repro_masked_group_source_20260826.py`

The minimized graph reduces groups of 16, slices off the final group-scale,
pads it with a scalar fill, and then broadcasts that result into a lane-width
division. The smallest tested failing shape is `B=2, D=48`.

## Result

| Source tree | eager | `nested_reduction=False` | `nested_reduction=True` |
| --- | --- | --- | --- |
| frozen #191775 | passes | passes, 2 kernels | compile assertion |
| F1 indexed forwarding | passes | passes, 2 kernels | same compile assertion |
| F2 lazy projection | passes | passes, 2 kernels | same compile assertion |

Both persistent and looped choices produce the same result. The assertion is:

```text
AssertionError: sub-parent reduction plan was lost before codegen
```

Each run records the imported `torch.__file__`; the editable source redirect
confirmed that the three rows used their named worktrees rather than the
currently installed editable source.

## Classification

This is not an F2 semantic regression. It is already present in the frozen
#191775 layer and is unchanged in F1 and F2.

It is more precisely an inherited fail-to-fallback bug than merely an
unsupported topology: disabling nested reduction cleanly emits the valid
two-kernel fallback, but enabling nested reduction admits a staged candidate
whose plan cannot be reconstructed at codegen and raises instead of declining
that candidate.

The repro should remain separate from the F2 green matrix. It does not provide
a valid end-to-end masked-callback oracle until the prerequisite fallback bug
is fixed.

## Commands

Representative passing fallback:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_WORKTREE=/data/users/eellison/pytorch/agent_space/followup_lazy_projection_wt \
PYTHONPATH=/data/users/eellison/pytorch \
conda run -n pytorch-3.12 python \
  /data/users/eellison/pytorch/agent_space/repro_masked_group_source_20260826.py \
  --batch 2 --dim 48 --no-nested --no-persistent
```

Representative failing staged attempt:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_WORKTREE=/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt \
PYTHONPATH=/data/users/eellison/pytorch \
conda run -n pytorch-3.12 python \
  /data/users/eellison/pytorch/agent_space/repro_masked_group_source_20260826.py \
  --batch 2 --dim 48 --nested --no-persistent
```

Logs:
`/data/users/eellison/pytorch/agent_space/masked_group_source_runs_20260826_redirected/`
