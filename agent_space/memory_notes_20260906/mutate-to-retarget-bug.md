---
name: mutate-to-retarget-bug
description: "Pre-existing mainline inductor silent-wrong-answer bug: mutate_to's pointer swing retargets an earlier MutationLayoutSHOULDREMOVE; found 2026-09-02, fix prototyped in agent_space/mutate_to_retarget_wt"
metadata:
  type: project
---

2026-09-02. Found by the `index_put` fuzzer written for the padded-scatter work
(seed 188). **Pre-existing mainline bug, unrelated to nested reductions**, CPU and
CUDA both.

`y = x.clone(); y[idx] = v; y[mask] = -5.0` silently drops the masked fill.
`MutationLayoutSHOULDREMOVE.get_buffer()` (ir.py) resolves `self.target` through
`MutableBox.data` at call time, and `mutate_to`'s fast path (lowering.py) swings
`changed_data.data = val.data`, retargeting the earlier scatter at the masked
fill's own output buffer.

Fix prototype (uncommitted, detached at `cdd22ade294`) in
`agent_space/mutate_to_retarget_wt`: skip the fast path when
`ir.try_get_name(changed_data) in V.graph.mutated_buffers`. Full write-up with
the generated-code evidence, trigger table, rejected alternative, and
verification: `agent_space/mutate_to_retarget_bug_20260902.md`.

Randomized mutation-sequence fuzzer:
`agent_space/pr191974_failure_investigation/tmp_mutseq_fuzz.py <lo> <hi>`.
300 seeds: 15 MISMATCH on main, 0 with the fix.

**Env gotcha:** torchvision in the `pytorch-3.12` env is built against a different
libtorch and raises `RuntimeError` (not `ImportError`) on import, escaping the
`try/except ImportError` in `torch/testing/_internal/common_quantization.py`. Any
inductor test importing that module dies at collection. Use
`/tmp/run_wt_novision.py` (stubs `sys.modules["torchvision"] = None`, then
delegates to `agent_space/run_wt.py`) in place of `run_wt.py`.

See [[nested-reduction-fuzz-campaigns]] and [[worktree-overlay-harness]].
