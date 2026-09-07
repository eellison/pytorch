# Sub-parent projection invariants

## IDENTITY

- Same output group: producer and consumer bodies are emitted from the same
  `_IterationSpace`. `MemoryDep.normalize()` equality is sufficient here because
  both node ranges are derived from the same row-major linear iteration ordinal.
  This permits shape-only flatten/reshape edges.
- Different output groups: name-based forwarding survives while
  `sub_parent_iteration_values()` changes. These edges require the strict
  source-frame proof: source axes match the consumer prefix axis-by-axis, added
  consumer axes are trailing and unused by the access, and canonicalized indices
  are equal. This rejects regrouped or transposed frames.
- A consumer cannot precede its internal producer's output group.
- The forwarded object is the producer's original `TritonCSEVariable`, so its
  existing `mask_vars` are preserved. No new source tail mask is inferred. The
  existing pointwise-cat path remains dependent on the consumer reapplying the
  same guard, as documented in `_PointwiseRemapHandler.load`.

## BROADCAST

- Parent/reduced-to-sub-parent broadcast uses the same strict source-prefix
  proof as cross-group internal forwarding.
- Indirect source or consumer indices are rejected by this proof.
- Plain `MemoryDep.normalize()` is not a frame proof: it merges axis boundaries
  and accepts the known `[2,4] -> [4,2,2]` regrouping counterexample.

## Fusion lifecycle and scoring

- `fusion_dependency_matches is not None` means a staged plan exists, including
  an empty tuple where every dependency matched normally. Such plans suppress
  later expansion, loop reordering, and index inversion, and still supplement
  exact-intersection scoring with normalized producer/consumer reuse.
- Projected scoring is added to ordinary exact-reuse scoring, but a producer
  write with an exact consumer read is excluded from supplemental scoring.
- Multiple projected reads of one producer write contribute one write score.

## Overlay unit tests

The editable install points at the parent checkout, so scheduler-only unit tests
were run by loading the worktree module before `runpy.run_path`:

```bash
python - <<'PY'
import importlib.util
import runpy
import sys
import torch._inductor

root = "/data/users/eellison/pytorch/agent_space/pr191775_simplify_wt"
name = "torch._inductor.scheduler"
spec = importlib.util.spec_from_file_location(name, f"{root}/torch/_inductor/scheduler.py")
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
sys.argv = [f"{root}/test/inductor/test_inductor_scheduler.py", "-k", "dependency_matches"]
runpy.run_path(sys.argv[0], run_name="__main__")
PY
```

Results:

- `-k dependency_matches`: 34 passed before the final score-test strengthening.
- `-k planned_dependency_matches`: 24 passed after the empty-plan scoring fix.
- `-k sub_parent_projection`: 2 passed after the group-aware IDENTITY proof.
- `-k preshuffled`: 6 passed after the empty-plan scoring fix.
- `python -m py_compile ...` and `git diff --check`: passed before the final
  test-only strengthening.

Do not treat the attempted MXFP6 overlay run as valid: mixing the worktree
`scheduler.py`/`simd.py` with the parent checkout's remaining modules failed on
source skew (`SIMDKernel.__init__(disable_multi_kernel)`). The last coherent
fresh-cache MXFP6 run before the group-aware refinement was 27 passed, 3 skipped.
