---
name: worktree-overlay-harness
description: "run_wt.py is the ONLY way to make a script import a worktree's torch._inductor; running the script directly silently uses the main repo"
metadata:
  type: project
---

`agent_space/run_wt.py` installs a meta_path finder that overlays
`torch._inductor` (plus a few extras) from `$PYTORCH_WORKTREE`. It takes any
script path plus its args, not just test files:
`PYTORCH_WORKTREE="$PWD" python ../run_wt.py /abs/path/repro.py <args>`.

**Why:** the conda env's torch is an editable install pointing at the main
checkout, so running a repro script directly imports the MAIN repo's
`torch._inductor` no matter which worktree you `cd` into. A worktree A/B done
that way compares a tree against itself and looks perfectly consistent, which
is what makes it dangerous. It cost a wrong conclusion on 2026-09-01 (a
"still crashes on the fix" result that was really the unfixed main repo).

**How to apply:** any A/B across worktrees must go through run_wt.py with an
absolute script path, and the sanity check is an instrumentation print: if a
debug print you just added to the worktree does not appear, the overlay is
not active and the run is meaningless. Related trap: Bash cwd does not
persist between calls, so always `cd <wt> &&` or `git -C <wt>` in the same
command. See [[nested-reduction-stack]].

Second trap (2026-09-03): the overlay covers only `torch._inductor` plus a
short `EXTRA` list, so a worktree on a *newer* main than the installed torch
dies on cross-module API drift -- e.g. current main's
`torch/_inductor/utils.py` calls `CudaInterface.is_gpu`, which the installed
`torch._dynamo.device_interface` does not have. Fix is to add the drifting
module to `EXTRA` in run_wt.py (`torch._dynamo.device_interface` is now
there). Symptom is an AttributeError at import of `torch._inductor.utils`,
not a test failure, so it is easy to misread as a broken environment.
