# Inductor autotune queue: local work snapshot

This branch preserves a local prototype, not a ready-to-merge upstream change.
It is based on an older PyTorch checkout (`6d95092b12d`) and needs rebasing,
review, and current-tree validation before a PR.

The prototype queues independent generated Triton autotuning calls. It compiles
candidate frontiers through the existing compile pool, benchmarks ready launchers
in groups, and keeps scalar fallback paths. The grouped timing primitive,
coordinate-descent and static-config queue paths, config/cache identity,
generated wrapper integration, and focused tests are all in this snapshot.

The main implementation is in `torch/_inductor/runtime/autotune_queue.py`,
`autotune_common.py`, `autotune_benchmarking.py`, `benchmarking.py`, and
`triton_heuristics.py`. Compile-pool integration is in
`torch/_inductor/async_compile.py` and `runtime/compile_tasks.py`; generated
wrapper integration is in `torch/_inductor/codegen/wrapper.py`. Focused tests
live under `test/inductor/test_autotune_queue*.py` alongside updates to the
existing Inductor test files.

The local starting point was the `Temp` snapshot commit (`7a1d03042ae`). The
follow-up edits in six source and test files are included in this backup. The
working split proposal lives outside this branch in the original checkout's
ignored `agent_space/autotune_queue_stack_split_current.md`; it is a dry-run
plan, not a set of submitted PRs. In brief, it separates grouped timing,
per-autotuner launcher timing, cache/config identity, process-pool compilation,
the compile-time queue, static-config integration, and MultiKernel safeguards.

This backup is not evidence that tests pass against a current checkout. The
original local notes report that focused queue tests passed as of 2026-08-24;
lint was not run because `spin` and a project `.venv` were unavailable then.
For this sharing-only backup, `lintrunner -a` was attempted and failed on the
older checkout, including a SET_LINTER crash and unrelated baseline type
errors. Its automatic edits were reverted. Re-run checks and review the full
diff before using or submitting this work.
