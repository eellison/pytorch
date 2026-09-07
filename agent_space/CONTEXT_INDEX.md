# wip-quant-context-20260906

Backup of the nested-reduction / quantization work context (2026-09-06).
Not for landing. Branches on this fork that carry the code:

- nested-reduction-lane-fold (6d243f4900c): validated, ready for review
- nested-reduction-multi-kernel (a30f04c119d): validated, ready for review
- nested-reduction-mutation-hoist (e5196889c55): validated, ready for review
- scheduler-reorder-fixpoint (adb23a4d464): validated, optional
- nested-reduction-persistent-heuristic (WIP, parked): must follow the lane fold
- nested-reduction-colwise-mxfp8 (WIP, experiment, flag-off)

Start with `memory_notes_20260906/MEMORY.md` (index of the working notes),
then `peak_cmp/REPORT.md` (peak-vs-peak vs flashinfer/QuACK, the lane-fold
root cause, MXFP8 findings, the Gluon prototypes), `quack_quant/REPORT.md`
(QuACK quantizers under nested reduction), `blockwise2d_design_proposal.md`
(dual dim0/dim1 design), `gluon_proto/` (Gluon row-wise and band prototypes).
Commit messages for the PR branches: `peak_cmp/*_commit_msg.txt`.
