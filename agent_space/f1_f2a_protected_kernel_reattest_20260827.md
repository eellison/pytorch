# F1/F2a protected-kernel re-attestation

Date: 2026-08-27

This is an AI-assisted scratch report for human review. No production files,
commits, or GitHub state were changed by this audit.

## Verdict

PASS.

- Final F1 matches all ten published protected-kernel hashes, with one staged
  kernel in every case.
- Final F2a matches the same ten protected forms, including every MXFP6 form.
- Across the 24-form NVFP4/MXFP4 matrix, F2a changes all and only the 12 NVFP4
  forms. Each change is the intended two-line cast-before-broadcast rewrite.
- All 12 MXFP4 forms are source-identical between F1 and F2a after metadata
  normalization. Kernel count, staged-kernel count, and structural counters are
  identical for all 24 pairs.

## Post-guard-removal F1 re-attestation

The final F1 cleanup was re-run after removing the resolver-owned guard model
and folded into commit `fc09fb38287d4639df6424c5c33cfb9749b000ba`. Its
complete `HEAD^` diff SHA256 is
`b42685fea074dbe98a7686ae175b343408eca86403ac1c4ec2a6dfe4dee442f3`.
All ten cases again emitted one kernel and matched the hashes below exactly.
Persisted sources are in
`agent_space/f1_no_guard_final_corpus_20260827/`.

## Earlier F1/F2a snapshots

Both worktrees are based on HEAD
`6b8ef64bd373b221c6ed1ba4f4b2f1edef2643c3`.

| Layer | Staged diff SHA256 | Unstaged diff SHA256 | Combined HEAD diff SHA256 |
| --- | --- | --- | --- |
| F1 `followup_indexed_rebase_wt` | `d9a050ef629b1d41712afb1ee7a2f3c65af52871eebd6c0f505b2eb241c2258f` | `1ca2b3a867ebcd1624598cd0395b48403f5663558ca89dfa86cbc3faf442fdfe` | `e4c858186e1d86ed2d630d748e630ab62e4b968d58318d4dd09f347ed5c48265` |
| F2a `followup_lazy_projection_wt` | `f7dec56b965764da13a758e31ea0e2d4913f7976ab84314d987bc7616fafbdbf` | `6b914c367513f9f27ac5f7a2fb16d3e218255ed5b2f058df6c82027fa433f661` | `5c51c2777d64c800d7da2c6ffde64d79f31884be7611424825eab3c33ffbc9bc` |

Both staged and unstaged `git diff --check` invocations pass.

## Ten protected cases

The authoritative baseline is the table in
`agent_space/indexed_forwarding_rebase_20260826.md`. The established harness
normalizes generated `triton_*` identifiers and temporary-directory prefixes
before hashing.

| Case | Published, final F1, and final F2a SHA256 |
| --- | --- |
| factor 2 persistent | `98b111d6f7798393c16439e9ed2c770ad98ddb8dee6cb731cb2d1640a75bc59c` |
| MXFP6 4:3 persistent | `b0fe88ce8d8fb83adc55d9d9d9e7c4c16bc360160bd14db3afe6343b42b57a79` |
| internal source persistent | `2efc561292dd1ba54ae8576515638189fcf7671122fcafbb64c0f987f3e9c887` |
| factor 2 looped | `6edb0230b628013056895dcbddd3643228608bb4be95d40250de15d7ec906961` |
| MXFP6 4:3 looped | `fcd5bda8ac08084965dfc92ad7142b0dadde8f3615469f3c4a8b28a44f6a976e` |
| internal source looped | `fb769f5dca3089dce0684cef2b3f4fcc6a5d9c232d736570d89aedd434be52ba` |
| reduced broadcast | `a907b3fbb5d83fdf8a61c7f292a4729d26aa01d4c8f9c9ed61f8562bb83b8258` |
| scale swizzle | `7f72ab3924703863f8e92edcf6849f6c70f9270e31f696cde87fd115571b48e9` |
| preshuffle | `710d6b80635a0add97076be321f2c19dd82301319f8cc85224f6421c99871497` |
| DCN preshuffle | `f3a02e5c076058f991b61a2ebb5bb8a54fb5d7c0e8f08f110018c2f99a869d19` |

Result: 10/10 F1 hashes match the published table; F2a produces the same ten
hashes. The four persisted F1/F2a sources are byte-identical after the harness
normalization. This protects the MXFP6 4:3, internal-source, scale-swizzle,
preshuffle, and DCN forms from unintended F2a changes.

## NVFP4/MXFP4 differential

The source-only matrix covers four shapes (`128x4096`, `4096x4096`,
`4096x4608`, `4096x8192`) and three modes (default, forced persistent, forced
looped) for each format.

| Comparison | Result | Normalized corpus SHA256 |
| --- | --- | --- |
| final F1, all 24 forms | matches archived F1 24/24 | `44761c936cc606bb038c33aca4245ed781ae366a53431e0277f585e108db05d0` |
| final F2a, all 24 forms | matches archived F2a 24/24 | `86e09a2aa6da07827fec6e1e48bc7d36a042dc99f784ba9a3f2f26582232ae21` |
| F1 vs F2a, MXFP4 12 forms | identical 12/12 | `f1a8e69e593f4d4f5e87de0a287c5a181099107c1adb5218730f15798555cf87` |
| F1 NVFP4 12 forms | expected pre-F2a form | `6924c663b50b42b3c85c9f02ae67e37f054e5cf62df40e1580476cfae26eec6a` |
| F2a NVFP4 12 forms | expected cast-before-broadcast form | `ae92cb523af1310add3c7548b488cee3b5b84da073832c3b9c0d8e0f6824bc92` |

Every NVFP4 pair has exactly one two-line replacement and no other normalized
source difference:

```text
F1:  FP8 -> uint8 bitcast -> broadcast -> FP8 bitcast -> FP32
F2a: FP8 -> FP32 at group width -> broadcast FP32
```

The aggregate structural manifest is identical for F1 and F2a:
`49256b3a9408d23b60fbf54c58006fefbc566d8793f264a77cead26233f04bc1`.
It covers generated/staged kernel counts, load/store counts, conversion counts,
split/broadcast/reshape counts, and reduction-loop counts. Every row emitted one
generated kernel and one staged kernel.

For historical comparisons, only nondeterministic metadata was removed:
`# kernel path`, AOT ID, backend hash, and `force_disable_caches`. Current F1
versus current F2a was also inspected as a line diff; the NVFP4 rewrite above
is the complete non-metadata difference.

## Commands

Common environment:

```bash
export LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```

Ten-case capture, run once with each worktree:

```bash
PYTORCH_WORKTREE=<WORKTREE> CUDA_VISIBLE_DEVICES=<GPU> \
conda run --no-capture-output -n pytorch-3.12 python \
  /data/users/eellison/pytorch/agent_space/run_wt.py \
  /data/users/eellison/pytorch/agent_space/compare_f1_indexed_sources.py \
  <WORKTREE> <OUTPUT_DIR>
```

The 24-form source-only capture, also run once per worktree:

```bash
PYTORCH_WORKTREE=<WORKTREE> CUDA_VISIBLE_DEVICES=<GPU> \
conda run --no-capture-output -n pytorch-3.12 python \
  /data/users/eellison/pytorch/agent_space/run_wt.py \
  /data/users/eellison/pytorch/agent_space/f1_perf_baseline_20260826.py \
  --formats nvfp4 mxfp4 \
  --label <LABEL> \
  --shapes 128x4096 4096x4096 4096x4608 4096x8192 \
  --modes default persistent looped \
  --skip-live-timing \
  --output <OUTPUT_DIR>/results.json \
  --generated-dir <OUTPUT_DIR>/generated
```

F1 used physical GPU 1 for the ten-case run and GPU 2 for the 24-form run.
F2a used physical GPU 1. All commands used the full-package `run_wt.py`
overlay and disabled both FX and Inductor caches.

## Artifacts

- F1 ten-case log and sources:
  `agent_space/f1_corpus_reattest_20260827/`
- F1 24-form result JSON and sources:
  `agent_space/f1_corpus_reattest_20260827/forms24/`
- F2a ten-case persisted sources:
  `agent_space/protected_kernel_reattest_20260827/f2a_ten/`
- F2a 24-form result JSON and sources:
  `agent_space/protected_kernel_reattest_20260827/f2a_24/`

Artifact hashes:

- F1 ten-case log:
  `c5f433d27a5d10a2019e1c46bff90d29bd3ca973c098bfb5b01ea71897c354d8`
- F1 24-form result JSON:
  `31cc221c65c4c0fd582d4baf86bec9a623956852c8ea035b4fdb582456bd76e2`
- F2a 24-form result JSON:
  `b222e495d23e9a372ef796d2fafa649760c529ccbd0b18439502f58da030dc99`
