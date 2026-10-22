# FlashInfer GDN chunked prefill: NaN in the last chunk of every sequence, in some processes

Standalone (no vLLM, no host tracing): `scripts/gdn_min.py` (minimal) and `repro_fi_gdn_nan.py` (with workspace / free-memory
patterns). FlashInfer 0.6.18 `flashinfer.gdn_prefill.chunk_gated_delta_rule` (the Blackwell CuTe DSL chunked GDN prefill that
vLLM 0.29 uses by default for Qwen3.5 / Qwen3.8 GDN layers on sm_10x), Qwen3.8-27B shapes (16 q/k heads, 48 v heads, head dim
128), bf16 inputs, fp32 zero initial state, finite inputs.

Observed on GB300 (sm_103a), CUDA 13.0, nvidia-cutlass-dsl 4.6.2, 2026-10-12, 9 fresh processes over GPUs 1-3:
- 4 processes: every output finite.
- 5 processes: the LAST 64-token chunk of every sequence is NaN, and the final state is NaN (1x64: tokens 0-63; 1x512:
  448-511; 4x128: 64-127 of each sequence; 1x100: 64-99), deterministically for the life of the process.
- Which kind a process is does not depend on the GPU; within a process it does not depend on the contents of the kernel's
  cached workspace or of free memory (repro_fi_gdn_nan.py: zero / 0xFF / random / NaN patterns all give the same NaN set).
Consequence seen in vLLM 0.29 (Qwen3.8-27B NVFP4, default GDN prefill backend): NaN outputs from vLLM's own eager forward in
about half the processes (from the start-up warm-up on); with gdn_prefill_backend=triton every process is clean.

Run: `python scripts/gdn_min.py` a few times (fresh processes). Not reported anywhere; the user decides.
