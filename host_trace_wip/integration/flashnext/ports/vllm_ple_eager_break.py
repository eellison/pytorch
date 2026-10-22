"""Flash-Next companion to r27's vllm_qwen_gdn_eager_break (correctness-forced deviation arm for vLLM 0.29 breakable graphs).

Qwen3.8-Flash-Next's PLE layer (models/qwen4_exp/nvidia/ple_layer.py) runs two request-dependent custom ops,
vllm::qwen4_exp_compute_ple_ngram_ids and vllm::qwen4_exp_ple_short_conv. Their Python bodies branch on the step's metadata
(prefill / decode / spec state indices, query_start_loc). Breakable capture records them into the PIECEWISE segment with the
capture-time dummy batch's branch, the same failure as Qwen's GDN core: breakable+gdnbreak still ran the short conv's decode
path at the 9 prefill / mixed sizes of at most 1024 tokens (kernel record g2chk: 20 PLE ATen kernels differ from ours).

Both ops write into a caller-provided output, so they qualify for vLLM's own eager_break_during_capture. This re-registers
their CUDA implementations wrapped by it (allow_override). Import after VLLM_USE_BREAKABLE_CUDAGRAPH=1 is set and before the
engine is built; the wrapper is a no-op outside a breakable capture and for FULL-mode captures."""
import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.models.qwen4_exp.nvidia import ple_layer as _ple

_LIB = torch.library.Library("vllm", "IMPL")
PATCHED = []
for _name in ("qwen4_exp_compute_ple_ngram_ids", "qwen4_exp_ple_short_conv"):
    _f = getattr(_ple, _name)
    _w = eager_break_during_capture(_f)
    if _w is not _f:
        _LIB.impl(_name, _w, "CUDA", allow_override=True)
        PATCHED.append(_name)
