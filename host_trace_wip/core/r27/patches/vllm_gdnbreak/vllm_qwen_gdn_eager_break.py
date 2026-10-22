"""Correctness-forced deviation for vLLM 0.29 breakable CUDA graphs (VLLM_USE_BREAKABLE_CUDAGRAPH=1) on Qwen GDN models.

vLLM's piecewise compile lists vllm::qwen_gdn_attention_core* as splitting ops, so the GDN layer runs eagerly with each step's
metadata. Breakable capture only breaks at functions decorated with @eager_break_during_capture (attention, Kimi's GDN _forward,
the XPU GDN op), and Qwen's CUDA GDN layer has no decorator: its body is captured into the PIECEWISE segments with the capture-time
dummy batch's branch (the decode path: fused_recurrent_gated_delta_rule_packed_decode + causal_conv1d_update) and replayed for real
prefill / mixed steps up to the capture range, which then never run the chunked prefill kernels (r27/out/BREAKABLE_GDN.md).

This makes the two GDN core methods eager breaks, as vLLM does for Kimi's GDN: inside a breakable capture they end the segment and
are re-run eagerly at replay with that step's forward context; FULL-mode captures (pure decode) keep them in the graph, as the
decorator itself does. Import it after VLLM_USE_BREAKABLE_CUDAGRAPH is set and before the engine is built (the decorator is a no-op
when breakable graphs are off):

    import vllm_qwen_gdn_eager_break  # noqa: F401
"""
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as _qgdn

PATCHED = []
for _name in ("_forward_core", "_forward_core_fused_norm_packed"):
    _f = getattr(_qgdn.QwenGatedDeltaNetAttention, _name)
    if not getattr(_f, "_gdn_eager_break", False):
        _w = eager_break_during_capture(_f)
        if _w is not _f:
            _w._gdn_eager_break = True
            setattr(_qgdn.QwenGatedDeltaNetAttention, _name, _w)
            PATCHED.append(_name)
