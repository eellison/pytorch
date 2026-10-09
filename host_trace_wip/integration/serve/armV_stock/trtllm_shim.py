# vllm_ht::trtllm_decode / trtllm_context: FlashInfer's trtllm-gen entry points as dispatcher ops, as in
# sglang/armF/trtllm_shim.py, for the calls vLLM's FlashInfer backend makes (v1/attention/backends/flashinfer.py).
# A traced forward sees one op with a fake impl instead of TVM-FFI's DLPack exchange; the CUDA impls call FlashInfer
# unchanged. install() rebinds the backend module's two imported names to wrappers that take the op for the covered
# form (bf16/fp16 K/V tuple, float scales, HND, trtllm-gen, one query token per request, no sinks / scale factors /
# lse / FP4 output) and otherwise call FlashInfer directly.
#
# Operand roles as in SGLang's shim: decode writes the workspace and the multi-CTA counters (reset to 0 by the
# kernel); context without lse reads neither. vLLM passes no counter buffer, so FlashInfer allocates a fresh zeroed
# one per call (an aten.zeros per layer); the ops get COUNTERS, one persistent zeroed buffer.
import torch

import flashinfer

_decode = flashinfer.decode.trtllm_batch_decode_with_kv_cache
_context = flashinfer.prefill.trtllm_batch_context_with_kv_cache

# FlashInfer needs round_up(max(bs * num_qo_heads, sm_count), 8) * 4 bytes: 1 MiB covers bs * heads <= 262144
COUNTERS = None

_lib = torch.library.Library("vllm_ht", "DEF")
_lib.define(
    "trtllm_decode(Tensor query, Tensor k_cache, Tensor v_cache, Tensor(a!) workspace, Tensor(b!) counters, Tensor block_tables, "
    "Tensor seq_lens, Tensor(c!) out, float bmm1_scale, float bmm2_scale, int max_seq_len, int window_left) -> ()"
)
_lib.define(
    "trtllm_context(Tensor query, Tensor k_cache, Tensor v_cache, Tensor workspace, Tensor counters, Tensor block_tables, "
    "Tensor seq_lens, Tensor cum_seq_lens_q, Tensor cum_seq_lens_kv, Tensor(a!) out, float bmm1_scale, float bmm2_scale, "
    "SymInt max_q_len, int max_kv_len, SymInt batch_size, int window_left) -> ()"
)


@torch.library.impl(_lib, "trtllm_decode", "CUDA")
def _decode_cuda(query, k_cache, v_cache, workspace, counters, block_tables, seq_lens, out, bmm1_scale, bmm2_scale, max_seq_len, window_left):
    _decode(query=query, kv_cache=(k_cache, v_cache), workspace_buffer=workspace, block_tables=block_tables, seq_lens=seq_lens,
            max_seq_len=max_seq_len, bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale, window_left=window_left, out=out,
            kv_layout="HND", backend="trtllm-gen", q_len_per_req=1, multi_ctas_kv_counter_buffer=counters)


@torch.library.impl(_lib, "trtllm_context", "CUDA")
def _context_cuda(query, k_cache, v_cache, workspace, counters, block_tables, seq_lens, cum_seq_lens_q, cum_seq_lens_kv, out,
                  bmm1_scale, bmm2_scale, max_q_len, max_kv_len, batch_size, window_left):
    _context(query=query, kv_cache=(k_cache, v_cache), workspace_buffer=workspace, block_tables=block_tables, seq_lens=seq_lens,
             max_q_len=max_q_len, max_kv_len=max_kv_len, bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale, batch_size=batch_size,
             cum_seq_lens_q=cum_seq_lens_q, cum_seq_lens_kv=cum_seq_lens_kv, window_left=window_left, out=out,
             kv_layout="HND", multi_ctas_kv_counter_buffer=counters)


torch.library.register_fake("vllm_ht::trtllm_decode", lambda *a: None, lib=_lib)
torch.library.register_fake("vllm_ht::trtllm_context", lambda *a: None, lib=_lib)

# keyword arguments the ops cover only at these values
_DECODE = {"sinks": None, "o_sf_scale": None, "kv_layout": "HND", "backend": "trtllm-gen", "q_len_per_req": 1, "max_q_len": None,
           "cum_seq_lens_q": None, "kv_cache_sf": None, "lse": None, "return_lse": False}
_CONTEXT = {"sinks": None, "o_sf_scale": None, "kv_cache_sf": None, "kv_layout": "HND"}


def _covered(kv_cache, bmm1_scale, bmm2_scale, out, kwargs, allowed):
    return (isinstance(kv_cache, tuple) and kv_cache[0].dtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)
            and isinstance(bmm1_scale, float) and isinstance(bmm2_scale, float) and isinstance(out, torch.Tensor)
            and all(k in allowed and (v is allowed[k] if allowed[k] is None else v == allowed[k]) for k, v in kwargs.items()))


def install():
    import vllm.v1.attention.backends.flashinfer as mod

    global COUNTERS
    if COUNTERS is None:
        COUNTERS = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")

    def decode(query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len, bmm1_scale=1.0, bmm2_scale=1.0,
               window_left=-1, out=None, **kwargs):
        if not _covered(kv_cache, bmm1_scale, bmm2_scale, out, kwargs, _DECODE):
            return _decode(query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len, bmm1_scale, bmm2_scale,
                           window_left, out, **kwargs)
        torch.ops.vllm_ht.trtllm_decode(query, *kv_cache, workspace_buffer, COUNTERS, block_tables, seq_lens, out, bmm1_scale,
                                        bmm2_scale, max_seq_len, window_left)
        return out

    def context(query, kv_cache, workspace_buffer, block_tables, seq_lens, max_q_len, max_kv_len, bmm1_scale, bmm2_scale,
                batch_size, cum_seq_lens_q, cum_seq_lens_kv, window_left=-1, out=None, **kwargs):
        if not _covered(kv_cache, bmm1_scale, bmm2_scale, out, kwargs, _CONTEXT):
            return _context(query, kv_cache, workspace_buffer, block_tables, seq_lens, max_q_len, max_kv_len, bmm1_scale,
                            bmm2_scale, batch_size, cum_seq_lens_q, cum_seq_lens_kv, window_left, out, **kwargs)
        torch.ops.vllm_ht.trtllm_context(query, *kv_cache, workspace_buffer, COUNTERS, block_tables, seq_lens, cum_seq_lens_q,
                                         cum_seq_lens_kv, out, bmm1_scale, bmm2_scale, max_q_len, max_kv_len, batch_size, window_left)
        return out

    mod.trtllm_batch_decode_with_kv_cache = decode
    mod.trtllm_batch_context_with_kv_cache = context
