"""DIAGNOSIS CONFIG (coordinator-approved 2026-10-13, check-only, never timed): deterministic QSA top-k order.

vLLM 0.29's QSA indexer (vllm/models/qwen4_exp/nvidia/ops/qsa.py, qsa_select_paged_tokens) writes each row's selected
compressed-block ids with _C.cooperative_topk / _C.persistent_topk in atomic-slot order (run-dependent), keeps that order
through _expand_qsa_indices_kernel, and the sparse attention accumulates in it, so outputs past 2048 tokens of KV (> 512
visible blocks) differ between identical eager runs. install() replaces vllm.models.qwen4_exp.nvidia.ops.qsa (before the
model is imported) by the same source with one asserted substitution: after the top-k call, _fn_det_order rewrites each
row's selected ids in ascending order, -1 for unused slots (bitmap + prefix sum per row in one Triton program per row; no
atomics). Applied to every arm of a check (stock reference included: stock_ref.py --qsa-det, the adapter's FN_QSA_DET=1).
Second source of run-to-run difference (fn_v4a: long prefills stayed unreproducible with the reorder alone): the indexer
score is sum_h relu(k . q_h) / divisor, so a block whose head dots are all <= 0 scores exactly 0, and a row with more than
512 visible blocks but fewer than 512 positive scores has exact ties at the top-k threshold; which tied blocks the _C top-k
keeps is run-dependent (the selected SET differs). Before the top-k, _fn_tie_break gives each exactly-zero visible score
the value -(column + 1) * 1e-30 (below every positive score, above -inf, distinct, larger for the lower column), so ties
at zero go to the lower block id. Selection is unchanged wherever the threshold is not at an exact tie (every row with
<= 512 visible blocks keeps all of them; rows with >= 512 positive scores keep the same set).
"""
import importlib.util
import os
import sys

_MOD = "vllm.models.qwen4_exp.nvidia.ops.qsa"
_CALL = "        topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\n"
_KERNEL = '''

@triton.jit
def _fn_det_order_kernel(blocks_ptr, mask_ptr, stride_b, stride_m, columns, K: tl.constexpr, BLOCK_K: tl.constexpr,
                         BLOCK_C: tl.constexpr) -> None:
    # flashnext lane diagnosis config (scripts/qsa_det.py): the row's selected ids rewritten in ascending order
    row = tl.program_id(0)
    for c0 in range(0, columns, BLOCK_C):
        cs = c0 + tl.arange(0, BLOCK_C)
        tl.store(mask_ptr + row * stride_m + cs, 0, mask=cs < columns)
    tl.debug_barrier()
    ks = tl.arange(0, BLOCK_K)
    b = tl.load(blocks_ptr + row * stride_b + ks, mask=ks < K, other=-1)
    valid = (ks < K) & (b >= 0) & (b < columns)
    tl.store(mask_ptr + row * stride_m + b, 1, mask=valid)
    tl.debug_barrier()
    base = 0
    for c0 in range(0, columns, BLOCK_C):
        cs = c0 + tl.arange(0, BLOCK_C)
        m = tl.load(mask_ptr + row * stride_m + cs, mask=cs < columns, other=0).to(tl.int32)
        pos = base + tl.cumsum(m, 0) - 1
        tl.store(blocks_ptr + row * stride_b + pos, cs, mask=(m > 0) & (pos < K))
        base += tl.sum(m, 0)
    tl.store(blocks_ptr + row * stride_b + ks, -1, mask=(ks >= base) & (ks < K))


@triton.jit
def _fn_tie_break_kernel(logits_ptr, visible_ptr, stride_l, BLOCK_C: tl.constexpr) -> None:
    # flashnext lane diagnosis config (scripts/qsa_det.py): exact-zero visible scores ordered by column
    row = tl.program_id(0)
    visible = tl.load(visible_ptr + row)
    for c0 in range(0, visible, BLOCK_C):
        cs = c0 + tl.arange(0, BLOCK_C)
        live = cs < visible
        v = tl.load(logits_ptr + row * stride_l + cs, mask=live, other=1.0)
        tl.store(logits_ptr + row * stride_l + cs, -(cs + 1).to(tl.float32) * 1e-30, mask=live & (v == 0.0))


def _fn_tie_break(logits: torch.Tensor, visible_blocks: torch.Tensor) -> None:
    _fn_tie_break_kernel[(logits.shape[0],)](logits, visible_blocks, logits.stride(0), BLOCK_C=1024, num_warps=4)


def _fn_det_order(blocks: torch.Tensor, columns: int) -> None:
    rows, k = blocks.shape
    mask = torch.empty((rows, columns), dtype=torch.int8, device=blocks.device)
    _fn_det_order_kernel[(rows,)](blocks, mask, blocks.stride(0), mask.stride(0), columns, K=k,
                                  BLOCK_K=triton.next_power_of_2(k), BLOCK_C=1024, num_warps=4)
'''


def patched_source() -> str:
    spec = importlib.util.find_spec(_MOD)
    src = open(spec.origin).read()
    if src.count(_CALL) != 1:
        raise RuntimeError(f"qsa_det: the top-k call is not found once in {spec.origin}")
    anchor = "\ndef qsa_select_paged_tokens("
    if src.count(anchor) != 1:
        raise RuntimeError("qsa_det: qsa_select_paged_tokens not found once")
    src = src.replace(_CALL, "        _fn_tie_break(logits, visible_blocks)\n" + _CALL + "        _fn_det_order(blocks, columns)\n")
    return src.replace(anchor, _KERNEL + "\n" + anchor)


def install() -> None:
    if _MOD in sys.modules and getattr(sys.modules[_MOD], "_fn_det_order", None) is not None:
        return
    if _MOD in sys.modules:
        raise RuntimeError("qsa_det.install() must run before the model's QSA ops are imported")
    # a real file (Triton reads a kernel's source through inspect), rewritten only when the patched text changes
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_qsa_det_patched.py")
    src = patched_source()
    if not os.path.exists(path) or open(path).read() != src:
        with open(path, "w") as f:
            f.write(src)
    spec = importlib.util.spec_from_file_location(_MOD, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD] = mod
    spec.loader.exec_module(mod)
    print("QSA_DET installed: deterministic QSA top-k order (diagnosis config)", flush=True)
