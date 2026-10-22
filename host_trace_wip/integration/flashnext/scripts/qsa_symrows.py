"""Overlay (flashnext lane, 10-21; labelled, tracing-forced, our arm only; stock arms never load it): QSA's row-chunk loop over the
symbolic row count.

vLLM 0.29's qsa_select_paged_tokens (vllm/models/qwen4_exp/nvidia/ops/qsa.py) chunks the query rows with
`for row_start in range(0, rows, rows_per_chunk)`, rows = q.shape[0]. Python's range needs an int, so under a host trace the
symbolic row count is concretized: one variant per token count in decode (rows = batch size), prefill and mixed (STATUS.md
10-21). install() loads the module from the same source with one asserted substitution: the loop written as
`row_start = 0; while row_start < rows: ...; row_start += rows_per_chunk`, exactly the same iterations and slices (row_end =
min(row_start + rows_per_chunk, rows) is unchanged), each comparison recording a range guard, so the variants are one per chunk
count (rows_per_chunk = 128 MiB // (4 * columns): thousands of rows). No kernel or output change. Composes with qsa_det.py
(FN_QSA_DET=1, check-only): its patched source is the base then."""
import importlib.util
import os
import sys

_MOD = "vllm.models.qwen4_exp.nvidia.ops.qsa"
_HEAD = "    for row_start in range(0, rows, rows_per_chunk):\n        row_end = min(row_start + rows_per_chunk, rows)\n"
_HEAD_NEW = ("    row_start = 0\n"
             "    while row_start < rows:  # flashnext overlay (scripts/qsa_symrows.py): range(0, rows, rows_per_chunk)'s iterations, the row count compared not concretized\n"
             "        row_end = min(row_start + rows_per_chunk, rows)\n")
_TAIL = "            out[row_slice],\n        )\n    return out\n"
_TAIL_NEW = "            out[row_slice],\n        )\n        row_start += rows_per_chunk\n    return out\n"


def stock_path() -> str:
    spec = importlib.util.find_spec(_MOD)
    if spec is None or not spec.origin:
        raise RuntimeError(f"{_MOD} not importable")
    return spec.origin


def patched_source(det: bool = False) -> str:
    if det:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import qsa_det

        src = qsa_det.patched_source()
    else:
        src = open(stock_path()).read()
    for old, new in ((_HEAD, _HEAD_NEW), (_TAIL, _TAIL_NEW)):
        if src.count(old) != 1:
            raise RuntimeError(f"qsa_symrows: expected exactly one occurrence of {old!r} in {_MOD}, found {src.count(old)}")
        src = src.replace(old, new)
    return src


def install(det: bool = False) -> None:
    if _MOD in sys.modules and getattr(sys.modules[_MOD], "_fn_symrows", None):
        return
    if _MOD in sys.modules:
        raise RuntimeError("qsa_symrows.install() must run before the model's QSA ops are imported")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_qsa_symrows_patched.py")
    src = patched_source(det) + "\n_fn_symrows = True\n"
    if not os.path.exists(path) or open(path).read() != src:
        with open(path, "w") as f:
            f.write(src)
    spec = importlib.util.spec_from_file_location(_MOD, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD] = mod
    spec.loader.exec_module(mod)
    print(f"FN_QSA_SYMROWS installed: QSA's row-chunk loop compares the symbolic row count (det base: {det})", flush=True)
