"""fnwrap.py DRIVER ARGS...: the Flash-Next lane's scripts/with_fi_autotune.py (one FlashInfer autotune file,
FN_FI_AUTOTUNE_FILE), but running the process's atexit handlers (diag27's end / prov records) before its os._exit."""
import atexit
import os
import runpy
import sys
import traceback
from pathlib import Path

path = os.environ.get("FN_FI_AUTOTUNE_FILE")
if path:
    import vllm.model_executor.warmup.flashinfer_autotune_cache as fac
    import vllm.model_executor.warmup.kernel_warmup as kw

    def resolve_flashinfer_autotune_file(runner):  # noqa: ARG001
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    fac.resolve_flashinfer_autotune_file = resolve_flashinfer_autotune_file
    kw.resolve_flashinfer_autotune_file = resolve_flashinfer_autotune_file
    print(f"FN_FI_AUTOTUNE_FILE {path} (exists: {Path(path).exists()})", flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
except BaseException:
    traceback.print_exc()
    code = 1
print(f"fnwrap: exit {code}", flush=True)
try:
    atexit._run_exitfuncs()
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
