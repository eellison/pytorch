"""Run a driver script (argv[1], its arguments after) with ONE FlashInfer autotune file for every arm of a timing set.

vLLM 0.29 keys the file by a hash of the vLLM config (warmup/flashinfer_autotune_cache.py), so fullnc, breakable and our V
(enforce_eager) would each tune separately and could pick different MoE tactics, i.e. different kernels. With
FN_FI_AUTOTUNE_FILE set, resolve_flashinfer_autotune_file returns that path for every arm: the first arm (fullnc) tunes and
saves it, the others load it (FlashInfer logs "Config cache hit ... source=config file" and "Saved N configs (0 new ...)";
check_autotune_hit.py fails a run whose log shows a new config). Only the cache location changes; vLLM's own load / save
code is unchanged.
    python_fn.sh scripts/with_fi_autotune.py DRIVER.py ARGS..."""
import os
import runpy
import sys
from pathlib import Path

path = os.environ.get("FN_FI_AUTOTUNE_FILE")
if path:
    import vllm.model_executor.warmup.flashinfer_autotune_cache as fac
    import vllm.model_executor.warmup.kernel_warmup as kw

    def resolve_flashinfer_autotune_file(runner):  # noqa: ARG001  (every arm: the one file)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    fac.resolve_flashinfer_autotune_file = resolve_flashinfer_autotune_file
    kw.resolve_flashinfer_autotune_file = resolve_flashinfer_autotune_file  # kernel_warmup imported the name
    print(f"FN_FI_AUTOTUNE_FILE {path} (exists: {Path(path).exists()})", flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))  # as `python script` would
# the driver's result is written by then; interpreter teardown with our native replay entries alive hung at 100% CPU in native
# code after t_f3id_v_r1 wrote its JSON (dev-box, 10-14 07:21), so leave through os._exit with the driver's exit code
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
except BaseException:
    import traceback

    traceback.print_exc()
    code = 1
    print("with_fi_autotune: driver raised, exit 1 pending", file=sys.stderr, flush=True)
print(f"with_fi_autotune: exit {code}", file=sys.stderr, flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(code)
