# r27 diagnosis only: import this before FlashInfer to load flashinfer PR #5335 (head 780baa1889f2, "fix(gdn): fence TMEM loads before cg1
# shared_acc releases", fixes #5334) applied to FlashInfer 0.6.18's two Blackwell GDN kernel files: gated_delta_net_chunked.py and
# gated_delta_net_cp_prefill.py (r27/fi_fence/pr5335/, the exact PR diff in r27/fi_fence/pr5335.diff). Evidence runs only.
import importlib.abc, importlib.util, os, sys

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pr5335", "flashinfer", "gdn_kernels", "blackwell")
_MODS = {f"flashinfer.gdn_kernels.blackwell.{n}": os.path.join(_DIR, f"{n}.py") for n in ("gated_delta_net_chunked", "gated_delta_net_cp_prefill")}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        f = _MODS.get(name)
        return importlib.util.spec_from_file_location(name, f) if f else None


sys.meta_path.insert(0, _Finder())
print("fi_fence_hook: PR #5335 files", sorted(_MODS.values()), flush=True)
