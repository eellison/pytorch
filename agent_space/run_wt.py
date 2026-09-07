"""Worktree overlay harness.

Same idea as agent_space/run_stochastic_worktree_test.py, but overlays the whole
torch._inductor package (and torch._higher_order_ops.inline_asm_elementwise)
from the worktree instead of a fixed module list, so nothing silently resolves
back to the main checkout.
"""

import importlib.abc
import importlib.util
import os
import runpy
import sys
from pathlib import Path


WORKTREE = Path(os.environ["PYTORCH_WORKTREE"])
PREFIXES = ("torch._inductor",)
EXTRA = {
    "torch._higher_order_ops.inline_asm_elementwise": WORKTREE
    / "torch/_higher_order_ops/inline_asm_elementwise.py",
    "torch._dynamo.exc": WORKTREE / "torch/_dynamo/exc.py",
    "torch.utils._triton": WORKTREE / "torch/utils/_triton.py",
    "torch._dynamo.device_interface": WORKTREE / "torch/_dynamo/device_interface.py",
    # Imports inductor internals; must come from the same tree as torch._inductor.
    "torch.testing._internal.inductor_utils": WORKTREE
    / "torch/testing/_internal/inductor_utils.py",
}


def _worktree_path(fullname):
    if path := EXTRA.get(fullname):
        return path if path.exists() else None
    if not any(
        fullname == p or fullname.startswith(p + ".") for p in PREFIXES
    ):
        return None
    rel = fullname.replace(".", "/")
    pkg_init = WORKTREE / "torch" / rel.split("torch/", 1)[1] / "__init__.py"
    mod = WORKTREE / "torch" / (rel.split("torch/", 1)[1] + ".py")
    if pkg_init.exists():
        return pkg_init
    if mod.exists():
        return mod
    return None


class WorktreeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        found = _worktree_path(fullname)
        if found is None:
            return None
        if found.name == "__init__.py":
            spec = importlib.util.spec_from_file_location(
                fullname, found, submodule_search_locations=[str(found.parent)]
            )
        else:
            spec = importlib.util.spec_from_file_location(fullname, found)
        return spec


sys.meta_path.insert(0, WorktreeFinder())
import torch  # noqa: E402

torch.__path__.insert(0, str(WORKTREE / "torch"))
test_file = Path(sys.argv[1])
if not test_file.is_absolute():
    test_file = WORKTREE / sys.argv[1]
sys.argv = [str(test_file), *sys.argv[2:]]
runpy.run_path(str(test_file), run_name="__main__")
