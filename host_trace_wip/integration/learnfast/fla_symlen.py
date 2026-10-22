"""Overlay (learnfast, 10-15; coordinator-approved, labelled): flash_linear_attention's chunked-prefill ops use .shape[0]
instead of len(); same values, removes SymInt concretization.

    import sys; sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast"); import fla_symlen

Import before vLLM loads vllm.third_party.flash_linear_attention.ops.*: an import hook rewrites `len(chunk_indices)` ->
`chunk_indices.shape[0]` and `len(cu_seqlens)` -> `cu_seqlens.shape[0]` in those modules' source as they load. Under a host
trace, Python's len() of a tensor with a symbolic first size concretizes it (a pin, Eq(s, n), one per prefill length);
.shape[0] is the same value as a SymInt. No kernel or launch changes (stock arms unaffected). PATCHED counts the rewritten
modules (10 in vLLM 0.29)."""
import importlib.abc
import importlib.machinery
import sys

PREFIX = "vllm.third_party.flash_linear_attention.ops."
PATCHED: list = []


def rewrite(src: str) -> str:
    return src.replace("len(chunk_indices)", "chunk_indices.shape[0]").replace("len(cu_seqlens)", "cu_seqlens.shape[0]")


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if not name.startswith(PREFIX):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None or not getattr(spec, "origin", None):
            return spec
        origin = spec.origin

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                src = open(origin).read()
                new = rewrite(src)
                if new != src:
                    PATCHED.append(name)
                exec(compile(new, origin, "exec"), module.__dict__)

        spec.loader = _Loader()
        return spec


if not any(isinstance(f, _Hook) for f in sys.meta_path):
    sys.meta_path.insert(0, _Hook())
