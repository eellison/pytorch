"""Overlay (learnfast, 10-15; PROPOSED, labelled, diagnostic until approved): Qwen GDN's prefill initial-state reset as a
masked_fill_ instead of boolean-mask setitem; same values.

    import sys; sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast"); import gdn_maskfill

vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py: `initial_state[~prefill_has_initial_state, ...] = 0` zeroes the
state rows of prefill sequences without an initial state. Under a host trace, Tensor.__setitem__ with a boolean mask
concretizes the mask's length (guard_int from the setitem path: Eq(s, n) on the number of prefill sequences), so every
sequence count is its own variant. `initial_state.masked_fill_((~mask).view(-1, *[1] * (initial_state.dim() - 1)), 0)` writes
the same zeros through one broadcast kernel with a symbolic size. Import before vLLM loads the module.
"""
import importlib.abc
import importlib.machinery
import sys

MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
OLD = "initial_state[~prefill_has_initial_state, ...] = 0"
NEW = "initial_state.masked_fill_((~prefill_has_initial_state).view(-1, *([1] * (initial_state.dim() - 1))), 0)"
PATCHED: list = []


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != MODULE:
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
                new = src.replace(OLD, NEW)
                if new != src:
                    PATCHED.append(name)
                exec(compile(new, origin, "exec"), module.__dict__)

        spec.loader = _Loader()
        return spec


if not any(isinstance(f, _Hook) for f in sys.meta_path):
    sys.meta_path.insert(0, _Hook())
