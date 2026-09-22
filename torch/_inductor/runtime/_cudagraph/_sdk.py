"""Enable raw CuTe host-IR reads before the SDK registers value casters."""

import importlib.abc
import importlib.machinery
import importlib.metadata
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


_raw_values_active = ContextVar("cutlass_mlir_raw_values", default=False)


@contextmanager
def raw_values() -> Iterator[None]:
    token = _raw_values_active.set(True)
    try:
        yield
    finally:
        _raw_values_active.reset(token)


class _RawValueLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        original = module.register_value_caster

        def register_value_caster(typeid, *, replace=False):
            def register(caster):
                def cast(value):
                    if _raw_values_active.get():
                        return module.Value(value)
                    return caster(value)

                original(typeid, replace=replace)(cast)
                return caster

            return register

        module.register_value_caster = register_value_caster
        module.raw_values = raw_values
        sys.meta_path.remove(_FINDER)


class _RawValueFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "cutlass._mlir.ir":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError("CuTe's MLIR Python module is unavailable")
        spec.loader = _RawValueLoader(spec.loader)
        return spec


_FINDER = _RawValueFinder()


def require_active() -> None:
    module = sys.modules.get("cutlass._mlir.ir")
    if getattr(module, "raw_values", None) is not raw_values:
        raise RuntimeError(
            "CuTe CUDA graph preparation requires SDK activation. "
            "Call `from torch._inductor.runtime._cudagraph import _sdk; _sdk.activate()` "
            "before importing cutlass or CuTe runtime entrypoints. "
            "If the SDK is already imported, restart the process and activate it first."
        )


def activate() -> None:
    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.6.2":
        raise RuntimeError("Parameterized CuTe graph tracing currently requires nvidia-cutlass-dsl 4.6.2")
    module = sys.modules.get("cutlass._mlir.ir")
    if module is not None:
        require_active()
    elif _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    importlib.import_module("cutlass.cute")
    # the host trace's DSL-level hooks (compiled programs under a trace) are
    # installed with the SDK, before any conversion snapshots the DSL's runtime
    from torch.cuda import _host_trace_cute_dsl

    _host_trace_cute_dsl.install()
