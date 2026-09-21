"""Experimental entry points for native CUDA graph replay."""

from importlib import import_module


_EXPORTS = {
    "DirectHost": 'torch._inductor.runtime._cudagraph.direct_host',
    "prepare_direct": 'torch._inductor.runtime._cudagraph.direct_host',
    "DirectCuTe": 'torch._inductor.runtime._cudagraph.direct_cute',
    "DirectTriton": 'torch._inductor.runtime._cudagraph.direct_triton',
    "ObservedOrdinaryEntry": 'torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner',
    "PythonEntry": 'torch._inductor.runtime._cudagraph._compiler.python_entry',
    "SignaturePolicy": 'torch._inductor.runtime._cudagraph._compiler.entry_signature',
    "InputContract": 'torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract',
    "TensorInput": 'torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract',
    "IntegerRange": 'torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract',
    "IntExpr": "torch._inductor.runtime.cudagraph_arg_mapping",
    "NativeTerminalPolicy": 'torch._inductor.runtime._cudagraph.policy',
}

__all__ = tuple(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)
