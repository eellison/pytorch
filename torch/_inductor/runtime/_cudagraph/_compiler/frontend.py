from __future__ import annotations

import io
import threading
from typing import Any


_TRACE_LOCK = threading.Lock()


def _snapshot(module: Any) -> tuple[str, bytes]:
    bytecode = io.BytesIO()
    module.operation.write_bytecode(bytecode)
    return str(module), bytecode.getvalue()


class _Collector:
    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.module: Any = None
        self.context: Any = None
        self.function_name = ""
        self.text = ""
        self.bytecode = b""
        self.calls = 0

    def __call__(self, owner: Any, module: Any, function_name: str) -> None:
        from cutlass._mlir import ir

        self.calls += 1
        hooks = tuple(owner._trace_finalize_hooks) + tuple(owner._scoped_trace_finalize_hooks.get())
        if owner is not self.owner or self.calls != 1 or hooks != (self,):
            raise RuntimeError("Expected one isolated CuTe frontend hook")
        if not isinstance(module, ir.Module) or not isinstance(function_name, str) or not function_name:
            raise RuntimeError("Expected the live typed host module and its function name")
        hosts = [op for op in module.body.operations
                 if op.operation.name == "func.func" and op.attributes["sym_name"].value == function_name]
        if len(hosts) != 1:
            raise RuntimeError("The frontend hook does not identify one host definition")
        self.module = module
        self.context = module.context
        self.function_name = function_name
        self.text, self.bytecode = _snapshot(module)
