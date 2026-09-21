"""Shared CuTe trace records without importing the optional SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sympy
    from torch._inductor.runtime._cudagraph._compiler.compiler_cute_handoff.invocation import InvocationEntry
    from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.conversion_trace import ConversionTrace
    from torch._inductor.runtime._cudagraph._compiler.python_entry import EntryCall
    from torch._inductor.runtime.cudagraph_arg_mapping import PointerSource
    from torch import SymInt
    from torch._subclasses.fake_tensor import FakeTensor

    from .cute_adapter import CuTeReceipt


@dataclass(frozen=True, eq=False)
class CuteInvokeEvent:
    entry: InvocationEntry
    operands: tuple[FakeTensor | int | SymInt, ...]
    call: EntryCall
    conversion: ConversionTrace | None = None


@dataclass(frozen=True)
class CuTeCall:
    bound: object
    pointers: tuple[PointerSource, ...]
    guards: tuple[sympy.Basic, ...]
    receipt: CuTeReceipt
