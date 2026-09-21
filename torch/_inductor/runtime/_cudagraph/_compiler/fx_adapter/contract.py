"""Compiler input authority and events recorded while generated host code runs."""

from dataclasses import dataclass

import sympy

import torch
from torch._inductor.runtime._cudagraph._compiler.selected_kernel import SelectedKernel
from torch._inductor.runtime.cudagraph_arg_mapping import IntegerInput, IntExpr
from torch.fx.experimental.symbolic_shapes import ShapeEnv


class FXTraceDeclined(ValueError):
    pass


@dataclass(frozen=True)
class IntegerRange:
    index: int
    lower: int = 1
    upper: int | None = None


@dataclass(frozen=True)
class TensorInput:
    index: int
    dtype: torch.dtype
    size: tuple[int | IntExpr, ...]
    stride: tuple[int | IntExpr, ...]
    device: torch.device | None = None
    pinned: bool = False


@dataclass(frozen=True)
class InputContract:
    kinds: tuple[str, ...]
    tensor_inputs: tuple[TensorInput, ...]
    integer_ranges: tuple[IntegerRange, ...]
    device_index: int = 0


@dataclass(frozen=True, eq=False)
class AllocateEvent:
    tensor: torch.Tensor
    size: tuple[int | torch.SymInt, ...]
    stride: tuple[int | torch.SymInt, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True, eq=False)
class ReinterpretEvent:
    source: torch.Tensor
    tensor: torch.Tensor
    size: tuple[int | torch.SymInt, ...]
    stride: tuple[int | torch.SymInt, ...]
    offset: int | torch.SymInt


@dataclass(frozen=True, eq=False)
class LayoutEvent:
    tensor: torch.Tensor
    size: tuple[int | torch.SymInt, ...]
    stride: tuple[int | torch.SymInt, ...]
    label: str | None = None


@dataclass(frozen=True, eq=False)
class NormalizeEvent:
    tensor: torch.Tensor


@dataclass(frozen=True, eq=False)
class InvokeEvent:
    kernel_name: str
    arguments: tuple[torch.Tensor | int | torch.SymInt, ...]


@dataclass(frozen=True, eq=False)
class CuTeInvokeEvent:
    entry_global: str
    operands: tuple[torch.Tensor, torch.Tensor]


HostEvent = (
    AllocateEvent
    | ReinterpretEvent
    | LayoutEvent
    | NormalizeEvent
    | InvokeEvent
    | CuTeInvokeEvent
)


@dataclass(frozen=True)
class FXTrace:
    graph_module: torch.fx.GraphModule | None
    contract: InputContract
    symbol_sources: dict[sympy.Symbol, IntExpr]
    selections: dict[str, SelectedKernel]
    shape_env: ShapeEnv
    placeholders: tuple[object, ...]
    events: tuple[HostEvent, ...]
    outputs: tuple[torch.Tensor | int | torch.SymInt | None, ...]
    compiler_binding: object | None = None
    integer_inputs: tuple[IntegerInput, ...] = ()
