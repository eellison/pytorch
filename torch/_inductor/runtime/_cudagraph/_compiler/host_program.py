"""Shared scratch representation; selected kernel objects stay outside this program."""

from dataclasses import dataclass

import torch
from torch._inductor.runtime.cudagraph_arg_mapping import (
    InputSource, IntExpr, KernelCallRecord, WrapperCallRecords,
)


@dataclass(frozen=True)
class Allocate:
    value_id: str
    dtype: torch.dtype
    size: tuple[int | IntExpr, ...]
    stride: tuple[int | IntExpr, ...]


@dataclass(frozen=True)
class Reinterpret:
    source_id: str
    result_id: str
    size: tuple[int | IntExpr, ...]
    stride: tuple[int | IntExpr, ...]
    offset: int


@dataclass(frozen=True)
class Normalize:
    input_index: int
    before_call: int


@dataclass(frozen=True)
class Invoke:
    call: KernelCallRecord
    bound_grid: tuple[IntExpr, IntExpr, IntExpr]


@dataclass(frozen=True)
class InputAssertion:
    source: InputSource
    size: tuple[int | IntExpr, ...]
    stride: tuple[int | IntExpr, ...]
    label: str | None


@dataclass(frozen=True)
class HostProgram:
    records: WrapperCallRecords
    events: tuple[Allocate | Reinterpret | Normalize | Invoke, ...]
    input_assertions: tuple[InputAssertion, ...]
