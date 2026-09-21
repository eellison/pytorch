from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

from .cudagraph_arg_mapping import BufferSource, IntExpr, WrapperCallRecords
from .cudagraph_launch_association import RecordedKernelLaunch

if TYPE_CHECKING:
    from ._cudagraph._compiler.selected_kernel import _SelectedCall


@dataclass
class _Preparation:
    records: WrapperCallRecords
    calls: "tuple[_SelectedCall, ...]"
    originals: tuple[torch.Tensor | int, ...]
    normalized: list[torch.Tensor | int]
    borrows: list[Any] = field(default_factory=list)
    launches: list[RecordedKernelLaunch] = field(default_factory=list)
    buffers: dict[BufferSource, torch.Tensor] = field(default_factory=dict)
    outputs: tuple[torch.Tensor | int | None, ...] = ()
    graph: Any = None
    instantiated: bool = False
    stream: Any = None
    capture_stream: Any = None
    entry: Any = None
    expression_values: dict[IntExpr, int] = field(default_factory=dict)
    source_calls: "tuple[_SelectedCall, ...]" = ()

    def abort(self) -> None:
        try:
            if self.entry is not None:
                self.entry.close()
            else:
                if self.stream is not None:
                    self.stream.synchronize()
                if self.capture_stream is not None:
                    self.capture_stream.synchronize()
                if self.graph is not None:
                    if not self.instantiated:
                        _failed_preparations.append(self)
                        return
                    batch = self.graph._prepare_kernel_pointer_updates((), 0)
                    cleanup = self.graph._make_replay_owner(batch, self.stream, (self,))
                    cleanup.close()
            self.borrows.clear()
        except BaseException:
            _failed_preparations.append(self)


# Public reset warns on destroy failures, so it cannot justify releasing borrows.
_failed_preparations: list[_Preparation] = []
