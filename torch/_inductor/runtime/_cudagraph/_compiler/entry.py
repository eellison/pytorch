from dataclasses import dataclass

from torch._inductor.runtime._cudagraph._compiler.components import DispatchComponents
from torch._inductor.runtime._cudagraph._compiler.join_components import JoinedComponents


@dataclass(frozen=True)
class DispatchEntry:
    components: DispatchComponents
    joined: JoinedComponents

    def check(self) -> None:
        self.components.check()
        self.joined.check()
        dispatch = self.components.dispatch
        if (self.joined.invocation.program is not dispatch.program
                or self.joined.invocation.lowering is not dispatch.joined.mapping.formals
                or self.joined.invocation.flow is not dispatch.joined.mapping.flow
                or self.joined.mapping is not self.components.plan.mapping):
            raise ValueError("Dispatch entry lost its original symbolic or compiler owners")
