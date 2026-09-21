"""Provider-neutral kernel invocation records."""

from dataclasses import dataclass

from torch._inductor.runtime.cudagraph_arg_mapping import CallArgument, IntExpr


@dataclass(frozen=True)
class Invocation:
    provider_key: str
    formals: tuple[str, ...]
    arguments: tuple[CallArgument, ...]
    grid: tuple[IntExpr, IntExpr, IntExpr]
    provider_identity: object | None = None
