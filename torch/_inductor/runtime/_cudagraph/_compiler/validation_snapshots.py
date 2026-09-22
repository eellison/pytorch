"""Share module snapshots during one synchronous, read-only validation pass."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


Snapshot = tuple[str, bytes]
Serializer = Callable[[Any], Snapshot]


@dataclass
class _Snapshots:
    operations: tuple[Any, ...]
    values: dict[tuple[Any, Serializer], tuple[Any, Snapshot]] = field(
        default_factory=dict
    )
    active: bool = True


_CURRENT: ContextVar[_Snapshots | None] = ContextVar(
    "cute_validation_snapshots", default=None
)


def read_snapshot(operation: Any, serialize: Serializer, value: Any) -> Snapshot:
    current = _CURRENT.get()
    if current is None or not current.active or operation not in current.operations:
        return serialize(value)
    key = operation, serialize
    if key not in current.values:
        current.values[key] = value, serialize(value)
    return current.values[key][1]


@contextmanager
def validation_snapshots(*modules: Any) -> Iterator[None]:
    state = _Snapshots(tuple(module.operation for module in modules))
    token = _CURRENT.set(state)
    try:
        yield
        for (operation, serialize), (value, expected) in state.values.items():
            with operation.context:
                if serialize(value) != expected:
                    raise RuntimeError(
                        "Compiler module changed during read-only validation"
                    )
    finally:
        state.active = False
        state.values.clear()
        state.operations = ()
        _CURRENT.reset(token)
