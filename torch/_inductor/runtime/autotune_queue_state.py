# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import threading
from typing import Any


_autotune_queue_state = threading.local()


def _get_active_coordinate_descent_batch() -> Any:
    return getattr(_autotune_queue_state, "current", None)


def get_active_autotune_queue() -> Any:
    return _get_active_coordinate_descent_batch()


def _set_active_autotune_queue(batch: Any) -> None:
    _autotune_queue_state.current = batch


def _clear_active_autotune_queue() -> None:
    try:
        del _autotune_queue_state.current
    except AttributeError:
        pass


@contextlib.contextmanager
def suspend_autotune_queue():
    batch = get_active_autotune_queue()
    if batch is None:
        yield
        return

    _clear_active_autotune_queue()
    try:
        yield
    finally:
        _set_active_autotune_queue(batch)
