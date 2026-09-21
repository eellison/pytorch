"""Intercept actual direct adapter calls within one ordinary or symbolic host run."""

from contextlib import contextmanager
from contextvars import ContextVar


ACTIVE = ContextVar("terminal_direct_invocation", default=None)


@contextmanager
def activate(handler):
    token = ACTIVE.set(handler)
    try:
        yield handler
    finally:
        ACTIVE.reset(token)
