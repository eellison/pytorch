"""Compatibility imports for the canonical complete host-tape adapter."""

from .direct_hosttrace import (
    _HostIntegers,
    HostTraceReplay,
    lower_host_trace,
    prepare_host_trace,
)


__all__ = ["_HostIntegers", "HostTraceReplay", "lower_host_trace", "prepare_host_trace"]
