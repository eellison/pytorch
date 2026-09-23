# Owner(s): ["module: inductor"]

from torch._inductor.runtime._cudagraph._compiler.owned_numeric import (
    dump_numeric,
    load_numeric,
)
from torch.testing._internal.common_utils import parametrize


numeric_transports = parametrize("transport", ("direct", "roundtrip"))


def transport_numeric(program, transport):
    if transport == "direct":
        return program
    if transport == "roundtrip":
        return load_numeric(dump_numeric(program))
    raise AssertionError(f"Unknown numeric test transport: {transport}")
