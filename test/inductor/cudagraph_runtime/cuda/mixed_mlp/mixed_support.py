"""CuTe fixtures and shared runtime support for the mixed model test."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_support import (
    IMPLEMENTATION,
    MixedTestCase,
    NativeObservation,
    POLICY,
    TerminalCall,
    counters,
    original_reference,
    terminal_policy,
    terminal_replay,
)
from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

import cutlass.cute as cute
from cutlass.base_dsl.jit_executor import ExecutionArgs, JitExecutor
from torch._inductor.runtime._cudagraph._compiler.compiler_cute_handoff.invocation import register_cute_entry
from torch._inductor.runtime._cudagraph._compiler.cute_dispatch import entry as ordinary_module
from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import TensorPolicy
from torch._inductor.runtime._cudagraph.api import ObservedOrdinaryEntry, SignaturePolicy
from torch._inductor.runtime._cudagraph.cute_types import CuteInvokeEvent, CuTeCall
