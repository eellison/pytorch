"""Thin host boundary around the unchanged upstream Ampere tensorcore GEMM."""

import importlib.util
from pathlib import Path
import sys

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from torch._inductor.runtime._cudagraph._compiler.entry_signature import SignaturePolicy
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import ObservedOrdinaryEntry
from torch._inductor.runtime._cudagraph._compiler.python_entry import PythonEntry


UPSTREAM = None
GEMM = None


@cute.jit
def invocation(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream):
    UPSTREAM.bmm(GEMM, a, b, c, stream)


def convert_arguments(a, b, c):
    if c.shape[0] != 1:
        raise ValueError("This GEMM conversion requires a singleton output batch")
    c = c.as_strided(c.shape, (0, *c.stride()[1:]))
    values = tuple(from_dlpack(tensor, assumed_align=16, use_32bit_stride=False) for tensor in (a, b, c))
    for index in (0, 2):
        values[index].mark_compact_shape_dynamic(1)
    return values


def make_owner(upstream_path):
    global UPSTREAM, GEMM

    path = Path(upstream_path).resolve()
    name = "_ordinary_upstream_tensorop_gemm"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None or name in sys.modules:
        raise RuntimeError("The upstream GEMM requires a fresh source module")
    UPSTREAM = importlib.util.module_from_spec(spec)
    sys.modules[name] = UPSTREAM
    spec.loader.exec_module(UPSTREAM)
    GEMM = UPSTREAM.TensorOpGemm(cutlass.Float16, cutlass.Float16, cutlass.Float32, (2, 2, 1))
    return ObservedOrdinaryEntry(PythonEntry(invocation), GEMM.kernel,
        policy=SignaturePolicy(32, 64, 16, "stream"), conversion=convert_arguments)
