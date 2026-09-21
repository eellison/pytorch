"""Ordinary user dictionaries select exact observed CuTe artifacts."""


from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

import kernels
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe, DirectHost, DirectTriton, InputContract, IntegerRange, IntExpr,
    ObservedOrdinaryEntry, PythonEntry, SignaturePolicy, TensorInput,
)
import torch
import triton


CACHE = {}
ADAPTERS = {}
OWNERS = {}
ADD = None


def invoke_cached(source, destination, rows):
    key = 2 if rows > 16 else 1
    selected = CACHE.get(key)
    if selected is None:
        selected = OWNERS[key].compile(source, destination, rows)
        CACHE[key] = selected
        ADAPTERS[selected] = DirectCuTe(OWNERS[key])
    ADAPTERS[selected](source, destination, rows)


def host(box):
    rows, source = box
    box.clear()
    temporary = torch.empty_strided((rows, 128), (128, 1), dtype=source.dtype, device=source.device)
    output = torch.empty_strided((rows, 128), (128, 1), dtype=source.dtype, device=source.device)
    count = rows * 128
    ADD[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](source, temporary, count, BLOCK=128)
    invoke_cached(temporary, output, rows)
    ADD[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](output, output, count, BLOCK=128)
    return (output,)


def make_runtime(device_index):
    global ADD

    if CACHE or ADAPTERS or OWNERS:
        raise RuntimeError("The cache workload requires fresh user dictionaries")
    for key, launch in ((1, kernels.launch_one), (2, kernels.launch_two)):
        OWNERS[key] = ObservedOrdinaryEntry(PythonEntry(launch), kernels.affine,
            policy=SignaturePolicy(32, 64, 16, "stream"), conversion=kernels.convert_arguments)
    ADD = DirectTriton(kernels.add_one)
    rows = IntExpr("boxed", 0)
    contract = InputContract(("integer", "tensor"),
        (TensorInput(1, torch.float32, (rows, 128), (128, 1)),),
        (IntegerRange(0, 2, 127),), device_index=device_index)
    return DirectHost(host, contract), ADD

