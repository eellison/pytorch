"""An ordinary Triton -> CuTe -> in-place Triton composition."""

from contextlib import ExitStack


from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe, DirectHost, DirectTriton, InputContract, IntegerRange, IntExpr,
    ObservedOrdinaryEntry, PythonEntry, SignaturePolicy, TensorInput,
)
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize_on_alignment=["source", "destination"])
def add_one(source, destination, count, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + index, index < count, other=0)
    tl.store(destination + index, value + 1, index < count)


@cute.kernel
def affine(source: cute.Tensor, destination: cute.Tensor, bias: cutlass.Int32):
    column, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    destination[row, column] = source[row, column] * 2.0 + bias


@cute.jit
def launch_affine(source: cute.Tensor, destination: cute.Tensor, bias: cutlass.Int32, stream: driver.CUstream):
    affine(source, destination, bias).launch(
        grid=(source.shape[0], 1, 1), block=(128, 1, 1), smem=0, stream=stream,
    )


def convert_arguments(source, destination, bias):
    values = tuple(from_dlpack(tensor, assumed_align=16, use_32bit_stride=False)
                   for tensor in (source, destination))
    for value in values:
        value.mark_compact_shape_dynamic(0, stride_order=(0, 1), divisibility=1)
    return (*values, cutlass.Int32(bias))


ADD = CUTE = None


def host(box):
    rows, source = box
    box.clear()
    temporary = torch.empty_strided((rows, 128), (128, 1), dtype=source.dtype, device=source.device)
    output = torch.empty_strided((rows, 128), (128, 1), dtype=source.dtype, device=source.device)
    count = rows * 128
    ADD[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](source, temporary, count, BLOCK=128)
    CUTE(temporary, output, rows)
    ADD[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](output, output, count, BLOCK=128)
    return (output,)


def make_example(device_index):
    global ADD, CUTE

    owner = ObservedOrdinaryEntry(PythonEntry(launch_affine), affine,
        policy=SignaturePolicy(32, 64, 16, "stream"), conversion=convert_arguments)
    ADD = DirectTriton(add_one)
    CUTE = DirectCuTe(owner)
    rows = IntExpr("boxed", 0)
    contract = InputContract(("integer", "tensor"),
        (TensorInput(1, torch.float32, (rows, 128), (128, 1)),),
        (IntegerRange(0, 2, 127),), device_index=device_index)
    return DirectHost(host, contract), owner, ADD


def main():
    device = torch.cuda.current_device()
    with ExitStack() as resources:
        runtime, owner, add = make_example(device)
        resources.callback(owner.close)
        resources.callback(add.close)
        resources.callback(runtime.close)
        samples = [(rows, torch.randn((rows, 128), device=device)) for rows in (5, 7, 35, 96)]
        outputs = [runtime([rows, source])[0] for rows, source in samples]
        torch.cuda.synchronize(device)
        print({"output_shapes": [tuple(value.shape) for value in outputs],
               "cached_variants": len(runtime.variants)})


if __name__ == "__main__":
    main()
