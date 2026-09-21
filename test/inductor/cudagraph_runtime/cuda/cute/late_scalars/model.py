"""Ordinary CuTe scalar arguments derived from input and allocated addresses."""


from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe, DirectHost, InputContract, IntegerRange, ObservedOrdinaryEntry,
    PythonEntry, SignaturePolicy, TensorInput,
)
import torch


@cute.kernel
def store_words(output: cute.Tensor, input_address: cutlass.Int64, owned_word: cutlass.Int32):
    output[0] = input_address
    output[1] = cutlass.Int64(owned_word)


@cute.jit
def launch_words(output: cute.Tensor, input_address: cutlass.Int64,
                 owned_bits: cutlass.Int32, stream: driver.CUstream):
    store_words(output, input_address, owned_bits + cutlass.Int32(7)).launch(
        grid=(1, 1, 1), block=(1, 1, 1), smem=0, stream=stream,
    )


def convert_arguments(output, input_address, owned_bits):
    tensor = from_dlpack(output, assumed_align=16, use_32bit_stride=False)
    return tensor, cutlass.Int64(input_address), cutlass.Int32(owned_bits)


CUTE = None


def host(box):
    count, source = box
    box.clear()
    owned = torch.empty_strided((count,), (1,), dtype=torch.float32, device=source.device)
    output = torch.empty_strided((2,), (1,), dtype=torch.int64, device=source.device)
    CUTE(output, source.data_ptr(), owned.const_data_ptr() % (1 << 30))
    return owned, output


def make_example(device_index):
    global CUTE

    owner = ObservedOrdinaryEntry(PythonEntry(launch_words), store_words,
        policy=SignaturePolicy(32, 64, 16, "stream"), conversion=convert_arguments)
    CUTE = DirectCuTe(owner)
    contract = InputContract(("integer", "tensor"),
        (TensorInput(1, torch.float32, (8,), (1,)),),
        (IntegerRange(0, 1, 1024),), device_index=device_index)
    return DirectHost(host, contract), owner
