"""Repeated generated reductions, unchanged CuTe GEMM, and user Triton SiLU."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph._compiler.compiler_cute_handoff.invocation import invoke_cute


BLOCKS = 3
WIDTH = 128


@triton.jit
def silu(source, destination, count, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + offsets, offsets < count, other=0).to(tl.float32)
    tl.store(destination + offsets, value * tl.sigmoid(value), offsets < count)


class RepeatedGemm(torch.nn.Module):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def forward(self, value, weight0, weight1, weight2):
        for weight in (weight0, weight1, weight2):
            centered = value.float() - value.float().mean(dim=1, keepdim=True)
            source = (centered * 0.125).to(value.dtype).unsqueeze(0)
            projection = torch.empty_like(source)
            invoke_cute(self.key, source, weight, projection)
            activated = torch.empty_like(projection)
            torch.library.wrap_triton(silu)[(triton.cdiv(projection.numel(), 256),)](
                projection, activated, projection.numel(), BLOCK=256,
            )
            value = (value.float() + activated.squeeze(0).float() * 0.125).to(value.dtype)
        return value, value[1:], value.transpose(0, 1)


def reference(value, *weights):
    for weight in weights:
        centered = value.float() - value.float().mean(dim=1, keepdim=True)
        source = (centered * 0.125).to(value.dtype).unsqueeze(0)
        projection = torch.bmm(source.float(), weight.float()).to(value.dtype)
        activated = torch.nn.functional.silu(projection)
        value = (value.float() + activated.squeeze(0).float() * 0.125).to(value.dtype)
    return value, value[1:], value.transpose(0, 1)
