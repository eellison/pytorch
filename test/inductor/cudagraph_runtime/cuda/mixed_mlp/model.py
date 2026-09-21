"""Residual MLP inference using generated GEMMs, user Triton, and CuTe."""

from torch._inductor.runtime._cudagraph._compiler.compiler_cute_handoff.invocation import invoke_cute
import torch
import triton
import triton.language as tl

from rmsnorm_fixture import EPSILON, WIDTH


BLOCKS = 3
EXPANSION = 1024


@triton.jit
def silu(source, destination, count, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + offsets, offsets < count, other=0)
    tl.store(destination + offsets, value * tl.sigmoid(value), offsets < count)


class ResidualMLP(torch.nn.Module):
    def __init__(self, key, device):
        super().__init__()
        self.key = key
        self.scales = torch.nn.ParameterList([
            torch.nn.Parameter(1 + torch.randn(WIDTH, device=device, dtype=torch.float32) * 0.05, requires_grad=False)
            for _ in range(BLOCKS)
        ])
        self.up = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(WIDTH, EXPANSION, device=device, dtype=torch.float32) / WIDTH**0.5,
                               requires_grad=False)
            for _ in range(BLOCKS)
        ])
        self.down = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(EXPANSION, WIDTH, device=device, dtype=torch.float32) / EXPANSION**0.5,
                               requires_grad=False)
            for _ in range(BLOCKS)
        ])

    def forward(self, value):
        for scale, up, down in zip(self.scales, self.up, self.down):
            normalized = torch.empty_like(value)
            invoke_cute(self.key, value, normalized)
            hidden = torch.mm(normalized * scale, up)
            activated = torch.empty_like(hidden)
            torch.library.wrap_triton(silu)[(triton.cdiv(hidden.numel(), 256),)](
                hidden, activated, hidden.numel(), BLOCK=256,
            )
            value = value + torch.mm(activated, down) * 0.25
        return (value,)


def eager_reference(model, value):
    for scale, up, down in zip(model.scales, model.up, model.down):
        normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + EPSILON)
        hidden = torch.mm(normalized * scale, up)
        value = value + torch.mm(torch.nn.functional.silu(hidden), down) * 0.25
    return (value,)
