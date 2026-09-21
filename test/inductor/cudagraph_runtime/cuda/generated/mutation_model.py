"""Ordinary generated foreach writes with caller-visible tensor aliases."""

import torch


WIDTH = 128


class ForeachMutation(torch.nn.Module):
    def forward(self, left, right):
        torch._foreach_add_([left, right], 1.0)
        torch._foreach_mul_([left, right], 2.0)
        return left, right, left[:, ::2], right[:, 1::2], left + right
