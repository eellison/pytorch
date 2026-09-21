import triton
import triton.language as tl


@triton.jit
def dynamic_user_add_one(x, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + offsets, offsets < n, other=0)
    tl.store(out + offsets, value + 1, offsets < n)
