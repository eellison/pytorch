"""Compiler-owned trailing scratch slots for CUDA Triton launches."""

from dataclasses import dataclass


class TritonScratchDeclined(ValueError):
    pass


@dataclass(frozen=True)
class ScratchSpec:
    size: int
    alignment: int


def scratch_specs(module) -> tuple[ScratchSpec, ...]:
    if type(module.num_ctas) is not int or module.num_ctas != 1:
        raise TritonScratchDeclined("Terminal Triton scratch requires one CTA per program")
    slots = []
    for present, size, alignment, profile in (
        (module.has_global_scratch, module.global_scratch_size, module.global_scratch_align, False),
        (module.has_profile_scratch, module.profile_scratch_size, 1, True),
    ):
        if type(present) is not bool or present != (size is not None):
            raise TritonScratchDeclined("Triton scratch metadata lost its trailing ABI slot")
        if not present:
            continue
        if type(size) is not int or not 0 <= size < 2**63:
            raise TritonScratchDeclined("Triton scratch size must be a nonnegative int64")
        if profile and size:
            raise TritonScratchDeclined("Nonempty Triton profile scratch is unsupported")
        if (type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1)
                or size and alignment > 256):
            raise TritonScratchDeclined("Triton scratch alignment exceeds the owned allocator contract")
        slots.append(ScratchSpec(size, alignment))
    return tuple(slots)
