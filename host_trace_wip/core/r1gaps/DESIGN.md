# DESIGN (for review by the coordinator and the user): eager steps that take a traced tensor's address (gap 1b)

Status: proposal only. Nothing below is implemented. r1gaps stops here for (ii)'s eager half.

## Problem
An eager step (an op the trace leaves eager: a foreign TVM-FFI function, a CuTe call the route declines, a
torch.library op with a Python impl) can take an int that is `t.data_ptr()` of a tensor the trace allocated. Under a
trace that int is `256*q_k + offset`, `q_k` the allocation's base symbol. `_TapeLowering.eager_call` lowers every SymInt
leaf as `ScalarSlot(row(v))`, and the integer program has no row for an allocation's or eager output's base (only for
arguments: `pointer i`). So lowering declines the whole trace ("sN is not read from the call's inputs"). One such call
sinks every variant. Before that, the warm-up check (`_check_witness`, fn's call order) compares int arguments by hint,
and the trace's hint is the placeholder address, the warm-up's the real one, so it declines there first
(core_gaps_repro gap_1b).

Where it bites now: R1 and Qwen3.8-Flash-Next, whose NVFP4 linears call FlashInfer's mm_fp4 cute-dsl GEMM with the
scale factors as make_ptr pointers (`data_ptr()` ints). The route takes DataPointer arguments now (r1gaps (ii) part 1),
but the GEMM's host function needs more route coverage (cute70/FROM_R1GAPS_DATAPOINTER.txt, items 1-8), so for now it
is an eager call with such ints.

## Proposal
A new eager-call leaf, an address: `LoweredAddress(root: Ref, displacement: row)`, for a SymInt leaf whose expression
holds exactly one root base symbol and splits as that base plus an offset (the split `pointer()` already does for
kernel pointer slots; argument roots keep today's ScalarSlot, which reads `pointer i`).
- Replay (Python and native): the step gets `base(root) + displacement`, the replay's address of the same bytes. To
  keep the C++ VariantSpec unchanged, the address leaf travels as a view leaf (kind 0: a 0-d uint8 view of the root at
  `displacement`) and the step's Python callable turns marked leaves into `data_ptr()` before calling the op.
- Memory plan: the view leaf is a use of the root at that step, so the root is live (not reused) across it. The
  callee may read any byte of the root's storage through the pointer; liveness is per root, so this holds.
- Warm-up check: an address argument is compared as (which root, offset) instead of its hint. The warm-up side needs
  the root's real base at the warm-up; the trace knows each allocation's warm-up address only if it records it, so
  the alternative is to leave address arguments out of the order key (weaker: a library that branches on an address
  bit would not be caught; such a branch would also be a hint read inside the library).
- Capture's lowering check (`check` per leaf): the displacement row against the traced offset.

## Cost / alternatives
- One eager step (a boundary) per such call, not a graph launch; the real fix for mm_fp4 is the route coverage.
- Alternative: keep declining (today). One unobserved or declined call with an address sinks the whole trace.
- Alternative: decline only the step (an eager fallback step without its address)? Not possible: the op needs the int.

## Asks
1. OK to add the address leaf (lower_tape, native flatten as a marked view, replay's eager-step callable, memory use,
   capture check)?
2. Warm-up check: record allocations' warm-up bases to compare (root, offset), or drop address ints from the key?
