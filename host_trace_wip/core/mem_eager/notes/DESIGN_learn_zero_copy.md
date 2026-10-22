# Design: learning an op costs at most one extra copy of its outputs (2026-10-14)

Invariant (coordinator/user): peak during a learn minus eager's own peak for that call is at most the op's output
bytes, plus bounded small terms (a workspace eager also allocates, the mirrors' windows, chunk buffers).

## Today's copies, per learned key (sizes from gh200-a's mobilenetv3 / MobileBert attributions), and their fate

| # | copy | where | today | fate |
|---|---|---|---|---|
| 1 | stand-in inputs | replay._learn :buffers | one per input or alias group, random-filled (MobileBert 1048 MB) | gone: the learn moves to the op's call (below), where the inputs are the call's live tensors |
| 2 | probe outputs | replay._learn `op(*args)` | one output (conv/attention dropped before captures by fix 3) | gone: the call's own output is the probe's result |
| 3 | capture outputs | each capture's fresh outputs, kept to the harvest's end so addresses differ | 2-4 x output (mobilenetv3 2 x 231 MB) | gone: the capture pool hands out reserved, unmapped VA (nothing runs during a capture), so captures cost VA, not memory |
| 4 | capture workspaces and plugs | capture pool | workspace-size, per capture | gone with #3 (also VA only) |
| 5 | verify's random operands | `refs`/`blocks()` | one input set (210 MB) | gone: verify launches on the call's live inputs; a written input keeps a copy (bounded by the op's written operands, as extern's cap) |
| 6 | verify's eager reference | `ref_out` | one output (822 MB) | gone: the call's own output is the reference |
| 7 | verify's eager rerun (spread) | second `fresh_outputs()` | one output (822 MB) | only on a mismatch of a floating output (an op eager may not reproduce bitwise), then one transient output; else none |
| 8 | the binding's outputs | `outs` | one output | stays: the one extra copy |
| 9 | compare buffer | `_same`'s `torch.equal` bool tensor | output/itemsize (411 MB) | bounded: chunked compare (16 MiB slices) |
| 10 | verify scratch | per scratch block | workspace-size | stays, bounded: eager allocates the same workspace in the call |
| 11 | digests, mirrors | small; mirror windows <= 66 MiB per constant range, spans persistent | stays, bounded |

All of #8-#11 come from one learn pool, released before the learn returns (as fix 2's per-harvest pool today).

## Moves, in order

1. Learn at the op's call during the trace (the in-trace twin of the replay-miss learn, _eager_step, which already
   learns "right after the call ran eagerly" on real tensors). At the OpaqueCall's eager run inside the trace, its
   inputs and fresh outputs are live: compute the key from the trace's values, suspend the trace, call provider.learn
   with the real operands. bind_opaque then only binds (no learn, no stand-ins). Removes #1, #2. The learning variant
   and background learns keep today's path (no call at hand).
2. Verify on the call's own tensors for every op (today only GEMMs, the in_place path): binding launched on the live
   inputs into a fresh output (#8), compared against the live output in chunks; a mismatch of a floating output runs
   eager once more into a transient buffer for the spread (#7). Removes #5, #6, bounds #9.
   - Risk: degenerate real data (an all-zero or constant input passes wrong bindings, the reason verify uses random
     operands). Guard: a per-input `amin == amax` check (one reduction); such an input gets a random stand-in, as
     today (rare, bounded by that input).
   - Inputs the op writes: copies, as today (extern's capped path).
3. Captures into VA, not memory: the capture pool becomes a MemPool over a pluggable allocator whose allocate is a
   slice of a reserved, never-mapped range (cuMemAddressReserve at the high hint). Captures record addresses only and
   their graphs are reset unlaunched, so no byte is touched. Removes #3, #4. Needs: cuBLAS/cuDNN never touch a pointer
   at capture (true today for operands, which already live in the unmapped operand arena); the plugging logic works on
   the VA allocator's blocks unchanged.
   - This is a new mechanism (a custom allocator for the capture pool): needs the user's OK.
4. Chunked bitwise compare (_same over 16 MiB slices of the integer view). Small, independent, first.

## The test (d)

For representative keys (cuBLAS mm/addmm bf16 at a large M, a cuDNN conv forward and backward with an 800 MB output,
cuDNN and flash SDPA forward/backward): measure eager's peak for the call alone, then a HostTraceReplay's first
traced call (which learns the key) from the same starting state; assert
`peak_learn - peak_eager <= max(output bytes) + workspace + 64 MiB`. Plus `memory_reserved()` back to its pre-learn
value after the learn (fix 2's test, extended to every pool).

## What this does to today's numbers (estimates, to be measured)

- mobilenetv3's 822 MB conv key: verify 2 x 822 + 411 + 225 MB -> 822 + 16 MB (+ captures: 2 x 231 MB -> 0).
- MobileBert: 1048 MB stand-ins -> 0.
- The HF rise on me2 (DistilBert +1.9 -> +2.4 GB) is most likely fix 2's private verify pool not reusing the default
  pool's cached free blocks (the trace call leaves many); with verify's working set down to one output, the term
  shrinks to that output whichever pool holds it. To be confirmed by gh200-a's DistilBert snapdiag.

## Order of work
4 (chunked compare) -> 1 (learn at the call) -> 2 (verify on live tensors) -> 3 (VA-only capture pool, after OK),
each gated and measured on gh200-a's flagged rows; the test (d) lands with 1+2.

## Target (user): captures cost 0 physical memory, verify costs at most one output, everything else is lent

Captured launches never execute: a capture needs valid, distinct addresses and no backing memory.
- The operand arena already works this way: a reserved, unmapped range at the high hint.
- Move 3 extends this to everything allocated inside the captured region: the call's workspaces and temporaries, its
  fresh outputs, the plugs and the conv holes. All come from the capture pool, so its allocator hands out slices of a
  reserved range (VA only) and never maps it.
- Physical memory during a learn is then: verify's one output-sized buffer (#8), the workspace eager also allocates
  (#10), mirror windows (bounded, #11), and chunk buffers (#9).

The only exceptions are library paths that do real device work outside the captured region during a learn:
- The op's first, uncaptured call. cuDNN builds its plan and parameter structs there, and with
  cudnn.benchmark=True ATen's algorithm search runs real kernels with real workspaces.
  - After move 1 this is the call itself, eager's own work with eager's own memory, so it adds nothing.
  - Under the replay-miss learn it already is the call.
  - Today's harvest adds a priming capture for cuDNN, which is captured and so becomes VA only after move 3.
- verify's launches (real by design), its eager rerun for the spread of a floating output that mismatched (one
  transient output, rare), its digests, and its mirror fills.
- Work that cannot run under capture: synchronous copies, host reads of device values such as .item() or nonzero's
  size, and blocking syncs. These fail the capture today, so the key is refused, not learned.
  - Nothing in the cuBLAS / cuBLASLt / cuDNN / SDPA / ATen-pointwise / reduce families we harvest does this.
  - Workspace-size and heuristic queries (cublasLtMatmulAlgoGetHeuristic, cudnnGet*WorkspaceSize, the cudnn-frontend
    heuristics) are host-only.
- A library's own internal allocations through cudaMalloc rather than the caching allocator (cuDNN/cuBLAS handle-level
  scratch made once per handle). These sit outside our pools, once per process, and are counted by eager too.
- Triton/CuTe autotuning happens at compile time, outside learns. Our harvest families don't JIT at a learn.

The test (d) checks the target directly:
- `peak_learn - peak_eager <= max(output bytes) + workspace + 64 MiB` for the representative keys.
- No physical growth from the captures: `memory_reserved` across a capture-only harvest step stays flat once the
  VA-only pool is in.

## Revised target (user, 2026-10-14): zero extra output copies

"We shouldn't even need the copies." Verify works in place, on the call's own output, comparing by digest:
1. The eager call has computed the real output (the learn happens at the call, move 1). Take a chunked GPU digest of
   its bytes: a strong hash of a few KB.
   - Each chunk's int64 words are mixed per word (xorshift-multiply with the word's index folded in) and summed. Every
     step is integer, so it is deterministic regardless of reduction order.
   - Several independent mixes give 4 x 64 bits.
   - Chunks of 16 MiB bound the temporaries.
2. Launch the binding into the same output buffer, on the call's live inputs, and digest again.
3. Equal: pass, and the buffer again holds the bitwise-identical values.
   Mismatch: rerun the eager op into the buffer to restore it, and refuse (or try the next channel: a memset, an owned
   scratch).
   - This is today's in-place GEMM path (row-sum digests plus an eager rerun on mismatch), generalized to every op with
     a strong digest.
   - Scratch fills 0x00/0xFF and the mirror poisoning stay, each a launch into the same buffer.

Exceptions:
- Ops that write their inputs: copies of only the written inputs (extern's capped path), or an eager rerun to restore.
- Floating outputs eager does not reproduce bitwise (_UNORDERED, atomics): the existing second eager run, which gives
  the spread for a tolerance compare. This is the one place a transient output-sized buffer remains.
- Degenerate live inputs (amin == amax): a random stand-in for that input only.

Learn memory target: digests + the op's own workspace + mirror windows, i.e. about eager.
- Captures cost no physical memory (move 3, address space only).
- Copies #6 (eager reference), #8 (the binding's output) and #9 (compare buffer) all go away. #7 stays only for the
  exceptions.

Bound test (d), revised: `peak_learn - peak_eager <= workspace + 64 MiB` for the representative keys. Ops that write an
input are allowed their written inputs' bytes; nondeterministic ops one output.

gh200-a's DistilBert/DistillGPT2 attribution (2026-10-15, Fsnap2) matches both halves:
- verify's own-pool buffers grew (+389 / +285 MB), and as a private pool they can't reuse the 1.8-2.6 GB of free
  default-pool blocks. Zero-copy verify removes them.
- The larger, older part: the learn runs while the trace call holds 4.2 / 2.7 GB live (the seam), so peak allocated
  is +1404 / +822 MB over default's. Learning at the call (move 1) runs the learn inside eager's own live set at that
  op, so this term goes away too.

Order unchanged: chunked digest compare (replacing _same and the row-sum digests) -> learn at the call -> verify in
place on live tensors -> address-space-only capture pool (prototype in parallel, lands after the user's OK).
