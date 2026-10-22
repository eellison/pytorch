# ms18's known conv coverage cost: the source fix options

STATUS (10-19, coordinator decision): kept refusal (C). The refusal stays, labelled. wip37 (B) and wip38 (M-keep) are reference
branches only. Why each option fails:
- (B) real memory for cuDNN operand sets and capture pool: learn - eager = +1.09 GB (attention case, bound 64 MiB) and
  +2.2 GB (conv_backward, bound 340 MB) in TestOutOfBandLearn, which gives up what me12 bought. It also leaves the refusal on
  class-1 leftovers from other families' address-only harvests (0xff07.. high halves at wgrad bytes 1704 / 1024, fprop byte
  968). B for all families would remove those, at GB-scale transient peaks on every key (blas included).
- (A) per-capture span offsets: doesn't touch class 1 (leftovers from earlier harvests), and the live flag class below makes
  any value classification moot.
- (M-verify), (M-keep), (M-narrow) map-on-verify: the field ms17 refused at strided dgrad byte 1424 is live. cuDNN leaves those
  bytes uninitialized in its host parameter struct, and CUTLASS stream-K dgrad / wgrad reads them, high half included, as a
  flag. Our high halves hang the kernel (ms17), the mirror's 4 GiB-shifted high half hangs it (wip38, 10-19: verify's first
  mirrored launch spun with the GPU at 100%, verify's host-zeroing hang path didn't release it, and the process had to be
  killed), and a zeroed high half hangs it (ms16 _canonical). Eager works only because its leftovers there are benign
  real-allocation high halves. No value-based proof can certify such a field, and the launch that would test it can spin the
  context unrecoverably. M-narrow (full-pointer qwords only) can't rule out the same spin: a stale pointer can be a semaphore.
- Filed (locally, nobody contacted) as an NVIDIA item: land/scratch/splitcost/NVIDIA_BUNDLE/cudnn_uninitialized_struct.md.

Unevaluated idea for after the talk (not worked on): run harvest's cuDNN host calls on a dedicated thread, so their malloc
chunks come from a separate glibc arena (and a separate cuDNN handle). That might stop our addresses leaking into the main
arena's chunks that the user's cuDNN calls reuse (class 1). It likely doesn't help the key's own capture (its uncaptured-path
struct is built on that thread too), so it's for evaluation only.

Unevaluated idea for after the talk (coordinator, 10-19; not worked on): c4 (pre-me12, real-memory harvest) graphed these rows
because its leftovers were real-allocation high halves, the same benign garbage eager and trees bake in. The refusal class
appeared once me12 moved harvest to the distinctive 0xFE00 / 0xFF00 reservations. Option: put the address-only reservations in
the ordinary device VA range, with high halves like real allocations', and detect our leftovers another way (record the
reservation ranges and check the full 64-bit value, not a distinctive high half). Tradeoff: an int32-under-high-half leftover
would then look exactly like eager's, which is what trees and c4 accept. Whether that is sound for the flag class depends on
the same "benign garbage" premise eager relies on; it matches eager rather than proving anything.


What is refused: cuDNN conv keys whose launch-parameter constants hold the high half of one of the harvest's own reservations
(_host_trace_harvest._STALE_STRUCT). Two classes are seen:
1. A malloc leftover. A cuDNN host struct reuses a heap chunk in which an earlier harvest's capture kept one of our addresses
   (operand arena 0xff00.., address-only span 0xfe00..), and an int32 is written into its low half. Seen at strided dgrad byte 1424
   (0xff00_0000_0020: the hang class) and at wgrad byte 712 (0xfe00_0000_4541).
2. A stale pointer in cuDNN's cached plan to our address-only span base (wgrad byte 1704 = 0xfe00_0000_0000): wgrad keeping a pointer
   into dgrad's workspace from an earlier capture.

Cost today: two Inductor conv-training rows refuse 2 keys each when run after other tests. The keys run as labelled eager steps;
correctness is intact.

Rejected:
- A heap scrub after each harvest (zero recently freed chunks of the parameter sizes). It is unreliable, because these structs are
  over glibc's per-thread cache size so chunk reuse order is not controlled. The user also rules out extra writes into the heap the
  user's own cuDNN calls use.
- M_PERTURB: it fills structs that eager calls build with 0xFF.

Real-fix options:
- (A) Per-capture span offsets. Each capture's address-only allocations start at a different offset, so a pointer a capture left in
  a cuDNN plan or heap chunk differs across captures. Classification then sees it vary (a role or loose bytes) instead of a constant.
  - It does not remove class 1, whose leftovers come from earlier harvests.
  - It needs the stale-pointer classification to handle a varying unmapped address (today "host state": refused).
  - Effort about 1 day. No memory cost.
- (B) Real memory for cuDNN conv operands in harvest captures (the caching allocator, as eager), so cuDNN only ever sees ordinary
  device addresses, as in eager.
  - Removes both classes for conv.
  - Cost: the conv call's operand bytes, allocated transiently during the harvest. For conv training activations that is up to
    the largest conv's input + output + weight bytes, released after the harvest. That is a transient peak, not a held one; it
    must be checked against TestMemoryGate's learn bound (eager peak + workspace + 64 MiB).
  - Effort about 1 day plus the memory gate.
- (C) Keep the refusal and widen nothing else. That is today's state.

## 10-19: option (M), map-on-verify. Soundness argument (written before code; coordinator decision 10-19)

Measured first: option (B) as specified (real memory for cuDNN operand sets and capture pool) costs learn - eager = +1.09 GB
(attention case, bound 64 MiB) and +2.2 GB (conv_backward, bound 340 MB) in TestOutOfBandLearn. It also leaves the refusal on
leftovers from other families' address-only harvests (0xff07.. high halves in wgrad byte 1704 / 1024, fprop byte 968). Dropped.
wip37 is kept as a reference.

What (M) changes. Today a constant qword in a cuDNN binding whose high half is one of the harvest's reservations (_own_highs:
the operand arenas, the address-only spans, the mirror spans), and which is neither a retired qword nor a mapped CUDA address,
is refused before verify (_stale_constant). Under (M) such a qword ("own-high qword") is not refused up front. It goes through
the proof verify already applies to every constant CUDA address and to constants into the current harvest's own memory
(ours_hits). A live result is refused, labelled as today.

The proof (the existing one, unchanged). verify launches the binding with each such qword q redirected to q + shift. shift is a
multiple of 4 GiB, so q's low 32 bits are kept. The target is a mirror window [q - 2 MiB, q + 64 MiB) of real memory mapped for
the check. The mirror is filled 0x00, then 0xFF. The qword passes as dead iff, for both fills:
  (1) every output equals eager's bitwise (on the call's fresh random operands, as for every binding), and
  (2) the mirror is not written (_Mirrors.touched).
A dead qword is kept as it is (q, its original bytes). This is the same check, with the same fills, windows and launches, that
decides a real retired workspace address today.

Q1. What proves the field dead? Conditions (1) and (2) together. Through q, the kernel writes nothing, and nothing it reads
changes any output under two opposite fills. For the int32-under-our-high-half shape, the redirected qword keeps the low half,
so a kernel reading the int32 alone (a stream-K mode flag) reads the same value at verify and at replay.

Q2. Why can't a kernel reading the field at replay escape the proof, with the VA unmapped again after the harvest? There are
three ways the field can be used:
  (a) Not used, or only its low int32 read as a value. Replay is identical to verify, so it's safe.
  (b) Dereferenced with a value-dependent effect, or written through. Then (1) or (2) fails at verify: refused.
  (c) Dereferenced with an effect independent of the value read (a read whose result both fills send the same way). This
      passes (1) and (2). With a real stale address the same read lands in mapped memory at replay; that is today's
      documented gap ("a live int32 under a stale high half that reads the mirror as it reads the allocation is the gap").
      With an own-high q, the VA is unmapped at replay, so the read faults: loud (illegal address), not silent, but a crash.
  To close (c) instead of crashing, the window must stay readable at replay. The cheapest way is one 2 MiB physical granule,
  shared, mapped read-only (CU_MEM_ACCESS_FLAGS_PROT_READ) at each such window inside our own reservation, for as long as a
  binding holds q. That is 2 MiB held in total, not per window: the same handle is aliased across windows. A kept qword then
  behaves exactly as a stale real address does in eager: a read sees memory, a write faults.
  This conflicts with condition 2 ("mapped only for verify, unmapped right after"), so it needs your call:
    (M-verify) unmapped after verify; case (c) is a loud replay fault;
    (M-keep) a read-only 2 MiB alias kept for kept qwords' windows; case (c) is safe.
  I recommend (M-keep): it makes own-high qwords exactly as sound as real stale addresses, for 2 MiB held.

Q3. If the proof says live: a labelled refusal ("<kernel> parameter <p> byte <o>: <q>, the harvest's own address high half in a
cuDNN parameter struct, live at verify (refused, never rewritten)"), as today. No scratch rebind. verify's existing path that
makes a live constant address "a slot into a scratch block of its own" is not used for own-high qwords, because their role
isn't known from classification.

Q4. Same proof as for real addresses? Yes. Same mirror windows and shift, same 0x00 / 0xFF fills, same bitwise and
not-written conditions, same launches. The only additions are routing own-high qwords into it and the read-only alias under
(M-keep).

Hang risk (why ms17 refused before any launch). The hang seen came from a rewritten high half (ms16 _canonical zeroed it), so
the kernel read address 0x20. Under (M) the launch reads the mirror (mapped). A kernel spinning on the mirror's value is
handled by verify's existing hang path (zeros from the host on another stream; the check fails: refused). A kernel still
spinning after that disables the provider (_Disabled), as for any binding today.

Scope. Own-high qwords in cuDNN bindings only (_CUDNN: conv fprop / dgrad / wgrad, cuDNN SDPA fwd / bwd). Mapping happens only
inside our own reservations, at minimum granularity. TestOutOfBandLearn's bound is unchanged. The me12 "learn maps no memory"
test allows the verify-scoped mirrors (already mapped today for real addresses), plus the 2 MiB alias under (M-keep).

Tests:
  - class 1 cross-family: a blas or attention harvest, then a conv in one process; the conv graphs with 0 refusals;
  - class 2: dgrad then wgrad (the stale plan pointer);
  - a live-field negative: a binding whose own-high qword is read value-dependently, refused with the label;
  - all bitwise vs eager, determinism forced with the fill off.

## 10-19: (M-keep) prototype hangs on the stream-K flag class (wip38, reference only)

wip38 = step_ms31 + (M-keep): own-high qwords routed into verify's mirror proof (live -> labelled refusal, never a scratch
slot), and the read-only alias granule with refcounted windows. The arena and the address-only span are reserved with guard
bands: the lower band is 512 MiB, because the driver places a hinted reservation only at a multiple of 512 MiB; the upper band
is 64 MiB. The arena reserves its high half's whole 4 GiB (address space only).

Result on conv_backward strided_dilated, channels_last, bf16 (the class ms17 refused at byte 1424): verify's first mirrored
launch never finishes. GPU at 100%, the process in _Mirrors.close -> synchronize. Verify's hang path (zeros from the host)
doesn't release it, so the context is stuck and the process has to be killed.

Reading. That qword (0xff00_0000_0020 here, an int32 under our high half) is a field cuDNN leaves uninitialized in its host
struct, and CUTLASS stream-K dgrad / wgrad reads it, high half included, as a flag. Its value decides whether the kernel spins:
  - our high halves hang it (ms17);
  - the mirror's shifted high half hangs it (now);
  - a zeroed high half hangs it (ms16 _canonical).
Eager works because its leftovers there are benign real-allocation high halves. So the field is live and its correct value is
whatever eager's heap garbage happens to be. No value-based proof can certify it dead, and launching it to find out can spin
the context unrecoverably, which is why ms17 refused it before any launch. (M) can't apply to this class.

Seen own-high qwords in the 13 rows, by shape:
  - int32 under our high half: dgrad byte 1424 (0xff00_0000_0200 / _0400, the hang class) and fprop byte 968 (0xff07_0000_000e);
  - full stale pointers: wgrad byte 1704 / 1024 (0xff07_0000_0000);
  - heap garbage: SDPA bprop byte 96 (0xff00_745f_666c, ASCII "lf_t" under our high half);
  - dgrad byte 576 (0xfe00_1220_00ff).
