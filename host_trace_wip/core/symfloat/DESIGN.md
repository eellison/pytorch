# Symbolic float arithmetic in traced host code (design for review, 2026-10-08)

Status: design only. Nothing in torch or the fork is changed. Paths are relative to agent_space/paramgraph. Base for
reading: land/int6/snap/step246 (Python) and land/int6/build_cpp8/src (C++). `SITE` = land/scratch/sglang/site/
flashinfer/data (FlashInfer 0.6.18). vLLM's JIT `fmha_gen` builds from a byte-identical copy (md5 3ab0f0eb...). `FORK` =
land/core/attn/fork/flashinfer/trtllm_trace.py. The scratch scripts behind the numbers below are in land/core/symfloat/scratch
(CPU only, no GPU used).

## 0. Summary

- **Representation:** a traced float is an IR node, either fp64 or fp32, one node per IEEE operation in the host's
  order. The program evaluator (Python `_step` and Program.cpp) gets explicit rows for each operation. No sympy is
  involved. Every float row holds a double; an fp32 row's double is always exactly an fp32 value. So the existing
  `feq`/`flt` rows compare both formats exactly, and widening fp32 to fp64 is free.
- **Guards:** a comparison of two traced floats is an ordinary boolean guard. Inside an op's host it is the op's guard
  (the existing owner sets), so a flip redispatches the op. `argmin` records exactly k-1 comparisons against the winner,
  with C++'s tie rule: the first minimum wins.
- **Contraction:** compiled C++ may contract `a*b + c` into an FMA, and the shipped binary does (see 0.1). For a C++
  `a*b + c` the IR carries every result a legal compilation could produce. A guard holds only where all of them decide
  the same way, and the op declines (named) at an input where they disagree. That is sound for whatever compiler built
  the user's FlashInfer, with nothing asserted about it.
- **Replaces `choice()`:** A3 (`choice`/`choice_range`) and its use in the fork are dropped. The batch pin goes away
  too, because batch enters the costs as an expression.

### 0.1 Facts found while reading (they shape the design)

1. **The C++ cost model is fp32.** In SITE include/flashinfer/trtllm/fmha/fmhaKernels.cuh:967-1095:
   - The cost tables are `std::unordered_map<int, float>` initialized from double literals (`2.2` -> double -> float).
     The locals are `float`, the factors are `1.0f`/`2.0f`/`128.0f`, and the running minimum starts at `FLT_MAX`.
   - The int32 quantities (seqLenPerCtaKv, mMaxNumCtasKv, numWaves) are converted to float by the usual arithmetic
     conversions.
2. **The shipped binary fuses one product.** Disassembly of vLLM's fmha_gen.so
   (land/scratch/vllm/cache/.cache/flashinfer/0.6.18/103a/cached_ops/fmha_gen, `selectTileSizeQForGqaGeneration` at
   0x23cf90, aarch64, host flags `-O3`, no `-ffp-contract`):
   ```
   scvtf s0, w1 ; fmul s1, s8, s1 ; fmul s1, s1, s0      // m = (factor*M)*float(seq), two roundings
   fmul s2, s2, s10                                     // R*128.0f (exact)
   fmadd s1, s2, s0, s1                                 // t = R128*float(kv) + m, ONE rounding
   scvtf s0, w0 ; fmul s0, s0, s1                       // t *= float(waves)
   fcmpe s0, s9 ; b.mi ...                              // strict <, so the first minimum wins
   ```
   GCC's default for C++ is `-ffp-contract=fast` on targets with FMA. On x86-64 with no `-march` (FlashInfer's JIT flags)
   no FMA instruction exists, so the same source compiles unfused there. The tile choice at a near-tie therefore depends
   on the user's platform.
3. **The fork's port has the right precision but the wrong evaluation order for this binary.** FORK:399 rounds
   `R128*kv` separately (`_f32(_f32(...) + _f32(_f32(R*128)*kv))`). A CPU sweep of the port's own candidate tuples over
   1.1M (config, max_kv) points (scratch/fma_sweep2.py; heads 16-128 per 4-8 KV heads, max_q 2-16, batch 1-16, 148/160
   SMs, max_kv up to 131072) found reachable inputs where the two orders pick different tiles:
   - Q/KV heads 32/8, max_q 16, batch 1, 148 or 152 SMs, max_kv 2049..2560 (512 values): the tile-32 and tile-16 costs
     tie in exact arithmetic (1448.96).
     - Unfused (the port, and an x86 build) picks tile 32.
     - The aarch64 binary's order picks tile 16.
   - The same tie shows up for 128/8 at max_q 4, 64/4 at max_q 8 and 32/4 at max_q 16 (see scratch output).
   - None at 160 SMs. No other disagreement anywhere else in the sweep.
   - The parity grid (land/core/attn STATUS 18:50) could not see this. At q_len 2 with 4 q heads per KV head,
     numTokensHeadsQ = 8 gives tile 8, and the model is skipped (`mTileSizeQ <= 8`). Only the 6-per-KV re-run reached
     it (16 vs 8). GPU confirmation of the 2049 point is test T5b.
   - Exact real-number ties are common in this model because its constants are decimals:
     `1.48*512 + 1.08*128*5 == 1.2*768 + 1.03*128*4`. scratch/tie_search.py finds them for every tile pair. Whether a
     reachable configuration lands on one decides whether rounding order matters.
4. **The existing float support** (`_host_trace_ir.py` "float lane", `_host_trace_program.py`):
   - fp64 only. `fadd`/`fsub`/`fmul` nodes exist but do not lower: IRLowering declines them.
   - Their sympy export is real arithmetic (`e(a) + e(b)`), which a consumer could simplify.
   - The only fp32 is the `f32div` row (ATen ReduceMoment's factor via `ht::Recorder::f32_div`): an integer row holding
     float bits.
   - Its Python reference rounds int -> double -> float, which is double rounding. Program.cpp rounds int64 -> float
     once. They disagree above 2^53: n = 2^53 + 2^29 + 1 gives 2^53 in Python and 2^53 + 2^30 in C++. This is a small
     existing bug, fixed by the shared rounding helper below.

## 1. Representation

- **No sympy, no ShapeEnv floats.**
  - torch's SymFloat under ShapeEnv is a sympy real. `sympy.Add`/`Mul` flatten, reassociate and fold Float constants at
    mpmath precision.
  - So no expression there denotes an IEEE operation sequence, and fp32 cannot be expressed at all.
  - Floats are therefore an IR-only feature (`symbolic = "ir"`, the default). Under `symbolic = "sympy"`, every fp32
    function declines by name.
- **IR nodes** (`_host_trace_ir.Ctx`, hash-consed like every node; the hint is the exact value at the traced call):
  - **fp64** (Python float semantics, C++ `double`): the existing `fconst fsym ffromint fadd fsub fmul fdiv fneg fsqrt`,
    plus `ffma` (std::fma).
  - **fp32** (C++ `float`):
    - `f32i(int)`: static_cast<float>(int64), rounded once to nearest-even.
    - `f32r(f64)`: static_cast<float>(double).
    - `f32add f32sub f32mul f32div f32fma`.
  - The hint of an fp32 node is a Python float that is exactly the fp32 value.
  - A format is a property of the node's op. A constant is accepted as an fp32 operand only if its double is exactly an
    fp32 value. Otherwise TypeError: "convert with f32()". That catches a port that forgets `float x = 2.2`'s rounding.
- **Python operators stay fp64.** Python operators on SymFloat (`a * b`), and c10::SymFloat's C++ operators, keep their
  language's meaning: fp64, unfused (CPython and c10::SymFloat evaluate each operation as one call).
  - An fp32 node used there is widened exactly (no node: its double is the value).
  - fp32 arithmetic happens only through the explicit functions in section 1.1. So the node graph's order is the host's
    order by construction.
- **Folding:**
  - Constant folding of a float node is allowed only when every operand is a constant. It is evaluated by the
    evaluator's own reference function.
  - No other rewrite on float nodes: no `x*1 -> x`, no `x+0 -> x` (`-0.0 + 0.0` is `+0.0`), no reassociation, no
    distribution. `Ctx.bounds` never reasons about float nodes.
- **Program rows** (`IntegerProgram`, Program.cpp). Each value is int64 bits of a double:
  ```
  existing: tofloat (int->f64), fsqrt, fdiv, feq, flt
  new f64:  fadd, fsub, fmul, ffma
  new f32:  f32i (int64->f32), f32r (f64->f32), f32add, f32sub, f32mul, f32div, f32fma
  new f->i: ftrunc, ffloor, fceil (to int64)
  ```
  - **Domain:** a float row whose result is non-finite or subnormal is FLOAT_DOMAIN, which is a miss. Under `domains`
    the row is made total and its domain row appended, as `fdiv`'s already is.
    - Excluding subnormals means FTZ/DAZ state (for example `torch.set_flush_denormal`) can never change a value. The
      domain is closed, so every operand is normal or zero.
    - The cost model's values are at least 1.
  - **Rounding mode:** round-to-nearest-even is assumed, the C/C++ default. torch never changes it.
- **Evaluators:**
  - **Python reference (`_step`):**
    - f32 add/sub/mul/div are computed in double, then rounded to fp32 with `struct`. This is exact: double rounding is
      innocuous when 53 >= 2*24 + 2.
    - `f32fma` and `f32i` above 2^53 round an exact `Fraction` once, through one helper, `_round_f32`.
    - fp64 `ffma` uses the same helper at 53 bits.
    - The helper also replaces `f32_bits`' conversion, which fixes the 0.1.4 bug.
  - **Program.cpp:** one IEEE operation per row. The result goes to the values array as bits, so no row's multiply can
    be fused with another row's add. `f32fma`/`ffma` call `std::fmaf`/`std::fma`, which are correctly rounded by
    contract.
  - **Reference check:** test T1 checks the two evaluators against each other, and both against numpy float32
    (unfused, one ufunc per operation) and an exact rational reference.

### 1.1 API (torch.cuda._host_trace, private; for review)

| function | meaning (C++ of the modelled host) | concrete in -> out |
| --- | --- | --- |
| `f32(x)` | `static_cast<float>(x)`; x an int/SymInt (round once) or a float/fp64 SymFloat | Python float (an fp32 value) |
| `f32_add/sub/mul/div(a, b)` | one fp32 operation, rounded once | same |
| `f32_fma(a, b, c)` | `std::fmaf(a, b, c)` | same |
| `f32_mad(a, b, c)` | the source's `a*b + c`, compiled with contraction allowed | the value if every legal result agrees, else declined |
| `f32_dot2(a, b, c, d)` | the source's `a*b + c*d`, same | same |
| `f64_fma/f64_mad/f64_dot2` | the fp64 counterparts (C++ `double` code) | same |
| `argmin(costs, below=None)` | `best = below; k = -1; for i: if (c[i] < best) {best = c[i]; k = i;}` (below None = no bound) | int |

- **Comparisons:** `<, <=, ==` on the SymFloats are exact (`fcmp`, lowered to `feq`/`flt`).
- **Subtraction under contraction:** write it as an addition of a negated operand. Negation is exact, and
  `fma(-a, b, c)` is what `fnmsub` computes, so the alternative sets are identical.
- **`f32_bits`** (an fp32 SymFloat as the int32 bit pattern a kernel parameter holds) would generalize the `f32div` row.
  It is not needed here and is left out.

## 2. Guards

- **A comparison is a guard where the host reads it.** `bool(t1 < t2)` goes through the existing `Env.guard_bool`
  record: decided at the hint, recorded as the node or its negation.
  - Inside a custom op's or dispatch_unit's host, the record lands in that op's owner set (`OpRec.guards`). A call where
    it fails redispatches the op. It never retraces the graph.
  - Lowering: `fcmp` -> `feq`/`flt` on the two rows. No NaN can reach a row, so every relation is `==` or `<` or a
    negation.
- **argmin, k-1 comparisons.** With w the winner at the hints, the guards are:
  - `c[w] < c[j]` for j < w (strict: an earlier equal cost would have won);
  - `c[w] <= c[j]` for j > w (a later equal cost loses);
  - `c[w] < below` when a bound is given;
  - if no cost beats `below`: `!(c[j] < below)` for every j.

  That is exactly the condition under which the C++ loop returns w. The loop itself is sound too, but it records the
  running-minimum comparisons, which also guard the order among the losers. Under `argmin`, only a change of the
  winner redispatches.
- **Contraction alternatives.**
  - `f32_mad`/`f32_dot2` return a value whose node is `falt(v1, ..., vn)`:
    - `a*b + c` -> {r(r(ab) + c), fma(a, b, c)};
    - `a*b + c*d` -> {r(r(ab) + r(cd)), fma(a, b, r(cd)), fma(c, d, r(ab))}.

    These are the contractions ISO C allows within an expression, and GCC's `fast` mode performs no others. It does not
    reassociate. A `-ffast-math` host build is out of scope: nothing can be exact there.
  - fp32 arithmetic on a `falt` maps over its alternatives (cross product, capped at 64, else declined).
  - A comparison of `falt` values evaluates every combination. Each evaluation is treated independently, so nothing is
    assumed about loop unrolling.
    - If all combinations agree at the hint, the guard is their `and` (true) or the negation of their `or` (false).
    - If they disagree, reading the comparison declines the op: "the decision depends on FMA contraction (an fp32
      near-tie)".
  - A `falt` may only flow into float arithmetic and comparisons. `int()`/floor, a pinned read (`guard_float`) or a
    select of one declines.
  - So `falt` never reaches lowering. Guards contain only `and`/`or`/`not` over `fcmp` of concrete-format nodes, and
    `Ctx.transfer` (redispatch) needs builders for the new ops only.
  - **Cost:** for the trtllm model, 5 candidates times 3 alternatives gives 4 pairs of 9 `flt` rows each.
  - **Effect:** in the 0.1.3 sweep the op declines at exactly the 512 x 4 inputs where legal compilations disagree, and
    nowhere else.
- **Ties:** strict `<` versus `<=` as above. FLT_MAX as the initial bound is `argmin(..., below=FLT_MAX)`, a constant
  fp32.

## 3. Conversions and constants

- **Int to float:**
  - `f32(n)` is the `f32i` row: one round-to-nearest-even of the exact int64. Above 2^24 it rounds (2^24 + 1 -> 2^24),
    as `scvtf` does.
  - fp64 `tofloat` is exact below 2^53 and rounded once above.
  - The C++ host computes these ints in int32. The IR's int64 rows equal them while no int32 overflow occurs. That width
    question is the C++ lane's existing one, not float-specific.
- **Float to int:** `ftrunc`/`ffloor`/`fceil` rows (C++ `(int)x`, `(int)std::floor(x)`, `(int)std::ceil(x)`). The
  operand must be finite and in int64 range, else FLOAT_DOMAIN.
  - The existing exact shortcut (floor/ceil of a quotient of ints -> floordiv/ceildiv) stays. It is the Python `math.ceil(a / b)` form.
- **Float constants:**
  - A C++ double literal assigned to a float is rounded twice: decimal -> double -> float. `f32(2.2)` in Python does
    exactly that, because Python's literal is the double.
  - A `2.2f` literal (decimal straight to float) would be `f32` of a decimal string. The trtllm source has none.
- **Device constants:**
  - The SM count is an int constant of the trace. In the fork it comes from `get_device_properties`. In the C++ lane it
    is `params.mMultiProcessorCount` (concrete in the traced pass).
  - That is valid because a replay entry is keyed by device (`key.device`), and the attribute is fixed for the device
    for the process's life.
- **max_kv and batch** enter as SymInts. `seq`, `kv` and `waves` are int expressions of them. The ceil_divs are
  expressions; the min()s become expressions too once the host uses `torch.sym_min`, else they are guards.
  - Both stay symbolic: no `guard_int`, no pin. The tile is guarded jointly on (max_kv, batch). The trtllm_cpp approval
    ("no pins on the spec-decode cost model") is met.

## 4. Where real-number simplification must not happen, and how evaluation stays exact

- **The IR:** none of the integer rewrites in the module docstring apply to float nodes (section 1, "Folding").
  Hash-consing is fine (same op, same operands, same value).
- **The sympy export** (`_SympyExport._expr`) is read by guard printing, `Tape.twin_guards`, and the oracle lowering
  (`direct=False`).
  - Every float operation exports as an opaque sympy Function: `F64Add`, `F32Add`, ..., `F32Fma`. No `eval` except
    all-constant folding through `_step`.
  - This includes the existing `fadd`/`fsub`/`fmul`/`fneg`, which today export as `+`/`*` (a sympy-route consumer
    could fold `1.2*(x*s)` at mpmath precision).
  - The sympy-route `Lowering` lowers each Function to its row, so both lowerings are row-for-row the same. The
    host_trace_ir_oracle test compares them, as it does for ints.
- **The C++ evaluator** must be built without `-ffast-math` (torch is not) and runs one operation per row.
- **The C++ host** (the user's FlashInfer) is not ours. Section 2's alternatives make its contraction choice irrelevant.
  Its FLT_EVAL_METHOD is 0 on every platform torch's CUDA build supports (x86-64 SSE, aarch64). x87 excess precision is
  out of scope.

## 5. Tests (after approval)

T1-T4 are CPU-only core tests; T5 needs the GPU.

- **T1 evaluator rows**, test/test_cuda_host_trace_program.py: table rows for each new op.
  - Edge values: ties-to-even, 2^24 + 1, 2^53 + 2^29 + 1 (also `f32div`, the fixed bug), FLT_MAX overflow, subnormal
    results, -0.0, int64 extremes for ftrunc/fceil.
  - Program.cpp == `_step` == numpy float32 / the exact reference on about 10^5 random operands per op.
- **T2 IR and lowering**, a new test/test_cuda_host_trace_float.py:
  - f32 expressions over size symbols lower to the expected rows. All-constant folding only.
  - Format errors (an fp64 constant that is not an fp32, an fp64 SymFloat into f32_add) raise.
  - The sympy export is opaque, and the IR and sympy lowerings are identical (oracle).
- **T3 guards at and around decision boundaries:**
  - Build the trtllm cost formula over symbols (seq, kv, waves as int expressions of max_kv and batch). Trace at a hint
    and record `argmin`'s guards.
  - Evaluate the guard program at every (max_kv, batch) in a box around each boundary, and compare with a concrete
    evaluation of the same alternative set at that point:
    - inside the guards, the same winner under every alternative;
    - outside, a different winner or a disagreement.
  - Cases:
    - near-ties: the exact real ties from scratch/tie_search.py, plus 1-ulp neighbours constructed by hand;
    - first-minimum ties, where equal costs pick the earlier index;
    - large ints: seq past 2^24, where f32(seq) rounds;
    - `below=FLT_MAX`.
- **T4 contraction:**
  - At an input where unfused and fused disagree (the 2049 tuple), the comparison declines with the named reason.
  - At a neighbour where they agree, the guard holds and covers exactly the agreeing set.
- **T5 real launcher, GPU (land/core/attn/tests):** the fork's `_best_tile_q` rewritten on the API (about 30 lines:
  `f32_dot2`, `f32_mul` by waves, `argmin(below=FLT_MAX)`, `sym_min`).
  - **(a) sweep:** for every config class that reaches the model on this box's cubins, capture the stock binding's
    launch (the parity harness) and compare its kernel/tile with the traced op's decision (trace once, then replay or
    redispatch at each point). Every point must be bitwise, or a counted contraction decline at exactly the predicted
    points.
    - Classes: Q/KV 32/8 at max_q 3/4/8/16, 48/8 at 2/4, 64/8 at 2/4, 128/8 at 4.
    - Points: batch 1-8, max_kv 1..8192 dense, and to 65536 at boundaries plus or minus 1.
  - **(b) disagreement point:** 32/8, max_q 16, batch 1, max_kv 2049..2560. If the box has 148/152 SMs, record which
    tile the stock binary takes (expected: 16). The fork's current port takes 32. That would confirm 0.1.3's mismatch
    on hardware.
  - **(c) redispatch count:** the core torch test (A3's
    `test_a_float_cost_model_choice_dispatches_again_where_it_changes`) is rewritten for fp64 Python costs and
    `argmin`. Redispatches == the regions first seen. 0 eager steps, 1 trace, bitwise.

## 6. For land/core/trtllm_cpp (consumption through c10::SymFloat)

- Values stay `c10::SymFloat` (PythonSymNodeImpl over IRSymNode). The traced body's `float` becomes an `fi_ht::F32` in
  the shim (that lane, about 80 lines): `std::variant<float, c10::SymFloat>`.
  - Concrete with concrete: plain float math, so the shim keeps the stock arithmetic.
  - Otherwise each operator calls `torch.cuda._host_trace.f32_*` through pybind, as ATen's `PyRecorder::f32_div` does
    today. A concrete operand is passed as its exact double.
  - `operator*` returns a lazy product, and `operator+` on lazy products calls `f32_mad`/`f32_dot2`. So the traced body
    keeps the source's expression text and gets the contraction alternatives with no edit.
  - `int -> F32` calls `f32(SymInt)`. Comparisons use c10::SymFloat's operators, which guard through `guard_bool`.
  - The loop's running minimum can stay (sound). Calling `argmin` gives tile-only redispatch, and that edit sits in the
    already-`#ifdef`'d body.
- No c10 or ATen change is needed. If ATen kernels later need host fp32 math, `ht::Recorder` gets fp32 methods (a C++
  API addition for a separate review).

## 7. Removed / changed

- **Dropped:** land/core/attn/patches/A3 (`choice`, `choice_range`) and the fork's `choice` call with its batch pin.
  The fork's TestTileChoiceGuard is replaced by T3/T5.
- **`f32_bits`:** uses the single-rounding helper. Program.cpp is unchanged for `f32div`.
- **Existing fp64 export:** `fadd`/`fsub`/`fmul`/`fneg` become opaque (printing changes only: they never lowered).

## 8. Size estimate

| part | lines |
| --- | --- |
| _host_trace_program.py: rows, `_round_f32`, domains | ~90 |
| _host_trace_ir.py: fp32 nodes, falt, comparisons over alternatives, transfer builders, opaque export | ~170 |
| _host_trace_lower.py: both lowerings of the new nodes/Functions | ~50 |
| _host_trace.py: API of 1.1 + sympy Functions | ~150 |
| Program.cpp/.h: ops, parse, step | ~70 |
| core tests T1-T4 | ~400 |
| fork port + T5 | ~30 changed + ~180 |

Roughly 530 lines of core and 580 of tests. About 2 days, plus the GPU sweep under the gpu locks (minutes per class).

## 9. Open questions for review

1. **Contraction semantics.** Choose between:
   - (a) This design: guard only where every legal compilation agrees, and decline at a disagreement. Sound for any
     build; costs a decline on near-tie inputs, e.g. 512 consecutive max_kv values in one 148/152-SM config.
   - (b) Declare this binary's contraction per site (it fuses here). Exact on this box, wrong on an x86 build. That is a
     caller-asserted property, so it is not proposed.

   A follow-up could remove (a)'s declines exactly: at a decline, the eager launch (which runs anyway) shows which
   alternative the binary took. Is that worth designing later?
2. Is the API surface in 1.1 acceptable: names, the `f64_*` counterparts, `argmin`, and keeping it private in
   torch.cuda._host_trace?
3. Is it acceptable that non-finite and subnormal float values are out of domain (a miss or decline)?
4. Should the fork port's fix (T5's rewrite) land in this lane? It is the first consumer and the GPU check of 0.1.3.

## 10. As implemented (after review, 2026-10-08 evening)

The user's answers simplified the design. What was built:

- **No contraction alternatives and no near-tie declines.** Each float expression is one operation sequence, the one
  the binary we run computes. For the trtllm model on this box that is `fmaf(R*128, ctasKv, (factor*M)*seq) * waves`.
  The GPU sweep against the real launcher is the check. `f32_mad`, `f32_dot2` and `falt` are not built.
- **No float domain.** The new float rows compute the IEEE result as the C++ host does: no FLOAT_DOMAIN for
  non-finite or subnormal values. The existing `fsqrt`/`fdiv` rows keep their checks.
- **Rows** (`_host_trace_program.py`, Program.cpp): `fadd fsub fmul` (double) and `f32i f32r f32add f32sub f32mul
  f32quot f32fma`. A float32 row holds its value as the double it converts to. The float32 division row is named
  `f32quot` because `f32div` (the int quotient's bits, ATen ReduceMoment) already exists.
  - The Python reference is `float_value`: double then round for + - * /, exact rational rounding for fmaf and for
    ints above 2^53.
  - `f32_bits` uses it, which fixes the double-rounding bug above 2^53.
- **API** (`torch.cuda._host_trace`): `f32`, `f32_add`, `f32_sub`, `f32_mul`, `f32_quot`, `f32_fma`, `argmin`,
  `FLT_MAX`. `f32_quot` instead of `f32_div` for the same name clash. Operands must be float32 values, else TypeError.
- **Sympy export:** each float operation (the double `fadd`/`fsub`/`fmul`/`fneg` too) is wrapped in the existing
  `Identity`. Measured (scratch/identity_probe.py; test TestFloatExport):
  - `simplify`, `expand`, `factor`, `cancel`, `together`, `doit`, `evalf`, `xreplace` and symbol substitution leave
    it unchanged. Substituting numbers keeps every Identity, so nothing folds across an operation.
  - What Identity does not block:
    1. `nsimplify` rewrites the Floats inside into decimal rationals.
    2. Identical Identity terms combine: `x + x -> 2*x`, `x - x -> 0`. That is exact in IEEE anyway.
    3. An Identity whose argument is all numbers folds at sympy precision, not float32. The IR folds all-constant
       operations itself first, so the export never builds one.
    4. The format is not in the sympy form: `f32add` and `fadd` both export as `Identity(a + b)`, and `f32r(x)` as
       `Identity(x)`. The fused form differs from mul-then-add, though: one Identity versus two.
  - So no consumer may evaluate the sympy form of a float guard. The one that did was the cover solver: its
    `_ir_vector` fell back to the export, and `_sympy`/`_VEC` evaluate Identity as its argument in real arithmetic.
    **Fix:** `_ir_vector` evaluates the float IR nodes by the rows' rule (`float_value`, per element) and compares
    them (`fcmp`).
  - The sympy-route Lowering declines Identity floats (no row), as it already declined `+`/`*` floats.
- **Port** (land/core/symfloat/fork, patch F3): `_best_tile_q` computes the costs symbolically through the API in the
  binary's order and picks with `argmin(below=FLT_MAX)`. No `choice()`, no batch pin, no hint replacement.
