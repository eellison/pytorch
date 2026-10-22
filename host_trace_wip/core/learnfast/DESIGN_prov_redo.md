# Design note (learnfast -> coordinator, for the user): redo only ops whose records depend on their own inputs

Status: prototype (learnfast/prov_redo.py + opdeps.py), measured on MiMo, 27B, Qwen3.5-9B (classification) and Flash-Next.
The idea is the user's (10-14, via the coordinator). Opt-in, the vLLM integration turns it on (the closure pattern); the
core default stays as today: a library-body op's redispatch redo is refused (no_lib_redo), the call retraces.

## Rule
A redispatch may redo a library-body op (a non-ATen operator or a dispatch unit) when both hold:
1. Static: every free symbol of the op's records is its own input's. Records: its launches (grid, block, smem, every
   slot, every root's address), its allocations' sizes and strides, the guards it owns (Tape.owners). Inputs: the
   traced tensors and SymInts of OpRec.call (sizes, strides, storage offset, root address) and the op's own allocations.
   With closure on, the declared inputs (the replay's static prefix: parameters, buffers, KV caches) count as constants.
   An op whose records hold an eager or opaque call is never eligible (its body is invisible).
2. Confirmed: on the op's first redispatch in the process, it is redone once at the variant's own trace values (each
   symbol's hint), outside any context, and must give back its recorded launches exactly and decide nothing the trace
   did not; else the operator is refused for the process. The confirmation never makes an entry.

## Why both
- The IR keeps a forward-context int as its own symbol (equality guards do not substitute), so a body that reads one
  shows it in its records and fails (1): GDN prefill, attention.
- (1) cannot see a read that coincides with an argument's symbol. Under trusted layouts the token count is one symbol
  across input_ids and every metadata tensor; opdeps found such "shared" symbols on every op of every model. A body that
  read only a per-step tensor's size through that symbol would pass (1); outside the step function the redo bakes the
  concrete value. (2) catches it: at the trace values the redo records a constant where the trace has a symbol.
- Neither sees host-visible side effects (an op caching a choice in Python that a later op reads). For a top-level op
  the later op's guards are its own; the residual risk is the trace contract's (Python state fixed at trace time).

## Implementation (core, estimated ~150 LOC + tests)
- `HostTraceReplay(..., redo_policy="refuse" | "inputs_only")`, default "refuse" (no_lib_redo in core).
- opdeps' classifier over Tape (computed once per tape, cached); the static prefix from the replay.
- confirm(): `_redo` with `program.values` / `lowering.row` read from hints (the prototype swaps them; core would pass
  a values source), compare launches (keyed-site nodes excluded, as `_redo` does) and guards.
- Closure interplay: "declared" counts only with closure on; reads of declared storages resolve by the closure rule.

## Measured (untimed, shared GPU 3; trace counts exact, seconds indicative)
| Model / workload | Baseline | Criterion (+closure) | redo_context + closure |
|---|---|---|---|
| MiMo, 11 decode-curve points, --check | 25 traces | 25 (fp8_gemm_nt allowed and confirmed; attention refused statically; moe_forward's confirmation fails on guard form) | 9 |
| 27B, learn 1..400 | 25 | not run: GDN reads forward-context ints (expected 25) | 3 |
| Flash-Next, 1K decode ladder bs {1,2,3,4,5,8,16,32}, --check | 12 traces, 8 decode variants | 12, 8 (low_latency_gemm confirmed, but its new-M path needs a cuBLAS keyed site the op lacked; moe_forward_shared confirmation fails; GDN reads forward-context ints) | - |

Why MiMo does not move: a redispatch attempt must redo every failing selector at that call; at each new decode size 50
fp8_gemm_nt, 48 attention and 47 MoE selectors fail together, and attention (forward-context ints) is refused, so the
call retraces whatever the GEMMs allow. The criterion pays off only where the size-dependent ops are all input-only.

Flash-Next: the exact-M plan lookup in qwen4_exp_low_latency_gemm is input-only and confirms, but for an M with no plan
the body calls aten.mm; a redo cannot create the keyed site that call needs (the op at the trace's M had none), so the
redo is refused as another dispatch. Collapsing that ladder also needs redo-created keyed sites.

Prototype instrumentation (diag27's torch-function and launch-intercept resolve hooks) slowed traces 1.4x (MiMo) to
4.6x (Flash-Next) in the criterion arms; warm-up seconds of those arms are upper bounds.

## Open points
- Relaxed confirmation (PROV_RELAX_GUARDS): extra guards that use only the op's own (or declared) symbols are another
  form of a decision on its inputs; accepting them admits moe_forward on MiMo. Implemented, not yet measured.
- A section a confirmed op produces still goes through check_section (and the adapters' section refusal).
