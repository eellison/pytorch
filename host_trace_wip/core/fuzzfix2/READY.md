# fuzzfix2: T2 (size lists through the pointwise witness) and T3 (cross-context node resolution)

STATUS: NOT handed to int6. Two independent patches, one per bug, each for both lines:
- even line (candidate2/src_cpp = snap/step188, build_cpp7): `patches/T2.int6cpp.patch`, `patches/T3.int6cpp.patch`
- odd line (candidate2/src_py = snap/step187): `patches/T2.int6.patch`, `patches/T3.int6.patch`
Trees: base_e/base_o (rsync of candidate2 src_cpp/src_py), T2_e/T2_o, T3_e/T3_o. Repros/probes: repro/. Census: census/.
GH200 (gh200-b) copy: <path> (even line only; logs/d = final round, synced to logs/gh/).

| patch | md5 (mkpatch.sh) | net LOC (non-blank) |
|---|---|---|
| T2.int6cpp.patch | 1bc9e2599faa | +111 -4 = 107 (product +88 -4, tests +23) |
| T2.int6.patch | 8131e04806d9 | same hunks, offsets |
| T3.int6cpp.patch | 9312f75f1514 | +106 -16 = 90 (product +44 -14, tests +62 -2) |
| T3.int6.patch | 20d1d1ff25b9 | same hunks |

Apply in either order (T3 on T2 checked on both lines). Both apply cleanly (dry run) on the fuzzfix lane's comb_e and
on N1_trusted_redispatch / 97_cute_tvm_ffi_spec_guards. Semantic overlap: N1 edits _redo; T3 does not touch
_host_trace_redispatch.py, but a trusted redispatch now gets T3's refusal for a closure's symbol. T2 and the ATen
soundness lane both edit _host_trace_tape.py (`_traced_pointwise`, `_pointwise_host`, the body route, tables after
`_NEW_FACTORIES` and before `_pointwise`).

## T2: constant_pad_nd / expand_copy (and every op of a size or integer argument) through the pointwise witness

Root cause (snap/step188 _host_trace_tape.py): `_traced_pointwise` exempted list-typed arguments from its
symbolic-scalar decline (:1257, "a size (full's) sets only the shapes"), the witness ran at the hints (:1296), and the
admission rule `_elementwise` (:2105) takes any untagged op whose output sizes are its operands' at the hints. A pad of
0 / an expand at 1 is then a copy: the plan reads the input and allocates the output from TensorIteratorSym's shape,
so the pad never enters the tape and nothing guards it. The census below shows the same hole without any symbolic
value: constant integers that make the op the identity at the hints (as_strided_copy's strides equal to x's,
narrow_copy/slice_copy of every column) replay as a copy at every later x (repro/t_constlists.py: wrong on base, GH200).

Fix (per the user: trace these symbolically, keep the witness to a small provably-sound set):
- The witness takes an op only where its host decides nothing but its TensorIterator's: ATen's pointwise tag (as
  before), fill, and `_UNTAGGED_ELEMENTWISE` (20 packets read from their CUDA hosts: one TensorIterator over their
  tensors, no integer argument). Any other op `_elementwise` admits is witnessed only for an empty tensor's reduction
  fill (its zero size is specialized by the trace; the existing zero-element-reduction feature). Check placed in
  `_pointwise_host` once the plan is known (`_witness_takes`).
- `_traced_pointwise` declines any symbolic operand, list or not (the exemption is gone).
- Ops leaving the witness trace their body (`_kernel_less` route, parts traced as any op): expand_copy, view_copy,
  as_strided_copy, narrow_copy, slice_copy, t/transpose/unsqueeze/alias_copy, full_like/ones_like and the `_to_copy`
  fallback need no new code (SymInt bodies). Int-signature bodies get a Python transcription used only for a call of
  a symbolic size (`_SYM_BODIES`): constant_pad_nd (PadNd.cpp: each `pad < 0` / `pad != 0` is a guard, its parts
  narrow/clone/empty/fill_/copy_), full and ones (TensorFactories.cpp: empty at infer_full_options' dtype, fill_);
  new_full/new_ones/new_zeros reach them through `_NEW_FACTORIES`. A call of constant sizes keeps eager's C++ body.
- Not added: a fake-metadata comparison of the witness's output. With no symbolic or integer argument the witness's
  output is its operands' broadcast or the fake kernel's (int target), so the T2 mismatch cannot arise; the
  comparison would cost a fake call per pointwise op. Can add if wanted.

Regression test (test_cuda_host_trace_aten.py): `test_a_size_or_integer_argument_is_not_witnessed`, 6 cases
(pad_right = the F2 repro, pad_left_crop_right, pad_rows, expand_copy at 1, narrow_copy, as_strided_copy constant).
Before: 6/6 fail (GH200, base). After: 6/6 pass, `entry.eager == 0` (traced, no eager step).

### Census (T2)
Method: census/census.py wraps `_Trace._traced_pointwise` and traces OpInfo samples (op_db, CUDA float32, first 2
samples per op) on base_e; plus census/static.py (schemas) and native_functions.yaml (structured_inherits). Raw:
census/census_base.json.
- Reaching the witness: 288 ATen overloads. 163 by the pointwise tag (160 traced at least once), 1 by `_NULLARY`
  (fill_), 124 by `_elementwise` only (27 traced at least once).
- Traced by `_elementwise` only (untagged): _prelu_kernel, _to_copy (the traced host's fallback: complex dtypes),
  alias_copy, all/any (.default/.dim), as_strided_copy, complex, copy_ (fallback, fft), flip, floor_divide, full,
  full_like, hardswish, huber_loss, linalg_matrix_sqrth, linalg_vector_norm, log_sigmoid_forward, new_full, new_ones,
  ones, ones_like, polar, t_copy, transpose_copy, unsqueeze_copy.
- Symbolic list arguments seen at the witness: full.size, ones.size, new_ones.size (traced: the exemption), and
  convolution.padding / roll.shifts (declined later by the witness). From the fuzzer: constant_pad_nd.pad,
  expand_copy.size. Symbolic scalar: nonzero_static.size (declined).
- Pointwise-tagged overloads (636, static): none takes an integer list; three take an integer that is a functor
  parameter (polygamma n, round decimals, mvlgamma p).
- Cheap to trace symbolically, now done: the *_copy view family (body), full_like/ones_like (body), _to_copy (body),
  full/ones/new_* (transcription), constant_pad_nd (transcription). Not done (now an eager step where the witness used
  to take a degenerate shape): flip, all/any, linalg_vector_norm, linalg_matrix_sqrth, huber_loss; repeat/tile of a
  symbolic count (int-signature, an eager step before and after).
- Resulting witness set: pointwise-tagged ops + fill + {copy_, floor_divide(_), complex, polar, zero_, hardswish(_),
  hardswish_backward, hardtanh_backward, log_sigmoid_forward, log_sigmoid_backward, mish_backward, _prelu_kernel,
  elu_backward, leaky_relu_backward, softplus_backward, hardshrink_backward, softshrink_backward,
  hardsigmoid_backward}, never with a symbolic operand; any other op only for an empty reduction's fill.
- Probed and correct on base (so not claimed as bugs): untagged ops degenerate at the hints (all at numel 1, any over a
  size-1 dim, vector_norm of one element, flip of size 1; repro/t_degenerate.py, bad 0 before and after).

## T3: a redispatch resolves the variant's SymInt (a closure's) by id in the fresh trace's context

Root cause (snap/step188): `Ctx.transfer` (_host_trace_ir.py:1008) looks a node up as `memo.get(n.id)` (:1013) and
`src.written.get(n.id)` (:1021) assuming n belongs to `src`. In `_redo` the grid lambda / dispatch_unit body still
closes over the variant's SymInt n, so the fresh trace records nodes of the variant's context; IRRename transfers
them with src = the fresh context, and the id hits an unrelated fresh node (the grid became the row `n % 16 != 0`).
The intended refusal ("dispatched on a symbol of no argument") only fired when the id missed and the symbol name was
not in the mapping. Two more holes on the same path: IRSymNode's binary ops build `ctx.mk(op, (self.node,
other.node))` whatever context `other` is of (a node of one context with a child of another), and a guard on the
closure's SymInt is recorded in the variant's own env (after its lowering), never in the entry's predicate.

Fix (cross-context resolution impossible by construction; foreign symbols are refused, as the _redo comment intends):
- `Ctx.owns(n)` (identity: `nodes[n.id] is n`); `transfer` raises KeyError on a node `src` does not own (every level)
  and on a symbol whose mapped node the target does not own; IRRename turns it into FoldRefused.
- `IRSymNode._of(other)`: an operand of another Env declines ("an operation on symbols of two traces").
- An env records guards only while its own trace runs or none does (`_ir.ACTIVE` is `_host_trace_tape._active`):
  `Env._record` and `_TraceShapeEnv._record` decline "a guard on a symbol of another trace" through the running
  trace's `decline` (it declines even where its host catches the exception). This is what refuses the repro: Triton's
  binder guards n.
- `IRLowering.lower` declines a value its context does not own (its relations are looked up by id).
Each refusal is reported (redispatch_refusals / trace declines); the call retraces, no eager fallback.

Sites audited (every lookup of a node or symbol across trace contexts):
1. `Ctx.transfer` memo/written by id, symbols by name — fixed (ownership, every node; mapped values owned).
2. `IRRename.mapping` (fold, `_redo`, borrowed dispatches) keyed by source symbol names — names come only from the
   source's own symbols; enforced through (1) (a sym leaf reached is always the source's).
3. IRSymNode binary ops (add, sub, mul, `_ints` users, truediv, float_pow, `_rel`, sym_and/or) — fixed (`_of`).
4. Guard records of an env other than the running trace's (IR `Env._record`, sympy `_TraceShapeEnv._record`) — fixed.
5. `IRLowering`: memo and sources by node identity (safe), relations by `ctx.written[n.id]` — fixed (top-level
   ownership; children are owned by (3)).
6. `_SympyExport._exprs/_relations` by id, `Env._record`'s `ctx.written[g.id]`, cover.py memo by id — reached only
   with an IRSymNode's own node or a lowered program's own nodes: owned by (3)/(5); unchanged.
7. Borrowed dispatch memo (`dispatches`, 81/84): reuse renames through IRRename and checks guards with
   `consts.transfer(g, c[0]...)` — both through (1); a redo that touched a foreign env declines before it is cached.
8. Redispatch name maps `old.shape_env.ctx.symbols[q]` (:335, :461): q from the variant's own roots — same context.
9. `_value_hash` (by node id) / `_value_eq` (identity): collisions harmless.
10. Sympy redo `rename` (by name): fresh names are disjoint from the variant's (`unique_ids` shared), so a closure
    symbol is refused; guards via (4). Residual, not reachable today: a third env's symbol of a colliding name.

Regression tests:
- test_cuda_host_trace_triton.py `test_a_closures_symbol_is_never_resolved_in_a_redispatch` (tuned = the F3 repro,
  plain grid lambda, dispatch_unit closure): before 3/3 fail (wrong output / driver error), after 3/3 pass, no
  redispatch, every refusal "a guard on a symbol of another trace".
- test_cuda_host_trace_guards.py `TestIRContexts` (CPU): transfer never resolves a node of another context by id
  (forced id collision), an operation on two traces' symbols declines, a guard on another trace's symbol declines and
  is recorded once no trace runs. Before: 2 fail + 1 error; after: pass.
- Test adjustment: `TestIRSymNodeDunders._state` compared two Envs' SymFloats with `==` (a cross-context node);
  it now compares their hints, as for SymInt/SymBool.

## Results

### Even line (int6cpp patches), GH200 gh200-b (sm_90), ht/fuzzfix2, logs/gh/{a,b,d,e,f,T2d,T3d,T2f,T3f,basef}
Repros (before = base, after = patched tree):
- F2_pad_zero_at_trace_noncontig: before 2/4 calls wrong shape ((4,13) for (4,16), (4,21) for (4,24)); after 4/4 OK.
- t_listargs / t_listargs2 / t_listargs3 (pads left/negative/rows/value/transposed, list-arg ops): 16 MISMATCH calls before, 0 after.
- t_constlists: before as_strided_copy, narrow_copy, slice_copy of constant arguments wrong at 2/4 calls each; after 0.
- t_factories (full/ones/new_*/zeros_like/full_like/expand_copy/view_copy/transpose_copy of symbolic sizes): 0 wrong
  before and after; 1 trace, 2 replays, no eager step after (still traced, now through bodies).
- F3_redo_closure_grid_wrong_row: before 872/1000 elements wrong at both redispatched calls; after OK (retrace).
- t_closure2: plain 872 wrong -> 0; busy/busy_tuple 0 -> 0. t_unit_closure: driver "invalid argument" -> all OK.
New tests: test_a_size_or_integer_argument_is_not_witnessed 6 fail -> 6 pass; test_a_closures_symbol_is_never_resolved
3 fail -> 3 pass; TestIRContexts 2 fail + 1 error -> 3 pass.
Gates (patched; every failure also fails on base in this environment):
| file | T2 | T3 | base / known |
|---|---|---|---|
| aten | 1675, 1 error | 1669, 1 error | the error = rms_norm native override: `No module named 'cutlass'` (fails on base here) |
| replay | 247 OK | 247 OK | |
| tape | 43 OK | 43 OK | |
| triton | - | 59 OK | |
| guards | - | 12 OK | |
| harvest | - | 220, 5 F + 1 E | the 4 known Hopper cuBLAS tests + test_persist_padding x2 (both fail on base here) |
| oob, opaque, capture, fullgraph, lower, lower_tape, cover, memory | OK | OK | OK |
| python_entry | 511, 5 F + 1 E | 511, 5 F + 1 E | identical 6 on base (harvest python-entry twins of the above) |
Trace time (repro/bench_trace.py, median of 15 traces, 4 interleaved rounds; GH200 is the correctness box, so rough):
base 306-314 ms, T2 307-313 ms, T3 305-316 ms: no difference beyond noise.

### Odd line (int6 patches), GB300 gb300-box GPU 1 (one gpu1.lock hold), logs/gb_odd, logs/T2o, logs/T3o
Repros: F2 2/4 wrong -> 0; t_constlists 6 MISMATCH -> 0; t_listargs2 6 -> 0; t_factories 0 -> 0 (traced);
F3 872/1000 wrong x2 -> OK; t_unit_closure driver "invalid argument" -> OK.
New tests: test_a_size_or_integer_argument_is_not_witnessed 6 fail -> 6 pass; test_a_closures_symbol_is_never_resolved
3 fail -> 3 pass; TestIRContexts 2 fail + 1 error -> 3 pass.
Gates: T2_o aten 1673 OK, tape 43 OK, replay 185 OK. T3_o triton 59 OK, guards 12 OK, replay 185 OK, tape 43 OK,
harvest 219 OK.

### Behavior changes to know
- A Triton launch whose grid (or a dispatch_unit body) closes over a traced SymInt no longer redispatches at a
  specialization flip: the redispatch is refused ("a guard on a symbol of another trace") and the call retraces
  (F3: 3 traces over its 4 calls instead of 1 trace + 2 wrong redispatches). Mapping a variant-owned closure symbol
  through the variant's own bindings would keep the redispatch; it needs the dispatch memo's key to include closure
  reads and the redo's hint-dependent decisions audited, so it is left as a follow-up.
- Untagged ops `_elementwise` used to hand the witness and that are not in `_UNTAGGED_ELEMENTWISE` now trace their
  body (copy family, factories, _to_copy) or run as a reported eager step (flip, all/any, vector_norm,
  matrix_sqrth, huber_loss at shapes where the witness used to take them). Not measured on the model suites.
