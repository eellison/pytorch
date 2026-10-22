# Core tasks from vLLM arm V/VP (candidate 2 + pinned lane), for the "core done" bar
Bar: stock vLLM traced with only the hook (model.forward -> replay entry, traced region over the prep), bitwise vs stock
vLLM eager on Qwen3-8B, no core-gap workaround in the adapter, the same kernels stock vLLM runs.
Each item: the repro, what core does today, what the fix should be. Paths are under land/scratch/integration/vllm.

## 0. (top priority, silent wrong output) run_buffer plan after a respec
See probe/runbuffer_nan_note.md. vLLM repro (1 min): fork + the silu_and_mul port, decode bs 5/8 -> NaN with
memory="run_buffer", bitwise with "eager". The respec'd variants' o_proj mm launches (seqs 34/35) come before the
allocations they use (seqs 37/38: output and the 32 MiB split-K scratch); plan_memory sees those intervals inverted or
starting late and reuses the bytes. Standalone attempt probe/runbuffer_nan_repro.py (respec via a dispatch_unit choice,
incl. a 1-node -> split-K mm topology change) renumbers consistently and does not trigger it; the vLLM case appends
re-created allocations (725 -> 1083) and orphans the old ones. Fix: respec/_splice must give a re-done or rebound op's
allocations seqs before its launches (or plan_memory must treat first use before seq as an error, not an interval).
Also worth a core assert: plan_memory refuses a temporary whose first user precedes its seq.

## a. aten.gather / aten.index.Tensor / int64 sum run eagerly (VP sampler tail)
Repro: probe/eager_ops_probe.py (torch only; bs 8, 149 vocab blocks). Today (candidate 3, gh200-b):
- gather: `an eager step in a variant: aten.gather.default (no traced implementation)`
- index: `an eager step in a variant: aten.index.Tensor (no traced implementation)`
- int64 sum: `aten.sum.dim_IntList's traced host declines: a Long sum`
Fix: traced hosts for gather (vLLM's gumbel_sample end) and index.Tensor; integral dtypes in the sum host. The symint
lane converts these. Adapter workaround to drop afterwards: _vp_gumbel's float32 one-hot select.

## b. Triton specialization of vLLM's block-table gather (num_reqs) and the indptr kernel
Today the adapter re-jits vLLM's _gather_block_tables_kernel with do_not_specialize=["num_reqs"] and uses a
do_not_specialize copy of its indptr kernel (_kv_indptr_fixed), only to cut variants (the == 1 / % 16 / alignment
specializations are guards in core). Per the user, the arms must run the stock kernels: ARMV_NO_REJIT=1 is now the
default. Measurement (pinned build, which records retrace_causes; range sweep with --check):
- stock kernels (out/c2_pinned_norejit_range.json): 969/969 bitwise, traces 4/5/9 = 18, 15 variants, 0 boundaries;
  redispatches per entry prefill/T2 14, mixed/T1 3, mixed/T2 2, prefill/T1 2, decode/T1 1.
- re-jitted (c2_pinned_rejit_range.json): 969/969, the same 18 traces / 15 variants; redispatches 2+1+2 = 5.
- So every Triton specialization flip of the stock kernels is a redispatch owned by that launch (17 more redispatches over
  the whole range, no extra trace). Step time (t_norejit_r* vs t_rejit_r*, medians of 2): decode 1/5/8/64/100
  4.35/4.27/4.41/4.97/5.58 vs 4.32/4.27/4.35/4.91/5.56 ms; prefill 64/512 5.19/7.74 vs 5.01/7.86; mixed 6.05 vs 6.15.
  Within noise: the stock kernels stay, nothing needed in core for this item.
- Retrace causes seen in the range (same in both): none is a Triton specialization. Three classes, each a separate item:
  1. ('dispatch', aten.copy_, (n - 1) == 0, adapter.py:437, "a memcpy in aten.copy_.default"): the adapter's own copy of
     combine_sampled_and_draft_tokens' output into its buffer flips between a memcpy and a copy kernel at n == 1, and a
     memcpy cannot be folded/redispatched (FoldRefused "a memcpy in", _host_trace_lower_tape.py:880), so it retraces.
     Core: let a redispatch/fold carry memcpy records. (The copy itself is adapter structure; see e.)
  2. ('graph', None, (T - 8192) < 0, vllm/v1/worker/gpu/block_table.py:222): `slot_mappings[:, :num_tokens_padded]` at
     T == width returns the tensor itself, so the slice guards T < width. Core: a symbolic full-width slice should not
     guard (same storage either way); today ARMV_WIDE_SLOTS (off) widens the buffer instead.
  3. ('graph', None, (n - 1) == 0, vllm/utils/torch_utils.py:145): `if 1 not in t.shape` in vLLM's stride canonicalizer:
     Python control flow on a size in vLLM code; legitimately graph-level (a variant per class).

## c. vLLM's five _C ops (rms_norm, fused_add_rms_norm, rotary_embedding, silu_and_mul, reshape_and_cache_flash)
Today: closed = harvested externs (correct, bitwise; a learn per new key, 6619 learned keys over the range sweep, most of
them per-T); open = Python launch ports (armV vllm_ops_trace) that re-derive each launch (grid T, block from hidden /
vec, the T>128 / T<256 switches) and check it against a captured launch (witness).
Without ports, two paths:
1. Harvest generalizing over sizes: the five launches are one kernel each whose grid is a size (T rows) and whose params are
   pointers plus constants, with one or two size switches (rms_norm block at T<256, silu's 256-bit template at T>128,
   T==1 views). A harvest keyed on the launch config (function, block, smem) with grid and size scalars as patched
   symbolic slots, and a guard at each observed switch, would serve every T from a couple of learns. Same mechanism as (d).
2. Trace the C++ host: needs a recording path for C++ host code (vLLM's .so computes grid/block in C++ and calls
   <<<>>>). Core has traced ATen hosts (aten/src/ATen/cuda/host_trace/Ops.h) but nothing for third-party C++; without
   rebuilding vLLM, only (1) applies.

## d. Closed trtllm-gen decode keyed on max_seq_len
Today: vllm_ht.trtllm_decode is a harvested extern; its key's scalars include max_seq_len, which grows every step, so
stock semantics (ARMV_DECODE_MAX_SEQ=actual) means a new key (a learn) per step; the adapter pins max_seq_len to
max_model_len instead (changes the split vs stock: not bitwise vs stock at bs < 64).
What max_seq_len changes in the launch (from the open launcher port, flashinfer/trtllm_trace.py, which mirrors the
host code of the closed op; the cubins are the closed part): for a generation (decode) call, max_kv enters only the
multi-CTA split count (ctas_kv -> grid z / the reduction) and two kernel-param fields (max_kv @1176, ctas_kv @1184 in the
packed params; the file's own comment: "max_kv enters a generation call's split count and its params only"). For a
context (prefill) call the tile size also takes it through a float cost model.
So for decode the harvest key should be the launch configuration (function, grid, block, smem, ctas_kv), with
max_kv / ctas_kv as patched param scalars, not the value of max_seq_len: a few keys per split region instead of one per
step. (A direct dump of two closed bindings in one split region needs a run that harvests at two max_seq_len values: the
exact-key run never repeats a key, so nothing was learned; TODO with a fixed-batch repeat.)
Open path measured: ARMV_DECODE_MAX_SEQ=sym (stock max_seq_len as a symbolic int into the fork's launcher):
drive 228/228 and range 969/969 bitwise vs stock vLLM eager, 19 traces / 19 variants, 0 boundaries (pinned policy: 19 / 20).

## e. vLLM's stock attention-metadata builder in the trace
Today the adapter does not trace vLLM's FlashInferMetadataBuilder.build: it copies a per-key template of vLLM's metadata
and recomputes the tensor fields in the trace (block-table gather, slot mappings, a Triton indptr kernel in place of
vLLM's host np.cumsum + pinned copy). Decline text from building vLLM's metadata inside the trace (ARMV_MD_STOCK_DIAG):
[results below when in]. Known from the code: the builder reads host state (num_decodes etc. via
split_decodes_and_prefills on CPU arrays, .item(), seq_lens_cpu / numpy cumsum into paged_kv_indptr.np, then a
pinned -> device copy). Needed in core: host steps for the numpy/CPU-tensor arithmetic (re-run per call, as the replay
already does for CPU tensors it tracks) and pinned copies (now in the pinned lane), so the stock build (same host math,
same copy) can live inside the entry; then the template and _kv_indptr go away.

## f. Escaping error instead of a decline (compiled forward)
vLLM's compiled forward (compile ON, cudagraph NONE) under the entry raises `RuntimeError: Cannot call numel() on tensor
with symbolic sizes/strides` out of entry(*args) instead of declining (logs/c2_Vc_trap2.log). Fix: a numel()/sizes read on a
traced tensor from Python must decline cleanly (the Vc arm itself is dropped).

## Kernel swaps in the adapter (the same-kernels rule)
- _gather_block_tables_kernel re-jit (b): removed by default (ARMV_NO_REJIT=1).
- _kv_indptr (Triton, adapter's own) in place of vLLM's host np.cumsum + copy (e).
- VP: _vp_gumbel's final gather replaced by a one-hot select (a); the logits/sampler/post_update kernels otherwise
  vLLM's own launches (gumbel kernel, get_num_sampled_and_rejected, post_update).
- trtllm_shim (closed) wraps vLLM's FlashInfer trtllm calls in dispatcher ops: same cubins. The fork (open) launches
  the same cubins through a Python port of the launcher.
- Ports of the five _C ops (open): same kernels, ported host (c).

## Deliberate engine-change arms (stay as labelled arms, not core gaps)
hostcuts (VAf and V's host cuts: one pinned staging copy, fused request-state writes, sampler dirty flag), VP (forward +
sampler + post_update in one entry; runner.sample / postprocess_sampled hand back its outputs), VP_UVA (measurement
stopgap; superseded by the pinned lane's memcpy nodes, ARMV_VP_PINNED), ARMV_PAD (A/B only), ARMV_WIDE_SLOTS (off).

## g. Retrace causes that should be redispatches (standalone repros, probe/retrace_class_repro.py, candidate 3 step246)
(i) memcpy refused for fold/redispatch. memcpy_flip: `out[:n].copy_(x[:, 0])` -- the strided column is contiguous only at
n == 1, so the traced copy_ host records a memcpy there and a copy kernel above. Traced at n = 4, a call at n = 1 retraces:
retrace_causes {('dispatch', 'aten.copy_.default', '((s61 + -1) != 0)', 'retrace_class_repro.py:17',
'a memcpy in aten.copy_.default'): 1}; traces 2, redispatches 0, all calls bitwise. Today: FoldRefused("a memcpy in
...") at _host_trace_lower_tape.py:880 (fold) / redispatch refuses, so the op-owned flip costs a whole-graph trace.
Fix: memcpy records (now with an H2D/D2H/D2D kind) in fold/redispatch tables like kernel records; the copy_'s flip then
re-dispatches that one node. In vLLM: the adapter's copy of combine_sampled_and_draft_tokens' output (adapter.py:437).
(ii) full-width slice guard. full_width: `buf[:, :n]` with buf width W = 64; traced at n = 8, a call at n = 64 retraces:
retrace_causes {('graph', None, '((-1*s13 + s75) < 0)', 'retrace_class_repro.py:22', None): 1}, i.e. the guard n < W at
graph level; all calls bitwise. What is guarded: the slice's end clamp, end = min(n, W) (and aten.slice's shortcut that
returns self for a full-width slice); the result is the same view of the first min(n, W) columns either way. Fix: a
symbolic Min(n, W) end, with no self shortcut under a trace (always a view), so no guard. In vLLM: block_table.py:222
`slot_mappings[:, :num_tokens_padded]` at T == max_num_batched_tokens (ARMV_WIDE_SLOTS was the adapter workaround, off).
(iii) vLLM's `if 1 not in t.shape` (utils/torch_utils.py:145): Python control flow on a size in vLLM code; legitimately
graph-level, no core change.

## e. update: the stock builder declines cleanly (sound)
With ARMV_MD_STOCK=1 (vLLM's FlashInferMetadataBuilder inside the trace), every trace declines at the builder's first host
read: `aten.slice.Tensor of an untraced tensor with symbolic arguments` (_host_trace_tape.py:583) at
vllm/v1/worker/gpu/model_states/default.py:196 `max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()`; all
forwards ran eagerly (66 of 122 checks differ from the max_seq-pinned reference, all bitwise vs stock: the eager
fallback runs stock max_seq_len). No host value reached a graph. CPU work is unsupported by design, so the stock-V
configuration runs vLLM's builder eagerly before the region (arm V-stock, ARMV_STOCK=1) with the metadata's tensors as
arguments and its ints lifted as symbolic arguments; the template stays as the labelled prep-in-graph arm.

## V-stock (criterion 5 configuration), pinned build (land/core/pinned, step242 + build_cpp8)
Definition (ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual, no
ports for the _C ops, ARMV_NO_REJIT=1): vLLM's input prep and FlashInferMetadataBuilder run eagerly as stock; the traced
region starts at model.forward; the metadata's device tensors are arguments and its ints (num_actual_tokens, num_decodes,
num_decode_tokens, num_prefills, num_prefill_tokens, decode.max_seq_len, prefill.max_q_len / max_seq_len) symbolic int
arguments; one entry per kind and metadata form. The five _C ops and cuBLAS are harvested (closed, no ports).
Results:
- drive --check (decode 1/8/64, prefill 64/512/2048, 4x128 + 8x128, mixed): 188/188 bitwise vs stock vLLM eager (stock
  max_seq_len, so this is the stock comparison); traces 4, variants 4, 0 boundaries.
- range --check: 958/958 bitwise; traces 3/0/6 = 9, 9 variants, 0 boundaries; 5995 keys learned (the closed _C ops and
  cuBLAS per size); redispatches prefill 39, decode 2, mixed 25; retrace causes: ('graph', utils/torch_utils.py:145
  size-1 check, legitimate), and in mixed the fork's split choice ('dispatch', trtllm_paged_attention_decode,
  min(ceil(max_kv/512), max(19 // ..., 1)) <= 1, "dispatches otherwise" / "another launch topology") x4.
- decode step ms, same build (median of 2): synced 5.10/5.06/5.92/6.87 (bs 1/8/64/128) vs default 4.56/4.59/5.11/5.98 and
  V (template) 4.17/4.36/4.85/5.97; async 5.70/5.90/7.77/8.38 vs default 5.25/5.75/6.93/8.74 and V 5.10/5.03/6.17/8.12.
  Prefill 64/512: 6.71/9.61 vs default 37.25/37.52, V 5.18/8.07. Mixed 8x128: 300-375 ms per step (see below).
Remaining non-hook replacements V-stock needs (the remaining core work for criterion 5):
1. The trtllm-gen fork (a Python port of FlashInfer's launcher over the same cubins) for attention: the closed op is keyed
   on max_seq_len (d), a learn per step with stock values. Needs (d): key the closed launch on its configuration, with
   max_kv / ctas_kv patched.
2. Attention layer use_direct_call = True: with vLLM's own custom-op call path (ARMV_DIRECT_CALL=0) the fork's kernels
   decline, "trtllm-gen context: fmha...Context was not loaded at the warm-up (a capture holds)" (and the same for decode),
   so attention becomes 36 eager steps per variant (still 109/109 bitwise). Needs: the kernel module loaded outside the
   capture (at the eager warm-up of the custom-op path too, or a load-and-retry outside capture), not a decline.
3. Mixed steps with stock max_kv: the context launcher pins max_kv (its tile choice goes through a float cost model), so
   every mixed step redispatches that op at a new max_kv: 1 trace, 12 replays, 5 redispatches over 4 timed mixed steps,
   each ~300 ms. Needs: a guarded (not pinned) tile choice, or a cheap redispatch for this op; same root as (d) for the
   closed op.
4. The five _C ops: harvested per size (learn cost, not a replacement); (c) would remove the learns.
5. Not replacements: the hook (runner.model.forward -> entry), _reparametrize_module (parameters passed as entry
   arguments inside the hook), the drive's own timing wrappers.

## h. (serve lane) Hybrid NVFP4 models: R1 Qwen3.8-27B, Qwen3.5-35B-A3B (repros under ../serve/repro, log ../serve/STATUS.md)
R1 under V-stock (pinned build, ../serve/armV_stock = armV_cand2 10-08 15:00 + the hybrid metadata path: every KV-cache
group's metadata built by vLLM eagerly, its CUDA tensors as arguments, its ints symbolic): 12 traces, 0 variants, every
step eager. Checks (eager path vs stock eager on the same metadata, recurrent state included) bitwise except prefill rows
where stock eager itself differs run to run (vLLM's FlashInfer GDN prefill, see below).
(i) harvest refill on float8 (decline of every R1 decode trace under V-stock). HarvestProvider._harvest's refill fills a
written floating operand with uniform_(-1, 1); float8 has no uniform_, so NotImplementedError ("check_uniform_bounds"
not implemented for 'Float8_e4m3fn') escapes as a trace decline (and the next call traces again: 11 'contract' retraces).
vLLM: _C.static_scaled_fp8_quant (fp8 query for trtllm attention, fp8 KV) harvested as an extern.
Torch repro: ../serve/repro/core_gaps_repro.py gap_2 (torch.library op writing float8, extern provider): pinned and
candidate 3 both decline exactly so. Fix: fill float8 through a float32 temp (or random bytes masked to finite codes); a
harvest exception refuses the key (eager step), never declines the trace.
(ii) eager step with an in-trace address scalar. FlashInfer's mm_fp4 cute-dsl GEMM (vLLM's default NVFP4 linear,
FlashInferCuteDslNvFp4LinearKernel) takes its scale factors as cute make_ptr pointers, i.e. data_ptr() ints. The CuTe DSL
route declines "argument 6 is a TVM-FFI DataPointer parameter"; the eager fallback's int argument is 256*s<allocK.base/256>,
which the eager-call lowering cannot express ("sN is not read from the call's inputs", _TapeLowering.eager_call ->
ScalarSlot), so the trace declines instead of keeping one eager step.
Torch repro: core_gaps_repro.py gap_1b (an op with a Python CUDA kernel taking t.data_ptr() of an in-trace allocation;
returns x+1 iff it gets x's address): today declines at the trace-vs-warm-up check ("call 2 is at_address(..., 5359283625290366976
[the placeholder]) in the trace; at the warm-up at_address(..., 281442726707200)"). FlashInfer repro (vLLM's kernel class,
random FP4 weights, no engine): ../serve/repro/repro_fp4_foreign.py (+ run_fp4.sh), declines at lowering.
Fix options: (a) the CuTe route takes DataPointer parameters (the pointer is a traced tensor's address: lower it as a
pointer slot, as Triton pointer arguments are); (b) eager steps accept address scalars of traced tensors (resolved to the
replay's address). (a) gives a graph, (b) a boundary instead of no variant.
Integration (not core, done in ../serve): observe cute.compile from worker init (ARMV_CUTE_OBSERVE) and a FlashInfer
workspace without objects compiled unobserved (../serve/fi_ws_ht); without them the GEMM is a foreign TVM-FFI call.
(iii) not ours: vLLM's FlashInfer GDN prefill (recipe default on sm10x) gives degenerate/NaN output on ~1/3 of single
requests in stock default and eager, varying run to run; Triton GDN prefill gives none. Checks on R1 are NaN-aware.
(iv) Qwen3.5-35B-A3B stock defaults need FlashInfer trtllm-gen cubins not on disk (MoE batched_gemm for the default
flashinfer_trtllm bf16 MoE; fp8 fmha ...P16VarSeqQ128Kv128PersistentContext); no-download run uses --moe-backend triton
--attention-backend TRITON_ATTN (results in ../serve/STATUS.md).
(v) side-stream work inside the region (Qwen3.5-35B-A3B, vLLM's MoE shared experts on their own stream inside the
vllm.moe_forward_shared custom op): "its kernel declines: aten.mm.default on a stream other than the trace's" -> an eager
step per MoE layer (40 boundaries per decode variant; decode otherwise replays: 239/241 steps, bitwise incl. state).
Torch repro: ../serve/repro/core_gaps_repro.py gap_3_inline (side stream + wait_stream both ways) / gap_3_op (same inside a
torch.library op). Want: fork/join edges in the graph (the side stream's work captured on its own branch).
(vi) unpinned CPU tensor arguments: vLLM's GDN prefill metadata carries unpinned host tensors (per-request chunk counts);
as arguments: "argN is on cpu; only CUDA tensors on one device and pinned CPU tensors are traced"; in the key: every
prefill length mix is a new key (first call eager). Torch repro: core_gaps_repro.py gap_4. Want: host inputs re-read
per call (the host steps of item e).

## h. Replay entry host cost before launch (decode gap vs vLLM's own full graph)
Decode bs 128 at KV ~7K (Qwen3-8B, attn line): V-stock and default-nc run identical kernels (same trtllm-gen Persistent decode
variant, same cuBLAS nvjet picks, GPU busy 22.2 vs 22.1 ms). V-stock is 1.0-1.4 ms slower per step because the replay entry spends
782 us (profiled) between the model forward call and cudaGraphLaunch, against 106 us for vLLM's replay:
- the adapter's Python form checks, about 150 us;
- 36 cudaGraphExecMemsetNodeSetParams per replay, 108 us (the memset nodes are re-parameterized on every call even when nothing changed);
- about 520 us unattributed in the C++ entry (argument binding, guards, node updates).
Candidates (core):
(1) skip a node update when its parameters equal the exec's current ones (the memset nodes here);
(2) attribute and shrink the C++ entry's per-argument work at bs 128 (record_function ranges or counters inside the entry);
(3) move the adapter's per-call form checks off the hot path (key the bound entry once per form).
Profile: integration/vllm/out/prof_dec128/, prof_dec_host.py.
(h) r1gaps lane notes (10-08 22:00, land/core/r1gaps/STATUS.md): (i) fixed in r1gaps (float8 fill in the harvest refill
and in HostTraceReplay._learn, which had the same uniform_). (ii) part 1 fixed (CuTe route takes TVM-FFI DataPointer
args as pointer slots); mm_fp4's block-scaled host function still needs CuTe-route coverage (TMA U4 / hierarchical U16
SF TMA, tile_to_shape, ...; list in core/cute70/FROM_R1GAPS_DATAPOINTER.txt), and its eager fallback's address int is gap
1b (core/r1gaps/DESIGN.md, awaiting review). Qwen3.8-Flash-Next does not use mm_fp4 (dense linears are bf16 in its
quant config; NVFP4 = routed experts only), so (ii) is R1-only.

## i. A redispatch's custom-op body cannot read module / forward-context state (vLLM use_direct_call=0)
vLLM V-stock with ARMV_DIRECT_CALL=0 (land/core/attn out/r7_vstock_nodirect.json) is 97/97 bitwise, but it has one one-off decline:
  `flashinfer_ht.trtllm_paged_attention_decode.default of a tensor the trace does not track`
It happens at a redispatch of vllm.unified_attention_with_output, whose op-owned guard is utils/torch_utils.py:145
`1 not in t.shape`. The call then traces again.
- Why: the op body reads state that is not among its arguments, through get_attention_context(layer_name):
  - the layer's kv_cache. This is the first untracked argument of the decode op (q, then k/v = kv_cache);
  - the trtllm workspace (a module global);
  - the forward context's decode metadata (block_tables, seq_lens).
  The adapter lifts all of these per call: it swaps the step's traced tensors into the modules and the forward context while the
  traced step runs (_swapped(attrs), override_forward_context, _reparametrize_module). A redispatch re-runs the op body alone
  ("no Python of the traced function runs"), so the body sees the real tensors.
- The adapter cannot express this. Its lifting acts only around the traced function, and the redo gives the adapter no hook or
  access to the redo's fresh traced stand-ins of the entry's arguments. Making the state an op argument would be an engine edit.
- Torch-level repro: integration/vllm/probe/redo_module_state_repro.py (CAND_LINE=attn, GPU 0, seconds). A custom op's body reads
  STATE["kv"]; the traced step swaps STATE["kv"] to its argument while it runs; an op-owned host branch on q.shape[0] == 1 keeps the
  same kernels on both sides:
  - module state: n = 4,4,4,1,... gives traces 2 and redispatches 0. The redo raises
    `aten.slice.Tensor of an untraced tensor with symbolic arguments`; redispatch_refusals {'repro_redo.attn.default dispatches
    otherwise': 1}; retrace_causes ('dispatch', ..., '((s75 + -1) != 0)', ...).
  - control, the same op with kv as an argument: traces 1, redispatches 1, no decline, no retrace. Bitwise everywhere.
  (Control without the add: out.copy_ at n == 1 is a memcpy, and redispatch refuses it ("a memcpy in ..."); that is the class
  already listed in g.)
- Fix options (core):
  (1) at a redo, resolve an untracked tensor that is (a view of the storage of) one of the call's arguments to that argument's
      fresh root. Here the KV and workspace are entry arguments (the static prefix); the metadata tensors are per-call arguments.
  (2) let the traced function register the attribute bindings (owner object, attribute name, argument index) that the trace saw,
      so the redo re-applies them with its fresh stand-ins.
(vii) malformed variant spec: a selector's nodes include a nested keyed site's node (Qwen3.8-Flash-Next under V-stock;
engine-killing). vLLM's QSA attention custom op vllm.qwen4_exp_qsa_with_output is traced through as a top-level op with
its own guards (a selector over its 7 launches); the KV-cache write inside it, _C_cache_ops.reshape_and_cache_flash, is a
harvested extern (a keyed site of 1 node). Both claim the same launch record (174, 251, ... one per QSA layer), so
torch._C._HostTraceVariant(spec) raises ValueError "a site's record 174" (VariantBuild.cpp:196) and
_host_trace_native.native_variant re-raises it as AssertionError("host_trace: the native variant rejects: a site's record
174"), which escapes the entry call and kills the engine. Two bugs: the lowering's spec (a launch owned by a keyed site must
not also be in an enclosing op's selector, or the selector must defer to the site), and the escape (a native rejection
should decline the trace, not raise out of the call).
Torch repro: ../serve/repro/core_gaps_repro.py gap_5 (a torch.library op whose Python CUDA kernel runs a harvested mm,
blas provider, then branches on x.shape[0] > 16): pinned build -> "escaped AssertionError: host_trace: the native variant
rejects: a site's record 1" at the first trace. Probe diagnosis in ../serve/logs/fn67.log (SPECFAIL line).

## From the symm lane (2026-10-08 ~23:00 PDT; for TP2 later hardware, not urgent): one adapter attribute entry
TP2's all-reduce on torch symmetric memory (config: `--disable-custom-all-reduce`, `VLLM_ALLREDUCE_USE_FLASHINFER=0`) runs
SymmMemCommunicator.all_reduce: `self.buffer[:n].copy_(inp)`, `torch.ops.symm_mem.two_shot_all_reduce_(self.buffer[:n])`,
copy out. With the symm lane's core patches (land/core/symm/READY.md: memcpy-kind fix + the symm host code on the symti
route) all three are graph nodes, provided the buffer is a traced argument: the collective's rendezvous is looked up from
the real argument behind the traced input and guarded on its address. The adapter's attribute lifting (ArmV.attrs ->
prefix, bound by _HostTraceBound) does not reach it today (not a model module attribute, not a listed global). Ask: one
more entry, the same pattern as fi.trtllm_workspace_buffer, when the TP group has one:
  comm = vllm.distributed.parallel_state.get_tp_group().device_communicator
  if getattr(comm, "symm_mem_comm", None) is not None and not comm.symm_mem_comm.disabled:
      self.attrs += [(comm.symm_mem_comm, "buffer", comm.symm_mem_comm.buffer)]
No engine edit. Without it the collective is an eager step (declined by name: "not a view of an argument"). vLLM-side
limits stay: the 4 MiB cap (SYMM_MEM_ALL_REDUCE_MAX_SIZES["10.3"][2], larger goes to pynccl) and the communicator's
multicast requirement (to check on 2 ranks).
(h, vi) R1 prefill's "argN is on cpu" identified (r1gaps, 10-09 00:10; armV_r1 ARMV_HOST_ARG_PATHS diagnostic, log
core/r1gaps/logs/r1_v3cpp.log): the only unpinned host tensor among the metadata arguments is
GDNAttentionMetadata.nums_dict[8]["nums"] (int32, one per prefill request: ceil(seqlen / 8); e.g. (13,) for a 100-token
prompt). vLLM: GDNAttentionMetadataBuilder.build (v1/attention/backends/gdn_attn.py) -> compute_causal_conv1d_metadata
(v1/attention/backends/utils.py:1042) builds, from query_start_loc_cpu: nums = -(-seqlens // 8) (seqlens =
query_start_loc_cpu.diff(): unpinned), tot = nums.sum().item(), mlist / offsetlist (PINNED, np_to_pinned_tensor /
pin_memory), mlist_len, and copies mlist / offsetlist non_blocking into the CUDA batch_ptr / token_chunk_offset_ptr.
What the forward does with it: nothing. causal_conv1d_fn (model_executor/layers/mamba/ops/causal_conv1d.py:560-700) with
metadata reads only nums_dict[8]["tot"] (a host int -> the Triton grid's first axis; lifted as a symbolic int by V-stock)
and the CUDA batch_ptr / token_chunk_offset_ptr; nums, mlist and offsetlist are read only when batch_ptr is None (never
with the builder's metadata). No host decision and no H2D copy in the forward depends on them; the H2D copies happen in
the builder, eagerly, before the region. The decline comes from arm V-stock passing every metadata tensor as an
argument (ARMV_HOST_ARGS=1); with ARMV_HOST_ARGS=0 their values key the call instead (every new length mix is a new key,
first call eager: s2c).
Torch-level repro: core/r1gaps/repro/host_arg.py (f(x, nums, tot, batch_ptr) with an unpinned CPU tensor it never
reads). Design options (not implemented): (a) core: an unpinned CPU argument is accepted unread and declines (or keys by
value) only at its first host read, so an untouched host argument costs nothing and CPU work stays unsupported;
(b) adapter: pass only host tensors the forward reads (needs a per-field rule, fragile); (c) adapter: keep them out of
args and out of the key (unsound if a later vLLM forward reads them).
  host_arg.py result (r1gaps attn tree, 10-09 01:37): traces 3, replays 0, variants 0, bitwise, decline "arg1 is on cpu; only CUDA tensors on one device and pinned CPU tensors are traced".

(viii) (serve lane, 10-12) Declined traces retry at every new exact class. A structural decline (for example "argN is on cpu; only
CUDA tensors ... and pinned CPU tensors are traced") is recorded per exact class (HostTraceReplay._declined.add(structural), where
structural = exact unless trusted). In vLLM mixed steps every step is a new exact class, so Qwen3.5-9B V-stock (q1) paid 377 full traces
in one run, each followed by an eager call (retrace cause 'contract' x377). The adapter-side fix for q1 is ARMV_HOST_ARGS=pinned. The
policy question for core: should a decline whose cause does not depend on the class's sizes (a CPU argument, a side stream) mark the
contract?

(ix) (serve lane, 10-12, fs6) An eager step that hands Python state to traced code. vLLM's SharedExperts (fused_moe/runner/
shared_experts.py) has forward() store its result in self._output[...], and a later reader takes it through the .output property and
resets it to None. In Flash-Next's prefill (Qwen4Exp + qwen3_next MoE, V-stock with the _m adapter, once prefill traces) a prefill
trace/replay leaves _output set. The next eager forward then asserts `self._output[self._output_idx] is None`
(shared_experts.py:182). The store happens inside an eager step (vllm.moe_forward_shared) and the read is host code that the replay
does not rerun, or the reverse, so the replay does not reproduce the Python-side handoff. Decode is not affected: 238 replays, checks
bitwise. A torch-level repro is still to write: a custom op whose Python body sets a module attribute that traced code reads and clears.
Interim (adapter): ARMV_EAGER_KINDS=prefill,mixed for Flash-Next.
(h, vii) "an operation on symbols of two traces" escaping a step (r1gaps 10-12, logs core/r1gaps/logs/v9meta.raw; the CuTe lane
saw the same in vllm.flashinfer_mm_fp4's dispatch): FlashInfer's AutoTuner.choose_one (flashinfer/autotuner/autotuner.py
~1680-1687), on a shape no tuned config covers, builds warn_key = (custom_op, signature-of-input_shapes) and keeps it in the
process-global set self._logged_cache_miss_oor. Under a trace the shapes are SymInts, so a trace's SymInts are stored in library
state; a later trace's `warn_key not in self._logged_cache_miss_oor` compares its SymInts with the stored ones ->
_host_trace_ir._of raises Declined("an operation on symbols of two traces"), here from _call_native -> custom op backend_impl
-> flashinfer_mm_fp4 -> mm_fp4 -> choose_one, and it escapes the engine step (crash), not a counted decline. Hit with R1
prefill now tracing (ARMV_HOST_ARGS=meta) and mm_fp4 at many M. Core question: library global state retaining trace SymInts
(and the escape from the native-call path). Not fixed in r1gaps.

(x) (r27 lane, 10-12) aten.index.Tensor / aten.index_put_ on the GDN state pool run as eager steps (symbolic TensorIterator
lane). Qwen3.8-27B NVFP4 V-stock with gdn_prefill_backend=triton (land/core/r27, vt1-3): every prefill / mixed variant has 96
eager steps / 97 segments = 48 GDN layers x (aten.index.Tensor + aten.index_put_.default) from vLLM's
`initial_state = ssm_state[prefill_state_indices]; initial_state[~prefill_has_initial_state, ...] = 0` and
`ssm_state[prefill_state_indices] = last_recurrent_state` (model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py ~1517-1536):
fp32 pool [N, 48, 128, 128] indexed by a CUDA int32 index tensor of the prefill requests and a bool mask. Decode variants are
already 0 eager / 1 segment. Want: traced hosts for index.Tensor (int index + bool mask) and index_put_ (int index) with symbolic
index length.
(xi) (r27 lane, 10-12) Mixed / prefill entries are keyed per request count: GDN metadata's unpinned host tensor
nums_dict[8]["nums"] (shape = number of prefill requests) becomes a meta stand-in whose shape is in the key
(ARMV_HOST_ARGS=pinned/meta), so every new request count is a new entry whose first call is eager (27B vt runs: ~28 mixed
entries of 1 call each). Separate from (x). Also open: a 1032-token mixed shape declines "sN is not read from the call's inputs".
(x) update (r27, 10-12 20:25): closed by the symti drop (land/core/symti install15 + wt314 = step314 + symti_py). 27B c5s
(land/core/r27/out/pm_c5s.json): prefill / mixed variants 0 eager / 1 segment, bitwise; Vc+VP decode (vLLM's own sampler) 0
eager / 1 segment (index.Tensor x3 + gather traced). One process aborted at interpreter exit ("_Py_GetConfig ... GIL is released",
rc 134) after writing its results: to the symti lane if it recurs.

(xi) (serve lane, 10-12) A cold Inductor autotuner first met under a host trace never converges. With the core's Inductor seams
(torch._inductor.cudagraph_host_trace._InductorKernel via vllm_ports/inductor_hosts.py, or the serve lane's earlier recorder), a
CachingAutotuner that still has several launchers declines with retry=True ("not autotuned"). In vLLM's compiled forward (Qwen3-8B Vc,
candidate 4, compile cache cold) the decline repeated at every call: run 1 of t_g4s_vc had 349 traces / 0 replays (decline
"triton_red_fused_fused_add_rms_norm_0: 2 launchers"), and the c5s q8vcs attempt had 428 traces / 0 replays. With a warm cache
(the tuned config on disk) the same runs replay: 7 traces, 367 replays. Expected: the declined call's eager fallback tunes the
autotuner (as the core's Triton Autotuner route relies on), and the next call traces. A standalone_compile repro (two artifacts, the
second first called under a trace) did not reproduce: its autotuners were already tuned at compile. vLLM-level repro:
serve/repro/run_vc_cold.sh (fresh VLLM_CACHE_ROOT / TORCHINDUCTOR_CACHE_DIR, DIAG_AUTOTUNE logs each autotune and each run's launcher
count, under a trace or not). Timing workaround: one untimed warm-up run per compiled arm.
  Update 10-13 04:25: both failing runs used the serve lane's first Inductor recorder (retired 10-12 ~20:31). With the core's seams
  (vllm_ports/inductor_hosts.py today), the cold-cache diagnostic on c5g (serve/repro/run_vc_cold.sh, fresh vLLM/Inductor/Triton
  caches, decode bs 1/8/64 + prefill + mixed 1024x64) replays: 3 traces, 62 replays. Every autotune ran outside a trace, at vLLM's
  warm-up, and no untuned autotuner was met under a trace. So this is not reproduced on the core path: likely the retired recorder's
  bug. Kept open (low) until a cold run of the full g4 workload confirms it.

(xii) (serve lane, 10-13) Per-replay patching costs ~2.8 us per changed kernel node, and in vLLM decode the changed set is one
node per layer. Qwen3-8B Vc + sampling, decode: 44 cuGraphExecKernelNodeSetParams per replay (52 KB of parameter images) at
bs 1 and at bs 64; patch phase 90-155 us of a 190-270 us entry (launch itself 13-22 us). 36 of the 44 are trtllm-gen attention
nodes whose ~1.2 KB by-value parameter block changes in one scalar (max_seq_len, eager's per-step maximum, +1 per step); 8 are
sampler kernels whose pointer arguments are vLLM's per-step InputBatch tensors. Mixed 1024x64: 78 node sets, patch 461 us of
757 us. With max_seq_len fixed (vLLM's capture convention) the same replay sets 8 nodes and the entry is 81-95 us. Asks: (1) is
2.8 us per node set the driver's floor on GB300 (cuGraphExecKernelNodeSetParams with a 1.2 KB block) or is part of it our
pack/compare path (packs are 116 per replay at bs 64 for 44 sets)? (2) a cheaper route for a node whose only change is a scalar
inside a large by-value block (e.g. keep a per-node device copy and patch through a memcpy node, if the kernel ABI allows it;
otherwise batch the sets). Repro: serve/repro/run_entry_prof.sh (ARMS=vcs_ch / vcsm_ch), serve/out/entry_prof/*.json
("ph_us", "patch", V.changed).
  Update 10-13 (coordinator, relaying the user): a device-updatable kernel node prototype (CUDA 12.4+ device-updatable kernel nodes, cudaGraphKernelNodeUpdatesApply /
  device-side node updates, so per-step scalars are written by a kernel instead of host node sets) is NOT being pursued now: the user
  isn't sure it's worth it, and decode is already at parity. Kept here as a possible later lever for the per-replay patch cost.
  Update 10-13 06:00, device-updatable kernel nodes, stage 1 only (stopped by the user's call: not convinced it is worth it).
  Standalone, the real trtllm-gen decode kernel (closed cubin) in torch.cuda.graph(keep_graph=True), 36 layer calls, GPU 2:
  serve/repro/devupd/{devupd_kernel.cu, devupd.cubin (nvcc -rdc + nvlink -lcudadevrt), devupd_probe.py, devupd_probe2.py}.
  - CUDA 13 / driver 580 accepts CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE (cuGraphKernelNodeSetAttribute) on the
    closed-cubin fmha nodes, whose single by-value parameter is a 1,344-byte block; a head kernel calling
    cudaGraphKernelNodeSetParam on the 36 nodes, with edges to them, makes them launch with the new bytes after cuGraphUpload.
  - max_seq_len is the int32 at byte 1176; byte 1000 is a per-capture scratch pointer (also differs between two captures at the
    same max_seq_len). bs 64 (Kv256Persistent kernel): 32/32 replays bitwise vs eager across max_seq_len 1800-8000, but this
    kernel's output does not depend on max_seq_len >= the longest sequence, so it is not a positive control. bs 1
    (MultiCtasKvCga kernel): max_seq_len also sets the split count (byte 1184) and grid x (1 at <= 300, 2 at 600, 4 at 1800,
    5 at 2049, 16 at 8000), and below ~600 and at 8000 the kernel function and shared memory change: a device route has to
    update grid on device (cudaGraphKernelNodeSetGridDim / FieldGridDim) and fall back to a host patch on a function change.
    The batch form (update records + grid on device, devupd_probe2.py) was not finished.
  - Cost: host, 36 cuGraphExecKernelNodeSetParams in isolation 49-55 us (1.4-1.5 us per node; the in-core 2.8 us per set
    includes pack and compare) vs ~1 us for one pinned write of the step's scalars; device, +4 to +10 us per replay with the
    head on the critical path (in a model graph it can run beside layer 0's work, edges only into the attention nodes).
  - Constraints from the CUDA docs: a graph with device-updatable nodes cannot be instantiated more than once or passed to
    cudaGraphExecUpdate, and host-side updates to those nodes need a re-upload before the next launch.
  Update (10-13 05:58): device-side updates (cudaGraphKernelNodeUpdatesApply from a head kernel) work on the trtllm-gen nodes
  (standalone, bitwise, serve/STATUS.md "Device-side node updates") but add +8-14 us of serial device time; deprioritized (host
  time overlaps the previous step, device time does not). The host route is the ask: (a) cut the per-node
  cuGraphExecKernelNodeSetParams cost (1.4 us standalone, ~2.8 us per set in the entry with pack + compare); (b) build records in
  C++ without re-packing the whole 1.2 KB image when one scalar row changed (the delta knows which leaves moved); (c) skip
  redundant work: 116 packs for 44 sets at bs 64.

(xiii) (serve lane, 10-13) A batch-size switch re-sets most of the graph. Qwen3-8B Vc + sampling, decode alternating A <-> B every
step (37<->50, 63<->65, 97<->100; vllm/round_sweep_line.sh, --alt-decode): 443 kernel node sets + 36 memory node sets per step
(of ~515 kernel nodes), patch 0.5-0.72 ms per step, step +0.7-1.0 ms over the same sizes held fixed (44 sets); vs vLLM default /
default-nc 0.90-0.99x (vLLM pads 4-5.5 rows and replays a captured graph, flat cost). Host-side levers to evaluate: the per-set
cost (xii), and keeping the last few shapes' instantiated execs per variant (an exec ring), so returning to a recent batch size
swaps execs instead of re-patching ~443 nodes. Data: vllm/out/t_swp_vcs_r1.json (c5g) and t_swph_* (c5h), ix alt_* rows ("patch",
"ph_us"); serve/out/churn/vcs_054937.json (GPU 2).
  Options (10-13, written only; the user holds all churn levers until the design is aligned; estimates from the c5g r1 rows):
  (1) Exec ring: keep the last K (2-4) instantiated execs per variant, each with its own held images; a step at a recently seen
      shape replays that exec and patches only what moved since its last use (~44 sets, as fixed batch). Alternating A<->B:
      patch 0.5-0.72 -> ~0.13 ms per step, step -0.4 to -0.6 ms (at or below vLLM in those rows). Costs: K execs' host memory and
      a first-use instantiation (ms); no help for a shape sequence that never repeats.
  (2) Cheaper sets: 1.4 us per cuGraphExecKernelNodeSetParams standalone vs ~2.8 us per set in the entry (pack + compare + set):
      C++ record building that packs only the moved fields could roughly halve patch time (443 sets: ~0.6 -> ~0.3 ms).
  (3) Max-size run buffer (allocate the run buffer at the variant's guard-range maximum): removes the pointer-only sets of
      temporaries; most of the 443 sets also change sizes or grids, so the saving is the pointer-only share (not measured; a
      held-image diff per switch, as ARMV_CHANGED does, would count it). Transient peak grows to the range maximum.
  (4) Grid bounds for size-dependent kernels: a batch change becomes an argument-only update, which avoids the successor
      penalty (~0.6-1.4 us device per re-gridded node's successor, paramgraph-exec-update-successor-penalty) but not the set.
      Alternating rows are +0.7-1.0 ms over fixed with +0.4-0.6 ms of extra patch, so up to ~0.3-0.4 ms may be device penalty.

(xii) (r27 lane, 10-13) RETRACTED (was: "re-trace stall after many step sizes"). The evidence was misread: the py-spy local `i: 1951`
belongs to another frame (it also shows in a later run whose loop never passes 1599), not to the learn-range loop. The 1-1600 learn
run with progress prints finished in 1506 s (61 traces, 3202 cuBLAS learns, about 1 s per size), so the 1-3072 run was most likely
just slow and would have hit its 3000 s timeout. No core issue is known here.

(xiv) (serve lane, 10-13) Our process carries ~900k unfrozen GC-tracked objects after warm-up (Qwen3-8B Vc + sampling: 898,854
tracked vs vLLM default's 6,305; vLLM's freeze_gc_heap() at engine-core init runs before our lazy traces / variants / replay
structures exist). A full collection then takes 0.66-0.74 s (vLLM ~5 ms), a gen-1 pass 0.14-0.38 s, and it lands wherever the
allocation thresholds trip: in swph1 one landed inside the ~1K-context bs-38 TPOT span in all 3 runs (mean 12-16 ms vs median
6.4). Options for the user: freeze after our warm-up (global state; where would the core do it?) or cut tracked objects in the
trace / replay structures. Data: serve/out/gc38b_{vcsl,default}_gpu3.json (drive_sweep DRIVE_GC_LOG=1).
  Update (xiv, 10-14 01:25): with gc.freeze() after our warm-up, tracked objects at measurement are ~8k (vLLM ~6k), but passes after
  any later trace still take 60-500 ms: gcdiag (DEBUG_SAVEALL + tracemalloc, vllm/out/t_gcdiag_vcsl_r1.json) shows 7 of 601 passes
  collecting 25k-72k objects each (tuple, cell, torch.SymInt, function, GenericAlias, host_trace_ir.IRSymNode, list, dict, Node),
  allocated by the tape recorder while tracing (_host_trace_tape.py:1825-1837 / 1810, :2774 from :485-487 / :507 via generated code,
  :726 via torch.utils._pytree closures, FakeTensor dispatch fake_tensor.py:1707 / 2512 / 3239). gc.garbage.clear() <= 3.4 ms: the
  cost is the cycle collection, not finalizers. Options: break the tape's cycles at trace end (real fix), or gc.collect(0/1) at the end
  of a trace (mitigation inside an already-slow call).

(xiv-b) (serve lane, 10-14, diag3) After the post-warm-up gc.freeze, one decode trace (a drain-tail decode variant at bs 4, long KV)
added 241k long-lived GC-tracked objects (3.5k -> 244.8k) and 25k cyclic garbage (1,427 SymInt / host-trace IR / fake-tensor objects
among them); every later gen-2 pass walks them (the swph1 300-600 ms one-step stalls). The replay path itself adds about 40-70 small
cyclic objects per step over vLLM's baseline (tuples, cells, getset_descriptors, GenericAlias; median 56-84 vs 17). Real fix (later,
core): a trace should not leave ~240k Python objects alive. Lane mitigation from cbh4 on: ARMV_REFREEZE=1 (opt-in adapter switch:
gc.collect(); gc.freeze() after a forward that traced or learned once the post-warm-up freeze happened). Data: vllm/out/t_diag3_vcsl_r2.json
(gc_steps), serve/ix/diag3_analyze.py.
