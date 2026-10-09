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
