# Candidate 4 merge queue (coordinator-maintained; int6 merges in this order when candidate 3 is promoted)

| item | status | patches | notes |
|---|---|---|---|
| pinned H2D/D2H memcpy nodes | READY 2026-10-08 | land/core/pinned/pinned_py.patch (on step242, net +80), pinned_csrc.patch (build_cpp8, net +28), py/test/test_cuda_host_trace_pinned.py (31 tests) | GB300 + GH200 green; vLLM VP UVA-off: 0 outside, 100/100 bitwise. Adapter: land/core/pinned/vllm/armV_pinned.patch |
| L1 learn index + changed-records repatch | in progress | land/core/learncost/ | fluctuating-batch blocker |
| symbolic TensorIterator (witness retired) | in progress | land/core/symti/ | templates; binary size to shrink; full ATen rebuild |
| hint-read fixes | in progress | land/core/hintfix/ | + Triton (next/triton_hint), CuTe sites |
| redispatch sections RD2 (cluster/attr changes, memcpy sections, SymBool args) | done on cand3 tops; rebasing to 269/270 + hf1 placeholder | land/core/redispatch2/ (even 26dccf15a406, odd c6438f6fdfe3) | vLLM fork decode 3 traces -> 1 + 2 sections |
| S1/S2 ATen/Inductor, F4b | in progress | land/core/perf/ | F4d -> symti |
| 97b_c2 cast-free layouts | in progress | core/cute70/ | |
| run_buffer NaN (vLLM open variant) | repro pending | vllm lane | silent wrong output; core test needed |
| F21 / F22 (GB300 fuzz) | confirming | land/scratch/fuzz/FINDINGS.md | |
| validity checks as post-graph assertions | in hintfix | | user items 1-2 |
| learn cost LC_core (L1 + L1c + item4 + item8 structural refusal + conv siblings); L1b exec copies NOT merged (default off) | READY pending split + full harvest rerun | land/core/learncost/L1_learn_index.patch | GB300 vLLM confirmation pending (c8l build) |
| entry arg binding ~200 us/call (many static args + lifted ints) | in progress (land/core/entry) | | from V-stock profile (out/prof_vstock) |
| attention harvest: key on launch config, module load outside capture, guarded tile choice | in progress | land/core/attn/ | V-stock criterion 5 |
| slice end-clamp guard (symbolic Min) | hintfix lane | probe/retrace_class_repro.py (ii) | |
| memcpy fold/redispatch | redispatch sections lane | probe/retrace_class_repro.py (i) | |
| MT5 workspace (98 follow-up, eqy per-op workspace as ordinary allocation) | CuTe lane, 247/248 | | blocks candidate 3 promote |
| CeilToInt(FloatTrueDiv(ToFloat(..))) has no integer lowering (XLNet dynamic declines both graphs, h/off 0.91x) | in progress (land/core/ceilint) | scratch/numbers/compile/gh200/NUMBERS.md | lower exact int ceil-div for float-of-int expressions, or guard |
| hint-read fixes hf1-hf8 (zero hint reads in scope; OFFENDERS 32 -> 15, owners marked; F17; slice full-width) | READY (both lines green) | land/core/hintfix/READY.md | validity checks before-graph (decided) |
| TRT-LLM attention via real C++ launcher (SymInt) | MILESTONES MET (lives in FlashInfer patch + extension, not int6) | land/core/trtllm_cpp/ | replaces fork; fork mixed fix = stopgap |
| R1 gaps: float8 harvest refill, FP4 GEMM pointer argument | in progress (land/core/r1gaps) | | from trtllm_cpp DESIGN notes |
| pinned build fuzz | 22.2k programs, 0 NEW | land/scratch/fuzz/FINDINGS.md | |
| attention: A1 warm-up descends custom-op bodies, F1 fork unpinned max_kv (max_q>1 declines; A3 choice() NOT adopted), F2 parity grid (+ A2/V1 closed patched scalars, optional) | READY (F1 needs the decline edit) | land/core/attn/patches/ (md5s in STATUS.md) | V-stock mixed 7.12 vs default 27.23 ms, bitwise |
| CuTe route: TVM-FFI calls with None / NamedTuple / float args left eager (vLLM FA4) | CuTe lane, after MT5/97b_c2 | attn_open/REPORT.md, repro_fa4.py | |
| replay keeps an extra intermediate in pointwise chains (peak 64.5 vs 43 KB) | FIXED peak1 (a3af7b860a54 even / a4764d9068ed odd), rebasing onto 275/280 | | replay memory must match eager |
| (2-GPU Qwen) memcpy node can't be re-pointed between allocator and symm_mem memory (driver invalid value) | in progress (land/core/symm) | scratch/integration/dsv41/partA/memcpy_symm_repro.py, PREP.md A.2.3 | blocks vLLM symm-mem copy-in |
| (2-GPU Qwen) symm-mem port Option A (~500 LOC + 480 tests, Meta kernels for two_shot_all_reduce_ etc.) | in progress (land/core/symm) | dsv41/PREP.md A.4 | |
| Triton zero hint reads (6 sites; guards owned by the launch) | READY (on hintfix tops) | land/core/next/triton_hint/TH_triton_no_hint_reads.hf.int6cpp.patch (879741dac041) / .int6.patch (bd3e6f26977b) | strict suites green; odd 2 failures = base (FZ09b/FZ05) |
| learn cost LC_core.step242.patch (md5 f5d348af2290) | READY | land/core/learncost/ | exec copies kept aside (c9c875646815) |
| (later) vLLM _C ops via real C++ host tracing (replace Python ports), after trtllm_cpp proves the pattern | unassigned | | removes 7 extern keys per new T |
| F24 pinned: segfault instantiating a form whose clone holds a freed pinned host ptr | pinned lane fixing | land/scratch/fuzz/repros/F24_pinned_form_stale_host_ptr.py | crash (not wrong output) |
| symbolic float (SymFloat, fp32-exact guards; replaces choice()) | approved, implementing | land/core/symfloat/ | |
| F20: in-place op on a slice of a view returns the slice instead of the base (silent wrong output; predates cand3) | FIXED as FZ11 (0f4664c7eeae even / e749dd49ab0a odd), rebasing | land/scratch/fuzz/repros/F20* + new no-out= case | core-done blocker (silent) |
| ceilint.int6cpp.patch (e6e8f95b5ccd; XLNet dyn 0.91x -> 1.49x, |n| <= 2^53 bound) | READY, sent to int6 | land/core/ceilint/ | |
| IR floor-of-ratio (math.floor(a/b) lowered as //) wrong past 2^53 | unassigned | land/core/ceilint/STATUS.md | edge-case wrong value; needs own node kind |
| Flash-Next V-stock: a nested harvested op (reshape_and_cache_flash keyed site) inside a traced custom op (QSA selector) -> both claim one record; AssertionError escapes | fuzzfix lane (repro gap_5) | serve/STATUS.md | blocks Flash-Next |
| redispatch cost: ~0.8 s per redispatch on vLLM 1k1k conc64 (36-layer op redo, Python-heavy) | unassigned | serve out/ix/ov1, attn prof_redispatch.py | must be cheap |
| side-stream work inside a custom op (Qwen3.5-35B-A3B MoE) | unassigned | CORE_TASKS h gap_3 | |
| unpinned CPU tensor arguments (Qwen3.5-35B-A3B; Flex metadata) | unassigned | CORE_TASKS h gap_4 | decline today |
| redispatch re-runs a custom-op body that reads module/context state (core task i) | redispatch lane: design note for user review | vllm/probe/redo_module_state_repro.py | needs core: lifting active during redo |
| R1_i float8 fill (909b2eb3fed3 on lc / 8a03a1976338 on attn) | READY, rebasing onto step278 | land/core/r1gaps/ | |
| step280 pinned x LC regression: host memcpy nodes skipped -> stale pinned registration (silent wrong data) | fix F24k (7ae969af262a csrc, b7d00a3d0101 py), validating | land/core/pinned/ | blocks cand4 finals |
| FZ12b native rejection at a miss declines (5e085652483a / f00a80cea51a) | READY -> int6 after FZ12 | land/core/fuzzfix/ | |
| redispatch an op that holds a keyed site (Flash-Next QSA op) | redispatch lane, design note first | fuzzfix READY FZ12 note | perf |
