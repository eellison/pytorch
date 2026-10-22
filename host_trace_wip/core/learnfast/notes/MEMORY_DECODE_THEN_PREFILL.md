# +107 GiB live memory in ours' 27B prefill after a decode phase (learnfast, 10-15)

## Finding
- 27B Qwen3.8-NVFP4, ours V (wip14 / build_cpp_ms9), sections on (Delta 3), ARMV_MEMORY=run_buffer (the adapter default),
  KV 64 GiB. max_memory_allocated (GiB) of single-request prefills after a decode warm-up phase:

| T (tokens) | ours run_buffer, decode bs edges to 128 | ours run_buffer, decode bs edges to 8 | ours ARMV_MEMORY=eager | vLLM eager |
|---|---|---|---|---|
| 1,024 | 96.49 | 96.49 | 89.56 | - |
| 4,096 | 117.02 | 117.02 | 90.14 | - |
| 16,376 | 199.07 | 199.07 | 92.45 | 92.54 |

  (89.4 GiB allocated at ready in every arm; reserved at 16,376: 215.7 / 215.4 / 97.4 / 93.4.)
- Linear in tokens (~6.7 MB per token), independent of the decode phase's batch sizes.
- Without the decode phase first (prefill edges straight away) ours run_buffer peaks at 91.71 GiB at 16,376, below eager.

## What it is
- Allocator snapshot (out/memsnap_decode128_p16k.pickle) across the 16,376-token prefill: at the peak, 109.7 GiB of the
  allocations made in that call are live, all from the native replay (_host_trace_replay._call_native): per GDN layer
  (x48) 7 x 192 MiB + 384 + 320 + 2 x 64 + 96 MiB, about 2.2 GiB per layer, plus one 2.06 GiB run buffer.
- These are the GDN ops' section temporaries. HostTraceVariant::allocate (EagerStep.cpp) in a run with temporaries and no
  eager order (the run_buffer plan) makes every selected section's temporaries first and holds them until the run's graph is
  queued (`for (const Entry* e : sections) make(*e, held);`), so the 48 layers' sections hold their temporaries at once;
  eager frees each layer's after its op. Delta 3's _lifetimes models exactly this (section temporaries held over the run).
- Why only after a decode phase: the decode warm-up's batched prefills make the prefill variant whose later size classes
  are redispatched with GDN sections; a prefill sweep run first traces those classes instead (same-chain entries, run-buffer
  temporaries placed first fit by lifetime, reused across layers).

## Slide-adjacent runs
- r27's timed sets (n27w / n27wf / n27wfg / n27m / n27s, every arm incl. vcvp / vsvp, decode then prefill / mixed in one
  process) refused sections (no_sections) and peak at 92.7-92.8 GiB allocated vs 92.8 for fullnc / breakable (2.46-2.52 GiB
  over init): not affected. Only the learnfast cold-start runs with sections on (rs1..rs6, fg2, mem2, snap*) carried it.

## Fix
- Config (immediate): ARMV_MEMORY=eager wherever sections are on. Eager order makes a section's temporaries before the op's
  first launch and frees them after its last (alloc_at / free_at), so layers reuse the bytes: 92.45 GiB at 16,376 vs vLLM
  eager's 92.54 (reserved 97.4 vs 93.4).
- Core (proposed, for the core agent): in a run-buffer run, place section temporaries in the run buffer with the op's
  [first, last] launch span as their lifetime (first fit, as the run's own temporaries), instead of holding each section's
  temporaries for the whole run. Then run_buffer + sections is eager-like too. Until then, run_buffer with sections on
  breaks "live memory = eager".

## After the core fix (step_ms15 / build_cpp_ms10, run_buffer + sections), 10-16
| point | ours allocated / reserved (GiB) | vLLM eager allocated / reserved |
|---|---|---|
| 16,376-token prefill after a decode phase | 92.64 / 100.56 | 92.54 / 93.4 |
| rs5-run mixed phase peak | 92.83 / 104.45 | 92.54 / 93.4 |
| 800-step stream | 90.58 / 106.02 | 90.53 / 93.4 |
- Live: +50-290 MiB at the peaks (0.1-0.3% of 92 GiB): within the user's small-overhead rule (coordinator 10-16). Not yet named;
  one allocator snapshot at the mixed peak is queued as low priority.
- Reserved: +7-13 GiB over vLLM eager: the known run-buffer / caching-allocator reserved growth (QUESTIONS #7, deferred to after
  the talk). Neither affects the slides: timed sets run without sections.
- Named (10-16, allocator snapshots over the rs5-style mixed phase, out/memsnap_mixed_{ours15,eager15}.pickle): of the
  allocations made in that phase, ours' live peak is 3,550 MiB = one 3,390 MiB run buffer (the step's temporaries, placed first
  fit by lifetime) + a 160 MiB tensor; vLLM eager's is 3,251 MiB of individual temporaries (fused_post_conv_prep 5 x 192 MiB,
  fused_add_rms_norm 4 x 160 MiB, a 512 MiB unquantized GEMM output, chunk_delta_h 384 MiB, causal_conv1d 320 MiB, ...). The
  ~300 MiB is the run buffer's packing over eager's dynamic reuse (about 9% of the step's temporary peak), not a leak or a held
  section. Within the small-overhead rule; tightening it would be a placement question for the run buffer (first fit vs best fit).
