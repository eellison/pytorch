# Design-note section (learnfast -> coordinator): overlapping our graph warm-up with weight loading (user's idea)

The idea is the user's: start the traces once the parameters exist, without waiting for their values. The cuBLAS learn
already overlaps: side processes run from process start, on synthetic operands.

## (1) What a trace needs real weight values for
- The symbolic trace: nothing.
  - It runs on traced tensors (fake metadata plus address symbols).
  - A host read of any device value is a decline (E24). Hint reads are never allowed.
  - So no launch, grid, guard or allocation in the tape can depend on a weight value. Under closure, weights
    contribute their storage base (fixed once allocated), and their sizes/strides (fixed at allocation).
- The witness / warm-up eager call (trace(warm_up=True)): it runs fn once on the real arguments, so that one-time
  work happens outside the trace (lazy inits, Triton/CuTe JIT, autotuner benchmarks).
  - Its outputs are discarded during startup, so garbage weights are harmless for correctness.
  - Two caveats:
    - An autotuner that benchmarks inside the witness times kernels on garbage data. Every candidate it can choose is
      correct, but the choice could differ from what vLLM picks with real weights. That breaks the "same kernels as
      the baseline" rule, not soundness. Mitigation: shared autotune caches, as today, and FlashInfer's autotune runs
      in vLLM's own kernel_warmup.
    - Data-dependent kernels (NaN/inf-dependent early exits) only change timing.
- Checks (--check, verify-on-load of persisted traces): these need real weights, so defer them until loading finishes.
  A variant is not used for serving until its deferred check passes, when checks are on.
- The cuBLAS harvest's verify: it uses its own synthetic operands, so no weights are needed (already overlapped).

## (2) Where the hook goes, and the saving
- vLLM init order: load_model (initialize_model allocates the parameters; load_weights copies values H2D) ->
  profile_run (one forward; memory measurement depends on shapes only) -> KV cache allocation -> compile/warm-up ->
  (vLLM) capture.
- Our traces close over the KV storages, so the KV cache must exist before a trace. The re-ordered schedule:
  1. initialize_model (parameters allocated, values not loaded);
  2. start load_weights on a background thread with its own CUDA stream. The safetensors reads and H2D copies mostly
     release the GIL;
  3. on the main thread: profile_run on unloaded parameters (memory only), allocate the KV cache, then compile (Vc)
     and our traces;
  4. join the loader; synchronize its stream before the first served step; run the deferred checks if on.
- The hook point is the model runner's load_model, between initialize_model and load_weights. An adapter can patch it
  from outside vLLM; no vLLM source change.
- Saving on 8B (repeat-2 timeline):
  - weights 5.75 s, model load 9.0 s in total;
  - our traces 5.7 s (compiled) / 9.8 s (eager);
  - for Vc, torch.compile (Dynamo + Inductor, 26 s) also needs no values and overlaps the same way.
  - Hidden: about min(9 s, traces) = 5.7-9 s per cold start. VR (eager forward) ready time ~51 -> ~42 s. Vc: the 9 s of
    loading hides under compile + traces.
  - With persisted traces (DESIGN_persisted_traces.md) the warm-up is ~1 s and the saving shrinks to that.

## (3) Soundness catches
- Values never enter the tape (no host reads, no hint reads), so tracing on unloaded parameters records the same tape
  as on loaded ones. The witness's outputs are not served.
- Storage addresses must not change when values arrive. load_weights must copy into the existing storages (vLLM's
  weight_loader does param.data.copy_). Under closure, any `.data =` / set_ / reassignment invalidates the variant.
- Parameters whose shapes are decided at load: quantized formats that repack or allocate scales during
  process_weights_after_loading (NVFP4, FP8 block scales). These reallocate after load, so the traces must start after
  process_weights_after_loading, or the variants touching those parameters are invalidated by the closure hooks and
  retraced. For bf16 8B nothing reallocates. For 27B NVFP4, check which layers repack.
- Ordering: the first served replay must follow the loader stream's completion (an event, recorded and waited on the
  serving stream). Otherwise kernels read half-loaded weights.
- Not allowed: making any guard depend on weight values to "validate" the early trace. The deferred check compares
  outputs only.
