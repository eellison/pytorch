# Design note (learnfast -> coordinator): a faster sound cuBLAS learn for cold start

Status: prototyped in learnfast/ (batchlearn.py, parf.sh, merge.py), on existing provider internals; core is unchanged.
Every number below is measured on GB300 GPU 2 (gpu2.lock) with load avg ~80-200, so treat wall times as coarse.

## Where the time goes today (Qwen3-8B key set: 5,944 keys from vllm/out/bindings_cbh.pkl)
- In-engine learn (t_cbh_learn): 123 s. That includes 15 s of init and 1,917 model steps. The cost per binding is not
  20 ms.
- Straight from keys, serial (HostTraceReplay._learn per key, harness.py): 17.0 s, 2.86 ms/key mean.
  - _harvest: 2.7 ms/key, of which digests 0.45, captures 0.29 (1.18 captures/key: 181 full harvests, the rest
    siblings), plug 0.11, and the remainder Python classify/verify.
  - _learn's per-key operand fill: ~2 ms/key average (fresh weights per key, up to 1.2 GB for lm_head).
  - The loop is CPU-bound Python: CPU time is about equal to wall.
- 27B: harvest is <1% of the learn (learn_s 9.0 s of 1506 s). The learn is whole-step traces; see
  DESIGN_redo_resolve.md and DESIGN_closure.md.

## Changes, each sound by construction
1. Operand buffers reused per operand slot, filled once (instead of fresh random fills per key): 17.0 -> 15.1 s.
   - The binding is harvested and verified exactly as before. Verify compares the binding with eager on these
     operands, whatever their values.
2. Verify's launch wait polls the event without time.sleep(1e-4): 15.1 -> 14.0 s. The hang deadline is unchanged.
3. Batched sibling checks:
   - A sibling key is captured and adopted by the provider's own _harvest with _VERIFY_SIBLINGS off.
   - The provider's exact in-place GEMM check is then enqueued and not waited on: eager into the output; digests of
     every operand; then per scratch fill (0x00, 0xFF) the output filled 0xFF, the binding launched, and digests
     again.
   - 64 keys go per sync, and all digest buffers are compared on the device.
   - These keys go through provider.learn unchanged: any that fails, any that holds a constant or misaligned CUDA
     address (verify's mirror route), and any that takes no sibling. That path does the full harvest, the memset /
     owned-slot repair, or the refusal.
   - So every binding kept passed the same bitwise comparisons as today.
   - 1 process: 11.2 s.
4. Processes, each on a shard, plus a merge (the saved format is already portable):
   - Without (3), processes do not scale. Per-process CPU time inflated ~4.5x, because the ~8 syncs per key wait
     through GPU context time-slicing and the driver spin-waits.
   - With (3): 4 procs 7.0 s, 8 procs 6.9 s (+1.7 s merge).
   - Threads in one process are not an option without core changes. learn() holds provider._lock and the module
     _LOCK; _DEVICES streams, the record-launches flag and the allocator history are process-global, so concurrent
     captures would corrupt each other's allocation logs; and the work is GIL-bound Python.
5. One fill when the scratch is unreachable:
   - A binding with no scratch slot and no memset into scratch launches once: its 0xFF launch would repeat the 0x00
     launch bit for bit.
   - The argument: a kernel reaches memory only through addresses in its parameters. Constant CUDA addresses are
     already excluded by (3)'s fallback. No slot points into scratch, so the fill is unobservable.
   - 99% of keys. It matters only where the learn is GPU-bound (below).

## Identity evidence
- The 5,944-key and 66,560-key sets, from every variant (serial harness, fastlearn, batchlearn at 1/4/8 processes, one
  and two fills, GPU 2 vs GPU 3), are identical to each other:
  - 5,724 / 66,340 keys byte-identical;
  - 220 keys differ only in their loose byte ranges, which are host / stack bytes the harvest keeps as capture A's.
  - Two serial runs of the same driver show the same 220-key loose-byte difference, so it is run-to-run host state,
    not the method.
- 0 refused, 0 failed, in all runs.

## The full warm-up range is GPU-bound
- 8B, T in [1, 16384] for the 4 layer GEMMs + lm_head in [1, 1024] = 66,560 keys, 8 procs:
  - 147 s with two fills;
  - ~98 s with one fill (quiet GPU 2), ~119 s on shared GPU 3.
  - That is about 1e17 FLOP of verify GEMMs (eager + binding per key), at ~1.05 PFLOP/s effective. More processes do
    not help (16 procs = 8 procs).
- 27B, 2 GEMMs x T in [1, 16384] + lm_head in [1, 128] = 32,896 keys, 8 procs, one fill: 45 s learn, 53 s wall.
- First-use learn of one new large size (standard path, quiet GPU 2): 2.0-6.7 ms per key. 8B has 4 keys per new T,
  so 10-25 ms against a step of hundreds of ms.

## Proposal for core (needs the user's review)
- (a) HarvestProvider.learn_many(keys, operands_for), which batches the sibling checks exactly as in (3).
- (b) The one-fill rule (5) inside verify.
- (c) An offline learner entry point: shards over processes, then merge(). This is a script around existing APIs; core
  only needs save/load, which exist.
- (d) The warm-up range API (W1) calls (a), so a cold start pre-learns the hot range in a few seconds, overlapped with
  engine init. The full range is ~100 s on 8B, bounded by the verify GEMMs, unless the user accepts the documented
  sibling-verify-off option (E-sibling-verify-option), which these numbers do not use.
