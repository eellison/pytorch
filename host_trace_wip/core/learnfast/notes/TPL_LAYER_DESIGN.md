# Cross-layer redo templates: core design sketch (for the user's review after the talk)

Prototype: learnfast/tpl_layer.py (on core's templates, step_ms10+). Labelled; used for cold-start measurements only.

## Problem
Core's templates (core_tpl.patch, landed in step_ms10) share a redo across layers when the layers' ops read their own
static-prefix weights. Two things still split every layer into its own key on vLLM models:
1. the op's call carries a per-layer constant: vLLM's layer name (a `LayerName` opaque value, or a str) that the op body
   uses to find the layer's module state;
2. each layer reads its own view into one shared cache buffer (GDN conv / SSM state, attention KV): a static-prefix
   argument whose offset differs per layer, keyed by value.
27B: GDN keys 48 per class (one per layer) -> 3 with both relaxations (the 3 vLLM KV-cache groups).

## Change (~120 LOC in _host_trace_redispatch + tests)
- `_reads`: (a) a constant of the call that is an opaque value-type object (torch's registered value-type opaque classes;
  vLLM's LayerName is one) is keyed as its type, not its value; the op is keyed by the roots its body reached instead
  (core's chain roots already in the key); (b) a static-prefix argument root whose offset is one plain symbol is keyed
  without the offset (the rest of its layout still by value).
- `_instantiate`: for a position whose root is another static-prefix argument with a different offset, each pointer slot
  into that root gets displacement + (o_j - o_i) * itemsize; `_for_instance` already renames the guards onto the
  instance's argument and value-pins both offsets.
- Soundness: the first instantiation of each (key, layer) also runs the full redo and compares with `_assert_same_entry`
  (layer = the instance's roots). A mismatch takes the full redo's entry, and that (key, layer) never takes a template
  nor contributes one afterwards. A per-layer Python scalar that reaches a launch, or a pointer not affine in the offset,
  shows up as a mismatch there. After it passes, the layer instantiates without re-verifying.
- Optional (guarded, labelled; lever (i)): confirm once per (op, layer) across variants, with a cheap check on later
  variants (fingerprint: the key with symbolic nodes as placeholders, the layer name, each root's kind / argument
  position / offset value / dtype); a differing fingerprint falls back to the full confirmation. Validation mode runs the
  full confirmation every time and counts cheap-pass / full-fail cases (expected 0).

## Evidence (prototype)
- 0 confirmation mismatches over 1,677 (key, layer) confirmations on 27B (vt RANGE=1, step_ms10 + Delta 3).
- n27wf --check with the prototype on: prefill part 97/97, mixed part 210/210 bitwise.
- Cost: a confirmation is one full redo (~65 ms on 27B); per-(key, layer) mode made 67 s of the 384 s warm-up.

## Tests to add with the port
- layers whose ops differ only in an opaque layer-name constant share one template (redos 1, instances layers-1), verify on;
- per-layer views into one buffer at different offsets share one template, outputs bitwise;
- a per-layer Python scalar reaching a launch: confirmation mismatch, that layer stays on full redos, outputs bitwise;
- a non-plain offset (an expression) keeps the value key.

## 10-19: dropped (coordinator)
27B ablation on step_ms32 (core agent, rs5, GPU 3): vt with TPL_LAYER_KEYOUT=0 was ready at 146.0 s (warm-up 109.5 s) vs vt
with it 179.2 s (125.3 s). Since core's template-key fix (ms22, TPL_DECLVAL) the GDN keys already collapse, and the key-out's
per-(key, layer) confirmations cost more than they save. Not ported.

## Correction (core agent, 10-19): the "dropped" ablation was invalid
startup27_abl.sh's KEYOUT=0 only unset TPL_LAYER_KEYOUT, which tpl_closure.py reads; on step_ms32 (which has _for_instance)
diag27's DIAG27_TPL=1 imports tpl_layer.py, whose layer-name and declared-offset key-outs are always on. So cs32vt_nokeyout ran the
vt configuration: 146.0 s ready vs vt r2 151.5 s (vt r1 179.2 s was a cold first round: init 37.4 s vs 25-26 s). Key-out's cost or
benefit is unmeasured; the valid ablation is DIAG27_TPL=0 (no tpl_layer). The drop is withdrawn pending that run.
