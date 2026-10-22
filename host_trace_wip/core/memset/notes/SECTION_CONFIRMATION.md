# Confirming a library body whose redo is a section (proposal, for the coordinator)

## Observation (27B vp, step_ms34)
16 GDN ops (vllm.qwen_gdn_attention_core_fused_norm_packed) refuse: "confirmation: its redo at its trace call takes another
chain". The size dump (logs/cs34dump*.log) shows the redo at the trace call gives the same 20 launches by name, and its 23
temporaries have the same dtype / size / stride formulas as the op's 23 allocations. _redo makes it a section because the chain
holds a D2D memcpy (aten.copy_ lowered as Memcpy): "a memcpy, which only a section's own node holds". The confirmation then
refuses any section. These refusals become 278 refused redispatches, i.e. retraces: most of the 27B vp vs vt gap.

## What the confirmation certifies
That the body's records are a function of its call: redone at its own trace call, it gives back its recorded launches launch for
launch (every compare=True field, values by node; roots by identity) and decides nothing beyond its own symbols. A section is a
statement about placement (its own scratch in the op's place, checked by check_section when the entry is added), not about what
the body read.

## Proposed rule
At the trace call, a redo that is a section confirms when, in order:
1. its temporaries pair one to one with the section's drops (the op's allocations it does not make), in allocation order, each
   pair with the same dtype and the same size and stride formulas (node equality, the static prefix's layouts at their values
   as elsewhere);
2. its launches equal the op's recorded chain launch for launch under that pairing (a temporary's root counts as its paired
   allocation's root; every other root by identity), with the same field comparison as today (_launch_difference); and
3. its guards satisfy today's rule.
Otherwise it refuses as today.

## Why it is sound
Under 1 and 2 the redo at the trace call produced the trace's records but for the identity of the op's own allocations, which
are the op's scratch: no record holds a constant where the trace has a symbol, and no extra decision was made. That is exactly the
evidence today's confirmation accepts for a non-section redo, so the inference "the body reads only its call" is unchanged. The
section entry's use at later calls is what core already does for an ATen op's section: check_section places its temporaries in the
op's slot of the memory plan and refuses otherwise (_Variant.refused_sections). Nothing about holders, lifted state or the
contract changes; a body that read a holder still records a constant where its trace has a symbol and fails 2.

## Cost
One extra pairing and comparison per confirmation that would have refused; none on replays.

## Repro
coldstart/repro/section_confirmation.py: a library op whose body copies its input into its own scratch (aten.copy_, a D2D
memcpy) and branches on its size. step_ms35: traces 2, redispatches 0, refusal "confirmation: its redo at its trace call takes
another chain". Under the proposed rule it would redispatch (1 trace), the section's temporaries placed by check_section.

## What would make it unsound (and why each is excluded)
- A temporary standing for an allocation that outlives the op (an output, or an allocation a later op reads): then the section's
  scratch would not be the variant's tensor. Excluded: _redo pairs the op's outputs with the variant's allocations before making
  temporaries; only the op's other allocations become drops, and check_section refuses a section whose temporaries or drops a
  later launch reads outside the op's slot.
- A pairing that maps a temporary onto an allocation with another role but equal formulas: the launch comparison under the pairing
  then sees the roots in the wrong places and refuses, unless the records are symmetric in the two, in which case the roles are
  interchangeable and the redo's records are the trace's.
- A record type that the section rewrites (a Memcpy in the redo where the trace had a kernel, or the reverse): the field
  comparison checks the record types launch for launch, so a type change refuses.
- A size or stride of the scratch that came from a holder (a constant where the trace has a symbol): rule 1 compares the formulas,
  so it refuses, as the launch comparison does for slots today.
- Relaxing more than this (a section with another launch count, or temporaries that do not pair): not proposed; it would no
  longer be "the trace's records back", which is the confirmation's whole evidence.

## 10-21: landed as step_ms36. The 16 GDN refusals under the rule (for the coordinator; nothing coded)

### Data (cs36d: one 27B vp round on step_ms36 with the PORTED_DEBUG_CHAIN dump in coldstart/ported.py)
- Library counters as ms34 / ms35 / cs36 r1-r2: 205 own confirmations, 3171 shared, 16 refused, confirmation_s 34.1; 278
  refused redispatches with the same four reasons; engine ready 299.8 s, ready-to-steady 442.8 s (init 31.2 s, a cold init).
  Logs: coldstart/logs/cs36d.log, learnfast/out/startup27_h3t_vp_rw_sec_rcs36d.json / .diag.jsonl.
- The 16 are one traced op each in 16 variants: the tape's first GDN op (op 15, vllm.qwen_gdn_attention_core_fused_norm_packed).
  A refusal ends the redispatch and is cached per traced op, so in those variants the other 47 GDN ops are never tried.
  189 confirmations, none a section: where op 15 confirms, its redo holds no memcpy.
- In all 16, _section_pairing succeeds (23 or 29 temporaries = the drops, in order) and every pair's dtype / size / stride
  formulas are equal (5696 declared symbols in play). The refusals:
  (a) 10 at the guard check, "90 / 93 guards its trace did not record (True)". The same guards classified in the non-section
      form (before fold_x): 195 guards = 169 recorded as is + 26 within own | declared as is (4 variants); 275 = 245 + 30 (3);
      277 = 247 + 30 (3). Zero are extra in that form.
  (b) 6 at the launch check, "launch 16 / 21 differs at its trace call (slots)": aten._index_put_impl_.default, the same
      CUfunction and the same four roots (p1813, p1924 / p1928, a34 / a39, a37 / a42), 14 slots in the redo's record for the
      trace's 15. The first differing position, 8: the trace has 65536*s26 (s26 a declared layout symbol, 3145728 at its
      value); the redo has 8*s7647 (or 8*s7665), a value over a symbol of the variant.

### (a) Fold vs rename: the exact difference
_confirmation's section path, after the launch comparison:
    exprs = [fold_x(g) for g in exprs]  # over the scratch's own addresses: the allocations'
with fold_x = _at_values(ctx, declared, scratch): a ChainMap of {scratch symbol -> the allocation's symbol node} and
{declared symbol -> its hint as a const}; a node whose free symbols meet either set is rebuilt by ctx.transfer in canonical
form, so a guard over declared symbols alone (Eq(d, 4), d a static-prefix layout symbol) rebuilds as the const true node. Then
    extra = [g for g in exprs if g not in recorded and not ((free := _symbols(g, set())) and free <= own | declared)]
counts it: true is no recorded guard and has no free symbols. The non-section path runs the same `extra` test on the guards as
_redo renamed them (over the variant's symbols; _redo already drops a guard that renames to true): a guard over declared symbols
passes by free <= own | declared, a recorded one by identity (nodes are interned per (op, args)). The two paths judge the same
guard on two forms; only the form differs. The rename the comment intends is _at_values(ctx, frozenset(), scratch): the
scratch symbols to the allocations' nodes, nothing at values. Under it a guard over the scratch (sectionN) becomes one over the
allocation's symbol aK, which `own` holds (op.allocs' q / sizes / strides) or which the trace recorded; a guard without scratch
symbols is unchanged.

### Why folding to true loses information
Rule 3 asks of each guard the redo decided: is it one the trace recorded, or one over symbols whose values the entry's predicate
already carries? A guard over declared symbols is of the second kind: under the static prefix's contract those layouts are the
same every call and the variant's recorded guards over them are checked at every call, so an extra one over them hides no
decision (which is why the non-section path accepts them). Folding it to true discards which symbols it was over; a constant
can neither be matched to a recorded guard nor be judged by its free symbols, so the only verdict left is "extra". The fold is
right for launch values (a redo that read a declared view at its constant layout holds the constant where the trace has the
symbol; comparing at the declared values is the sheet's rule) and wrong for a membership test, which is about symbols, not
values. The guard is still checked at the declared values: by the variant's own recorded guard over the same symbols, or, for
an extra one, by nothing at all on either path today (the non-section path drops it the same way), which is the existing rule.
Repro with a static prefix (coldstart/repro/section_confirmation_static_prefix.py on step_ms36,
coldstart/logs/repro_static_prefix_ms36.log): the body branches on w.shape[0] (w the static-prefix argument) and on x.shape[0];
the twin without scratch (scale_w) confirms (1 trace, 1 redispatch); the twin with scratch (copy_scale_w) is refused
"1 guards its trace did not record (True)" with all 18 of its guards recorded as is.

### Proposed change (one line; not coded)
In _confirmation's section path, `exprs = [rename(g) for g in exprs]` with `rename = _at_values(tape.shape_env.ctx,
frozenset(), scratch)`; fold / fold_x stay as they are for the launch comparison and the layout pairs. No condition is relaxed:
rule 3 is applied to a section's guards on the form the non-section path uses. No cost beyond the existing pass.

### Proposed test (fails on step_ms36 with "guards its trace did not record (True)")
test_a_library_section_with_a_guard_over_the_static_prefix_confirms (replay, beside the other library-section tests): the
repro's twins as host_trace_direct ops (copy_scale_w / scale_w: `f = 2 if w.shape[0] > 16 else 3`, `g = 2 if x.shape[0] > 16
else 3`, the first copying x into an empty_like scratch), step(w, x) = op(x, w) + w[:1], HostTraceReplay(step, static_prefix=1)
with one w of 32 rows over x rows (8, 8, 32, 32, 8, 64): outputs bitwise, traces == 1, redispatches > 0 (sections > 0 for the
scratch twin), _LIBRARY_COUNTS["refused"] unchanged. Negative twin kept refused: the same body with the guard over a holder the
step fills from an int (HOLDER["n"] > 16), as test_a_library_section_with_a_holder_sized_scratch_is_refused does for a size.

### (b) The slot differences at launches 16 / 21: not the guard fold; the op stays refused
A natively traced kernel's slots are the parameter words the C++ host recorded as tracked (Tape._launch_of: fields =
(param, offset, width, value, pointer); a constant word is in `images` only). The trace's slot 8 of aten._index_put_impl_ is
65536*s26, a stride of the state tensor (a declared size times 65536 bytes per row); at the redo that tensor is reached through
close_over and read at its constant layout (sheet item 4), so the stride is a constant in the parameter bytes and no slot: 14
slots for 15, and the positional comparison fails at 8 (by the counts the redo's 8*s7647 is the trace's next slot; the dump
prints the first difference only, so that is inferred, not read). This is a structural difference of the record ("the trace's
records back" fails at the launch), not a guard-form asymmetry; the rename above does not touch it. Reconciling it would mean
comparing the packed parameter bytes at the declared values instead of the slot lists (a change of what is compared), or reading
a declared view with symbolic strides at the redo (a change of close_over's semantics): neither is proposed. Under the rule
these 6 variants' op 15 stays refused and the variants retrace. Which variants carry the symbolic stride is per variant (the
10 guard-group variants pass the same launch): a follow-up dump of both full slot lists would show it, if wanted.

### Not proposed
No change to fold / fold_x for launch values and pairs; no reconciliation of (b); no change to what a section drops or to the
drops check. Expected effect of the one-line change, to be measured under the full gate (the new test, suites, parity8b,
fn_gate, 8B and 27B churn checks, run27wf, one vp round): the 10 guard-group variants' op 15 confirms (their launches already
match) and the redispatch goes on to the variant's other GDN ops (each confirms on its own or shares, or refuses for its own
reason); the 6 slot-group variants keep retracing.

## 10-21 decision (coordinator): (a) approved as ms37; (b) a structural limitation, recorded
- (a) ms37 on step_ms36: the section path's guard membership test runs on the guards with the scratch renamed only (fold_x
  stays for launch values and pair layouts), so a section's guards are judged on the form the non-section path uses; a guard
  over declared symbols is still enforced by the static prefix's contract (statics_changed invalidates), so no check is lost.
  With the static-prefix twin test (fails on step_ms36) and the holder-guard negative kept refused; full gate; 2 vp rounds
  reporting engine ready, ready-to-steady, the counters, how many of the 16 variants' op 15 now confirm and what their other
  47 GDN ops do.
- (b) Structural limitation, not a defect of the rule: a close_over'd tensor is read at its constant layout at a redo, so a
  stride of it that is a slot (a tracked parameter word) in the trace's record is packed bytes in the redo's record; the two
  records differ in structure (one slot fewer) and "the trace's records back" fails at that launch. The 6 variants' op 15
  (aten._index_put_impl_ at launch 16 / 21, slot 8 = 65536*s26) stay refused and those variants retrace. Future item, not
  proposed now: tracking a declared view's strides as slots at the redo (symbolic at their declared values), so the record
  keeps the trace's structure and the existing value fold compares them; it changes how close_over reads a view (sheet item
  4) and what a redo's record holds, so it needs its own sheet entry and the user's OK.

## 10-21 ms37 candidate (snap/wip48 = step_ms36 + rename-only guard membership): measurements and follow-up candidates
Two 27B vp rounds (cs37_r1 with the dump, cs37_r2 clean; both under a loaded host: another user's build, init 50.7 / 58.7 s vs
17-20 s on the ms36 rounds, so engine ready is not quoted from them; quiet-host rounds follow).
- Counters (both rounds): refused 1 (ms36: 16), own confirmations 196 (205) of which 96 sections (0), shared 1005 (3171),
  confirmed 1200 (3360); refused redispatches 22 (278), all one variant's op 15 at launch 16 (the slot case, (b)); traces during
  the run 5 (20), miss events 8 (23), steps with a trace or learn 12 (26).
- Per variant (r1 dump): tape 40976 all 48 GDN ops confirm (non-section) + 20 other ops; tapes 45696 and 54928 all 48 GDN ops
  confirm as sections, each its own confirmation (2 x 48 = 96) + 5 / 21 other ops; tape 53120 op 15 refused (slots), its other
  47 never tried. The variants that retraced before are served by redos now: the 16376 / 3541 / 4273 / ... prefill traces are
  gone (5 traces per run instead of 20).
- Warm-up breakdown (range_warm.phases, s; ms36 r1 / r2 -> ms37 r1 / r2; vt cs32vt_r3 for reference):
    decode    61.1 / 62.3 -> 74.6 / 81.6   (vt 54.4; same 3 traces, 16 redispatches; the host load)
    prefill1  74.7 / 68.7 -> 64.5 / 61.7   (vt 17.6; traces 4 -> 1, redispatches 98 -> 37 but their time 13.6 -> 40.0 / 37.2)
    prefill2  44.1 / 42.1 -> 31.5 / 28.4   (vt  4.6; traces 3 -> 0, redispatches 21 -> 9, their time 1.7 -> 25.3 / 21.7)
    mixed     71.0 / 65.1 -> 74.9 / 72.0   (vt 38.0; traces 3 -> 1, redispatches 53 -> 17, their time 16.4 -> 41.6 / 40.7)
  Slow-call seconds in the warm-up 210 / 205 -> 55 / 51 by the miss events; the redispatch accumulators rose 55 -> 107 s because
  a redispatch that succeeds (48 confirmations, section entries built and placed) costs far more than one refused at op 15 (its
  retrace was counted under trace). Serving phase: stream to steady 146.8 / 143.1 -> 112.7 / 107.6 s, steady tok/s 1202 / 1239
  -> 1363 / 1363.
- Follow-up candidate 1 (not implemented): confirmation sharing across the GDN layers of these variants. In all three confirming
  variants no GDN op shared, section or not (48 own confirmations each; "confirmations with no confirmed key" = all, so the 48
  ops' template keys (_reads) differ, not their record match), whereas in the decode-shaped variants GDN ops share (ms36: 3171).
  Measured cost in r1: 48 non-section confirmations in 8.9 s (0.20 s each), 48 section confirmations in 15.2 and 20.7 s
  (0.30-0.40 s each): ~45 s per run of the ~107 s the warm-up spends in redispatches, if 47 of 48 shared. Sharing a section
  confirmation is the same argument as the non-section one (the records are the op's with the read-walk roots renamed by
  position; the section's temporaries are the op's own allocations, which the bijection already covers) once the keys agree; the
  first step is to find which part of the key differs per layer in the prefill variants (a per-layer close_over'd state tensor
  among the chain roots is the suspect). Needs the coordinator's OK before code.
- Follow-up candidate 2 (not proposed, see (b)): the 6 slot-case variants.
- Quiet-host rounds (cs37q_r1 / r2, own-cores 0.069 / 0.086 busy, PSI 0.29 / 0.31; the numbers to quote; engine ready / to steady
  s): 266.6 / 365.4, 248.7 / 348.2 (ms36 289.2 / 435.2, 271.6 / 414.0; vt 148.6 / 207.0, 151.5 / 211.3). Warm-up 223.1 / 215.0
  (ms36 254.5 / 241.0; vt 116.0 / 112.6): decode 66.5 / 60.2 (vt 50-54), prefill1 60.3 / 57.1 (vt 17-18), prefill2 25.5 / 25.3
  (vt 4.5), mixed 65.2 / 66.9 (vt 38-40); redispatch seconds 120.5 / 115.6 (vt 40-42), trace seconds 38 / 37 (vt 30-31). Serving:
  stream to steady 99.4 / 100.1 s (ms36 146.8 / 143.1; vt 59-60), steady tok/s 1385 / 1409, first-request TTFT 339 ms (ms36
  2149 / 2173 ms: the first request traced on ms36).

### Follow-up candidate 1, diagnosed (cs37k: one vp round with the _reads key diff in the dump; logs/cs37k.log)
Why the 48 GDN ops of one variant never share a confirmation: their _reads keys are not None (the walk classifies them: 147 items
in the decode-shaped variant, 427 / 517 in the prefill ones, 14 / 39 / 48 roots) and differ, op 15 vs ops 55 / 95 / 181, in three
kinds of positions and no other:
(i)   the call: one per-layer `vllm.utils.torch_utils.LayerName` object, keyed as (type, object) by identity (every pair; the body
      looks its layer's state up by it in the forward context);
(ii)  per-layer layout symbols of the layer's close_over'd state tensors (s7726 vs s7733 ...): not in `declared` (those tensors are
      not static-prefix inputs, they live on declared storage), so they enter as ('int', symbol) through the op's own allocation
      sizes (48*s7726: scratch sized by the state) and the chain roots' layouts: 2 + 3 positions in the decode-shaped variant,
      30 + 20 and 40 + 29 in the prefill ones (the same symbol repeated);
(iii) for a layer of another KV-cache group (op 181 / 183), the declared views' offsets keyed at their values: ('value', 0) vs
      ('value', 2146729984) and ('value', 15360) vs ('value', 1073380352).
No launch field differs (the (type, function, attributes, programmatic) items are equal), so there is no per-layer constant in a
launch: the keys differ because the per-layer declared state enters the key by identity (its own symbols and offsets) and by the
LayerName that selects it. For comparison: the trtllm decode ops differ by (iii) only (3 call values: 0 vs 4293459968, 256 vs
4293460224: the per-layer KV view offsets), the FP4 GEMMs by real shapes (3072 vs 2560) and dtypes.
Reading: (ii) and (iii) are the case the coordinator named (the declared storage entering by identity): on the key side the
declared views' layouts would enter by role (a declared root at that position, its layout pinned per instance as a static-prefix
input's is) and sharing follows from the existing rule once the records match under the bijection (the record match must then
compare those layouts at their values too, as it does for static-prefix layouts). (i) is a call constant: by the existing rule a
different call; keying a LayerName by role (an opaque per-layer token that selects roots) would be a new argument (the records
match as the evidence that it selected nothing else), for the coordinator / user. Expected saving if all 48 share: the own GDN
confirmations, 8.9 + 15.2 + 20.7 s in cs37_r1 (0.20 s non-section, 0.30-0.40 s section each), ~45 s of the quiet rounds' 116-121 s
of warm-up redispatch time; the same applies to the 3 trtllm decode confirmations per variant (small). No code.

### Follow-up candidate 1, corrected (cs37k3: the differing key items' symbol sources; logs/cs37k3.log)
static_prefix = 1916 of 1942-1966 inputs. The per-layer symbols of (ii) are not close_over'd storage: they are inputs at positions
>= static_prefix, the lifted metadata tensors arg1922..arg1926 (size / stride / offset symbols s7638..s7656) and lifted ints
arg1935 / arg1942 / arg1949 (s7726 / s7733 / s7740; 7 ints apart). Ops 15 / 55 / 95 (GDN layers 0, 1, 2) each read a different
lifted tensor and int set; op 181 (layer 4) reads the same lifted inputs as op 15 and differs from it only by the LayerName (i)
and two static-prefix layout values (iii): the metadata is lifted per vLLM KV-cache group and the hybrid groups interleave the
GDN layers. (iii) carries no symbol: ('value', 0) vs ('value', 2146729984) are static-prefix inputs' layout values (the layer's
state view offset inside its group's buffer), keyed at value by the existing rule. The trtllm decode ops differ by (iii) only.
So: the "declared storage by role" change (ms38 as approved 10-21) would alter none of these keys, and a record match at values
cannot share (iii). Options, for the coordinator: (a) static-prefix layouts by role in the key and renamed by root position in
_records_match (what _for_instance already does for a static root at another position: subs + per-instance root_pins), which
shares the trtllm decode ops (3 per variant) and, within a KV group, GDN ops but for (i); (b) (ii) is correct as is (different
inputs are different reads); lifting one metadata per group is the adapter's choice; (c) (i) held for the user. The ~45 s of own
GDN confirmations sit behind (i); (a) alone saves ~1-2 s per run. No code.

### 10-21 decision (coordinator): ms38 as approved cancelled; (a) held, bundled with (i)
(a) (static-prefix layouts by role in the key, renamed by root position in the record match) is the template rule applied to
the match, but alone it is worth 1-2 s per run and changes the compared form: held. If the user approves (i) (LayerName by role,
put to them together with the cross-layer key-out item), (a) is built with it as one change, one argument, one gate; if (i) is
rejected, (a) stays unbuilt. (ii) stays as is (different lifted inputs are different reads).
