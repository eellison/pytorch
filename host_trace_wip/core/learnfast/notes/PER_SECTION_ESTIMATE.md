# Per-section composition: what it would cut (analysis only, learnfast 10-16)

Data: rs6 (wip14, edge + derived warm-up phases, 800-step stream; DIAG27_FAILING_GUARDS logs, per redispatch attempt, the
failing guards of the first unselected selector of each op) and the ms15 cut reports (memset/split/out/27b_*_rms15_r1_*.json:
4 pieces per run, bounds e.g. prefill [0, 43, 215, 903, 3628], mixed [0, 47, 235, 987, 3948]).

| quantity | today (rs6) | with per-section composition (estimate) |
|---|---|---|
| redispatch attempts | 82 warm-up + 10 stream | one per distinct (section, op class): 25-50 |
| op-class records (op x attempt) | 119 (106 warm-up, 13 stream) | - |
| distinct (op, failing-guard set) | 25 (23 in warm-up; 2 new in the stream) | lower bound on the classes |
| warm-up redispatch seconds | 207 s (2.5 s per attempt) | about 60-110 s |
| stream redispatches | 10 | about 2 (only the 2 classes the warm-up never reached) |
| traces | 6 (3 first calls of an entry form, 2 graph-level count guards, 1 GDN pin + graph guard) | 3-6 (form traces stay; the count-guard ones only go if their guard is section-local) |

- Per op: mm_fp4 56 records / 4 signatures, GDN 30 / 17, attention decode 27 / 1, silu_and_mul 3 / 1, embedding 3 / 2.
- The signature undercounts classes where one guard expression covers several classes (attention's split count
  min(ceil(L/512), ...) fails with the same text for every L class; mm_fp4's pow2 buckets likewise). Taking the classes the
  guards actually separate (attention ~9 split counts up to the stream's contexts, mm_fp4 ~16-20 buckets / tile classes, GDN 17,
  3 others) gives the ~50 upper bound.
- Why it cuts: today a class is redone once per variant it is reached in (11 of the 13 stream records repeat a signature an
  earlier attempt already redispatched, in another variant), and each attempt redoes every failing op of the call together.
  Composed per section, a class is redone once and every variant's section takes it.
- Pieces from the cut report: 4 per run; the last piece holds ~75% of the nodes (2725 of 3628 prefill records), so piece-level
  sections give little isolation beyond op classes; the saving comes from cross-variant reuse, not from the piece split itself.
