# Phase 6 step 4c diagnostic: where the guards that split a family's variants are recorded. Runs bench/rangesweep.py
# (same args) with IR.Env._record wrapped; every new guard record whose text matches GUARD_RE gets its Python stack
# (outside torch/cuda/_host_trace*) filed under the guard text. Prints the distinct stacks per guard.
#   CUDA_VISIBLE_DEVICES=2 GUARD_RE='...' bash python_vllm_cpp.sh bench/guardorigin.py --out bench/out/range/X.json [rangesweep args]
import collections, json, os, re, runpy, sys, traceback

import torch.cuda._host_trace_ir as IR

PAT = re.compile(os.environ.get("GUARD_RE", ".*"))  # rendered IR text (not sympy); all records by default, dumped to OUT.guards.json
STACKS = collections.defaultdict(collections.Counter)  # guard text -> stack -> count
_record = IR.Env._record


def record(self, g, written):
    n = len(self.records)
    _record(self, g, written)
    if len(self.records) > n:
        txt = str(g)
        if PAT.search(txt):
            fr = [f for f in traceback.extract_stack()[:-1] if "_host_trace" not in f.filename and "/torch/utils/" not in f.filename]
            STACKS[txt]["".join(traceback.format_list(fr[-8:]))] += 1


IR.Env._record = record
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "rangesweep.py"), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
out = sys.argv[sys.argv.index("--out") + 1] + ".guards.json"
with open(out, "w") as f:
    json.dump({t: [[c, s] for s, c in st.most_common(3)] for t, st in STACKS.items()}, f, indent=1)
print("=== guard origins:", len(STACKS), "guards; dumped to", out, flush=True)
SHOW = re.compile(os.environ.get("GUARD_SHOW", r"\b(1|16|17|8192)\b"))
for txt, st in sorted(STACKS.items()):
    if not SHOW.search(txt) or len(txt) > 80:
        continue
    print(f"### {txt}  records {sum(st.values())} stacks {len(st)}", flush=True)
    for s, c in st.most_common(3):
        print(f"--- x{c}\n{s}", flush=True)
