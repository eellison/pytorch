"""merge_top.py <top>: symti's Python (py/, over its base tmp/mkpatches/base) three-way merged onto
int6/snap/<top> in tmp/merge_<top>; writes patches/symti_py.<top>.patch (top -> merged). Conflicts are
resolved by the rules below (the witness and its helpers are symti's to delete); any other conflict stops it."""
import os, re, shutil, subprocess, sys

L = "/data/eellison/src/pytorch/agent_space/paramgraph/land"
S = f"{L}/core/symti"
top = sys.argv[1]
T = f"{L}/int6/snap/{top}"
M = f"{S}/tmp/merge_{top}"
files = [l[6:].strip() for l in open(f"{S}/patches/symti_py.patch") if l.startswith("--- a/")]
shutil.rmtree(M, ignore_errors=True)
subprocess.run(f"mkdir -p {M} && cd {T} && tar cf - torch/cuda torch/_meta_registrations.py test/test_cuda_host_trace_*.py | (cd {M} && tar xf -)", shell=True, check=True)


def blocks(text):
    out, i, lines = [], 0, text.split("\n")
    res = []
    while i < len(lines):
        if lines[i].startswith("<<<<<<< "):
            j = i + 1
            ours = []
            while not lines[j].startswith("======="):
                ours.append(lines[j]); j += 1
            j += 1
            theirs = []
            while not lines[j].startswith(">>>>>>> "):
                theirs.append(lines[j]); j += 1
            res.append(("c", ours, theirs)); i = j + 1
        else:
            res.append(("t", lines[i])); i += 1
    return res


WITNESS = ("_pointwise_host", "_traced_pointwise", "_witness_takes", "_UNTAGGED_ELEMENTWISE", "_NULLARY", "_to_copy(", "witness")


def choose(f, ours, theirs):
    o, t = "\n".join(ours), "\n".join(theirs)
    if f.endswith("_host_trace_hint_audit.py"):
        return [l for l in theirs if not ('(_TAPE, "_elementwise")' in l or '"_Trace._pointwise_host' in l or "owner: the symti lane" in l)]
    if "_INT_READS" in o and "_is_non_overlapping_and_dense(" in t:
        return theirs + [l for l in ours if "_INT_READS" in l or "tr.int_read" in l]
    if any(w in t for w in WITNESS):
        keep = [l for l in theirs if l.startswith("_TO_COPY_KWARGS")]
        return keep + ([""] * 2 if keep else []) + ours
    if o.strip() == "self.unconverted = tuple(tr.unconverted)":
        return theirs + ours  # both add a field at the Tape's end
    if f.endswith("test_cuda_host_trace_aten.py"):
        return ours
    if f.endswith("test_cuda_host_trace_replay.py") and not t.strip():
        return theirs  # a native test the odd line leaves out
    raise SystemExit(f"unresolved conflict in {f}:\n--- ours\n{o[:2000]}\n--- theirs\n{t[:2000]}")


for f in files:
    r = subprocess.run(["git", "merge-file", "-p", f"{S}/py/{f}", f"{S}/tmp/mkpatches/base/{f}", f"{M}/{f}"], capture_output=True, text=True)
    if r.returncode < 0 or r.returncode > 127:
        raise SystemExit(f"git merge-file failed on {f}: {r.stderr}")
    out = []
    for b in blocks(r.stdout):
        out += [b[1]] if b[0] == "t" else choose(f, b[1], b[2])
    open(f"{M}/{f}", "w").write("\n".join(out))
    print(f, "conflicts", r.returncode, r.stderr.strip()[:200])

p = f"{M}/torch/cuda/_host_trace_tape.py"
s = open(p).read()
if "_witness_pools = threading.local()\n" in s:
    a = s.index("_witness_pools = threading.local()\n")
    b = s.index("            outer += pools\n", a) + len("            outer += pools\n")
    s = s[:a] + s[b:].lstrip("\n")
    s = s.replace("with _witness_pools_freed(), private_stream(device) as stream,", "with private_stream(device) as stream,")
if "threading." not in s:
    s = s.replace("\nimport threading\n", "\n", 1)
if "_TO_COPY_KWARGS)" in s and "_TO_COPY_KWARGS = " not in s:
    raise SystemExit("_TO_COPY_KWARGS used but gone")
open(p, "w").write(s)
p = f"{M}/test/test_cuda_host_trace_tape.py"
s = open(p).read()
s = s.replace("""    def test_witnesses_free_their_pools(self):
        # a pointwise witness's output allocates in its capture's pool, which
        # the trace frees, not leaves cached where no one captures again
""", """    def test_a_trace_leaves_no_private_pool_memory(self):
        # a trace of pointwise ops caches nothing in a private pool, where no
        # one allocates again
""")
open(p, "w").write(s)
p = f"{M}/test/test_cuda_host_trace_replay.py"
s = open(p).read()
if '"aten.index.Tensor returned other metadata"' in open(f"{T}/test/test_cuda_host_trace_replay.py").read():
    # the redispatch lane's tops (RD2-RD5): its redispatch reruns index's own host
    old = """        # traced at len(i) == len(j) (one symbol): at (1, 8) index's own host expands i (eager's
        # expand_outplace), which the trace's dispatch did not: the redispatch refuses, a second variant
        cases[((8, 8), (1, 8), (8, 8), (4, 4))] = (2, 2, 0, {"aten.index.Tensor dispatches otherwise": 1})
"""
    new = """        # traced at len(i) == len(j) (one symbol): at (1, 8) the redispatch runs index's own host
        # over the new sizes (its expand_outplace of i included)
        cases[((8, 8), (1, 8), (8, 8), (4, 4))] = (1, 3, 1, {})
"""
    if old not in s:
        raise SystemExit("the index guard test moved")
    s = s.replace(old, new)
open(p, "w").write(s)
out = f"{S}/patches/symti_py.{top}.patch"
with open(out, "w") as fh:
    for f in sorted(os.listdir(f"{M}/torch/cuda")):
        pass
    names = [f"torch/cuda/{n}" for n in sorted(os.listdir(f"{M}/torch/cuda")) if n.startswith("_host_trace") and n.endswith(".py")]
    names += ["torch/_meta_registrations.py"]
    names += [f"test/{n}" for n in sorted(os.listdir(f"{M}/test")) if n.startswith("test_cuda_host_trace_") and n.endswith(".py")]
    for n in names:
        d = subprocess.run(["diff", "-u", "--label", f"a/{n}", "--label", f"b/{n}", f"{T}/{n}", f"{M}/{n}"], capture_output=True, text=True).stdout
        fh.write(d)
flk = subprocess.run([sys.executable, "-m", "pyflakes"] + [f"{M}/{n}" for n in names], capture_output=True, text=True).stdout
bad = [l for l in flk.splitlines() if "undefined name" in l or "redefinition of unused" in l or ("imported but unused" in l and "_host_trace_tape" in l)]
print("pyflakes:", bad or "clean")
print("wrote", out)
