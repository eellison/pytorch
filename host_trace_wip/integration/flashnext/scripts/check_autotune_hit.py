"""A timing run's log against the unified FlashInfer autotune file (scripts/with_fi_autotune.py): which file it used, whether
FlashInfer loaded it ("Config cache hit"), and how many configs it added ("Saved N configs ... (M new"). Every arm but the
one that primes the file must add 0.
    python3 scripts/check_autotune_hit.py LOG"""
import re
import sys

text = open(sys.argv[1], errors="replace").read()
used = re.findall(r"FN_FI_AUTOTUNE_FILE (\S+) \(exists: (\w+)\)", text)
hits = len(re.findall(r"Config cache hit for flashinfer::", text))
new = [int(n) for n in re.findall(r"Saved \d+ configs to .*?\((\d+) new", text)]
primed = bool(used) and used[0][1] == "False"
ok = bool(used) and (primed or (hits > 0 and not any(new)))
print(f"AUTOTUNE {sys.argv[1].rsplit('/', 1)[-1]}: file {used[0][0] if used else None} existed {used[0][1] if used else None}; "
      f"cache hits {hits}; new configs {new}; {'OK' if ok else 'CHECK: this arm tuned on its own'}")
