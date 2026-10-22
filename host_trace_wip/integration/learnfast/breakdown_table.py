"""breakdown_table.py TITLE label=RUN.json:RUN.log[:OFFLINE_S] ...: breakdown27.py's rows as one markdown table, a column per run."""
import subprocess
import sys

cols, rows = [], {}
for spec in sys.argv[2:]:
    label, rest = spec.split("=", 1)
    parts = rest.split(":")
    out = subprocess.run([sys.executable, __file__.replace("breakdown_table.py", "breakdown27.py"), *parts], capture_output=True, text=True, check=True).stdout
    cols.append(label)
    for line in out.splitlines():
        k, v = (x.strip() for x in line.split("\t", 1))
        rows.setdefault(k, {})[label] = v
print(f"## {sys.argv[1]}\n")
print("| phase | " + " | ".join(cols) + " |")
print("|---|" + "---|" * len(cols))
for k, vals in rows.items():
    if all(vals.get(c, "-") in ("-", "None / None") for c in cols):
        continue
    print(f"| {k} | " + " | ".join(vals.get(c, "") for c in cols) + " |")
