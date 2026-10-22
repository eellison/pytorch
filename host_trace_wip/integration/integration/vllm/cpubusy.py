# Busy cores on CPUs [lo, hi] over a window from /proc/stat: python3 cpubusy.py [secs] [lo] [hi] -> "busy_cores max_core_busy%"
import sys, time
secs, lo, hi = float(sys.argv[1]) if len(sys.argv) > 1 else 5, int(sys.argv[2]) if len(sys.argv) > 2 else 0, int(sys.argv[3]) if len(sys.argv) > 3 else 71
def snap():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split(); c = int(f[0][3:]); v = list(map(int, f[1:9]))
            if lo <= c <= hi: out[c] = (sum(v), v[3] + v[4])
    return out
a = snap(); time.sleep(secs); b = snap()
busy = [1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in a]
print(f"{sum(busy):.1f} {100 * max(busy):.0f}")
