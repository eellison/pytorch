# CUDA runtime/driver improvement suggestions (with measurements)

## 2026-10-08: per-node graph exec updates (learn-cost lane, GB300, land/core/learncost/probe_setparams.py)
- Today: cuGraphExecKernelNodeSetParams costs ~0.42 us host per node when params change (0.15 us unchanged), plus ~0.27 us per node applied at the next launch. A fluctuating-batch decode step changes ~360-900 nodes, each differing only by a 2-byte size scalar.
- cudaGraphExecUpdate of the whole graph is no cheaper (0.47 us/node).
- Suggestions: (1) a batched setter, cuGraphExecKernelNodesSetParams(exec, nodes[], params[], n), or an (offset, value) patch list per exec; (2) graph-level parameter slots: a kernel parameter read from a device location (like conditional handles), so a size change is one 8-byte copy instead of N node updates; (3) cheaper device-side grid updates (25-30 us each today).
