"""Overlay (flashnext lane, 10-21; coordinator-approved, labelled; tracing-forced, our arm only; stock arms never load it).

vLLM's _qwen4_exp_low_latency_gemm (vllm/models/qwen4_exp/nvidia/low_latency_gemm.py) picks its skinny-GEMM plan with
QWEN4_EXP_GEMM_PLANS[(N, K)].get(x.shape[0]). Under a host trace a dict lookup on the symbolic token count M specializes to the
guard M == <value> on every call, hit or miss, so each token count traces its own variant (STATUS.md, 10-21 census: ~330 of 352
variants), although for M > 16 the body always falls through to F.linear. install() replaces each plan dict whose keys are a
subset of {1, 2, 4, 8, 16} by a dict subclass whose get(M) answers None for M > 16 before any lookup (the trace records the
range guard M > 16 there) and, for M <= 16, compares M with each plan key by == instead of hashing it (coordinator 10-21, 1b):
a hit records M == k and takes that plan, a miss records the inequalities M != k for the plan keys and takes F.linear, so every
missing token count below 17 shares one variant instead of pinning its own. Exactly equivalent to the original lookup: the
same plan for a key, None otherwise, the same kernels and outputs. A plan dict with any other key is left untouched (the
original lookup, exact-M guards as before) and reported. Upstream vLLM candidate (for the user to decide):
`if x.shape[0] > 16: return F.linear(x, weight)` before the lookup."""
import os

LIMIT = 16
ALLOWED = frozenset({1, 2, 4, 8, 16})
STATS: dict = {"installed": [], "fell_through": []}


class _RangePlan(dict):
    """A plan dict (M -> SkinnyGemmConfig) whose keys lie within ALLOWED: get(M) is None for M > LIMIT without hashing M, and
    below that compares M with each key by == (a traced SymInt records each relation; nothing hashes M)."""

    def __init__(self, plan):
        super().__init__(plan)
        self.keys_sorted = tuple(sorted(dict.keys(self)))

    def get(self, m, default=None):
        if m > LIMIT:  # the range guard
            return default
        for k in self.keys_sorted:
            if m == k:  # M == k on a hit; M != k for each key on a miss
                return dict.get(self, k)
        return default


def install() -> dict:
    import vllm.models.qwen4_exp.nvidia.low_latency_gemm as llg

    plans = llg.QWEN4_EXP_GEMM_PLANS
    for nk, plan in list(plans.items()):
        if isinstance(plan, _RangePlan):
            continue
        keys = set(plan.keys())
        if keys <= ALLOWED:
            plans[nk] = _RangePlan(plan)
            STATS["installed"].append((nk, sorted(keys)))
        else:
            STATS["fell_through"].append((nk, sorted(keys)))
    print(f"FN_PLAN_RANGE: {len(STATS['installed'])} plan dicts answer M > {LIMIT} by a range guard (keys within {sorted(ALLOWED)}); "
          f"left as the original lookup: {STATS['fell_through']}", flush=True)
    return STATS
