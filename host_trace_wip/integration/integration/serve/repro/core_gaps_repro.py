# Torch-level repros for the two core gaps behind R1-V's 0 variants (CORE_TASKS.md, serve items). No vLLM.
#   gap_1b  an eager step (an op with no traced implementation, here a torch.library op with a Python CUDA kernel) whose
#           int argument is t.data_ptr() of a tensor allocated inside the trace. vLLM/FlashInfer: mm_fp4's cute-dsl GEMM
#           takes its scale factors as make_ptr pointers (data_ptr() ints); unobserved or declined by the CuTe route it is
#           an eager step with such a scalar. Today: the lowering of the eager step declines the whole trace
#           ("sN is not read from the call's inputs", allocK.base). Want: a variant whose eager step gets the replay's
#           address (the op returns x + 1 when the address it gets is x's, x + 100 otherwise).
#   gap_4   an unpinned CPU tensor argument (vLLM's GDN prefill metadata carries host bookkeeping tensors, e.g. the per-
#           request chunk counts, built unpinned on the host). Today: "argN is on cpu; only CUDA tensors on one device and
#           pinned CPU tensors are traced". Want: a host input re-read per call (or a clear rule that host values key the call).
#   gap_5   a keyed (harvested) op inside a traced custom op that has its own guards. Flash-Next: vLLM's
#           vllm.qwen4_exp_qsa_with_output (direct_register_custom_op, traced through: a selector over its 7 launches) calls
#           _C_cache_ops.reshape_and_cache_flash (an extern, a keyed site); both claim the cache-write launch and the native
#           variant rejects the spec: ValueError "a site's record 174" from torch._C._HostTraceVariant, re-raised as
#           AssertionError("host_trace: the native variant rejects: ...") out of the entry call (the engine dies). Here: a
#           torch.library op whose Python CUDA kernel branches on a size (its own guard) around a harvested mm.
#   gap_2   an extern (harvested) op that writes a float8 output. HarvestProvider's refill fills written float operands with
#           uniform_, which float8 does not implement: NotImplementedError escapes as a trace decline. Want: a harvested
#           binding (or at worst a refusal of the key, an eager step), replay bitwise at two sizes.
#   gap_3   work on a side stream inside the region (vLLM's MoE shared experts run on their own stream inside the
#           vllm.moe_forward_shared custom op: "aten.mm.default on a stream other than the trace's"). Inline form
#           (side stream + wait_stream both ways) and the custom-op form. Want: the side-stream work in the graph (fork /
#           join edges), replay bitwise; today an eager step (custom op) or a decline.
# Run: CUDA_VISIBLE_DEVICES=1 bash serve/python_vllm_cand3.sh serve/repro/core_gaps_repro.py   (any host-trace build)
import torch
import torch.cuda._host_trace_replay as R
from torch.cuda._host_trace_harvest import HarvestProvider

lib = torch.library.Library("htrepro", "DEF")
lib.define("at_address(Tensor x, int address) -> Tensor")
lib.define("to_fp8(Tensor x, Tensor(a!) out) -> ()")


@torch.library.impl(lib, "at_address", "CUDA")
def _at_address(x, address):
    return x + (1 if address == x.data_ptr() else 100)


@torch.library.impl(lib, "to_fp8", "CUDA")
def _to_fp8(x, out):
    out.copy_(x.to(torch.float8_e4m3fn))


lib.define("shared_side(Tensor x, Tensor w) -> Tensor")
_side = None


@torch.library.impl(lib, "shared_side", "CUDA")
def _shared_side(x, w):
    global _side
    _side = _side or torch.cuda.Stream()
    cur = torch.cuda.current_stream()
    _side.wait_stream(cur)
    with torch.cuda.stream(_side):
        s = x @ w
    y = torch.relu(x @ w)
    cur.wait_stream(_side)
    return y + s


torch.library.register_fake("htrepro::shared_side", lambda x, w: torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=x.dtype), lib=lib)
lib.define("nested_extern(Tensor x, Tensor w) -> Tensor")


@torch.library.impl(lib, "nested_extern", "CUDA")
def _nested_extern(x, w):
    y = torch.mm(x, w)  # a keyed site (blas provider) inside the op
    return y * 2 if x.shape[0] > 16 else y + 1  # the op's own guard


torch.library.register_fake("htrepro::nested_extern", lambda x, w: torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=x.dtype), lib=lib)
torch.library.register_fake("htrepro::at_address", lambda x, address: torch.empty_like(x), lib=lib)
torch.library.register_fake("htrepro::to_fp8", lambda x, out: None, lib=lib)


def gap_1b(x):
    t = torch.empty_like(x)
    t.copy_(x * 2)
    return (torch.ops.htrepro.at_address(t, t.data_ptr()),)


def gap_2(x):
    out = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    torch.ops.htrepro.to_fp8.default(x * 0.5, out)
    return (out.view(torch.uint8) + 0,)


W = torch.randn(64, 64, device="cuda")
SIDE = torch.cuda.Stream()


def gap_3_inline(x):
    cur = torch.cuda.current_stream()
    SIDE.wait_stream(cur)
    with torch.cuda.stream(SIDE):
        s = x @ W
    y = torch.relu(x @ W)
    cur.wait_stream(SIDE)
    return (y + s,)


def gap_3_op(x, w):
    return (torch.ops.htrepro.shared_side(x * 2, w) + 1,)  # traced work around the op, so the op is one step of a variant


def gap_5(x, w):
    return (torch.ops.htrepro.nested_extern(x + 1, w) - 1,)


def gap_4(x, n):
    # n: an unpinned CPU int64 tensor of per-row counts, read on the host to size the work (as vLLM's nums_dict)
    k = int(n.sum())
    return (x[:k] * 2,)


cases = (("gap_5", gap_5, (HarvestProvider(("blas",)),)), ("gap_4", gap_4, ()), ("gap_3_inline", gap_3_inline, (HarvestProvider(("blas",)),)), ("gap_3_op", gap_3_op, (HarvestProvider(("blas",)),)), ("gap_1b", gap_1b, ()), ("gap_2", gap_2, (HarvestProvider(("extern",), extern_ops=(torch.ops.htrepro.to_fp8.default,)),)))
for name, fn, opaque in cases:
    entry = R.HostTraceReplay(fn, opaque=opaque, memory="run_buffer")
    ok = []
    err = None
    for m in (8, 40, 8, 40, 100):
        x = torch.randn(m, 64, device="cuda")
        extra = (W,) if fn in (gap_3_op, gap_5) else (torch.tensor([m // 4, m // 4], dtype=torch.int64),) if fn is gap_4 else ()
        try:
            got = entry(x, *extra)
        except Exception as e:  # an escaping error (should have been a decline)
            err = f"{type(e).__name__}: {e}"
            break
        want = fn(x, *extra)
        ok.append(all(torch.equal(g, h) for g, h in zip(got, want)))
    print(name, "traces", entry.traces, "replays", entry.replays, "eager", entry.eager, "variants", len(entry.variants), "bitwise", ok,
          "declines", sorted({str(d) for d in entry.declines})[:1], "retrace_causes", dict(list(getattr(entry, "retrace_causes", {}).items())[:3]), "escaped", err, flush=True)
