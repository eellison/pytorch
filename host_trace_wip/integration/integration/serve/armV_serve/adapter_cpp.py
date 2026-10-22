# Arm V-cpp level 2 (install_sgl_cpp, read only; launcher python_vllm_cpp.sh; bench/drive.py picks it there unless
# ARMV_BOUND=0): decode steps with traced metadata and traced prepare_inputs go through torch._C._HostTraceBound
# (src_sglcpp CHANGES C3, C6, C7): one C++ call checks the plan's guards, builds the entry's arguments behind the
# bound static prefix and dispatches. ArmV.forward's Python key build (_split, _input_sig) and argument list are what
# it replaces; any call no plan takes (prefill, mixed, a first call, a non-hit) runs ArmV's Python path unchanged.
#
# The plan's only host guard is the identity of p.key, a decode key object of self.entries set by ms.prepare_attn's
# wrapper from (num_reqs, recorded launches) once the Python path computed that key for them. Under _fast_key's
# conditions (no prefill, one token per request, no padding, cudagraph mode NONE, not for capture) the Python key is a
# function of those two: _split gives (T, n, n, 0, 0, None, None) and the recorded launches' host values are n and
# constants. ARMV_BOUND_VERIFY=1 recomputes the Python key on every fast step and asserts it is p.key.
import os

import torch

import adapter
from adapter import _Pending, ArmV

BOUND = os.environ.get("ARMV_BOUND", "1") == "1" and hasattr(torch._C, "_HostTraceBound") and adapter.STATIC_PREFIX
VERIFY = os.environ.get("ARMV_BOUND_VERIFY") == "1"
P, ADAPTER, NB = 0, 1, 2  # the bound's sources: the step's _Pending (derived: adapter.cur), the adapter, block_tables.num_blocks
HEAD = ("b_idx", "b_qsl", "b_seq")  # p's attributes for input_batch.idx_mapping, query_start_loc, seq_lens


class ArmVcpp(ArmV):
    def __init__(self, runner):
        super().__init__(runner)
        if not (BOUND and self.md_trace and self.in_trace):
            raise RuntimeError("arm V-cpp: needs _HostTraceBound, static_prefix, traced metadata and inputs")
        self.cur = None  # the step's _Pending, set by prepare_attn; cleared by a hit
        self.dec_keys = {}  # (num_reqs, recorded launches) -> the decode key (an object of self.entries) its plan guards
        self.tnames = []  # p's attributes for the recorded launches' tensors
        self.bound = torch._C._HostTraceBound(self.prefix, 2, 0, (self, runner.block_tables.num_blocks), derived=((0, "cur"),),
                                              resets=((P, "inputs_done", True), (ADAPTER, "cur", None)), trust_statics=True)

    def install(self):
        super().install()
        r, ms = self.runner, self.runner.model_state
        orig_attn, orig_ms = r.prepare_attn, ms.prepare_attn

        def prepare_attn(input_batch):
            p, slots = orig_attn(input_batch)
            self.cur = p
            return p, slots

        def ms_prepare_attn(*args, **kw):
            res = orig_ms(*args, **kw)
            if isinstance(res, _Pending):
                res.key = self._fast_key(res)
            return res
        r.prepare_attn, ms.prepare_attn = prepare_attn, ms_prepare_attn

    def _fast_form(self, p):
        """(num_reqs, launches) when p's step is one _fast_key covers, else None."""
        ib, md = p.input_batch, p.md_args
        n = ib.num_reqs
        if (md is None or md[0] != self.cg_none or md[3] or not p.inputs or ib.has_prefill or ib.num_tokens != n
                or ib.num_tokens_after_padding != n or ib.num_reqs_after_padding != n):
            return None
        return n, len(p.inputs)

    def _fast_key(self, p):
        form = self._fast_form(p)
        key = self.dec_keys.get(form) if form is not None else None
        if key is None:
            return None
        ib = p.input_batch
        tensors = [a for _, args, kw, out in p.inputs for a in (*args, *kw.values(), out) if isinstance(a, torch.Tensor)]
        if len(tensors) > len(self.tnames):
            self.tnames += [f"t{i}" for i in range(len(self.tnames), len(tensors))]
        vars(p).update(zip(HEAD, (ib.idx_mapping, ib.query_start_loc, ib.seq_lens)))
        vars(p).update(zip(self.tnames, tensors))
        if VERIFY:
            want = self._entry_key("decode", ib.num_tokens_after_padding, self._split(ib), self._input_sig(p.inputs)[0])
            if want != key:
                raise AssertionError(f"arm V-cpp: fast key {key} is not the Python key {want}")
        return key

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        p = self.cur
        if (p is not None and p.key is not None and not self.check and intermediate_tensors is None and inputs_embeds is None
                and p.input_batch.input_ids is input_ids and p.input_batch.positions is positions):
            if adapter.VP and not self._vp_ok(p.input_batch):
                return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
            out = self.bound(input_ids, positions)
            if out is not NotImplemented:
                if len(out) > 1:
                    self.vp_res = (p.input_batch, out[1:])
                return out[0]
        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)

    def _forward_deferred(self, t0, fc, input_ids, positions):
        p = fc.attn_metadata
        out = super()._forward_deferred(t0, fc, input_ids, positions)
        form = self._fast_form(p)
        if form is not None and form not in self.dec_keys and self._same_inputs(p.input_batch, input_ids, positions):
            ib = p.input_batch
            sig, tensors = self._input_sig(p.inputs)
            ints = self._split(ib)
            key = self._entry_key("decode", input_ids.shape[0], ints, sig)
            entry = self.entries.get(key)
            if entry is not None and ("decode", input_ids.shape[0], ints, "md", sig) not in self.bad:
                key = next(k for k in self.entries if k == key)  # the entries' own object: the plan guards its identity
                self.dec_keys[form] = key
                names = list(self.tnames) + [f"t{i}" for i in range(len(self.tnames), len(tensors))]
                self.tnames = names
                spec = [(P, n) for n in HEAD] + [(NB, "gpu")] + [(P, n) for n in names[:len(tensors)]]
                self.bound.add(entry, (_Pending,), ((P, "key", key),), (), (), tuple(spec), (), None, ())
        return out

    def summary(self):
        res = super().summary()
        res["bound"] = {"hits": sum(self.bound.hits()), "plans": len(self.dec_keys)}
        return res
