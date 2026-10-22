"""Overlay (core agent, 10-22; ms40, the user's decision via the coordinator; labelled, our arm only): Qwen GDN's fused-norm-packed
core as a sibling custom op whose body reaches no layer object. The stock op takes `layer_name` and looks the layer up in the
forward context for its state (kv_cache[0] conv state, kv_cache[1] ssm state), weights (conv1d weight / bias, A_log, dt_bias,
norm weight / bias) and config; under a host trace that name token and the per-layer lookup are what keep the 48 layers' ops
apart. Here the per-layer tensors are explicit arguments, the config scalars come from one template layer (asserted identical
for every layer), and the step's metadata is found by the layer's KV-cache GROUP (a small int argument: the layers of a group
share one GDNAttentionMetadata object), so the ops of a group's layers are identical by construction and nothing per-layer is
keyed. Same kernels, same outputs: the body is the stock method, run on a proxy carrying the call's tensors.

    import sys; sys.path.insert(0, ".../land/core/learnfast"); import gdn_explicit   # before vLLM loads the module

Rewrite (import hook, as gdn_maskfill / fla_symlen): the layer forward's call of torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed
becomes _ht_explicit_core(self, ...), and the module gets the sibling op (vllm.qwen_gdn_attention_core_explicit), the proxy and
the group table appended. gdn_maskfill's rewrite is applied too when that overlay is loaded. PATCHED counts the rewrite.
"""
import importlib.abc
import importlib.machinery
import sys

MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
OLD = """            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                mixed_qkvz,
                ba,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
            )
"""
NEW = """            _ht_explicit_core(self, mixed_qkvz, ba, core_attn_out)  # host-trace overlay gdn_explicit: no layer lookup
"""
APPEND = '''

# ---- host-trace overlay (land/core/learnfast/gdn_explicit.py, ms40): the per-layer state as explicit arguments ----
from vllm.utils.torch_utils import direct_register_custom_op as _ht_register

_HT_FIELDS = ("key_dim", "value_dim", "tp_size", "head_k_dim", "head_v_dim", "num_k_heads", "num_v_heads", "activation",
              "layer_norm_epsilon", "gdn_decode_kernel", "enable_packed_recurrent_decode", "enable_fused_gdn_decode")
_HT_TEMPLATE: dict = {}  # the config shared by the GDN layers (from the first layer seen; asserted equal for the others)
_HT_GROUP: dict = {}  # layer prefix -> KV-cache group index (the layers whose prefixes map to one metadata object)
_HT_GROUP_PREFIX: dict = {}  # group index -> one prefix of the group: the metadata lookup key


class _HtHolder:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _HtProxy(QwenGatedDeltaNetAttention):
    """The stock core methods run on this instead of the layer: the state and the weights are the call's, the config the
    template's, the metadata the group's. nn.Module.__init__ is not run: the fields go straight into __dict__."""

    def __init__(self, conv_state, ssm_state, conv_weight, conv_bias, A_log, dt_bias, norm_weight, norm_bias, group):
        t, d = _HT_TEMPLATE, self.__dict__
        for k in _HT_FIELDS:
            d[k] = t[k]
        d["kv_cache"] = [conv_state, ssm_state]
        d["conv1d"] = _HtHolder(weight=conv_weight, bias=conv_bias)
        d["norm"] = _HtHolder(weight=norm_weight, bias=norm_bias, **t["norm"])
        d["A_log"], d["dt_bias"] = A_log, dt_bias
        d["chunk_gated_delta_rule"] = t["chunk_gated_delta_rule"]
        d["prefix"] = _HT_GROUP_PREFIX.get(group, "")

    def __getattr__(self, name):
        # any other attribute the stock methods read (config fields such as disable_tp_for_ba_proj, helper objects) comes from the
        # template layer, never a tensor or a module holding one: those are the call's explicit arguments
        v = getattr(_HT_TEMPLATE["layer"], name)
        if isinstance(v, torch.Tensor) or (isinstance(v, torch.nn.Module) and any(True for _ in v.parameters())):
            raise AttributeError(f"gdn_explicit: {name} is per-layer state the sibling op does not take")
        return v


def qwen_gdn_attention_core_explicit(
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor | None,
    group: int,
) -> None:
    proxy = _HtProxy(conv_state, ssm_state, conv_weight, conv_bias, A_log, dt_bias, norm_weight, norm_bias, group)
    QwenGatedDeltaNetAttention._forward_core_fused_norm_packed(proxy, mixed_qkvz=mixed_qkvz, ba=ba, core_attn_out=core_attn_out)


def _qwen_gdn_attention_core_explicit_fake(
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor | None,
    group: int,
) -> None:
    return


_ht_register(
    op_name="qwen_gdn_attention_core_explicit",
    op_func=qwen_gdn_attention_core_explicit,
    mutates_args=["core_attn_out", "conv_state", "ssm_state"],
    fake_impl=_qwen_gdn_attention_core_explicit_fake,
)


def _ht_group_of(layer):
    """The layer's KV-cache group: the index (in registration order) of the metadata object its prefix maps to among the
    GDN layers' objects; -1 while the forward context carries no metadata dict (vLLM's profiling and dummy runs)."""
    if not _HT_TEMPLATE:
        _HT_TEMPLATE.update({k: getattr(layer, k) for k in _HT_FIELDS})
        _HT_TEMPLATE["layer"] = layer  # the config template for attributes outside _HT_FIELDS (never its tensors)
        n = layer.norm
        _HT_TEMPLATE["norm"] = {"eps": n.eps, "group_size": n.group_size, "norm_before_gate": n.norm_before_gate, "activation": n.activation}
        _HT_TEMPLATE["chunk_gated_delta_rule"] = layer.chunk_gated_delta_rule
    else:
        for k in _HT_FIELDS:
            if getattr(layer, k) != _HT_TEMPLATE[k]:
                raise RuntimeError(f"gdn_explicit: {layer.prefix}.{k} = {getattr(layer, k)!r} differs from the template's {_HT_TEMPLATE[k]!r}")
    md = get_forward_context().attn_metadata
    if not isinstance(md, dict) or layer.prefix not in md:
        return -1
    groups: dict = {}
    for prefix, m in md.items():
        if isinstance(m, GDNAttentionMetadata):
            g = groups.setdefault(id(m), len(groups))
            _HT_GROUP[prefix] = g
            _HT_GROUP_PREFIX.setdefault(g, prefix)
    return _HT_GROUP.get(layer.prefix, -1)


def _ht_explicit_core(layer, mixed_qkvz, ba, core_attn_out):
    g = _HT_GROUP.get(layer.prefix)
    if g is None:
        g = _ht_group_of(layer)
    if g < 0 or "kv_cache" not in layer.__dict__:
        # vLLM's profiling and dummy runs: no metadata dict, no KV cache bound yet; the stock op runs its warm-up path there
        # (nothing is traced in those phases)
        torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(mixed_qkvz, ba, core_attn_out, layer_name=_encode_layer_name(layer.prefix))
        return
    torch.ops.vllm.qwen_gdn_attention_core_explicit(
        mixed_qkvz, ba, core_attn_out, layer.kv_cache[0], layer.kv_cache[1], layer.conv1d.weight, layer.conv1d.bias,
        layer.A_log, layer.dt_bias, layer.norm.weight, layer.norm.bias, g,
    )
'''
PATCHED: list = []


def rewrite(src: str) -> str:
    if OLD not in src:
        raise RuntimeError("gdn_explicit: the fused-norm-packed call site is not the expected text (vLLM version changed?)")
    new = src.replace(OLD, NEW) + APPEND
    mf = sys.modules.get("gdn_maskfill")
    if mf is not None:
        new = new.replace(mf.OLD, mf.NEW)
    return new


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None or not getattr(spec, "origin", None):
            return spec
        origin = spec.origin

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                src = open(origin).read()
                new = rewrite(src)
                PATCHED.append(name)
                exec(compile(new, origin, "exec"), module.__dict__)

        spec.loader = _Loader()
        return spec


if not any(isinstance(f, _Hook) for f in sys.meta_path):
    sys.meta_path.insert(0, _Hook())
