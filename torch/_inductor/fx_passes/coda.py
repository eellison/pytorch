# mypy: allow-untyped-defs
import math
import operator
import threading

import torch
from torch._dynamo.utils import counters
from torch._higher_order_ops.flex_gemm import flex_gemm_hop
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    fwd_only,
    init_once_fakemode,
    joint_fwd_bwd,
    Match,
    register_replacement,
)
from torch._inductor.utils import ensure_cute_available
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq
from torch.utils._ordered_set import OrderedSet


aten = torch.ops.aten
_RMS_PARTIAL_GROUP = 16
_MIN_M = 128
_MIN_HIDDEN = 1024
_MIN_OUTPUT = 128
_MAX_FUSIBLE_WALK_DEPTH = 8
_MAX_FUSIBLE_WALK_NODES = 32
_CODA_INIT_LOCK = threading.Lock()
_CODA_INITIALIZED = False


def _rms_norm_mm_pattern(x, w0, residual, gamma, w1, eps):
    hidden = torch.mm(x, w0) + residual
    normalized = torch.nn.functional.rms_norm(hidden, (hidden.shape[-1],), gamma, eps)
    return torch.mm(normalized, w1)


def _rms_norm_mm_pattern_residual_first(x, w0, residual, gamma, w1, eps):
    hidden = residual + torch.mm(x, w0)
    normalized = torch.nn.functional.rms_norm(hidden, (hidden.shape[-1],), gamma, eps)
    return torch.mm(normalized, w1)


def _rms_norm_mm_reassociation_with_aux(x, w0, residual, gamma, w1, eps):
    hidden = torch.mm(x, w0) + residual
    hidden_float = hidden.float()
    inv_rms = torch.rsqrt(hidden_float.square().mean(dim=-1, keepdim=True) + eps)
    gamma_row = gamma.view(1, -1)
    weighted = (hidden_float * gamma_row).to(hidden.dtype)
    projected = torch.mm(weighted, w1)
    projected_float = projected.float()
    result = (projected_float * inv_rms).to(projected.dtype)
    return result, hidden_float, inv_rms, gamma_row, weighted, projected_float


def _rms_norm_mm_reassociation(x, w0, residual, gamma, w1, eps):
    result, *_ = _rms_norm_mm_reassociation_with_aux(x, w0, residual, gamma, w1, eps)
    return result


def _rms_norm_mm_replacement(x, w0, residual, gamma, w1, eps):
    counters["inductor"]["coda_rms_norm_rewrite"] += 1
    return _rms_norm_mm_reassociation(x, w0, residual, gamma, w1, eps)


def _rms_norm_mm_flex_replacement(x, w0, residual, gamma, w1, eps):
    counters["inductor"]["coda_rms_norm_fusion"] += 1
    gamma_row = gamma.view(1, -1)

    def first_body(a, b, residual_arg, gamma_arg):
        hidden = torch.mm(a, b) + residual_arg
        hidden_float = hidden.float()
        partial_rms = (
            hidden_float.square()
            .view(hidden.shape[0], -1, _RMS_PARTIAL_GROUP)
            .sum(dim=-1)
        )
        weighted = (hidden_float * gamma_arg).to(hidden.dtype)
        return weighted, partial_rms

    weighted, partial_rms = flex_gemm_hop(
        aten.mm.default,
        first_body,
        (x, w0, residual, gamma_row),
        {},
        {"backend": "QUACK"},
    )
    inv_rms = torch.rsqrt(partial_rms.sum(dim=-1, keepdim=True) / gamma.shape[0] + eps)

    def second_body(a, b, inv_rms_arg):
        projected = torch.mm(a, b)
        return (projected.float() * inv_rms_arg).to(projected.dtype)

    return flex_gemm_hop(
        aten.mm.default,
        second_body,
        (weighted, w1, inv_rms),
        {},
        {"backend": "QUACK"},
    )


def _rms_norm_mm_flex_training_replacement(x, w0, residual, gamma, w1, eps):
    counters["inductor"]["coda_rms_norm_fusion"] += 1
    gamma_row = gamma.view(1, -1)

    # Return the forward values consumed by the already-jointed backward as
    # auxiliary HOP outputs, so FlexGEMM does not need an autograd formula.
    def first_body(a, b, residual_arg, gamma_arg):
        hidden = torch.mm(a, b) + residual_arg
        hidden_float = hidden.float()
        partial_rms = (
            hidden_float.square()
            .view(hidden.shape[0], -1, _RMS_PARTIAL_GROUP)
            .sum(dim=-1)
        )
        weighted = (hidden_float * gamma_arg).to(hidden.dtype)
        return weighted, partial_rms, hidden_float

    weighted, partial_rms, hidden_float = flex_gemm_hop(
        aten.mm.default,
        first_body,
        (x, w0, residual, gamma_row),
        {},
        {"backend": "QUACK"},
    )
    inv_rms = torch.rsqrt(partial_rms.sum(dim=-1, keepdim=True) / gamma.shape[0] + eps)

    def second_body(a, b, inv_rms_arg):
        projected = torch.mm(a, b)
        projected_float = projected.float()
        result = (projected_float * inv_rms_arg).to(projected.dtype)
        return result, projected_float

    result, projected_float = flex_gemm_hop(
        aten.mm.default,
        second_body,
        (weighted, w1, inv_rms),
        {},
        {"backend": "QUACK"},
    )
    return result, hidden_float, inv_rms, gamma_row, weighted, projected_float


def _node_tensor(match: Match, name: str) -> tuple[torch.fx.Node, torch.Tensor] | None:
    node = match.kwargs.get(name)
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    return node, value


def _has_fusible_reduction_consumer(node: torch.fx.Node) -> bool:
    from .fusion_regions import is_fusible_node

    pending = [(user, 0) for user in node.users]
    visited = OrderedSet[torch.fx.Node]()
    while pending and len(visited) < _MAX_FUSIBLE_WALK_NODES:
        user, depth = pending.pop()
        if user in visited or user.op != "call_function":
            continue
        visited.add(user)
        target = user.target
        if (
            isinstance(target, torch._ops.OpOverload)
            and torch.Tag.reduction in target.tags
        ):
            return True
        if depth < _MAX_FUSIBLE_WALK_DEPTH and (
            target is operator.getitem or is_fusible_node(user)
        ):
            pending.extend((next_user, depth + 1) for next_user in user.users)
    return False


def _has_competing_consumer(node: torch.fx.Node) -> bool:
    non_output_users = [user for user in node.users if user.op != "output"]
    if len(non_output_users) > 1 or _has_fusible_reduction_consumer(node):
        return True
    return any(
        user.op == "call_function"
        and user.target
        in (
            aten.mm.default,
            aten.addmm.default,
            aten.bmm.default,
            aten.baddbmm.default,
        )
        for user in non_output_users
    )


def _has_external_internal_users(match: Match) -> bool:
    outputs = OrderedSet(node for node in match.output_nodes() if node is not None)
    matched_nodes = OrderedSet(match.nodes)
    return any(
        node not in outputs and any(user not in matched_nodes for user in node.users)
        for node in matched_nodes
    )


def _is_major_contiguous(value: torch.Tensor) -> bool:
    return value.ndim == 2 and any(
        statically_known_true(stride == 1) for stride in value.stride()
    )


def _is_definitely_too_small(dim: int | torch.SymInt, minimum: int) -> bool:
    return statically_known_true(dim < minimum)


def _is_nvidia_sm100_or_later(device: torch.device) -> bool:
    return torch.version.hip is None and torch.cuda.get_device_capability(device) >= (
        10,
        0,
    )


def _rms_norm_mm_common_check(match: Match) -> bool:
    if _has_external_internal_users(match):
        return False

    args = {
        name: _node_tensor(match, name)
        for name in ("x", "w0", "residual", "gamma", "w1")
    }
    if any(arg is None for arg in args.values()):
        return False
    values = {name: arg[1] for name, arg in args.items() if arg is not None}

    x, w0 = values["x"], values["w0"]
    residual, gamma, w1 = values["residual"], values["gamma"], values["w1"]
    if not (
        x.ndim == w0.ndim == residual.ndim == w1.ndim == 2
        and gamma.ndim == 1
        and all(value.layout is torch.strided for value in values.values())
        and all(value.device == x.device for value in values.values())
        and x.device.type == "cuda"
        and x.dtype
        == w0.dtype
        == residual.dtype
        == gamma.dtype
        == w1.dtype
        == torch.bfloat16
        and all(_is_major_contiguous(value) for value in (x, w0, residual, w1))
        and statically_known_true(gamma.stride(0) == 1)
    ):
        return False

    m = x.shape[0]
    hidden = w0.shape[1]
    output = w1.shape[1]
    if any(
        _is_definitely_too_small(dim, minimum)
        for dim, minimum in (
            (m, _MIN_M),
            (hidden, _MIN_HIDDEN),
            (output, _MIN_OUTPUT),
        )
    ):
        return False

    eps = match.kwargs.get("eps")
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps < 0:
        return False
    return not _has_competing_consumer(match.output_node())


def _rms_norm_mm_extra_check(match: Match) -> bool:
    if not (config.coda_rms_norm_rewrite or config.coda_rms_norm_fusion):
        return False
    return _rms_norm_mm_common_check(match)


def _rms_norm_mm_flex_extra_check(match: Match) -> bool:
    if not (
        config.coda_rms_norm_fusion
        and ensure_cute_available()
        and _rms_norm_mm_common_check(match)
    ):
        return False
    x = _node_tensor(match, "x")
    gamma = _node_tensor(match, "gamma")
    residual = _node_tensor(match, "residual")
    if x is None or gamma is None or residual is None:
        return False
    m = x[1].shape[0]
    hidden = gamma[1].shape[0]
    residual_shape = residual[1].shape
    return (
        _is_nvidia_sm100_or_later(x[1].device)
        and statically_known_true(x[1].shape[1] % 16 == 0)
        and statically_known_true(hidden % _RMS_PARTIAL_GROUP == 0)
        and any(
            statically_known_true(sym_eq(residual_shape, shape))
            for shape in ((m, hidden), (1, hidden), (m, 1))
        )
    )


def _rms_norm_mm_flex_training_extra_check(match: Match) -> bool:
    if not _rms_norm_mm_flex_extra_check(match):
        return False
    matched_nodes = OrderedSet(match.nodes)
    return any(
        any(user not in matched_nodes for user in output.users)
        for output in match.output_nodes()[1:]
        if output is not None
    )


@init_once_fakemode
def _coda_init_impl(input_device: torch.device) -> None:
    from .joint_graph import pass_patterns, patterns

    inference_inputs = [
        torch.empty(8, 16, device=input_device, dtype=torch.bfloat16),
        torch.empty(16, 32, device=input_device, dtype=torch.bfloat16),
        torch.empty(8, 32, device=input_device, dtype=torch.bfloat16),
        torch.empty(32, device=input_device, dtype=torch.bfloat16),
        torch.empty(32, 24, device=input_device, dtype=torch.bfloat16),
    ]
    training_inputs = [
        torch.empty_like(input, memory_format=torch.preserve_format, requires_grad=True)
        for input in inference_inputs
    ]
    for name, search_fn in (
        ("coda_rms_norm_mm", _rms_norm_mm_pattern),
        ("coda_rms_norm_mm_residual_first", _rms_norm_mm_pattern_residual_first),
    ):
        register_replacement(
            search_fn,
            _rms_norm_mm_replacement,
            training_inputs,
            joint_fwd_bwd,
            # pyrefly: ignore [bad-argument-type]
            patterns,
            extra_check=_rms_norm_mm_extra_check,
            scalar_workaround={"eps": 0.12345},
            skip_duplicates=True,
            pattern_name=f"{name}_training",
        )
        register_replacement(
            search_fn,
            _rms_norm_mm_replacement,
            inference_inputs,
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            patterns,
            extra_check=_rms_norm_mm_extra_check,
            scalar_workaround={"eps": 0.12345},
            skip_duplicates=True,
            pattern_name=f"{name}_inference",
        )

    flex_patterns = pass_patterns[1]
    register_replacement(
        _rms_norm_mm_reassociation_with_aux,
        _rms_norm_mm_flex_training_replacement,
        inference_inputs,
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        flex_patterns,
        extra_check=_rms_norm_mm_flex_training_extra_check,
        scalar_workaround={"eps": 0.12345},
        skip_duplicates=True,
        pattern_name="coda_rms_norm_flex_gemm_training",
    )
    register_replacement(
        _rms_norm_mm_reassociation,
        _rms_norm_mm_flex_replacement,
        inference_inputs,
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        flex_patterns,
        extra_check=_rms_norm_mm_flex_extra_check,
        scalar_workaround={"eps": 0.12345},
        skip_duplicates=True,
        pattern_name="coda_rms_norm_flex_gemm_inference",
    )


def _coda_init(input_device: torch.device | None = None) -> None:
    if input_device is None:
        if not torch.cuda.is_available():
            return
        input_device = torch.device("cuda")
    if input_device.type != "cuda" or torch.version.hip is not None:
        return

    global _CODA_INITIALIZED
    with _CODA_INIT_LOCK:
        if _CODA_INITIALIZED:
            return
        _coda_init_impl(input_device)
        _CODA_INITIALIZED = True
