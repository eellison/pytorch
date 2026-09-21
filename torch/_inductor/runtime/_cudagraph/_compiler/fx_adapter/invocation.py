"""Diagnostic FX markers; semantic host events are recorded at their call sites."""

from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.node import has_side_effect
from torch.utils import _pytree as pytree

from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import FXTraceDeclined


def _record(target, args):
    mode = get_proxy_mode()
    if mode is None:
        raise FXTraceDeclined("Host markers require an active FX proxy trace")
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)
    mode.tracer.create_proxy("call_function", target, proxy_args, {})


@has_side_effect
def invoke(kernel_name, arguments):
    _record(invoke, (kernel_name, arguments))


@has_side_effect
def assert_layout(tensor, size, stride, label=None):
    _record(assert_layout, (tensor, size, stride, label))


@has_side_effect
def normalize(tensor):
    _record(normalize, (tensor,))
