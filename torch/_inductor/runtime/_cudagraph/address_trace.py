"""Track host address reads with the same storage roots used by replay lowering."""

from dataclasses import dataclass

import sympy
import torch
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import AllocateEvent, FXTraceDeclined, NormalizeEvent, ReinterpretEvent
from torch._dynamo.source import LocalSource
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, InputSource
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot, track_tensor_tree
from torch.fx.experimental.symbolic_shapes import DimDynamic
from torch.overrides import TorchFunctionMode


UINTPTR_MAX = (1 << 64) - 1


@dataclass(frozen=True, eq=False)
class AddressResolution:
    root: InputSource | BufferSource
    root_tensor: FakeTensor
    byte_offset: int | torch.SymInt
    alignment: int
    generation: int = 0


class TensorAddressRoots:
    """Consume the trace's existing root identities and recorded view offsets."""

    def __init__(self, mode):
        self.mode = mode
        self.values = {}
        self.inputs = {}
        self.events = {}
        self.allocations = []

    def _check(self, tensor):
        if type(tensor) is not FakeTensor or tensor.fake_mode is not self.mode:
            raise FXTraceDeclined("Address source is outside the local FakeTensorMode")

    def bind_input(self, tensor, index):
        self._check(tensor)
        if id(tensor) in self.values:
            raise FXTraceDeclined("Address root was already recorded")
        resolution = AddressResolution(InputSource(index), tensor, 0, tensor.dtype.itemsize)
        self.values[id(tensor)] = (tensor, resolution)
        self.inputs[index] = resolution

    def record(self, event):
        if type(event) is AllocateEvent:
            self._check(event.tensor)
            root = BufferSource(f"trace_{len(self.allocations)}")
            resolution = AddressResolution(root, event.tensor, 0, 256)
            self.allocations.append(resolution)
            self.values[id(event.tensor)] = (event.tensor, resolution)
        elif type(event) is ReinterpretEvent:
            source = self(event.source)
            self._check(event.tensor)
            offset = event.offset
            if not (type(offset) is int or type(offset) is torch.SymInt
                    and offset.node.shape_env is self.mode.shape_env):
                raise FXTraceDeclined("Address view offset has no local integer source")
            if event.tensor.dtype != event.source.dtype:
                raise FXTraceDeclined("Address view changed its recorded element representation")
            resolution = AddressResolution(source.root, source.root_tensor,
                                           source.byte_offset + offset * event.source.element_size(),
                                           source.alignment, source.generation)
            self.values[id(event.tensor)] = (event.tensor, resolution)
        elif type(event) is NormalizeEvent:
            source = self(event.tensor)
            resolution = AddressResolution(source.root, source.root_tensor, source.byte_offset,
                                           16, source.generation + 1)
            for key, (tensor, old) in tuple(self.values.items()):
                if old.root == source.root:
                    self.values[key] = (tensor, AddressResolution(old.root, old.root_tensor, old.byte_offset,
                                                                 16, resolution.generation))
        else:
            return
        self.events[id(event)] = (event, resolution)

    def event(self, event):
        item = self.events.get(id(event))
        if item is None or item[0] is not event:
            raise FXTraceDeclined("Host event lost its recorded storage resolution")
        return item[1]

    def __call__(self, tensor):
        self._check(tensor)
        item = self.values.get(id(tensor))
        if item is None or item[0] is not tensor:
            raise FXTraceDeclined("Address tensor has no recorded storage root")
        return item[1]


@dataclass(frozen=True, eq=False)
class AddressBinding:
    root: InputSource | BufferSource
    generation: int
    root_tensor: FakeTensor
    alignment: int
    value: torch.SymInt
    source: LocalSource
    node: torch.fx.Node
    symbol: sympy.Symbol


class SymbolicTensorAddresses(TorchFunctionMode):
    def __init__(self, mode, resolve, input_hints):
        super().__init__()
        self.mode = mode
        self.resolve = resolve
        self.input_hints = dict(input_hints)
        self.bindings = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is not torch.Tensor.data_ptr and func is not torch.Tensor.const_data_ptr:
            return func(*args, **({} if kwargs is None else kwargs))
        if len(args) != 1 or kwargs:
            raise FXTraceDeclined("Expected the exact Tensor address accessor")
        tensor = args[0]
        resolution = self.resolve(tensor)
        if tensor.numel() == 0:
            return 0
        if resolution.root_tensor.numel() == 0:
            raise FXTraceDeclined("A nonempty view cannot use an empty input's null data pointer as its root")
        return self.bind(resolution).value + resolution.byte_offset

    def bind(self, resolution):
        key = (resolution.root, resolution.generation)
        binding = self.bindings.get(key)
        if binding is None:
            proxy_mode = get_proxy_mode()
            if proxy_mode is None:
                raise FXTraceDeclined("Symbolic address reads require the active FX tracer")
            source = LocalSource(f"address_{len(self.bindings)}")
            environment = self.mode.shape_env
            if type(resolution.root) is InputSource and resolution.generation == 0:
                hint = self.input_hints.get(resolution.root)
                if type(hint) is not int or not 0 <= hint <= UINTPTR_MAX:
                    raise FXTraceDeclined("Input address requires its exact unsigned ordinary hint")
                symbol = environment.create_symbol(
                    hint, source, dynamic_dim=DimDynamic.DYNAMIC, positive=None,
                    do_not_specialize_zero_one=True,
                )
                value = environment.create_symintnode(symbol, hint=hint, source=source)
            else:
                value = environment.create_unbacked_symint(source)
                symbol = value.node.expr
            environment.constrain_symbol_range(symbol, 0, UINTPTR_MAX)
            tracer = proxy_mode.tracer
            tensor_proxy = get_proxy_slot(resolution.root_tensor, tracer).proxy
            proxy = tracer.create_proxy("call_method", "data_ptr", (tensor_proxy,), {})
            track_tensor_tree(value, proxy, constant=None, tracer=tracer)
            binding = AddressBinding(resolution.root, resolution.generation, resolution.root_tensor,
                                     resolution.alignment, value, source, proxy.node, symbol)
            self.bindings[key] = binding
        elif (binding.root_tensor is not resolution.root_tensor
              or binding.alignment != resolution.alignment):
            raise FXTraceDeclined("Address root changed its trace identity or alignment fact")
        return binding
