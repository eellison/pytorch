"""Exact Triton host tensor-map arguments and their selected physical ABI."""

import ctypes
import re
from dataclasses import dataclass

import sympy
import torch
from torch._inductor.runtime.cudagraph_arg_mapping import ExpressionSource, IntegerSource, IntExpr
from torch._inductor.runtime.cudagraph_boxed_replay import _KernelModule, _PhysicalField, _TensorMapField
from torch.cuda._utils import _check_cuda_bindings


class TritonTmaDeclined(ValueError):
    pass


@dataclass(frozen=True)
class DescriptorSpec:
    dtype: str
    block_shape: tuple[int, ...]
    element_size: int
    box_dimensions: tuple[int, ...]
    data_type: int
    swizzle: int

    @property
    def abi_types(self):
        return "M" + "i" * len(self.block_shape) + "l" * len(self.block_shape)


@dataclass(frozen=True)
class DescriptorValue:
    base: object
    shape: tuple
    strides: tuple
    padding: str
    round_f32_to_tf32: bool


def descriptor_specs(module, formals):
    from triton.backends.nvidia.driver import TMA_DTYPE_DEVICE_TO_HOST

    rows = tuple(row for row in formals if row.triton_type.startswith("tensordesc<"))
    if not rows:
        return ()
    metadata = module.tensordesc_meta
    if type(metadata) not in (list, tuple) or len(metadata) != len(rows):
        raise TritonTmaDeclined("Host tensor maps require exact selected descriptor metadata")
    specs = []
    for row, meta in zip(rows, metadata):
        match = re.fullmatch(r"tensordesc<([^\[>]+)\[([0-9, ]+)\]>", row.triton_type)
        required = {"swizzle", "elem_size", "elem_type", "block_size", "fp4_padded"}
        if (match is None or type(meta) is not dict or not required.issubset(meta)
                or meta.keys() - required - {"is_im2col"}
                or meta["fp4_padded"] is not False or meta.get("is_im2col", False) is not False):
            raise TritonTmaDeclined("Only ordinary tiled, unpacked Triton tensor maps are supported")
        block_shape = tuple(int(value.strip()) for value in match[2].split(","))
        box = meta["block_size"]
        if (not 1 <= len(block_shape) <= 5 or type(box) not in (tuple, list)
                or len(box) != len(block_shape)
                or any(type(value) is not int or not 1 <= value <= 256 for value in box)
                or type(meta["elem_size"]) is not int or meta["elem_size"] not in (1, 2, 4, 8)
                or type(meta["elem_type"]) is not int or meta["elem_type"] not in TMA_DTYPE_DEVICE_TO_HOST
                or type(meta["swizzle"]) is not int or not 0 <= meta["swizzle"] <= 3):
            raise TritonTmaDeclined("Selected tensor-map metadata exceeds the tiled encoder contract")
        data_type = TMA_DTYPE_DEVICE_TO_HOST[meta["elem_type"]]
        if not 0 <= data_type <= 12:
            raise TritonTmaDeclined("Selected tensor-map dtype requires an unsupported packed format")
        specs.append((row.source_arg_index, DescriptorSpec(
            match[1], block_shape, meta["elem_size"], tuple(reversed(box)), data_type, meta["swizzle"],
        )))
    return tuple(specs)


def snapshot_descriptor(value, spec, device_index):
    from triton._utils import canonicalize_dtype
    from triton.tools.tensor_descriptor import TensorDescriptor

    if (type(value) is not TensorDescriptor or not isinstance(value.base, torch.Tensor)
            or value.base.device != torch.device("cuda", device_index)
            or canonicalize_dtype(value.base.dtype) != spec.dtype or value.base.dtype.itemsize != spec.element_size
            or type(value.block_shape) not in (list, tuple)
            or any(type(axis) is not int for axis in value.block_shape)
            or tuple(value.block_shape) != spec.block_shape
            or type(value.shape) not in (list, tuple, torch.Size)
            or type(value.strides) not in (list, tuple)
            or len(value.shape) != len(spec.block_shape) or len(value.strides) != len(spec.block_shape)
            or any(type(axis) not in (int, torch.SymInt) for axis in (*value.shape, *value.strides))
            or type(value.padding) is not str or value.padding not in ("zero", "nan")
            or value.padding == "nan" and not value.base.dtype.is_floating_point
            or type(value.round_f32_to_tf32) is not bool
            or value.round_f32_to_tf32 and spec.dtype != "fp32"):
        raise TritonTmaDeclined("TensorDescriptor differs from its selected compiler type")
    return DescriptorValue(value.base, tuple(value.shape), tuple(value.strides),
                           value.padding, value.round_f32_to_tf32)


def descriptor_guards(value, spec):
    def expression(axis):
        return sympy.Integer(axis) if type(axis) is int else axis.node.expr

    guards = []
    for axis in value.shape:
        guards.extend((sympy.Ge(expression(axis), 1), sympy.Le(expression(axis), 2**31 - 1)))
    for axis in value.strides:
        guards.extend((sympy.Ge(expression(axis), 0), sympy.Le(expression(axis), 2**63 - 1)))
    guards.append(sympy.Eq(expression(value.strides[-1]), 1))
    for axis in value.strides[:-1]:
        byte_stride = expression(axis) * spec.element_size
        guards.extend((sympy.Lt(byte_stride, 2**40), sympy.Eq(sympy.Mod(byte_stride, 16), 0)))
    return tuple(guards)


def lower_descriptor(parameter, value, spec, pointer, numeric):
    from triton.backends.nvidia.driver import TMA_DTYPE_DEVICE_TO_HOST, TMA_TF32

    data_type = TMA_DTYPE_DEVICE_TO_HOST[TMA_TF32] if value.round_f32_to_tf32 else spec.data_type
    tensor_map = _TensorMapField(
        parameter, pointer, tuple(numeric(axis) for axis in reversed(value.shape)),
        tuple(IntExpr("multiply", None, (numeric(axis), IntExpr("constant", spec.element_size)))
              for axis in reversed(value.strides[:-1])),
        spec.box_dimensions, data_type, spec.swizzle, value.padding == "nan",
    )
    fields = []
    for index, (kind, axis) in enumerate(
        (*(("i32", axis) for axis in value.shape), *(("i64", axis) for axis in value.strides)),
        start=parameter + 1,
    ):
        source = IntegerSource(axis) if type(axis) is int else ExpressionSource(numeric(axis))
        fields.append(_PhysicalField(index, 0, kind, source))
    return tuple(fields), tensor_map


class TritonTmaModule(_KernelModule):
    def __init__(self, owner):
        self.owner = owner
        self.check()

    @property
    def function(self):
        return self.owner.module.function

    @property
    def parameter_layout(self):
        return self.owner.abi_layout

    @property
    def parameter_sizes(self):
        return tuple(size for _, size in self.parameter_layout)

    @property
    def shared(self):
        return self.owner.module.shared

    @property
    def block(self):
        return (self.owner.module.num_warps * 32, 1, 1)

    def check(self):
        self.owner.check()

    def _borrow_for_cudagraph(self):
        self.check()
        return self.owner.module._borrow_for_cudagraph()

    def launch(self, images, grid, stream):
        from cuda.bindings import driver

        self.check()
        if (type(images) is not tuple or tuple(len(image) for image in images) != self.parameter_sizes
                or any(type(image) is not bytes for image in images)
                or type(grid) is not tuple or len(grid) != 3
                or any(type(axis) is not int or axis <= 0 for axis in grid)
                or type(stream) is not int or not 0 <= stream < 2**64
                or self.owner.module._graph_borrows <= 0):
            raise TritonTmaDeclined("Host tensor-map capture requires exact arguments and a held graph borrow")
        if _check_cuda_bindings(driver.cuStreamIsCapturing(stream)) != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            raise TritonTmaDeclined("Host tensor-map launch is only supported during graph capture")
        storage = tuple(ctypes.create_string_buffer(image, len(image)) for image in images)
        arguments = (ctypes.c_void_p * len(storage))(*(ctypes.addressof(image) for image in storage))
        _check_cuda_bindings(driver.cuLaunchKernel(
            self.function, *grid, *self.block, self.shared, stream, ctypes.addressof(arguments), 0,
        ))
