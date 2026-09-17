from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml

import torchgen.api.ufunc as ufunc
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    CType,
    Expr,
    NamedCType,
    opmath_t,
    scalar_t,
    StructuredImplSignature,
    VectorizedCType,
)
from torchgen.context import with_native_function
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    ScalarType,
    UfuncInnerLoop,
    UfuncKey,
)
from torchgen.utils import OrderedSet
from torchgen.yaml_utils import YamlLoader


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torchgen.api.ufunc import UfunctorBindings
    from torchgen.utils import FileManager


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                                  CUDA STUFF
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# NB: not bothering to generate dispatch stub forward declaration in header,
# we can just paste it wherever necessary

# TODO: use BackendIndex
# dispatch_key: DispatchKey  # only CPU/CUDA right now


# Represents functors for implementing CUDA ufuncs.
# Functors are templated by scalar_t because when USERS instantiate functors
# they are templated.  A functor looks something like this:
#
#   template <typename scalar_t>
#   struct CUDAFunctorOnSelf_add {
#     using opmath_t = at::opmath_type<scalar_t>;
#     opmath_t other_;
#     opmath_t alpha_;
#     CUDAFunctorOnSelf_add(opmath_t other, opmath_t alpha)
#         : other_(other), alpha_(alpha) {}
#     __device__ scalar_t operator()(scalar_t self) {
#       return ufunc::add(static_cast<opmath_t>(self), other_, alpha_);
#     }
#   };
#
@dataclass(frozen=True)
class UfunctorSignature:
    g: NativeFunctionsGroup
    scalar_tensor_idx: int | None
    name: str

    def arguments(self) -> UfunctorBindings:
        return ufunc.ufunctor_arguments(
            self.g, scalar_tensor_idx=self.scalar_tensor_idx, scalar_t=scalar_t
        )

    def fields(self) -> list[Binding]:
        # fields are renamed to have a trailing underscore, as is conventional
        return [b.rename(f"{b.name}_") for b in self.arguments().ctor]

    def returns_type(self) -> CType:
        # TODO: don't hardcode; return type will be inferred based on tags on
        # the native function
        return BaseCType(scalar_t)

    def decl_fields(self) -> str:
        return "\n".join(f"{f.type} {f.name};" for f in self.fields())

    def inline_defn_ctor(self) -> str:
        args_str = ", ".join(a.decl() for a in self.arguments().ctor)
        # NB: hypothetically could do this with translate but the
        # transition here is very regular
        init_str = ", ".join(f"{a.name}_({a.name})" for a in self.arguments().ctor)
        return f"{self.name}({args_str}) : {init_str} {{}}"

    def decl_apply(self) -> str:
        args_str = ", ".join(a.decl() for a in self.arguments().apply)
        return f"{self.returns_type().cpp_type()} operator()({args_str}) const"


@dataclass(frozen=True)
class UfuncSignature:
    g: NativeFunctionsGroup
    name: str
    compute_t: CType

    def arguments(self) -> list[Binding]:
        return ufunc.ufunc_arguments(self.g, compute_t=self.compute_t)

    def call(self, ctx: Sequence[Binding | Expr]) -> str:
        return f"{self.name}({', '.join(a.expr for a in translate(ctx, self.arguments()))})"


# steps:
#   1. take the functional signature
#   2. use api.ufunc to convert it to template signature.  this establishes
#      the type of the template function
#   3. use api.ufunc (II) to generate a split struct / operator() signature.
#      this establish context in which we call the template signature
#
# StructuredImplSignature context
#   ~> functor constructor sig
#
# Functor constructor context
#   ~> functor fields sig
#
# Functor apply context (functor fields + functor apply sig)
#   ~> template sig
#


def eligible_for_binary_scalar_specialization(g: NativeFunctionsGroup) -> bool:
    num_tensors = sum(
        1 for a in g.functional.func.arguments.flat_non_out if a.type.is_tensor_like()
    )
    return num_tensors == 2


def compute_ufunc_cuda_functors(
    g: NativeFunctionsGroup,
) -> tuple[dict[ScalarType, dict[UfuncKey, UfunctorSignature]], str]:
    # First, build the functors.
    ufunctor_sigs: dict[ScalarType, dict[UfuncKey, UfunctorSignature]] = {}
    ufunctors: list[str] = []
    loops = g.out.ufunc_inner_loop
    scalar_tensor_idx_lookup = {
        UfuncKey.CUDAFunctorOnSelf: 1,
        UfuncKey.CUDAFunctorOnOther: 0,
        UfuncKey.CUDAFunctor: None,
    }
    if eligible_for_binary_scalar_specialization(g):
        keys = [
            UfuncKey.CUDAFunctorOnSelf,
            UfuncKey.CUDAFunctorOnOther,
            UfuncKey.CUDAFunctor,
        ]
    else:
        keys = [UfuncKey.CUDAFunctor]
        for k in [UfuncKey.CUDAFunctorOnSelf, UfuncKey.CUDAFunctorOnOther]:
            if k in loops:
                raise AssertionError(f"cannot use {k} on non-binary function")
    for k in keys:
        # If the key was directly defined, skip functor codegen; we assume the
        # user already done it for us
        if k in loops:
            ufunctor_sig = UfunctorSignature(
                g, scalar_tensor_idx=scalar_tensor_idx_lookup[k], name=loops[k].name
            )
            for dtype in loops[k].supported_dtypes:
                ufunctor_sigs.setdefault(dtype, {})[k] = ufunctor_sig
            continue

        # Note [ScalarOnly and Generic must match names for CUDA]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Otherwise, look in ANY of the generic entries.  For simplicity of
        # codegen, both ScalarOnly and Generic are defined, the ufunc name
        # must match  (if they didn't match, we'd have to generate distinct
        # functors per dtype, which is awful, so we're not going to do it unless
        # someone really forces us to)
        ufunc_name = None
        supported_dtypes: OrderedSet[ScalarType] = OrderedSet()
        for lk in [UfuncKey.ScalarOnly, UfuncKey.Generic]:
            if lk not in loops:
                continue
            if ufunc_name is None:
                ufunc_name = loops[lk].name
            else:
                # See Note [ScalarOnly and Generic must match names for CUDA]
                if ufunc_name != loops[lk].name:
                    raise AssertionError(
                        "ScalarOnly and Generic must have same ufunc name"
                    )
            supported_dtypes |= loops[lk].supported_dtypes
        if ufunc_name is None:
            raise AssertionError("ufunc_name must be non-None")

        name = f"{k}_{ufunc_name}"
        ufunctor_sig = UfunctorSignature(
            g, scalar_tensor_idx=scalar_tensor_idx_lookup[k], name=name
        )
        for dtype in supported_dtypes:
            ufunctor_sigs.setdefault(dtype, {})[k] = ufunctor_sig

        ufunc_sig = UfuncSignature(
            g, name=f"ufunc::{ufunc_name}", compute_t=BaseCType(opmath_t)
        )
        apply_ctx = ufunctor_sig.fields() + ufunctor_sig.arguments().apply
        ufunctors.append(
            f"""
template <typename scalar_t>
struct {ufunctor_sig.name} {{
  using opmath_t = at::opmath_type<scalar_t>;
  {ufunctor_sig.decl_fields()}
  {ufunctor_sig.inline_defn_ctor()}
  __device__ {ufunctor_sig.decl_apply()} {{
    return {ufunc_sig.call(apply_ctx)};
  }}
}};
"""
        )

    return ufunctor_sigs, "\n".join(ufunctors)


@dataclass(frozen=True)
class BinaryScalarSpecializationConfig:
    scalar_idx: int
    ctor_tensor: str
    ufunc_key: UfuncKey


BinaryScalarSpecializationConfigs = [
    BinaryScalarSpecializationConfig(
        scalar_idx=0,
        ctor_tensor="self",
        ufunc_key=UfuncKey.CUDAFunctorOnOther,
    ),
    BinaryScalarSpecializationConfig(
        scalar_idx=1,
        ctor_tensor="other",
        ufunc_key=UfuncKey.CUDAFunctorOnSelf,
    ),
]


def compute_ufunc_cuda_dtype_body(
    g: NativeFunctionsGroup,
    dtype: ScalarType,
    inner_loops: dict[UfuncKey, UfunctorSignature],
    parent_ctx: Sequence[Binding],
) -> str:
    body = "using opmath_t = at::opmath_type<scalar_t>;"
    body += "if (false) {}\n"  # for ease of codegen
    for config in BinaryScalarSpecializationConfigs:
        if config.ufunc_key not in inner_loops:
            continue
        ufunctor_sig = inner_loops[config.ufunc_key]
        scalar_idx = config.scalar_idx + 1
        # Make a copy and at the same time widen the type (not permissible
        # without copy; we don't want to mutate the input argument anyway)
        ctx: list[Expr | Binding] = list(parent_ctx)
        ctx.append(
            Expr(
                expr=f"iter.scalar_value<opmath_t>({scalar_idx})",
                type=NamedCType(config.ctor_tensor, BaseCType(opmath_t)),
            )
        )
        ufunctor_ctor_exprs_str = ", ".join(
            a.expr for a in translate(ctx, ufunctor_sig.arguments().ctor)
        )

        # NB: ufunctor must be allocated before iter.remove_operand is called,
        # as it relies on iter
        body += f"""\
else if (iter.is_cpu_scalar({scalar_idx})) {{
  {ufunctor_sig.name}<scalar_t> ufunctor({ufunctor_ctor_exprs_str});
  iter.remove_operand({scalar_idx});
  gpu_kernel(iter, ufunctor);
}}"""

    ufunctor_sig = inner_loops[UfuncKey.CUDAFunctor]
    ufunctor_ctor_exprs_str = ", ".join(
        a.expr for a in translate(parent_ctx, ufunctor_sig.arguments().ctor)
    )
    body += f"""
else {{
  gpu_kernel(iter, {ufunctor_sig.name}<scalar_t>({ufunctor_ctor_exprs_str}));
}}
    """
    return body


@with_native_function
def compute_ufunc_cuda(g: NativeFunctionsGroup) -> str:
    # First, build the functors, indexing them by dtype
    ufunctor_sigs, ufunctors = compute_ufunc_cuda_functors(g)

    # Next, build the conditionals
    sig = StructuredImplSignature(g, ufunc.kernel_name(g, DispatchKey.CUDA))
    dtype_cases = []
    for dtype, inner_ufunc_sigs in ufunctor_sigs.items():
        dtype_cases.append(
            f"""
AT_DISPATCH_CASE(at::ScalarType::{dtype},
  [&]() {{
    {compute_ufunc_cuda_dtype_body(g, dtype, inner_ufunc_sigs, sig.arguments())}
  }}
)
"""
        )

    dtype_cases_str = "\n".join(dtype_cases)

    stub_sig = StubSignature(g)

    return f"""
{ufunctors}

{stub_sig.type_defn()};
{stub_sig.dispatch_decl()}

{stub_sig.kernel_defn()} {{
  AT_DISPATCH_SWITCH(iter.common_dtype(), "{sig.name}",
    {dtype_cases_str}
  );
}}
REGISTER_DISPATCH({stub_sig.name}, &{stub_sig.kernel_name})

{sig.defn()} {{
  {stub_sig.direct_call(sig.arguments())};
}}
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                                   CPU STUFF
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@dataclass(frozen=True)
class StubSignature:
    g: NativeFunctionsGroup

    @property
    def name(self) -> str:
        return f"{str(self.g.functional.func.name.name)}_stub"

    @property
    def kernel_name(self) -> str:
        return f"{str(self.g.functional.func.name.name)}_kernel"

    @property
    def type_name(self) -> str:
        return f"{str(self.g.functional.func.name.name)}_fn"

    def arguments(self) -> list[Binding]:
        return ufunc.stub_arguments(self.g)

    def type(self) -> str:
        cpp_args = self.arguments()
        return f"void(*)(TensorIteratorBase&, {', '.join(a.type for a in cpp_args)})"

    def dispatch_decl(self) -> str:
        return f"DECLARE_DISPATCH({self.type_name}, {self.name})"

    def dispatch_defn(self) -> str:
        return f"DEFINE_DISPATCH({self.name})"

    def kernel_defn(self) -> str:
        return f"void {self.kernel_name}(TensorIteratorBase& iter, {', '.join(a.defn() for a in self.arguments())})"

    def type_defn(self) -> str:
        return f"using {self.type_name} = {self.type()}"

    # must be called from context where this is TensorIteratorBase*
    def call(self, ctx: Sequence[Binding]) -> str:
        return f"{self.name}(device_type(), *this, {', '.join(a.expr for a in translate(ctx, self.arguments()))})"

    # used in CUDA to skip the unnecessary dynamic dispatch
    def direct_call(self, ctx: Sequence[Binding]) -> str:
        return f"{self.kernel_name}(*this, {', '.join(a.expr for a in translate(ctx, self.arguments()))})"


@with_native_function
def compute_ufunc_cpu(g: NativeFunctionsGroup) -> str:
    stub_sig = StubSignature(g)
    sig = StructuredImplSignature(g, ufunc.kernel_name(g, DispatchKey.CPU))

    return f"""
{stub_sig.type_defn()};
{stub_sig.dispatch_decl()}
{stub_sig.dispatch_defn()};

{sig.defn()} {{
  {stub_sig.call(sig.arguments())};
}}
"""


def compute_ufunc_cpu_dtype_body(
    g: NativeFunctionsGroup,
    dtype: ScalarType,
    inner_loops: dict[UfuncKey, UfuncSignature],
    parent_ctx: Sequence[Binding],
) -> str:
    if UfuncKey.CPUScalar not in inner_loops:
        raise AssertionError(f"{dtype}, {inner_loops.keys()}")
    if not inner_loops.keys() <= {UfuncKey.CPUScalar, UfuncKey.CPUVector}:
        raise AssertionError(
            f"inner_loops keys must be subset of CPUScalar/CPUVector, got {inner_loops.keys()}"
        )
    scalar_loop = inner_loops[UfuncKey.CPUScalar]
    vec_loop = None
    if UfuncKey.CPUVector in inner_loops:
        vec_loop = inner_loops[UfuncKey.CPUVector]

    # NB: We DON'T use translate here, because translate is
    # incapable of CSE'ing the scalar accesses in case it is also
    # used by Vectorized; also, the unpacking here is very simple
    # and only affects Scalar; everything else is implicitly captured
    # by the lambda

    # Setup scalar in scope
    body = []
    ctx = []
    for b in parent_ctx:
        if isinstance(b.argument, Argument) and b.argument.type != BaseType(
            BaseTy.Scalar
        ):
            continue
        body.append(f"auto _s_{b.name} = {b.name}.to<scalar_t>();")
        ctx.append(Expr(f"_s_{b.name}", NamedCType(b.nctype.name, BaseCType(scalar_t))))
    if vec_loop is not None:
        for b in parent_ctx:
            if isinstance(b.argument, Argument) and b.argument.type != BaseType(
                BaseTy.Scalar
            ):
                continue
            body.append(
                f"auto _v_{b.name} = at::vec::Vectorized<scalar_t>(_s_{b.name});"
            )
            ctx.append(
                Expr(
                    f"_v_{b.name}",
                    NamedCType(b.nctype.name, VectorizedCType(BaseCType(scalar_t))),
                )
            )

    # Setup lambda signature
    # NB: simplified version of ufunctor_arguments
    scalar_bindings = []
    vec_bindings = []
    for a in g.functional.func.arguments.flat_non_out:
        if not a.type.is_tensor_like():
            continue
        if a.type != BaseType(BaseTy.Tensor):
            raise AssertionError(f"Expected Tensor type, got {a.type}")
        scalar_bindings.append(
            Binding(
                name=a.name,
                nctype=NamedCType(a.name, BaseCType(scalar_t)),
                argument=a,
            )
        )
        if vec_loop is not None:
            vec_bindings.append(
                Binding(
                    name=a.name,
                    nctype=NamedCType(a.name, VectorizedCType(BaseCType(scalar_t))),
                    argument=a,
                )
            )

    def with_ctx(b: Sequence[Binding]) -> list[Expr | Binding]:
        r: list[Expr | Binding] = []
        r.extend(ctx)
        r.extend(b)
        return r

    body_str = "\n".join(body)
    if vec_loop is not None:
        return f"""
{body_str}
cpu_kernel_vec(iter,
  [=]({", ".join(b.decl() for b in scalar_bindings)}) {{ return {scalar_loop.call(with_ctx(scalar_bindings))}; }},
  [=]({", ".join(b.decl() for b in vec_bindings)}) {{ return {vec_loop.call(with_ctx(vec_bindings))}; }}
);
"""
    else:
        return f"""
{body_str}
cpu_kernel(iter,
  [=]({", ".join(b.decl() for b in scalar_bindings)}) {{ return {scalar_loop.call(with_ctx(scalar_bindings))}; }}
);
"""


@with_native_function
def compute_ufunc_cpu_kernel(g: NativeFunctionsGroup) -> str:
    stub_sig = StubSignature(g)

    # Reindex the ufunc by dtypes; processing generic/scalaronly as well
    loops = g.out.ufunc_inner_loop
    ufunc_sigs: dict[ScalarType, dict[UfuncKey, UfuncSignature]] = {}
    for k in [UfuncKey.CPUScalar, UfuncKey.CPUVector]:
        lks = []
        # ORDER MATTERS: this specifies overriding precedence
        if k in loops:  # should happen rarely
            lks.append(k)
        if UfuncKey.ScalarOnly in loops and k is UfuncKey.CPUScalar:
            lks.append(UfuncKey.ScalarOnly)
        if UfuncKey.Generic in loops:
            lks.append(UfuncKey.Generic)
        # TODO: don't hardcode ufunc:: namespace here, should be centralized smh
        for lk in lks:
            for dtype in loops[lk].supported_dtypes:
                compute_t: CType
                if k is UfuncKey.CPUScalar:
                    compute_t = BaseCType(scalar_t)
                elif k is UfuncKey.CPUVector:
                    compute_t = VectorizedCType(BaseCType(scalar_t))
                else:
                    raise AssertionError
                inner_ufunc_sigs = ufunc_sigs.setdefault(dtype, {})
                if k not in inner_ufunc_sigs:
                    inner_ufunc_sigs[k] = UfuncSignature(
                        g, name=f"ufunc::{loops[lk].name}", compute_t=compute_t
                    )

    # Build the conditionals
    dtype_cases = []
    for dtype, inner_ufunc_sigs in ufunc_sigs.items():
        dtype_cases.append(
            f"""
AT_DISPATCH_CASE(at::ScalarType::{dtype},
  [&]() {{
    {compute_ufunc_cpu_dtype_body(g, dtype, inner_ufunc_sigs, stub_sig.arguments())}
  }}
)
"""
        )

    dtype_cases_str = "\n".join(dtype_cases)
    return f"""
namespace {{

{stub_sig.kernel_defn()} {{
  AT_DISPATCH_SWITCH(iter.common_dtype(), "{stub_sig.name}",
    {dtype_cases_str}
  );
}}

}} // anonymous namespace

{stub_sig.type_defn()};
{stub_sig.dispatch_decl()}
REGISTER_DISPATCH({stub_sig.name}, &{stub_sig.kernel_name})
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                       HOST-TRACING SIBLINGS (CUDA)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# The traced sibling of an elementwise op (aten/src/ATen/cuda/host_trace/ti):
# eager's own functor launched on the sibling iterator, with the functor's
# constant members (a CPU scalar operand, a Scalar argument) as proxy fields
# of the tape.  The spec is an op's ufunc_inner_loop (add) or an entry of
# cuda/host_trace/ti/siblings.yaml for an op whose eager kernel host is
# hand-written.  Per op the generator writes two headers for the eager
# translation unit that hosts the kernel (E36: one instantiation, one function
# object for eager and the replay):
#   HostTraceFunctor_<name>.cuh  the functor eager's host launches in place
#                                of its lambda, in namespace at::native, its
#                                body the ufunc header's (the only copy of the
#                                device body); not written for a `functor`
#                                spec, whose type eager's TU defines itself,
#                                nor for a ufunc op, whose UfuncCUDA_<name>.cu
#                                defines its functors
#   HostTraceSibling_<name>.cuh  the entry, its proxies and strided views;
#                                eager's .cu includes it after its namespace
#                                closes (UfuncCUDA.cu's template does the same)
# HostTraceSiblingOps.h declares the entries, HostTraceSiblingBindings.h binds
# them with a table for torch/cuda/_host_trace_ti.py's registry, and
# tools/pyi/gen_pyi.py writes their stubs from the same specs.
#
# Two launch shapes.  "ufunc": the functors compute_ufunc_cuda builds (a CPU
# scalar operand and the Scalar arguments as opmath fields, the
# CUDAFunctorOnSelf / CUDAFunctorOnOther specializations); the shape of a
# ufunc op.  "eager" (the default): one functor over scalar_t with the op's
# non-tensor arguments as fields, launched as the eager host launches
# (gpu_kernel, gpu_kernel_with_scalars or the symmetric form), a CPU scalar
# operand held in scalar_t as eager holds it.  Bitwise under E21 includes NaN
# payloads, which the opmath round trip of the ufunc shape canonicalizes for
# pass-through ops (max / min return an operand's bits).

# the dtypes the sibling iterator and the field types cover: no complex, no
# float8; bool only for a functor without fields; the rest decline by name
SIBLING_DTYPES: OrderedSet[ScalarType] = OrderedSet(
    [
        ScalarType.Byte,
        ScalarType.Char,
        ScalarType.Short,
        ScalarType.Int,
        ScalarType.Long,
        ScalarType.Half,
        ScalarType.Float,
        ScalarType.Double,
        ScalarType.BFloat16,
    ]
)

# a field's conversion from the argument: "opmath" is eager's
# `.to<opmath_t>()`, "scalar" eager's `.to<scalar_t>()` (the value rounded to
# the tensor dtype, then widened: the functor carries it in opmath_t, which
# compares and converts back bit for bit), "acc" eager's `.to<acc_type>()`
FIELD_CONVERSIONS = ("opmath", "scalar", "acc")
# the tensor dtypes an argument may be fixed to beside the scalar_t operands
ARG_TYPES = {"bool": ScalarType.Bool}
SPEC_KEYS = {
    "func",
    "ufunc_inner_loop",
    "header",
    "shape",
    "symmetric",
    "scalars",
    "fields",
    "arg_types",
    "variants",
    "host_args",
    "inputs",
    "output",
    "entry",
    "constants",
    "functor",
}


@dataclass(frozen=True)
class SiblingSpec:
    # the functional overload as native_functions.yaml names it ("sigmoid",
    # "lerp.Scalar")
    name: str
    loops: dict[UfuncKey, UfuncInnerLoop]
    # the ufunc header with the device body (None for a `functor` spec)
    header: str | None
    shape: str = "eager"
    # eager shape, binary: the symmetric AUnaryFunctor form for a CPU scalar on
    # either side (opmath_symmetric_gpu_kernel_with_scalars), or
    # gpu_kernel_with_scalars; neither: plain gpu_kernel and a CPU scalar
    # operand declines (eager's plain gpu_kernel asserts on one)
    symmetric: bool = False
    scalars: bool = False
    # non-tensor argument -> conversion (FIELD_CONVERSIONS); unlisted Scalar,
    # float and int arguments convert as "opmath"
    fields: dict[str, str] = field(default_factory=dict)
    # tensor argument -> a fixed dtype (a bool mask beside scalar_t operands)
    arg_types: dict[str, str] = field(default_factory=dict)
    # member -> literal: a value eager's lambda captures that is not an
    # argument of the op (threshold_backward's value 0), a field after the
    # arguments, converted as `fields` says
    constants: dict[str, str] = field(default_factory=dict)
    # (str argument, its values): one device body per value, selected at the
    # entry (gelu_backward's approximate)
    variants: tuple[str, tuple[str, ...]] | None = None
    # arguments the registry's hand entry consumes before the launch (a
    # reduction mode); they are not part of the generated entry
    host_args: tuple[str, ...] = ()
    # the iterator's input order where the eager meta differs from the schema
    inputs: tuple[str, ...] | None = None
    # "iterator": the sibling iterator allocates the functional output;
    # "suggest_memory_format": eager allocates empty_like(first input,
    # suggest_memory_format()) and hands it to the iterator
    output: str = "iterator"
    # the registry does not register the generic entry: a hand entry in
    # torch/cuda/_host_trace_ti.py wraps the binding (the op's own checks)
    hand_entry: bool = False
    # a functor type eager's translation unit defines itself (AbsKernel.cu's
    # AbsFunctor), launched as it is: no functor header, no restated body
    functor: str | None = None

    @property
    def file_name(self) -> str:
        return self.name.replace(".", "_")

    @property
    def base(self) -> str:
        return self.name.split(".")[0]


def parse_host_trace_siblings(path: str) -> dict[str, SiblingSpec]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        entries = yaml.load(f, Loader=YamlLoader) or []
    specs: dict[str, SiblingSpec] = {}
    for e in entries:
        name = e["func"]
        unknown = set(e) - SPEC_KEYS
        if unknown:
            raise AssertionError(f"sibling {name}: unknown keys {sorted(unknown)}")
        loops = {}
        for k, v in e["ufunc_inner_loop"].items():
            key = UfuncKey.parse(k)
            loops[key] = UfuncInnerLoop.parse(v, key)
        shape = e.get("shape", "eager")
        if shape not in ("ufunc", "eager"):
            raise AssertionError(f"sibling {name}: unknown shape {shape}")
        fields = dict(e.get("fields", {}))
        for a, conv in fields.items():
            if conv not in FIELD_CONVERSIONS:
                raise AssertionError(f"sibling {name}: field {a} converts as {conv}")
        arg_types = dict(e.get("arg_types", {}))
        for a, t in arg_types.items():
            if t not in ARG_TYPES:
                raise AssertionError(f"sibling {name}: argument {a} typed {t}")
        variants = None
        if "variants" in e:
            ((selector, values),) = e["variants"].items()
            variants = (selector, tuple(values))
        output = e.get("output", "iterator")
        if output not in ("iterator", "suggest_memory_format"):
            raise AssertionError(f"sibling {name}: unknown output {output}")
        entry = e.get("entry", "generic")
        if entry not in ("generic", "hand"):
            raise AssertionError(f"sibling {name}: unknown entry {entry}")
        host_args = tuple(e.get("host_args", ()))
        functor = e.get("functor")
        if functor is not None and "header" in e:
            raise AssertionError(f"sibling {name}: a functor spec has no ufunc header")
        header = (
            None
            if functor is not None
            else e.get("header", f"ATen/native/ufunc/{name.split('.')[0]}.h")
        )
        specs[name] = SiblingSpec(
            name=name,
            loops=loops,
            header=header,
            shape=shape,
            symmetric=bool(e.get("symmetric", False)),
            scalars=bool(e.get("scalars", False)),
            fields=fields,
            arg_types=arg_types,
            variants=variants,
            host_args=host_args,
            inputs=tuple(e["inputs"]) if "inputs" in e else None,
            constants={k: str(v) for k, v in e.get("constants", {}).items()},
            output=output,
            hand_entry=entry == "hand" or bool(host_args),
            functor=functor,
        )
    return specs


@dataclass(frozen=True)
class SiblingOp:
    spec: SiblingSpec
    functional: NativeFunction
    # the overloads the registry serves through the same entry: the in-place
    # form (self is the destination), the out= form, and the .Scalar twins
    # (the same kernel over a wrapped number)
    inplace: NativeFunction | None
    out: NativeFunction | None
    scalar_twins: tuple[NativeFunction, ...]
    # the ufunc shape reuses compute_ufunc_cuda_functors, which needs the group
    group: NativeFunctionsGroup | None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def arguments(self) -> list[Argument]:
        # the entry's arguments: the schema's, minus what the hand entry consumes
        return [
            a
            for a in self.functional.func.arguments.flat_non_out
            if a.name not in self.spec.host_args
        ]

    @property
    def tensors(self) -> list[Argument]:
        return [a for a in self.arguments if a.type == BaseType(BaseTy.Tensor)]

    @property
    def values(self) -> list[Argument]:
        # the non-tensor arguments that become functor fields, in schema order
        selector = self.spec.variants[0] if self.spec.variants else None
        return [
            a
            for a in self.arguments
            if a.type != BaseType(BaseTy.Tensor) and a.name != selector
        ]

    @property
    def inputs(self) -> list[Argument]:
        # the iterator's inputs, the schema's tensor order unless the spec reorders
        if self.spec.inputs is None:
            return self.tensors
        by_name = {a.name: a for a in self.tensors}
        if sorted(self.spec.inputs) != sorted(by_name):
            raise AssertionError(
                f"sibling {self.name}: inputs {self.spec.inputs} are not the tensor arguments {sorted(by_name)}"
            )
        return [by_name[n] for n in self.spec.inputs]

    def overloads(self) -> list[NativeFunction]:
        return [
            f for f in (self.inplace, self.out, *self.scalar_twins) if f is not None
        ]


def sibling_argument_cpp(a: Argument) -> str:
    t = a.type
    if t == BaseType(BaseTy.Tensor):
        return "const at::Tensor&"
    if t == BaseType(BaseTy.Scalar):
        return "const at::Scalar&"
    if t == BaseType(BaseTy.str):
        return "std::string_view"
    r = cpp.valuetype_type(t, binds=a.name, symint=False)
    if r is None or t not in (
        BaseType(BaseTy.float),
        BaseType(BaseTy.int),
        BaseType(BaseTy.bool),
    ):
        raise AssertionError(f"sibling argument {a.name} of type {t} is not supported")
    return r.cpp_type()


def scalar_twins_of(
    functional: NativeFunction, by_name: dict[str, NativeFunction]
) -> tuple[NativeFunction, ...]:
    # <base>.Scalar and <base>_.Scalar: the functional's signature with the
    # second tensor argument a Scalar (BinaryOps.cpp's wrapped-number forms)
    args = functional.func.arguments.flat_non_out
    tensors = [i for i, a in enumerate(args) if a.type == BaseType(BaseTy.Tensor)]
    if len(tensors) != 2:
        return ()
    want = [(a.name, str(a.type)) for a in args]
    want[tensors[1]] = (args[tensors[1]].name, "Scalar")
    base = functional.func.name.name.base
    twins = []
    for candidate in (f"{base}.Scalar", f"{base}_.Scalar"):
        f = by_name.get(candidate)
        if f is None:
            continue
        got = [(a.name, str(a.type)) for a in f.func.arguments.flat_non_out]
        if got == want:
            twins.append(f)
    return tuple(twins)


def sibling_ops_for(
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    specs: dict[str, SiblingSpec],
) -> list[SiblingOp]:
    by_name: dict[str, NativeFunction] = {}
    for g in grouped_native_functions:
        for f in g.functions() if isinstance(g, NativeFunctionsGroup) else (g,):
            by_name[str(f.func.name)] = f
    ops = []
    seen: set[str] = set()
    for g in grouped_native_functions:
        if isinstance(g, NativeFunctionsGroup):
            functional, inplace, out, group = g.functional, g.inplace, g.out, g
        else:
            functional, inplace, out, group = g, None, None, None
        name = str(functional.func.name)
        if group is not None and group.out.ufunc_inner_loop:
            base = functional.func.name.name.base
            spec = SiblingSpec(base, group.out.ufunc_inner_loop, None, shape="ufunc")
        elif name in specs:
            spec = specs[name]
        else:
            continue
        if spec.shape == "ufunc" and group is None:
            raise AssertionError(
                f"sibling {name}: the ufunc shape needs the op's structured group"
            )
        seen.add(name)
        ops.append(
            SiblingOp(
                spec,
                functional,
                inplace,
                out,
                scalar_twins_of(functional, by_name),
                group,
            )
        )
    missing = set(specs) - seen
    if missing:
        raise AssertionError(
            f"siblings.yaml names ops not in native_functions.yaml: {sorted(missing)}"
        )
    return ops


# ---- the ufunc shape (compute_ufunc_cuda's functors on the sibling iterator)


def sibling_entry_arguments(g: NativeFunctionsGroup) -> list[Binding]:
    # the impl signature's bindings without the out tensor: the functional
    # arguments as the kernel host sees them (const at::Tensor &, const at::Scalar &)
    sig = StructuredImplSignature(g, ufunc.kernel_name(g, DispatchKey.CUDA))
    outs = {a.name for a in g.out.func.arguments.out}
    return [b for b in sig.arguments() if b.name not in outs]


def sibling_functor_proxy(name: str, fields: Sequence[tuple[str, str]]) -> str:
    # Traced<F>: each constant member as a field of the tape at its offset,
    # named as the member (`fields` are (member, type) pairs, the type an
    # alias of F).  The storage is raw bytes: a ufunc functor has a
    # constructor and no default one, and zeroed bytes give the same image at
    # the trace and at the build.
    if not fields:
        return ""
    P = f"at::native::{name}<scalar_t>"
    members = "\n".join(
        f'  ti::gen::field_t<typename P::{t}, offsetof(P, {m})> {m}{{this, "{m}"}};'
        for m, t in fields
    )
    return f"""
// the proxy through which {name}'s fields reach the kernel as values
template <class scalar_t>
struct Traced<{P}> : TracedBase {{
  using P = {P};
  alignas(P) unsigned char pod_bytes[sizeof(P)] = {{}};
{members}
  Traced() : TracedBase(pod_bytes, sizeof(P)) {{}}
}};
"""


def sibling_functor_view(name: str, fields: Sequence[tuple[str, str]]) -> str:
    if not fields:
        return ""
    F = f"at::native::{name}<scalar_t>"
    slots = "\n".join(f"  gen::slot_t<typename F::{t}> {m};" for m, t in fields)
    inits = ",\n".join(
        f'        {m}(o, base + offsetof(F, {m}), SlotName{{&this->nm, "{m}"}})'
        for m, _ in fields
    )
    assigns = "\n".join(
        f"    {m} = static_cast<gen::sym_t<typename F::{t}>>(s.{m});" for m, t in fields
    )
    return f"""
// {name}'s fields inside a StridedOp proxy
template <class scalar_t>
struct FunctorView<{F}> {{
  using F = {F};
  TracedBase* o;
  size_t base;
  SlotName nm;
{slots}
  FunctorView(TracedBase* o, size_t base, SlotName nm)
      : o(o), base(base), nm(nm),
{inits} {{}}
  FunctorView(const FunctorView&) = delete;
  FunctorView& operator=(const Traced<F>& s) {{
{assigns}
    return *this;
  }}
}};
"""


def wrapped_proxy(F: str) -> str:
    # Traced<BinaryFunctor<..., F>> (Loops.cuh; what gpu_kernel_with_scalars
    # launches for two tensor operands) as F's own proxy: the wrapper holds the
    # functor as its only member, at offset 0 (standard layout, the same size
    # and alignment; nvcc's front end does not grant is_layout_compatible to a
    # private member against a public twin), so the fields sit at the same
    # offsets and the launch image is the functor's bytes
    B = f"at::native::BinaryFunctor<scalar_t, scalar_t, scalar_t, {F}>"
    return f"""
// the same fields inside the BinaryFunctor eager's gpu_kernel_with_scalars wraps around the functor
template <class scalar_t>
struct Traced<{B}> : Traced<{F}> {{
  using P = {B};
  static_assert(sizeof(P) == sizeof({F}) && alignof(P) == alignof({F}) && std::is_standard_layout_v<P>, "BinaryFunctor holds its functor as its only member");
}};
"""


def wrapped_view(F: str) -> str:
    B = f"at::native::BinaryFunctor<scalar_t, scalar_t, scalar_t, {F}>"
    return f"""
// the wrapped functor inside a StridedOp proxy: the functor's view
template <class scalar_t>
struct FunctorView<{B}> : FunctorView<{F}> {{
  using FunctorView<{F}>::FunctorView;
  FunctorView& operator=(const Traced<{B}>& s) {{
    FunctorView<{F}>::operator=(static_cast<const Traced<{F}>&>(s));
    return *this;
  }}
}};
"""


def ufunc_fields(sig: UfunctorSignature) -> list[tuple[str, str]]:
    return [(f.name, "opmath_t") for f in sig.fields()]


def sibling_launch(sig: UfunctorSignature, ctx: Sequence[Binding | Expr]) -> str:
    F = f"at::native::{sig.name}<scalar_t>"
    fields = sig.fields()
    if not fields:
        return f"  gpu_kernel(iter, {F}{{}});\n"
    exprs = translate(ctx, sig.arguments().ctor)
    assigns = "\n".join(
        f"  f.{fld.name} = sibling_value({e.expr});" for fld, e in zip(fields, exprs)
    )
    return f"  Traced<{F}> f;\n{assigns}\n  gpu_kernel(iter, f);\n"


def sibling_dtype_body(
    inner_loops: dict[UfuncKey, UfunctorSignature], parent_ctx: Sequence[Binding]
) -> str:
    # compute_ufunc_cuda_dtype_body's shape: a CPU scalar on either side is
    # read as a value, removed from the iterator and launched as the
    # CUDAFunctorOnSelf / CUDAFunctorOnOther functor; otherwise the plain functor
    body = "using opmath_t = at::opmath_type<scalar_t>;\nif (false) {}\n"
    for config in BinaryScalarSpecializationConfigs:
        if config.ufunc_key not in inner_loops:
            continue
        scalar_idx = config.scalar_idx + 1
        ctx: list[Expr | Binding] = list(parent_ctx)
        ctx.append(
            Expr(
                expr=f"iter.scalar_value<opmath_t>({scalar_idx})",
                type=NamedCType(config.ctor_tensor, BaseCType(opmath_t)),
            )
        )
        body += f"else if (iter.is_cpu_scalar({scalar_idx})) {{\n"
        body += sibling_launch(inner_loops[config.ufunc_key], ctx)
        body += f"  iter.remove_operand({scalar_idx});\n}}\n"
    body += (
        "else {\n"
        + sibling_launch(inner_loops[UfuncKey.CUDAFunctor], list(parent_ctx))
        + "}\n"
    )
    # the operand is removed after the functor read its value; the launch must
    # follow the removal, so reorder the two lines the helper emitted
    for i in (1, 2):
        body = body.replace(
            f"  gpu_kernel(iter, f);\n  iter.remove_operand({i});\n",
            f"  iter.remove_operand({i});\n  gpu_kernel(iter, f);\n",
        )
    return body


def ufunc_shape_parts(op: SiblingOp) -> tuple[str, str, str, dict[ScalarType, str]]:
    # the functors are eager's own, defined by compute_ufunc_cuda in the
    # UfuncCUDA_<name>.cu that includes the fragment: nothing to emit but the
    # proxies and views over them
    g = op.group
    if g is None:
        raise AssertionError(
            f"sibling {op.name}: the ufunc shape needs a structured group"
        )
    ufunctor_sigs, _ = compute_ufunc_cuda_functors(g)
    seen: dict[str, UfunctorSignature] = {}
    for inner in ufunctor_sigs.values():
        for sig in inner.values():
            seen.setdefault(sig.name, sig)
    proxies = "".join(
        sibling_functor_proxy(sig.name, ufunc_fields(sig)) for sig in seen.values()
    )
    views = "".join(
        sibling_functor_view(sig.name, ufunc_fields(sig)) for sig in seen.values()
    )
    args = sibling_entry_arguments(g)
    bodies = {
        d: sibling_dtype_body(ufunctor_sigs[d], args)
        for d in ufunctor_sigs
        if d in SIBLING_DTYPES
    }
    return "", proxies, views, bodies


# ---- the eager shape (the hand-written kernel host's functor)


def eager_field_type(conv: str) -> str:
    return "acc_t" if conv == "acc" else "opmath_t"


def eager_field_expr(name: str, is_scalar: bool, conv: str) -> str:
    # the host conversion eager applies to the argument (a Scalar, or a plain
    # number: a float / int argument or a constant) before the launch
    if is_scalar:
        if conv == "scalar":
            return f"sibling_value(static_cast<opmath_t>(({name}).to<scalar_t>()))"
        return f"sibling_value(({name}).to<{eager_field_type(conv)}>())"
    if conv == "scalar":
        return f"sibling_value(static_cast<opmath_t>(static_cast<scalar_t>({name})))"
    return f"sibling_value(static_cast<{eager_field_type(conv)}>({name}))"


def eager_shape_parts(op: SiblingOp) -> tuple[str, str, str, dict[ScalarType, str]]:
    spec = op.spec
    dtypes: OrderedSet[ScalarType] = OrderedSet()
    ufunc_name = None
    for lk in (UfuncKey.ScalarOnly, UfuncKey.Generic):
        if lk in spec.loops:
            ufunc_name = spec.loops[lk].name
            dtypes |= spec.loops[lk].supported_dtypes
    if ufunc_name is None:
        raise AssertionError(f"sibling of {spec.name}: no Generic / ScalarOnly loop")
    for a in spec.fields:
        if a not in {v.name for v in op.values} | set(spec.constants):
            raise AssertionError(
                f"sibling of {spec.name}: field {a} is not a non-tensor argument or constant"
            )
    for a in spec.arg_types:
        if a not in {t.name for t in op.tensors}:
            raise AssertionError(
                f"sibling of {spec.name}: arg_types names {a}, not a tensor argument"
            )
    # (member, type alias, host expression) per field: the non-tensor
    # arguments in schema order, then the constants
    conv = {
        n: spec.fields.get(n, "opmath")
        for n in [a.name for a in op.values] + list(spec.constants)
    }
    members = [
        (
            f"{a.name}_",
            eager_field_type(conv[a.name]),
            eager_field_expr(a.name, a.type == BaseType(BaseTy.Scalar), conv[a.name]),
        )
        for a in op.values
    ] + [
        (f"{n}_", eager_field_type(conv[n]), eager_field_expr(v, False, conv[n]))
        for n, v in spec.constants.items()
    ]
    fields = [(m, t) for m, t, _ in members]
    if fields and spec.symmetric:
        raise AssertionError(
            f"sibling of {spec.name}: LoopsSym holds no fields inside AUnaryFunctor"
        )
    if spec.arg_types and (spec.symmetric or spec.scalars):
        raise AssertionError(
            f"sibling of {spec.name}: a fixed argument dtype needs the plain launch"
        )
    if len(op.tensors) not in (1, 2, 3):
        raise AssertionError(f"sibling of {spec.name}: {len(op.tensors)} tensor inputs")
    if spec.functor is not None and (fields or spec.arg_types or spec.variants):
        raise AssertionError(
            f"sibling of {spec.name}: a named functor takes no fields, arg_types or variants"
        )
    variants = [f"_{v}" for v in spec.variants[1]] if spec.variants else [""]
    # the aliases the field conversions use (an unused alias is a warning)
    convs = set(conv.values())
    acc = "  using acc_t = at::acc_type<scalar_t, true>;\n" if "acc" in convs else ""
    opmath = (
        "  using opmath_t = at::opmath_type<scalar_t>;\n"
        if convs & {"opmath", "scalar"} or "opmath_t" in (spec.functor or "")
        else ""
    )
    params = ", ".join(
        f"{spec.arg_types.get(a.name, 'scalar_t')} {a.name}" for a in op.inputs
    )
    call = ", ".join([a.name for a in op.inputs] + [m for m, _ in fields])
    decls = "".join(f"  {t} {m};\n" for m, t in fields)
    functors = ""
    proxies = ""
    views = ""
    # eager's gpu_kernel_with_scalars wraps the functor in BinaryFunctor for
    # two tensor operands; with fields the wrapped proxy (emitted beside the
    # functor's own) is launched directly and a CPU scalar operand declines
    # (its AUnaryFunctor would hold the scalar beside the fields)
    wrapped = bool(fields) and spec.scalars and len(op.tensors) == 2
    for suffix in variants if spec.functor is None else ():
        functor = f"CUDAFunctor_{spec.file_name}{suffix}"
        functors += f"""
template <typename scalar_t>
struct {functor} {{
  using opmath_t = at::opmath_type<scalar_t>;
{acc}{decls}  __device__ scalar_t operator()({params}) const {{
    return ufunc::{ufunc_name}{suffix}({call});
  }}
}};
"""
        proxies += sibling_functor_proxy(functor, fields)
        views += sibling_functor_view(functor, fields)
        if wrapped:
            proxies += wrapped_proxy(f"at::native::{functor}<scalar_t>")
            views += wrapped_view(f"at::native::{functor}<scalar_t>")
    if len(op.tensors) == 1 or not (spec.symmetric or spec.scalars) or wrapped:
        helper = "gpu_kernel"
    elif spec.symmetric:
        helper = "opmath_symmetric_gpu_kernel_with_scalars<scalar_t>"
    else:
        helper = "gpu_kernel_with_scalars"
    assigns = "".join(f"    f.{m} = {e};\n" for m, _, e in members)
    launches = []
    for suffix in variants:
        functor = (
            spec.functor
            or f"at::native::CUDAFunctor_{spec.file_name}{suffix}<scalar_t>"
        )
        if wrapped:
            functor = (
                f"at::native::BinaryFunctor<scalar_t, scalar_t, scalar_t, {functor}>"
            )
        if fields:
            launches.append(
                f"    Traced<{functor}> f;\n{assigns}    {helper}(iter, f);\n"
            )
        else:
            launches.append(f"    {helper}(iter, {functor}{{}});\n")
    if spec.variants:
        selector, values = spec.variants
        body = opmath + acc + "  if (false) {\n  }"
        for value, launch in zip(values, launches):
            body += f' else if ({selector} == "{value}") {{\n{launch}  }}'
        either = (
            f"either {values[0]} or {values[1]}"
            if len(values) == 2
            else "one of " + ", ".join(values)
        )
        body += f' else {{\n    TORCH_CHECK(false, "{selector} argument must be {either}.");\n  }}\n'
    else:
        body = opmath + acc + launches[0]
    allowed = (
        SIBLING_DTYPES if fields else OrderedSet([*SIBLING_DTYPES, ScalarType.Bool])
    )
    bodies = {d: body for d in dtypes if d in allowed}
    return functors, proxies, views, bodies


def compute_host_trace_sibling_cuda(op: SiblingOp) -> dict[str, str]:
    spec = op.spec
    if spec.shape == "eager":
        functors, proxies, views, bodies = eager_shape_parts(op)
    else:
        functors, proxies, views, bodies = ufunc_shape_parts(op)
    dtypes = list(bodies)
    supported = " || ".join(f"dtype == ScalarType::{d}" for d in dtypes)
    cases = "\n".join(
        f"""    AT_DISPATCH_CASE(at::ScalarType::{d}, [&]() {{
{bodies[d]}    }})"""
        for d in dtypes
    )
    inputs = [a.name for a in op.inputs]
    plain = [a for a in inputs if a not in spec.arg_types]
    builder = {1: "unary_op", 2: "binary_op", 3: "ternary_op"}[len(inputs)]
    destination = "result" if spec.output == "suggest_memory_format" else "out"
    checks = ""
    if spec.arg_types:
        # the operands of a fixed dtype beside the scalar_t ones: the iterator
        # neither promotes nor checks; the entry fixes the dispatch dtype to
        # the first plain operand's and requires the rest to match it
        config = (
            "  TensorIteratorSymConfig config;\n"
            f"  config.allow_cpu_scalars_ = {'true' if len(inputs) > 1 else 'false'};\n"
            "  config.check_all_same_dtype_ = false;\n"
            f"  config.static_dtype_ = {plain[0]}.scalar_type();\n"
        )
        iterator = (
            f"TensorIteratorSym::{builder}({destination}, {', '.join(inputs)}, config)"
        )
        for a, t in spec.arg_types.items():
            checks += (
                f"  if ({a}.scalar_type() != ScalarType::{ARG_TYPES[t]}) {{\n"
                f'    decline(c10::str("host_trace: {spec.name} with a ", {a}.scalar_type(), " {a} is not traced (declined)"));\n'
                "  }\n"
            )
        for a in plain[1:]:
            checks += (
                f"  if ({a}.scalar_type() != {plain[0]}.scalar_type()) {{\n"
                f'    decline(c10::str("host_trace: {spec.name} on ", {plain[0]}.scalar_type(), " and ", {a}.scalar_type(), ": type promotion is not traced (declined)"));\n'
                "  }\n"
            )
    else:
        config = ""
        iterator = f"TensorIteratorSym::{builder}({destination}, {', '.join(inputs)})"
    if spec.output == "suggest_memory_format":
        checks += f"  const Tensor result = out.defined() ? out : at::empty_like({plain[0]}, {plain[0]}.suggest_memory_format());\n"
    scalars = ""
    wrapped = (
        spec.shape == "eager" and spec.scalars and bool(op.values or spec.constants)
    )
    if (
        len(inputs) > 1
        and spec.shape == "eager"
        and (not (spec.symmetric or spec.scalars) or wrapped)
    ):
        # eager's plain gpu_kernel asserts on a CPU scalar operand; the
        # sibling declines it by name before any launch
        cond = " || ".join(f"iter.is_cpu_scalar({i + 1})" for i in range(len(inputs)))
        scalars = (
            f"  if ({cond}) {{\n"
            f'    decline("host_trace: {spec.name} with a CPU scalar operand is not traced (declined)");\n'
            "  }\n"
        )
    params = [f"{sibling_argument_cpp(a)} {a.name}" for a in op.arguments] + [
        "const Tensor& out"
    ]
    signature = f"Tensor {spec.file_name}_traced({', '.join(params)})"
    entry = f"""
{signature} {{
{checks}{config}  TensorIteratorSym iter = {iterator};
{scalars}  const ScalarType dtype = iter.dtype(0);
  if (!({supported})) {{
    decline(c10::str("host_trace: {spec.name} on ", dtype, " is not traced (declined)"));
  }}
  AT_DISPATCH_SWITCH(dtype, "{spec.file_name}_traced",
{cases}
  );
  return iter.output();
}}
"""
    functor_header = f"HostTraceFunctor_{spec.file_name}.cuh" if functors else ""
    return {
        "name": spec.name,
        "header": spec.header or "",
        "functor_header": functor_header,
        "functor_include": f"#include <ATen/{functor_header}>\n" if functors else "",
        "functors": functors,
        "proxies": proxies,
        "views": views,
        "entry": entry,
        "declaration": f"TORCH_CUDA_CU_API {signature};",
    }


def sibling_binding(op: SiblingOp) -> tuple[str, str]:
    spec = op.spec
    ntensors = len(op.tensors)
    params, calls, captures = [], [], []
    for a in op.arguments:
        if a.type == BaseType(BaseTy.Tensor):
            if ntensors >= 2:
                params.append(f"const py::handle& {a.name}")
                calls.append(f"operand({a.name})")
                if "operand" not in captures:
                    captures.append("operand")
            else:
                params.append(f"const at::Tensor& {a.name}")
                calls.append(a.name)
        elif a.type == BaseType(BaseTy.Scalar):
            params.append(f"const py::handle& {a.name}")
            calls.append(f"scalar_arg({a.name})")
            if "scalar_arg" not in captures:
                captures.append("scalar_arg")
        elif a.type == BaseType(BaseTy.str):
            params.append(f"const std::string& {a.name}")
            calls.append(a.name)
        else:
            params.append(f"{sibling_argument_cpp(a)} {a.name}")
            calls.append(a.name)
    params.append("const std::optional<at::Tensor>& out")
    calls.append("out.value_or(at::Tensor())")
    binding = f"_host_trace_ti_gen_{spec.file_name}"
    text = f"""  m.def("{binding}", [{", ".join(captures)}]({", ".join(params)}) {{
    return gen::{spec.file_name}_traced({", ".join(calls)});
  }});
"""
    overloads = ", ".join(f'"{f.func.name}"' for f in op.overloads())
    hand = "true" if spec.hand_entry else "false"
    row = f'      {{"{op.functional.func.name}", "{binding}", {{{overloads}}}, {ntensors}, {hand}}},'
    return text, row


def sibling_pyi_stub(op: SiblingOp) -> str:
    ntensors = len(op.tensors)
    params = []
    for a in op.arguments:
        t = a.type
        if t == BaseType(BaseTy.Tensor):
            hint = "Tensor | Number" if ntensors >= 2 else "Tensor"
        elif t == BaseType(BaseTy.Scalar):
            hint = "Number"
        elif t == BaseType(BaseTy.str):
            hint = "str"
        else:
            hint = {"float": "_float", "int": "_int", "bool": "_bool"}[str(t)]
        params.append(f"{a.name}: {hint}")
    params.append("out: Tensor | None")
    return f"def _host_trace_ti_gen_{op.spec.file_name}({', '.join(params)}) -> Tensor: ..."


# the fragment's place in eager's translation unit: after eager's namespace
# closes, under this banner (the hand-written eager files carry the same
# lines; UfuncCUDA.cu's template takes them from ufunc_cuda_sibling_env)
def sibling_include(name: str, host: str) -> str:
    return f"""
// ---- host tracing (ATen/cuda/host_trace): the traced sibling of {host}, compiled here
// so the sibling and the host above instantiate the one kernel (DECISIONS E36): the tape's
// launch is eager's function object, not a twin. Generated by torchgen from
// cuda/host_trace/ti/siblings.yaml or the op's ufunc_inner_loop; the entry, its proxies and
// its strided views. Outside a trace it runs the same launches in ordinary mode.
#include <ATen/HostTraceSibling_{name}.cuh>
"""


def ufunc_cuda_sibling_env(g: NativeFunctionsGroup, rocm: bool) -> dict[str, str]:
    # UfuncCUDA_<name>.cu: the traced sibling lives in this file (the
    # functors are eager's, defined above it), which needs the method
    # operators (TensorIteratorSym is at::Tensor); ROCm has no host tracing
    if rocm:
        return {
            "operators_guard": "TORCH_ASSERT_NO_OPERATORS",
            "host_trace_sibling": "",
        }
    name = g.functional.func.name.name.base
    return {
        "operators_guard": "TORCH_ASSERT_ONLY_METHOD_OPERATORS",
        "host_trace_sibling": sibling_include(name, f"{name}_kernel"),
    }


def gen_host_trace_siblings(
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    specs: dict[str, SiblingSpec],
    fm: FileManager,
) -> None:
    ops = sibling_ops_for(grouped_native_functions, specs)
    if not ops:
        return
    computed = [compute_host_trace_sibling_cuda(op) for op in ops]
    bindings = [sibling_binding(op) for op in ops]
    fm.write_with_template(
        "HostTraceSiblingOps.h",
        "HostTraceSiblingOps.h",
        lambda: {"declarations": "\n".join(c["declaration"] for c in computed)},
    )
    fm.write_with_template(
        "HostTraceSiblingBindings.h",
        "HostTraceSiblingBindings.h",
        lambda: {
            "bindings": "".join(b[0] for b in bindings),
            "table": "\n".join(b[1] for b in bindings),
        },
    )
    for op, c in zip(ops, computed):
        if c["functors"]:
            fm.write_with_template(
                c["functor_header"],
                "HostTraceFunctor.cuh",
                lambda c=c: c,  # type: ignore[misc]
            )
        fm.write_with_template(
            f"HostTraceSibling_{op.spec.file_name}.cuh",
            "HostTraceSibling.cuh",
            lambda c=c: c,  # type: ignore[misc]
        )
