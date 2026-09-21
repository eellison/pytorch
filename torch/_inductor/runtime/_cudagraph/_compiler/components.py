from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import version
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.accessors import analyze_accessors, emit_accessors
from torch._inductor.runtime._cudagraph._compiler.aggregate_plan import AggregatePlan, build_aggregate_plan
from torch._inductor.runtime._cudagraph._compiler.compiler_boundary import _snapshot
from torch._inductor.runtime._cudagraph._compiler.compiler_owner import TaggedProgram, _TaggedCollector
from torch._inductor.runtime._cudagraph._compiler.continuation import DispatchCompilation, _DispatchCollector, _metadata_signature, _uses, bind_dispatch_consumers
from torch._inductor.runtime._cudagraph._compiler.dispatch_join import join_dispatch
from torch._inductor.runtime._cudagraph._compiler.entry_signature import snapshot_metadata
from torch._inductor.runtime._cudagraph._compiler.frontend import _TRACE_LOCK
from torch._inductor.runtime._cudagraph._compiler.helpers import ScalarRequest, emit_dispatch_helpers
from torch._inductor.runtime._cudagraph._compiler.launch_events import CloneBundle
from torch._inductor.runtime._cudagraph._compiler.lowering import bind_host_formals
from torch._inductor.runtime._cudagraph._compiler.mapping import bind_dispatch_sites
from torch._inductor.runtime._cudagraph._compiler.source_dispatch import check_dispatch_source


class _ComponentDispatchCollector(_DispatchCollector):
    def __init__(self, owner: Any, signature: Any, arguments: tuple[Any, ...], keywords: dict[str, Any]) -> None:
        super().__init__(owner, signature, arguments, keywords)
        self.accessors: tuple[Any, ...] = ()

    def __call__(self, owner: Any, module: Any, function_name: str) -> None:
        from cutlass._mlir import ir
        from cutlass.cute.metadata import build_function_metadata

        _TaggedCollector.__call__(self, owner, module, function_name)
        metadata = build_function_metadata(function_name=function_name, signature=self.signature,
                                           args=self.arguments, kwonlyargs=self.keywords)
        with module.context, ir.raw_values():
            self.original_text, self.original_bytecode = _snapshot(module)
            self.accessors = emit_accessors(module, function_name, metadata)
            source = check_dispatch_source(module, function_name, snapshot_metadata(metadata))
            prefix = "__cudagraph_dispatch_" + sha256(function_name.encode()).hexdigest()[:16]
            requests = tuple(ScalarRequest(f"{prefix}_{number}", role, index, value, site)
                             for number, (site, role, index, value) in enumerate(_uses(source)))
            self.helpers = emit_dispatch_helpers(source, requests)
            self.text, self.bytecode = _snapshot(module)


@dataclass(frozen=True)
class DispatchComponents:
    dispatch: DispatchCompilation
    plan: AggregatePlan
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        if (self.dispatch is not self._owners[0] or self.plan is not self._owners[1]
                or type(self.dispatch) is not DispatchCompilation or type(self.plan) is not AggregatePlan
                or self.plan.mapping.program is not self.dispatch.program):
            raise RuntimeError("Dispatch/component compiler ownership changed")
        self.dispatch.check()
        self.plan.check()
        formals = {formal.ir_arg_index: formal for formal in self.dispatch.joined.mapping.formals.formals
                   if formal.metadata.kind == "Tensor"}
        if set(formals) != {formal.source_arg_index for formal in self.plan.formals}:
            raise ValueError("Tensor components do not cover the original dispatch tensor formals")
        for tensor in self.plan.formals:
            formal = formals[tensor.source_arg_index]
            if tensor.llvm_arg_index != formal.llvm_arg_index or tensor.llvm_type != formal.llvm_type:
                raise ValueError("Tensor aggregate differs from its exact dispatch LLVM formal")


def compile_dispatch_components(host: Any, kernel: Any, *arguments: Any, arch: str = "sm_80",
                                **keywords: Any) -> DispatchComponents:
    import cutlass.compiler as compiler
    import cutlass.cute as cute
    from cutlass._mlir import ir
    from cutlass.cutlass_dsl.cutlass import CuTeDSL

    if version("nvidia-cutlass-dsl") != "4.6.2" or type(arch) is not str or re.fullmatch(r"sm_[0-9]+[af]?", arch) is None:
        raise ValueError("Expected CuTe 4.6.2 and an explicit supported architecture spelling")
    if ir.Context.current is not None or not _TRACE_LOCK.acquire(blocking=False):
        raise RuntimeError("Dispatch compilation requires an isolated frontend context")
    try:
        bundle = CloneBundle.create(host, kernel)
        dsl = CuTeDSL._get_dsl()
        if (type(dsl) is not CuTeDSL or dsl._trace_finalize_hooks or dsl._scoped_trace_finalize_hooks.get()
                or dsl.envar.keep_ir_clean):
            raise RuntimeError("Dispatch compilation requires the standard isolated frontend")
        collector = _ComponentDispatchCollector(dsl, inspect.signature(bundle.original_host.__wrapped__), arguments, keywords)
        try:
            source = cute.compile.to_precompiled_mlir(bundle.host, *arguments, **keywords, options=f"--gpu-arch {arch}",
                no_jit_engine=True, trace_finalize_hooks=collector)
        finally:
            bundle.check_originals()
            if dsl._trace_finalize_hooks or dsl._scoped_trace_finalize_hooks.get():
                raise RuntimeError("Frontend hook scope was not restored")
        if (collector.calls != 1 or collector.helpers is None or not collector.accessors
                or type(source) is not compiler.PreCompiledMlirArtifact or source.is_consumed
                or source.get_bitcode() != collector.bytecode):
            raise RuntimeError("Helpers do not belong to the actual frontend artifact")
        names = tuple(metadata.symbol_name for metadata in source.metadata)
        if names != (collector.function_name,):
            raise ValueError("Frontend artifact metadata does not identify the original host")
        actual = snapshot_metadata(source.metadata[0])
        if _metadata_signature(actual) != _metadata_signature(collector.helpers.source.metadata):
            raise ValueError("Actual frontend signature metadata differs from helper source admission")
        collector.helpers.check()
    finally:
        _TRACE_LOCK.release()
    serialized = compiler.serialize_compilation_artifact(source)
    snapshot = compiler.deserialize_compilation_artifact(serialized)
    if (type(snapshot) is not compiler.PreCompiledMlirArtifact or snapshot.is_consumed
            or snapshot.get_bitcode() != collector.bytecode or compiler.serialize_compilation_artifact(snapshot) != serialized):
        raise RuntimeError("Owned metadata snapshot differs from the genuine frontend artifact")
    metadata = tuple(snapshot.metadata)
    native = compiler.CuteCompiler()
    native.set_device_target(arch)
    native.set_host_target(compiler.Compiler.detect_host_triple())
    native.set_abi(compiler.Abi.CutlassCall)
    compiled = native.compile_to(source, compiler.ArtifactType.CompiledMlir)
    if (not source.is_consumed or type(compiled) is not compiler.CompiledMlirArtifact or compiled.is_consumed
            or tuple(item.symbol_name for item in compiled.metadata) != names):
        raise RuntimeError("Expected genuine compiled continuation of the helper-bearing source")
    bytecode = compiled.get_bitcode()
    context = ir.Context()
    with context, ir.Location.unknown(), ir.raw_values():
        module = ir.Module.parse(bytecode)
        if not module.operation.verify():
            raise RuntimeError("Compiled dispatch Module failed verification")
        text, module_bytecode = _snapshot(module)
    program = TaggedProgram(bundle, collector.module, collector.context, source, collector.text, collector.bytecode,
        compiled, module, context, bytecode, text, module_bytecode, collector.function_name, arch,
        sha256(collector.bytecode).hexdigest(), sha256(bytecode).hexdigest(), snapshot, metadata, serialized,
        compiler.serialize_compilation_artifact(compiled), (snapshot, metadata))
    program.check()
    formals = bind_host_formals(program, metadata[0], collector.helpers.source.source_types)
    joined = join_dispatch(bind_dispatch_sites(program, formals))
    consumers = bind_dispatch_consumers(joined, collector.helpers)
    owners = program, collector.helpers, joined, consumers, collector.original_text, collector.original_bytecode
    dispatch = DispatchCompilation(*owners, owners)
    plan = build_aggregate_plan(analyze_accessors(program, collector.accessors))
    result = DispatchComponents(dispatch, plan, (dispatch, plan))
    result.check()
    return result
