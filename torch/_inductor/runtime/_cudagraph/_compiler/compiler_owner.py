from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.compiler_boundary import CompiledProgram
from torch._inductor.runtime._cudagraph._compiler.frontend import _Collector


SOURCE_ARGUMENT = "cudagraph.source_arg"


class _TaggedCollector(_Collector):
    def __call__(self, owner: Any, module: Any, function_name: str) -> None:
        from cutlass._mlir import ir

        hooks = tuple(owner._trace_finalize_hooks) + tuple(owner._scoped_trace_finalize_hooks.get())
        if owner is not self.owner or self.calls or hooks != (self,):
            raise RuntimeError("Expected one isolated provenance-tagging hook")
        with module.context, ir.raw_values():
            hosts = [view.operation for view in module.body.operations
                     if view.operation.name == "func.func"
                     and view.operation.attributes["sym_name"].value == function_name]
            if len(hosts) != 1:
                raise RuntimeError("Expected the original source host definition")
            host = hosts[0]
            count = len(host.regions[0].blocks[0].arguments)
            previous = host.attributes.get("arg_attrs")
            if previous is not None and len(previous) != count:
                raise RuntimeError("Source argument attributes disagree with the formals")
            tagged = []
            for index in range(count):
                attrs = {} if previous is None else {item.name: item.attr for item in previous[index]}
                if SOURCE_ARGUMENT in attrs:
                    raise RuntimeError("Source provenance attribute is already assigned")
                attrs[SOURCE_ARGUMENT] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), index)
                tagged.append(ir.DictAttr.get(attrs))
            host.attributes["arg_attrs"] = ir.ArrayAttr.get(tagged)
            if not module.operation.verify():
                raise RuntimeError("Tagged source Module failed verification")
        super().__call__(owner, module, function_name)


@dataclass(frozen=True)
class TaggedProgram(CompiledProgram):
    source_snapshot: Any = field(repr=False)
    source_metadata: tuple[Any, ...] = field(repr=False)
    source_serialized: bytes = field(repr=False)
    compiled_serialized: bytes = field(repr=False)
    _snapshot_owners: tuple[Any, Any] = field(repr=False)

    def check(self) -> None:
        import cutlass.compiler as compiler

        super().check()
        if (self.source_snapshot is not self._snapshot_owners[0]
                or self.source_metadata is not self._snapshot_owners[1]
                or type(self.source_snapshot) is not compiler.PreCompiledMlirArtifact
                or self.source_snapshot.is_consumed
                or self.source_snapshot.get_bitcode() != self.source_bytecode
                or compiler.serialize_compilation_artifact(self.source_snapshot) != self.source_serialized
                or compiler.serialize_compilation_artifact(self.artifact) != self.compiled_serialized):
            raise RuntimeError("Original compiler metadata snapshot changed")
