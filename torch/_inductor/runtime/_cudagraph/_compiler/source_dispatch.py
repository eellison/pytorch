from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.accessors import _function, _snapshot
from torch._inductor.runtime._cudagraph._compiler.branch_abi import _exact_uses
from torch._inductor.runtime._cudagraph._compiler.emitter_v2 import _kernel, _validate_tree, _walk
from torch._inductor.runtime._cudagraph._compiler.entry_signature import MetadataSnapshot, snapshot_metadata
from torch._inductor.runtime._cudagraph._compiler.launch_config import _constant, _DEFAULT_ATTRIBUTES, _owner
from torch._inductor.runtime._cudagraph._compiler.tma_requirements import read_tma_requirements, TmaDimensionRequirement, TmaStrideRequirement


SOURCE_DISPATCH_VERSION = 1


@dataclass(frozen=True)
class SourceDiagnostic:
    kind: str
    operation: Any
    predicate: Any
    requested: Any
    kernel_query: Any | None
    limit: int | None
    arch: str | None
    expected: bool = False


@dataclass(frozen=True)
class SourceLaunchSite:
    arm: bool | None
    block: Any
    launch: Any
    config: Any
    kernel: Any
    callee: tuple[str, ...]
    arguments: tuple[Any, ...]
    argument_types: tuple[str, ...]
    stream_source_arg_index: int
    block_dims: tuple[Any, ...]
    grid: tuple[Any, ...]
    shared: Any
    attributes: tuple[Any, ...]
    diagnostics: tuple[SourceDiagnostic, ...]
    status: tuple[Any, Any]
    yield_op: Any
    cluster: tuple[int, int, int] | None = None
    tma_strides: tuple[TmaStrideRequirement, ...] = ()
    tma_dimensions: tuple[TmaDimensionRequirement, ...] = ()


def _tma_requirements(values: tuple[Any, ...]) -> tuple[tuple[TmaStrideRequirement, ...], tuple[TmaDimensionRequirement, ...]]:
    from cutlass._mlir import ir

    visited, strides, dimensions = set(), [], []

    def visit(value: Any) -> None:
        operation = _owner(value)
        if not isinstance(operation, ir.Operation) or operation in visited:
            return
        visited.add(operation)
        for operand in operation.operands:
            visit(operand)
        if operation.name in {"cute_nvgpu.atom.make_non_exec_tiled_tma_load",
                              "cute_nvgpu.atom.make_non_exec_tiled_tma_store"}:
            stride_rows, dimension_rows = read_tma_requirements(operation)
            strides.extend(stride_rows)
            dimensions.extend(dimension_rows)

    for value in values:
        visit(value)
    return tuple(strides), tuple(dimensions)


def _pure(module: Any, operation: Any, kernels: dict[Any, Any]) -> None:
    from cutlass._mlir import ir

    if operation.name == "scf.if":
        if (operation.attributes or operation.successors or len(operation.operands) != 1
                or str(operation.operands[0].type) != "i1" or not operation.results
                or any(str(value.type) not in {"i1", "i32", "i64"} for value in operation.results)
                or len(operation.regions) != 2):
            raise ValueError("Expected a pure scalar-result scf.if with two regions")
        for region in operation.regions:
            if len(region.blocks) != 1 or region.blocks[0].arguments:
                raise ValueError("Scalar branches require one block without arguments")
            body = tuple(view.operation for view in region.blocks[0].operations)
            if (not body or body[-1].name != "scf.yield" or body[-1].attributes or body[-1].results
                    or body[-1].regions or body[-1].successors
                    or tuple(value.type for value in body[-1].operands) != tuple(value.type for value in operation.results)):
                raise ValueError("Scalar branch yield does not preserve exact result types")
            for nested in body[:-1]:
                _pure(module, nested, kernels)
    else:
        # The source owner snapshots the whole module; dispatch only needs kernel identities.
        _validate_tree(module, operation, kernels, snapshot_kernels=False)
    for nested in _walk(operation):
        for value in (*nested.operands, *nested.results):
            if isinstance(value.type, ir.IntegerType) and str(value.type) not in {"i1", "i32", "i64"}:
                raise ValueError("Unsupported source scalar width")


def _diagnostic_body(operation: Any) -> tuple[Any, Any]:
    from cutlass._mlir import ir

    if (operation.name != "scf.if" or operation.results or operation.attributes or operation.successors
            or len(operation.operands) != 1 or str(operation.operands[0].type) != "i1"
            or len(operation.regions) != 2 or len(operation.regions[0].blocks) != 1
            or operation.regions[0].blocks[0].arguments or operation.regions[1].blocks):
        raise ValueError("Unsupported shared-memory diagnostic control flow")
    body = tuple(view.operation for view in operation.regions[0].blocks[0].operations)
    if (len(body) != 2 or tuple(op.name for op in body) != ("cute.print", "scf.yield")
            or any(op.results or op.regions or op.successors for op in body)
            or body[1].operands or body[1].attributes
            or set(body[0].attributes) != {"fmt"} or not isinstance(body[0].attributes["fmt"], ir.StringAttr)):
        raise ValueError("Unsupported shared-memory diagnostic effects")
    comparison = _owner(operation.operands[0])
    if (not isinstance(comparison, ir.Operation) or comparison.name != "arith.cmpi"
            or comparison.regions or comparison.successors or len(comparison.results) != 1
            or len(comparison.operands) != 2 or set(comparison.attributes) != {"predicate"}
            or not isinstance(comparison.attributes["predicate"], ir.IntegerAttr)):
        raise ValueError("Shared-memory diagnostic requires an exact typed comparison")
    _exact_uses(comparison.results[0], [(operation, 0)])
    return body[0], comparison


def _is_i64_request(value: Any, requested: Any) -> bool:
    from cutlass._mlir import ir

    if str(requested.type) == "i64":
        return value == requested
    operation = _owner(value)
    return (str(requested.type) == "i32" and isinstance(operation, ir.Operation)
            and operation.name == "arith.extsi" and not operation.attributes
            and not operation.regions and not operation.successors
            and tuple(operation.operands) == (requested,) and len(operation.results) == 1
            and str(value.type) == "i64")


def _diagnostics(module: Any, operations: tuple[Any, ...], config: Any,
                 kernel: Any, callee: tuple[str, ...]) -> tuple[SourceDiagnostic, ...]:
    from cutlass._mlir import ir
    from cutlass.cutlass_dsl.cutlass import SMEM_CAPACITY_MAP

    diagnostics = [op for op in operations if op.name == "scf.if" and not op.results]
    if len(diagnostics) not in (1, 2) or any(operations.index(op) >= operations.index(config) for op in diagnostics):
        raise ValueError("Expected complete compiler shared-memory diagnostics before each configuration")
    upper = diagnostics[-1]
    printing, comparison = _diagnostic_body(upper)
    requested, limit_value = comparison.operands
    if (comparison.attributes["predicate"].value != 4 or str(requested.type) not in {"i32", "i64"}
            or tuple(printing.operands) != (requested,) or not _is_i64_request(config.operands[3], requested)):
        raise ValueError("Wrong upper-bound diagnostic predicate or requested bytes")
    limit = _constant(limit_value, int(str(requested.type)[1:]))
    text = printing.attributes["fmt"].value
    prefix = f"\nError: kernel '@{callee[0]}::@{callee[1]}' launch shared memory exceeds current GPU arch "
    suffix = f" allowed. Allocated: {{}} bytes. Max: {limit} bytes.\n\n"
    if not text.startswith(prefix) or not text.endswith(suffix):
        raise ValueError("Wrong upper-bound diagnostic message or limit")
    arch = text[len(prefix):-len(suffix)]
    match = re.fullmatch(r"(sm_[0-9]+)[af]?", arch)
    if match is None or SMEM_CAPACITY_MAP.get(match[1]) != limit:
        raise ValueError("Shared-memory diagnostic differs from the compiler architecture capacity")
    result = []
    if len(diagnostics) == 2:
        lower = diagnostics[0]
        printing, comparison = _diagnostic_body(lower)
        query = _owner(comparison.operands[1])
        if (comparison.attributes["predicate"].value != 2
                or not _is_i64_request(comparison.operands[0], requested)
                or not isinstance(query, ir.Operation) or query.name != "cute.kernel_smem_size"
                or _kernel(module, query) != kernel or query not in operations
                or operations.index(query) >= operations.index(lower)
                or tuple(printing.operands) != (requested, query.results[0])):
            raise ValueError("Wrong kernel-need diagnostic predicate, request or exact kernel query")
        expected = (f"\nError: shared memory usage in '@{callee[0]}::@{callee[1]}' "
                    "may exceed available memory set in kernel launch. Allocated: {} bytes. Used: {} bytes.\n\n")
        if printing.attributes["fmt"].value != expected:
            raise ValueError("Wrong kernel-need diagnostic message")
        result.append(SourceDiagnostic("kernel_need", lower, lower.operands[0], requested, query, None, None))
    else:
        query = _owner(requested)
        if (not isinstance(query, ir.Operation) or query.name != "cute.kernel_smem_size"
                or _kernel(module, query) != kernel or query not in operations
                or operations.index(query) >= operations.index(upper)):
            raise ValueError("Explicit shared-memory request is missing its kernel-need diagnostic")
    result.append(SourceDiagnostic("architecture_limit", upper, upper.operands[0], requested, query, limit, arch))
    return tuple(result)


def _success_return(suffix: tuple[Any, ...]) -> None:
    if (len(suffix) != 2 or tuple(op.name for op in suffix) != ("arith.constant", "func.return")
            or len(suffix[1].operands) != 1 or suffix[1].results or suffix[1].attributes
            or any(op.regions or op.successors for op in suffix)
            or suffix[1].operands[0] != suffix[0].results[0] or _constant(suffix[1].operands[0], 32) != 0):
        raise ValueError("Dispatch must finish with the exact root success return")
    _exact_uses(suffix[0].results[0], [(suffix[1], 0)])


def _site(module: Any, block: Any, arm: bool | None, arguments: tuple[Any, ...],
          operations: tuple[Any, ...]) -> SourceLaunchSite:
    from cutlass._mlir import ir

    launches = [op for op in operations if op.name == "cuda.launch_ex"]
    configs = [op for op in operations if op.name == "cuda.launch_cfg.create"]
    if len(launches) != 1 or len(configs) != 1:
        raise ValueError("Each dispatch arm requires one unconditional launch and configuration")
    launch, config = launches[0], configs[0]
    if (set(launch.attributes) != {"callee", "assume_kernel_attr"}
            or not isinstance(launch.attributes["callee"], ir.SymbolRefAttr)
            or launch.attributes["assume_kernel_attr"] != ir.Attribute.parse("#cuda.assume_kernel_attr<true>")
            or launch.regions or launch.successors or len(launch.results) != 1
            or str(launch.results[0].type) != "!cuda.result"):
        raise ValueError("Unsupported source launch attributes or result")
    callee = tuple(launch.attributes["callee"].value)
    if len(callee) != 2:
        raise ValueError("Launch requires exact module and kernel symbols")
    gpu = _function(module, callee[0], "gpu.module")
    kernels = [view.operation for view in gpu.regions[0].blocks[0].operations
               if view.operation.name == "cuda.kernel" and view.operation.attributes["sym_name"].value == callee[1]]
    if (len(kernels) != 1 or tuple(kernels[0].attributes["function_type"].value.inputs)
            != tuple(value.type for value in launch.operands[1:])):
        raise ValueError("Source launch operands do not match the exact kernel signature")
    kernel = kernels[0]
    if (len(config.operands) != 8 or len(config.results) != 1 or config.regions or config.successors
            or set(config.attributes) != {"maxNumAttrs"}
            or not isinstance(config.attributes["maxNumAttrs"], ir.IntegerAttr)
            or str(config.attributes["maxNumAttrs"].type) != "i32" or config.attributes["maxNumAttrs"].value < 2
            or tuple(str(value.type) for value in config.operands) != (
                "i32", "i32", "i32", "i64", "i32", "i32", "i32", "!cuda.stream")
            or not launch.operands or launch.operands[0] != config.results[0]
            or operations.index(config) >= operations.index(launch)):
        raise ValueError("Unsupported semantic launch-configuration fields")
    stream_indices = [index for index, value in enumerate(arguments) if value == config.operands[7]]
    if len(stream_indices) != 1:
        raise ValueError("Launch stream must be an original host formal")
    defaults = tuple(op for op in operations if op.name in _DEFAULT_ATTRIBUTES)
    if len(defaults) != 2 or {op.name for op in defaults} != _DEFAULT_ATTRIBUTES:
        raise ValueError("Expected both disabled default launch attributes")
    for op in defaults:
        if (op.results or op.regions or op.successors or op.attributes or len(op.operands) != 2
                or op.operands[0] != config.results[0] or _constant(op.operands[1], 32) != 0
                or not operations.index(config) < operations.index(op) < operations.index(launch)):
            raise ValueError("Nondefault or misplaced launch attributes")
    clusters = tuple(op for op in operations if op.name == "cuda.launch_cfg.cluster_dim")
    if len(clusters) > 1:
        raise ValueError("Expected at most one fixed cluster dimension attribute")
    cluster = None
    if clusters:
        op, = clusters
        if (op.results or op.regions or op.successors or op.attributes or len(op.operands) != 4
                or op.operands[0] != config.results[0]
                or not operations.index(config) < operations.index(op) < operations.index(launch)):
            raise ValueError("Unsupported or misplaced cluster dimension attribute")
        cluster = tuple(_constant(value, 32) for value in op.operands[1:])
        if any(value <= 0 for value in cluster):
            raise ValueError("Fixed cluster dimensions must be positive compiler constants")
    attributes = tuple(op for op in operations if op in (*defaults, *clusters))
    _exact_uses(config.results[0], [(op, 0) for op in attributes] + [(launch, 0)])
    diagnostics = _diagnostics(module, operations, config, kernel, callee)
    suffix = operations[operations.index(launch) + 1:]
    status = suffix[:2]
    if (len(status) != 2 or tuple(op.name for op in status) != ("cuda.cast", "cuda.return_if_error")
            or tuple(suffix[0].operands) != (launch.results[0],) or len(suffix[0].results) != 1
            or str(suffix[0].results[0].type) != "i32"
            or tuple(suffix[1].operands) != (suffix[0].results[0],) or suffix[1].results
            or any(op.attributes or op.regions or op.successors for op in status)):
        raise ValueError("Unsupported launch status protocol")
    yield_op = None
    if arm is None:
        if len(suffix) != 2:
            raise ValueError("Unconditional launch must end with its exact status protocol")
    else:
        if (len(suffix) != 3 or suffix[2].name != "scf.yield" or suffix[2].operands
                or suffix[2].results or suffix[2].attributes or suffix[2].regions or suffix[2].successors):
            raise ValueError("Unsupported dispatch-arm yield protocol")
        yield_op = suffix[2]
    _exact_uses(launch.results[0], [(suffix[0], 0)])
    _exact_uses(suffix[0].results[0], [(suffix[1], 0)])
    special = {launch, config, *attributes, *(item.operation for item in diagnostics), *suffix}
    queried = {}
    for op in operations:
        if op not in special:
            _pure(module, op, queried)
    if any(queried_kernel != kernel for queried_kernel in queried):
        raise ValueError("Arm computation queries a different exact kernel")
    return SourceLaunchSite(arm, block, launch, config, kernel, callee, tuple(launch.operands[1:]),
        tuple(str(value.type) for value in launch.operands[1:]), stream_indices[0], tuple(config.operands[:3]),
        tuple(config.operands[4:7]), config.operands[3], attributes, diagnostics, status, yield_op, cluster,
        *_tma_requirements(tuple(launch.operands[1:])))


def _inspect(module: Any, function_name: str, metadata: MetadataSnapshot) -> tuple[Any, ...]:
    host = _function(module, function_name, "func.func")
    if (len(host.regions) != 1 or len(host.regions[0].blocks) != 1
            or set(host.attributes) - {"sym_name", "function_type", "llvm.emit_c_interface", "arg_attrs"}):
        raise ValueError("Expected one original host block with standard attributes")
    block = host.regions[0].blocks[0]
    arguments = tuple(block.arguments)
    indices = [param.ir_arg_index for param in metadata.params if param.ir_arg_index is not None]
    if (metadata.symbol_name != function_name or len(indices) != len(set(indices))
            or set(indices) != set(range(len(arguments)))):
        raise ValueError("Metadata must identify every original source formal exactly once")
    if (tuple(host.attributes["function_type"].value.inputs) != tuple(value.type for value in arguments)
            or tuple(str(value) for value in host.attributes["function_type"].value.results) != ("i32",)):
        raise ValueError("Unsupported original host function signature or return type")
    operations = tuple(view.operation for view in block.operations)
    launches = [op for op in operations if op.name == "cuda.launch_ex"]
    if launches:
        sites, start = [], 0
        for launch in launches:
            end = operations.index(launch) + 3
            sites.append(_site(module, block, None, arguments, operations[start:end]))
            start = end
        suffix = operations[start:]
        _success_return(suffix)
        if (len({site.stream_source_arg_index for site in sites}) != 1
                or len({site.diagnostics[-1].arch for site in sites}) != 1):
            raise ValueError("Sequential launches require one original stream and compiler architecture")
        return host, arguments, (), None, None, tuple(sites), suffix
    branches = [op for op in operations if op.name == "scf.if" and not op.results]
    if len(branches) != 1:
        raise ValueError("Expected one root resultless dispatch")
    branch = branches[0]
    if (branch.attributes or branch.successors or len(branch.operands) != 1
            or str(branch.operands[0].type) != "i1" or len(branch.regions) != 2
            or any(len(region.blocks) != 1 or region.blocks[0].arguments for region in branch.regions)):
        raise ValueError("Dispatch requires exactly two single-block resultless arms")
    suffix = operations[operations.index(branch) + 1:]
    _success_return(suffix)
    prefix = operations[:operations.index(branch)]
    queried = {}
    for operation in prefix:
        _pure(module, operation, queried)
    if queried:
        raise ValueError("Kernel resource queries must be local to their dispatch arm")
    sites = tuple(_site(module, region.blocks[0], arm, arguments,
                        tuple(view.operation for view in region.blocks[0].operations))
                  for region, arm in zip(branch.regions, (True, False)))
    if sites[0].callee == sites[1].callee:
        raise ValueError("This dispatch subset requires two distinct exact kernel callees")
    if sites[0].diagnostics[-1].arch != sites[1].diagnostics[-1].arch:
        raise ValueError("Dispatch arms disagree on compiler architecture")
    return host, arguments, prefix, branch, branch.operands[0], sites, suffix


@dataclass(frozen=True)
class SourceDispatch:
    module: Any = field(repr=False)
    context: Any = field(repr=False)
    function_name: str
    metadata: MetadataSnapshot
    host: Any = field(repr=False)
    arguments: tuple[Any, ...] = field(repr=False)
    source_types: tuple[str, ...]
    prefix: tuple[Any, ...] = field(repr=False)
    dispatch: Any = field(repr=False)
    predicate: Any = field(repr=False)
    sites: tuple[SourceLaunchSite, ...]
    root_return: tuple[Any, ...] = field(repr=False)
    _snapshot: tuple[str, bytes] = field(repr=False)
    _seal: tuple[Any, ...] = field(repr=False)

    def _state(self) -> tuple[Any, ...]:
        return (id(self.module), id(self.context), self.function_name, id(self.metadata), id(self.host),
                id(self.arguments), self.source_types, id(self.prefix), id(self.dispatch), id(self.predicate),
                id(self.sites), id(self.root_return), self._snapshot)

    def check(self) -> None:
        from cutlass._mlir import ir

        if self._state() != self._seal or self.module.context != self.context:
            raise RuntimeError("Source dispatch ownership or specification changed")
        with self.context, ir.raw_values():
            if _snapshot(self.module.operation) != self._snapshot:
                raise RuntimeError("Original dispatch Module changed")
            actual = _inspect(self.module, self.function_name, self.metadata)
            if (actual != (self.host, self.arguments, self.prefix, self.dispatch, self.predicate, self.sites, self.root_return)
                    or tuple(str(value.type) for value in self.arguments) != self.source_types):
                raise RuntimeError("Source dispatch lost its exact operations, sites or operands")


def check_dispatch_source(module: Any, function_name: str, metadata: MetadataSnapshot) -> SourceDispatch:
    """Read a complete original host; this structural record does not certify a compiled artifact."""
    from cutlass._mlir import ir

    if not isinstance(module, ir.Module) or type(metadata) is not MetadataSnapshot or type(function_name) is not str:
        raise TypeError("Expected an owned Module and typed original metadata snapshot")
    with module.context, ir.raw_values():
        before = _snapshot(module.operation)
        if not module.operation.verify():
            raise ValueError("Original dispatch Module failed verification")
        host, arguments, prefix, branch, predicate, sites, suffix = _inspect(module, function_name, metadata)
        if _snapshot(module.operation) != before:
            raise RuntimeError("Source dispatch inspection changed original IR")
        result = SourceDispatch(module, module.context, function_name, metadata, host, arguments,
                                tuple(str(value.type) for value in arguments), prefix, branch, predicate, sites, suffix, before, ())
        return replace(result, _seal=result._state())


@dataclass(frozen=True)
class DispatchAdmission:
    program: Any = field(repr=False)
    source: SourceDispatch
    _owners: tuple[Any, Any] = field(repr=False)

    def check(self) -> None:
        if (len(self._owners) != 2 or self.program is not self._owners[0] or self.source is not self._owners[1]):
            raise RuntimeError("Dispatch admission changed its original owners")
        self.program.check()
        self.source.check()
        if (self.source.module is not self.program.source_module or self.source.context is not self.program.source_context
                or self.source.function_name != self.program.function_name or len(self.program.source_metadata) != 1
                or snapshot_metadata(self.program.source_metadata[0]) != self.source.metadata
                or self.source.sites[0].diagnostics[-1].arch != self.program.arch):
            raise RuntimeError("Dispatch source differs from the genuine program metadata or target")


def admit_dispatch_source(program: Any) -> DispatchAdmission:
    program.check()
    if len(program.source_metadata) != 1:
        raise ValueError("Expected one original host metadata record")
    source = check_dispatch_source(program.source_module, program.function_name, snapshot_metadata(program.source_metadata[0]))
    result = DispatchAdmission(program, source, (program, source))
    result.check()
    return result
