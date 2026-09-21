from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.argument_flow import ArgumentFlow, LaunchFlow, ParameterFlow
from torch._inductor.runtime._cudagraph._compiler.argument_flow import analyze_argument_flow
from torch._inductor.runtime._cudagraph._compiler.lowering import FormalLowering, FormalLoweringSet


@dataclass(frozen=True)
class ParameterBinding:
    source_parameter_index: int | None
    device_parameter_index: int
    source_value: Any = field(repr=False, compare=False)
    formal: FormalLowering | None
    parameter: ParameterFlow


@dataclass(frozen=True)
class SiteMapping:
    source_launch: Any = field(repr=False, compare=False)
    source_path: tuple[tuple[int, int, int], ...]
    callee: tuple[str, str]
    source_arguments: tuple[Any, ...] = field(repr=False, compare=False)
    argument_types: tuple[str, ...]
    kernel: Any = field(repr=False, compare=False)
    launch: LaunchFlow
    parameters: tuple[ParameterBinding, ...]


def _parameter_bindings(arguments, source_arguments, launch, by_llvm, *, require_direct=False):
    formals = []
    for index, parameter in enumerate(launch.parameters):
        value = parameter.source
        if parameter.index != index or parameter.llvm_type != value.llvm_type:
            return None
        formal = None
        if (value.kind == "argument" and value.value is None and not value.path
                and not value.operands and not value.attributes and value.argument in by_llvm):
            formal = by_llvm[value.argument]
            if parameter.llvm_type != formal.llvm_type:
                return None
        formals.append(formal)
    direct = len(source_arguments) == len(formals) and all(
        formal is not None and source == arguments[formal.ir_arg_index]
        and str(source.type) == formal.source_type
        for source, formal in zip(source_arguments, formals)
    )
    if require_direct and not direct:
        return None
    # Static source operands can disappear during compiler lowering.
    return tuple(ParameterBinding(index if direct else None, parameter.index,
                                  source_arguments[index] if direct else None, formal, parameter)
                 for index, (formal, parameter) in enumerate(zip(formals, launch.parameters)))


def _check_sequence(program, sites):
    from torch._inductor.runtime._cudagraph._compiler.argument_flow import _function
    from torch._inductor.runtime._cudagraph._compiler.branch_abi import _exact_uses, _integer, _ordered, _owner
    from cutlass._mlir import ir

    with program.context, ir.raw_values():
        host = _function(program.module, program.function_name)
        blocks = tuple(host.regions[0].blocks)
        bodies = tuple(tuple(view.operation for view in block.operations) for block in blocks)
        calls, successes = {}, {}
        for ordinal, site in enumerate(sites):
            launch = site.launch
            position = launch.block_index, launch.operation_index
            if (position in calls or launch.block_index in successes
                    or not 0 <= launch.block_index < len(bodies)
                    or not 0 <= launch.operation_index < len(bodies[launch.block_index])):
                raise ValueError("Sequential launch lost its unique compiled operation position")
            body = bodies[launch.block_index]
            call, terminator = body[launch.operation_index], body[-1]
            if (call.name != "llvm.call" or call.attributes["callee"].value != launch.callee
                    or len(call.results) != 1 or str(call.results[0].type) != "i32"
                    or terminator.name != "llvm.cond_br" or len(terminator.operands) != 1
                    or len(terminator.successors) != 2):
                raise ValueError("Sequential launch requires its exact direct status branch")
            status = call.results[0]
            comparison = _owner(terminator.operands[0], "llvm.icmp")
            if (comparison.attributes["predicate"].value != 0 or len(comparison.operands) != 2
                    or comparison.operands[0] != status or _integer(comparison.operands[1]) != 0):
                raise ValueError("Sequential launch status must select success or its own error return")
            _ordered(call, comparison, terminator)
            error = tuple(view.operation for view in terminator.successors[1].operations)
            if (len(error) != 1 or error[0].name != "llvm.return"
                    or tuple(error[0].operands) != (status,)):
                raise ValueError("Sequential launch failure must return its unchanged status")
            _exact_uses(status, [(comparison, 0), (error[0], 0)])
            calls[position] = ordinal
            successes[launch.block_index] = blocks.index(terminator.successors[0])

        pending, visited, reached, returns = [(0, 0)], set(), set(), set()
        while pending:
            block_index, prefix = pending.pop()
            state = block_index, prefix
            if state in visited:
                continue
            visited.add(state)
            body = bodies[block_index]
            if not body:
                raise ValueError("Sequential compiled host contains an empty block")
            for index, operation in enumerate(body):
                position = block_index, index
                if position in calls:
                    if calls[position] != prefix:
                        raise ValueError("Compiled launch order differs from the exact source sequence")
                    prefix += 1
                    reached.add(position)
            terminator = body[-1]
            if block_index in successes:
                pending.append((successes[block_index], prefix))
            elif terminator.name in ("llvm.br", "llvm.cond_br"):
                expected = 1 if terminator.name == "llvm.br" else 2
                if len(terminator.successors) != expected:
                    raise ValueError("Sequential compiled host has an unsupported branch")
                pending.extend((blocks.index(block), prefix) for block in terminator.successors)
            elif terminator.name == "llvm.return":
                if (prefix != len(sites) or len(terminator.operands) != 1
                        or _integer(terminator.operands[0]) != 0):
                    raise ValueError("Normal return does not complete the source launch sequence")
                returns.add(block_index)
            elif (terminator.name == "llvm.unreachable" and len(body) >= 2
                  and body[-2].name == "llvm.intr.trap"):
                continue
            else:
                raise ValueError("Sequential compiled host has an unsupported exit")
        if reached != set(calls) or not returns:
            raise ValueError("Compiled successful paths do not cover the complete source launch sequence")


def _sites(program: Any, formals: FormalLoweringSet, flow: ArgumentFlow) -> tuple[SiteMapping, ...]:
    from cutlass._mlir import ir

    program.check()
    formals.check()
    flow.check()
    if (formals.program is not program or flow.module is not program.module
            or flow.context is not program.context or flow.function_name != program.function_name):
        raise ValueError("Dispatch mapping requires one original compiler and formal owner")
    by_llvm = {item.llvm_arg_index: item for item in formals.formals}
    if (set(by_llvm) != set(range(len(flow.host_types)))
            or any(by_llvm[index].llvm_type != typ for index, typ in enumerate(flow.host_types))):
        raise ValueError("Tagged formals do not cover the actual LLVM entry arguments")
    with program.source_context, ir.raw_values():
        hosts = [view.operation for view in program.source_module.body.operations
                 if view.operation.name == "func.func"
                 and view.operation.attributes["sym_name"].value == program.function_name]
        if len(hosts) != 1 or len(hosts[0].regions[0].blocks) != 1:
            raise ValueError("Expected one original structured host definition")
        host = hosts[0]
        arguments = tuple(host.regions[0].blocks[0].arguments)
        source_sites = []

        def visit(operation: Any, path: tuple[tuple[int, int, int], ...]) -> None:
            for region_index, region in enumerate(operation.regions):
                for block_index, block in enumerate(region.blocks):
                    for operation_index, view in enumerate(block.operations):
                        child = view.operation
                        location = (*path, (region_index, block_index, operation_index))
                        if child.name == "cuda.launch_ex":
                            source_sites.append((child, location))
                        visit(child, location)

        visit(host, ())
        if not source_sites or len(source_sites) != len(flow.launches):
            raise ValueError("Source and compiled launch sets differ")
        result, used_launches = [], set()
        for source, path in source_sites:
            callee = tuple(source.attributes["callee"].value)
            if len(callee) != 2:
                raise ValueError("Each source site requires an exact module and kernel symbol")
            gpu = [view.operation for view in program.source_module.body.operations
                   if view.operation.name == "gpu.module"
                   and view.operation.attributes["sym_name"].value == callee[0]]
            kernels = [] if len(gpu) != 1 else [view.operation for view in gpu[0].regions[0].blocks[0].operations
                if view.operation.name == "cuda.kernel" and view.operation.attributes["sym_name"].value == callee[1]]
            source_arguments = tuple(source.operands)[1:]
            if (len(kernels) != 1 or tuple(kernels[0].attributes["function_type"].value.inputs)
                    != tuple(value.type for value in source_arguments)):
                raise ValueError("Original callee does not own the source operand signature")
            candidates = [launch for launch in flow.launches if launch.registration.kernel_symbol == callee[1]]
            matches = []
            for launch in candidates:
                bindings = _parameter_bindings(arguments, source_arguments, launch, by_llvm,
                                                require_direct=len(candidates) > 1)
                if bindings is not None:
                    matches.append((launch, bindings))
            if len(matches) != 1:
                raise ValueError("Source launch lacks one unique compiled registration and ordered formal pack")
            launch, bindings = matches[0]
            if launch.index in used_launches:
                raise ValueError("A compiled launch cannot identify two source sites")
            used_launches.add(launch.index)
            images = [image for image in flow.binaries if image.library_slot == launch.registration.library_slot]
            if (len(images) != 1 or launch.registration not in flow.registrations
                    or images[0].global_name != launch.registration.binary_global
                    or images[0].sha256 != launch.registration.binary_sha256):
                raise ValueError("Selected registration lost its actual embedded library")
            result.append(SiteMapping(source, path, callee, source_arguments,
                                      tuple(str(value.type) for value in source_arguments), kernels[0],
                                      launch, bindings))
        if used_launches != {launch.index for launch in flow.launches}:
            raise ValueError("Compiled launch coverage is incomplete")
        result = tuple(result)
    if len(result) > 1 and all(len(site.source_path) == 1 for site in result):
        _check_sequence(program, result)
    return result


def _site_state(site: SiteMapping) -> tuple[Any, ...]:
    return (id(site), id(site.source_launch), site.source_path, site.callee,
            tuple(id(value) for value in site.source_arguments), site.argument_types, id(site.kernel),
            id(site.launch), tuple((id(item), item.source_parameter_index, item.device_parameter_index,
                                  id(item.source_value), item.formal, item.parameter) for item in site.parameters))


@dataclass(frozen=True)
class DispatchMapping:
    program: Any = field(repr=False)
    formals: FormalLoweringSet
    flow: ArgumentFlow
    sites: tuple[SiteMapping, ...]
    _owners: tuple[Any, ...] = field(repr=False)
    _site_seals: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        owned = self.program, self.formals, self.flow, self.sites
        if (len(owned) != len(self._owners) or any(value is not owner for value, owner in zip(owned, self._owners))
                or tuple(_site_state(site) for site in self.sites) != self._site_seals):
            raise RuntimeError("Dispatch site or compiler ownership changed")
        actual = _sites(self.program, self.formals, self.flow)
        for expected, found in zip(self.sites, actual):
            if (expected != found or expected.source_launch != found.source_launch
                    or expected.source_arguments != found.source_arguments or expected.kernel != found.kernel
                    or tuple(item.source_value for item in expected.parameters)
                    != tuple(item.source_value for item in found.parameters)):
                raise RuntimeError("Dispatch site lost its exact source/compiled correspondence")
        if len(actual) != len(self.sites):
            raise RuntimeError("Dispatch site coverage changed")


def bind_dispatch_sites(program: Any, formals: FormalLoweringSet, flow: ArgumentFlow | None = None) -> DispatchMapping:
    """Bind source uses to real compiled packs; this does not admit source control/effects."""
    if type(formals) is not FormalLoweringSet:
        raise TypeError("Expected the compiler-owned tagged formal correspondence")
    if flow is None:
        from torch._inductor.runtime._cudagraph._compiler.compiler_type_layout import compile_type_layouts
        from cutlass._mlir.dialects import llvm

        with program.context:
            pointer_layout = compile_type_layouts((llvm.PointerType.get(),), device_target=program.arch)
        flow = analyze_argument_flow(program.module, program.function_name,
                                     local_pointer_width=pointer_layout.layouts[0].size * 8)
    if type(flow) is not ArgumentFlow:
        raise TypeError("Expected the unchanged generic argument-flow reader result")
    sites = _sites(program, formals, flow)
    return DispatchMapping(program, formals, flow, sites, (program, formals, flow, sites),
                           tuple(_site_state(site) for site in sites))
