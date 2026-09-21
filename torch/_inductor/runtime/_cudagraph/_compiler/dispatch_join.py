from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.mapping import DispatchMapping, SiteMapping
from torch._inductor.runtime._cudagraph._compiler.source_dispatch import admit_dispatch_source, DispatchAdmission, SourceLaunchSite


@dataclass(frozen=True)
class BoundDispatchSite:
    source: SourceLaunchSite
    compiled: SiteMapping


@dataclass(frozen=True)
class JoinedDispatch:
    admission: DispatchAdmission
    mapping: DispatchMapping
    sites: tuple[BoundDispatchSite, ...]
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        owned = self.admission, self.mapping, self.sites
        if (len(owned) != len(self._owners)
                or any(value is not owner for value, owner in zip(owned, self._owners))):
            raise RuntimeError("Joined dispatch ownership changed")
        self.admission.check()
        self.mapping.check()
        if (self.admission.program is not self.mapping.program
                or self.admission.source.module is not self.mapping.program.source_module
                or len(self.sites) != len(self.admission.source.sites)
                or len(self.sites) != len(self.mapping.sites)):
            raise ValueError("Source dispatch and compiled launch mapping have different owners or coverage")
        used = set()
        for bound, source in zip(self.sites, self.admission.source.sites):
            compiled = bound.compiled
            if (bound.source is not source or not any(compiled is item for item in self.mapping.sites)
                    or id(compiled) in used or compiled.source_launch != source.launch
                    or compiled.callee != source.callee or compiled.kernel != source.kernel
                    or compiled.source_arguments != source.arguments or compiled.argument_types != source.argument_types):
                raise ValueError("An admitted dispatch arm lost its exact compiled launch site")
            used.add(id(compiled))
            for item in compiled.parameters:
                if item.source_parameter_index is None:
                    if item.source_value is not None:
                        raise ValueError("Compiled-only parameter acquired an unproven source operand")
                elif source.arguments[item.source_parameter_index] != item.source_value:
                    raise ValueError("Compiled parameter lost its exact admitted source operand")
        if used != {id(site) for site in self.mapping.sites}:
            raise ValueError("An actual compiled launch lacks an admitted source arm")


def join_dispatch(mapping: DispatchMapping) -> JoinedDispatch:
    if type(mapping) is not DispatchMapping:
        raise TypeError("Expected the genuine compiler-owned dispatch mapping")
    mapping.check()
    admission = admit_dispatch_source(mapping.program)
    sites = []
    for source in admission.source.sites:
        found = [item for item in mapping.sites if item.source_launch == source.launch]
        if len(found) != 1:
            raise ValueError("An admitted source launch lacks one exact compiled site")
        sites.append(BoundDispatchSite(source, found[0]))
    sites = tuple(sites)
    result = JoinedDispatch(admission, mapping, sites, (admission, mapping, sites))
    result.check()
    return result
