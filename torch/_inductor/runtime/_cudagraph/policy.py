"""Warm an ordinary compiled artifact, then publish its traced native replay."""

from dataclasses import replace
from threading import Lock
from weakref import ref

import sympy
import torch
from torch._logging import trace_structured
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import FXTraceDeclined
from torch._inductor.cudagraph_utils import CUDAGraphPolicy
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource
from torch._inductor.utils import ALIGNMENT, _InputAlignmentWrapper

from . import prepared
from .frontend import lower_terminal, trace_warmed_wrapper
from .guard_export import prepare_guard
from .metadata import check_terminal_attachment, MetadataDeclined, read_terminal_metadata
from .prepared import PreparedVariant
from .replay import prepare_terminal
from .trace_views import TraceViewDeclined


_FAILED_VARIANTS = prepared._FAILED_VARIANTS


def _static_inputs(artifact, contract):
    kwargs = artifact.fx_kwargs
    indices = kwargs.get("static_input_idxs") if type(kwargs) is dict else None
    if (type(indices) not in (list, tuple)
            or any(type(index) is not int or not 0 <= index < len(contract.kinds)
                   or contract.kinds[index] != "tensor" for index in indices)
            or len(set(indices)) != len(indices)):
        raise MetadataDeclined("Static input indices lost their compiler tensor slots")
    return tuple(indices)


class Installation:
    _boxed_call = True

    def __init__(self, artifact, original, metadata, prepare):
        self._artifact = ref(artifact)
        self._original = ref(original)
        ordinary = artifact.current_callable
        self._ordinary = ref(ordinary)
        self._ordinary_owner = ordinary if type(ordinary) is _InputAlignmentWrapper else None
        self._alignment_inputs = ordinary.inputs_to_check if self._ordinary_owner is not None else ()
        self.metadata = metadata
        self._attachment = original._cudagraph_terminal_attachment
        self._static_inputs = _static_inputs(artifact, metadata.inputs)
        self._prepare = prepare
        self._lock = Lock()
        self.entry = None
        self.variants = []
        self.status = "armed"
        self.decline = None

    @property
    def artifact(self):
        return self._artifact()

    @property
    def original(self):
        return self._original()

    @property
    def ordinary(self):
        return self._ordinary()

    def _check(self, expected):
        artifact, original = self.artifact, self.original
        if (artifact is None or original is None or artifact.current_callable is not expected
                or artifact._cudagraph_original_callable is not original or self.ordinary is None
                or self._ordinary_owner is not None and self._ordinary_owner.model is not original):
            raise RuntimeError("Terminal policy lost its original compiled artifact")
        if (check_terminal_attachment(original, self._attachment) is not self.metadata
                or _static_inputs(artifact, self.metadata.inputs) != self._static_inputs):
            raise RuntimeError("Terminal policy compiler input contract changed")
        return artifact, original

    def __call__(self, box):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Terminal policy installation is busy")
        try:
            if self.status != "armed" or type(box) is not list:
                raise RuntimeError("Terminal policy requires an armed boxed invocation")
            artifact, original = self._check(self)
            inputs = tuple(box)
            self.status = "warming"
            result = self.ordinary(box)
            self._check(self)
            self.status = "preparing"
            try:
                variant = self._prepare(original, inputs, static_inputs=self._static_inputs,
                                        alignment_inputs=self._alignment_inputs)
            except (FXTraceDeclined, TraceViewDeclined, UnsupportedCapture) as error:
                self._check(self)
                self.decline = str(error)
                artifact.current_callable = self.ordinary
                self._ordinary_owner = None
                self.status = "declined"
                return result
            try:
                self._check(self)
                entry = variant.entry
                if variant.guard is not None:
                    entry = torch._C._cuda_make_boxed_dispatch(((entry, variant.guard.registration),), self._miss)
            except BaseException as error:
                variant.abort(error)
                raise
            self.variants.append(variant)
            self.entry = entry
            artifact.current_callable = entry
            self.status = "ready"
            return result
        except BaseException:
            self.status = "failed"
            raise
        finally:
            self._lock.release()

    def _miss(self, box):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Terminal policy installation is busy")
        try:
            if self.status != "ready" or self.entry is None:
                raise RuntimeError("Terminal guard miss requires a ready installation")
            _, original = self._check(self.entry)
            inputs = tuple(box)
            result = self.ordinary(box)
            self._check(self.entry)
            try:
                variant = self._prepare(original, inputs, static_inputs=self._static_inputs,
                                        alignment_inputs=self._alignment_inputs)
            except (FXTraceDeclined, TraceViewDeclined, UnsupportedCapture) as error:
                self._check(self.entry)
                self.decline = str(error)
                return result
            try:
                self._check(self.entry)
                if variant.guard is None:
                    variant = replace(variant, guard=prepare_guard(variant.program, inputs, required=True))
                self.entry.append(variant.entry, variant.guard.registration)
            except BaseException as error:
                variant.abort(error)
                raise
            self.variants.append(variant)
            return result
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Terminal policy installation is busy")
        try:
            if self.status == "closed":
                return
            entry = self.entry
            if entry is not None:
                entry.close()
            for variant in self.variants:
                variant.close()
            artifact = self.artifact
            if artifact is not None and (artifact.current_callable is self
                                         or entry is not None and artifact.current_callable is entry):
                self._check(artifact.current_callable)
                artifact.current_callable = self.ordinary
            self._ordinary_owner = None
            self.entry = None
            self.status = "closed"
        finally:
            self._lock.release()


class NativeTerminalPolicy(CUDAGraphPolicy):
    trace_terminal = True

    def __init__(self):
        self._artifacts = []
        self._installations = []
        self._declines = []
        self._lock = Lock()
        self._closed = False

    def __deepcopy__(self, memo):
        # Config snapshots share this runtime owner and its live installations.
        return self

    @property
    def installations(self):
        return tuple(self._installations)

    @property
    def declines(self):
        return tuple((artifact(), reason) for artifact, reason in self._declines)

    @property
    def artifacts(self):
        result = []
        for artifact_ref, original_ref in self._artifacts:
            artifact, original = artifact_ref(), original_ref()
            if artifact is not None and original is not None:
                result.append((artifact, original))
        return tuple(result)

    def should_wrap(self, artifact):
        return False

    def wrap_output(self, artifact):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Terminal policy is busy")
        try:
            if self._closed:
                raise RuntimeError("Terminal policy is closed")
            if any(item.artifact is artifact for item in self._installations):
                return artifact
            original = getattr(artifact, "_cudagraph_original_callable", None)
            if callable(original):
                self._artifacts.append((ref(artifact), ref(original)))
            try:
                original, metadata = read_terminal_metadata(artifact)
                installation = Installation(artifact, original, metadata, self.prepare)
            except MetadataDeclined as error:
                reason = str(error)
                self._declines.append((ref(artifact), reason))
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {"name": "cudagraph_trace_decline", "encoding": "string"},
                    payload_fn=lambda: reason,
                )
                return artifact
            installation._check(installation.ordinary)
            self._installations.append(installation)
            artifact.current_callable = installation
            return artifact
        finally:
            self._lock.release()

    def prepare(self, wrapper, inputs, *, static_inputs=(), alignment_inputs=()):
        metadata = check_terminal_attachment(wrapper, wrapper._cudagraph_terminal_attachment)
        trace, views = trace_warmed_wrapper(wrapper, metadata.inputs, inputs)
        addresses = {binding.root.index: binding.symbol for binding in trace.address_bindings
                     if type(binding.root) is InputSource and binding.generation == 0}
        if not set(alignment_inputs).issubset(addresses):
            raise FXTraceDeclined("Alignment wrapper lost its original input addresses")
        guards = tuple(sympy.Eq(sympy.Mod(addresses[index], ALIGNMENT), 0) for index in alignment_inputs)
        program = lower_terminal(trace, views, extra_guards=guards)
        try:
            guard = prepare_guard(program, inputs)
            return PreparedVariant(prepare_terminal(program, inputs), guard, program)
        except BaseException as error:
            try:
                program.close()
            except BaseException as cleanup:
                error.add_note(f"Terminal preparation cleanup retained live resources: {cleanup}")
            raise

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Terminal policy is busy")
        try:
            if self._closed:
                return
            for installation in self._installations:
                installation.close()
            self._closed = True
        finally:
            self._lock.release()
