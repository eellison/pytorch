"""CuTe launch records of a tape (torch/cuda/_host_trace_cute.py), lowered into
the shared physical calls through the runtime's own CuTe binder.

A record's invocation was bound at the trace (the entry signature, the ordinary
binding, the artifact, the selected sites); here the runtime's invocation
lowering (cute_adapter.lower_cute_invocation) runs once per invocation over the
tape's symbol correspondence: the operands' pointers are the tape's roots plus
their displacements, integers are the lowering's numeric plan over the tape's
size, stride and offset loads, and every obligation the binder emits (the
dispatch predicate, the shared-memory and TMA obligations, the integer ranges
and divisibilities) comes back as a sympy relation over the tape's symbols and
joins the tape's predicate. Each selected site's CuTeCall is one physical call
of the tape, its module the runtime's CuTeKernelOwner behind a thin wrapper
that holds the graph borrow the owner requires at capture and releases the
receipt when the replay closes.
"""

from __future__ import annotations

import dataclasses

import sympy

from torch._inductor.runtime.cudagraph_boxed_replay import _KernelModule


class _TapeOrigin:
    """The tape's symbol correspondence for cute_adapter.lower_cute_invocation:
    the tape's ShapeEnv and traced values, its size / stride / offset symbols
    keyed as the numeric plan loads them (the IntExpr leaves the lowering
    emits), no address scalars (an integer operand over an address declined at
    the trace), and pointer alignment as guarded at the trace."""

    def __init__(self, tape, symbols, guarded):
        self.shape_env = tape.shape_env
        self.hints = dict(tape.shape_env.backed_var_to_val)
        self.storage_offset_indices = tuple(range(len(tape.inputs)))
        self.symbols = {}
        self.metadata_symbols = set()
        for symbol, prop in symbols.by_symbol.items():
            if prop.kind in ("size", "stride"):
                self.symbols[prop.kind, prop.index, prop.dim] = symbol
            elif prop.kind == "offset":
                self.symbols["storage_offset", prop.index] = symbol
                self.metadata_symbols.add(symbol)
        self.address_symbols = set()
        self._guarded = guarded  # id(PointerSource) -> alignment guarded at the trace

    def symbolic(self, value):
        from torch._inductor.runtime._cudagraph.address_scalars import symbolic_integer
        from torch._inductor.runtime._cudagraph.cute_adapter import CuTeDeclined

        return symbolic_integer(value, self.symbols, CuTeDeclined)

    def late_scalar(self, value, abi_type):
        return None

    def alignment_guard(self, pointer, alignment, root_alignments):
        from torch._inductor.runtime._cudagraph.cute_adapter import CuTeDeclined

        guarded = self._guarded.get(id(pointer))
        if guarded is None or alignment > guarded:
            raise CuTeDeclined(
                f"CuTe pointer alignment requirement {alignment} exceeds the {guarded} the trace guarded"
            )
        return sympy.true


class _HostTraceCuTeModule(_KernelModule):
    """A CuTe launch site of the tape: the runtime's CuTeKernelOwner (its loaded
    kernel, parameter layout, block, shared bytes and launch) behind the tape
    preparation's module interface. The owner launches only under a held graph
    borrow, so the wrapper takes one at the first launch and hands it to the
    replay when it borrows (cudagraph_boxed_replay._make_replay)."""

    def __init__(self, receipt, name, device):
        self.receipt = receipt
        self.owner = receipt.owner
        self.name = name
        self.device_index = device
        self.host_symbol = None
        self._borrow = None

    @property
    def function(self):
        return self.owner.function

    @property
    def parameter_layout(self):
        return self.owner.parameter_layout

    @property
    def parameter_sizes(self):
        return self.owner.parameter_sizes

    @property
    def shared(self):
        return self.owner.shared

    @property
    def block(self):
        return self.owner.block

    def check(self):
        self.receipt.check()

    def _borrow_for_cudagraph(self):
        borrow, self._borrow = self._borrow, None
        if borrow is None:
            borrow = self.owner._borrow_for_cudagraph()
        return borrow

    def launch(self, images, grid, stream, shared=None, block=None):
        if block is not None and tuple(block) != tuple(self.owner.block):
            raise RuntimeError(
                f"host_trace preparation: CuTe kernel {self.name} launched with block {tuple(block)}; "
                f"the compiled site's is {tuple(self.owner.block)}"
            )
        if self._borrow is None:
            self._borrow = self.owner._borrow_for_cudagraph()
        self.owner.launch(tuple(images), tuple(grid), stream, shared=shared)

    def release(self):
        """Close the receipt (its owners and the ordinary borrow) once no graph
        holds the kernel; a receipt still borrowed by a live graph stays open,
        its loader retaining the code until the borrow is dropped."""
        self._borrow = None
        try:
            self.receipt.close()
        except RuntimeError:
            return False
        return True


class CuTeLowering:
    """One lower_tape's CuTe records: per invocation the runtime's invocation
    lowering runs once (its sites' calls in order); each record takes its site's
    call, and the binder's obligations join the tape's predicate."""

    def __init__(self, tape, lowering, symbols, device):
        import torch
        from torch._inductor.runtime._cudagraph.direct_hosttrace import (
            _int_or_expr,
            _pointer_source,
            HostTraceLoweringDeclined,
        )

        self.tape, self.lowering, self.symbols, self.device = (
            tape,
            lowering,
            symbols,
            device,
        )
        self.stream = torch.cuda.current_stream(device).cuda_stream
        self.root_alignments = dict(symbols.mapping.root_alignments)
        self._declined = HostTraceLoweringDeclined
        self._pointer_source, self._int_or_expr = _pointer_source, _int_or_expr
        self._calls = {}  # id(invocation) -> its sites' CuTeCalls

    def lower(self, launch):
        from torch._inductor.runtime._cudagraph.cute_adapter import CuTeDeclined
        from torch._inductor.runtime._cudagraph.direct_hosttrace import _PhysicalCall

        invocation = launch["cute"].invocation
        calls = self._calls.get(id(invocation))
        if calls is None:
            calls = self._calls[id(invocation)] = self._lower_invocation(
                invocation, launch["kernel"]
            )
        try:
            call = calls[launch["cute"].site_index]
            for guard in call.guards:
                self.lowering.require(guard)
        except CuTeDeclined as error:
            raise self._declined(
                f"host_trace lowering: CuTe kernel {launch['kernel']}: {error}"
            ) from error
        module = _HostTraceCuTeModule(call.receipt, launch["kernel"], self.device)
        bound = call.bound
        if type(bound) is not _PhysicalCall:
            raise self._declined(
                f"host_trace lowering: CuTe kernel {launch['kernel']} lost its physical compiler binding"
            )
        return dataclasses.replace(bound, module=module, storage_sources=call.pointers)

    def _lower_invocation(self, invocation, name):
        import torch
        from torch._inductor.runtime._cudagraph.cute_adapter import (
            _InvocationResources,
            CuTeDeclined,
            lower_cute_invocation,
        )
        from torch._subclasses.fake_tensor import FakeTensor

        lowering, symbols = self.lowering, self.symbols
        if invocation.owner._device != torch.device("cuda", self.device):
            raise self._declined(
                f"host_trace lowering: CuTe kernel {name} was compiled for {invocation.owner._device}, "
                f"the tape is lowered for cuda:{self.device}"
            )
        # the operands' pointers over the tape's roots (a converted view's offset
        # included), and the alignment each was guarded to at the trace
        tensors, guarded = {}, {}
        for twin_id, (root, offset, dtype) in invocation.origins.items():
            source, _ = self._pointer_source(
                lowering, symbols, root.sym + offset * dtype.itemsize
            )
            tensors[twin_id] = source
            alignment = invocation.alignments.get(twin_id)
            if alignment is not None:
                guarded[id(source)] = alignment
        sources = tuple(
            tensors[id(value)]
            for value in invocation.operands
            if type(value) is FakeTensor
        )
        origin = _TapeOrigin(self.tape, symbols, guarded)
        local = invocation.local
        # the entry's ordinary borrow lives with the invocation (its receipts name
        # it); a tape lowered again after its replay closed takes a fresh one
        if local.borrow is None or local.borrow._token is None:
            try:
                local.borrow = invocation.adapter.borrow_native()
            except RuntimeError as error:
                raise self._declined(
                    f"host_trace lowering: CuTe kernel {name}: {error}"
                ) from error
        resources = _InvocationResources(local.borrow)
        try:
            local.check()
            invocation.artifact.check()
            return lower_cute_invocation(
                origin,
                local,
                invocation.artifact,
                invocation.binding,
                tensors,
                sources,
                lambda value: self._int_or_expr(lowering, value),
                stream=self.stream,
                resources=resources,
                root_alignments=self.root_alignments,
                sites=invocation.sites,
                owner_provider=invocation.owner_provider,
            )
        except BaseException as error:
            try:
                resources.close()
            except BaseException as cleanup:
                error.add_note(
                    f"CuTe preparation resources retained after cleanup failed: {cleanup}"
                )
            if isinstance(
                error, (CuTeDeclined, ValueError, TypeError, RuntimeError)
            ) and not isinstance(error, self._declined):
                raise self._declined(
                    f"host_trace lowering: CuTe kernel {name}: {type(error).__name__}: {error}"
                ) from error
            raise
