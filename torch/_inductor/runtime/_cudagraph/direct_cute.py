"""Direct ordinary CuTe calls with the existing compiler and lifetime owner."""

from .direct_invocation import ACTIVE


class _NativeBorrow:
    def __init__(self, entry, owner):
        self._seal = entry, owner
        self._token = None

    @property
    def entry(self):
        return None if self._seal is None else self._seal[0]

    @property
    def owner(self):
        return None if self._seal is None else self._seal[1]

    def check(self):
        from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry

        if self._token is None:
            raise RuntimeError("Direct CuTe native borrow is closed")
        DirectCuTe.check(self.entry)
        if self.entry.owner is not self.owner:
            raise RuntimeError("Direct CuTe native borrow lost its ordinary owner")
        OrdinaryEntry._check_native_borrow(self.owner, self._token)

    def close(self):
        from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry

        if self._token is not None:
            OrdinaryEntry._release_native_borrow(self.owner, self._token)
            self._token = None
            self._seal = None

    def __reduce_ex__(self, protocol):
        raise TypeError("Direct CuTe native borrows cannot be serialized")


class DirectCuTe:
    def __init__(self, owner):
        from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry
        from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import ObservedOrdinaryEntry

        if type(owner) is not ObservedOrdinaryEntry:
            raise TypeError("Direct CuTe requires the actual observed ordinary owner")
        OrdinaryEntry.check(owner)
        names = owner.argument_names
        arguments = ", ".join(names)
        namespace = {"ENTRY": owner.entry}
        exec(f"def invoke({arguments}):\n    ENTRY({arguments})\n", namespace)
        self.owner = owner
        self._host = namespace["invoke"]
        self._run = OrdinaryEntry._invoke_owned.__get__(owner, type(owner))
        self._seal = (owner, owner.entry, self._host, self._host.__code__,
                      self._run, OrdinaryEntry._invoke_owned.__code__)

    def check(self):
        from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry

        owner, entry, host, code, run, run_code = self._seal
        if (self.owner is not owner or owner.entry is not entry or self._host is not host
                or host.__code__ is not code or host.__globals__.get("ENTRY") is not entry
                or self._run is not run or run.__self__ is not owner
                or run.__func__ is not OrdinaryEntry._invoke_owned or run.__func__.__code__ is not run_code):
            raise RuntimeError("Direct CuTe invocation changed its ordinary callable")
        OrdinaryEntry.check(owner)

    def invoke(self, *arguments):
        active = ACTIVE.get()
        if active is not None:
            return active.cute(self, arguments)
        return self._invoke_ordinary(*arguments)

    def _invoke_ordinary(self, *arguments):
        self.check()
        self._run(self._host, *arguments)

    __call__ = invoke

    def borrow_native(self):
        from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry

        self.check()
        if self.owner._owned_executor is None:
            raise RuntimeError("Direct CuTe native replay requires an ordinary warm call")
        borrow = _NativeBorrow(self, self.owner)
        borrow._token = OrdinaryEntry._acquire_native_borrow(self.owner)
        return borrow

    def __reduce_ex__(self, protocol):
        raise TypeError("Live direct CuTe calls cannot be serialized")
