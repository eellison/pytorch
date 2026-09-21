import inspect

from ..entry_signature import EntrySignature


class EntrySourceGuard:
    __slots__ = ("signature", "_state", "_parameters", "_return_annotation", "_seal")

    def __new__(cls):
        raise TypeError("EntrySourceGuard must be prepared from an entry signature")

    def __setattr__(self, name, value):
        raise AttributeError("Prepared entry source guards are immutable")

    def check(self, signature: EntrySignature) -> None:
        owned = self, self.signature, self._state, self._parameters, self._return_annotation
        if (type(self) is not EntrySourceGuard or type(signature) is not EntrySignature
                or signature is not self.signature or type(self._seal) is not tuple
                or len(self._seal) != len(owned)
                or any(actual is not old for actual, old in zip(owned, self._seal))):
            raise RuntimeError("Prepared entry source guard changed ownership")
        signature.trace.check()
        if (not any(call is signature.call for call in signature.trace.calls)
                or signature.call.target is not signature.target
                or signature.call.entry.target is not signature.target
                or signature._state() != self._state):
            raise RuntimeError("Prepared entry source association changed")
        current = signature.signature
        if (type(current) is not inspect.Signature
                or current.return_annotation is not self._return_annotation
                or len(current.parameters) != len(self._parameters)):
            raise RuntimeError("Prepared Python signature changed")
        for (name, parameter), old in zip(current.parameters.items(), self._parameters):
            if (name != old[0] or parameter is not old[1] or parameter.name != old[2]
                    or parameter.kind is not old[3] or parameter.default is not old[4]
                    or parameter.annotation is not old[5]):
                raise RuntimeError("Prepared Python parameter binding changed")


def prepare_entry_source_guard(signature: EntrySignature) -> EntrySourceGuard:
    if type(signature) is not EntrySignature or type(signature.signature) is not inspect.Signature:
        raise TypeError("Expected the original entry and Python signature")
    signature.check()
    parameters = tuple((name, parameter, parameter.name, parameter.kind, parameter.default, parameter.annotation)
                       for name, parameter in signature.signature.parameters.items())
    if any(type(item[1]) is not inspect.Parameter for item in parameters):
        raise TypeError("Expected original Python signature parameters")
    state = signature._state()
    result = object.__new__(EntrySourceGuard)
    object.__setattr__(result, "signature", signature)
    object.__setattr__(result, "_state", state)
    object.__setattr__(result, "_parameters", parameters)
    object.__setattr__(result, "_return_annotation", signature.signature.return_annotation)
    object.__setattr__(result, "_seal", (result, signature, state, parameters, signature.signature.return_annotation))
    result.check(signature)
    return result
