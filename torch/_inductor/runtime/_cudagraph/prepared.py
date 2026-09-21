"""Shared ownership of a prepared terminal replay and its cleanup."""

from dataclasses import dataclass


_FAILED_VARIANTS = []


@dataclass(frozen=True, eq=False)
class PreparedVariant:
    entry: object
    guard: object
    program: object

    def close(self):
        self.entry.close()
        self.program.close()
        _FAILED_VARIANTS[:] = [item for item in _FAILED_VARIANTS if item is not self]

    def abort(self, error):
        try:
            self.close()
        except BaseException as cleanup:
            if not any(item is self for item in _FAILED_VARIANTS):
                _FAILED_VARIANTS.append(self)
            error.add_note(f"Terminal variant cleanup retained live resources: {cleanup}")
