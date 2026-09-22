"""Eager as the oracle of the native host-trace replay.

One tape, prepared on the native replay (``direct_hosttrace.HostTraceReplay``
from that tape, no trace of its own, one variant: a call it does not serve is a
miss, never a re-trace), and eager itself as the reference: a served call runs
eager on fresh copies of the tensor arguments (the same sizes, strides, offsets,
dtype, device and pinning over another storage, so an in-place write lands beside
the replay's) and compares the two output for output and argument for argument,
bitwise through the bytes; a miss is the native entry's, named.

Two uses. A native suite's oracle test builds an ``Oracle`` and calls ``check``
(eager and native at the call's inputs, bitwise) or ``expect_miss``. The stack's
suites (test/test_cuda_host_trace*.py, whose replay is ``host_trace_testing.build``)
run unchanged under ``install()``, which makes ``build`` return an
``OracleVariant``: the suite's variant with eager run beside every served call;
a tape the native line refuses by name is served by the eager form alone, counted,
and a call made under a profiler (a test counting the replay's device work counts
one replay) runs the native side alone.

    python -m torch.testing._internal.host_trace_oracle test/test_cuda_host_trace_ti.py

runs a stack suite that way; ``HOST_TRACE_ORACLE_LOG=<file>`` appends the per-test
counts (builds, native builds, refusals, calls compared, misses) as JSON.

The replay runs under the caller's grad mode without reading it (E32, O48): a
served call's outputs carry no autograd history, so the comparisons say nothing
about grad mode; the runners call the entries under no_grad.
"""

import atexit
import inspect
import json
import os
import runpy
import sys
import threading
import unittest

import torch


def _ht():
    from torch.cuda import _host_trace

    return _host_trace


def _adapter():
    from torch._inductor.runtime._cudagraph import direct_hosttrace

    return direct_hosttrace


def _symm(t):
    try:
        from torch.distributed._symmetric_memory import is_symm_mem_tensor
    except Exception:
        return False
    try:
        return bool(is_symm_mem_tensor(t))
    except Exception:
        return False


def clone_like(t, storages=None):
    """A tensor with the same bytes, sizes, strides, storage offset, dtype, device and
    pinning over a fresh storage: eager's argument beside the replay's. Arguments
    over one storage share its copy (`storages`: the originals' storages seen so
    far), so the aliasing the tape's guards relate is the same. A symmetric-memory
    buffer (its handle is the storage's) and a math-bit tensor (the recorder
    declines it; the bit is what the call must see) are passed as they are."""
    if not isinstance(t, torch.Tensor) or _symm(t) or t.is_neg() or t.is_conj():
        return t
    storage = t.untyped_storage()
    key = storage._cdata  # the StorageImpl: data_ptr() would materialize a lazy clone
    raw = None if storages is None else storages.get(key)
    if raw is None:
        nbytes = storage.nbytes()
        if t.is_cuda:
            raw = torch.empty(nbytes, dtype=torch.uint8, device=t.device)
        else:
            raw = torch.empty(nbytes, dtype=torch.uint8, pin_memory=t.is_pinned())
        raw.untyped_storage().copy_(storage)
        if storages is not None:
            storages[key] = raw
    return torch.empty((0,), dtype=t.dtype, device=t.device).set_(
        raw.untyped_storage(), t.storage_offset(), t.shape, t.stride()
    )


def _clone_args(args):
    storages: dict = {}
    return tuple(clone_like(a, storages) for a in args)


_BITS = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _bits(t):
    # NaN payloads and signed zeros compare as the bytes they are
    if t.dtype in _BITS.values() or t.dtype == torch.bool:
        return t
    if t.is_complex():
        t = torch.view_as_real(t)
    return t.contiguous().view(_BITS[t.element_size()])


def _same(a, b, what, values=True):
    if isinstance(a, torch.Tensor) != isinstance(b, torch.Tensor):
        raise AssertionError(
            f"oracle: {what}: {type(a).__name__} vs {type(b).__name__}"
        )
    if not isinstance(a, torch.Tensor):
        if a != b:
            raise AssertionError(f"oracle: {what}: {a!r} vs {b!r}")
        return
    facts = [(a.dtype, b.dtype), (a.shape, b.shape), (a.stride(), b.stride())]
    facts.append((a.device, b.device))
    for x, y in facts:
        if x != y:
            raise AssertionError(f"oracle: {what}: {x} vs {y}")
    if values and a.numel() and not torch.equal(_bits(a), _bits(b)):
        diff = (
            (a.float() - b.float()).abs().max().item()
            if a.is_floating_point() or a.is_complex()
            else "int"
        )
        raise AssertionError(f"oracle: {what}: values differ (max abs {diff})")


def _as_list(out):
    return [out] if isinstance(out, torch.Tensor) else list(out)


def compare(native_out, args, eager_out, clones, unwritten=()):
    """The native replay's outputs at `args` against eager's at `clones` (copies of
    `args`), output for output, and every argument the replay may have written
    against eager's copy of it. An output no node writes (eager's at::empty
    returned as it is, `unwritten`) is compared by its metadata: its values are
    indeterminate on every path."""
    if len(native_out) != len(eager_out):
        raise AssertionError(
            f"oracle: {len(native_out)} native outputs, {len(eager_out)} eager"
        )
    for k, (a, b) in enumerate(zip(native_out, eager_out)):
        _same(a, b, f"output {k} against eager", values=k not in unwritten)
    for k, (a, c) in enumerate(zip(args, clones)):
        if c is not a:
            _same(a, c, f"argument {k} after the call")


_STATS: dict = {}
_STATS_LOCK = threading.Lock()


def _test_id():
    for frame in inspect.stack():
        self = frame.frame.f_locals.get("self")
        if isinstance(self, unittest.TestCase):
            return f"{type(self).__name__}.{self._testMethodName}"
    return "<no test>"


def _note(key, value=1):
    with _STATS_LOCK:
        row = _STATS.setdefault(
            _test_id(),
            {
                "builds": 0,
                "native": 0,
                "refused": [],
                "compared": 0,
                "fallback": 0,
                "missed": 0,
                "rerun": 0,
                "other_stream": 0,
                "profiled": 0,
            },
        )
        if key == "refused":
            row["refused"].append(value)
        else:
            row[key] += value


def _testing():
    """test/host_trace_testing.py, the stack's test module (the eager form of a
    tape a native line refused): beside the suite on sys.path, else from the
    repo's test directory."""
    try:
        import host_trace_testing
    except ImportError:
        test_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__))), "test"
        )
        if test_dir not in sys.path:
            sys.path.append(test_dir)
        import host_trace_testing
    return host_trace_testing


def _native_entry(fn, tape, args, device, staging_depth):
    """The native entry prepared from `tape` at `args` (no trace, nothing runs), or
    (None, why) when the native line refuses the tape by name."""
    ht = _ht()
    adapter = _adapter()
    from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture

    if any(f.filename.endswith("host_trace_two_hint.py") for f in inspect.stack()):
        # the two-hint property check (E23) re-runs every case of a suite under other
        # hints to compare the tapes: the native side ran in the case's own run
        _note("rerun")
        return None, None
    _note("builds")
    try:
        native = adapter.HostTraceReplay(
            fn,
            args,
            tape=tape,
            max_variants=1,
            staging_depth=staging_depth,
            device=device,
        )
    except (UnsupportedCapture, ht.Miss, ValueError) as e:
        # UnsupportedCapture: the lowering or the runtime declines by name; Miss: a
        # region template the harvest refused; ValueError: the runtime's registry
        # refusing what it cannot hold (a region whose call launches nothing)
        _note("refused", f"{type(e).__name__}: {e}")
        return None, str(e)
    _note("native")
    return native, None


def _native_call(native, args):
    """The native entry's outputs as a list, or its miss as `ht.Miss` (the cap error
    of a one-variant entry read as a miss, with the entry's reason); another
    stream's refusal (O29) propagates as the RuntimeError it is."""
    ht = _ht()
    try:
        out = native(*args)
    except RuntimeError as e:
        if "misses all 1 variants" not in str(e):
            raise
        why = native.miss_log[-1][1] if native.miss_log else str(e)
        raise ht.Miss(why) from None
    return _as_list(out)


class OracleVariant:
    """A stack suite's variant (`host_trace_testing.build`'s: the native replay, or
    the eager form where the native line refused the tape) with eager run beside
    every served call and compared to it."""

    def __init__(self, variant):
        self.variant = variant

    _own = frozenset(("variant",))

    def __getattr__(self, name):
        return getattr(self.variant, name)

    def __setattr__(self, name, value):
        if name in self._own:
            object.__setattr__(self, name, value)
        else:
            setattr(self.variant, name, value)

    def matches(self, args):
        return self.variant.matches(args)

    def __call__(self, args):
        return self.replay(args)

    def replay(self, args):
        return self._both(tuple(args), miss_ok=False)

    def try_replay(self, args):
        return self._both(tuple(args), miss_ok=True)

    def _both(self, args, miss_ok):
        ht = _ht()
        variant = self.variant
        if variant.native is None or torch.autograd._profiler_enabled():
            # the eager form (nothing to compare it with) or a profiled call (a test
            # that counts the replay's device work counts one replay, not eager's
            # beside it): the variant alone
            if variant.native is not None:
                _note("profiled")
            return variant.try_replay(args) if miss_ok else variant.replay(args)
        clones = _clone_args(args)
        device = variant.device
        state = torch.cuda.get_rng_state(device)
        try:
            out = variant.replay(args)
        except ht.Miss:
            # the variant's miss (the tape's guards, checked against the native
            # predicate by host_trace_testing.NativeReplay): eager is not run
            _note("missed")
            if miss_ok:
                return None
            raise
        # eager draws from the state the served call started at; the served
        # call's own advancement is what the caller sees afterwards, as with
        # any replay (the runtime team's review of the first cut, cascade 18)
        served = torch.cuda.get_rng_state(device)
        torch.cuda.set_rng_state(state, device)
        eager = _as_list(variant.fn(*clones))
        torch.cuda.set_rng_state(served, device)
        compare(out, args, eager, clones, variant.lowered.unwritten_outputs)
        _note("compared")
        return out


_ORIGINAL_BUILD = None
_TESTING = None


def install():
    """Make `host_trace_testing.build` return an OracleVariant (idempotent)."""
    global _ORIGINAL_BUILD, _TESTING
    if _ORIGINAL_BUILD is not None:
        return
    import host_trace_testing as testing

    _TESTING = testing
    _ORIGINAL_BUILD = original = testing.build

    def build(tape, fn, args, device=None, *, staging_depth=2, backend=None):
        variant = original(
            tape, fn, args, device, staging_depth=staging_depth, backend=backend
        )
        _note("builds")
        if variant.native is not None:
            _note("native")
        elif variant.refused is not None:
            _note("refused", variant.refused)
        return OracleVariant(variant)

    testing.build = build


def uninstall():
    global _ORIGINAL_BUILD
    if _ORIGINAL_BUILD is not None:
        _TESTING.build = _ORIGINAL_BUILD
        _ORIGINAL_BUILD = None


class Oracle:
    """One tape, the native replay prepared from it and eager as its reference, for
    a native suite's oracle test."""

    def __init__(
        self, fn, args, *, tape=None, device=None, staging_depth=2, warm_up=True
    ):
        ht = _ht()
        self.fn = fn
        args = tuple(args)
        self.tape = tape if tape is not None else ht.trace(fn, args, warm_up=warm_up)
        self.device = device if device is not None else self.tape.device.index
        self.native, self.refused = _native_entry(
            fn, self.tape, args, device, staging_depth
        )

    def close(self):
        if self.native is not None:
            self.native.close()

    def check(self, args, reference=None, *, miss_ok=False):
        """Eager (or `reference`) and the native entry at `args`, bitwise, output
        for output and argument for argument; returns the native outputs, or
        None when the entry misses and `miss_ok`. Eager runs first on its own
        copies from the device's RNG state at the call, which is restored for
        the native entry; the entry runs once and its RNG advancement stays, as
        any replay's does. A tape the native line refused by name is checked on
        the eager form of the stack's suites (host_trace_testing.build: the
        tape's own predicate, eager as the executor), so the class it serves at
        other shapes is exercised and its result is eager's."""
        ht = _ht()
        args = tuple(args)
        if self.native is None:
            if self.refused is None:
                raise AssertionError(
                    "oracle: no native entry and no refusal; nothing to check"
                )
            variant = _testing().build(
                self.tape, self.fn, args, self.device, backend="eager"
            )
            unwritten = ()
        else:
            unwritten = self.native.lowered.unwritten_outputs
        clones = _clone_args(args)
        state = torch.cuda.get_rng_state(self.device)
        if reference is None:
            reference = self.fn(*clones)
            torch.cuda.set_rng_state(state, self.device)
        reference = _as_list(reference)
        try:
            if self.native is None:
                out = variant.replay(args)
            else:
                out = _native_call(self.native, args)
        except ht.Miss:
            _note("missed")
            if miss_ok:
                return None
            raise
        compare(out, args, reference, clones, unwritten)
        _note("compared" if self.native is not None else "fallback")
        return out

    def try_check(self, args):
        """`check`, or None when the native entry misses at `args` (the entry runs
        once either way)."""
        return self.check(args, miss_ok=True)

    def expect_miss(self, args):
        """The native entry misses at `args` (its predicate, named in the miss log)."""
        ht = _ht()
        if self.native is None:
            raise AssertionError(
                f"oracle: the native line refused this tape ({self.refused}); nothing to check"
            )
        try:
            _native_call(self.native, tuple(args))
        except ht.Miss:
            _note("missed")
            return
        raise AssertionError("oracle: the call was served")


def _report():
    if not _STATS:
        return
    compared = sum(1 for row in _STATS.values() if row["compared"])
    refused = sum(1 for row in _STATS.values() if row["refused"] and not row["native"])
    fallback = sum(1 for row in _STATS.values() if row["fallback"])
    line = f"host_trace_oracle: {len(_STATS)} tests built variants, {compared} compared at least one served call with eager, {refused} on the eager form only (refused by name), {fallback} checked a refused tape's eager form"
    print(line, file=sys.stderr, flush=True)
    path = os.environ.get("HOST_TRACE_ORACLE_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"argv": sys.argv, "tests": _STATS}) + "\n")


atexit.register(_report)


if __name__ == "__main__":
    path = os.path.abspath(sys.argv[1])
    sys.argv = [path, *sys.argv[2:]]
    sys.path.insert(0, os.path.dirname(path))  # the suite's own helpers beside it
    install()
    runpy.run_path(path, run_name="__main__")
