"""The interim Python replay as a test oracle for the native host-trace line.

One tape, two backends prepared from it: the interim ``Variant``
(``torch.cuda._host_trace.build``: the ordinary host captured and checked against the
tape byte for byte) and the native entry (``direct_hosttrace.HostTraceReplay`` prepared
from that same tape, no trace of its own, one variant: a call it does not serve is a
miss, never a re-trace). A served call runs both, the native side on fresh copies of
the tensor arguments (the same sizes, strides, offsets, dtype, device and pinning over
another storage, so an in-place write lands beside the interim's), and compares them
output for output and argument for argument; a miss must be a miss on both sides.

Two uses. A native suite's oracle test builds an ``Oracle`` and calls ``check`` (eager,
native and interim at the call's inputs, bitwise) or ``expect_miss``. The interim suites
run unchanged under ``install()``, which makes ``_host_trace.build`` return an
``OracleVariant``: the interim ``Variant`` for every attribute the tests read, with a
native entry checked on every replay; interim-only, counted, when the native line
refuses the tape by name (a cell the native path does not serve yet), and for a call
made under a profiler (a test counting the replay's device work counts one replay).

    python -m torch.testing._internal.host_trace_oracle test/test_cuda_host_trace_ti.py

runs an interim suite that way; ``HOST_TRACE_ORACLE_LOG=<file>`` appends the per-test
counts (builds, native builds, refusals, calls served by both, misses on both) as JSON.

Both backends replay under the caller's grad mode without reading it (E32, O48): a
served call's outputs carry no autograd history on either side, so the oracle's
comparisons say nothing about grad mode; the runners call the entries under no_grad.
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
    pinning over a fresh storage: the native side's argument. Arguments over one
    storage share its copy (`storages`: the originals' data pointers seen so far), so
    the aliasing the tape's guards relate is the same. A symmetric-memory buffer (its
    handle is the storage's) and a math-bit tensor (the recorder declines it; the bit
    is what the call must see) are passed as they are."""
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
                "both": 0,
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


def _native_entry(fn, tape, args, device, staging_depth):
    """The native entry prepared from `tape` at copies of `args`, or (None, why) when
    the native line refuses the tape by name."""
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
            _clone_args(args),
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


class NativeMiss(Exception):
    pass


class OtherStream(Exception):
    pass


class OracleVariant:
    """The interim Variant with a native entry checked beside it on every replay."""

    def __init__(self, interim, native, refused):
        self.interim = interim
        self.native = native
        self.refused = refused

    _own = frozenset(("interim", "native", "refused"))

    def __getattr__(self, name):
        return getattr(self.interim, name)

    def __setattr__(self, name, value):
        # a test that swaps an internal of the interim Variant (its exec) writes
        # through to it
        if name in self._own:
            object.__setattr__(self, name, value)
        else:
            setattr(self.interim, name, value)

    def close(self):
        if self.native is not None:
            self.native.close()

    def matches(self, args):
        return self.interim.matches(args)

    def __call__(self, args):
        return self.replay(args)

    def replay(self, args):
        return self._both(tuple(args), miss_ok=False)

    def try_replay(self, args):
        return self._both(tuple(args), miss_ok=True)

    def wait_for_h2d(self):
        self.interim.wait_for_h2d()
        if self.native is not None:
            self.native.wait_for_h2d()

    def _native_call(self, clones):
        native = self.native
        try:
            out = native(*clones)
        except RuntimeError as e:
            if "bound device and stream" in str(e):
                # the call is on another stream than the entry's (O29): the
                # native line refuses it by contract, the interim serves it
                raise OtherStream(str(e)) from None
            if "misses all 1 variants" not in str(e):
                raise
            why = native.miss_log[-1][1] if native.miss_log else str(e)
            raise NativeMiss(why) from None
        return [out] if isinstance(out, torch.Tensor) else list(out)

    def _both(self, args, miss_ok):
        ht = _ht()
        interim = self.interim
        if self.native is None:
            return interim.try_replay(args) if miss_ok else interim.replay(args)
        if torch.autograd._profiler_enabled():
            # a profiled call is the interim's alone: a test that counts the replay's
            # device work counts one replay, not the native call's plus its argument
            # clones (its un-profiled calls compared both backends)
            _note("profiled")
            return interim.try_replay(args) if miss_ok else interim.replay(args)
        clones = _clone_args(args)
        device = interim.device
        state = torch.cuda.get_rng_state(device)
        native_out, native_miss = None, None
        try:
            native_out = self._native_call(clones)
        except NativeMiss as e:
            native_miss = str(e)
        except OtherStream:
            torch.cuda.set_rng_state(state, device)
            _note("other_stream")
            return interim.try_replay(args) if miss_ok else interim.replay(args)
        torch.cuda.set_rng_state(state, device)
        try:
            out = interim.replay(args)
        except (ht.Miss, ht.TopologyMiss) as e:
            # a TopologyMiss (E28): the call's shapes select another node chain than
            # the variant's exec holds; an entry builds the same tape at these inputs,
            # the one-variant native side reports the cap (a miss here)
            if native_miss is None:
                raise AssertionError(
                    f"oracle: the interim replay missed ({e}) where the native entry served"
                ) from e
            _note("missed")
            if miss_ok:
                return None
            raise
        if native_miss is not None:
            raise AssertionError(
                f"oracle: the native entry missed ({native_miss}) where the interim replay served"
            )
        if len(out) != len(native_out):
            raise AssertionError(
                f"oracle: {len(out)} interim outputs, {len(native_out)} native"
            )
        unwritten = self.native.lowered.unwritten_outputs
        for k, (a, b) in enumerate(zip(out, native_out)):
            # an output no node writes (eager's at::empty returned as it is) is
            # compared by its metadata: its values are indeterminate on every path
            _same(a, b, f"output {k}", values=k not in unwritten)
        for k, (a, c) in enumerate(zip(args, clones)):
            if c is not a:
                _same(a, c, f"argument {k} after the call")
        _note("both")
        return out


_ORIGINAL_BUILD = None


def install():
    """Make `_host_trace.build` return an OracleVariant (idempotent)."""
    global _ORIGINAL_BUILD
    ht = _ht()
    if _ORIGINAL_BUILD is not None:
        return
    _ORIGINAL_BUILD = original = ht.build

    def build(tape, fn, args, device=None, *, warm_up=True, staging_depth=2):
        interim = original(
            tape, fn, args, device, warm_up=warm_up, staging_depth=staging_depth
        )
        native, refused = _native_entry(fn, tape, tuple(args), device, staging_depth)
        return OracleVariant(interim, native, refused)

    ht.build = build


def uninstall():
    global _ORIGINAL_BUILD
    if _ORIGINAL_BUILD is not None:
        _ht().build = _ORIGINAL_BUILD
        _ORIGINAL_BUILD = None


class Oracle:
    """One tape, both backends, for a native suite's oracle test."""

    def __init__(
        self, fn, args, *, tape=None, device=None, staging_depth=2, warm_up=True
    ):
        ht = _ht()
        self.fn = fn
        args = tuple(args)
        self.tape = tape if tape is not None else ht.trace(fn, args, warm_up=warm_up)
        interim = (_ORIGINAL_BUILD or ht.build)(
            self.tape, fn, args, device, warm_up=False, staging_depth=staging_depth
        )
        native, refused = _native_entry(fn, self.tape, args, device, staging_depth)
        self.variant = OracleVariant(interim, native, refused)

    @property
    def native(self):
        return self.variant.native

    @property
    def interim(self):
        return self.variant.interim

    @property
    def refused(self):
        return self.variant.refused

    def close(self):
        self.variant.close()

    def check(self, args, reference=None):
        """Eager (or `reference`), the native entry and the interim replay at `args`,
        bitwise; returns the interim's outputs. Eager runs on its own copies."""
        args = tuple(args)
        device = self.interim.device
        if reference is None:
            state = torch.cuda.get_rng_state(device)
            reference = self.fn(*_clone_args(args))
            torch.cuda.set_rng_state(state, device)
        reference = (
            [reference] if isinstance(reference, torch.Tensor) else list(reference)
        )
        out = self.variant.replay(args)
        if len(out) != len(reference):
            raise AssertionError(f"oracle: {len(out)} outputs, eager {len(reference)}")
        native = self.native
        unwritten = () if native is None else native.lowered.unwritten_outputs
        for k, (a, b) in enumerate(zip(out, reference)):
            _same(a, b, f"output {k} against eager", values=k not in unwritten)
        return out

    def expect_miss(self, args):
        """Both backends miss at `args` (None from try_replay)."""
        if self.variant.try_replay(tuple(args)) is not None:
            raise AssertionError("oracle: the call was served")


def _report():
    if not _STATS:
        return
    oracle = sum(1 for row in _STATS.values() if row["both"])
    refused = sum(1 for row in _STATS.values() if row["refused"] and not row["native"])
    line = f"host_trace_oracle: {len(_STATS)} tests built variants, {oracle} compared at least one served call on both backends, {refused} interim-only (refused by name)"
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
