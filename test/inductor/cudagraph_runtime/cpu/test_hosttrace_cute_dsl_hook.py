# Owner(s): ["module: inductor"]
import unittest
from inspect import Parameter, Signature, signature
from types import SimpleNamespace
from unittest import mock

import torch
from torch.cuda import _host_trace_cute_dsl as hook
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


try:
    from cutlass.base_dsl.common import DSLUserCodeError
    from cutlass.base_dsl.jit_executor import ExecutionArgs
except ImportError:
    ExecutionArgs = None


class _Compiled:
    pass


@unittest.skipIf(ExecutionArgs is None, "CuTe DSL is not installed")
@instantiate_parametrized_tests
class TestCuTeDslHookContract(TestCase):
    def synth(self, signature, adapter=lambda *args: None, kind="integer"):
        return SimpleNamespace(
            signature=signature,
            formals=tuple(
                hook._Formal(name, kind, True) for name in signature.parameters
            ),
            adapter=adapter,
            name="fixture",
        )

    @parametrize("kind", ("duplicate", "unexpected", "keyword_only"))
    def test_invalid_call_matches_sdk_binder(self, kind):
        signature = Signature((Parameter("value", Parameter.POSITIONAL_OR_KEYWORD),))
        args, kwargs = (3,), {"value": 4} if kind == "duplicate" else {"unknown": 4}
        if kind == "keyword_only":
            signature = Signature((Parameter("value", Parameter.KEYWORD_ONLY),))
            args, kwargs = (3,), {}
        sdk = ExecutionArgs(signature, "fixture")
        with self.assertRaises(DSLUserCodeError):
            sdk.get_rectified_args(args, kwargs)
        with self.assertRaises(hook._Refusal):
            hook._arguments(self.synth(signature), args, kwargs, None)

    @parametrize("use_default", (False, True))
    def test_valid_call_matches_sdk_binder(self, use_default):
        signature = Signature(
            (
                Parameter("value", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("scale", Parameter.KEYWORD_ONLY, default=7),
            )
        )
        kwargs = {} if use_default else {"scale": 11}
        expected = ExecutionArgs(signature, "fixture").get_rectified_args((3,), kwargs)
        actual, read_only = hook._arguments(self.synth(signature), (3,), kwargs, None)
        self.assertEqual(actual, tuple(expected))
        self.assertEqual(read_only, frozenset())

    @parametrize("error_type", (RuntimeError, hook._Refusal))
    def test_adapter_error_never_retries_original(self, error_type):
        calls = []
        error = error_type("after adapter side effect")

        def adapter():
            calls.append("adapter")
            raise error

        def original(compiled):
            calls.append("original")

        compiled = _Compiled()
        hook._entries[compiled] = self.synth(Signature(), adapter)
        with hook.observing():
            with self.assertRaises(error_type) as caught:
                hook._observe(compiled, original, (), {})
            self.assertIs(caught.exception, error)
            self.assertFalse(hook._state.inside)
        self.assertEqual(calls, ["adapter"])

    def test_operand_refusal_falls_back_before_adapter(self):
        compiled = _Compiled()
        adapter = mock.Mock()
        original = mock.Mock(return_value="ordinary")
        signature = Signature((Parameter("value", Parameter.POSITIONAL_OR_KEYWORD),))
        hook._entries[compiled] = self.synth(signature, adapter, "tensor")
        with (
            hook.observing(),
            mock.patch.object(
                hook,
                "_real_operand",
                side_effect=hook._Refusal("unrepresentable operand"),
            ),
        ):
            self.assertEqual(hook._observe(compiled, original, (3,), {}), "ordinary")
        adapter.assert_not_called()
        original.assert_called_once_with(compiled, 3)

    def test_original_positional_only_wrapper_stays_positional(self):
        calls = []

        def original(compiled, *args):
            calls.append(args)

        signature = Signature((Parameter("value", Parameter.POSITIONAL_OR_KEYWORD),))
        compiled = _Compiled()
        adapter = mock.Mock()
        hook._entries[compiled] = self.synth(signature, adapter)
        with hook.observing(), self.assertRaises(TypeError):
            hook._hooked_call(original)(compiled, value=3)
        adapter.assert_not_called()
        self.assertEqual(calls, [])

    @parametrize("negative", (False, True))
    def test_f32_baked_constant_distinguishes_signed_zero(self, negative):
        constant = -0.0 if negative else 0.0
        sig = Signature((Parameter("value", Parameter.POSITIONAL_OR_KEYWORD),))
        synth = self.synth(sig)
        synth.formals = (hook._Formal("value", "float", True, constant),)
        self.assertEqual(
            hook._arguments(synth, (constant,), {}, None), ((), frozenset())
        )
        with self.assertRaises(hook._Refusal):
            hook._arguments(synth, (-constant,), {}, None)

    @parametrize("kind", ("float", "rounded", "typed"))
    def test_f32_baked_constant_compares_declared_width(self, kind):
        import cutlass

        sig = Signature((Parameter("value", Parameter.POSITIONAL_OR_KEYWORD),))
        synth = self.synth(sig)
        synth.formals = (hook._Formal("value", "float", True, 1.0),)
        value = {"float": 1.0, "rounded": 1.0 + 2**-25, "typed": cutlass.Float32(1.0)}[
            kind
        ]
        self.assertEqual(hook._arguments(synth, (value,), {}, None), ((), frozenset()))
        with self.assertRaises(hook._Refusal):
            hook._arguments(synth, (1.0 + 2**-22,), {}, None)

    @parametrize("value", (float("inf"), -float("inf"), float("nan")))
    def test_nonfinite_baked_constant_declines_before_entry(self, value):
        import cutlass
        from cutlass.cute.runtime import make_fake_tensor

        from torch._inductor.runtime._cudagraph import _sdk, api

        sig = Signature(
            (
                Parameter("source", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("value", Parameter.POSITIONAL_OR_KEYWORD),
            )
        )
        compiled = _Compiled()
        compiled.execution_args = ExecutionArgs(sig, "fixture")
        fake = make_fake_tensor(cutlass.Float32, (8,), (1,))
        record = hook._Compile(
            lambda source, value: None, (fake, cutlass.Float32(0.0)), {}
        )
        with (
            mock.patch.object(_sdk, "require_active"),
            mock.patch.object(api, "ObservedOrdinaryEntry") as owner,
            self.assertRaisesRegex(hook._Refusal, "non-finite baked constant"),
        ):
            hook._synthesize(compiled, record, (torch.empty(8), value), {})
        owner.assert_not_called()

    @parametrize("use_default", (False, True))
    def test_synthesized_call_preserves_keyword_only(self, use_default):
        import cutlass
        from cutlass import cute
        from cutlass.cute.runtime import make_fake_tensor

        from torch._inductor.runtime._cudagraph import _sdk, api

        calls = []

        def op(source, *, scale=7):
            calls.append((source, scale))

        compiled = _Compiled()
        compiled.execution_args = ExecutionArgs(signature(op), "fixture")
        op.__wrapped__ = lambda: None
        fake = make_fake_tensor(cutlass.Float32, (8,), (1,))
        record = hook._Compile(op, (fake,), {"scale": cutlass.Int32(7)})
        tensor = torch.empty(8)
        kwargs = {} if use_default else {"scale": 11}
        with (
            mock.patch.object(_sdk, "require_active"),
            mock.patch.object(cute, "jit", side_effect=lambda fn: fn),
            mock.patch.object(hook, "_real_operand", return_value=(tensor, False)),
            mock.patch.object(api, "PythonEntry", side_effect=lambda fn: fn),
            mock.patch.object(
                api, "ObservedOrdinaryEntry", side_effect=lambda host, *a, **kw: host
            ),
            mock.patch.object(api, "DirectCuTe", side_effect=lambda owner: owner),
        ):
            synth = hook._synthesize(compiled, record, (tensor,), kwargs)
            arguments, _ = hook._arguments(
                synth, (tensor,), kwargs, lambda t: (t, False)
            )
            synth.adapter(*arguments, 0)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], tensor)
        self.assertEqual(calls[0][1], 7 if use_default else 11)

    def test_nested_phase_cleanup(self):
        previous = getattr(hook._state, "phase", None)
        with hook.observing() as observations:
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with hook.tracing(observations):
                    raise RuntimeError("fixture")
            self.assertEqual(hook._state.phase, "observe")
            self.assertIs(hook._state.observations, observations)
        self.assertIs(getattr(hook._state, "phase", None), previous)

    def test_idle_original_error_is_unchanged(self):
        original = mock.Mock(side_effect=ValueError("original"))
        compiled = _Compiled()
        with self.assertRaisesRegex(ValueError, "original"):
            hook._hooked_call(original)(compiled, 3)
        original.assert_called_once_with(compiled, 3)


if __name__ == "__main__":
    run_tests()
