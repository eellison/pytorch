"""Keep shared callable ownership across actual CuTe SDK preprocessing."""

from contextvars import Context
from inspect import getattr_static


from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

from cutlass import cute
from cutlass.base_dsl.dsl import BaseDSL
from torch._inductor.runtime._cudagraph._compiler.frontend import _TRACE_LOCK
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import _CallableState, _callable_state, _observe_preprocessing
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


def make_function():
    @cute.jit
    def shared(value):
        return value + 1

    return shared.__wrapped__


def replacement(value):
    return value - 1


class TestPreprocessingOwnership(TestCase):
    def test_two_snapshots_accept_actual_sdk_transition(self):
        function = make_function()
        first = _CallableState(function, _callable_state(function))
        second = _CallableState(function, _callable_state(function))
        original = getattr_static(BaseDSL, "_preprocess_and_replace_code")
        with _TRACE_LOCK, _observe_preprocessing():
            BaseDSL._preprocess_and_replace_code(function)
        self.assertIsNot(function.__code__, first.state[3])
        self.assertIs(getattr_static(BaseDSL, "_preprocess_and_replace_code"), original)
        first.check()
        second.check()
        _CallableState(function, _callable_state(function)).check()

    @parametrize("change", ("replacement", "rollback", "defaults"))
    def test_unobserved_changes_still_fail(self, change):
        function = make_function()
        state = _CallableState(function, _callable_state(function))
        with _TRACE_LOCK, _observe_preprocessing():
            BaseDSL._preprocess_and_replace_code(function)
        state.check()
        if change == "replacement":
            function.__code__ = replacement.__code__
        elif change == "rollback":
            function.__code__ = state.state[3]
        else:
            function.__defaults__ = (7,)
        with self.assertRaisesRegex(RuntimeError, "callable code, defaults or receiver changed"):
            state.check()

    def test_other_context_is_not_recorded(self):
        function = make_function()
        state = _CallableState(function, _callable_state(function))
        with _TRACE_LOCK, _observe_preprocessing():
            Context().run(BaseDSL._preprocess_and_replace_code, function)
        self.assertIsNot(function.__code__, state.state[3])
        with self.assertRaisesRegex(RuntimeError, "callable code, defaults or receiver changed"):
            state.check()

    def test_scope_restores_sdk_descriptor_after_error(self):
        function = make_function()
        state = _CallableState(function, _callable_state(function))
        original = getattr_static(BaseDSL, "_preprocess_and_replace_code")
        with self.assertRaisesRegex(RuntimeError, "test scope failure"):
            with _TRACE_LOCK, _observe_preprocessing():
                BaseDSL._preprocess_and_replace_code(function)
                raise RuntimeError("test scope failure")
        self.assertIs(getattr_static(BaseDSL, "_preprocess_and_replace_code"), original)
        state.check()


instantiate_parametrized_tests(TestPreprocessingOwnership)

if __name__ == "__main__":
    run_tests()
