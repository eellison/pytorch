# Owner(s): ["module: inductor"]

import gc
import weakref
from unittest import mock

import torch
from torch._inductor.runtime._cudagraph._sdk import activate


activate()

import cutlass
import cutlass.cute as cute

from torch.cuda import (
    _host_trace as ht,
    _host_trace_cute_desc as desc,
    _host_trace_cute_dsl as hook,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class _Compiled:
    pass


@instantiate_parametrized_tests
class TestCuTeDescriptorOwnership(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from cute_host.test_captured_descriptor import _host

        fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (cute.sym_int(32),), (1,), assumed_align=16
        )
        compiled = cute.compile(
            _host,
            fake,
            None,
            fake,
            cutlass.Float32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--gpu-arch sm_90a --enable-tvm-ffi",
            no_jit_engine=True,
        )
        cls.descriptor = desc.Descriptor.from_json(
            hook._compiles[compiled].descriptor.to_json()
        )

    def setUp(self):
        super().setUp()
        self.context = (0, 101)
        self.calls = []
        for name in ("_compiles", "_eager_facts"):
            patch = mock.patch.object(hook, name, weakref.WeakKeyDictionary())
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(
            hook, "_cuda_context", side_effect=lambda: self.context
        )
        self.context_query = patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(
            torch.cuda, "current_device", side_effect=lambda: self.context[0]
        )
        patch.start()
        self.addCleanup(patch.stop)
        self.arguments = (object(), None, object(), 0.0)
        self.nodes = tuple(
            {"name": site.registration.kernel_symbol, "func": 123 + index}
            for index, site in enumerate(self.descriptor.payload.sites)
        )

    def compiled(self, loaded=False):
        compiled = (
            hook.LoadedProgram(self.ordinary, self.descriptor, object(), True)
            if loaded
            else _Compiled()
        )
        hook._compiles[compiled] = hook._Compile(
            None,
            (),
            {},
            descriptor=self.descriptor,
            name="same_name",
            module="torch._native.fixture",
        )
        return compiled

    def ordinary(self, *args):
        self.calls.append("ordinary")
        return "result"

    def observe(self, compiled, capture_context=None):
        def capture(call, device):
            call()
            if capture_context is not None:
                self.context = capture_context
            return self.nodes

        with (
            hook.observing() as observations,
            mock.patch.object(desc, "read_eager_launches", side_effect=capture),
        ):
            self.assertEqual(
                hook._observe(compiled, self.ordinary, self.arguments, {}), "result"
            )
        return observations

    @parametrize("loaded", (False, True))
    def test_capture_owner_survives_until_last_launch_is_released(self, loaded):
        compiled = self.compiled(loaded)
        owner = weakref.ref(compiled)
        observations = self.observe(compiled)
        self.assertEqual(self.calls, ["ordinary", "ordinary"])
        del compiled
        gc.collect()
        self.assertIsNotNone(owner())
        launches = []

        def record(trace, program, args, kwargs, operand, name):
            self.assertIs(program.kernel_owner, owner())
            captured = desc._captured_launches(
                program, program.descriptor.payload.sites
            )
            self.assertEqual(tuple(launch.function for launch in captured), (123, 124))
            launches.extend((launch, program.borrow_native()) for launch in captured)

        with hook.tracing(observations), mock.patch.object(desc, "record", new=record):
            hook._record(None, owner(), self.arguments, {}, observations[0])
        observations.clear()
        gc.collect()
        self.assertTrue(
            all(
                launch.owner is owner() and borrow.program.kernel_owner is owner()
                for launch, borrow in launches
            )
        )
        del launches[:-1]
        gc.collect()
        self.assertIsNotNone(owner())
        launches.clear()
        gc.collect()
        self.assertIsNone(owner())
        self.assertEqual(len(hook._eager_facts), 0)

    @parametrize("warm_up", (False, True))
    def test_same_name_different_owner_is_rejected(self, warm_up):
        compiled = self.compiled()
        observations = self.observe(compiled)
        replacement = self.compiled()
        with (
            hook.tracing(observations if warm_up else None),
            mock.patch.object(desc, "record") as record,
            self.assertRaises(ht.Declined),
        ):
            hook._record(None, replacement, self.arguments, {}, observations[0])
        record.assert_not_called()

    @parametrize("warm_up", (False, True))
    @parametrize("other", ((1, 101), (0, 202)))
    def test_device_or_context_change_is_rejected(self, warm_up, other):
        compiled = self.compiled()
        observations = self.observe(compiled)
        self.context = other
        with (
            hook.tracing(observations if warm_up else None),
            mock.patch.object(desc, "record") as record,
            self.assertRaisesRegex(ht.Declined, "another CUDA device or context"),
        ):
            hook._record(None, compiled, self.arguments, {}, observations[0])
        record.assert_not_called()

    def test_new_warm_up_does_not_rewrite_old_observation(self):
        compiled = self.compiled()
        old = self.observe(compiled)
        self.context = (1, 202)
        current = self.observe(compiled)
        with (
            hook.tracing(old),
            mock.patch.object(desc, "record") as record,
            self.assertRaisesRegex(ht.Declined, "another CUDA device or context"),
        ):
            hook._record(None, compiled, self.arguments, {}, old[0])
        record.assert_not_called()
        with hook.tracing(current), mock.patch.object(desc, "record") as record:
            hook._record(None, compiled, self.arguments, {}, current[0])
        record.assert_called_once()
        self.assertIs(record.call_args.args[1].eager, current[0].eager.nodes)
        with hook.tracing(), mock.patch.object(desc, "record") as record:
            hook._record(None, compiled, self.arguments, {})
        record.assert_called_once()
        self.assertIs(record.call_args.args[1].eager, current[0].eager.nodes)
        self.assertIs(record.call_args.args[1].kernel_owner, compiled)
        self.assertEqual(record.call_args.args[1].context, self.context[1])
        self.assertEqual(record.call_args.args[1]._device.index, self.context[0])

    @parametrize("other", ((0, 0), (1, 101), (0, 202)))
    def test_invalid_capture_context_publishes_no_facts(self, other):
        compiled = self.compiled()
        if other == (0, 0):
            self.context = other
        observations = self.observe(compiled, capture_context=other)
        self.assertNotIn(compiled, hook._eager_facts)
        with (
            hook.tracing(observations),
            mock.patch.object(desc, "record") as record,
            self.assertRaises(ht.Declined),
        ):
            hook._record(None, compiled, self.arguments, {}, observations[0])
        record.assert_not_called()

    def test_failed_verification_publishes_no_facts(self):
        compiled = self.compiled()
        with mock.patch.object(
            desc.Descriptor, "check", side_effect=desc.Unexpressed("mismatch")
        ):
            observations = self.observe(compiled)
        self.assertNotIn(compiled, hook._eager_facts)
        with (
            hook.tracing(observations),
            mock.patch.object(desc, "record") as record,
            self.assertRaisesRegex(ht.Declined, "mismatch"),
        ):
            hook._record(None, compiled, self.arguments, {}, observations[0])
        record.assert_not_called()

    def test_idle_call_does_not_query_context(self):
        compiled = self.compiled()
        with mock.patch.object(hook._state, "phase", None, create=True):
            self.assertEqual(hook._hooked_call(self.ordinary)(compiled), "result")
        self.context_query.assert_not_called()


if __name__ == "__main__":
    run_tests()
