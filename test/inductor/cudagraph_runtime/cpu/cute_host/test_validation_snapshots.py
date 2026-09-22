# Owner(s): ["module: inductor"]

from contextvars import copy_context
from unittest import mock

from torch._inductor.runtime._cudagraph._sdk import activate


activate()

from cutlass._mlir import ir

from torch._inductor.runtime._cudagraph._compiler import accessors, compiler_boundary
from torch._inductor.runtime._cudagraph._compiler.validation_snapshots import (
    validation_snapshots,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _module():
    return ir.Module.parse(
        """
module {
  func.func @getter() -> i64 {
    %c = arith.constant 3 : i64
    return %c : i64
  }
}
"""
    )


def _change_constant(module, value):
    function = next(iter(module.body.operations)).operation
    constant = next(iter(function.regions[0].blocks[0].operations)).operation
    constant.attributes["value"] = ir.IntegerAttr.get(
        ir.IntegerType.get_signless(64), value
    )


@instantiate_parametrized_tests
class TestValidationSnapshots(TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(ir.Context())
        self.enterContext(ir.Location.unknown())
        self.enterContext(ir.raw_values())

    @parametrize("count", (1, 8))
    def test_repeated_reads_preserve_each_module_and_serializer(self, count):
        modules = (_module(), _module())
        expected_local = tuple(accessors._snapshot(m.operation) for m in modules)
        expected_full = tuple(compiler_boundary._snapshot(m) for m in modules)
        with (
            mock.patch.object(
                accessors, "_take_snapshot", wraps=accessors._take_snapshot
            ) as local,
            mock.patch.object(
                compiler_boundary,
                "_take_snapshot",
                wraps=compiler_boundary._take_snapshot,
            ) as full,
        ):
            with validation_snapshots(*modules):
                for _ in range(count):
                    for index, module in enumerate(modules):
                        self.assertEqual(
                            accessors._snapshot(module.operation), expected_local[index]
                        )
                        self.assertEqual(
                            compiler_boundary._snapshot(module), expected_full[index]
                        )
                self.assertEqual(local.call_count, 2)
                self.assertEqual(full.call_count, 2)
            self.assertEqual(local.call_count, 4)
            self.assertEqual(full.call_count, 4)

    def test_unrelated_module_and_getter_are_not_cached(self):
        admitted, unrelated = _module(), _module()
        getter = next(iter(admitted.body.operations)).operation
        with mock.patch.object(
            accessors, "_take_snapshot", wraps=accessors._take_snapshot
        ) as snapshots:
            with validation_snapshots(admitted):
                accessors._snapshot(admitted.operation)
                for _ in range(3):
                    accessors._snapshot(getter)
                    accessors._snapshot(unrelated.operation)
                self.assertEqual(snapshots.call_count, 7)
            self.assertEqual(snapshots.call_count, 8)

    @parametrize("changed", (0, 1))
    def test_mutation_of_either_module_is_rejected_at_exit(self, changed):
        modules = (_module(), _module())
        before = accessors._snapshot(modules[changed].operation)
        with self.assertRaisesRegex(RuntimeError, "module changed during read-only"):
            with validation_snapshots(*modules):
                for module in modules:
                    accessors._snapshot(module.operation)
                _change_constant(modules[changed], 4)
        self.assertTrue(modules[changed].operation.verify())
        self.assertNotEqual(accessors._snapshot(modules[changed].operation), before)

    def test_nested_and_subsequent_scopes_take_fresh_snapshots(self):
        module = _module()
        with mock.patch.object(
            accessors, "_take_snapshot", wraps=accessors._take_snapshot
        ) as snapshots:
            with validation_snapshots(module):
                first = accessors._snapshot(module.operation)
                with validation_snapshots(module):
                    self.assertEqual(accessors._snapshot(module.operation), first)
                self.assertEqual(snapshots.call_count, 3)
                self.assertEqual(accessors._snapshot(module.operation), first)
            self.assertEqual(snapshots.call_count, 4)
            _change_constant(module, 5)
            with validation_snapshots(module):
                self.assertNotEqual(accessors._snapshot(module.operation), first)
            self.assertEqual(snapshots.call_count, 6)

    def test_failed_body_preserves_error_and_cannot_reuse_snapshots(self):
        module = _module()
        error = ValueError("validation body failed")
        with self.assertRaisesRegex(ValueError, "validation body failed") as caught:
            with validation_snapshots(module):
                first = accessors._snapshot(module.operation)
                _change_constant(module, 6)
                raise error
        self.assertIs(caught.exception, error)
        after = accessors._snapshot(module.operation)
        self.assertNotEqual(after, first)
        with validation_snapshots(module):
            self.assertEqual(accessors._snapshot(module.operation), after)

    def test_exit_serializer_error_does_not_leave_a_cached_snapshot(self):
        module = _module()
        serialize = accessors._take_snapshot
        error = OSError("snapshot writer failed")
        calls = 0

        def fail_on_exit(operation):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise error
            return serialize(operation)

        with mock.patch.object(accessors, "_take_snapshot", new=fail_on_exit):
            with self.assertRaisesRegex(OSError, "snapshot writer failed") as caught:
                with validation_snapshots(module):
                    first = accessors._snapshot(module.operation)
            self.assertIs(caught.exception, error)
            _change_constant(module, 9)
            with validation_snapshots(module):
                self.assertNotEqual(accessors._snapshot(module.operation), first)
            self.assertEqual(calls, 4)

    def test_location_only_mutation_is_rejected_at_exit(self):
        module = _module()
        before = accessors._snapshot(module.operation)
        with self.assertRaisesRegex(RuntimeError, "module changed during read-only"):
            with validation_snapshots(module):
                accessors._snapshot(module.operation)
                function = next(iter(module.body.operations)).operation
                function.location = ir.Location.file("changed.mlir", 5, 2)
        after = accessors._snapshot(module.operation)
        self.assertEqual(after[0], before[0])
        self.assertNotEqual(after[1], before[1])

    def test_copied_context_after_exit_cannot_reuse_snapshots(self):
        module = _module()
        with mock.patch.object(
            accessors, "_take_snapshot", wraps=accessors._take_snapshot
        ) as snapshots:
            with validation_snapshots(module):
                first = accessors._snapshot(module.operation)
                copied = copy_context()
            self.assertEqual(snapshots.call_count, 2)
            _change_constant(module, 7)
            after = copied.run(accessors._snapshot, module.operation)
            self.assertNotEqual(after, first)
            _change_constant(module, 8)
            self.assertNotEqual(
                copied.run(accessors._snapshot, module.operation), after
            )
            self.assertEqual(snapshots.call_count, 4)


if __name__ == "__main__":
    run_tests()
