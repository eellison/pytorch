# Owner(s): ["module: inductor"]

import contextlib
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from torch._inductor.runtime._cudagraph import _sdk


_sdk.activate()

from cutlass._mlir import ir

from torch._inductor.runtime._cudagraph._compiler import source_dispatch
from torch._inductor.runtime._cudagraph._compiler.accessors import _function, _snapshot
from torch._inductor.runtime._cudagraph._compiler.cfg_values import read_cfg_function
from torch._inductor.runtime._cudagraph._compiler.continuation import (
    BoundDispatchHelper,
    check_dispatch_consumers,
)
from torch._inductor.runtime._cudagraph._compiler.dispatch_join import JoinedDispatch
from torch._inductor.runtime._cudagraph._compiler.entry_signature import (
    MetadataSnapshot,
    ParameterMetadata,
)
from torch._inductor.runtime._cudagraph._compiler.helpers import DispatchScalarHelper
from torch._inductor.runtime._cudagraph._compiler.source_dispatch import SourceDispatch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestDispatchCheckSnapshots(TestCase):
    def setUp(self):
        super().setUp()
        self.contexts = contextlib.ExitStack()
        self.addCleanup(self.contexts.close)
        self.context = self.contexts.enter_context(ir.Context())
        self.contexts.enter_context(ir.Location.unknown())
        self.contexts.enter_context(ir.raw_values())
        self.module = ir.Module.parse("""module {
          llvm.func @host() -> i64 attributes {arg_attrs = []} {
            %zero = llvm.mlir.constant(0 : i64) : i64
            llvm.return %zero : i64
          }
          llvm.func @helper() -> i64 attributes {arg_attrs = []} {
            %zero = llvm.mlir.constant(0 : i64) : i64
            llvm.return %zero : i64
          }
        }""")
        host = _function(self.module, "host", "llvm.func")
        operation = _function(self.module, "helper", "llvm.func")
        value = host.regions[0].blocks[0].operations[0].results[0]
        metadata = MetadataSnapshot(
            "host",
            "host",
            "test",
            (),
            (),
            ParameterMetadata("Scalar", "ret", None, None),
        )
        source = SourceDispatch(
            self.module,
            self.context,
            "host",
            metadata,
            host,
            (),
            (),
            (),
            None,
            value,
            (),
            (),
            _snapshot(self.module.operation),
            (),
        )
        self.source = replace(source, _seal=source._state())
        self.compiled_module = ir.Module.parse(str(self.module))
        for name in ("host", "helper"):
            function = _function(self.compiled_module, name, "llvm.func")
            function.attributes["arg_attrs"] = ir.ArrayAttr.get([])
        self.cfg = read_cfg_function(self.compiled_module, "helper")
        self.operation = operation
        self.value = value
        program = SimpleNamespace(
            source_module=self.module,
            source_context=self.context,
            module=self.compiled_module,
            context=self.context,
            function_name="host",
        )
        self.joined = JoinedDispatch(
            SimpleNamespace(source=self.source),
            SimpleNamespace(program=program),
            (),
            (),
        )
        # Focus this fixture on source snapshots and consumer ownership, not IR lowering.
        inspected = (host, (), (), None, value, (), ())
        self.contexts.enter_context(
            mock.patch.object(source_dispatch, "_inspect", return_value=inspected)
        )
        self.join_checks = self.contexts.enter_context(
            mock.patch.object(JoinedDispatch, "check")
        )
        self.bodies = self.contexts.enter_context(
            mock.patch.object(DispatchScalarHelper, "_check_body", autospec=True)
        )
        self.snapshots = self.contexts.enter_context(
            mock.patch.object(source_dispatch, "_snapshot", wraps=_snapshot)
        )

    def consumers(self, count, distinct_sources=False):
        consumers = []
        for _ in range(count):
            source = replace(self.source) if distinct_sources else self.source
            helper = DispatchScalarHelper(
                "helper",
                "predicate",
                0,
                self.operation,
                source,
                None,
                self.value,
                (),
                (),
                (),
                "i64",
                ("prefix", 0, 0),
                (),
                (),
            )
            helper = replace(helper, _seal=helper._state())
            consumer = BoundDispatchHelper(
                self.joined,
                helper,
                None,
                "predicate",
                0,
                self.value,
                self.cfg,
                (),
                (),
            )
            consumers.append(replace(consumer, _seal=consumer._state()))
        return tuple(consumers)

    @parametrize("count", (0, 1, 24))
    def test_shared_source_snapshot_count(self, count):
        consumers = self.consumers(count)
        check_dispatch_consumers(self.joined, consumers)
        self.assertEqual(self.snapshots.call_count, 2 if count else 0)
        self.assertEqual(self.bodies.call_count, count)
        self.assertEqual(self.join_checks.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in self.bodies.call_args_list],
            [c.helper for c in consumers],
        )

    def test_distinct_source_owners_are_each_checked(self):
        consumers = self.consumers(3, distinct_sources=True)
        check_dispatch_consumers(self.joined, consumers)
        self.assertEqual(self.snapshots.call_count, 6)
        self.assertEqual(self.bodies.call_count, 3)

    def test_mutation_after_a_helper_is_rejected(self):
        consumers = self.consumers(2)

        def mutate(helper, source):
            self.module.operation.attributes["test.changed"] = ir.UnitAttr.get()

        self.bodies.side_effect = mutate
        with self.assertRaisesRegex(RuntimeError, "Original dispatch Module changed"):
            check_dispatch_consumers(self.joined, consumers)
        self.assertEqual(self.bodies.call_count, 2)

    def test_mutation_between_checks_is_rejected_then_restored(self):
        consumers = self.consumers(2)
        check_dispatch_consumers(self.joined, consumers)
        self.module.operation.attributes["test.changed"] = ir.UnitAttr.get()
        with self.assertRaisesRegex(RuntimeError, "Original dispatch Module changed"):
            check_dispatch_consumers(self.joined, consumers)
        self.assertEqual(self.bodies.call_count, 2)
        del self.module.operation.attributes["test.changed"]
        check_dispatch_consumers(self.joined, consumers)
        self.assertEqual(self.bodies.call_count, 4)

    def test_failed_body_does_not_reuse_an_earlier_snapshot(self):
        consumers = self.consumers(2)
        self.bodies.side_effect = ValueError("helper failed")
        with self.assertRaisesRegex(ValueError, "helper failed"):
            check_dispatch_consumers(self.joined, consumers)
        self.bodies.side_effect = None
        self.module.operation.attributes["test.changed"] = ir.UnitAttr.get()
        with self.assertRaisesRegex(RuntimeError, "Original dispatch Module changed"):
            check_dispatch_consumers(self.joined, consumers)
        self.assertEqual(self.bodies.call_count, 1)

    def test_direct_consumer_check_still_checks_its_source(self):
        (consumer,) = self.consumers(1)
        consumer.check()
        self.assertEqual(self.snapshots.call_count, 1)
        self.assertEqual(self.bodies.call_count, 1)
        self.module.operation.attributes["test.changed"] = ir.UnitAttr.get()
        with self.assertRaisesRegex(RuntimeError, "Original dispatch Module changed"):
            consumer.check()


if __name__ == "__main__":
    run_tests()
