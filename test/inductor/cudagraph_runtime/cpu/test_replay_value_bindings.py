# Owner(s): ["module: inductor"]
"""Late kernel fields remain correct when copy nodes add numeric inputs."""

import struct
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.runtime import cudagraph_compiled_evaluation as evaluation
from torch._inductor.runtime.cudagraph_arg_mapping import (
    InputSource,
    IntegerInput,
    IntExpr,
    ParameterSource,
    PointerSource,
)
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _KernelModule,
    _make_replay,
    _NumericProgram,
    _PhysicalCall,
    _PhysicalField,
)
from torch._inductor.runtime.cudagraph_launch_association import (
    RecordedGraphNode,
    RecordedKernelLaunch,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class _Module(_KernelModule):
    function = 17
    parameter_layout = ((0, 8),)
    shared = 0

    def check(self):
        pass

    def _borrow_for_cudagraph(self):
        return self


class _Graph:
    def __init__(self, pointer):
        self.argument = struct.pack("q", pointer + 32)
        self.fields = None
        self.copies = None

    def _inspect_captured_kernel_nodes(self, nodes):
        kernel = (
            40,
            17,
            0,
            0,
            (1, 1, 1),
            (32, 1, 1),
            0,
            False,
            ((0, 8, self.argument),),
        )
        return 30, 20, (40, 50), (kernel,)

    def _prepare_kernel_replay_updates(
        self, pointers, count, fields, grids, value_count, **kwargs
    ):
        self.fields = fields
        self.copies = kwargs
        return object()

    def _make_boxed_replay(self, *args, **kwargs):
        return self


@instantiate_parametrized_tests
class TestReplayValueBindings(TestCase):
    @parametrize("operation", ("memcpy", "memset"))
    def test_copy_expression_does_not_displace_late_pointer_field(self, operation):
        source = torch.empty(128, dtype=torch.uint8)
        inputs = (source, 5)
        numeric = _NumericProgram(
            SimpleNamespace(
                input_names=("source", "count"),
                integer_inputs=(IntegerInput("count", 1),),
            ),
            inputs,
        )
        offset = IntExpr("constant", 32)
        late = ParameterSource("pointer", 64, PointerSource(InputSource(0), offset))
        one = IntExpr("constant", 1)
        call = _PhysicalCall(
            (_PhysicalField(0, 0, "pointer", late),), _Module(), (one, one, one), ()
        )
        graph = _Graph(source.data_ptr())
        launch = RecordedKernelLaunch(
            10, (1, 20, 30, ()), (1, 20, 30, ((40, bytes(8)),)), 17, (graph.argument,)
        )
        count = IntExpr("multiply", args=(IntExpr("boxed", 1), IntExpr("constant", 8)))
        destination = PointerSource(InputSource(0), IntExpr("constant", 0))
        kwargs = (
            {"memsets": ((50, destination, count, 0),)}
            if operation == "memset"
            else {"memcpys": ((50, destination, destination, count),)}
        )
        event = RecordedGraphNode(
            launch.stream, launch.after, (1, 20, 30, ((50, bytes(8)),)), operation
        )
        compiled = []
        compile_evaluation = evaluation.compile_evaluation

        def remember(*args, **kwargs):
            result = compile_evaluation(*args, **kwargs)
            compiled.append(result)
            return result

        with mock.patch.object(evaluation, "compile_evaluation", side_effect=remember):
            _make_replay(
                graph,
                2,
                (),
                (),
                (),
                (call,),
                (launch,),
                {},
                None,
                numeric=numeric,
                capture_inputs=inputs,
                capture_events=(launch, event),
                **kwargs,
            )
        program = compiled[0]
        field = graph.fields[0]
        copy = graph.copies[operation + "_bindings"][0]
        for size in (5, 9):
            current = torch.empty_like(source)
            status, early = program.early.evaluate_leaves(
                program.early.bind_inputs((current, size))
            )
            self.assertEqual(status, evaluation.EarlyStatus.SUCCESS)
            status, late_values = program.late.evaluate(early, (current.data_ptr(), 0))
            self.assertEqual(status, evaluation.LateStatus.SUCCESS)
            values = (*early, *late_values)
            self.assertEqual(values[field[-1]], current.data_ptr() + 32)
            self.assertEqual(values[copy[-1]], size * 8)


if __name__ == "__main__":
    run_tests()
