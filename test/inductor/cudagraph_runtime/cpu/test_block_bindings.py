# Owner(s): ["module: inductor"]
"""Symbolic block recipes retain their exact captured launch correspondence."""

import struct
from dataclasses import replace
from types import SimpleNamespace

from torch._inductor.runtime.cudagraph_arg_mapping import (
    IntegerInput,
    IntegerSource,
    IntExpr,
)
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _KernelModule,
    _make_replay,
    _NumericProgram,
    _PhysicalCall,
    _PhysicalField,
)
from torch._inductor.runtime.cudagraph_launch_association import (
    RecordedKernelLaunch,
    UnsupportedCapture,
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

    def __init__(self, shared):
        self._shared = shared

    @property
    def shared(self):
        return self._shared

    def check(self):
        pass

    def _borrow_for_cudagraph(self):
        return self


class _Graph:
    def __init__(self, kernel):
        self.kernel = kernel
        self.grid_bindings = None

    def _inspect_captured_kernel_nodes(self, nodes):
        return (30, 20, tuple(nodes), (self.kernel,))

    def _prepare_kernel_replay_updates(
        self, pointers, count, scalars, grids, value_count
    ):
        self.grid_bindings = grids
        return object()

    def _make_boxed_replay(self, *args, **kwargs):
        return args, kwargs


@instantiate_parametrized_tests
class TestBlockBindings(TestCase):
    def prepare(self, *, dynamic_block=True, dynamic_shared=False, fault=None):
        numeric = _NumericProgram(
            SimpleNamespace(
                input_names=("bx", "by", "bz", "smem"),
                integer_inputs=tuple(
                    IntegerInput(name, i)
                    for i, name in enumerate(("bx", "by", "bz", "smem"))
                ),
            ),
            (8, 4, 1, 16),
        )
        dimensions = tuple(IntExpr("boxed", axis) for axis in range(3))
        module = _Module(None if dynamic_shared else 16)
        call = _PhysicalCall(
            (_PhysicalField(0, 0, "i64", IntegerSource(7)),),
            module,
            tuple(IntExpr("constant", value) for value in (2, 1, 1)),
            (),
            shared=IntExpr("boxed", 3) if dynamic_shared else None,
            block=dimensions if dynamic_block else None,
        )
        kernel = (
            40,
            17,
            0,
            0,
            (2, 1, 1),
            (8, 4, 1),
            16,
            False,
            ((0, 8, struct.pack("q", 7)),),
        )
        launch = RecordedKernelLaunch(
            10,
            (1, 20, 30, ()),
            (1, 20, 30, ((40, bytes(8)),)),
            17,
            (struct.pack("q", 7),),
        )
        if fault == "different_block":
            kernel = (*kernel[:5], (4, 8, 1), *kernel[6:])
        elif fault == "two_axes":
            call = replace(call, block=dimensions[:2])
        elif fault == "list_axes":
            call = replace(call, block=list(dimensions))
        elif fault == "different_function":
            kernel = (40, 18, *kernel[2:])
        elif fault == "different_node":
            kernel = (41, *kernel[1:])
        graph = _Graph(kernel)
        result = _make_replay(
            graph, 4, (), (), (), (call,), (launch,), {}, None, numeric=numeric
        )
        return graph, numeric, result

    @parametrize("dynamic_block", (False, True))
    @parametrize("dynamic_shared", (False, True))
    def test_old_and_extended_binding_forms(self, dynamic_block, dynamic_shared):
        graph, numeric, _ = self.prepare(
            dynamic_block=dynamic_block, dynamic_shared=dynamic_shared
        )
        (row,) = graph.grid_bindings
        self.assertEqual(len(row), 4 + int(dynamic_shared) + 3 * int(dynamic_block))
        self.assertEqual(row[0], 40)
        self.assertEqual(tuple(numeric.values[index] for index in row[1:4]), (2, 1, 1))
        if dynamic_shared:
            self.assertEqual(numeric.values[row[4]], 16)
        if dynamic_block:
            self.assertEqual(
                tuple(numeric.values[index] for index in row[-3:]), (8, 4, 1)
            )

    @parametrize(
        "fault",
        (
            "different_block",
            "two_axes",
            "list_axes",
            "different_function",
            "different_node",
        ),
    )
    def test_block_transport_preserves_capture_checks(self, fault):
        with self.assertRaises(UnsupportedCapture):
            self.prepare(fault=fault)


if __name__ == "__main__":
    run_tests()
