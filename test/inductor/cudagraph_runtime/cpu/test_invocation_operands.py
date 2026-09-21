"""Multiple Tensor sources retain the single explicit destination contract."""

import torch
from torch._inductor import config
from torch._inductor.runtime._cudagraph._compiler.compiler_cute_handoff.invocation import (
    InvocationDeclined,
    invoke_cute,
    register_cpu_standin,
)
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def add_sources(left, right, destination):
    destination.copy_(left + right)


def add_one(source, destination):
    destination.copy_(source + 1)


@instantiate_parametrized_tests
class TestInvocationOperands(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        self.enterContext(torch.no_grad())
        self.enterContext(config.patch(cudagraph_policy=None, cpp_wrapper=False))

    @parametrize("compiled", (False, True))
    @parametrize("source_count", (1, 2))
    def test_tensor_operands(self, compiled, source_count):
        entry = register_cpu_standin(add_one if source_count == 1 else add_sources)
        self.addCleanup(entry.close)
        self.assertEqual(entry.formals, ("source", "destination") if source_count == 1 else
                         ("left", "right", "destination"))

        def operation(*sources):
            output = torch.empty_like(sources[0])
            invoke_cute(entry.key, *sources, output)
            return output, *sources

        function = torch.compile(operation, fullgraph=True) if compiled else operation
        held = []
        for index in range(2):
            left = torch.arange(32, dtype=torch.float32).reshape(4, 8) + index
            right = torch.full_like(left, index + 2)
            sources = (left,) if source_count == 1 else (left, right)
            source_values = tuple(source.clone() for source in sources)
            expected = left + 1 if source_count == 1 else left + right
            if compiled and index == 0:
                actual, code = run_and_get_code(function, *sources)
                self.assertEqual(len(code), 1)
                self.assertNotIn("compiler_cute_handoff.descriptor", code[0])
                self.assertNotIn("compiler_cute_handoff.envelope", code[0])
                self.assertNotIn("_cudagraph_mixed_ir", code[0])
                self.assertNotIn("_cudagraph_terminal_metadata", code[0])
            else:
                actual = function(*sources)
            self.assertEqual(actual[0], expected)
            for returned, source, original in zip(actual[1:], sources, source_values, strict=True):
                self.assertIs(returned, source)
                self.assertEqual(source, original)
                self.assertNotEqual(actual[0].data_ptr(), source.data_ptr())
            held.append((actual, expected))
        entry.close()
        for actual, expected in held:
            self.assertEqual(actual[0], expected)

    @parametrize("fault", ("missing", "extra", "source_scalar", "destination_scalar"))
    def test_invalid_operands_rejected_before_invocation(self, fault):
        entry = register_cpu_standin(add_sources)
        self.addCleanup(entry.close)
        left, right, destination = (torch.full((4,), value) for value in (1.0, 2.0, -1.0))
        operands = {
            "missing": (left, destination),
            "extra": (left, right, destination, destination),
            "source_scalar": (left, 2, destination),
            "destination_scalar": (left, right, 0),
        }[fault]
        with self.assertRaisesRegex(InvocationDeclined, "Tensor"):
            invoke_cute(entry.key, *operands)
        with self.assertRaisesRegex(InvocationDeclined, "Tensor"):
            entry.invoke(*operands)
        self.assertEqual(destination, torch.full((4,), -1.0))

    @parametrize("fault", ("count", "scalar"))
    def test_dynamo_rejects_invalid_registered_call(self, fault):
        entry = register_cpu_standin(add_sources)
        self.addCleanup(entry.close)

        def operation(left, right):
            output = torch.empty_like(left)
            if fault == "count":
                invoke_cute(entry.key, left, output)
            else:
                invoke_cute(entry.key, left, 2, output)
            return output

        compiled = torch.compile(operation, fullgraph=True)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "CuTe invocation"):
            compiled(torch.ones(4), torch.ones(4))


if __name__ == "__main__":
    run_tests()
