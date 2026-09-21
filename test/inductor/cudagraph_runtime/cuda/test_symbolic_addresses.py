"""Reuse direct host variants selected by current input and derived view addresses."""

import json
import sys
from unittest import mock


import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph import direct_host
from torch._inductor.runtime._cudagraph.api import DirectHost, DirectTriton, InputContract, IntegerRange, TensorInput
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, PointerSource
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@triton.jit(do_not_specialize_on_alignment=["source", "destination"])
def add_by_branch(source, destination, count, VALUE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < count
    tl.store(destination + index, tl.load(source + index, mask, other=0) + VALUE, mask)


ADD = METHOD = None


def host(box):
    count, source = box
    box.clear()
    first = source[1:]
    view = first[1:]
    pointer = view.data_ptr() if METHOD == "data_ptr" else view.const_data_ptr()
    if pointer - source.const_data_ptr() != 8:
        raise AssertionError("Nested views must retain their storage byte offset")
    output = torch.empty_strided((count,), (1,), dtype=source.dtype, device=source.device)
    value = 1 if pointer % 16 == 0 else 2
    ADD[(triton.cdiv(count, 128),)](view, output, count, VALUE=value, BLOCK=128)
    return (output,)


class TestSymbolicAddresses(TestCase):
    @parametrize("method", ("data_ptr", "const_data_ptr"))
    def test_current_addresses_choose_cached_variant(self, device, method):
        with torch.cuda.device(device):
            adapter = DirectTriton(add_by_branch)
            self.addCleanup(adapter.close)
            self.enterContext(mock.patch.dict(globals(), {"ADD": adapter, "METHOD": method}))
            contract = InputContract(("integer", "tensor"),
                (TensorInput(1, torch.float32, (102,), (1,)),),
                (IntegerRange(0, 1, 102),), device_index=torch.cuda.current_device())
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            observations = self.enterContext(mock.patch.object(direct_host, "_observe_direct",
                                                                wraps=direct_host._observe_direct))
            preparations = self.enterContext(mock.patch.object(direct_host, "_prepare_observed",
                                                               wraps=direct_host._prepare_observed))
            samples = []
            for offset in (2, 3, 6, 7, 10, 11):
                storage = torch.randn(128, device=device)
                self.assertEqual(storage.data_ptr() % 16, 0)
                samples.append(storage[offset:offset + 102])
            self.assertEqual(len({source.data_ptr() for source in samples}), len(samples))
            held, frames = [], []

            def profile(frame, event, result):
                if event == "call":
                    frames.append((frame.f_code.co_filename, frame.f_code.co_name))

            for index, source in enumerate(samples):
                box = [100, source]
                if index < 2:
                    actual, = runtime(box)
                else:
                    try:
                        sys.setprofile(profile)
                        actual, = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                ordinary, = host([100, source])
                value = 1 if source[2:].data_ptr() % 16 == 0 else 2
                self.assertEqual(value, 1 if index % 2 == 0 else 2)
                self.assertEqual(actual, source[2:] + value)
                self.assertEqual(actual, ordinary)
                held.append((actual, actual.clone()))
                self.assertEqual(len(runtime.variants), min(index + 1, 2))
            self.assertEqual(observations.call_count, 2)
            self.assertEqual(preparations.call_count, 2)
            for variant in runtime.variants:
                trace = variant.program.guards.trace
                self.assertEqual(len(trace.address_bindings), 1)
                binding, = trace.address_bindings
                self.assertEqual(binding.root, InputSource(1))
                self.assertEqual(binding.generation, 0)
                self.assertEqual(binding.node.target, "data_ptr")
                placeholders = [node for node in trace.graph_module.graph.nodes if node.op == "placeholder"]
                self.assertIs(binding.node.args[0], placeholders[1])
                call, = [event for event in variant.program.events if type(event) is DirectKernelCall]
                source, = [arg.source for arg in call.arguments if type(arg.source) is PointerSource
                           and arg.source.root == InputSource(1)]
                self.assertEqual(source.byte_offset.op, "constant")
                self.assertEqual(source.byte_offset.value, 8)
                self.assertEqual(len(variant.program.allocations), 1)
            runtime.close()
            adapter.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)
            print("SYMBOLIC_ADDRESS_RESULT=" + json.dumps({
                "accepted": True,

                "method": method, "samples": 6, "ordinary_observations": 2,
                "preparations": 2, "variants": 2, "native_hits": 4,
                "torch_references": 6, "ordinary_references": 6,
                "python_frames": frames, "held_outputs_after_close": len(held),
            }, sort_keys=True))


instantiate_device_type_tests(TestSymbolicAddresses, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
