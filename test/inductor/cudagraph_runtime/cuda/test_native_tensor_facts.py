# Owner(s): ["module: inductor"]
"""Size and stride loads in the numeric plan, and Tensor fact sources in the dispatch predicate."""

import ctypes

import triton
import triton.language as tl

import torch
from torch._inductor.codecache import CppCodeCache
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


SENTINEL = 424242


@triton.jit(
    do_not_specialize=["value"], do_not_specialize_on_alignment=["value", "output"]
)
def store_value(value, output):
    tl.store(output, value)


def _predicate(source):
    library = CppCodeCache.load(
        "#include <cstdint>\n"
        'extern "C" int8_t guard(int64_t* int_values, double* float_values) {\n'
        f"  return {source};\n"
        "}\n"
    )
    return library, ctypes.cast(library.guard, ctypes.c_void_p).value


class TestNativeTensorFacts(TestCase):
    def capture_entry(
        self, device, instructions, value_index, input_count=1, integer_inputs=()
    ):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            captured_output = torch.empty((1,), dtype=torch.int64, device=device)
            binary = store_value[(1,)](-(1 << 40), captured_output)
            slots = {
                name: index
                for index, (name, _) in enumerate(binary.src.signature.items())
            }
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                store_value[(1,)](-(1 << 40), captured_output)
                frontier = torch._C._cuda_get_capture_frontier(stream.cuda_stream)
                node = frontier[3][0][0]
            graph.instantiate()
            batch = graph._prepare_kernel_replay_updates(
                ((node, slots["output"], 0),),
                input_count,
                ((node, slots["value"], 8, value_index),),
                (),
                len(instructions),
            )
            entry = graph._make_boxed_replay(
                batch,
                stream,
                (binary, captured_output),
                input_count,
                (0,),
                (),
                None,
                (("input", 0),),
                (integer_inputs, instructions),
            )
        self.addCleanup(entry.close)
        self.addCleanup(stream.synchronize)
        return entry, stream

    def test_size_and_stride_loads_read_the_boxed_tensor(self, device):
        # value = size(0) * 1000 + stride(0) of the output tensor itself
        instructions = (
            ("size", 0, 0),
            ("constant", 1000),
            ("multiply", 0, 1),
            ("stride", 0, 0),
            ("add", 2, 3),
        )
        entry, stream = self.capture_entry(device, instructions, 4)
        with torch.cuda.stream(stream):
            base = torch.full((12,), SENTINEL, dtype=torch.int64, device=device)
            for view, expected in (
                (base[:3], 3001),
                (base[::4], 3004),
                (base[:1], 1001),
            ):
                (result,) = entry([view])
                self.assertIs(result, view)
                self.assertEqual(int(result[0].item()), expected)
            scalar = torch.full((), SENTINEL, dtype=torch.int64, device=device)
            with self.assertRaisesRegex(ValueError, "metadata dimension"):
                entry([scalar])

    def test_plan_rejects_loads_of_integer_inputs(self, device):
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            captured_output = torch.empty((1,), dtype=torch.int64, device=device)
            binary = store_value[(1,)](-(1 << 40), captured_output)
            slots = {
                name: index
                for index, (name, _) in enumerate(binary.src.signature.items())
            }
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                store_value[(1,)](-(1 << 40), captured_output)
                node = torch._C._cuda_get_capture_frontier(stream.cuda_stream)[3][0][0]
            graph.instantiate()
            batch = graph._prepare_kernel_replay_updates(
                ((node, slots["output"], 1),), 2, ((node, slots["value"], 8, 0),), (), 1
            )
            with self.assertRaisesRegex(
                ValueError, "Tensor metadata loads must name declared Tensor inputs"
            ):
                graph._make_boxed_replay(
                    batch,
                    stream,
                    (binary, captured_output),
                    2,
                    (1,),
                    (),
                    None,
                    (("input", 1),),
                    ((0,), (("size", 0, 0),)),
                )
        stream.synchronize()

    def test_fact_sources_select_the_variant(self, device):
        entry, stream = self.capture_entry(device, (("stride", 0, 0),), 0)
        misses = []
        dtype_code = torch._C._cuda_scalar_type_code(torch.int64)
        # stride == 4, rank == 1, dtype int64, device index, no math bits, size >= 2
        library, address = _predicate(
            "int_values[0] == 4 && int_values[1] == 1 && int_values[2] == "
            f"{dtype_code} && int_values[3] == {torch.device(device).index or 0} && "
            "int_values[4] == 0 && int_values[5] == 0 && int_values[6] >= 2"
        )
        facts = (
            ("stride", 0, 0),
            ("rank", 0, 0),
            ("dtype", 0, 0),
            ("device", 0, 0),
            ("neg", 0, 0),
            ("conj", 0, 0),
            ("size", 0, 0),
        )
        dispatch = torch._C._cuda_make_boxed_dispatch(
            ((entry, ((), address, library, (), (), facts)),),
            lambda box: misses.append(box) or (box[0],),
        )
        self.addCleanup(dispatch.close)
        with torch.cuda.stream(stream):
            base = torch.full((16,), SENTINEL, dtype=torch.int64, device=device)
            hit = base[::4]
            (result,) = dispatch([hit])
            self.assertIs(result, hit)
            self.assertEqual(int(hit[0].item()), 4)
            self.assertEqual(misses, [])
            for miss in (
                base[:4],
                base[::4][:1],
                base[::4].view(torch.float64),
                base[::4].reshape(2, 2),
            ):
                dispatch([miss])
            self.assertEqual(len(misses), 4)
            # the misses never launched: only the hit's element 0 changed
            self.assertEqual(int(base[0].item()), 4)
            self.assertTrue(bool((base[1:] == SENTINEL).all().item()))

    @parametrize("binding_order", ("grouped", "interleaved"))
    def test_multi_input_fact_order_and_rebinding(self, device, binding_order):
        instructions = (
            ("size", 1, 0),
            ("constant", 100000),
            ("multiply", 0, 1),
            ("size", 1, 1),
            ("constant", 1000),
            ("multiply", 3, 4),
            ("add", 2, 5),
            ("stride", 1, 0),
            ("constant", 10),
            ("multiply", 7, 8),
            ("add", 6, 9),
            ("stride", 1, 1),
            ("add", 10, 11),
            ("boxed", 2),
            ("boxed", 3),
            ("boxed", 4),
            ("boxed", 5),
        )
        entry, stream = self.capture_entry(device, instructions, 12, 6, (2, 3, 4, 5))
        source_facts = (
            ("size", 1, 0),
            ("stride", 1, 0),
            ("size", 1, 1),
            ("stride", 1, 1),
            ("rank", 1, 0),
            ("dtype", 1, 0),
        )
        output_facts = (
            ("rank", 0, 0),
            ("stride", 0, 0),
            ("dtype", 0, 0),
            ("device", 0, 0),
            ("size", 0, 0),
            ("pinned", 0, 0),
        )
        facts = (
            source_facts + output_facts
            if binding_order == "grouped"
            else tuple(row for pair in zip(source_facts, output_facts) for row in pair)
        )
        slots = {fact: f"int_values[{7 + i}]" for i, fact in enumerate(facts)}
        encoded = (
            f"{slots['size', 1, 0]} * 100000 + {slots['size', 1, 1]} * 1000 + "
            f"{slots['stride', 1, 0]} * 10 + {slots['stride', 1, 1]}"
        )
        dtype = torch._C._cuda_scalar_type_code(torch.int64)
        device_index = torch.device(device).index or 0
        conditions = (
            f"int_values[0] == ({encoded})",
            "int_values[1] == int_values[5]",
            "int_values[2] == int_values[4]",
            "int_values[3] == int_values[6]",
            f"{slots['rank', 1, 0]} == 2",
            f"{slots['dtype', 1, 0]} == {dtype}",
            f"{slots['rank', 0, 0]} == 1",
            f"{slots['stride', 0, 0]} == 2",
            f"{slots['dtype', 0, 0]} == {dtype}",
            f"{slots['device', 0, 0]} == {device_index}",
            f"{slots['size', 0, 0]} == 4",
            f"{slots['pinned', 0, 0]} == 0",
        )
        library, address = _predicate(" && ".join(conditions))
        misses = []
        dispatch = torch._C._cuda_make_boxed_dispatch(
            ((entry, ((2, 3, 4, 5), address, library, (1, 0), (1,), facts)),),
            lambda box: misses.append(box) or (box[0],),
        )
        self.addCleanup(dispatch.close)
        with torch.cuda.stream(stream):
            output = torch.full((9,), SENTINEL, dtype=torch.int64, device=device)[1::2]
            source = torch.empty((128,), dtype=torch.int64, device=device)
            for shape, strides, offset, expected in (
                ((2, 3), (3, 1), 0, 203031),
                ((3, 2), (1, 3), 5, 302013),
                ((3, 3), (9, 2), 11, 303092),
            ):
                source.as_strided_(shape, strides, offset)
                box = [
                    output,
                    source,
                    expected,
                    output.data_ptr(),
                    source.data_ptr(),
                    offset,
                ]
                (result,) = dispatch(box)
                self.assertIs(result, output)
                self.assertEqual(
                    result,
                    torch.tensor(
                        [expected, SENTINEL, SENTINEL, SENTINEL],
                        dtype=torch.int64,
                        device=device,
                    ),
                )
                self.assertEqual(len(misses), 0)
            source.as_strided_((6,), (1,), 0)
            output.fill_(SENTINEL)
            (result,) = dispatch(
                [output, source, 0, output.data_ptr(), source.data_ptr(), 0]
            )
            self.assertIs(result, output)
            self.assertEqual(len(misses), 1)
            self.assertEqual(result, torch.full_like(output, SENTINEL))
            (result,) = dispatch([output, object(), 0, output.data_ptr(), 0, 0])
            self.assertIs(result, output)
            self.assertEqual(len(misses), 2)
            self.assertEqual(result, torch.full_like(output, SENTINEL))
            source.as_strided_((2, 3), (3, 1), 17)
            (result,) = dispatch(
                [output, source, 203031, output.data_ptr(), source.data_ptr(), 17]
            )
            self.assertIs(result, output)
            self.assertEqual(result[0], 203031)
            self.assertEqual(len(misses), 2)

    def test_fact_sources_are_validated(self, device):
        entry, _ = self.capture_entry(device, (("stride", 0, 0),), 0)
        library, address = _predicate("1")
        for bad, message in (
            (("area", 0, 0), "Unsupported Tensor metadata kind"),
            (
                ("size", 3, 0),
                "Tensor metadata must name declared Tensor inputs",
            ),
            ((0, 0, 0), "Tensor metadata kinds must be exact strings"),
        ):
            with self.assertRaisesRegex((TypeError, ValueError), message):
                torch._C._cuda_make_boxed_dispatch(
                    ((entry, ((), address, library, (), (), (bad,))),), lambda box: box
                )


instantiate_device_type_tests(TestNativeTensorFacts, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
