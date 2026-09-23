# Owner(s): ["module: inductor"]

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch._inductor.runtime._cudagraph._sdk import activate


activate()

import cutlass

from cuda.bindings import driver
from cutlass import cute

from torch.cuda import (
    _host_trace,
    _host_trace_cute_desc as descriptor,
    _host_trace_cute_dsl as hook,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@cute.kernel
def mark_grid(source: cute.Tensor, output: cute.Tensor):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    if thread == 0 and block < output.shape[0]:
        output[block] = source[0] + cutlass.Float32(block) + cutlass.Float32(1)


@cute.jit
def launch_grid(source: cute.Tensor, output: cute.Tensor, stream: driver.CUstream):
    rows = cutlass.Int32(source.shape[0])
    blocks = cutlass.Int32(129) + cutlass.Int32(cutlass.Int8(rows))
    mark_grid(source, output).launch(
        grid=(blocks, 1, 1), block=(32, 1, 1), smem=0, stream=stream
    )


class TestCuTeCachedNumeric(TestCase):
    @parametrize("transport", ("inprocess", "persisted"))
    def test_grid_source_narrowing_refuses_native_trace(self, device, transport):
        hook.install()
        # Select the descriptor route for this test's real compiler artifact.
        self.enterContext(
            patch.object(hook, "_NATIVE_MODULES", (*hook._NATIVE_MODULES, __name__))
        )
        fake_source = cute.runtime.make_fake_tensor(
            cutlass.Float32, (cute.sym_int(32),), (1,), assumed_align=16
        )
        fake_output = cute.runtime.make_fake_tensor(
            cutlass.Float32, (512,), (1,), assumed_align=16
        )
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            launch_grid,
            fake_source,
            fake_output,
            fake_stream,
            options="--enable-tvm-ffi",
        )
        text = hook.descriptor_json(compiled)
        self.assertIsNotNone(text)
        saved = descriptor.Descriptor.from_json(text)
        reason = "Unsupported source scalar width"
        self.assertIn(reason, saved.declined)
        self.assertIsNone(saved.payload)
        with self.assertRaisesRegex(descriptor.Unexpressed, reason):
            saved.check()
        self.assertEqual(saved.to_json(), text)
        if transport == "persisted":
            directory = self.enterContext(TemporaryDirectory())
            path = Path(directory) / "grid.o"
            compiled.export_to_c(object_file_path=str(path), function_name="func")
            module = cute.runtime.load_module(str(path), enable_tvm_ffi=True)
            entry = hook.loaded_program(module["func"], module, text)
            self.assertTrue(entry.native)
        else:
            entry = compiled
        self.assertEqual(hook.descriptor_json(entry), text)

        def host(source):
            output = torch.zeros(512, device=source.device, dtype=source.dtype)
            entry(source, output)
            return output

        cases = ((7, 0, 136), (256, 4, 129), (7, 8, 136))
        inputs = [
            torch.randint(-8, 8, (rows + offset,), device=device, dtype=torch.float32)[
                offset:
            ]
            for rows, offset, _ in cases
        ]
        for source, (_, _, blocks) in zip(inputs, cases):
            actual = host(source)
            expected = torch.zeros(512, device=device)
            expected[:blocks] = (
                source[0] + torch.arange(blocks, device=device, dtype=torch.float32) + 1
            )
            self.assertEqual(actual, expected, atol=0, rtol=0)
            with self.assertRaisesRegex(_host_trace.Declined, reason) as refused:
                _host_trace.trace(host, (source,))
            self.assertTrue(refused.exception.warm_up_ran)
            self.assertEqual(
                refused.exception.warm_up_outputs, expected, atol=0, rtol=0
            )


instantiate_device_type_tests(TestCuTeCachedNumeric, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
