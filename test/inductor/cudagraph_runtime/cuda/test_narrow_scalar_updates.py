# Owner(s): ["module: inductor"]

import ctypes
import struct
import sys
import unittest

import torch
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


_PTX = b"""
.version 6.0
.target sm_50
.address_size 64
.visible .entry narrow_fields(
    .param .u64 output_ptr,
    .param .align 2 .b8 fields[8]
)
{
    .reg .u64 %p, %v;
    .reg .s32 %s;
    .reg .u32 %u;
    ld.param.u64 %p, [output_ptr];
    ld.param.s8 %s, [fields];
    cvt.s64.s32 %v, %s;
    st.global.u64 [%p], %v;
    ld.param.u8 %u, [fields+1];
    cvt.u64.u32 %v, %u;
    st.global.u64 [%p+8], %v;
    ld.param.s16 %s, [fields+2];
    cvt.s64.s32 %v, %s;
    st.global.u64 [%p+16], %v;
    ld.param.u16 %u, [fields+4];
    cvt.u64.u32 %v, %u;
    st.global.u64 [%p+24], %v;
    ld.param.u16 %u, [fields+6];
    cvt.u64.u32 %v, %u;
    st.global.u64 [%p+32], %v;
    ret;
}
"""


class TestNarrowScalarUpdates(TestCase):
    @unittest.skipIf(torch.version.hip is not None, "CUDA driver PTX test")
    def test_packed_signed_unsigned_fields_and_neighbors(self, device):
        from cuda.bindings import driver

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        module = _check_cuda_bindings(driver.cuModuleLoadData(_PTX))
        entry = None
        try:
            function = _check_cuda_bindings(
                driver.cuModuleGetFunction(module, b"narrow_fields")
            )
            with torch.cuda.stream(stream):
                output = torch.empty(5, dtype=torch.int64, device=device)
                with torch.cuda.graph(graph, stream=stream):
                    pass
                payload = struct.pack(
                    "Q6BH", output.data_ptr(), 0, 0, 0, 0, 0, 0, 0x5AA5
                )
                storage = ctypes.create_string_buffer(payload, len(payload))
                size = ctypes.c_size_t(len(payload))
                extra = (ctypes.c_void_p * 5)(
                    int(driver.CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT),
                    ctypes.addressof(storage),
                    int(driver.CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT),
                    ctypes.addressof(size),
                    int(driver.CU_LAUNCH_PARAM_END_AS_INT),
                )
                params = driver.CUDA_KERNEL_NODE_PARAMS()
                params.func = function
                params.gridDimX = params.gridDimY = params.gridDimZ = 1
                params.blockDimX = params.blockDimY = params.blockDimZ = 1
                params.extra = ctypes.addressof(extra)
                node = int(
                    _check_cuda_bindings(
                        driver.cuGraphAddKernelNode(
                            graph.raw_cuda_graph(), [], 0, params
                        )
                    )
                )
                graph.instantiate()
                (snapshot,) = graph._inspect_captured_kernel_nodes((node,))[3]
                self.assertEqual(
                    tuple((field[0], field[1]) for field in snapshot[8]),
                    ((0, 8), (8, 8)),
                )
                bindings = tuple(
                    (node, 1, offset, width, 0)
                    for offset, width in ((0, 1), (1, 1), (2, 2), (4, 2))
                )
                with self.assertRaisesRegex(ValueError, "Scalar width"):
                    graph._prepare_kernel_replay_updates(
                        (), 2, ((node, 1, 0, 3, 0),), (), 1
                    )
                with self.assertRaisesRegex(ValueError, "Scalar field exceeds"):
                    graph._prepare_kernel_replay_updates(
                        (), 2, ((node, 1, 7, 2, 0),), (), 1
                    )
                batch = graph._prepare_kernel_replay_updates(
                    ((node, 0, 0),), 2, bindings, (), 1
                )
                entry = graph._make_boxed_replay(
                    batch,
                    stream,
                    (module, output),
                    2,
                    (0,),
                    (),
                    None,
                    (("input", 0),),
                    ((1,), (("boxed", 1),)),
                )
                output.fill_(-777)
                for invalid in (-(2**63) - 1, 2**63):
                    with self.assertRaises(OverflowError):
                        entry([output, invalid])
                    self.assertEqual(output, torch.full_like(output, -777))
                self.assertFalse(entry.failed)
                samples = (
                    0,
                    127,
                    128,
                    255,
                    256,
                    32767,
                    32768,
                    65535,
                    65536,
                    -1,
                    -129,
                    -32769,
                    -(2**63),
                    2**63 - 1,
                    17,
                    17,
                    65553,
                )
                for fresh in (False, True):
                    for value in samples:
                        if fresh:
                            backing = torch.full(
                                (7,), -777, dtype=torch.int64, device=device
                            )
                            output = backing[1:6]
                        else:
                            output.fill_(-777)
                        byte, half = value & 255, value & 65535
                        expected = (
                            byte if byte < 128 else byte - 256,
                            byte,
                            half if half < 32768 else half - 65536,
                            half,
                            0x5AA5,
                        )
                        box = [output, value]
                        frames = []

                        def profile(frame, event, result):
                            if event == "call":
                                frames.append(frame.f_code.co_name)

                        try:
                            sys.setprofile(profile)
                            (result,) = entry(box)
                        finally:
                            sys.setprofile(None)
                        self.assertEqual(frames, [])
                        self.assertEqual(box, [])
                        self.assertIs(result, output)
                        self.assertEqual(
                            result,
                            torch.tensor(expected, dtype=torch.int64, device=device),
                        )
                        if fresh:
                            self.assertEqual(
                                backing[[0, 6]],
                                torch.full(
                                    (2,), -777, dtype=torch.int64, device=device
                                ),
                            )
                self.assertFalse(entry.failed)
                entry.close()
                self.assertTrue(entry.closed)
                self.assertEqual(
                    result, torch.tensor(expected, dtype=torch.int64, device=device)
                )
        finally:
            stream.synchronize()
            if entry is not None:
                entry.close()
            graph.reset()
            _check_cuda_bindings(driver.cuModuleUnload(module))


instantiate_device_type_tests(TestNarrowScalarUpdates, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
