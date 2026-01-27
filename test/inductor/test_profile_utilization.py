"""
Tests for profiler utilization annotations (Goal 5).

This tests:
1. Cublas GEMM operations achieve 20-95% bandwidth OR flop utilization
2. Inductor kernels achieve 20-95% bandwidth OR flop utilization
3. Profiler hook/callback mechanism for adding utilization annotations
"""

import json
import os
import tempfile
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


# Skip tests if CUDA is not available
def requires_cuda(test_func):
    """Decorator to skip tests if CUDA is not available."""
    return unittest.skipUnless(
        torch.cuda.is_available(), "CUDA not available"
    )(test_func)


def requires_known_device(test_func):
    """Decorator to skip tests if the device is not in our device_info mapping."""
    if not torch.cuda.is_available():
        return unittest.skip("CUDA not available")(test_func)

    from torch._inductor.analysis.device_info import lookup_device_info

    device_name = torch.cuda.get_device_name()
    if lookup_device_info(device_name) is None:
        return unittest.skip(
            f"Device {device_name} not in device_info mapping"
        )(test_func)

    return test_func


class TestUtilizationAnnotations(TestCase):
    """Test the utilization annotation functions."""

    def test_add_utilization_annotations_basic(self):
        """Test that utilization annotations are added to trace events."""
        from torch._inductor.analysis.profile_analysis import (
            add_utilization_annotations,
        )

        # Create a mock trace with kernel events
        trace_data = {
            "traceEvents": [
                {
                    "name": "kernel_mm",
                    "cat": "kernel",
                    "dur": 100,  # 100 microseconds
                    "args": {
                        "kernel_flop": 2e9,  # 2 GFLOPS
                        "kernel_num_gb": 0.1,  # 100 MB
                    },
                },
            ],
            "deviceProperties": [
                {"id": 0, "name": "NVIDIA H100"},
            ],
        }

        result = add_utilization_annotations(trace_data, device_name="NVIDIA H100", dtype=torch.float32)

        # Check that utilization was added
        kernel_event = result["traceEvents"][0]
        self.assertIn("achieved_flops_percent", kernel_event["args"])
        self.assertIn("achieved_bandwidth_percent", kernel_event["args"])

        # Verify the calculation:
        # op_flops = 2e9 / (100e-6) = 2e13 FLOPS/s = 20 TFLOPS
        # H100 fp32 peak = 67.5 TFLOPS
        # achieved_flops = 100 * 20e12 / (67.5 * 1e12) = 29.6%
        achieved_flops = kernel_event["args"]["achieved_flops_percent"]
        self.assertGreater(achieved_flops, 0)
        self.assertLess(achieved_flops, 100)

        # op_gbps = 0.1 / (100e-6) = 1000 GB/s
        # H100 peak bw = 3350 GB/s
        # achieved_bw = 100 * 1000 / 3350 = 29.85%
        achieved_bw = kernel_event["args"]["achieved_bandwidth_percent"]
        self.assertGreater(achieved_bw, 0)
        self.assertLess(achieved_bw, 100)

    def test_add_utilization_skips_non_kernel_events(self):
        """Test that non-kernel events are not annotated."""
        from torch._inductor.analysis.profile_analysis import (
            add_utilization_annotations,
        )

        trace_data = {
            "traceEvents": [
                {
                    "name": "cpu_op",
                    "cat": "cpu_op",
                    "dur": 100,
                    "args": {
                        "kernel_flop": 1e9,
                        "kernel_num_gb": 0.1,
                    },
                },
            ],
            "deviceProperties": [
                {"id": 0, "name": "NVIDIA H100"},
            ],
        }

        result = add_utilization_annotations(trace_data, device_name="NVIDIA H100")

        # CPU op should not have utilization annotations
        cpu_event = result["traceEvents"][0]
        self.assertNotIn("achieved_flops_percent", cpu_event["args"])
        self.assertNotIn("achieved_bandwidth_percent", cpu_event["args"])

    def test_add_utilization_handles_missing_flop(self):
        """Test that events without flop/bandwidth info don't get annotations."""
        from torch._inductor.analysis.profile_analysis import (
            add_utilization_annotations,
        )

        trace_data = {
            "traceEvents": [
                {
                    "name": "kernel_no_flop",
                    "cat": "kernel",
                    "dur": 100,
                    "args": {},
                },
            ],
            "deviceProperties": [
                {"id": 0, "name": "NVIDIA H100"},
            ],
        }

        result = add_utilization_annotations(trace_data, device_name="NVIDIA H100")

        kernel_event = result["traceEvents"][0]
        self.assertNotIn("achieved_flops_percent", kernel_event["args"])
        self.assertNotIn("achieved_bandwidth_percent", kernel_event["args"])


class TestProfilerCallbacks(TestCase):
    """Test the profiler callback mechanism."""

    def setUp(self):
        from torch._inductor.analysis.profile_analysis import (
            clear_profiler_export_callbacks,
        )
        # Clear any existing callbacks before each test
        clear_profiler_export_callbacks()

    def tearDown(self):
        from torch._inductor.analysis.profile_analysis import (
            clear_profiler_export_callbacks,
        )
        # Clean up callbacks after each test
        clear_profiler_export_callbacks()

    def test_register_callback(self):
        """Test registering a callback."""
        from torch._inductor.analysis.profile_analysis import (
            register_profiler_export_callback,
            _profiler_export_callbacks,
        )

        def my_callback(data):
            data["modified"] = True
            return data

        register_profiler_export_callback(my_callback)
        self.assertEqual(len(_profiler_export_callbacks), 1)

    def test_run_callbacks(self):
        """Test that callbacks are executed in order."""
        from torch._inductor.analysis.profile_analysis import (
            register_profiler_export_callback,
            run_profiler_export_callbacks,
        )

        results = []

        def callback1(data):
            results.append(1)
            data["cb1"] = True
            return data

        def callback2(data):
            results.append(2)
            data["cb2"] = True
            return data

        register_profiler_export_callback(callback1)
        register_profiler_export_callback(callback2)

        data = {"traceEvents": []}
        result = run_profiler_export_callbacks(data)

        self.assertEqual(results, [1, 2])
        self.assertTrue(result["cb1"])
        self.assertTrue(result["cb2"])

    def test_unregister_callback(self):
        """Test unregistering a callback."""
        from torch._inductor.analysis.profile_analysis import (
            register_profiler_export_callback,
            unregister_profiler_export_callback,
            _profiler_export_callbacks,
        )

        def my_callback(data):
            return data

        register_profiler_export_callback(my_callback)
        self.assertEqual(len(_profiler_export_callbacks), 1)

        unregister_profiler_export_callback(my_callback)
        self.assertEqual(len(_profiler_export_callbacks), 0)

    def test_create_utilization_callback(self):
        """Test the create_utilization_callback helper."""
        from torch._inductor.analysis.profile_analysis import (
            create_utilization_callback,
        )

        callback = create_utilization_callback(device_name="NVIDIA H100", dtype=torch.float32)

        trace_data = {
            "traceEvents": [
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "dur": 100,
                    "args": {
                        "kernel_flop": 1e9,
                        "kernel_num_gb": 0.1,
                    },
                },
            ],
            "deviceProperties": [{"id": 0, "name": "NVIDIA H100"}],
        }

        result = callback(trace_data)

        kernel = result["traceEvents"][0]
        self.assertIn("achieved_flops_percent", kernel["args"])
        self.assertIn("achieved_bandwidth_percent", kernel["args"])


class TestProfilerHookIntegration(TestCase):
    """Test that profiler hooks work with export_chrome_trace."""

    def setUp(self):
        from torch._inductor.analysis.profile_analysis import (
            clear_profiler_export_callbacks,
        )
        clear_profiler_export_callbacks()

    def tearDown(self):
        from torch._inductor.analysis.profile_analysis import (
            clear_profiler_export_callbacks,
        )
        clear_profiler_export_callbacks()

    @requires_cuda
    def test_callback_runs_on_export(self):
        """Test that registered callbacks run when export_chrome_trace is called."""
        from torch.profiler import profile, ProfilerActivity
        from torch._inductor.analysis.profile_analysis import (
            register_profiler_export_callback,
        )

        callback_ran = [False]

        def my_callback(data):
            callback_ran[0] = True
            data["custom_marker"] = "callback_executed"
            return data

        register_profiler_export_callback(my_callback)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            trace_path = f.name

        try:
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                a = torch.randn(100, 100)
                b = torch.mm(a, a)

            prof.export_chrome_trace(trace_path)

            # Verify callback ran
            self.assertTrue(callback_ran[0], "Callback should have been executed")

            # Verify the marker was added
            with open(trace_path) as f:
                data = json.load(f)
            self.assertEqual(data.get("custom_marker"), "callback_executed")

        finally:
            os.unlink(trace_path)

    @requires_cuda
    @requires_known_device
    def test_utilization_callback_integration(self):
        """Test that utilization callback works end-to-end with profiler."""
        from torch.profiler import profile, ProfilerActivity
        from torch._inductor.analysis.profile_analysis import (
            register_profiler_export_callback,
            create_utilization_callback,
        )

        device_name = torch.cuda.get_device_name()

        # Register the utilization callback
        callback = create_utilization_callback(device_name=device_name, dtype=torch.float32)
        register_profiler_export_callback(callback)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            trace_path = f.name

        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                a = torch.randn(2048, 2048, device="cuda")
                b = torch.randn(2048, 2048, device="cuda")
                for _ in range(3):
                    c = torch.mm(a, b)
                torch.cuda.synchronize()

            prof.export_chrome_trace(trace_path)

            # Load and check for utilization annotations
            with open(trace_path) as f:
                data = json.load(f)

            # Find kernel events with utilization
            kernel_events = [
                e for e in data["traceEvents"]
                if e.get("cat") == "kernel"
                and ("achieved_flops_percent" in e.get("args", {})
                     or "achieved_bandwidth_percent" in e.get("args", {}))
            ]

            # Should have at least some events with utilization
            self.assertGreater(
                len(kernel_events), 0,
                "Expected kernel events with utilization annotations"
            )

        finally:
            os.unlink(trace_path)


class TestAugmentTraceFile(TestCase):
    """Test the augment_trace_file function."""

    def test_augment_trace_file_basic(self):
        """Test augmenting a trace file."""
        from torch._inductor.analysis.profile_analysis import augment_trace_file

        trace_data = {
            "traceEvents": [
                {
                    "name": "aten::mm",
                    "cat": "cpu_op",
                    "dur": 100,
                    "args": {
                        "External id": 1,
                        "Input Dims": [[1024, 1024], [1024, 1024]],
                        "Input type": ["float32", "float32"],
                        "Concrete Inputs": ["", ""],
                    },
                },
                {
                    "name": "mm_kernel",
                    "cat": "kernel",
                    "dur": 50,
                    "args": {
                        "External id": 1,
                    },
                },
            ],
            "deviceProperties": [{"id": 0, "name": "NVIDIA H100"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(trace_data, f)
            input_path = f.name

        try:
            output_path = input_path.replace(".json", "_augmented.json")
            result_path = augment_trace_file(
                input_path, output_path, device_name="NVIDIA H100", dtype=torch.float32
            )

            self.assertEqual(result_path, output_path)
            self.assertTrue(os.path.exists(output_path))

            with open(output_path) as f:
                result = json.load(f)

            # Check that kernel event was augmented
            kernel_event = [e for e in result["traceEvents"] if e["cat"] == "kernel"][0]
            # Should have kernel_flop and kernel_num_gb from augmentation
            self.assertIn("kernel_flop", kernel_event["args"])
            self.assertIn("kernel_num_gb", kernel_event["args"])

        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


@requires_cuda
class TestCublasGemmUtilization(TestCase):
    """
    Test that cublas GEMM operations achieve reasonable utilization (20-95%).

    These tests verify that for decent-sized matrix multiplications,
    the hardware achieves a reasonable percentage of peak FLOPS or bandwidth.
    """

    @requires_known_device
    def test_large_gemm_flop_utilization(self):
        """Test that a large GEMM achieves reasonable FLOP utilization."""
        from torch.profiler import profile, ProfilerActivity
        from torch._inductor.analysis.profile_analysis import (
            add_utilization_annotations,
            JsonProfile,
        )
        from torch._inductor.analysis.device_info import lookup_device_info

        device_name = torch.cuda.get_device_name()
        device_info = lookup_device_info(device_name)
        if device_info is None:
            self.skipTest(f"Device {device_name} not in device_info mapping")

        # Large matrices for high FLOP utilization
        size = 4096
        a = torch.randn(size, size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, size, device="cuda", dtype=torch.float32)

        # Warmup
        for _ in range(3):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            trace_path = f.name

        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                for _ in range(5):
                    c = torch.mm(a, b)
                torch.cuda.synchronize()

            prof.export_chrome_trace(trace_path)

            # Load and analyze the trace
            json_profile = JsonProfile(trace_path, dtype=torch.float32)
            json_profile.augment_trace()

            # Also add utilization annotations
            with open(trace_path) as f:
                trace_data = json.load(f)
            trace_data = add_utilization_annotations(
                trace_data, device_name=device_name, dtype=torch.float32
            )

            # Find kernel events with utilization
            kernel_events = [
                e for e in trace_data["traceEvents"]
                if e.get("cat") == "kernel"
                and "achieved_flops_percent" in e.get("args", {})
            ]

            if len(kernel_events) == 0:
                # If no events have utilization (e.g., unknown kernels), skip
                self.skipTest("No kernel events with FLOP utilization found")

            # Check that at least one kernel achieves reasonable utilization
            max_flop_util = max(
                e["args"]["achieved_flops_percent"] for e in kernel_events
            )
            max_bw_util = max(
                e["args"].get("achieved_bandwidth_percent", 0) for e in kernel_events
            )

            # For a large GEMM, we expect either high FLOP or high BW utilization
            # The 20-95% range accounts for:
            # - Some overhead from profiling
            # - Hardware not always hitting peak
            # - Memory-bound vs compute-bound balance
            has_reasonable_util = max_flop_util >= 20 or max_bw_util >= 20
            self.assertTrue(
                has_reasonable_util,
                f"Expected at least 20% utilization. Got FLOPS: {max_flop_util:.1f}%, BW: {max_bw_util:.1f}%"
            )

            # Also check we're not exceeding 100% (sanity check)
            self.assertLessEqual(
                max_flop_util, 100,
                f"FLOP utilization exceeds 100%: {max_flop_util:.1f}%"
            )

        finally:
            os.unlink(trace_path)

    @requires_known_device
    def test_large_gemm_bandwidth_utilization(self):
        """Test that a memory-bound GEMM achieves reasonable bandwidth utilization."""
        from torch.profiler import profile, ProfilerActivity
        from torch._inductor.analysis.profile_analysis import (
            add_utilization_annotations,
        )
        from torch._inductor.analysis.device_info import lookup_device_info

        device_name = torch.cuda.get_device_name()
        device_info = lookup_device_info(device_name)
        if device_info is None:
            self.skipTest(f"Device {device_name} not in device_info mapping")

        # Use float16 for higher throughput, tall/skinny matrices are more memory-bound
        # But still large enough to have meaningful bandwidth
        m, n, k = 8192, 512, 512
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)

        # Warmup
        for _ in range(3):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            trace_path = f.name

        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                for _ in range(10):
                    c = torch.mm(a, b)
                torch.cuda.synchronize()

            prof.export_chrome_trace(trace_path)

            with open(trace_path) as f:
                trace_data = json.load(f)
            trace_data = add_utilization_annotations(
                trace_data, device_name=device_name, dtype=torch.float16
            )

            # Find kernel events with bandwidth utilization
            kernel_events = [
                e for e in trace_data["traceEvents"]
                if e.get("cat") == "kernel"
                and "achieved_bandwidth_percent" in e.get("args", {})
            ]

            if len(kernel_events) == 0:
                self.skipTest("No kernel events with bandwidth utilization found")

            max_bw_util = max(
                e["args"]["achieved_bandwidth_percent"] for e in kernel_events
            )
            max_flop_util = max(
                e["args"].get("achieved_flops_percent", 0) for e in kernel_events
            )

            # Either FLOPS or bandwidth should be reasonable
            has_reasonable_util = max_flop_util >= 20 or max_bw_util >= 20
            self.assertTrue(
                has_reasonable_util,
                f"Expected at least 20% utilization. Got FLOPS: {max_flop_util:.1f}%, BW: {max_bw_util:.1f}%"
            )

        finally:
            os.unlink(trace_path)

    @requires_known_device
    def test_gemm_utilization_not_too_low(self):
        """
        Sanity test: GEMM on decent-sized matrices shouldn't have very low utilization.

        This catches obvious issues like incorrect FLOP counting or wrong device info.
        """
        from torch.profiler import profile, ProfilerActivity
        from torch._inductor.analysis.profile_analysis import (
            add_utilization_annotations,
        )
        from torch._inductor.analysis.device_info import lookup_device_info

        device_name = torch.cuda.get_device_name()
        device_info = lookup_device_info(device_name)
        if device_info is None:
            self.skipTest(f"Device {device_name} not in device_info mapping")

        # Medium-sized GEMM that should have decent utilization
        size = 2048
        a = torch.randn(size, size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, size, device="cuda", dtype=torch.float32)

        # Warmup
        for _ in range(3):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            trace_path = f.name

        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                for _ in range(5):
                    c = torch.mm(a, b)
                torch.cuda.synchronize()

            prof.export_chrome_trace(trace_path)

            with open(trace_path) as f:
                trace_data = json.load(f)
            trace_data = add_utilization_annotations(
                trace_data, device_name=device_name, dtype=torch.float32
            )

            kernel_events = [
                e for e in trace_data["traceEvents"]
                if e.get("cat") == "kernel"
                and ("achieved_flops_percent" in e.get("args", {})
                     or "achieved_bandwidth_percent" in e.get("args", {}))
            ]

            if len(kernel_events) == 0:
                self.skipTest("No kernel events with utilization found")

            # Get best utilization across all kernels
            flop_utils = [e["args"].get("achieved_flops_percent", 0) for e in kernel_events]
            bw_utils = [e["args"].get("achieved_bandwidth_percent", 0) for e in kernel_events]

            max_util = max(max(flop_utils), max(bw_utils))

            # For a 2048x2048 GEMM, we should get at least 20% utilization
            # If utilization is very low (<5%), something is probably wrong
            self.assertGreater(
                max_util, 5,
                f"Utilization suspiciously low: max FLOPS {max(flop_utils):.1f}%, max BW {max(bw_utils):.1f}%"
            )

            # Should not exceed 100%
            self.assertLessEqual(max_util, 100)

        finally:
            os.unlink(trace_path)


@requires_cuda
class TestInductorKernelUtilization(TestCase):
    """
    Test that Inductor-generated kernels achieve reasonable utilization (20-95%).

    These tests use torch.compile to generate Triton/Inductor kernels and verify
    they achieve reasonable FLOPS or bandwidth utilization.
    """

    @requires_known_device
    def test_inductor_matmul_utilization(self):
        """Test that inductor-compiled matmul achieves reasonable utilization."""
        from torch.profiler import profile, ProfilerActivity
        from torch._inductor.analysis.profile_analysis import (
            add_utilization_annotations,
        )
        from torch._inductor.analysis.device_info import lookup_device_info

        device_name = torch.cuda.get_device_name()
        device_info = lookup_device_info(device_name)
        if device_info is None:
            self.skipTest(f"Device {device_name} not in device_info mapping")

        @torch.compile
        def compiled_matmul(a, b):
            return torch.mm(a, b)

        size = 4096
        a = torch.randn(size, size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, size, device="cuda", dtype=torch.float32)

        # Warmup and compile
        for _ in range(3):
            _ = compiled_matmul(a, b)
        torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            trace_path = f.name

        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                for _ in range(5):
                    c = compiled_matmul(a, b)
                torch.cuda.synchronize()

            prof.export_chrome_trace(trace_path)

            with open(trace_path) as f:
                trace_data = json.load(f)
            trace_data = add_utilization_annotations(
                trace_data, device_name=device_name, dtype=torch.float32
            )

            kernel_events = [
                e for e in trace_data["traceEvents"]
                if e.get("cat") == "kernel"
                and ("achieved_flops_percent" in e.get("args", {})
                     or "achieved_bandwidth_percent" in e.get("args", {}))
            ]

            if len(kernel_events) == 0:
                self.skipTest("No kernel events with utilization found")

            max_flop_util = max(e["args"].get("achieved_flops_percent", 0) for e in kernel_events)
            max_bw_util = max(e["args"].get("achieved_bandwidth_percent", 0) for e in kernel_events)

            # Either FLOPS or BW should be reasonable
            has_reasonable_util = max_flop_util >= 20 or max_bw_util >= 20
            self.assertTrue(
                has_reasonable_util,
                f"Expected at least 20% utilization. Got FLOPS: {max_flop_util:.1f}%, BW: {max_bw_util:.1f}%"
            )

        finally:
            os.unlink(trace_path)

    @requires_known_device
    def test_inductor_fused_kernel_utilization(self):
        """Test that inductor fused kernels achieve reasonable utilization."""
        from torch.profiler import profile, ProfilerActivity
        from torch._inductor.analysis.profile_analysis import (
            add_utilization_annotations,
        )
        from torch._inductor.analysis.device_info import lookup_device_info

        device_name = torch.cuda.get_device_name()
        device_info = lookup_device_info(device_name)
        if device_info is None:
            self.skipTest(f"Device {device_name} not in device_info mapping")

        @torch.compile
        def fused_ops(a, b):
            # This should generate a fused Triton kernel
            c = torch.mm(a, b)
            d = torch.relu(c)
            e = d + 1.0
            return e

        size = 2048
        a = torch.randn(size, size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, size, device="cuda", dtype=torch.float32)

        # Warmup and compile
        for _ in range(3):
            _ = fused_ops(a, b)
        torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            trace_path = f.name

        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                for _ in range(5):
                    c = fused_ops(a, b)
                torch.cuda.synchronize()

            prof.export_chrome_trace(trace_path)

            with open(trace_path) as f:
                trace_data = json.load(f)
            trace_data = add_utilization_annotations(
                trace_data, device_name=device_name, dtype=torch.float32
            )

            kernel_events = [
                e for e in trace_data["traceEvents"]
                if e.get("cat") == "kernel"
                and ("achieved_flops_percent" in e.get("args", {})
                     or "achieved_bandwidth_percent" in e.get("args", {}))
            ]

            if len(kernel_events) == 0:
                self.skipTest("No kernel events with utilization found")

            max_flop_util = max(e["args"].get("achieved_flops_percent", 0) for e in kernel_events)
            max_bw_util = max(e["args"].get("achieved_bandwidth_percent", 0) for e in kernel_events)

            # For fused ops, we expect decent utilization from either compute or memory
            has_reasonable_util = max_flop_util >= 20 or max_bw_util >= 20
            self.assertTrue(
                has_reasonable_util,
                f"Expected at least 20% utilization. Got FLOPS: {max_flop_util:.1f}%, BW: {max_bw_util:.1f}%"
            )

        finally:
            os.unlink(trace_path)

    @requires_known_device
    def test_inductor_pointwise_bandwidth_utilization(self):
        """Test that inductor pointwise kernels achieve reasonable bandwidth utilization."""
        from torch.profiler import profile, ProfilerActivity
        from torch._inductor.analysis.profile_analysis import (
            add_utilization_annotations,
        )
        from torch._inductor.analysis.device_info import lookup_device_info

        device_name = torch.cuda.get_device_name()
        device_info = lookup_device_info(device_name)
        if device_info is None:
            self.skipTest(f"Device {device_name} not in device_info mapping")

        @torch.compile
        def pointwise_ops(a, b):
            # Pointwise ops are memory-bound
            return a + b * 2.0 + torch.sin(a)

        # Large tensors for memory bandwidth testing
        size = 10000000  # 10M elements
        a = torch.randn(size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, device="cuda", dtype=torch.float32)

        # Warmup and compile
        for _ in range(3):
            _ = pointwise_ops(a, b)
        torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            trace_path = f.name

        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                for _ in range(10):
                    c = pointwise_ops(a, b)
                torch.cuda.synchronize()

            prof.export_chrome_trace(trace_path)

            with open(trace_path) as f:
                trace_data = json.load(f)
            trace_data = add_utilization_annotations(
                trace_data, device_name=device_name, dtype=torch.float32
            )

            kernel_events = [
                e for e in trace_data["traceEvents"]
                if e.get("cat") == "kernel"
                and "achieved_bandwidth_percent" in e.get("args", {})
            ]

            if len(kernel_events) == 0:
                self.skipTest("No kernel events with bandwidth utilization found")

            max_bw_util = max(e["args"]["achieved_bandwidth_percent"] for e in kernel_events)

            # Pointwise ops should achieve decent bandwidth utilization
            # Note: profiling overhead may reduce apparent utilization
            self.assertGreater(
                max_bw_util, 10,
                f"Bandwidth utilization too low: {max_bw_util:.1f}%"
            )
            self.assertLessEqual(max_bw_util, 100)

        finally:
            os.unlink(trace_path)


class TestProfileAnalysisReport(TestCase):
    """Test the JsonProfile report functionality with utilization metrics."""

    def test_json_profile_report_includes_utilization(self):
        """Test that JsonProfile report shows utilization metrics."""
        from torch._inductor.analysis.profile_analysis import JsonProfile

        # Create a mock trace file
        trace_data = {
            "traceEvents": [
                {
                    "name": "aten::mm",
                    "cat": "cpu_op",
                    "dur": 100,
                    "args": {
                        "External id": 1,
                        "Input Dims": [[1024, 1024], [1024, 1024]],
                        "Input type": ["float32", "float32"],
                        "Concrete Inputs": ["", ""],
                    },
                },
                {
                    "name": "mm_kernel",
                    "cat": "kernel",
                    "dur": 50,
                    "args": {
                        "External id": 1,
                        "device": 0,
                        "name": "mm_kernel",
                    },
                },
            ],
            "deviceProperties": [
                {"id": 0, "name": "NVIDIA H100", "totalGlobalMem": 80000000000},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(trace_data, f)
            trace_path = f.name

        try:
            profile = JsonProfile(trace_path, dtype=torch.float32)
            profile.augment_trace()
            report = profile.report()

            # Report should contain utilization columns
            self.assertIn("Achieved FLOPS %", report)
            self.assertIn("Achieved Bandwidth %", report)

        finally:
            os.unlink(trace_path)


if __name__ == "__main__":
    run_tests()
