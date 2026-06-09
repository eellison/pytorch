# Owner(s): ["module: inductor"]

import functools
import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import skipUnless
from unittest.mock import MagicMock, Mock, patch

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters
from torch._inductor import config, metrics
from torch._inductor.runtime.triton_compat import HAS_WARP_SPEC, OutOfResources
from torch._inductor.utils import clone_preserve_strides
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    runOnRocm,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    HAS_GPU_AND_TRITON,
    requires_gpu_with_enough_memory,
)


try:
    import triton  # @manual
    import triton.language as tl  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904

from torch._inductor.codegen.common import REMOVED
from torch._inductor.codegen.triton_combo_kernel import ComboKernel
from torch._inductor.runtime.hints import (
    AttrsDescriptorWrapper,
    AutotuneHint,
    DeviceProperties,
    HeuristicType,
    TRITON_MAX_BLOCK,
)
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_heuristics import (
    autotune_hints_to_configs,
    CachingAutotuner,
    cached_autotune,
    hash_configs,
    template,
    triton_config,
)
from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner
from torch._inductor.test_case import run_tests, TestCase



@triton.jit
def amd_sqr_kernel(in_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data = tl.load(in_ptr + offsets, mask=offsets < numel)
    sqr = data * data
    tl.store(out_ptr + offsets, sqr, mask=offsets < numel)


@functools.lru_cache
def get_autotuned_amd_sqr_kernel():
    return triton.autotune(
        configs=[
            triton.Config(
                {
                    "BLOCK_SIZE": 64,
                    "waves_per_eu": 3,
                }
            )
        ],
        key=[],
    )(amd_sqr_kernel)


@instantiate_parametrized_tests
class TestTritonHeuristics(TestCase):
    device_type = GPU_TYPE

    def test_triton_config(self):
        """
        Make sure block size does not exceed the maximum defined in inductor config.
        """
        cfg = triton_config({"x": 2048, "y": 2}, 64, 64)
        for label in "XYZ":
            key = f"{label}BLOCK"
            if key not in cfg.kwargs:
                continue
            self.assertTrue(cfg.kwargs[key] <= TRITON_MAX_BLOCK[label])

    def _test_artificial_zgrid(self):
        def forward(primals_1, primals_2, primals_5):
            view = torch.ops.aten.reshape.default(primals_5, [-1, 2, 4])
            primals_5 = None
            permute = torch.ops.aten.permute.default(view, [0, 2, 1])
            clone = torch.ops.aten.clone.default(
                permute, memory_format=torch.contiguous_format
            )
            permute = None
            view_1 = torch.ops.aten.reshape.default(clone, [-1, 4])
            clone = None
            permute_1 = torch.ops.aten.permute.default(primals_1, [1, 0])
            primals_1 = None
            addmm = torch.ops.aten.addmm.default(primals_2, view_1, permute_1)
            primals_2 = None
            return addmm

        s0 = 16777472
        s1 = 8

        args = [
            torch.rand([2, 4], device=GPU_TYPE),
            torch.rand([2], device=GPU_TYPE),
            torch.rand([s0, s1], device=GPU_TYPE),
        ]
        torch._dynamo.mark_dynamic(args[-1], 0)
        foo_c = torch.compile(forward)

        self.assertEqual(forward(*args), foo_c(*args))

        args = [
            torch.rand([2, 4], device=GPU_TYPE),
            torch.rand([2], device=GPU_TYPE),
            torch.rand([s0, s1], device=GPU_TYPE),
        ]
        self.assertEqual(forward(*args), foo_c(*args))

    def test_artificial_zgrid(self):
        self._test_artificial_zgrid()

    @config.patch("cpp_wrapper", True)
    def test_artificial_grid_cpp_wrapper(self):
        self._test_artificial_zgrid()

    @staticmethod
    def _get_cos_kernel_caching_autotuner_args():
        @triton.jit
        def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
            xnumel = 16
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:]
            xmask = xindex < xnumel
            x0 = xindex
            tmp0 = tl.load(in_ptr0 + (x0), xmask)
            tmp1 = tl_math.cos(tmp0)
            tl.store(out_ptr0 + (x0), tmp1, xmask)

        triton_meta = {
            "signature": {"in_ptr0": "*fp32", "out_ptr0": "*fp32", "xnumel": "i32"},
            "device": DeviceProperties.create(torch.device(GPU_TYPE)),
            "constants": {},
            "configs": [
                AttrsDescriptorWrapper(divisible_by_16=(0, 1, 2), equal_to_1=())
            ],
        }

        configs = [
            triton_config({"x": 16}, 64),
            triton_config({"x": 256}, 64),
        ]

        inductor_meta = {}

        return {
            "fn": triton_,
            "triton_meta": triton_meta,
            "configs": configs,
            "save_cache_hook": False,
            "mutated_arg_names": [],
            "reset_to_zero_arg_names": [],
            "optimize_mem": True,
            "heuristic_type": HeuristicType.POINTWISE,
            "inductor_meta": inductor_meta,
        }

    def test_pre_hook_assert(self):
        # assert if any of the configs passed to the CachingAutotuner have pre-hooks
        args = self._get_cos_kernel_caching_autotuner_args()

        def pre_hook(kwargs):
            if "in_ptr0" in kwargs:
                kwargs["in_ptr0"].zero_()

        for cfg in args["configs"]:
            cfg.pre_hook = pre_hook

        with self.assertRaisesRegex(AssertionError, "pre_hook"):
            CachingAutotuner(**args)

    def test_coordinate_descent_batch_inactive_fallback_uses_scalar_benchmarking(
        self,
    ):
        order = []

        class Launcher:
            config = triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1)
            cache_hash = "hash"
            store_cubin = False

            def __call__(self, *args, stream, **kwargs):
                order.append(("launch", stream))

        launcher = Launcher()
        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(__name__="kernel")
        autotuner.triton_meta = {"signature": {}}
        autotuner.inductor_meta = {
            "coordinate_descent_tuning": True,
            "coordinate_descent_tuning_batch": True,
            "coordinate_descent_tuning_batch_policy": "all",
        }
        autotuner.device_props = SimpleNamespace(type="cpu")
        autotuner.triton_interpret = False
        autotuner.configs = None
        autotuner.launchers = [launcher]
        autotuner._cached_launcher = None
        autotuner._cache_eligible = False
        autotuner._coordinate_descent_batch_enabled = lambda: True
        autotuner._recheck_coordesc_cache_before_runtime_tuning = (
            lambda: order.append("recheck")
        )

        def prepare_for_benchmark(*args, **kwargs):
            runtime_coordesc_cache_recheck = kwargs.get(
                "runtime_coordesc_cache_recheck"
            )
            order.append(("prepare", runtime_coordesc_cache_recheck))
            if runtime_coordesc_cache_recheck:
                autotuner._recheck_coordesc_cache_before_runtime_tuning()

        autotuner.prepare_for_benchmark = prepare_for_benchmark
        autotuner._pre_launch = lambda *args, **kwargs: order.append("pre")
        autotuner._post_launch = lambda: order.append("post")

        def coordinate_descent_tuning(launcher, *args, **kwargs):
            order.append(("coordesc", kwargs.get("use_batch_benchmarking")))
            launcher.config.found_by_coordesc = True
            return launcher

        autotuner.coordinate_descent_tuning = coordinate_descent_tuning

        with (
            patch(
                "torch._inductor.runtime.triton_heuristics.TritonBundler.put_winner"
            ),
            patch(
                "torch._inductor.runtime.triton_heuristics.triton.set_allocator",
                create=True,
            ),
        ):
            autotuner.run("arg", stream=17)

        self.assertEqual(
            order,
            [
                ("prepare", True),
                "recheck",
                ("coordesc", False),
                "pre",
                ("launch", 17),
                "post",
            ],
        )

    def test_coordinate_descent_batch_skips_benchmark_run(self):
        order = []
        test_case = self

        class Launcher:
            config = triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1)
            cache_hash = "hash"
            store_cubin = False

            def __call__(self, *args, stream, **kwargs):
                order.append(("launch", stream))

        class Batch:
            disposable_args = True

            def enqueue(self, *args, **kwargs):
                test_case.fail("benchmark_run should not enqueue coordesc")

        launcher = Launcher()
        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(__name__="kernel")
        autotuner.triton_meta = {"signature": {}}
        autotuner.inductor_meta = {
            "coordinate_descent_tuning": True,
            "coordinate_descent_tuning_batch": True,
            "coordinate_descent_tuning_batch_policy": "all",
        }
        autotuner.device_props = SimpleNamespace(type="cpu")
        autotuner.triton_interpret = False
        autotuner.configs = None
        autotuner.launchers = [launcher]
        autotuner._cached_launcher = None
        autotuner._cache_eligible = False
        autotuner._pre_launch = lambda *args, **kwargs: order.append("pre")
        autotuner._post_launch = lambda: order.append("post")

        with (
            patch(
                "torch._inductor.runtime.triton_heuristics.get_active_autotune_queue",
                return_value=Batch(),
            ),
            patch(
                "torch._inductor.runtime.triton_heuristics.TritonBundler.put_winner"
            ),
            patch(
                "torch._inductor.runtime.triton_heuristics.triton.set_allocator",
                create=True,
            ),
        ):
            autotuner.run("arg", stream=11, benchmark_run=True)

        self.assertEqual(order, ["pre", ("launch", 11), "post"])

    def test_autotune_to_one_config_can_skip_cache_save(self):
        class Launcher:
            def __init__(self, name):
                self.name = name
                self.config = triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1)
                self.cache_hash = name
                self.n_regs = 0
                self.n_spills = 0
                self.shared = 0

        slow = Launcher("slow")
        fast = Launcher("fast")
        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(__name__="kernel")
        autotuner.inductor_meta = {
            "coordinate_descent_tuning": True,
            "coordinate_descent_tuning_batch": True,
            "coordinate_descent_tuning_batch_policy": "all",
        }
        autotuner.launchers = [slow, fast]
        autotuner.precompile_time_taken_ns = 0
        autotuner.benchmark_failure_reasons = {}
        autotuner.benchmark_all_configs = lambda *args, **kwargs: {
            slow: 2.0,
            fast: 1.0,
        }
        autotuner.save_cache_hook = MagicMock(
            side_effect=AssertionError("benchmark_run should not save autotune cache")
        )

        with patch(
            "torch._inductor.runtime.triton_heuristics.TritonBundler.put_winner"
        ) as put_winner:
            autotuner.autotune_to_one_config("arg", save_cache=False)

        self.assertEqual(autotuner.launchers, [fast])
        put_winner.assert_called_once_with("fast")
        autotuner.save_cache_hook.assert_not_called()

    def test_autotune_to_one_config_does_not_save_pre_coordesc_as_coordesc(self):
        class Launcher:
            def __init__(self, name):
                self.name = name
                self.config = triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1)
                self.cache_hash = name
                self.n_regs = 0
                self.n_spills = 0
                self.shared = 0

        slow = Launcher("slow")
        fast = Launcher("fast")
        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(__name__="kernel")
        autotuner.inductor_meta = {
            "coordinate_descent_tuning": True,
            "coordinate_descent_tuning_batch": True,
            "coordinate_descent_tuning_batch_policy": "all",
        }
        autotuner.launchers = [slow, fast]
        autotuner.precompile_time_taken_ns = 0
        autotuner.benchmark_failure_reasons = {}
        autotuner.benchmark_all_configs = lambda *args, **kwargs: {
            slow: 2.0,
            fast: 1.0,
        }
        autotuner.save_cache_hook = MagicMock()

        with patch(
            "torch._inductor.runtime.triton_heuristics.TritonBundler.put_winner"
        ):
            autotuner.autotune_to_one_config("arg")

        autotuner.save_cache_hook.assert_called_once()
        kwargs = autotuner.save_cache_hook.call_args.kwargs
        self.assertFalse(kwargs["found_by_coordesc"])
        self.assertFalse(kwargs["coordinate_descent_tuning_batch"])
        self.assertIsNone(kwargs["coordinate_descent_tuning_batch_policy"])

    @staticmethod
    def _coordesc_batch_patch(**overrides):
        config_values = {
            "compile_threads": 2,
            "coordinate_descent_tuning": True,
            "coordinate_descent_tuning_batch": True,
        }
        config_values.update(overrides)
        return config.patch(config_values)

    def test_finish_coordinate_descent_tuning_clears_process_pool_submitter(self):
        config0 = triton.Config({"XBLOCK": 1}, num_warps=4, num_stages=1)
        launcher = SimpleNamespace(
            config=config0,
            cache_hash="winner-hash",
        )
        autotuner = object.__new__(CachingAutotuner)
        autotuner.autotune_time_taken_ns = 0
        autotuner.fn = SimpleNamespace(src="def kernel(): pass")
        autotuner.inductor_meta = {
            "coordinate_descent_tuning_batch_policy": "auto",
        }
        autotuner.save_cache_hook = None
        autotuner.size_hints = {"x": 1}
        autotuner._config_compile_submitter = Mock()

        winner = autotuner._finish_coordinate_descent_tuning(
            config0,
            {config0: launcher},
            coordesc_time_taken_ns=1,
            save_cache=False,
        )

        self.assertIs(winner, launcher)
        self.assertIsNone(autotuner._config_compile_submitter)

    def test_sync_triton_deferred_static_precompile_gets_process_pool_submitter(self):
        from torch._inductor.async_compile import AsyncCompile, CompiledTritonKernels

        source = "@triton.jit\ndef kernel_sync_deferred():\n    pass\n"
        precompile_calls = []
        kernel = object.__new__(CachingAutotuner)
        kernel._config_compile_submitter = None
        kernel._static_config_compile_submitter = None
        kernel._static_triton_bundle_key = None
        kernel.set_compile_info = lambda compile_id, is_backward: None
        kernel.autotune_cache_info = {}

        def precompile(**kwargs):
            precompile_calls.append(kwargs)
            self.assertIsNone(kernel._config_compile_submitter)
            self.assertIsNotNone(kernel._static_config_compile_submitter)
            self.assertIsNotNone(kernel._static_triton_bundle_key)

        kernel.precompile = precompile
        pool = SimpleNamespace(submit=Mock(return_value="config-future"))
        candidate_config = triton.Config({"XBLOCK": 2}, num_warps=4, num_stages=1)

        try:
            with (
                config.patch({"compile_threads": 2}),
                patch.object(AsyncCompile, "use_process_pool", return_value=False),
                patch.object(AsyncCompile, "process_pool", return_value=pool),
                patch(
                    "torch._inductor.async_compile._load_triton_kernel_from_source",
                    return_value=kernel,
                ),
                patch(
                    "torch._inductor.runtime.autotune_common._should_defer_static_autotune_precompile",
                    return_value=True,
                ),
            ):
                result = AsyncCompile().triton("kernel_sync_deferred", source)
                self.assertEqual(
                    result._static_config_compile_submitter([candidate_config]),
                    "config-future",
                )

            self.assertIs(result, kernel)
            self.assertIsNone(result._config_compile_submitter)
            self.assertIsNotNone(result._static_config_compile_submitter)
            self.assertEqual(
                precompile_calls,
                [
                    {
                        "warm_cache_only": False,
                        "static_triton_bundle_key": None,
                        "max_configs": 1,
                    }
                ],
            )
            pool.submit.assert_called_once()
        finally:
            CompiledTritonKernels.remove_future(source)

    def test_triton_future_result_is_idempotent_for_deferred_static(self):
        from torch._inductor.async_compile import AsyncCompile, CompiledTritonKernels

        class Future:
            def __init__(self, result):
                self._result = result
                self.result_calls = 0

            def result(self, timeout=None):
                self.result_calls += 1
                return self._result

        source = "@triton.jit\ndef kernel_idempotent():\n    pass\n"
        precompile_calls = []
        restore_calls = []
        kernel = object.__new__(CachingAutotuner)
        kernel.set_compile_info = lambda compile_id, is_backward: None
        kernel.restore_after_unpickle = lambda old_values: restore_calls.append(
            old_values
        )
        kernel.precompile = lambda **kwargs: precompile_calls.append(kwargs)
        kernel.autotune_cache_info = {}
        task = Future((kernel, 1))
        pool = SimpleNamespace(submit=Mock(return_value=task))

        try:
            with (
                config.patch({"compile_threads": 2}),
                patch.object(AsyncCompile, "use_process_pool", return_value=True),
                patch.object(AsyncCompile, "process_pool", return_value=pool),
                patch(
                    "torch._inductor.runtime.autotune_common._has_deferred_static_autotune_precompile",
                    return_value=True,
                ),
            ):
                future = AsyncCompile().triton("kernel_idempotent", source)
                first = future.result()
                second = future.result()

            self.assertIs(first, second)
            self.assertEqual(task.result_calls, 1)
            self.assertEqual(restore_calls, [None])
            self.assertEqual(len(precompile_calls), 1)
            self.assertIsNone(CompiledTritonKernels.get(source))
        finally:
            CompiledTritonKernels.remove_future(source)

    def test_caching_state_strips_process_pool_compile_submitters(self):
        autotuner = object.__new__(CachingAutotuner)
        autotuner.launchers = []
        autotuner.compile_results = []
        autotuner._config_compile_submitter = lambda configs: None
        autotuner._static_config_compile_submitter = lambda configs: None
        autotuner._static_triton_bundle_key = "static-key"
        autotuner.lock = object()

        state = autotuner.__getstate__()
        self.assertIsNone(state["_config_compile_submitter"])
        self.assertIsNone(state["_static_config_compile_submitter"])
        self.assertIsNone(state["_static_triton_bundle_key"])

        autotuner.prepare_for_caching()
        self.assertIsNone(autotuner._config_compile_submitter)
        self.assertIsNone(autotuner._static_config_compile_submitter)
        self.assertIsNone(autotuner._static_triton_bundle_key)

    def test_partial_precompile_skips_invalid_until_first_valid_config(self):
        bad_config = triton.Config({"XBLOCK": 1}, num_warps=4, num_stages=1)
        good_config = triton.Config({"XBLOCK": 2}, num_warps=4, num_stages=1)
        later_config = triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1)
        autotuner = object.__new__(CachingAutotuner)
        autotuner.compile_results = []
        autotuner.launchers = []
        autotuner.configs = [bad_config, good_config, later_config]

        def precompile_config(config):
            if config is bad_config:
                raise OutOfResources(2, 1, "shared memory")
            return SimpleNamespace(
                config=config,
                kernel=SimpleNamespace(hash=f"hash-{config.kwargs['XBLOCK']}"),
            )

        autotuner._precompile_config = precompile_config

        autotuner._precompile_worker(max_configs=1)

        self.assertEqual(
            [result.config for result in autotuner.compile_results],
            [good_config],
        )
        self.assertEqual(autotuner.configs, [later_config])

    def test_benchmark_clone_args_does_not_clone_unmutated_args(self):
        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(arg_names=["inp", "out"])
        autotuner.inductor_meta = {}
        autotuner.mutated_arg_names = []

        inp = torch.ones(2)
        out = torch.zeros(2)
        cloned_args, cloned_kwargs = autotuner.maybe_clone_args(set(), inp, out)

        self.assertEqual(cloned_kwargs, {})
        self.assertIs(cloned_args[0], inp)
        self.assertIs(cloned_args[1], out)

    def test_benchmark_clone_args_clones_mutated_inputs(self):
        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(arg_names=["inp", "out"])
        autotuner.inductor_meta = {
            "mutated_input_arg_names": ["inp"],
        }
        autotuner.mutated_arg_names = []

        inp = torch.ones(2)
        out = torch.zeros(2)
        cloned_args, cloned_kwargs = autotuner.maybe_clone_args(set(), inp, out)

        self.assertEqual(cloned_kwargs, {})
        self.assertIsNot(cloned_args[0], inp)
        self.assertIs(cloned_args[1], out)
        cloned_args[0].add_(1)
        self.assertEqual(inp, torch.ones_like(inp))

    def test_benchmark_clone_args_clones_reset_to_zero_workspace(self):
        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(arg_names=["inp", "workspace"])
        autotuner.inductor_meta = {
            "mutated_input_arg_names": [],
        }
        autotuner.mutated_arg_names = []
        autotuner.reset_to_zero_arg_names = ["workspace"]

        inp = torch.ones(2)
        workspace = torch.zeros(2)
        cloned_args, cloned_kwargs = autotuner.maybe_clone_args(
            set(), inp, workspace
        )

        self.assertEqual(cloned_kwargs, {})
        self.assertIs(cloned_args[0], inp)
        self.assertIsNot(cloned_args[1], workspace)
        cloned_args[1].add_(1)
        self.assertEqual(workspace, torch.zeros_like(workspace))

    def test_legacy_benchmark_clone_args_clones_mutated_args(self):
        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(arg_names=["inp", "out"])
        autotuner.inductor_meta = {}
        autotuner.mutated_arg_names = ["inp"]

        inp = torch.ones(2)
        out = torch.zeros(2)
        cloned_args, cloned_kwargs = autotuner.maybe_clone_args(set(), inp, out)

        self.assertEqual(cloned_kwargs, {})
        self.assertIsNot(cloned_args[0], inp)
        self.assertIs(cloned_args[1], out)
        cloned_args[0].add_(1)
        self.assertEqual(inp, torch.ones_like(inp))

    def test_combo_kernel_mutated_input_args_include_inplace_inputs(self):
        combo = object.__new__(ComboKernel)
        combo.sub_kernels = [
            SimpleNamespace(
                mutations=["inplace"],
                removed_buffers=set(),
                args=SimpleNamespace(
                    input_buffers={"inplace": "in_out_ptr0"},
                    inplace_buffers={},
                    output_buffers={"out": "out_ptr0", "removed": REMOVED},
                ),
            ),
            SimpleNamespace(
                mutations=[],
                removed_buffers=set(),
                args=SimpleNamespace(
                    input_buffers={},
                    inplace_buffers={},
                    output_buffers={"out": "out_ptr1"},
                ),
            ),
        ]

        self.assertEqual(
            combo.get_mutated_input_args_sub_kernels(),
            ["in_out_ptr0"],
        )

    @skipUnless(HAS_GPU, "requires GPU")
    def test_copy_args_to_cpu_excludes_unmutated_args(self):
        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(arg_names=["inp", "out"])
        autotuner.inductor_meta = {}
        autotuner.mutated_arg_names = []
        autotuner.optimize_mem = True

        inp = torch.ones(8, device=GPU_TYPE)
        out = torch.zeros(8, device=GPU_TYPE)

        with (
            patch("torch.accelerator.current_accelerator", return_value=object()),
            patch("torch.accelerator.max_memory_allocated", return_value=0),
            patch("torch.accelerator.memory_allocated", return_value=0),
        ):
            cpu_copies = autotuner.copy_args_to_cpu_if_needed(inp, out)

        self.assertEqual(cpu_copies, {})

    @skipUnless(HAS_GPU, "requires GPU")
    def test_copy_args_to_cpu_includes_mutated_inputs(self):
        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(arg_names=["inp", "out"])
        autotuner.inductor_meta = {
            "mutated_input_arg_names": ["inp"],
        }
        autotuner.mutated_arg_names = []
        autotuner.optimize_mem = True

        inp = torch.ones(8, device=GPU_TYPE)
        out = torch.zeros(8, device=GPU_TYPE)

        with (
            patch("torch.accelerator.current_accelerator", return_value=object()),
            patch("torch.accelerator.max_memory_allocated", return_value=0),
            patch("torch.accelerator.memory_allocated", return_value=0),
        ):
            cpu_copies = autotuner.copy_args_to_cpu_if_needed(inp, out)

        self.assertEqual(list(cpu_copies), ["inp"])
        self.assertIs(cpu_copies["inp"][0], inp)

    def test_prepare_for_benchmark_runs_coordesc_without_cache(self):
        class Launcher:
            def __init__(self, name):
                self.name = name
                self.config = triton.Config(
                    {"XBLOCK": 4}, num_warps=4, num_stages=1
                )

        base = Launcher("base")
        tuned = Launcher("tuned")
        calls = []
        autotuner = object.__new__(CachingAutotuner)
        autotuner.launchers = [base]
        autotuner.inductor_meta = {"coordinate_descent_tuning": True}

        def coordinate_descent_tuning(launcher, *args, save_cache=True, **kwargs):
            calls.append((launcher.name, args, save_cache, kwargs))
            tuned.config.found_by_coordesc = True
            return tuned

        autotuner.coordinate_descent_tuning = coordinate_descent_tuning

        result = autotuner.prepare_for_benchmark(
            "arg",
            coordinate_descent=True,
            save_cache=False,
            kw="value",
        )

        self.assertIs(result, tuned)
        self.assertEqual(autotuner.launchers, [tuned])
        self.assertEqual(calls, [("base", ("arg",), False, {"kw": "value"})])

    def test_prepare_for_benchmark_installs_allocator_before_tuning(self):
        class Launcher:
            def __init__(self):
                self.config = triton.Config(
                    {"XBLOCK": 4}, num_warps=4, num_stages=1
                )

        launcher = Launcher()
        calls = []
        autotuner = object.__new__(CachingAutotuner)
        autotuner.launchers = [launcher]
        autotuner.device_props = SimpleNamespace(type="cpu")
        autotuner.inductor_meta = {"coordinate_descent_tuning": True}

        def coordinate_descent_tuning(launcher, *args, save_cache=True, **kwargs):
            calls.append("coordesc")
            return launcher

        autotuner.coordinate_descent_tuning = coordinate_descent_tuning

        def set_allocator(alloc_fn):
            calls.append("allocator")

        with patch(
            "torch._inductor.runtime.triton_heuristics.triton.set_allocator",
            side_effect=set_allocator,
            create=True,
        ):
            autotuner.prepare_for_benchmark("arg", coordinate_descent=True)

        self.assertEqual(calls, ["allocator", "coordesc"])

    def test_prepare_for_benchmark_forwards_save_cache_to_combo(self):
        class Launcher:
            def __init__(self):
                self.config = triton.Config(
                    {"XBLOCK": 4}, num_warps=4, num_stages=1
                )

        launcher = Launcher()
        calls = []
        autotuner = object.__new__(CachingAutotuner)
        autotuner.launchers = [launcher]
        autotuner.inductor_meta = {"combo_tuning_groups": [{"group": 0}]}
        autotuner.compile_id = None
        autotuner.is_backward = False

        def combo(launcher, *args, save_cache=True, **kwargs):
            calls.append((args, save_cache, kwargs))
            launcher.config.found_by_combo_autotune = True
            return launcher

        autotuner._combo_sequential_autotune = combo

        result = autotuner.prepare_for_benchmark(
            "arg",
            coordinate_descent=False,
            save_cache=False,
            kw="value",
        )

        self.assertIs(result, launcher)
        self.assertEqual(calls, [(("arg",), False, {"kw": "value"})])

    def test_coordinate_descent_heuristic_type_maps_split_scan(self):
        from torch._inductor.codegen.triton import _coordinate_descent_heuristic_type

        self.assertEqual(
            _coordinate_descent_heuristic_type("split_scan"),
            HeuristicType.SPLIT_SCAN,
        )

    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    @skipIfXpu(msg="queued coordinate descent tuning requires CUDA")
    def test_coordinate_descent_batch_generated_compile_time_queue(self):
        def fn(a):
            x = (torch.sin(a) * torch.cos(a)).sum(dim=1)
            return (x * x).sum()

        inp = torch.randn(128, 256, device=GPU_TYPE)

        metrics.reset()
        counters.clear()
        torch._dynamo.reset()
        with config.patch(
            {
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_min_kernels": 1,
                "compile_threads": 2,
                "triton.autotune_at_compile_time": True,
                "autotune_local_cache": False,
                "autotune_remote_cache": False,
                "fx_graph_cache": False,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(inp)

        torch.get_device_module(GPU_TYPE).synchronize()
        self.assertEqual(actual, fn(inp))
        self.assertGreaterEqual(metrics.generated_kernel_count, 2)
        self.assertGreaterEqual(
            counters["inductor"]["autotune_queue_tasks"], 1
        )
        self.assertGreater(
            counters["inductor"]["autotune_queue_process_pool_compiles"], 0
        )

    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    @skipIfXpu(msg="queued coordinate descent tuning requires CUDA")
    def test_coordinate_descent_batch_per_kernel_alloc_opts_out(self):
        def fn(a):
            x = (torch.sin(a) * torch.cos(a)).sum(dim=1)
            return (x * x).sum()

        inp = torch.randn(128, 256, device=GPU_TYPE)

        metrics.reset()
        counters.clear()
        torch._dynamo.reset()
        with config.patch(
            {
                "aot_inductor.autotune_per_kernel_alloc": True,
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_min_kernels": 1,
                "compile_threads": 2,
                "triton.autotune_at_compile_time": True,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(inp)

        torch.get_device_module(GPU_TYPE).synchronize()
        self.assertEqual(actual, fn(inp))
        self.assertGreaterEqual(metrics.generated_kernel_count, 2)
        self.assertEqual(counters["inductor"]["autotune_queue_tasks"], 0)

    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    @skipIfXpu(msg="queued static autotune requires CUDA")
    def test_static_config_batch_generated_no_coordesc(self):
        def fn(a):
            return torch.sum(torch.sin(a), dim=1)

        inp = torch.randn(64, 8192, device=GPU_TYPE)

        metrics.reset()
        counters.clear()
        torch._dynamo.reset()
        with config.patch(
            {
                "coordinate_descent_tuning": False,
                "max_autotune": True,
                "autotune_queue": True,
                "autotune_queue_min_kernels": 1,
                "compile_threads": 2,
                "triton.autotune_at_compile_time": True,
                "autotune_local_cache": False,
                "autotune_remote_cache": False,
                "fx_graph_cache": False,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(inp)

        torch.get_device_module(GPU_TYPE).synchronize()
        self.assertEqual(actual, fn(inp))
        self.assertGreaterEqual(metrics.generated_kernel_count, 1)
        self.assertGreaterEqual(
            counters["inductor"]["autotune_queue_tasks"], 1
        )
        self.assertEqual(counters["inductor"]["coordesc_tuning_bench"], 0)

    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    @skipIfXpu(msg="queued coordinate descent tuning requires CUDA")
    def test_coordinate_descent_batch_generated_sample_inputs_stays_scalar(self):
        def fn(a):
            x = (torch.sin(a) * torch.cos(a)).sum(dim=1)
            return (x * x).sum()

        inp = torch.randn(64, 128, device=GPU_TYPE)

        metrics.reset()
        counters.clear()
        torch._dynamo.reset()
        with config.patch(
            {
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_min_kernels": 1,
                "compile_threads": 2,
                "triton.autotune_at_compile_time": True,
                "triton.autotune_with_sample_inputs": True,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(inp)

        torch.get_device_module(GPU_TYPE).synchronize()
        self.assertEqual(actual, fn(inp))
        self.assertGreaterEqual(metrics.generated_kernel_count, 2)
        self.assertEqual(
            counters["inductor"]["autotune_queue_tasks"], 0
        )

    @skipUnless(HAS_GPU_AND_TRITON and GPU_TYPE == "cuda", "requires CUDA and Triton")
    def test_coordinate_descent_batch_generated_split_scan(self):
        from torch._inductor.runtime import triton_heuristics

        def fn(a):
            return torch.cumsum(a, 0)

        scan_numel = (
            (1 << 19) + 1
            if torch.cuda.get_device_capability()[0] >= 10
            else 8193
        )
        inp = torch.ones(scan_numel, device=GPU_TYPE, dtype=torch.int32)
        seen_split_scan_autotuners = []
        orig_split_scan = triton_heuristics.split_scan

        def record_split_scan(*args, **kwargs):
            decorator = orig_split_scan(*args, **kwargs)

            def wrap(fn):
                autotuner = decorator(fn)
                seen_split_scan_autotuners.append(autotuner)
                return autotuner

            return wrap

        metrics.reset()
        counters.clear()
        torch._dynamo.reset()
        with (
            config.patch(
                {
                    "coordinate_descent_tuning": True,
                    "coordinate_descent_tuning_batch": True,
                    "coordinate_descent_tuning_batch_policy": "auto",
                    "coordinate_descent_tuning_batch_min_kernels": 1,
                    "compile_threads": 2,
                    "triton.autotune_at_compile_time": True,
                    "autotune_local_cache": False,
                    "autotune_remote_cache": False,
                    "fx_graph_cache": False,
                }
            ),
            patch(
                "torch._inductor.runtime.triton_heuristics.split_scan",
                side_effect=record_split_scan,
            ),
            patch.object(CoordescTuner, "has_improvement", return_value=False),
        ):
            actual = torch.compile(fn, fullgraph=True)(inp)

        torch.get_device_module(GPU_TYPE).synchronize()
        self.assertEqual(actual, fn(inp))
        self.assertEqual(len(seen_split_scan_autotuners), 1)
        split_scan_autotuner = seen_split_scan_autotuners[0]
        self.assertEqual(split_scan_autotuner.heuristic_type, HeuristicType.SPLIT_SCAN)
        self.assertTrue(
            split_scan_autotuner.inductor_meta["coordinate_descent_tuning_batch"]
        )
        self.assertGreaterEqual(metrics.generated_kernel_count, 1)
        self.assertGreaterEqual(
            counters["inductor"]["autotune_queue_tasks"], 1
        )
        self.assertGreaterEqual(counters["inductor"]["coordesc_tuning_bench"], 1)

    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    @skipIfXpu(msg="queued coordinate descent tuning requires CUDA")
    def test_coordinate_descent_batch_generated_default_min_compile_time_queue(self):
        def fn(a, b, c):
            x = (torch.sin(a) * torch.cos(a)).sum(dim=1)
            y = (torch.tanh(b) * torch.cos(b)).sum(dim=1)
            z = (torch.sigmoid(c) * torch.cos(c)).sum(dim=1)
            return ((x * x) + (y * y) + (z * z)).sum()

        inputs = tuple(torch.randn(32, 64, device=GPU_TYPE) for _ in range(3))

        metrics.reset()
        counters.clear()
        torch._dynamo.reset()
        with config.patch(
            {
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_min_kernels": 3,
                "compile_threads": 2,
                "triton.autotune_at_compile_time": True,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(*inputs)

        torch.get_device_module(GPU_TYPE).synchronize()
        self.assertEqual(actual, fn(*inputs))
        self.assertGreaterEqual(metrics.generated_kernel_count, 3)
        self.assertGreaterEqual(
            counters["inductor"]["autotune_queue_tasks"], 2
        )
        self.assertGreater(
            counters["inductor"][
                "autotune_queue_max_live_retained_arg_bytes"
            ],
            0,
        )
        self.assertGreater(counters["inductor"]["autotune_queue_frontiers"], 0)

    def test_coordinate_descent_tuning_func_many_fallback_skips_batch_cache_save(self):
        class Launcher:
            def __init__(self, config):
                self.config = config
                self.n_regs = 1
                self.n_spills = 0
                self.shared = 0
                self.cache_hash = f"hash-{config.kwargs['XBLOCK']}"
                self.store_cubin = False

        def timing(launcher):
            return abs(launcher.config.kwargs["XBLOCK"] - 8) + 1.0

        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(
            __name__="kernel",
            src="def kernel():\n    pass\n",
        )
        autotuner.lock = threading.Lock()
        autotuner.coordesc_tuner = CoordescTuner(
            name="kernel",
            size_hints={"x": 16},
            frozen_fields={"num_warps"},
        )
        autotuner.size_hints = {"x": 16}
        autotuner.heuristic_type = HeuristicType.REDUCTION
        autotuner.benchmark_failure_reasons = {}
        autotuner.autotune_time_taken_ns = 0
        autotuner._ensure_kernel_loaded = lambda: None
        autotuner._precompile_config = lambda config: SimpleNamespace(
            make_launcher=lambda: Launcher(config)
        )
        save_calls = []

        def finish_coordesc(
            best_config, config2launcher, elapsed_ns, save_cache=True, **kwargs
        ):
            save_calls.append((save_cache, kwargs))
            winner = config2launcher[best_config]
            winner.config.found_by_coordesc = True
            return winner

        autotuner._finish_coordinate_descent_tuning = finish_coordesc
        grouped_calls = []

        def benchmark_all_launchers(launchers, *args, **kwargs):
            grouped_calls.append(
                [launcher.config.kwargs["XBLOCK"] for launcher in launchers]
            )
            if len(launchers) > 1:
                raise RuntimeError("batched timing failed")
            return {launcher: timing(launcher) for launcher in launchers}

        autotuner.benchmark_all_launchers = benchmark_all_launchers
        baseline = Launcher(triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1))

        counters.clear()
        winner = autotuner._coordinate_descent_tuning(
            baseline,
            use_batch_benchmarking=True,
        )

        self.assertEqual(winner.config.kwargs["XBLOCK"], 8)
        self.assertIn([8, 2], grouped_calls)
        self.assertEqual(len(save_calls), 1)
        save_cache, kwargs = save_calls[0]
        self.assertFalse(save_cache)
        self.assertFalse(kwargs["coordinate_descent_tuning_batch"])
        self.assertGreater(
            counters["inductor"]["coordesc_tuning_batch_ungrouped_cache_skips"],
            0,
        )

    def test_autotune_hints_to_configs(self):
        device_props = DeviceProperties.create(torch.device(GPU_TYPE))
        device_props = device_props._replace(warp_size=8)

        hints = {AutotuneHint.ONE_ELEMENT_PER_THREAD}
        size_hints = (1024,)
        block_size = 256

        seen_num_elements_per_warp = set()

        def mock_triton_config(
            size_hints,
            x,
            y=None,
            z=None,
            num_stages=None,
            num_elements_per_warp=None,
            min_elem_per_thread=None,
        ):
            seen_num_elements_per_warp.add(num_elements_per_warp)
            return None

        with unittest.mock.patch(
            "torch._inductor.runtime.triton_heuristics.triton_config",
            mock_triton_config,
        ):
            _ = autotune_hints_to_configs(hints, size_hints, block_size, device_props)

        self.assertTrue(8 in seen_num_elements_per_warp)

    @unittest.skipIf(not HAS_WARP_SPEC, "FBCODE Triton is required for this test")
    def test_template_function_ws(self):
        triton_meta = {"device": MagicMock()}
        num_stages = 2
        num_warps = 4
        num_consumer_groups = 3
        num_buffers_warp_spec = 5

        with patch(
            "torch._inductor.runtime.triton_heuristics.cached_autotune"
        ) as mock_cached_autotune:
            template(
                num_stages=num_stages,
                num_warps=num_warps,
                triton_meta=triton_meta,
                num_consumer_groups=num_consumer_groups,
                num_buffers_warp_spec=num_buffers_warp_spec,
            )
            mock_cached_autotune.assert_called_once()
            configs = mock_cached_autotune.call_args[0][1]
            self.assertEqual(configs[0].num_consumer_groups, num_consumer_groups)
            self.assertEqual(configs[0].num_buffers_warp_spec, num_buffers_warp_spec)

    @runOnRocm
    def test_amd_special_config_args(self):
        """
        waves_per_eu is an example of a special config arg on AMD; if it is explicitly specified
        in a config, the kwarg will exist in the kwargs but not in the function signature.
        """

        @torch.library.triton_op("test_triton_heuristics::triton_sqr", mutates_args=())
        def triton_sqr(x: torch.Tensor) -> torch.Tensor:
            y = torch.empty_like(x)

            def grid(meta):
                return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

            torch.library.wrap_triton(get_autotuned_amd_sqr_kernel())[grid](
                x, y, x.numel()
            )

        def fn(x):
            return triton_sqr(x)

        x = torch.randn(32, device=GPU_TYPE)
        ref = fn(x)
        res = torch.compile(fn)(x)
        self.assertEqual(ref, res)

    @skipIfXpu(
        msg="lack _get_exceeding_shared_memory_checker support - torch-xpu-ops: 2331"
    )
    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @parametrize("do_pruning", [False, True])
    def test_prune_configs_over_shared_memory_limit(self, do_pruning):
        from torch._inductor.template_heuristics.triton import (
            CUDAConfigHeuristic,
            GemmConfig,
            ROCmConfigHeuristic,
        )

        expected_count = 1 if do_pruning else 2
        mm_configs = [
            GemmConfig(32, 32, 32, 1, 8, group_m=8),
            GemmConfig(
                128, 128, 128, 100, 8, group_m=4
            ),  # intentionally large to exceed shared memory limit
        ]
        with config.patch(
            {"max_autotune_prune_choices_based_on_shared_mem": do_pruning}
        ):
            if torch.version.hip:
                config_heuristic = ROCmConfigHeuristic()
            else:
                config_heuristic = CUDAConfigHeuristic()
            config_heuristic.should_scale_configs = False
            config_heuristic.mm_configs = mm_configs
            configs = list(
                config_heuristic.get_mm_configs()(3, 3, 3, dtype_size=4, op_name="mm")
            )
            self.assertEqual(len(configs), expected_count)


class TestArgumentCloneAndRestore(TestCase):
    # Our tensor is large enough. If a unexpected copy happens, the
    # peak memory increase should be larger than tolerance and the test
    # will fail.
    MEM_TOLERANCE = int(256 * 1e6)

    def _create_caching_autotuner(self):
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        args["optimize_mem"] = True
        args["mutated_arg_names"] = ["in_ptr0"]
        autotuner = CachingAutotuner(**args)
        return autotuner

    def _create_tensor(self, pad=1, with_offset=False):
        """
        Create a GPU tensor of about 1GB size.
        """
        M = 2
        N = 2**29 // 4
        out = rand_strided((M, N), (N + pad, 1), device=GPU_TYPE)
        if with_offset:
            out = out[:, 1:]
        return out

    def _do_test(self, gpu_tensor):
        torch.get_device_module(GPU_TYPE).reset_peak_memory_stats()
        autotuner = self._create_caching_autotuner()

        old_storage_offset = gpu_tensor.storage_offset()
        gpu_tensor_clone = clone_preserve_strides(gpu_tensor)

        peak_mem_before = torch.get_device_module(GPU_TYPE).max_memory_allocated()
        cpu_copies = autotuner.copy_args_to_cpu_if_needed(gpu_tensor)
        self.assertTrue(len(cpu_copies) == 1)

        # Mutate the arg
        gpu_tensor.add_(1)

        # will restore gpu_tensor
        autotuner.restore_args_from_cpu(cpu_copies)
        self.assertTrue(gpu_tensor is not gpu_tensor_clone)
        self.assertEqual(gpu_tensor.size(), gpu_tensor_clone.size())
        self.assertEqual(gpu_tensor.stride(), gpu_tensor_clone.stride())
        self.assertEqual(gpu_tensor.storage_offset(), old_storage_offset)

        # Note: torch.allclose somehow allocates large amount of extra memory.
        # Record peak memory before that.
        peak_mem_after = torch.get_device_module(GPU_TYPE).max_memory_allocated()

        self.assertTrue(torch.allclose(gpu_tensor, gpu_tensor_clone))
        self.assertTrue(
            peak_mem_after <= peak_mem_before + self.MEM_TOLERANCE,
            f"{peak_mem_before=} v.s. {peak_mem_after=}",
        )

        # Avoid OOM in CI
        self.assertTrue(peak_mem_after < 1e10)

    @requires_gpu_with_enough_memory(1e10)
    def test_clone_contiguous_args(self):
        arg = self._create_tensor(pad=0)
        self.assertTrue(arg.is_contiguous())
        self.assertTrue(arg.storage_offset() == 0)
        self._do_test(arg)

    @requires_gpu_with_enough_memory(1e10)
    def test_clone_non_contiguous_args(self):
        arg = self._create_tensor(pad=1)
        self.assertFalse(arg.is_contiguous())
        self.assertTrue(arg.storage_offset() == 0)
        self._do_test(arg)

    @requires_gpu_with_enough_memory(1e10)
    def test_clone_args_with_non_zero_offset(self):
        arg = self._create_tensor(pad=1, with_offset=True)
        self.assertFalse(arg.is_contiguous())
        self.assertTrue(arg.storage_offset() > 0)

        self._do_test(arg)


class TestDumpLaunchTensors(TestCase):
    """Test the _dump_launch_tensors functionality"""

    def setUp(self):
        super().setUp()
        # Create a temporary directory for test dumps
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary directory
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        super().tearDown()

    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_dump_launch_tensors(self):
        """
        Test that dump_launch_tensors functions correctly:
        1. Creates the dump directory when torch.compile() runs
        2. Saves tensor files that can be loaded
        3. Loads tensors to match the original values
        4. Properly rotates when max_kernel_dump_occurrences is reached
        """
        from torch._inductor.config import triton as inductor_triton_config
        from torch._inductor.runtime.runtime_utils import cache_dir

        # Clear any existing state
        inductor_triton_config.debug_dump_kernel_inputs.clear()

        # Define a simple function that will generate Triton kernels
        def simple_model(x):
            y = x * 2.0
            z = y + 1.0
            return z.sum()

        old_dump_env = os.environ.get("TORCHINDUCTOR_DUMP_LAUNCH_TENSORS")
        os.environ["TORCHINDUCTOR_DUMP_LAUNCH_TENSORS"] = "1"

        try:
            compiled_fn = torch.compile(simple_model)

            # Run the compiled function multiple times to test rotation
            max_runs = inductor_triton_config.max_kernel_dump_occurrences

            for i in range(max_runs + 2):
                test_input = torch.randn(100, 100, device=GPU_TYPE) * (i + 1)
                _ = compiled_fn(test_input)

            # After multiple runs, verify rotation and tensor correctness
            kernel_bases = {}
            verified_tensor_load = False

            for root, dirs, files in os.walk(cache_dir()):
                for d in dirs:
                    if "_run_" in d:
                        full_path = os.path.join(root, d)
                        tensor_files = [
                            f
                            for f in os.listdir(full_path)
                            if f.startswith("tensor_") and f.endswith(".pt")
                        ]
                        if not tensor_files:
                            continue

                        dir_name = os.path.basename(full_path)
                        base_name = dir_name.rsplit("_run_", 1)[0]
                        run_idx = int(dir_name.rsplit("_run_", 1)[1])

                        # Track run indices per kernel
                        if base_name not in kernel_bases:
                            kernel_bases[base_name] = []
                        kernel_bases[base_name].append(run_idx)

                        # Verify we can successfully load at least one saved tensor
                        if not verified_tensor_load:
                            first_tensor_file = os.path.join(full_path, tensor_files[0])
                            loaded_tensor = torch.load(first_tensor_file)

                            # Verify it's a valid tensor with expected properties
                            self.assertIsInstance(loaded_tensor, torch.Tensor)
                            self.assertEqual(loaded_tensor.device.type, GPU_TYPE)
                            verified_tensor_load = True

            # Verify rotation constraints
            if kernel_bases:
                for base_name, indices in kernel_bases.items():
                    self.assertLessEqual(
                        len(indices),
                        max_runs,
                        f"Kernel {base_name} has more runs ({len(indices)}) than max ({max_runs})",
                    )

                    # Verify the indices are within [0, max_runs)
                    for idx in indices:
                        self.assertLess(
                            idx,
                            max_runs,
                            f"Run index {idx} exceeds max_runs-1 ({max_runs - 1})",
                        )

        finally:
            # Restore environment variable
            if old_dump_env is None:
                os.environ.pop("TORCHINDUCTOR_DUMP_LAUNCH_TENSORS", None)
            else:
                os.environ["TORCHINDUCTOR_DUMP_LAUNCH_TENSORS"] = old_dump_env


class TestRecheckAutotuneCache(TestCase):
    """Tests for CachingAutotuner.recheck_autotune_cache"""

    @staticmethod
    def _make_compile_result(cfg):
        """Create a mock StaticTritonCompileResult with the given config."""
        from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

        result = MagicMock(spec=StaticTritonCompileResult)
        result.config = cfg
        return result

    @staticmethod
    def _make_autotuner_with_results(configs, compile_results):
        """
        Create a CachingAutotuner and inject compile_results directly,
        bypassing actual Triton compilation.
        """
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        args["configs"] = configs
        autotuner = CachingAutotuner(**args)
        autotuner.compile_results = compile_results
        return autotuner

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_single_config_enters_cache_hit_block(self):
        """
        When there is exactly 1 config and the autotune cache returns a hit,
        recheck_autotune_cache should narrow compile_results to that single
        matching result (not skip the block due to len(configs) == 1).
        """
        cfg = triton_config({"x": 16}, 64)
        cfg.found_by_coordesc = False
        compile_result = self._make_compile_result(cfg)

        autotuner = self._make_autotuner_with_results([cfg], [compile_result])

        # Cache returns the same config as the best
        cached_cfg = triton_config({"x": 16}, 64)
        cached_cfg.found_by_coordesc = True

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([cached_cfg], None, {"autotune_cache_state": "hit"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        # The compile_results should be narrowed to just the matching one
        self.assertEqual(len(autotuner.compile_results), 1)
        self.assertIs(autotuner.compile_results[0], compile_result)
        # And found_by_coordesc must be propagated
        self.assertTrue(autotuner.compile_results[0].config.found_by_coordesc)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_propagates_found_by_coordesc_true(self):
        """
        When the cached best config has found_by_coordesc=True,
        it must be propagated to the compile result's config.
        """
        cfg_a = triton_config({"x": 16}, 64)
        cfg_b = triton_config({"x": 256}, 64)
        cfg_a.found_by_coordesc = False
        cfg_b.found_by_coordesc = False
        result_a = self._make_compile_result(cfg_a)
        result_b = self._make_compile_result(cfg_b)

        autotuner = self._make_autotuner_with_results(
            [cfg_a, cfg_b], [result_a, result_b]
        )

        # Cache says cfg_b is the best, found via coordesc
        cached_cfg = triton_config({"x": 256}, 64)
        cached_cfg.found_by_coordesc = True

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([cached_cfg], None, {"autotune_cache_state": "hit"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        self.assertEqual(len(autotuner.compile_results), 1)
        self.assertIs(autotuner.compile_results[0], result_b)
        # The flag must be propagated from the cached config
        self.assertTrue(autotuner.compile_results[0].config.found_by_coordesc)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_propagates_found_by_coordesc_false(self):
        """
        When the cached best config has found_by_coordesc=False, it must be
        propagated so that coordinate descent can still run if enabled.
        """
        cfg = triton_config({"x": 16}, 64)
        cfg.found_by_coordesc = True
        compile_result = self._make_compile_result(cfg)

        autotuner = self._make_autotuner_with_results([cfg], [compile_result])

        cached_cfg = triton_config({"x": 16}, 64)
        cached_cfg.found_by_coordesc = False

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([cached_cfg], None, {"autotune_cache_state": "hit"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        self.assertEqual(len(autotuner.compile_results), 1)
        self.assertFalse(autotuner.compile_results[0].config.found_by_coordesc)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_no_cache_hit_leaves_results_unchanged(self):
        """
        When there's no autotune cache hit, compile_results should not change.
        """
        cfg_a = triton_config({"x": 16}, 64)
        cfg_b = triton_config({"x": 256}, 64)
        result_a = self._make_compile_result(cfg_a)
        result_b = self._make_compile_result(cfg_b)

        autotuner = self._make_autotuner_with_results(
            [cfg_a, cfg_b], [result_a, result_b]
        )

        # Cache returns no hit (empty list)
        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([], None, {"autotune_cache_state": "miss"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        # Nothing should change
        self.assertEqual(len(autotuner.compile_results), 2)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_uses_original_configs_hash(self):
        """
        A runtime static autotuner can be reconstructed with a narrowed config
        list after max-autotune, but coordesc winners are cached under the
        original full config-set hash.
        """
        cfg = triton_config({"x": 16}, 64)
        compile_result = self._make_compile_result(cfg)

        autotuner = self._make_autotuner_with_results([cfg], [compile_result])
        autotuner.autotune_configs_hash = "original-config-set-hash"

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([], None, {"autotune_cache_state": "miss"}),
        ) as mock_check:
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        mock_check.assert_called_once()
        self.assertEqual(
            mock_check.call_args.kwargs["configs_hash"],
            "original-config-set-hash",
        )

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_tolerates_missing_original_configs_hash(self):
        cfg = triton_config({"x": 16}, 64)
        compile_result = self._make_compile_result(cfg)
        autotuner = self._make_autotuner_with_results([cfg], [compile_result])
        del autotuner.autotune_configs_hash

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([], None, {"autotune_cache_state": "miss"}),
        ) as mock_check:
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        mock_check.assert_called_once()
        self.assertIsNone(mock_check.call_args.kwargs["configs_hash"])

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_cached_autotune_preserves_original_configs_hash(self):
        """
        cached_autotune computes the config-set hash before cache narrowing and
        passes it to both the initial cache check and the runtime autotuner.
        """
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        expected_hash = hash_configs(args["configs"])
        mock_cls = MagicMock(return_value="autotuner")

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=(args["configs"][:1], None, {"autotune_cache_state": "hit"}),
        ) as mock_check:
            decorator = cached_autotune(
                size_hints=[16],
                configs=args["configs"],
                triton_meta=args["triton_meta"],
                heuristic_type=args["heuristic_type"],
                filename="kernel.py",
                inductor_meta=args["inductor_meta"],
                caching_autotuner_cls=mock_cls,
            )
            autotuner = decorator(args["fn"])

        self.assertEqual(autotuner, "autotuner")
        self.assertEqual(mock_check.call_args.kwargs["configs_hash"], expected_hash)
        self.assertEqual(
            mock_cls.call_args.kwargs["autotune_configs_hash"],
            expected_hash,
        )

    def test_cached_autotune_records_effective_coordesc_batch_mode(self):
        def kernel(XBLOCK):
            pass

        def run_case(heuristic_type, native_matmul, expected_batch):
            configs = [triton.Config({"XBLOCK": 16}, num_warps=4, num_stages=1)]
            inductor_meta = {
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_policy": "auto",
            }
            triton_meta = {
                "device_type": "cuda",
                "native_matmul": native_matmul,
            }
            mock_cls = MagicMock(return_value="autotuner")

            with (
                config.patch({"autotune_queue": True}),
                patch(
                    "torch._inductor.runtime.autotune_common._coordinate_descent_batch_has_compile_parallelism",
                    return_value=True,
                ),
                patch(
                    "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
                    return_value=(configs, None, None),
                ) as mock_check,
            ):
                decorator = cached_autotune(
                    size_hints=[16],
                    configs=configs,
                    triton_meta=triton_meta,
                    heuristic_type=heuristic_type,
                    filename="kernel.py",
                    inductor_meta=inductor_meta,
                    caching_autotuner_cls=mock_cls,
                )
                autotuner = decorator(SimpleNamespace(fn=kernel))

            self.assertEqual(autotuner, "autotuner")
            self.assertEqual(
                mock_check.call_args.args[2]["coordinate_descent_tuning_batch"],
                expected_batch,
            )
            self.assertEqual(
                mock_cls.call_args.kwargs["inductor_meta"][
                    "coordinate_descent_tuning_batch"
                ],
                expected_batch,
            )

        run_case(HeuristicType.POINTWISE, False, False)
        run_case(HeuristicType.REDUCTION, False, True)
        run_case(HeuristicType.POINTWISE, True, True)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_runtime_coordesc_recheck_rebuilds_launchers_once(self):
        """
        Runtime coordesc should do one cache recheck before scalar tuning. If
        that recheck finds a coordesc winner, launchers are rebuilt for the
        narrowed result and the recheck is not repeated on later calls.
        """
        cfg = triton_config({"x": 16}, 64)
        cfg.found_by_coordesc = False
        compile_result = self._make_compile_result(cfg)

        autotuner = self._make_autotuner_with_results([cfg], [compile_result])
        autotuner.filename = "kernel.py"
        autotuner.inductor_meta["coordinate_descent_tuning"] = True
        autotuner.launchers = [MagicMock()]
        autotuner._cached_launcher = MagicMock()

        def recheck_cache(reload_kernel_from_src):
            compile_result.config.found_by_coordesc = True

        autotuner.recheck_autotune_cache = MagicMock(side_effect=recheck_cache)
        autotuner._make_launchers = MagicMock()

        autotuner._recheck_coordesc_cache_before_runtime_tuning()
        autotuner._recheck_coordesc_cache_before_runtime_tuning()

        autotuner.recheck_autotune_cache.assert_called_once()
        autotuner._make_launchers.assert_called_once()
        self.assertEqual(autotuner.launchers, [])
        self.assertIsNone(autotuner._cached_launcher)

    def test_prepare_runtime_coordesc_recheck_runs_before_autotune(self):
        """
        Runtime cache recheck must happen before max-autotune can save a
        non-coordesc cache entry over a compile-time coordesc winner.
        """
        order = []
        autotuner = object.__new__(CachingAutotuner)
        autotuner.launchers = []
        autotuner.compile_results = []
        autotuner.inductor_meta = {}
        autotuner._install_triton_allocator = lambda: None

        def make_launcher(found_by_coordesc):
            launcher = MagicMock()
            launcher.config.found_by_coordesc = found_by_coordesc
            return launcher

        def precompile():
            order.append("precompile")
            autotuner.launchers = [
                make_launcher(False),
                make_launcher(False),
            ]

        def recheck():
            order.append("recheck")
            if order.count("recheck") == 1:
                return
            autotuner.launchers = [make_launcher(True)]

        def autotune_to_one_config(*args, **kwargs):
            order.append("autotune")

        autotuner.precompile = precompile
        autotuner._recheck_coordesc_cache_before_runtime_tuning = recheck
        autotuner.autotune_to_one_config = autotune_to_one_config

        launcher = autotuner.prepare_for_benchmark(
            runtime_coordesc_cache_recheck=True
        )

        self.assertEqual(order, ["recheck", "precompile", "recheck"])
        self.assertTrue(launcher.config.found_by_coordesc)


@triton.jit
def hip_autotune_kernel(
    in_ptr,
    out_ptr,
    numel,
    XBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < numel
    data = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, data * 2.0, mask=mask)


@functools.lru_cache
def get_hip_autotune_kernel_with_invalid_config():
    # num_warps=32 with HIP warp_size=64 = 2048 threads, exceeds 1024 limit
    return triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 128}, num_warps=32, num_stages=1),  # invalid
            triton.Config({"XBLOCK": 256}, num_warps=1, num_stages=1),  # valid
        ],
        key=["numel"],
    )(hip_autotune_kernel)


class TestHIPInvalidConfigHandling(TestCase):
    @runOnRocm
    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_benchmark_returns_inf_on_invalid_config(self):
        from torch._inductor.runtime.benchmarking import TritonBenchmarker

        benchmarker = TritonBenchmarker()

        def failing_callable():
            raise RuntimeError(
                "Triton Error [HIP]: Code: 9, Message: invalid configuration argument"
            )

        result = benchmarker.benchmark(
            fn=failing_callable,
            device=GPU_TYPE,
            is_vetted_benchmarking=True,
        )
        self.assertEqual(result, float("inf"))

    @runOnRocm
    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_autotune_skips_invalid_hip_config_and_succeeds(self):
        numel = 1024 * 1024
        x = torch.randn(numel, device=GPU_TYPE, dtype=torch.float32)
        y = torch.empty_like(x)

        kernel = get_hip_autotune_kernel_with_invalid_config()

        def grid(meta):
            return (triton.cdiv(numel, meta["XBLOCK"]),)

        kernel[grid](x, y, numel)

        expected = x * 2.0
        torch.testing.assert_close(y, expected)


class TestGridExprMaximum(TestCase):
    def test_maximum_cpp_mode_casts_int_constants_to_long(self):
        from torch._inductor.runtime.triton_heuristics import Grid1D

        grid = Grid1D(inductor_meta={}, mode="cpp")
        # Mixed str/int: int constants must be cast to (long) for std::max
        result = grid.maximum(["ynumel_0", "ynumel_1", 4480])
        self.assertIn("(long)4480", result)
        self.assertIn("std::max", result)
        # All strings: no cast needed
        result = grid.maximum(["xnumel", "ynumel"])
        self.assertNotIn("(long)", result)
        # All ints: constant-folds
        self.assertEqual(grid.maximum([10, 20, 5]), 20)


class TestGrid2DWithYZOverflowZeroYnumel(TestCase):
    """Regression test for https://github.com/pytorch/pytorch/issues/178530"""

    def test_grid2d_yz_overflow_zero_ynumel_python(self):
        from torch._inductor.runtime.triton_heuristics import Grid2DWithYZOverflow

        grid = Grid2DWithYZOverflow(inductor_meta={}, mode="python")
        grid.generate({"XBLOCK": 128, "YBLOCK": 128})
        # ynumel=0 must not raise ZeroDivisionError
        x, y, z = grid.eval_slow(
            {"xnumel": 256, "ynumel": 0, "XBLOCK": 128, "YBLOCK": 128}
        )
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_grid2d_yz_overflow_zero_ynumel_cpp(self):
        from torch._inductor.runtime.triton_heuristics import Grid2DWithYZOverflow

        grid = Grid2DWithYZOverflow(inductor_meta={}, mode="cpp")
        grid.generate({"XBLOCK": 128, "YBLOCK": 128})
        # cpp mode: the generated expression should contain a zero-guard
        self.assertIn("== 0", str(grid.y_grid))

    def test_grid2d_yz_overflow_nonzero_ynumel_unchanged(self):
        from torch._inductor.runtime.triton_heuristics import Grid2DWithYZOverflow

        grid = Grid2DWithYZOverflow(inductor_meta={}, mode="python")
        grid.generate({"XBLOCK": 128, "YBLOCK": 128})
        # Normal case: ynumel > 0 still works correctly
        x, y, z = grid.eval_slow(
            {"xnumel": 256, "ynumel": 256, "XBLOCK": 128, "YBLOCK": 128}
        )
        self.assertEqual(x, 2)
        self.assertEqual(y, 2)
        self.assertEqual(z, 1)

    def test_grid2d_yz_overflow_large_ynumel(self):
        from torch._inductor.runtime.triton_heuristics import Grid2DWithYZOverflow

        grid = Grid2DWithYZOverflow(inductor_meta={}, mode="python")
        grid.generate({"XBLOCK": 128, "YBLOCK": 128})
        # Large ynumel that requires overflow splitting across y and z
        x, y, z = grid.eval_slow(
            {"xnumel": 128, "ynumel": 128 * 131070, "XBLOCK": 128, "YBLOCK": 128}
        )
        self.assertEqual(x, 1)
        # y * z must cover all y blocks
        self.assertGreaterEqual(y * z, 131070)


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
