# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import dataclasses
import itertools
import logging
import time
from typing import Any

import torch
from torch._dynamo.utils import counters
from torch.utils._ordered_set import OrderedSet

from ..triton_bundler import TritonBundler
from .autotune_benchmarking import (
    _benchmark_launcher_requests,
    _device_guard,
    _LauncherBenchmarkRequest,
)
from .autotune_common import (
    _COORDESC_BENCHMARK_MODE_GROUPED,
    _COORDESC_BENCHMARK_MODE_UNGROUPED,
    _COORDESC_UNGROUPED_BENCHMARK_KEY,
    _coordinate_descent_batch_has_compile_parallelism,
    _has_ungrouped_coordesc_benchmark,
    _is_expected_compile_config_failure,
    _requires_benchmark_arg_clones,
)
from .benchmarking import is_invalid_configuration_error
from .coordinate_descent_tuner import get_field
from .autotune_queue_state import (
    _clear_active_autotune_queue,
    _get_active_coordinate_descent_batch,
    _set_active_autotune_queue,
    get_active_autotune_queue,
    suspend_autotune_queue,
)
from .runtime_utils import dynamo_timed, red_text, triton_config_to_hashable


log = logging.getLogger(__name__)


def _clear_deferred_compile_state(autotuner: Any) -> None:
    clear = getattr(autotuner, "clear_deferred_compile_state", None)
    if clear is not None:
        clear()
        return
    autotuner._config_compile_submitter = None
    autotuner._static_config_compile_submitter = None
    autotuner._static_triton_bundle_key = None


def _estimate_retained_arg_bytes(args, kwargs) -> int:
    retained_bytes = 0
    seen: set[tuple[str, int, int]] = set()
    for value in itertools.chain(args, kwargs.values()):
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            continue
        storage = value.untyped_storage()
        key = (
            value.device.type,
            value.device.index or 0,
            storage.data_ptr(),
        )
        if key in seen:
            continue
        seen.add(key)
        retained_bytes += storage.nbytes()
    return retained_bytes


def _save_coordesc_winner_kernel(autotuner, winner, stream, save_kernel) -> None:
    if not save_kernel or not winner.store_cubin:
        return
    if autotuner.device_props.type == "cpu":
        autotuner.cpu_kernel_saved = False
        autotuner.save_cpu_kernel(winner)
    elif stream is not None:
        autotuner.cuda_kernel_saved = False
        autotuner.save_gpu_kernel(stream, winner)


def _run_scalar_coordesc_fallback(
    autotuner, launcher, args, kwargs, stream, save_kernel
) -> None:
    device_idx = getattr(getattr(autotuner, "device_props", None), "index", None)
    device_idx = device_idx if isinstance(device_idx, int) else None
    with _device_guard(autotuner.get_device_interface(), device_idx):
        winner = autotuner.coordinate_descent_tuning(
            launcher,
            *args,
            use_batch_benchmarking=None,
            clone_args=False,
            **kwargs,
        )
    _save_coordesc_winner_kernel(autotuner, winner, stream, save_kernel)
    autotuner.launchers = [winner]
    autotuner._cached_launcher = None


class _AutotuneQueueTaskBase:
    def _init_queue_task_state(
        self,
        autotuner,
        args,
        kwargs,
        stream,
        save_kernel,
    ) -> None:
        self.autotuner = autotuner
        self.args = args
        self.kwargs = kwargs
        self.stream = stream
        self.save_kernel = save_kernel
        self.device_idx = getattr(
            getattr(autotuner, "device_props", None), "index", None
        )
        self.device_idx = self.device_idx if isinstance(self.device_idx, int) else None
        self.retained_arg_bytes = _estimate_retained_arg_bytes(args, kwargs)
        self.queue_bytes = self.retained_arg_bytes
        self.pending_candidates = None
        self.done = False
        self.finalized = False
        self.benchmark_group_state: dict[str, int] = {}
        self.start_time_ns = time.time_ns()

    def benchmark_mode(self) -> str:
        if self.has_ungrouped_batch_benchmark():
            return _COORDESC_BENCHMARK_MODE_UNGROUPED
        return _COORDESC_BENCHMARK_MODE_GROUPED

    def mark_ungrouped_batch_benchmark(self) -> None:
        self.benchmark_group_state[_COORDESC_UNGROUPED_BENCHMARK_KEY] = 1

    def has_ungrouped_batch_benchmark(self) -> bool:
        return _has_ungrouped_coordesc_benchmark(self.benchmark_group_state)

    def benchmark_request(
        self,
        launcher,
        *,
        benchmark_group_key=None,
        benchmark_group_state=None,
    ):
        return _LauncherBenchmarkRequest(
            autotuner=self.autotuner,
            launcher=launcher,
            args=self.args,
            kwargs=self.kwargs,
            device_idx=self.device_idx,
            wait_ready_fn=None,
            clone_args=False,
            benchmark_group_key=(
                self if benchmark_group_key is None else benchmark_group_key
            ),
            benchmark_group_state=(
                self.benchmark_group_state
                if benchmark_group_state is None
                else benchmark_group_state
            ),
        )

    def benchmark_launcher(self, launcher):
        with _device_guard(self.autotuner.get_device_interface(), self.device_idx):
            return self.autotuner.bench(
                launcher,
                *self.args,
                clone_args=False,
                **self.kwargs,
            )

    def _lookup_timing(self, config) -> float:
        timing = self.autotuner.coordesc_tuner.lookup_in_cache(
            config,
            benchmark_mode=self.benchmark_mode(),
        )
        return float("inf") if timing is None else timing

    def _uncached_configs(self, candidate_configs):
        return [
            config
            for config in candidate_configs
            if self.autotuner.coordesc_tuner.lookup_in_cache(
                config,
                benchmark_mode=self.benchmark_mode(),
            )
            is None
        ]

    def has_pending_results(self) -> bool:
        return self.pending_candidates is not None

    def should_fallback_on_launch_exception(self, exc: Exception) -> bool:
        return False


class _CoordinateDescentTask(_AutotuneQueueTaskBase):
    can_scalar_drain = True

    def __init__(
        self,
        autotuner,
        launcher,
        args,
        kwargs,
        stream,
        save_kernel,
    ) -> None:
        self.initial_launcher = launcher
        self._init_queue_task_state(
            autotuner,
            args,
            kwargs,
            stream,
            save_kernel,
        )
        self.config2launcher = {launcher.config: launcher}
        self.best_config = launcher.config
        self.best_timing = autotuner.coordesc_tuner.lookup_in_cache(
            self.best_config,
            benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
        )
        self.best_timing_mode = (
            _COORDESC_BENCHMARK_MODE_GROUPED
            if self.best_timing is not None
            else None
        )
        self.phase = "fields" if self.best_timing is not None else "baseline"
        self.field_index = 0
        self.improved_this_pass = False
        self.old_best_timing_before_all_directions = None

        ensure_loaded_start_ns = time.time_ns()
        autotuner._ensure_kernel_loaded()
        counters["inductor"]["coordesc_tuning_batch_ensure_loaded_ns"] += (
            time.time_ns() - ensure_loaded_start_ns
        )
        self.can_scalar_drain = False

    def submit_compile_job(self, config):
        launcher = self.config2launcher.get(config)
        if launcher is not None:
            return launcher
        submitter = getattr(self.autotuner, "_config_compile_submitter", None)
        if submitter is None:
            return None
        counters["inductor"]["autotune_queue_process_pool_compiles"] += 1
        return submitter([config])

    def resolve_compile_result(self, config, result):
        if not isinstance(result, tuple):
            return result
        compile_result = self.autotuner._accept_deferred_static_compile_result(
            config, result, append=False
        )
        if compile_result is None:
            return None
        launcher = compile_result.make_launcher()
        self.config2launcher[config] = launcher
        return launcher

    def launcher_for_config(self, config):
        return self.autotuner._make_coordesc_launcher_for_config(
            self.config2launcher, config
        )

    def cache_benchmark_result(self, config, timing) -> None:
        self.autotuner.coordesc_tuner.cache_benchmark_result(
            config,
            timing,
            benchmark_mode=self.benchmark_mode(),
        )
        counters["inductor"]["coordesc_tuning_bench"] += 1

    def _refresh_best_timing_mode(self) -> None:
        if self.best_timing is None:
            return
        benchmark_mode = self.benchmark_mode()
        if self.best_timing_mode == benchmark_mode:
            return
        launcher = self.launcher_for_config(self.best_config)
        try:
            timing = self.benchmark_launcher(launcher)
        except Exception as e:
            if not is_invalid_configuration_error(e):
                raise
            log.debug("COORDESC: got exception %s", e)
            timing = float("inf")
        self.cache_benchmark_result(self.best_config, timing)
        self.best_timing = timing
        self.best_timing_mode = benchmark_mode

    def _prepare_candidates(self, candidate_configs):
        uncached_configs = self._uncached_configs(candidate_configs)
        if uncached_configs:
            self.pending_candidates = candidate_configs
            return uncached_configs
        return None

    def prepare_next_configs(self):
        while not self.done:
            if self.pending_candidates is not None:
                uncached = self._uncached_configs(self.pending_candidates)
                if uncached:
                    return uncached
                self.accept_pending_results()
                continue

            if self.phase == "baseline":
                uncached = self._prepare_candidates([self.best_config])
                if uncached:
                    return uncached
                self._accept_baseline_timing()
                continue

            if self.phase == "fields":
                fields = self.autotuner.coordesc_tuner.tunable_fields
                while self.field_index < len(fields):
                    field = fields[self.field_index]
                    self.field_index += 1
                    if get_field(self.best_config, field) is None:
                        continue
                    candidate_configs = (
                        self.autotuner.coordesc_tuner.get_neighbour_configs(
                            self.best_config, field
                        )
                    )
                    uncached = self._prepare_candidates(candidate_configs)
                    if uncached:
                        return uncached
                    self._accept_candidate_timings(candidate_configs)

                if self.improved_this_pass:
                    self.improved_this_pass = False
                    self.field_index = 0
                    continue

                if self.autotuner.coordesc_tuner.inductor_meta.get(
                    "coordinate_descent_check_all_directions"
                ):
                    self.phase = "all_directions"
                    self.old_best_timing_before_all_directions = self.best_timing
                    continue

                self.done = True
                return []

            assert self.phase == "all_directions", self.phase
            candidate_configs = self.autotuner.coordesc_tuner.get_all_tuning_directions(
                self.best_config
            )
            uncached = self._prepare_candidates(candidate_configs)
            if uncached:
                return uncached
            self._accept_all_direction_timings(candidate_configs)

        return []

    def accept_pending_results(self) -> None:
        assert self.pending_candidates is not None
        candidate_configs = self.pending_candidates
        if self._uncached_configs(candidate_configs):
            return
        self.pending_candidates = None

        if self.phase == "baseline":
            self._accept_baseline_timing()
        elif self.phase == "fields":
            self._accept_candidate_timings(candidate_configs)
        else:
            assert self.phase == "all_directions", self.phase
            self._accept_all_direction_timings(candidate_configs)

    def _accept_baseline_timing(self) -> None:
        self.best_timing = self._lookup_timing(self.best_config)
        self.best_timing_mode = self.benchmark_mode()
        log.debug(
            "= Do coordinate descent tuning for %s =",
            self.autotuner.coordesc_tuner.name,
        )
        log.debug(
            "%s: Baseline Config %s, baseline timing %f",
            self.autotuner.coordesc_tuner.name,
            self.best_config,
            self.best_timing,
        )
        self.phase = "fields"

    def _accept_candidate_timings(self, candidate_configs) -> None:
        self._refresh_best_timing_mode()
        assert self.best_timing is not None
        for candidate_config in candidate_configs:
            candidate_timing = self._lookup_timing(candidate_config)
            if self.autotuner.coordesc_tuner.has_improvement(
                self.best_timing, candidate_timing
            ):
                log.debug(
                    "Tune from %s %f -> %s %f",
                    self.best_config,
                    self.best_timing,
                    candidate_config,
                    candidate_timing,
                )
                self.improved_this_pass = True
                self.best_config = candidate_config
                self.best_timing = candidate_timing
                self.best_timing_mode = self.benchmark_mode()

    def _accept_all_direction_timings(self, candidate_configs) -> None:
        self._refresh_best_timing_mode()
        assert self.best_timing is not None
        improved = False
        for candidate_config in candidate_configs:
            candidate_timing = self._lookup_timing(candidate_config)
            if self.autotuner.coordesc_tuner.has_improvement(
                self.best_timing, candidate_timing
            ):
                log.debug(
                    "Tune from %s %f -> %s %f",
                    self.best_config,
                    self.best_timing,
                    candidate_config,
                    candidate_timing,
                )
                improved = True
                self.best_config = candidate_config
                self.best_timing = candidate_timing
                self.best_timing_mode = self.benchmark_mode()

        if improved:
            msg = red_text(
                "%s: Coordinate descent tuning found improvement of %.3fx by looking in all directions."
            )
            old_best_timing = self.old_best_timing_before_all_directions
            if old_best_timing is not None:
                log.debug(
                    msg,
                    self.autotuner.coordesc_tuner.name,
                    old_best_timing / self.best_timing,
                )
            self.phase = "fields"
            self.field_index = 0
            self.improved_this_pass = False
        else:
            self.done = True

    def finalize(self, *, save_cache=True, save_kernel=None) -> None:
        if self.finalized:
            return

        if save_kernel is None:
            save_kernel = self.save_kernel

        coordesc_time_taken_ns = time.time_ns() - self.start_time_ns
        log.debug(
            "%s: Improve from %s %f -> %s %f, %.3fx",
            self.autotuner.coordesc_tuner.name,
            self.initial_launcher.config,
            self._lookup_timing(self.initial_launcher.config),
            self.best_config,
            self._lookup_timing(self.best_config),
            self._lookup_timing(self.initial_launcher.config)
            / self._lookup_timing(self.best_config),
        )
        used_ungrouped_batch_benchmark = self.has_ungrouped_batch_benchmark()
        if used_ungrouped_batch_benchmark and save_cache:
            counters["inductor"]["coordesc_tuning_batch_ungrouped_cache_skips"] += 1
        winner = self.autotuner._finish_coordinate_descent_tuning(
            self.best_config,
            self.config2launcher,
            coordesc_time_taken_ns,
            save_cache=save_cache and not used_ungrouped_batch_benchmark,
            coordinate_descent_tuning_batch=not used_ungrouped_batch_benchmark,
        )
        if save_kernel:
            self._save_winner_kernel(winner)
        self.autotuner.launchers = [winner]
        self.autotuner._cached_launcher = None
        _clear_deferred_compile_state(self.autotuner)
        self.finalized = True

    def run_scalar(
        self,
        *,
        save_cache=True,
        save_kernel=None,
        use_batch_benchmarking: bool | None = None,
    ) -> None:
        if self.finalized:
            return
        if save_kernel is None:
            save_kernel = self.save_kernel
        with _device_guard(self.autotuner.get_device_interface(), self.device_idx):
            winner = self.autotuner.coordinate_descent_tuning(
                self.initial_launcher,
                *self.args,
                save_cache=save_cache,
                use_batch_benchmarking=use_batch_benchmarking,
                clone_args=False,
                **self.kwargs,
            )
        if save_kernel:
            self._save_winner_kernel(winner)
        self.autotuner.launchers = [winner]
        self.autotuner._cached_launcher = None
        _clear_deferred_compile_state(self.autotuner)
        self.finalized = True

    def _save_winner_kernel(self, winner) -> None:
        _save_coordesc_winner_kernel(
            self.autotuner, winner, self.stream, self.save_kernel
        )


class _StaticAutotuneTask(_AutotuneQueueTaskBase):
    can_scalar_drain = True

    def __init__(
        self,
        autotuner,
        args,
        kwargs,
        stream,
        save_kernel,
    ) -> None:
        self._init_queue_task_state(
            autotuner,
            args,
            kwargs,
            stream,
            save_kernel,
        )
        self.launchers = list(autotuner.launchers)
        self.config2launcher = {launcher.config: launcher for launcher in self.launchers}
        self.config2compile_result = {
            triton_config_to_hashable(result.config): result
            for result in autotuner.compile_results
        }
        seen_configs = OrderedSet()
        self.configs = []
        for launcher in self.launchers:
            key = triton_config_to_hashable(launcher.config)
            seen_configs.add(key)
            self.configs.append(launcher.config)
        for config in autotuner.configs or ():
            key = triton_config_to_hashable(config)
            if key in seen_configs:
                continue
            seen_configs.add(key)
            self.configs.append(config)
        self.config_order = {
            triton_config_to_hashable(config): idx
            for idx, config in enumerate(self.configs)
        }
        self.can_scalar_drain = not bool(autotuner.configs)
        self.best_launcher = self.launchers[0]

    def launcher_for_config(self, config):
        launcher = self.config2launcher.get(config)
        if launcher is not None:
            return launcher

        key = triton_config_to_hashable(config)
        compile_result = self.config2compile_result.get(key)
        if compile_result is None:
            compile_result = self.autotuner._precompile_config(config)
            compile_result.config = config
            self.autotuner.compile_results.append(compile_result)
            self.config2compile_result[key] = compile_result
        launcher = compile_result.make_launcher()
        self.config2launcher[config] = launcher
        self.launchers.append(launcher)
        return launcher

    def submit_compile_job(self, config):
        launcher = self.config2launcher.get(config)
        if launcher is not None:
            return launcher
        submitter = getattr(self.autotuner, "_static_config_compile_submitter", None)
        if submitter is None:
            return None
        counters["inductor"]["autotune_queue_process_pool_compiles"] += 1
        return submitter([config])

    def resolve_compile_result(self, config, result):
        if not isinstance(result, tuple):
            return result
        compile_result = self.autotuner._accept_deferred_static_compile_result(
            config, result
        )
        if compile_result is None:
            return None
        self.config2compile_result[triton_config_to_hashable(config)] = compile_result
        launcher = compile_result.make_launcher()
        self.config2launcher[config] = launcher
        self.launchers.append(launcher)
        return launcher

    def cache_benchmark_result(self, config, timing) -> None:
        self.autotuner.coordesc_tuner.cache_benchmark_result(
            config,
            timing,
            benchmark_mode=self.benchmark_mode(),
        )

    def prepare_next_configs(self):
        if self.done:
            return []
        candidate_configs = self.configs
        if self.pending_candidates is None:
            self.pending_candidates = candidate_configs
        uncached = self._uncached_configs(self.pending_candidates)
        if uncached:
            return uncached
        self.accept_pending_results()
        return []

    def accept_pending_results(self) -> None:
        assert self.pending_candidates is not None
        if self._uncached_configs(self.pending_candidates):
            return
        self.pending_candidates = None
        best_config = min(
            self.configs,
            key=lambda config: (
                self._lookup_timing(config),
                self.config_order[triton_config_to_hashable(config)],
            ),
        )
        self.best_launcher = self.launcher_for_config(best_config)
        self.done = True

    def finalize(self, *, save_cache=True, save_kernel=None) -> None:
        if self.finalized:
            return
        if save_kernel is None:
            save_kernel = self.save_kernel

        benchmark_time_taken_ns = time.time_ns() - self.start_time_ns
        best_compile_result = self.config2compile_result.get(
            triton_config_to_hashable(self.best_launcher.config)
        )
        if best_compile_result is not None:
            self.autotuner.compile_results = [best_compile_result]
        self.autotuner.configs = None
        self.autotuner.launchers = [self.best_launcher]
        self.autotuner.autotune_time_taken_ns = (
            self.autotuner.precompile_time_taken_ns + benchmark_time_taken_ns
        )
        TritonBundler.put_winner(self.best_launcher.cache_hash)
        if save_cache and self.autotuner.save_cache_hook:
            self.autotuner.save_cache_hook(
                self.best_launcher.config,
                self.autotuner.autotune_time_taken_ns,
                found_by_coordesc=False,
                coordinate_descent_tuning_batch=False,
                coordinate_descent_tuning_batch_policy=None,
                triton_cache_hash=self.best_launcher.cache_hash,
            )
        static_triton_bundle_key = getattr(
            self.autotuner, "_static_triton_bundle_key", None
        )
        if (
            static_triton_bundle_key is not None
            and self.autotuner.is_statically_launchable()
        ):
            TritonBundler.put_static_autotuner(
                static_triton_bundle_key, self.autotuner
            )
        _clear_deferred_compile_state(self.autotuner)
        if save_kernel:
            _save_coordesc_winner_kernel(
                self.autotuner, self.best_launcher, self.stream, save_kernel
            )
        self.autotuner._cached_launcher = None
        self.finalized = True

    def run_scalar(
        self,
        *,
        save_cache=True,
        save_kernel=None,
        use_batch_benchmarking: bool | None = False,
    ) -> None:
        if self.finalized:
            return
        if save_kernel is None:
            save_kernel = self.save_kernel
        if self.autotuner.configs:
            self.autotuner._finish_deferred_static_precompile(
                use_process_pool=self.autotuner._static_config_compile_submitter
                is not None
            )
            self.launchers = list(self.autotuner.launchers)
            self.config2launcher = {
                launcher.config: launcher for launcher in self.launchers
            }
        else:
            self.autotuner.launchers = list(self.launchers)
        with _device_guard(self.autotuner.get_device_interface(), self.device_idx):
            self.autotuner.autotune_to_one_config(
                *self.args,
                save_cache=save_cache,
                **self.kwargs,
            )
        if save_kernel:
            (winner,) = self.autotuner.launchers
            _save_coordesc_winner_kernel(
                self.autotuner, winner, self.stream, save_kernel
            )
        self.autotuner._cached_launcher = None
        _clear_deferred_compile_state(self.autotuner)
        self.finalized = True

    def should_fallback_on_launch_exception(self, exc: Exception) -> bool:
        return is_invalid_configuration_error(exc)


@dataclasses.dataclass
class _CoordinateDescentCompileJob:
    task: Any
    config: Any
    future_or_launcher: Any


class _AutotuneQueueDrainState:
    def __init__(self, queue, tasks) -> None:
        self.queue = queue
        self.idle_tasks = collections.deque(task for task in tasks if not task.done)
        self.in_flight = []
        self.remaining_by_task: dict[Any, int] = {}
        self.compiled_by_task: dict[Any, list[tuple[Any, Any, Any]]] = (
            collections.defaultdict(list)
        )
        self.ready_tasks: OrderedSet[Any] = OrderedSet()

    def run(self) -> None:
        while self.idle_tasks or self.in_flight or self.ready_tasks:
            self._submit_idle_frontiers()
            self._record_max_inflight_compiles()
            self._collect_ready_compile_jobs()
            if self.ready_tasks:
                self._benchmark_ready_tasks()
                continue
            self._wait_for_compile_progress()

    def _submit_idle_frontiers(self) -> None:
        while self.idle_tasks:
            self._submit_task_frontier(self.idle_tasks.popleft())

    def _submit_task_frontier(self, task) -> None:
        configs = task.prepare_next_configs()
        if not task.has_pending_results():
            return

        pending_configs = [(task, config) for config in configs]
        self.remaining_by_task[task] = 0
        counters["inductor"]["autotune_queue_frontiers"] += 1
        counters["inductor"]["autotune_queue_candidates"] += len(
            pending_configs
        )
        for pending_config_chunk in self.queue._pending_config_chunks(
            pending_configs
        ):
            for pending_task, config in pending_config_chunk:
                job = self.queue._submit_compile_job(pending_task, config)
                if job is None:
                    continue
                self.remaining_by_task[pending_task] += 1
                self.in_flight.append(job)
        if self.remaining_by_task[task] == 0:
            self.ready_tasks.add(task)

    def _record_max_inflight_compiles(self) -> None:
        counters["inductor"]["autotune_queue_max_inflight_compiles"] = max(
            counters["inductor"]["autotune_queue_max_inflight_compiles"],
            len(self.in_flight),
        )

    def _collect_ready_compile_jobs(self) -> None:
        compile_start_ns = time.time_ns()
        for job in list(self.in_flight):
            if not self.queue._compile_job_ready(job):
                continue
            self.in_flight.remove(job)
            compiled = self.queue._resolve_compile_job(job)
            task = job.task
            self.remaining_by_task[task] -= 1
            if compiled is not None:
                self.compiled_by_task[task].append(compiled)
            if self.remaining_by_task[task] == 0:
                self.ready_tasks.add(task)
        counters["inductor"]["autotune_queue_compile_wait_ns"] += (
            time.time_ns() - compile_start_ns
        )

    def _benchmark_ready_tasks(self) -> None:
        if not self.ready_tasks:
            return

        counters["inductor"]["autotune_queue_ready_subset_waves"] += 1
        ready = list(self.ready_tasks)
        self.ready_tasks.clear()
        compiled = []
        for task in ready:
            compiled.extend(self.compiled_by_task.pop(task, ()))

        self.queue._benchmark_compiled_launchers(compiled)

        for task in ready:
            task.accept_pending_results()
            self.remaining_by_task.pop(task, None)
            if not task.done:
                self.idle_tasks.append(task)

    def _wait_for_compile_progress(self) -> None:
        if not self.in_flight:
            return
        wait_start_ns = time.time_ns()
        self.queue._wait_for_ready_compile_job(self.in_flight)
        counters["inductor"]["autotune_queue_compile_wait_ns"] += (
            time.time_ns() - wait_start_ns
        )


class _AutotuneQueue:
    def __init__(
        self,
        expected_calls: int | None = None,
        *,
        disposable_args: bool = False,
    ) -> None:
        self.tasks: list[Any] = []
        self._previous = None
        self._owns_context = False
        self.expected_calls = expected_calls
        self.disposable_args = disposable_args
        self._live_retained_arg_bytes = 0

    def _record_max_live_bytes(self, live_retained_arg_bytes: int) -> None:
        counters["inductor"][
            "autotune_queue_max_live_retained_arg_bytes"
        ] = max(
            counters["inductor"][
                "autotune_queue_max_live_retained_arg_bytes"
            ],
            live_retained_arg_bytes,
        )
        counters["inductor"]["autotune_queue_max_live_bytes"] = max(
            counters["inductor"]["autotune_queue_max_live_bytes"],
            live_retained_arg_bytes,
        )

    def _live_bytes_budget(self) -> int:
        return torch._inductor.config.coordinate_descent_tuning_batch_max_live_bytes

    def _exceeds_single_task_live_bytes_budget(self, queue_bytes: int) -> bool:
        live_bytes_budget = self._live_bytes_budget()
        if live_bytes_budget <= 0:
            return False
        if queue_bytes <= live_bytes_budget:
            return False
        counters["inductor"]["autotune_queue_live_bytes_skips"] += 1
        return True

    def _drain_for_live_bytes_budget(self, queue_bytes: int) -> None:
        live_bytes_budget = self._live_bytes_budget()
        if live_bytes_budget <= 0:
            return
        if (
            self.tasks
            and self._live_retained_arg_bytes + queue_bytes > live_bytes_budget
        ):
            counters["inductor"]["autotune_queue_live_bytes_drains"] += 1
            self.drain()

    def _prepare_for_live_bytes_budget(self, queue_bytes: int) -> bool:
        if self._exceeds_single_task_live_bytes_budget(queue_bytes):
            return False
        self._drain_for_live_bytes_budget(queue_bytes)
        return True

    def _drain_for_live_kernel_budget(self) -> None:
        live_kernel_budget = (
            torch._inductor.config.coordinate_descent_tuning_batch_max_live_kernels
        )
        if live_kernel_budget <= 0:
            return
        if len(self.tasks) >= live_kernel_budget:
            counters["inductor"]["autotune_queue_live_kernel_drains"] += 1
            self.drain()

    @staticmethod
    def _chunks(items, max_items):
        if max_items <= 0 or len(items) <= max_items:
            yield items
            return
        for start in range(0, len(items), max_items):
            yield items[start : start + max_items]

    def _frontier_candidate_limit(self) -> int:
        return (
            torch._inductor.config.coordinate_descent_tuning_batch_max_frontier_candidates
        )

    def _pending_config_chunks(self, pending_configs):
        max_frontier_candidates = self._frontier_candidate_limit()
        is_split = (
            max_frontier_candidates > 0
            and len(pending_configs) > max_frontier_candidates
        )
        if is_split:
            counters["inductor"]["autotune_queue_frontier_splits"] += 1
        for chunk in self._chunks(pending_configs, max_frontier_candidates):
            if is_split:
                counters["inductor"]["autotune_queue_frontier_chunks"] += 1
            yield chunk

    def __enter__(self):
        self._previous = get_active_autotune_queue()
        if self._previous is not None:
            return self._previous
        if not self.disposable_args:
            return self
        if not torch._inductor.config.coordinate_descent_tuning_batch:
            return self
        if not _coordinate_descent_batch_has_compile_parallelism():
            counters["inductor"]["autotune_queue_compile_thread_skips"] += 1
            return self

        min_kernels = max(
            1, torch._inductor.config.coordinate_descent_tuning_batch_min_kernels
        )
        if self.expected_calls is not None and self.expected_calls < min_kernels:
            return self

        if self.expected_calls is not None:
            from torch._inductor.async_compile import AsyncCompile

            AsyncCompile.wakeup()
        self._owns_context = True
        _set_active_autotune_queue(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if not self._owns_context:
            return False
        try:
            if exc_type is None:
                self.drain()
        finally:
            if self._previous is None:
                _clear_active_autotune_queue()
            else:
                _set_active_autotune_queue(self._previous)
        return False

    def _add_task(self, task):
        if not self._prepare_for_live_bytes_budget(task.queue_bytes):
            task.run_scalar()
            return False

        counters["inductor"]["autotune_queue_tasks"] += 1
        counters["inductor"]["autotune_queue_max_live_tasks"] = max(
            counters["inductor"]["autotune_queue_max_live_tasks"],
            len(self.tasks) + 1,
        )
        self.tasks.append(task)
        self._live_retained_arg_bytes += task.retained_arg_bytes
        self._record_max_live_bytes(self._live_retained_arg_bytes)
        return task

    def remove_task(self, task) -> None:
        try:
            self.tasks.remove(task)
        except ValueError:
            return
        self._live_retained_arg_bytes = max(
            0, self._live_retained_arg_bytes - task.retained_arg_bytes
        )

    def enqueue(self, autotuner, launcher, args, kwargs, stream, save_kernel):
        if get_active_autotune_queue() is not self:
            return False
        if not self.disposable_args:
            return False
        if _requires_benchmark_arg_clones(autotuner):
            counters["inductor"]["autotune_queue_clone_arg_skips"] += 1
            return False

        estimated_queue_bytes = _estimate_retained_arg_bytes(args, kwargs)
        if not self._prepare_for_live_bytes_budget(estimated_queue_bytes):
            _run_scalar_coordesc_fallback(
                autotuner,
                launcher,
                args,
                kwargs,
                stream,
                save_kernel,
            )
            return False
        self._drain_for_live_kernel_budget()

        task = _CoordinateDescentTask(
            autotuner,
            launcher,
            args,
            kwargs,
            stream,
            save_kernel,
        )
        return self._add_task(task)

    def enqueue_static(self, autotuner, args, kwargs, stream, save_kernel):
        if get_active_autotune_queue() is not self:
            return False
        if not self.disposable_args:
            return False
        if _requires_benchmark_arg_clones(autotuner):
            counters["inductor"]["autotune_queue_clone_arg_skips"] += 1
            return False

        estimated_queue_bytes = _estimate_retained_arg_bytes(args, kwargs)
        if not self._prepare_for_live_bytes_budget(estimated_queue_bytes):
            return False
        self._drain_for_live_kernel_budget()

        task = _StaticAutotuneTask(
            autotuner,
            args,
            kwargs,
            stream,
            save_kernel,
        )
        return self._add_task(task)

    def _submit_compile_job(self, task, config):
        try:
            submit_compile_job = getattr(task, "submit_compile_job", None)
            future_or_launcher = (
                submit_compile_job(config)
                if submit_compile_job is not None
                else task.config2launcher.get(config)
            )
            if future_or_launcher is None:
                raise RuntimeError(
                    "Queued autotune config compilation requires a process-pool "
                    "compile submitter"
                )
            return _CoordinateDescentCompileJob(task, config, future_or_launcher)
        except Exception as e:
            if not _is_expected_compile_config_failure(e):
                raise
            log.debug("COORDESC: got exception %s", e)
            task.cache_benchmark_result(config, float("inf"))
            return None

    @staticmethod
    def _compile_job_ready(job) -> bool:
        future_or_launcher = job.future_or_launcher
        return not hasattr(future_or_launcher, "done") or future_or_launcher.done()

    @staticmethod
    def _wait_for_ready_compile_job(jobs) -> None:
        if any(_AutotuneQueue._compile_job_ready(job) for job in jobs):
            return
        from concurrent.futures import FIRST_COMPLETED, Future, wait

        futures = [
            job.future_or_launcher
            for job in jobs
            if isinstance(job.future_or_launcher, Future)
        ]
        if futures:
            wait(futures, return_when=FIRST_COMPLETED)
            return
        for job in jobs:
            if hasattr(job.future_or_launcher, "result"):
                job.future_or_launcher.result()
                return

    def _resolve_compile_job(self, job):
        task = job.task
        config = job.config
        future_or_launcher = job.future_or_launcher
        try:
            if hasattr(future_or_launcher, "result"):
                result = future_or_launcher.result()
            else:
                result = future_or_launcher
            resolve_compile_result = getattr(task, "resolve_compile_result", None)
            if resolve_compile_result is not None:
                result = resolve_compile_result(config, result)
                if result is None:
                    task.cache_benchmark_result(config, float("inf"))
                    return None
            return task, config, result
        except Exception as e:
            from torch._inductor.compile_worker.subproc_pool import SubprocException

            if isinstance(e, SubprocException):
                e = e.with_name(task.autotuner.fn.__name__)
            if not _is_expected_compile_config_failure(e):
                raise e
            log.debug("COORDESC: got exception %s", e)
            task.cache_benchmark_result(config, float("inf"))
            return None

    def _benchmark_compiled_launchers(self, compiled) -> None:
        for compiled_chunk in self._chunks(compiled, self._frontier_candidate_limit()):
            if not compiled_chunk:
                continue
            requests = [
                task.benchmark_request(launcher)
                for task, config, launcher in compiled_chunk
            ]
            try:
                benchmark_start_ns = time.time_ns()
                timings = _benchmark_launcher_requests(requests)
                counters["inductor"]["autotune_queue_benchmark_ns"] += (
                    time.time_ns() - benchmark_start_ns
                )
            except Exception as e:
                counters["inductor"]["autotune_queue_grouped_fallbacks"] += 1
                log.debug("COORDESC: grouped benchmark failed: %s", e)
                timings = []
                for task, config, launcher in compiled_chunk:
                    task.mark_ungrouped_batch_benchmark()
                    try:
                        timing = task.benchmark_launcher(launcher)
                    except Exception as e:
                        if not is_invalid_configuration_error(e):
                            raise
                        log.debug("COORDESC: got exception %s", e)
                        timing = float("inf")
                    timings.append(timing)

            for (task, config, launcher), timing in zip(compiled_chunk, timings):
                task.cache_benchmark_result(config, timing)
                log.debug(
                    "COORDESC: %s: %f, nreg %d, nspill %d, #shared-mem %d",
                    launcher.config,
                    timing,
                    launcher.n_regs,
                    launcher.n_spills,
                    launcher.shared,
                )

    def drain(self) -> None:
        if not self.tasks:
            return

        drain_start_ns = time.time_ns()
        if len(self.tasks) == 1 and getattr(self.tasks[0], "can_scalar_drain", True):
            counters["inductor"]["autotune_queue_single_task_drains"] += 1
            task = self.tasks.pop()
            task.run_scalar(use_batch_benchmarking=None)
            self._live_retained_arg_bytes = 0
            counters["inductor"]["autotune_queue_drain_ns"] += (
                time.time_ns() - drain_start_ns
            )
            return

        with dynamo_timed(
            "CachingAutotuner.autotune_queue_drain",
            log_pt2_compile_event=False,
            metadata={"num_kernels": len(self.tasks)},
            dynamo_compile_column_us="runtime_triton_autotune_time_us",
            log_waitcounter=True,
            waitcounter_name_override="triton_autotuner",
        ):
            _AutotuneQueueDrainState(self, self.tasks).run()

        drained_tasks = self.tasks
        self.tasks = []
        self._live_retained_arg_bytes = 0
        finalize_start_ns = time.time_ns()
        for task in drained_tasks:
            task.finalize()
        counters["inductor"]["autotune_queue_finalize_ns"] += (
            time.time_ns() - finalize_start_ns
        )
        counters["inductor"]["autotune_queue_drain_ns"] += (
            time.time_ns() - drain_start_ns
        )


def autotune_queue(
    expected_calls: int | None = None,
    *,
    disposable_args: bool = False,
):
    return _AutotuneQueue(
        expected_calls,
        disposable_args=disposable_args,
    )


def coordinate_descent_batch(
    expected_calls: int | None = None,
    *,
    disposable_args: bool = False,
):
    return autotune_queue(
        expected_calls,
        disposable_args=disposable_args,
    )


_CoordinateDescentBatch = _AutotuneQueue
