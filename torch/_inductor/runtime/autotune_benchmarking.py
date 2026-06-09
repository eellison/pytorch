# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import contextlib
import dataclasses
from typing import Any, Callable

import torch
from torch.utils._debug_mode import DebugMode

from .autotune_common import (
    BenchmarkFailureReason,
    _COORDESC_UNGROUPED_BENCHMARK_KEY,
)
from .benchmarking import benchmarker


@dataclasses.dataclass
class _LauncherBenchmarkRequest:
    autotuner: Any
    launcher: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    device_idx: int | None = None
    wait_ready_fn: Callable[[], Any] | None = None
    setup_fn: Callable[[], Any] | None = None
    clone_args: bool = True
    benchmark_group_key: Any | None = None
    benchmark_group_state: dict[str, int] | None = None


def _device_guard(device_interface, device_idx):
    device_idx = device_idx if isinstance(device_idx, int) else None
    if device_idx is None or not device_interface.is_available():
        return contextlib.nullcontext()
    return device_interface.device(device_idx)


def _mark_ungrouped_coordesc_benchmark(request: _LauncherBenchmarkRequest) -> None:
    if (
        request.benchmark_group_key is not None
        and request.benchmark_group_state is not None
    ):
        request.benchmark_group_state[_COORDESC_UNGROUPED_BENCHMARK_KEY] = 1


def _benchmark_many_supports_grouped_launcher_requests(device) -> bool:
    benchmark_device = torch.device(device) if isinstance(device, str) else device
    if benchmark_device is None or benchmark_device.type != "cuda":
        return False
    return bool(getattr(benchmarker, "supports_grouped_benchmark_many", False))


def _benchmark_device(device_type: str, device_idx: int | None):
    if device_type == "hip":
        device_type = "cuda"
    device_idx = device_idx if isinstance(device_idx, int) else None
    if device_idx is None:
        return device_type
    if device_type in ("cuda", "xpu"):
        return torch.device(device_type, device_idx)
    return device_type


def _benchmark_kwargs(device_type: str) -> dict[str, Any]:
    return {} if device_type == "cpu" else {"rep": 40, "is_vetted_benchmarking": True}


def _benchmark_launcher_requests(
    requests: list[_LauncherBenchmarkRequest],
) -> list[float]:
    if not requests:
        return []

    results: list[float | None] = [None] * len(requests)
    calls_by_device: dict[
        tuple[str, int | None],
        list[tuple[int, Callable[[], None], Callable[[], Any], Any | None]],
    ] = collections.defaultdict(list)

    for idx, request in enumerate(requests):
        autotuner = request.autotuner
        launcher = request.launcher
        if autotuner._skip_config_due_to_register_spilling(launcher):
            results[idx] = float("inf")
            continue

        device_interface = autotuner.get_device_interface()
        device_idx = (
            request.device_idx
            if request.device_idx is not None
            else autotuner.device_props.index
        )
        device_idx = device_idx if isinstance(device_idx, int) else None
        device_type = autotuner.device_props.type
        if (
            device_idx is None
            and device_type != "cpu"
            and device_interface.is_available()
        ):
            device_idx = device_interface.current_device()
        with _device_guard(device_interface, device_idx):
            if device_type == "cpu":
                stream = 0
            else:
                # benchmark_many records CUDA events on the active stream, so launch
                # the candidate on that stream instead of a saved runtime stream.
                stream = device_interface.get_raw_stream(
                    device_idx
                    if device_idx is not None
                    else device_interface.current_device()
                )
            wait_ready_fn = request.wait_ready_fn or (lambda: None)
            setup_fn = request.setup_fn or (lambda: None)
            wait_ready_fn()
            cpu_copies = (
                autotuner.copy_args_to_cpu_if_needed(*request.args, **request.kwargs)
                if request.clone_args
                else {}
            )
            benchmark_call = autotuner._make_benchmark_call(
                launcher,
                cpu_copies,
                stream,
                request.args,
                request.kwargs,
                clone_args=request.clone_args,
            )
            if cpu_copies:
                _mark_ungrouped_coordesc_benchmark(request)
                with DebugMode._benchmarking_inductor(), _device_guard(
                    device_interface, device_idx
                ):
                    result = benchmarker.benchmark(
                        benchmark_call,
                        device=_benchmark_device(device_type, device_idx),
                        **_benchmark_kwargs(device_type),  # type: ignore[arg-type]
                    )
                results[idx] = result
                if result == float("inf"):
                    autotuner.benchmark_failure_reasons[launcher] = (
                        BenchmarkFailureReason.INVALID_CONFIG
                    )
                continue
            calls_by_device[(device_type, device_idx)].append(
                (
                    idx,
                    benchmark_call,
                    setup_fn,
                    request.benchmark_group_key,
                )
            )

    for (device_type, device_idx), indexed_calls in calls_by_device.items():
        benchmark_kwargs = _benchmark_kwargs(device_type)
        first_request = requests[indexed_calls[0][0]]
        device_interface = first_request.autotuner.get_device_interface()
        benchmark_group_keys = [group_key for _, _, _, group_key in indexed_calls]
        has_grouped_requests = any(
            group_key is not None for group_key in benchmark_group_keys
        )
        benchmark_device = _benchmark_device(device_type, device_idx)
        use_grouped_benchmarking = (
            has_grouped_requests
            and _benchmark_many_supports_grouped_launcher_requests(benchmark_device)
        )

        def benchmark_call_group(calls, call_kwargs):
            with DebugMode._benchmarking_inductor(), _device_guard(
                device_interface, device_idx
            ):
                return benchmarker.benchmark_many(
                    [call for _, call, _, _ in calls],
                    device=benchmark_device,
                    setup_fns=[setup_fn for _, _, setup_fn, _ in calls],
                    **call_kwargs,  # type: ignore[arg-type]
                )

        if use_grouped_benchmarking:
            benchmark_call_groups = [
                (
                    [call for call in indexed_calls if call[3] is not None],
                    {
                        **benchmark_kwargs,
                        "benchmark_group_keys": [
                            group_key
                            for _, _, _, group_key in indexed_calls
                            if group_key is not None
                        ],
                        "benchmark_group_states": [
                            requests[idx].benchmark_group_state
                            for idx, _, _, group_key in indexed_calls
                            if group_key is not None
                        ],
                    },
                ),
                (
                    [call for call in indexed_calls if call[3] is None],
                    benchmark_kwargs,
                ),
            ]
        else:
            if has_grouped_requests:
                for idx, _, _, group_key in indexed_calls:
                    if group_key is not None:
                        _mark_ungrouped_coordesc_benchmark(requests[idx])
            benchmark_call_groups = [(indexed_calls, benchmark_kwargs)]

        for benchmark_calls, call_kwargs in benchmark_call_groups:
            if not benchmark_calls:
                continue
            benchmark_results = benchmark_call_group(benchmark_calls, call_kwargs)
            if len(benchmark_results) != len(benchmark_calls):
                raise RuntimeError(
                    "Grouped launcher benchmark returned "
                    f"{len(benchmark_results)} results for "
                    f"{len(benchmark_calls)} requests"
                )
            for (idx, _, _, _), result in zip(benchmark_calls, benchmark_results):
                results[idx] = result
                if result == float("inf"):
                    request = requests[idx]
                    request.autotuner.benchmark_failure_reasons[request.launcher] = (
                        BenchmarkFailureReason.INVALID_CONFIG
                    )

    return [float("inf") if result is None else result for result in results]


def benchmark_launchers(
    autotuner: Any,
    launchers,
    args,
    kwargs,
    *,
    device_idx: int | None = None,
    clone_args: bool = True,
    benchmark_group_key=None,
    benchmark_group_state=None,
):
    launchers = list(launchers)
    timings = _benchmark_launcher_requests(
        [
            _LauncherBenchmarkRequest(
                autotuner=autotuner,
                launcher=launcher,
                args=args,
                kwargs=kwargs,
                device_idx=device_idx,
                clone_args=clone_args,
                benchmark_group_key=benchmark_group_key,
                benchmark_group_state=benchmark_group_state,
            )
            for launcher in launchers
        ]
    )
    return dict(zip(launchers, timings))
