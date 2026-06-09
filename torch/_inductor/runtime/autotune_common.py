# mypy: allow-untyped-defs
from __future__ import annotations

import enum
import logging
from typing import Any

import torch
from torch.utils._ordered_set import OrderedSet

from .hints import HeuristicType
from .triton_compat import IntelGPUError, OutOfResources, PTXASError


log = logging.getLogger(__name__)


_InductorMetaTy = dict[str, object]
_COORDESC_BATCH_POLICIES = {"auto", "all", "reductions", "none"}
_COORDESC_BATCH_REQUESTED_KEY = "coordinate_descent_tuning_batch_requested"
_COORDESC_BATCH_EFFECTIVE_KEY = "coordinate_descent_tuning_batch"
_STATIC_AUTOTUNE_BATCH_MIN_CONFIGS = 3


def _coordinate_descent_batch_requested_from_meta(
    inductor_meta: _InductorMetaTy,
) -> bool:
    from torch._inductor import config

    return bool(
        inductor_meta.get(
            _COORDESC_BATCH_REQUESTED_KEY,
            inductor_meta.get(
                _COORDESC_BATCH_EFFECTIVE_KEY,
                config.autotune_queue,
            ),
        )
    )


def _coordinate_descent_batch_policy(inductor_meta: _InductorMetaTy) -> str:
    from torch._inductor import config

    return str(
        inductor_meta.get(
            "coordinate_descent_tuning_batch_policy",
            config._coordinate_descent_tuning_batch_default_policy,
        )
    ).lower()


def _coordinate_descent_batch_has_compile_parallelism() -> bool:
    from torch._inductor import config
    from torch._inductor.async_compile import get_compile_threads

    return get_compile_threads() > 1 and not config.triton.proton_profiling


def _coordinate_descent_batch_enabled_from_meta(
    inductor_meta: _InductorMetaTy,
    *,
    check_compile_parallelism: bool = True,
) -> bool:
    if not inductor_meta.get("coordinate_descent_tuning", False):
        return False
    if not inductor_meta.get(_COORDESC_BATCH_EFFECTIVE_KEY, False):
        return False
    policy = _coordinate_descent_batch_policy(inductor_meta)
    if policy not in _COORDESC_BATCH_POLICIES or policy == "none":
        return False
    if (
        check_compile_parallelism
        and not _coordinate_descent_batch_has_compile_parallelism()
    ):
        return False
    return True


def _has_effective_coordinate_descent_batch_metadata(
    inductor_meta: _InductorMetaTy,
) -> bool:
    return _COORDESC_BATCH_EFFECTIVE_KEY in inductor_meta


def _has_explicit_coordinate_descent_batch_request_metadata(
    inductor_meta: _InductorMetaTy,
) -> bool:
    return _COORDESC_BATCH_REQUESTED_KEY in inductor_meta


def _can_coordinate_descent_tune_for(
    heuristic_type: HeuristicType,
    deterministic_mode: bool,
) -> bool:
    if heuristic_type in (
        HeuristicType.TEMPLATE,
        HeuristicType.USER_AUTOTUNE,
        HeuristicType.FIXED,
    ):
        return False

    if deterministic_mode and heuristic_type in (
        HeuristicType.REDUCTION,
        HeuristicType.PERSISTENT_REDUCTION,
        HeuristicType.SPLIT_SCAN,
    ):
        return False

    return True


def _coordinate_descent_batch_supported_device(
    triton_meta: dict[str, Any],
) -> bool:
    device_type = triton_meta.get("device_type")
    if device_type is None:
        device = triton_meta.get("device")
        device_type = getattr(device, "type", None)
        if device_type is None and isinstance(device, torch.device):
            device_type = device.type
        elif device_type is None and isinstance(device, str):
            device_type = device
    if isinstance(device_type, torch.device):
        device_type = device_type.type
    elif isinstance(device_type, str):
        device_type = device_type.split(":", 1)[0]

    return device_type in ("cuda", "hip")


def _autotune_batch_enabled_for(
    inductor_meta: dict[str, Any],
    triton_meta: dict[str, Any],
    *,
    check_compile_parallelism: bool = True,
) -> bool:
    if not _coordinate_descent_batch_requested_from_meta(inductor_meta):
        return False
    if _coordinate_descent_batch_policy(inductor_meta) == "none":
        return False
    if not _coordinate_descent_batch_supported_device(triton_meta):
        return False
    if (
        check_compile_parallelism
        and not _coordinate_descent_batch_has_compile_parallelism()
    ):
        return False
    return True


def _static_autotune_batch_enabled_for(
    inductor_meta: dict[str, Any],
    triton_meta: dict[str, Any],
    heuristic_type: HeuristicType,
    *,
    check_compile_parallelism: bool = True,
) -> bool:
    policy = _coordinate_descent_batch_policy(inductor_meta)
    if policy not in ("auto", "all"):
        return False
    if policy == "auto" and not (
        heuristic_type
        in (
            HeuristicType.REDUCTION,
            HeuristicType.PERSISTENT_REDUCTION,
            HeuristicType.SPLIT_SCAN,
        )
        or bool(triton_meta.get("native_matmul", False))
    ):
        return False
    return _autotune_batch_enabled_for(
        inductor_meta,
        triton_meta,
        check_compile_parallelism=check_compile_parallelism,
    )


def _coordinate_descent_batch_enabled_for(
    inductor_meta: dict[str, Any],
    triton_meta: dict[str, Any],
    heuristic_type: HeuristicType,
    deterministic_mode: bool,
    *,
    check_compile_parallelism: bool = True,
) -> bool:
    if not inductor_meta.get("coordinate_descent_tuning", False):
        return False
    if not _can_coordinate_descent_tune_for(heuristic_type, deterministic_mode):
        return False
    if not _autotune_batch_enabled_for(
        inductor_meta,
        triton_meta,
        check_compile_parallelism=check_compile_parallelism,
    ):
        return False

    policy = _coordinate_descent_batch_policy(inductor_meta)
    if policy == "none":
        return False
    if policy == "all":
        return True

    is_reduction = heuristic_type in (
        HeuristicType.REDUCTION,
        HeuristicType.PERSISTENT_REDUCTION,
        HeuristicType.SPLIT_SCAN,
    )
    if policy == "reductions":
        return is_reduction
    if policy == "auto":
        return is_reduction or bool(triton_meta.get("native_matmul", False))

    log.warning(
        "Unknown coordinate_descent_tuning_batch_policy=%s; disabling autotune queue",
        policy,
    )
    return False


def _coordinate_descent_batch_enabled_for_kernel(
    kernel: Any,
    *,
    check_compile_parallelism: bool = True,
) -> bool:
    inductor_meta = kernel.inductor_meta
    if _has_explicit_coordinate_descent_batch_request_metadata(
        inductor_meta
    ) and _has_effective_coordinate_descent_batch_metadata(inductor_meta):
        return _coordinate_descent_batch_enabled_from_meta(
            inductor_meta,
            check_compile_parallelism=check_compile_parallelism,
        )
    return _coordinate_descent_batch_enabled_for(
        inductor_meta,
        kernel.triton_meta,
        kernel.heuristic_type,
        kernel.deterministic_mode,
        check_compile_parallelism=check_compile_parallelism,
    )


def apply_effective_coordinate_descent_queue_metadata(
    inductor_meta: dict[str, Any],
    triton_meta: dict[str, Any],
    heuristic_type: HeuristicType,
    deterministic_mode: bool,
    *,
    keep_disabled: bool = False,
) -> bool:
    from torch._inductor import config

    if not inductor_meta.get("coordinate_descent_tuning", False):
        return False

    enabled = (
        config.autotune_queue
        and _coordinate_descent_batch_enabled_for(
            inductor_meta,
            triton_meta,
            heuristic_type,
            deterministic_mode,
            check_compile_parallelism=False,
        )
    )
    if enabled or keep_disabled:
        inductor_meta[_COORDESC_BATCH_EFFECTIVE_KEY] = enabled
    else:
        inductor_meta.pop(_COORDESC_BATCH_EFFECTIVE_KEY, None)
    return enabled


def _static_autotune_config_count(kernel: Any) -> int:
    launchers = getattr(kernel, "launchers", ())
    configs = getattr(kernel, "configs", ())
    compile_results = getattr(kernel, "compile_results", ())
    if configs is not None and launchers:
        return len(launchers) + len(configs)
    if len(launchers) > 1:
        return len(launchers)
    if configs is not None and compile_results:
        return len(compile_results) + len(configs)
    if len(compile_results) > 1:
        return len(compile_results)
    return 0 if configs is None else len(configs)


def _mutated_input_arg_names(autotuner: Any) -> OrderedSet[str]:
    inductor_meta = getattr(autotuner, "inductor_meta", {})
    if "mutated_input_arg_names" in inductor_meta:
        return OrderedSet(inductor_meta.get("mutated_input_arg_names", ()))
    return OrderedSet(getattr(autotuner, "mutated_arg_names", ()))


def _benchmark_clone_arg_names(autotuner: Any) -> OrderedSet[str]:
    out = OrderedSet(_mutated_input_arg_names(autotuner))
    for name in getattr(autotuner, "reset_to_zero_arg_names", ()):
        out.add(name)
    return out


def _requires_benchmark_arg_clones(autotuner: Any) -> bool:
    clone_arg_names = getattr(autotuner, "_benchmark_clone_arg_names", None)
    clone_names = (
        clone_arg_names()
        if clone_arg_names is not None and hasattr(autotuner, "inductor_meta")
        else _benchmark_clone_arg_names(autotuner)
    )
    return bool(clone_names)


def _is_caching_autotuner_like(kernel: Any) -> bool:
    return (
        hasattr(kernel, "inductor_meta")
        and hasattr(kernel, "triton_meta")
        and hasattr(kernel, "heuristic_type")
        and hasattr(kernel, "launchers")
    )


def _kernel_would_defer_static_autotune(kernel: Any) -> bool:
    if not _is_caching_autotuner_like(kernel):
        return False
    if _requires_benchmark_arg_clones(kernel):
        return False
    if kernel.inductor_meta.get("coordinate_descent_tuning", False):
        return False
    if kernel.inductor_meta.get("combo_tuning_groups"):
        return False
    if not _static_autotune_batch_enabled_for(
        kernel.inductor_meta,
        kernel.triton_meta,
        kernel.heuristic_type,
        check_compile_parallelism=False,
    ):
        return False

    return _static_autotune_config_count(kernel) >= _STATIC_AUTOTUNE_BATCH_MIN_CONFIGS


def _should_defer_static_autotune_precompile(kernel: Any) -> bool:
    from torch._inductor import config

    if not config.autotune_queue_static_precompile:
        return False
    if not _is_caching_autotuner_like(kernel):
        return False
    if kernel.inductor_meta.get("profile_bandwidth", False):
        return False
    if not _coordinate_descent_batch_has_compile_parallelism():
        return False
    if kernel.compile_results or kernel.launchers:
        return False
    if not _kernel_would_defer_static_autotune(kernel):
        return False
    if (
        kernel.heuristic_type == HeuristicType.REDUCTION
        and kernel.inductor_meta.get("dynamic_scale_rblock", True)
    ):
        return False
    return True


def _has_deferred_static_autotune_precompile(kernel: Any) -> bool:
    if not _is_caching_autotuner_like(kernel):
        return False
    if not kernel.compile_results or kernel.launchers or not kernel.configs:
        return False
    return _kernel_would_defer_static_autotune(kernel)


def _kernel_would_defer_coordinate_descent(kernel: Any) -> bool:
    if not _is_caching_autotuner_like(kernel):
        return False
    if _requires_benchmark_arg_clones(kernel):
        return False
    if not kernel.inductor_meta.get("coordinate_descent_tuning", False):
        return False
    if not _coordinate_descent_batch_enabled_for_kernel(
        kernel,
        check_compile_parallelism=False,
    ):
        return False

    launchers = getattr(kernel, "launchers", ())
    if len(launchers) == 1 and getattr(
        launchers[0].config, "found_by_coordesc", False
    ):
        return False
    return True


def expected_coordinate_descent_batch_calls(kernels: Any) -> int:
    return sum(
        1
        for kernel in kernels
        if _kernel_would_defer_coordinate_descent(kernel)
        or _kernel_would_defer_static_autotune(kernel)
    )


def expected_autotune_queue_calls(kernels: Any) -> int:
    return expected_coordinate_descent_batch_calls(kernels)


class BenchmarkFailureReason(enum.Enum):
    """Reasons why a triton config benchmark may return float('inf')."""

    REGISTER_SPILLING = "register_spilling"
    INVALID_CONFIG = "invalid_config"


_COORDESC_UNGROUPED_BENCHMARK_KEY = "_coordinate_descent_ungrouped_benchmark"
_COORDESC_BENCHMARK_MODE_GROUPED = "grouped"
_COORDESC_BENCHMARK_MODE_UNGROUPED = "ungrouped"


class NoTritonConfigsError(RuntimeError):
    pass


def _has_ungrouped_coordesc_benchmark(state: dict[str, int] | None) -> bool:
    return bool(state and state.get(_COORDESC_UNGROUPED_BENCHMARK_KEY))


def _is_expected_compile_config_failure(e: Exception) -> bool:
    if isinstance(
        e,
        (OutOfResources, PTXASError, IntelGPUError, torch.cuda.OutOfMemoryError),
    ):
        return True
    if isinstance(e, NoTritonConfigsError):
        return any(
            name in str(e)
            for name in (
                "OutOfResources",
                "PTXASError",
                "IntelGPUError",
                "OutOfMemoryError",
            )
        )

    from torch._inductor.compile_worker.subproc_pool import SubprocException

    if isinstance(e, SubprocException):
        return any(
            name in e.details
            for name in (
                "OutOfResources",
                "PTXASError",
                "IntelGPUError",
                "OutOfMemoryError",
                "No valid triton configs",
            )
        )
    return False
