from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("bench_worker.py")
WORKTREE_RUNNER = ROOT / "agent_space" / "run_worktree_script.py"
PYTHON = Path("/home/eellison/.conda/envs/pytorch-3.12/bin/python")
SENTINEL = "PADDING_BENCH_RESULT="

BLOCKED_CORE = ((19, 4096), (128, 4096), (129, 4128), (989, 4096))
XDL_CORE = ((95, 3072), (96, 3072), (97, 3104), (2048, 3072))
BLOCKED_FULL = BLOCKED_CORE + ((1, 4096), (1024, 4096), (16000, 8192))
XDL_FULL = XDL_CORE + ((1, 3072), (192, 3072), (2049, 3200))


def case(
    workload: str,
    variant: str,
    shape: tuple[int, int],
    tuning: str,
) -> dict[str, Any]:
    return {
        "workload": workload,
        "variant": variant,
        "rows": shape[0],
        "hidden": shape[1],
        "tuning": tuning,
    }


def make_cases(
    preset: str,
    include_pad_scatter: bool,
    include_predicated: bool,
    include_auxiliary: bool,
) -> list[dict[str, Any]]:
    blocked_shapes = BLOCKED_FULL if preset == "full" else BLOCKED_CORE
    xdl_shapes = XDL_FULL if preset == "full" else XDL_CORE
    if preset == "smoke":
        blocked_shapes = BLOCKED_CORE[:2]
        xdl_shapes = XDL_CORE[1:3]

    cases = []
    blocked_variants = ["fpad", "fill_scatter"]
    if include_pad_scatter:
        blocked_variants.append("pad_scatter")
    if include_predicated:
        blocked_variants.append("predicated")
    xdl_variants = ["fpad", "fill_scatter"]
    if include_auxiliary:
        xdl_variants.append("auxiliary")

    for shape in blocked_shapes:
        for variant in blocked_variants:
            cases.append(case("swizzle_mxfp4", variant, shape, "default"))
    for shape in xdl_shapes:
        for variant in xdl_variants:
            cases.append(case("swizzle_dcn", variant, shape, "default"))

    for workload in (
        "mxfp4_quant",
        "rmsnorm_mxfp4",
        "rmsnorm_nvfp4",
        "mxfp8_quant",
        "rmsnorm_mxfp8",
    ):
        for shape in blocked_shapes:
            for tuning in ("default", "coordesc"):
                for variant in blocked_variants:
                    cases.append(case(workload, variant, shape, tuning))
            cases.append(case(workload, "flashinfer", shape, "external"))

    for workload in ("mxfp6_quant", "rmsnorm_mxfp6", "dcn_mxfp6"):
        for shape in xdl_shapes:
            for tuning in ("default", "coordesc"):
                for variant in xdl_variants:
                    cases.append(case(workload, variant, shape, tuning))
    return cases


def case_key(item: dict[str, Any]) -> str:
    return ":".join(
        str(item[key])
        for key in ("workload", "variant", "rows", "hidden", "tuning")
    )


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "ok":
            completed.add(record.get("case_key", case_key(record)))
    return completed


def run_case(
    args: argparse.Namespace, item: dict[str, Any], gpu: str
) -> dict[str, Any]:
    command = [
        str(PYTHON),
        str(WORKTREE_RUNNER),
        str(args.worktree),
        str(WORKER),
        "--workload",
        item["workload"],
        "--variant",
        item["variant"],
        "--rows",
        str(item["rows"]),
        "--hidden",
        str(item["hidden"]),
        "--tuning",
        "default" if item["tuning"] == "external" else item["tuning"],
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--calls-per-graph",
        str(args.calls_per_graph),
        "--cooldown",
        str(args.cooldown),
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
        }
    )
    started = time.time()
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    record = {
        **item,
        "case_key": case_key(item),
        "matrix_tuning": item["tuning"],
        "physical_gpu": gpu,
        "status": "error",
        "returncode": process.returncode,
        "process_seconds": time.time() - started,
    }
    for line in process.stdout.splitlines():
        if line.startswith(SENTINEL):
            record.update(json.loads(line.removeprefix(SENTINEL)))
            record["matrix_tuning"] = item["tuning"]
            record["status"] = "ok"
    if record["status"] != "ok":
        record["output"] = process.stdout[-12000:]
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "core", "full"), default="core")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--gpus")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--cooldown", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--no-pad-scatter", action="store_true")
    parser.add_argument("--predicated", action="store_true")
    parser.add_argument("--auxiliary", action="store_true")
    parser.add_argument("--workloads")
    parser.add_argument("--variants")
    parser.add_argument("--shapes")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    args.worktree = args.worktree.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = make_cases(
        args.preset,
        not args.no_pad_scatter,
        args.predicated,
        args.auxiliary,
    )
    if args.workloads:
        workloads = set(args.workloads.split(","))
        cases = [item for item in cases if item["workload"] in workloads]
    if args.variants:
        variants = set(args.variants.split(","))
        cases = [item for item in cases if item["variant"] in variants]
    if args.shapes:
        shapes = {
            tuple(map(int, value.lower().split("x")))
            for value in args.shapes.split(",")
        }
        cases = [
            item for item in cases if (item["rows"], item["hidden"]) in shapes
        ]
    random.Random(args.seed).shuffle(cases)
    completed = set() if args.no_resume else load_completed(args.output)
    pending = [item for item in cases if case_key(item) not in completed]
    print(
        f"matrix={len(cases)} complete={len(cases) - len(pending)} "
        f"pending={len(pending)} output={args.output}",
        flush=True,
    )
    output = args.output.open("a")
    lock = threading.Lock()

    def run_queue(gpu: str, queue: list[dict[str, Any]]) -> None:
        for index, item in enumerate(queue, 1):
            print(
                f"[gpu {gpu} {index}/{len(queue)}] {case_key(item)}",
                flush=True,
            )
            result = run_case(args, item, gpu)
            with lock:
                output.write(json.dumps(result, sort_keys=True) + "\n")
                output.flush()
            if result["status"] == "ok":
                print(
                    f"  {result['median_us']:.3f} us, "
                    f"{result['kernel_count']} kernels",
                    flush=True,
                )
            else:
                print(f"  ERROR returncode={result['returncode']}", flush=True)

    gpus = args.gpus.split(",") if args.gpus else [str(args.gpu)]
    queues = {gpu: [] for gpu in gpus}
    for item in pending:
        group = f"{item['workload']}:{item['rows']}:{item['hidden']}"
        gpu = gpus[zlib.crc32(group.encode()) % len(gpus)]
        queues[gpu].append(item)
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(run_queue, gpu, queue) for gpu, queue in queues.items()]
            for future in futures:
                future.result()
    finally:
        output.close()


if __name__ == "__main__":
    main()
