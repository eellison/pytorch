from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("status") == "ok":
                records.append(record)
    return records


def shape(record: dict[str, Any]) -> str:
    return "x".join(str(value) for value in record["shape"])


def ratio(numerator: float, denominator: float) -> str:
    return f"{numerator / denominator:.2f}x"


def timing(record: dict[str, Any] | None) -> str:
    if record is None:
        return "-"
    return f"{record['median_us']:.3f} ({record['kernel_count']}k)"


def table(headers: list[str], rows: list[list[str]]) -> str:
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    result.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(result)


def output_key(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(item["sha256"] for item in record["outputs"])


def correctness(records: list[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    errors = []
    groups = defaultdict(list)
    for record in records:
        if record["variant"] != "flashinfer":
            key = (
                record["workload"],
                tuple(record["shape"]),
                record.get("matrix_tuning", record["tuning"]),
            )
            groups[key].append(record)
        if record["variant"] != "flashinfer" and record["padding"]["padding_wrong"]:
            errors.append(
                f"{record['workload']} {shape(record)} {record['variant']}: "
                f"{record['padding']['padding_wrong']} padding values wrong"
            )
    rows = []
    for (workload, dimensions, tuning), group in sorted(groups.items()):
        hashes = {output_key(record) for record in group}
        rows.append(
            [
                workload,
                "x".join(map(str, dimensions)),
                tuning,
                str(len(group)),
                "yes" if len(hashes) == 1 else "NO",
                str(group[0]["padding"]["padding_values"]),
            ]
        )
        if len(hashes) != 1:
                errors.append(
                    f"{workload} {dimensions} {tuning}: padding variants differ"
                )
    return errors, rows


def layout_table(records: list[dict[str, Any]]) -> list[list[str]]:
    groups = defaultdict(dict)
    for record in records:
        if record["workload"].startswith("swizzle"):
            groups[(record["workload"], tuple(record["shape"]))][record["variant"]] = record
    rows = []
    for (workload, dimensions), variants in sorted(groups.items()):
        fpad = variants.get("fpad")
        fill = variants.get("fill_scatter")
        sparse = variants.get("pad_scatter")
        predicated = variants.get("predicated")
        auxiliary = variants.get("auxiliary")
        rows.append(
            [
                workload,
                "x".join(map(str, dimensions)),
                f"{fpad['padding']['padding_fraction']:.1%}" if fpad else "-",
                timing(fpad),
                timing(fill),
                ratio(fill["median_us"], fpad["median_us"]) if fill and fpad else "-",
                timing(sparse),
                ratio(sparse["median_us"], fpad["median_us"]) if sparse and fpad else "-",
                timing(predicated or auxiliary),
                ratio((predicated or auxiliary)["median_us"], fpad["median_us"])
                if (predicated or auxiliary) and fpad
                else "-",
            ]
        )
    return rows


def pipeline_table(records: list[dict[str, Any]]) -> list[list[str]]:
    groups = defaultdict(dict)
    flashinfer = {}
    for record in records:
        if record["workload"].startswith("swizzle"):
            continue
        key = (record["workload"], tuple(record["shape"]))
        if record["variant"] == "flashinfer":
            flashinfer[key] = record
            continue
        tuning = record.get("matrix_tuning", record["tuning"])
        groups[(*key, tuning)][record["variant"]] = record
    rows = []
    for (workload, dimensions, tuning), variants in sorted(groups.items()):
        fpad = variants.get("fpad")
        fill = variants.get("fill_scatter")
        sparse = variants.get("pad_scatter")
        predicated = variants.get("predicated")
        auxiliary = variants.get("auxiliary")
        candidates = [
            item
            for item in (fpad, fill, sparse, predicated, auxiliary)
            if item is not None
        ]
        best = min(candidates, key=lambda item: item["median_us"])
        external = flashinfer.get((workload, dimensions))
        rows.append(
            [
                workload,
                "x".join(map(str, dimensions)),
                tuning,
                timing(fpad),
                timing(fill),
                timing(sparse),
                timing(predicated or auxiliary),
                f"{best['variant']} {best['median_us']:.3f}",
                timing(external),
                ratio(best["median_us"], external["median_us"]) if external else "-",
            ]
        )
    return rows


def ranges(records: list[dict[str, Any]]) -> list[str]:
    messages = []
    for workload in sorted({record["workload"] for record in records}):
        if workload.startswith("swizzle"):
            continue
        ratios = []
        groups = defaultdict(dict)
        for record in records:
            if record["workload"] != workload or record["variant"] == "flashinfer":
                continue
            key = (tuple(record["shape"]), record.get("matrix_tuning", record["tuning"]))
            groups[key][record["variant"]] = record
        for variants in groups.values():
            if "fpad" in variants and "fill_scatter" in variants:
                ratios.append(
                    variants["fill_scatter"]["median_us"]
                    / variants["fpad"]["median_us"]
                )
        if ratios:
            messages.append(
                f"- {workload}: fill+scatter / F.pad {min(ratios):.2f}-{max(ratios):.2f}x "
                f"(median {statistics.median(ratios):.2f}x)."
            )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = load(args.input)
    errors, correctness_rows = correctness(records)
    revisions = sorted({record["worktree_git"] for record in records})
    gpus = sorted({record["gpu"] for record in records})
    failures = sum(
        1
        for path in args.input
        for line in path.read_text().splitlines()
        if '"status": "error"' in line
    )
    report = [
        "# Padded swizzle and quantization matrix",
        "",
        "Sources: " + ", ".join(f"`{path}`" for path in args.input),
        f"Revisions: {', '.join(f'`{revision}`' for revision in revisions)}",
        f"GPUs: {', '.join(gpus)}",
        f"Successful cells: {len(records)}; failed cells: {failures}.",
        "",
        "Every cell ran in a fresh process. Timings are CUDA-graph replay medians; "
        "each replay contains the number of calls recorded in the JSON result.",
        "",
        "## Correctness",
        "",
        "All padding lanes contain the required sentinel and all Inductor padding "
        "variants for a workload/shape produce byte-identical payload and scale "
        "outputs." if not errors else "Errors: " + "; ".join(errors),
        "",
        table(
            [
                "workload",
                "shape",
                "tuning",
                "cells",
                "byte-identical",
                "padding values",
            ],
            correctness_rows,
        ),
        "",
        "## Padding-only swizzle",
        "",
        "Times are `us (generated kernels)`. Ratios above 1 mean the scatter form is slower.",
        "",
        table(
            [
                "workload",
                "shape",
                "padding",
                "F.pad",
                "fill+scatter",
                "fill/F.pad",
                "sparse pad scatter",
                "sparse/F.pad",
                "experimental",
                "experimental/F.pad",
            ],
            layout_table(records),
        ),
        "",
        "## Full pipelines",
        "",
        table(
            [
                "workload",
                "shape",
                "tuning",
                "F.pad",
                "fill+scatter",
                "sparse pad scatter",
                "experimental",
                "best Inductor",
                "FlashInfer",
                "best/FI",
            ],
            pipeline_table(records),
        ),
        "",
        "## Padding mechanism ranges",
        "",
        *ranges(records),
        "",
    ]
    text = "\n".join(report)
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
