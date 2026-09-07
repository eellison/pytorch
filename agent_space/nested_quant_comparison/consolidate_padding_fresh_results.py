from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
INPUTS = {
    "default": (
        RESULTS / "padding_fresh_idle_default.json",
        RESULTS / "padding_fresh_idle_remaining_default.json",
    ),
    "coordesc": (
        RESULTS / "padding_fresh_idle_coordesc.json",
        RESULTS / "padding_fresh_idle_remaining_coordesc.json",
    ),
    "coordesc_force_cat": (
        RESULTS / "padding_fresh_idle_coordesc_force_cat.json",
        RESULTS / "padding_fresh_idle_remaining_coordesc_force_cat.json",
    ),
}
OUTPUT = RESULTS / "padding_fresh_idle_consolidated.json"


def load_mode(paths: tuple[Path, Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(json.loads(path.read_text())["results"])
    return rows


def main() -> None:
    by_mode = {mode: load_mode(paths) for mode, paths in INPUTS.items()}
    inductor = {
        mode: [row for row in rows if row["implementation"].startswith("inductor")]
        for mode, rows in by_mode.items()
    }
    flashinfer = [
        row
        for row in by_mode["coordesc"]
        if row["implementation"].startswith("flashinfer")
    ]
    order = {
        (1, 4096): 0,
        (19, 4096): 1,
        (99, 4096): 2,
        (129, 4096): 3,
        (989, 4096): 4,
        (129, 4128): 5,
    }
    summaries = []
    for shape in order:
        for suite in ("rmsnorm_nvfp4", "rmsnorm_mxfp4", "rmsnorm_mxfp8"):
            mode_rows = {
                mode: next(
                    row
                    for row in rows
                    if tuple(row["shape"]) == shape and row["suite"] == suite
                )
                for mode, rows in inductor.items()
            }
            fi = next(
                row
                for row in flashinfer
                if tuple(row["shape"]) == shape and row["suite"] == suite
            )
            default_us = mode_rows["default"]["median_us"]
            coord_us = mode_rows["coordesc"]["median_us"]
            force_us = mode_rows["coordesc_force_cat"]["median_us"]
            fi_us = fi["median_us"]
            summaries.append(
                {
                    "shape": list(shape),
                    "suite": suite,
                    "default_us": default_us,
                    "coordesc_us": coord_us,
                    "coordesc_force_cat_us": force_us,
                    "flashinfer_pdl_false_us": fi_us,
                    "coordesc_speedup_vs_default": default_us / coord_us,
                    "force_cat_speedup_vs_coordesc": coord_us / force_us,
                    "force_cat_ratio_vs_flashinfer": force_us / fi_us,
                    "kernel_counts": {
                        mode: row["kernel_count"] for mode, row in mode_rows.items()
                    },
                    "pad_rewritten_as_cat": {
                        mode: row["pad_rewritten_as_cat"]
                        for mode, row in mode_rows.items()
                    },
                }
            )
    first = json.loads(INPUTS["default"][0].read_text())
    output = {
        "authoritative": True,
        "environment": first["environment"],
        "protocol": {
            "execution": "three sequential fresh processes on idle GPU7",
            "warmup": 20,
            "samples": 50,
            "calls_per_graph": 100,
            "inductor_internal_cudagraphs": False,
            "external_cuda_graph": True,
            "flashinfer_pdl": False,
            "correctness_region": "logical scale values after unswizzle; padded bytes excluded",
        },
        "excluded_non_authoritative_artifacts": [
            "padded_quant_main.json",
            "padding_force_pointwise_cat_ablation.json",
            "padding_fresh_default.json",
            "padding_fresh_coordesc.json",
        ],
        "source_files": {
            mode: [str(path) for path in paths] for mode, paths in INPUTS.items()
        },
        "summary": summaries,
        "results": [
            *inductor["default"],
            *inductor["coordesc"],
            *inductor["coordesc_force_cat"],
            *flashinfer,
        ],
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
