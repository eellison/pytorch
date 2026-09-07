from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
MAIN = RESULTS / "d117_aten_padded_quant_main_cdd22ad.json"
D117 = RESULTS / "d117_aten_padded_quant_stack_7fe8160.json"
OUTPUT = RESULTS / "d117_aten_padded_quant_consolidated.json"
REPEAT = RESULTS / "d117_aten_padded_quant_coordesc_repeat.json"
MAIN_FIXED = RESULTS / "d117_aten_padded_quant_main_fixed_default_config.json"
D117_FIXED = RESULTS / "d117_aten_padded_quant_stack_fixed_default_config.json"
PDL_PAIR = RESULTS / "d117_aten_padded_quant_pdl_pair.json"
D117_PDL = RESULTS / "d117_aten_padded_quant_inductor_pdl_on.json"
MAIN_PDL = RESULTS / "d117_aten_padded_quant_main_inductor_pdl_on.json"
D117_MXFP8_PDL = RESULTS / "d117_aten_padded_quant_mxfp8_inductor_pdl_on.json"
MAIN_MXFP8_PDL = RESULTS / "d117_aten_padded_quant_main_mxfp8_inductor_pdl_on.json"
ONE_KERNEL_CEILING = RESULTS / "d117_uninitialized_padding_ceiling_pdl_on.json"


def padding_kernel_proof(row: dict) -> dict:
    candidate = next(
        kernel
        for kernel in row["padding_kernel_candidates"]
        if "zeros" in kernel["name"]
    )
    name = candidate["name"]
    for source_file in row["source_files"]:
        path = Path(source_file["path"])
        if not path.exists():
            continue
        source = path.read_text()
        start = source.find(f"def {name}(")
        if start < 0:
            continue
        end = source.find("'''", start)
        kernel_source = source[start:] if end < 0 else source[start:end]
        return {
            "name": name,
            "source_file": str(path),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "tl_load_count": kernel_source.count("tl.load("),
            "tl_store_count": kernel_source.count("tl.store("),
            "device_assert_count": kernel_source.count("tl.device_assert("),
            "source": kernel_source,
        }
    raise RuntimeError(f"source not found for {name}")


def pdl_kernel_proof(row: dict) -> dict:
    kernels = []
    for selected in row["selected_kernels"]:
        name = selected["name"]
        for source_file in row["source_files"]:
            path = Path(source_file["path"])
            if not path.exists():
                continue
            source = path.read_text()
            function_start = source.find(f"def {name}(")
            if function_start < 0:
                continue
            decorator_start = source.rfind("@triton_heuristics", 0, function_start)
            end = source.find("'''", function_start)
            kernel_source = source[
                decorator_start if decorator_start >= 0 else function_start :
                end if end >= 0 else None
            ]
            kernels.append(
                {
                    "name": name,
                    "source_file": str(path),
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "metadata_launch_pdl": "'launch_pdl': True" in kernel_source,
                    "gdc_wait_count": kernel_source.count(
                        "tl.extra.cuda.gdc_wait()"
                    ),
                    "gdc_launch_count": kernel_source.count(
                        "tl.extra.cuda.gdc_launch_dependents()"
                    ),
                }
            )
            break
    return {
        "all_kernel_metadata_launch_pdl": all(
            kernel["metadata_launch_pdl"] for kernel in kernels
        ),
        "has_gdc_wait_and_launch": any(
            kernel["gdc_wait_count"] > 0 and kernel["gdc_launch_count"] > 0
            for kernel in kernels
        ),
        "kernels": kernels,
    }


def main() -> None:
    main_data = json.loads(MAIN.read_text())
    d117_data = json.loads(D117.read_text())
    main_rows = [
        row
        for row in main_data["results"]
        if row["implementation"] == "inductor_main_cdd22ad"
    ]
    d117_rows = [
        row
        for row in d117_data["results"]
        if row["implementation"] == "d117896257_v0_3_301421b_stack_7fe8160"
    ]
    flashinfer_rows = [
        row
        for row in d117_data["results"]
        if row["implementation"].startswith("flashinfer")
    ]
    repeat_data = json.loads(REPEAT.read_text())
    main_fixed_data = json.loads(MAIN_FIXED.read_text())
    d117_fixed_data = json.loads(D117_FIXED.read_text())
    pdl_data = json.loads(PDL_PAIR.read_text())
    d117_pdl_data = json.loads(D117_PDL.read_text())
    main_pdl_data = json.loads(MAIN_PDL.read_text())
    d117_mxfp8_pdl_data = json.loads(D117_MXFP8_PDL.read_text())
    main_mxfp8_pdl_data = json.loads(MAIN_MXFP8_PDL.read_text())
    ceiling_data = json.loads(ONE_KERNEL_CEILING.read_text())
    summary = []
    for main_row in main_rows:
        shape = main_row["shape"]
        suite = main_row["suite"]
        d117_row = next(
            row
            for row in d117_rows
            if row["shape"] == shape and row["suite"] == suite
        )
        flashinfer_row = next(
            row
            for row in flashinfer_rows
            if row["shape"] == shape and row["suite"] == suite
        )
        main_padding = padding_kernel_proof(main_row)
        d117_padding = padding_kernel_proof(d117_row)
        main_row["padding_kernel_proof"] = main_padding
        d117_row["padding_kernel_proof"] = d117_padding
        summary.append(
            {
                "shape": shape,
                "suite": suite,
                "main_us": main_row["median_us"],
                "d117_us": d117_row["median_us"],
                "flashinfer_pdl_false_us": flashinfer_row["median_us"],
                "d117_speedup_vs_main": main_row["median_us"]
                / d117_row["median_us"],
                "d117_ratio_vs_flashinfer": d117_row["median_us"]
                / flashinfer_row["median_us"],
                "kernel_counts": {
                    "main": main_row["kernel_count"],
                    "d117": d117_row["kernel_count"],
                },
                "padding_kernel_load_counts": {
                    "main": main_padding["tl_load_count"],
                    "d117": d117_padding["tl_load_count"],
                },
            }
        )
    focused_confirmation = []
    for shape, suite in (
        ([19, 4096], "rmsnorm_mxfp8"),
        ([129, 4096], "rmsnorm_mxfp4"),
        ([989, 4096], "rmsnorm_nvfp4"),
    ):
        main_tuned = next(
            row
            for row in main_rows
            if row["shape"] == shape and row["suite"] == suite
        )
        d117_tuned = next(
            row
            for row in d117_rows
            if row["shape"] == shape and row["suite"] == suite
        )
        d117_repeat = next(
            row
            for row in repeat_data["results"]
            if row["shape"] == shape and row["suite"] == suite
            and row["implementation"].startswith("d117")
        )
        main_fixed = next(
            row
            for row in main_fixed_data["results"]
            if row["shape"] == shape and row["suite"] == suite
            and row["implementation"].startswith("inductor")
        )
        d117_fixed = next(
            row
            for row in d117_fixed_data["results"]
            if row["shape"] == shape and row["suite"] == suite
            and row["implementation"].startswith("d117")
        )
        for row in (d117_repeat, main_fixed, d117_fixed):
            row["padding_kernel_proof"] = padding_kernel_proof(row)
        main_reduction = next(
            kernel
            for kernel in main_tuned["selected_kernels"]
            if "red_" in kernel["name"]
        )
        d117_reduction = next(
            kernel
            for kernel in d117_tuned["selected_kernels"]
            if "red_" in kernel["name"]
        )
        repeat_reduction = next(
            kernel
            for kernel in d117_repeat["selected_kernels"]
            if "red_" in kernel["name"]
        )
        main_fixed_reduction = next(
            kernel
            for kernel in main_fixed["selected_kernels"]
            if "red_" in kernel["name"]
        )
        d117_fixed_reduction = next(
            kernel
            for kernel in d117_fixed["selected_kernels"]
            if "red_" in kernel["name"]
        )
        focused_confirmation.append(
            {
                "shape": shape,
                "suite": suite,
                "end_to_end_tuned": {
                    "main_us": main_tuned["median_us"],
                    "d117_us": d117_tuned["median_us"],
                    "d117_repeat_us": d117_repeat["median_us"],
                    "main_reduction_config": main_reduction,
                    "d117_reduction_config": d117_reduction,
                    "d117_repeat_reduction_config": repeat_reduction,
                },
                "fixed_default_reduction_config": {
                    "main_us": main_fixed["median_us"],
                    "d117_us": d117_fixed["median_us"],
                    "d117_speedup_vs_main": main_fixed["median_us"]
                    / d117_fixed["median_us"],
                    "main_reduction_config": main_fixed_reduction,
                    "d117_reduction_config": d117_fixed_reduction,
                    "main_padding_kernel": main_fixed["padding_kernel_proof"],
                    "d117_padding_kernel": d117_fixed["padding_kernel_proof"],
                },
            }
        )
    pdl_comparison = []
    all_d117_pdl_results = [
        *d117_pdl_data["results"],
        *d117_mxfp8_pdl_data["results"],
    ]
    all_main_pdl_results = [
        *main_pdl_data["results"],
        *main_mxfp8_pdl_data["results"],
    ]
    for d117_pdl_row in all_d117_pdl_results:
        if not d117_pdl_row["implementation"].startswith("d117"):
            continue
        shape = d117_pdl_row["shape"]
        suite = d117_pdl_row["suite"]
        d117_off_rows = [
            row
            for row in pdl_data["results"]
            if row["shape"] == shape
            and row["suite"] == suite
            and row["implementation"].startswith("d117")
        ]
        if not d117_off_rows:
            d117_off_rows = [
                row
                for row in d117_rows
                if row["shape"] == shape and row["suite"] == suite
            ]
        d117_off = d117_off_rows[0]
        fi_false = next(
            row
            for row in all_d117_pdl_results
            if row["shape"] == shape
            and row["suite"] == suite
            and row["implementation"].startswith("flashinfer")
            and row["implementation"].endswith("pdl_false")
        )
        fi_true = next(
            row
            for row in all_d117_pdl_results
            if row["shape"] == shape
            and row["suite"] == suite
            and row["implementation"].startswith("flashinfer")
            and row["implementation"].endswith("pdl_true")
        )
        main_pdl_rows = [
            row
            for row in all_main_pdl_results
            if row["shape"] == shape
            and row["suite"] == suite
            and row["implementation"].startswith("inductor")
        ]
        d117_pdl_row["pdl_codegen_proof"] = pdl_kernel_proof(d117_pdl_row)
        if main_pdl_rows:
            main_pdl_rows[0]["pdl_codegen_proof"] = pdl_kernel_proof(
                main_pdl_rows[0]
            )
        pdl_comparison.append(
            {
                "shape": shape,
                "suite": suite,
                "d117_pdl_off_us": d117_off["median_us"],
                "d117_pdl_on_us": d117_pdl_row["median_us"],
                "d117_pdl_speedup": d117_off["median_us"]
                / d117_pdl_row["median_us"],
                "main_pdl_on_us": (
                    main_pdl_rows[0]["median_us"] if main_pdl_rows else None
                ),
                "flashinfer_pdl_false_us": fi_false["median_us"],
                "flashinfer_pdl_true_us": fi_true["median_us"],
                "flashinfer_pdl_speedup": fi_false["median_us"]
                / fi_true["median_us"],
                "d117_pdl_on_ratio_vs_flashinfer_pdl_true": d117_pdl_row[
                    "median_us"
                ]
                / fi_true["median_us"],
                "d117_pdl_codegen_proof": d117_pdl_row["pdl_codegen_proof"],
                "main_pdl_codegen_proof": (
                    main_pdl_rows[0]["pdl_codegen_proof"]
                    if main_pdl_rows
                    else None
                ),
            }
        )
    one_kernel_ceiling = []
    d117_pdl_rows = [
        row
        for row in [*d117_pdl_data["results"], *d117_mxfp8_pdl_data["results"]]
        if row["implementation"].startswith("d117")
    ]
    for ceiling_row in ceiling_data["results"]:
        if not ceiling_row["implementation"].startswith("d117"):
            continue
        shape = ceiling_row["shape"]
        suite = ceiling_row["suite"]
        two_kernel = next(
            row
            for row in d117_pdl_rows
            if row["shape"] == shape and row["suite"] == suite
        )
        flashinfer = next(
            row
            for row in ceiling_data["results"]
            if row["shape"] == shape
            and row["suite"] == suite
            and row["implementation"].startswith("flashinfer")
        )
        ceiling_row["pdl_codegen_proof"] = pdl_kernel_proof(ceiling_row)
        one_kernel_ceiling.append(
            {
                "shape": shape,
                "suite": suite,
                "two_kernel_d117_pdl_on_us": two_kernel["median_us"],
                "one_kernel_ceiling_pdl_on_us": ceiling_row["median_us"],
                "flashinfer_pdl_on_us": flashinfer["median_us"],
                "one_kernel_speedup_vs_two_kernel": two_kernel["median_us"]
                / ceiling_row["median_us"],
                "one_kernel_ratio_vs_flashinfer": ceiling_row["median_us"]
                / flashinfer["median_us"],
                "kernel_count": ceiling_row["kernel_count"],
                "external_op_call_count": ceiling_row["generated_source_checks"][
                    "external_op_call_count"
                ],
                "padding_bytes": ceiling_row["padding_bytes"],
                "pdl_codegen_proof": ceiling_row["pdl_codegen_proof"],
            }
        )
    payload = {
        "authoritative": True,
        "revisions": {
            "main": "cdd22ade2948699188c3e2d0d80b5a396a8489ae",
            "d117_worktree_head": "7fe816090ad8a6ac9707cfb468105e8401a97e01",
            "d117_original_v0_3": "301421b0c8f930f35d3a824dd173df88039a1b93",
            "d117_prerequisite_rebased": [
                "64143220551f711e32916bc06766b493234440df",
                "0ab10c182033a552b62f647ad08766f5880f0ee1",
            ],
        },
        "protocol": d117_data["protocol"],
        "focused_tests": {
            "test_padded_scatter_selected": "5 passed",
            "test_masked_scatter_masks_output_index": "1 passed",
        },
        "padding_kernel_proof": {
            "all_main_padding_kernels_load_output": all(
                row["padding_kernel_proof"]["tl_load_count"] == 1
                for row in main_rows
            ),
            "all_d117_padding_kernels_have_no_loads": all(
                row["padding_kernel_proof"]["tl_load_count"] == 0
                for row in d117_rows
            ),
        },
        "inductor_pdl_proof": {
            "all_d117_kernels_metadata_launch_pdl": all(
                row["d117_pdl_codegen_proof"]["all_kernel_metadata_launch_pdl"]
                for row in pdl_comparison
            ),
            "all_d117_graphs_emit_gdc_wait_and_launch": all(
                row["d117_pdl_codegen_proof"]["has_gdc_wait_and_launch"]
                for row in pdl_comparison
            ),
            "all_measured_main_graphs_metadata_launch_pdl": all(
                row["main_pdl_codegen_proof"] is None
                or row["main_pdl_codegen_proof"]["all_kernel_metadata_launch_pdl"]
                for row in pdl_comparison
            ),
            "all_measured_main_graphs_emit_gdc_wait_and_launch": all(
                row["main_pdl_codegen_proof"] is None
                or row["main_pdl_codegen_proof"]["has_gdc_wait_and_launch"]
                for row in pdl_comparison
            ),
        },
        "one_kernel_ceiling_proof": {
            "all_kernel_counts_one": all(
                row["kernel_count"] == 1 for row in one_kernel_ceiling
            ),
            "all_external_op_call_counts_zero": all(
                row["external_op_call_count"] == 0 for row in one_kernel_ceiling
            ),
            "all_kernel_metadata_launch_pdl": all(
                row["pdl_codegen_proof"]["all_kernel_metadata_launch_pdl"]
                for row in one_kernel_ceiling
            ),
            "all_graphs_emit_gdc_wait_and_launch": all(
                row["pdl_codegen_proof"]["has_gdc_wait_and_launch"]
                for row in one_kernel_ceiling
            ),
            "padding_bytes": "unspecified",
        },
        "summary": summary,
        "focused_confirmation": focused_confirmation,
        "pdl_comparison": pdl_comparison,
        "one_kernel_ceiling": one_kernel_ceiling,
        "supporting_results": {
            "d117_coordesc_repeat": repeat_data["results"],
            "main_fixed_default_config": main_fixed_data["results"],
            "d117_fixed_default_config": d117_fixed_data["results"],
            "d117_flashinfer_pdl_pair": pdl_data["results"],
            "d117_inductor_pdl_on": d117_pdl_data["results"],
            "main_inductor_pdl_on": main_pdl_data["results"],
            "main_mxfp8_inductor_pdl_on": main_mxfp8_pdl_data["results"],
            "d117_mxfp8_inductor_pdl_on": d117_mxfp8_pdl_data["results"],
            "d117_one_kernel_ceiling": ceiling_data["results"],
        },
        "results": [*main_rows, *d117_rows, *flashinfer_rows],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
