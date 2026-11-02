#!/usr/bin/env python3
"""
ORBIT-FMIB Code 12 — Fixed-subset trajectory organization and QA.

Consumes the authoritative corrected Code-11 fixed-subset biological score map
and organizes it into 15 subset trajectories across 31 ESM-2 representation
stages.

No neural training. No pooled-by-order MI estimation. No retention calculation.
No null subtraction. No clamping of negative near-floor neural-DV scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CODE_NAME = "ORBIT-FMIB CODE 12 — FIXED-SUBSET TRAJECTORY ORGANIZATION"

LAYERS = tuple(range(31))
SUBSETS_BY_ORDER = {
    1: (1, 2, 4, 8),
    2: (3, 5, 6, 9, 10, 12),
    3: (7, 11, 13, 14),
    4: (15,),
}
SUBSET_ORDER = tuple(
    (k, mask)
    for k in (1, 2, 3, 4)
    for mask in SUBSETS_BY_ORDER[k]
)

EXPECTED_CODE11_SOURCE_SHA256 = (
    "8ff6654c9352e56f2eff7368ac1ed14b90929f3729cb3bacc5b5dd592b1209e3"
)

EXPECTED_SAMPLE_REGIME = {
    "n_total_per_subset": 8210,
    "n_train": 7352,
    "n_val": 245,
    "n_test": 613,
}

SOURCE_COLUMNS = (
    "layer",
    "order_k",
    "subset_mask",
    "score_mean",
    "critic_seed_sd",
    "score_min",
    "score_max",
    "mean_best_val",
    "mean_best_step",
    "mean_final_training_step",
    "mean_product_shuffle_sd",
    "mean_joint_minus_mismatch_score",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalized_subsets_by_order(raw: dict) -> Dict[int, Tuple[int, ...]]:
    return {int(k): tuple(int(x) for x in v) for k, v in raw.items()}


def validate_manifest(manifest: dict, manifest_path: Path) -> Path:
    if manifest.get("overall_pass") is not True:
        raise RuntimeError("Code-11 manifest does not report overall_pass=True.")

    if manifest.get("production_source_sha256") != EXPECTED_CODE11_SOURCE_SHA256:
        raise RuntimeError("Code-11 production source hash mismatch.")

    outputs = manifest.get("outputs", {})
    subset_name = outputs.get("subset_summary")
    if subset_name != "gb1_subset_mi_summary.csv":
        raise RuntimeError(f"Unexpected Code-11 subset summary: {subset_name!r}")

    cond = manifest.get("conditioning_contract", {})
    if cond.get("one_subset_mask_per_critic") is not True:
        raise RuntimeError("Code-11 manifest is not fixed-subset conditioned.")

    got_subsets = normalized_subsets_by_order(cond.get("subsets_by_order", {}))
    if got_subsets != SUBSETS_BY_ORDER:
        raise RuntimeError(f"Code-11 subset structure mismatch: {got_subsets!r}")

    for key, expected in EXPECTED_SAMPLE_REGIME.items():
        if int(cond.get(key, -1)) != expected:
            raise RuntimeError(
                f"Code-11 conditioning contract mismatch for {key}: "
                f"{cond.get(key)!r} != {expected}"
            )

    downstream = " ".join(str(x).lower() for x in manifest.get("downstream_contract", []))
    required_phrases = (
        "subset-level summary is the authoritative biological score map",
        "within each fixed subset trajectory first",
        "only then aggregated within order",
        "not subtracted",
        "negative",
        "retained",
    )
    for phrase in required_phrases:
        if phrase not in downstream:
            raise RuntimeError(
                f"Code-11 downstream contract missing phrase: {phrase!r}"
            )

    subset_path = manifest_path.parent / subset_name
    if not subset_path.exists():
        raise FileNotFoundError(subset_path)

    recorded_hashes = manifest.get("file_sha256", {})
    expected_hash = recorded_hashes.get(subset_name)
    if not expected_hash:
        raise RuntimeError(f"Manifest lacks SHA-256 for {subset_name}.")
    if sha256_file(subset_path) != expected_hash:
        raise RuntimeError("Code-11 subset-summary hash mismatch.")

    audit = manifest.get("computational_audit", {})
    observed = audit.get("observed", {})
    if audit.get("overall_pass") is not True:
        raise RuntimeError("Code-11 computational audit did not pass.")
    if int(observed.get("subset_cells", -1)) != 465:
        raise RuntimeError("Code-11 manifest does not report 465 subset cells.")
    if int(observed.get("primary_fits", -1)) != 1395:
        raise RuntimeError("Code-11 manifest does not report 1395 primary fits.")
    if observed.get("all_results_finite") is not True:
        raise RuntimeError("Code-11 manifest does not report finite results.")

    return subset_path


def validate_rows(rows: List[dict]) -> None:
    if len(rows) != 465:
        raise RuntimeError(f"Expected 465 Code-11 subset cells; found {len(rows)}.")

    if not rows:
        raise RuntimeError("Code-11 subset summary is empty.")

    missing_columns = set(SOURCE_COLUMNS) - set(rows[0].keys())
    if missing_columns:
        raise RuntimeError(
            f"Code-11 subset summary missing columns: {sorted(missing_columns)}"
        )

    seen = set()
    by_layer: Dict[int, set] = {layer: set() for layer in LAYERS}
    numeric_fields = (
        "score_mean",
        "critic_seed_sd",
        "score_min",
        "score_max",
        "mean_best_val",
        "mean_best_step",
        "mean_final_training_step",
        "mean_product_shuffle_sd",
        "mean_joint_minus_mismatch_score",
    )

    for i, r in enumerate(rows, start=2):
        layer = int(r["layer"])
        k = int(r["order_k"])
        mask = int(r["subset_mask"])

        if layer not in LAYERS:
            raise RuntimeError(f"Row {i}: unexpected layer {layer}.")
        if k not in SUBSETS_BY_ORDER:
            raise RuntimeError(f"Row {i}: unexpected order k={k}.")
        if mask not in SUBSETS_BY_ORDER[k]:
            raise RuntimeError(f"Row {i}: invalid subset mask {mask} for k={k}.")
        if bin(mask).count("1") != k:
            raise RuntimeError(f"Row {i}: subset mask popcount != k.")

        key = (layer, k, mask)
        if key in seen:
            raise RuntimeError(f"Duplicate layer/order/subset cell: {key}.")
        seen.add(key)
        by_layer[layer].add((k, mask))

        for field in numeric_fields:
            value = float(r[field])
            if not math.isfinite(value):
                raise RuntimeError(f"Row {i}: non-finite {field}.")

        if float(r["critic_seed_sd"]) < 0:
            raise RuntimeError(f"Row {i}: negative critic_seed_sd.")

        score = float(r["score_mean"])
        lo = float(r["score_min"])
        hi = float(r["score_max"])
        if not (lo <= score <= hi):
            raise RuntimeError(f"Row {i}: score_mean outside [score_min, score_max].")

    expected_set = set(SUBSET_ORDER)
    for layer in LAYERS:
        if by_layer[layer] != expected_set:
            raise RuntimeError(f"Layer {layer} subset grid mismatch.")

    if len(seen) != 465:
        raise RuntimeError(f"Expected 465 unique cells; found {len(seen)}.")


def sorted_rows(rows: Iterable[dict]) -> List[dict]:
    order_index = {pair: i for i, pair in enumerate(SUBSET_ORDER)}
    return sorted(
        rows,
        key=lambda r: (
            int(r["layer"]),
            order_index[(int(r["order_k"]), int(r["subset_mask"]))],
        ),
    )


def write_long(rows: List[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SOURCE_COLUMNS))
        w.writeheader()
        for r in sorted_rows(rows):
            w.writerow({col: r[col] for col in SOURCE_COLUMNS})


def matrix_column(k: int, mask: int) -> str:
    return f"k{k}_S{mask}"


def write_matrix(rows: List[dict], path: Path, value_field: str) -> None:
    lookup = {
        (int(r["layer"]), int(r["order_k"]), int(r["subset_mask"])): r[value_field]
        for r in rows
    }
    columns = ["layer"] + [matrix_column(k, mask) for k, mask in SUBSET_ORDER]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for layer in LAYERS:
            row = [layer]
            for k, mask in SUBSET_ORDER:
                row.append(lookup[(layer, k, mask)])
            w.writerow(row)


def write_subset_index(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trajectory_column", "order_k", "subset_mask", "n_layers"])
        for k, mask in SUBSET_ORDER:
            w.writerow([matrix_column(k, mask), k, mask, len(LAYERS)])


def run_self_test() -> None:
    rows = []
    for layer in LAYERS:
        for k, mask in SUBSET_ORDER:
            score = (layer * 0.001) + (mask * 0.0001)
            rows.append(
                {
                    "layer": str(layer),
                    "order_k": str(k),
                    "subset_mask": str(mask),
                    "score_mean": repr(score),
                    "critic_seed_sd": "0.001",
                    "score_min": repr(score - 0.001),
                    "score_max": repr(score + 0.001),
                    "mean_best_val": "0.0",
                    "mean_best_step": "10.0",
                    "mean_final_training_step": "20.0",
                    "mean_product_shuffle_sd": "0.001",
                    "mean_joint_minus_mismatch_score": "0.0",
                }
            )

    validate_rows(rows)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        long_path = td / "long.csv"
        matrix_path = td / "matrix.csv"
        sd_path = td / "sd.csv"
        index_path = td / "index.csv"

        write_long(rows, long_path)
        write_matrix(rows, matrix_path, "score_mean")
        write_matrix(rows, sd_path, "critic_seed_sd")
        write_subset_index(index_path)

        assert len(read_csv(long_path)) == 465
        assert len(read_csv(matrix_path)) == 31
        assert len(read_csv(sd_path)) == 31
        assert len(read_csv(index_path)) == 15

    print(f"{CODE_NAME} SELF-TEST: PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--code11-manifest", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    root = project_root()
    exp = root / "04 Experiments" / "EXP02_GB1_ESM2"

    manifest_path = (
        args.code11_manifest
        if args.code11_manifest is not None
        else exp / "gb1_subset_mi_manifest.json"
    ).resolve()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else exp.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(manifest_path)
    subset_path = validate_manifest(manifest, manifest_path)

    rows = read_csv(subset_path)
    validate_rows(rows)

    long_path = output_dir / "gb1_subset_trajectories_long.csv"
    matrix_path = output_dir / "gb1_subset_trajectory_matrix.csv"
    sd_matrix_path = output_dir / "gb1_subset_trajectory_seed_sd_matrix.csv"
    index_path = output_dir / "gb1_subset_trajectory_index.csv"
    out_manifest_path = output_dir / "gb1_subset_trajectory_manifest.json"

    write_long(rows, long_path)
    write_matrix(rows, matrix_path, "score_mean")
    write_matrix(rows, sd_matrix_path, "critic_seed_sd")
    write_subset_index(index_path)

    long_rows = read_csv(long_path)
    matrix_rows = read_csv(matrix_path)
    sd_rows = read_csv(sd_matrix_path)
    index_rows = read_csv(index_path)

    if len(long_rows) != 465:
        raise RuntimeError("Generated trajectory-long file does not have 465 rows.")
    if len(matrix_rows) != 31:
        raise RuntimeError("Generated trajectory matrix does not have 31 rows.")
    if len(sd_rows) != 31:
        raise RuntimeError("Generated seed-SD matrix does not have 31 rows.")
    if len(index_rows) != 15:
        raise RuntimeError("Generated trajectory index does not have 15 rows.")

    output_paths = (long_path, matrix_path, sd_matrix_path, index_path)

    out_manifest = {
        "schema_version": 2,
        "code_name": CODE_NAME,
        "purpose": (
            "Organize the authoritative corrected Code-11 fixed-subset "
            "biological score map into 15 trajectories across 31 ESM-2 "
            "representation stages, without computing retention."
        ),
        "code12_source_sha256": sha256_file(Path(__file__).resolve()),
        "upstream": {
            "code11_manifest": manifest_path.name,
            "code11_manifest_sha256": sha256_file(manifest_path),
            "code11_production_source_sha256": manifest.get("production_source_sha256"),
            "subset_summary": subset_path.name,
            "subset_summary_sha256": sha256_file(subset_path),
        },
        "trajectory_contract": {
            "authoritative_unit": "fixed positional subset S",
            "n_layers": 31,
            "layers": list(LAYERS),
            "n_subsets": 15,
            "subsets_by_order": {str(k): list(v) for k, v in SUBSETS_BY_ORDER.items()},
            "n_layer_subset_cells": 465,
            "matrix_rows": "ESM-2 representation stages L0..L30",
            "matrix_columns": [matrix_column(k, mask) for k, mask in SUBSET_ORDER],
            "score_field": "Code-11 score_mean across critic seeds",
            "negative_scores_clamped": False,
            "null_scores_subtracted": False,
            "pooled_by_order_mi_estimation": False,
            "retention_computed": False,
        },
        "audit": {
            "source_subset_rows": len(rows),
            "unique_layer_subset_cells": 465,
            "trajectory_count": 15,
            "layers_per_trajectory": 31,
            "all_source_scores_finite": True,
            "source_grid_complete": True,
            "generated_long_rows": len(long_rows),
            "generated_matrix_rows": len(matrix_rows),
            "generated_sd_matrix_rows": len(sd_rows),
            "generated_index_rows": len(index_rows),
            "overall_pass": True,
        },
        "outputs": {
            "trajectory_long": long_path.name,
            "trajectory_matrix": matrix_path.name,
            "trajectory_seed_sd_matrix": sd_matrix_path.name,
            "trajectory_index": index_path.name,
        },
        "file_sha256": {p.name: sha256_file(p) for p in output_paths},
        "interpretation_boundary": [
            "Code 12 is descriptive trajectory organization and QA only.",
            "The 15 fixed-subset trajectories are preserved independently.",
            "No critic pools different positional subsets.",
            "No retention statistic is computed in Code 12.",
            "No order-level confirmatory statistic is computed in Code 12.",
            "Negative near-floor neural-DV scores are retained.",
            "Randomized-null scores are not subtracted from biological scores.",
            "Any future retention rule must be defined within each fixed subset trajectory first and only then aggregated within interaction order.",
        ],
        "overall_pass": True,
    }

    with out_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(out_manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\n{CODE_NAME}: PASS")
    print("Source fixed-subset cells: 465")
    print("Trajectories: 15")
    print("Layers per trajectory: 31")
    print("Retention computed: NO")
    print("Pooled-by-order MI estimation: NO")
    print(f"Manifest: {out_manifest_path}")


if __name__ == "__main__":
    main()
