#!/usr/bin/env python3
"""
ORBIT-FMIB CODE 13 — FIXED-SUBSET FLOOR-SAFE RETENTION

Scientific contract
-------------------
Primary input:
    04 Experiments/EXP02_GB1_ESM2/gb1_subset_trajectories_long.csv

Estimator-floor input:
    04 Experiments/EXP02_GB1_ESM2/
        gb1_subset_conditioned_diag_null_by_derangement.csv

Frozen retention rule (pre-specified before HO-vs-LO computation)
-----------------------------------------------------------------
tau = Q_0.99(|D_null|), computed from the authoritative fixed-S
cube-derangement null-score means.

For each fixed positional subset S:
    P_S = max_l D_{l,S}

If P_S > tau:
    Retain_S = D_{30,S} / P_S
else:
    Retain_S = NA (not estimable)

Separately:
    final_near_floor = |D_{30,S}| <= tau

Constraints:
- Compute retention within fixed S before any order aggregation.
- Do not clamp negative neural-DV scores.
- Do not subtract null scores from biological scores.
- Do not use a floor-level peak as a denominator.
- If any subset required for an order is non-estimable, that order's
  confirmatory retention is NA rather than averaging only eligible subsets.
- If any required order is NA, the corresponding LO/HO/Delta contrast is NA.
- Critic seeds are estimator replicates, not biological replicates.

This script computes point-estimate retention organization only.
It does NOT perform the final cube-/subset-aware inferential resampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CODE_NAME = "ORBIT-FMIB CODE 13 — FIXED-SUBSET FLOOR-SAFE RETENTION"
SCHEMA_VERSION = 1

PROJECT_DEFAULT = Path(".")
TRAJ_REL = Path(
    "04 Experiments/EXP02_GB1_ESM2/IID_Production_20260910/Code12/gb1_subset_trajectories_long.csv"
)
NULL_REL = Path(
    "04 Experiments/EXP02_GB1_ESM2/"
    "gb1_subset_conditioned_diag_null_by_derangement.csv"
)
OUT_DIR_REL = Path("04 Experiments/EXP02_GB1_ESM2/IID_Production_20260910/Code13")

SUBSET_OUT = "gb1_subset_retention.csv"
ORDER_OUT = "gb1_order_retention_summary.csv"
CONTRAST_OUT = "gb1_retention_contrast.csv"
MANIFEST_OUT = "gb1_retention_manifest.json"

EXPECTED_LAYERS = list(range(31))
EXPECTED_SUBSETS_BY_ORDER = {
    1: [1, 2, 4, 8],
    2: [3, 5, 6, 9, 10, 12],
    3: [7, 11, 13, 14],
    4: [15],
}
EXPECTED_SUBSET_TO_ORDER = {
    mask: order
    for order, masks in EXPECTED_SUBSETS_BY_ORDER.items()
    for mask in masks
}
EXPECTED_TRAJECTORY_ROWS = 465
EXPECTED_NULL_ROWS = 225
FINAL_LAYER = 30
NULL_QUANTILE = 0.99


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def linear_quantile(values: Sequence[float], q: float) -> float:
    """NumPy/pandas-style linear interpolation quantile."""
    if not values:
        raise ContractError("Cannot compute quantile of an empty sequence.")
    if not (0.0 <= q <= 1.0):
        raise ContractError(f"Invalid quantile q={q}.")
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return v[lo]
    return v[lo] + (pos - lo) * (v[hi] - v[lo])


def parse_float(value: str, field: str, row_no: int) -> float:
    try:
        x = float(value)
    except Exception as exc:
        raise ContractError(
            f"Row {row_no}: field {field!r} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(x):
        raise ContractError(
            f"Row {row_no}: field {field!r} is non-finite: {value!r}"
        )
    return x


def parse_int(value: str, field: str, row_no: int) -> int:
    x = parse_float(value, field, row_no)
    if int(x) != x:
        raise ContractError(
            f"Row {row_no}: field {field!r} must be integer-valued, got {value!r}"
        )
    return int(x)


def read_csv_dicts(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        raise ContractError(f"Missing required input: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ContractError(f"No header found in {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def require_columns(
    columns: Iterable[str], required: Iterable[str], label: str
) -> None:
    cset = set(columns)
    missing = [c for c in required if c not in cset]
    if missing:
        raise ContractError(f"{label} is missing required columns: {missing}")


def load_null_scores(path: Path) -> Tuple[List[float], Dict[str, float]]:
    columns, rows = read_csv_dicts(path)
    require_columns(columns, ["null_score_mean"], "Fixed-S null file")
    if len(rows) != EXPECTED_NULL_ROWS:
        raise ContractError(
            f"Expected {EXPECTED_NULL_ROWS} fixed-S null rows, found {len(rows)}."
        )
    scores = [
        parse_float(r["null_score_mean"], "null_score_mean", i + 2)
        for i, r in enumerate(rows)
    ]
    abs_scores = [abs(x) for x in scores]
    stats = {
        "n": len(scores),
        "min": min(scores),
        "median": linear_quantile(scores, 0.50),
        "q95_abs": linear_quantile(abs_scores, 0.95),
        "q99_abs": linear_quantile(abs_scores, 0.99),
        "max_abs": max(abs_scores),
    }
    return scores, stats


def load_trajectories(
    path: Path,
) -> Tuple[
    Dict[int, List[Dict[str, float]]],
    List[str],
    List[Dict[str, str]],
]:
    columns, raw_rows = read_csv_dicts(path)
    require_columns(
        columns,
        [
            "layer",
            "order_k",
            "subset_mask",
            "score_mean",
            "critic_seed_sd",
            "score_min",
            "score_max",
        ],
        "Code 12 trajectory file",
    )

    if len(raw_rows) != EXPECTED_TRAJECTORY_ROWS:
        raise ContractError(
            f"Expected {EXPECTED_TRAJECTORY_ROWS} trajectory rows, "
            f"found {len(raw_rows)}."
        )

    by_subset: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    seen = set()

    for i, r in enumerate(raw_rows, start=2):
        layer = parse_int(r["layer"], "layer", i)
        order = parse_int(r["order_k"], "order_k", i)
        mask = parse_int(r["subset_mask"], "subset_mask", i)
        score = parse_float(r["score_mean"], "score_mean", i)
        seed_sd = parse_float(r["critic_seed_sd"], "critic_seed_sd", i)
        score_min = parse_float(r["score_min"], "score_min", i)
        score_max = parse_float(r["score_max"], "score_max", i)

        if mask not in EXPECTED_SUBSET_TO_ORDER:
            raise ContractError(f"Row {i}: unexpected subset_mask={mask}.")
        if EXPECTED_SUBSET_TO_ORDER[mask] != order:
            raise ContractError(
                f"Row {i}: subset_mask={mask} should have order "
                f"{EXPECTED_SUBSET_TO_ORDER[mask]}, got {order}."
            )
        if layer not in EXPECTED_LAYERS:
            raise ContractError(f"Row {i}: unexpected layer={layer}.")
        key = (layer, mask)
        if key in seen:
            raise ContractError(f"Duplicate layer/subset cell: {key}")
        seen.add(key)

        by_subset[mask].append(
            {
                "layer": layer,
                "order_k": order,
                "subset_mask": mask,
                "score_mean": score,
                "critic_seed_sd": seed_sd,
                "score_min": score_min,
                "score_max": score_max,
            }
        )

    if set(by_subset) != set(EXPECTED_SUBSET_TO_ORDER):
        missing = sorted(set(EXPECTED_SUBSET_TO_ORDER) - set(by_subset))
        extra = sorted(set(by_subset) - set(EXPECTED_SUBSET_TO_ORDER))
        raise ContractError(
            f"Subset grid mismatch. Missing={missing}, extra={extra}"
        )

    for mask, rows in by_subset.items():
        rows.sort(key=lambda r: int(r["layer"]))
        layers = [int(r["layer"]) for r in rows]
        if layers != EXPECTED_LAYERS:
            raise ContractError(
                f"subset_mask={mask} does not contain exactly layers 0..30."
            )

    if len(seen) != EXPECTED_TRAJECTORY_ROWS:
        raise ContractError(
            f"Expected {EXPECTED_TRAJECTORY_ROWS} unique layer/subset cells, "
            f"found {len(seen)}."
        )

    return dict(by_subset), columns, raw_rows


def compute_subset_retention(
    by_subset: Dict[int, List[Dict[str, float]]], tau: float
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []

    for mask in sorted(by_subset):
        rows = by_subset[mask]
        order = EXPECTED_SUBSET_TO_ORDER[mask]
        peak_row = max(rows, key=lambda r: (r["score_mean"], -r["layer"]))
        peak_score = float(peak_row["score_mean"])
        peak_layer = int(peak_row["layer"])

        final_rows = [r for r in rows if int(r["layer"]) == FINAL_LAYER]
        if len(final_rows) != 1:
            raise ContractError(
                f"subset_mask={mask}: expected one final-layer row, "
                f"found {len(final_rows)}."
            )
        final_score = float(final_rows[0]["score_mean"])

        eligible = peak_score > tau
        retention: Optional[float]
        if eligible:
            retention = final_score / peak_score
            if not math.isfinite(retention):
                raise ContractError(
                    f"subset_mask={mask}: non-finite retention."
                )
            # Since layer 30 is included in the peak search and peak>tau>0,
            # a positive final score cannot exceed 1 by construction.
            if retention > 1.0 + 1e-12:
                raise ContractError(
                    f"subset_mask={mask}: retention > 1 unexpectedly: "
                    f"{retention}"
                )
        else:
            retention = None

        final_near_floor = abs(final_score) <= tau

        out.append(
            {
                "order_k": order,
                "subset_mask": mask,
                "n_layers": len(rows),
                "peak_layer": peak_layer,
                "peak_score": peak_score,
                "final_layer": FINAL_LAYER,
                "final_score": final_score,
                "tau_q99_abs_fixed_subset_null": tau,
                "retention_eligible": eligible,
                "final_near_floor": final_near_floor,
                "retain_s": retention,
                "retain_s_status": (
                    "estimable" if eligible else "not_estimable_peak_at_or_below_floor"
                ),
            }
        )

    return out


def compute_order_retention(
    subset_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    by_order: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for r in subset_rows:
        by_order[int(r["order_k"])].append(r)

    out: List[Dict[str, object]] = []
    for order in sorted(EXPECTED_SUBSETS_BY_ORDER):
        rows = sorted(by_order[order], key=lambda r: int(r["subset_mask"]))
        expected_masks = EXPECTED_SUBSETS_BY_ORDER[order]
        observed_masks = [int(r["subset_mask"]) for r in rows]
        if observed_masks != expected_masks:
            raise ContractError(
                f"Order k={order}: expected masks {expected_masks}, "
                f"observed {observed_masks}."
            )

        eligible_rows = [r for r in rows if bool(r["retention_eligible"])]
        all_estimable = len(eligible_rows) == len(rows)

        if all_estimable:
            vals = [float(r["retain_s"]) for r in rows]
            retain_k: Optional[float] = sum(vals) / len(vals)
            status = "estimable_all_subsets"
        else:
            retain_k = None
            status = "not_estimable_one_or_more_subsets"

        out.append(
            {
                "order_k": order,
                "n_subsets_expected": len(rows),
                "n_subsets_estimable": len(eligible_rows),
                "all_subsets_estimable": all_estimable,
                "retain_k": retain_k,
                "retain_k_status": status,
            }
        )

    return out


def compute_contrast(
    order_rows: List[Dict[str, object]]
) -> Dict[str, object]:
    omap = {int(r["order_k"]): r for r in order_rows}

    def val(k: int) -> Optional[float]:
        x = omap[k]["retain_k"]
        return None if x is None else float(x)

    r1, r2, r3, r4 = val(1), val(2), val(3), val(4)

    lo = None if r1 is None or r2 is None else (r1 + r2) / 2.0
    ho = None if r3 is None or r4 is None else (r3 + r4) / 2.0
    delta = None if lo is None or ho is None else ho - lo

    return {
        "retain_1": r1,
        "retain_2": r2,
        "retain_3": r3,
        "retain_4": r4,
        "lo_retention": lo,
        "ho_retention": ho,
        "delta_ho_minus_lo": delta,
        "directional_hypothesis": "delta_ho_minus_lo < 0",
        "contrast_estimable": delta is not None,
        "inferential_test_performed": False,
        "note": (
            "Point contrast only. Final cube-/subset-aware inferential "
            "resampling is not performed by Code 13."
        ),
    }


def csv_value(x: object) -> object:
    return "" if x is None else x


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: csv_value(row.get(k)) for k in fieldnames})


def self_test() -> None:
    # Quantile interpolation check.
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert linear_quantile(vals, 0.50) == 2.0
    assert abs(linear_quantile(vals, 0.25) - 1.0) < 1e-15

    # Frozen eligibility rule check.
    tau = 0.01
    fake: Dict[int, List[Dict[str, float]]] = {}
    for mask, order in EXPECTED_SUBSET_TO_ORDER.items():
        rows = []
        for layer in EXPECTED_LAYERS:
            # mask 1 is deliberately floor-level and therefore ineligible.
            if mask == 1:
                score = 0.005 if layer == 10 else 0.001
            else:
                score = 0.02 if layer == 10 else (0.01 if layer == 30 else 0.002)
            rows.append(
                {
                    "layer": layer,
                    "order_k": order,
                    "subset_mask": mask,
                    "score_mean": score,
                    "critic_seed_sd": 0.0,
                    "score_min": score,
                    "score_max": score,
                }
            )
        fake[mask] = rows

    subset = compute_subset_retention(fake, tau)
    m1 = next(r for r in subset if r["subset_mask"] == 1)
    assert m1["retention_eligible"] is False
    assert m1["retain_s"] is None

    m2 = next(r for r in subset if r["subset_mask"] == 2)
    assert m2["retention_eligible"] is True
    assert abs(float(m2["retain_s"]) - 0.5) < 1e-15

    order = compute_order_retention(subset)
    o1 = next(r for r in order if r["order_k"] == 1)
    assert o1["retain_k"] is None  # strict: one ineligible subset => order NA
    o2 = next(r for r in order if r["order_k"] == 2)
    assert abs(float(o2["retain_k"]) - 0.5) < 1e-15

    contrast = compute_contrast(order)
    assert contrast["contrast_estimable"] is False

    print(f"{CODE_NAME} SELF-TEST: PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-root",
        default=str(PROJECT_DEFAULT),
        help="ORBIT-FMIB project root (default: current directory)",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal logic tests only; do not read/write production data.",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate production inputs and frozen threshold, but do not write outputs.",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    root = Path(args.project_root).resolve()
    traj_path = root / TRAJ_REL
    null_path = root / NULL_REL
    out_dir = root / OUT_DIR_REL

    null_scores, null_stats = load_null_scores(null_path)
    tau = float(null_stats["q99_abs"])

    by_subset, _, _ = load_trajectories(traj_path)

    # Freeze-rule integrity check against the already documented validated run.
    # This is a guardrail, not the source of tau: tau is computed from the CSV.
    documented_tau = 0.006777383486429849
    if abs(tau - documented_tau) > 1e-15:
        raise ContractError(
            "Computed q99 absolute fixed-S null threshold differs from the "
            f"documented frozen value. computed={tau!r}, "
            f"documented={documented_tau!r}. "
            "Stop and audit the null input before proceeding."
        )

    print(f"{CODE_NAME} PREFLIGHT: PASS")
    print(f"Fixed-S null rows: {len(null_scores)}")
    print(f"Frozen tau = Q0.99(|D_null|): {tau:.17g}")
    print(f"Trajectory cells: {sum(len(v) for v in by_subset.values())}")
    print(f"Fixed subsets: {len(by_subset)}")
    print(f"Layers per subset: {len(EXPECTED_LAYERS)}")

    if args.preflight_only:
        print("Production outputs written: NO")
        return 0

    subset_rows = compute_subset_retention(by_subset, tau)
    order_rows = compute_order_retention(subset_rows)
    contrast = compute_contrast(order_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    subset_path = out_dir / SUBSET_OUT
    order_path = out_dir / ORDER_OUT
    contrast_path = out_dir / CONTRAST_OUT
    manifest_path = out_dir / MANIFEST_OUT

    subset_fields = [
        "order_k",
        "subset_mask",
        "n_layers",
        "peak_layer",
        "peak_score",
        "final_layer",
        "final_score",
        "tau_q99_abs_fixed_subset_null",
        "retention_eligible",
        "final_near_floor",
        "retain_s",
        "retain_s_status",
    ]
    order_fields = [
        "order_k",
        "n_subsets_expected",
        "n_subsets_estimable",
        "all_subsets_estimable",
        "retain_k",
        "retain_k_status",
    ]
    contrast_fields = [
        "retain_1",
        "retain_2",
        "retain_3",
        "retain_4",
        "lo_retention",
        "ho_retention",
        "delta_ho_minus_lo",
        "directional_hypothesis",
        "contrast_estimable",
        "inferential_test_performed",
        "note",
    ]

    write_csv(subset_path, subset_fields, subset_rows)
    write_csv(order_path, order_fields, order_rows)
    write_csv(contrast_path, contrast_fields, [contrast])

    source_path = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "status": "complete_pass",
        "overall_pass": True,
        "purpose": (
            "Compute pre-specified floor-safe retention within each fixed "
            "positional subset before interaction-order aggregation."
        ),
        "scientific_contract": {
            "primary_measurement_unit": "fixed positional subset S",
            "retention_formula_if_eligible": "Retain_S = D_30,S / max_l D_l,S",
            "eligibility_rule": "max_l D_l,S > tau",
            "noneligible_rule": "Retain_S = NA / not estimable",
            "final_near_floor_rule": "|D_30,S| <= tau",
            "order_aggregation": (
                "mean Retain_S within order only if every expected subset "
                "for that order is estimable"
            ),
            "lo_definition": "(Retain_1 + Retain_2) / 2",
            "ho_definition": "(Retain_3 + Retain_4) / 2",
            "delta_definition": "HO - LO",
            "directional_hypothesis": "HO - LO < 0",
            "negative_scores_clamped": False,
            "null_scores_subtracted": False,
            "critic_seeds_are_biological_replicates": False,
            "inferential_resampling_performed": False,
        },
        "floor_contract": {
            "source": str(NULL_REL),
            "statistic": "Q0.99(abs(null_score_mean))",
            "quantile_method": "linear interpolation",
            "n_null_rows": len(null_scores),
            "tau": tau,
            "null_summary": null_stats,
            "threshold_computed_from_source_file": True,
            "threshold_hardcoded_as_source_of_truth": False,
            "documented_tau_guardrail": documented_tau,
        },
        "grid_audit": {
            "trajectory_rows": sum(len(v) for v in by_subset.values()),
            "trajectory_count": len(by_subset),
            "layers_per_trajectory": len(EXPECTED_LAYERS),
            "expected_subsets_by_order": EXPECTED_SUBSETS_BY_ORDER,
        },
        "retention_audit": {
            "n_subsets_total": len(subset_rows),
            "n_subsets_estimable": sum(
                1 for r in subset_rows if r["retention_eligible"]
            ),
            "n_subsets_not_estimable": sum(
                1 for r in subset_rows if not r["retention_eligible"]
            ),
            "n_final_near_floor": sum(
                1 for r in subset_rows if r["final_near_floor"]
            ),
            "contrast_estimable": bool(contrast["contrast_estimable"]),
        },
        "inputs": {
            "trajectory_file": str(TRAJ_REL),
            "trajectory_sha256": sha256_file(traj_path),
            "fixed_subset_null_file": str(NULL_REL),
            "fixed_subset_null_sha256": sha256_file(null_path),
        },
        "outputs": {
            "subset_retention": SUBSET_OUT,
            "order_retention_summary": ORDER_OUT,
            "point_contrast": CONTRAST_OUT,
        },
        "production_source": os.path.relpath(source_path, root)
        if source_path.is_relative_to(root)
        else str(source_path),
        "production_source_sha256": sha256_file(source_path),
        "interpretation_limits": [
            "Neural-DV scores are comparative dependence scores, not calibrated exact MI in nats.",
            "Near-floor failure to detect dependence is not proof of zero MI.",
            "Code 13 computes point retention only; confirmatory uncertainty requires later cube-/subset-aware resampling.",
            "If a required subset/order is not estimable, the confirmatory contrast remains NA rather than changing the estimand post hoc.",
        ],
    }

    # Write once, then hash non-manifest outputs and add them.
    manifest["output_sha256"] = {
        SUBSET_OUT: sha256_file(subset_path),
        ORDER_OUT: sha256_file(order_path),
        CONTRAST_OUT: sha256_file(contrast_path),
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"{CODE_NAME}: PASS")
    print(f"Retention-eligible subsets: {manifest['retention_audit']['n_subsets_estimable']}/15")
    print(f"Final-layer near-floor subsets: {manifest['retention_audit']['n_final_near_floor']}/15")
    print(f"Point HO-vs-LO contrast estimable: {contrast['contrast_estimable']}")
    print("Inferential resampling performed: NO")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
