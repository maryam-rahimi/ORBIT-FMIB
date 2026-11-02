#!/usr/bin/env python3
"""
ORBIT-FMIB CODE 15 — LAYERWISE DYNAMICS ANALYSIS

Purpose
-------
Analyze the frozen corrected-IID GB1 + ESM-2 fixed-subset dependence trajectories
without refitting any critic and without changing the frozen Code-13 retention
estimand.

Authoritative inputs
--------------------
04 Experiments/EXP02_GB1_ESM2/IID_Production_20260910/
  gb1_subset_mi_seed_results.csv
  gb1_subset_mi_manifest.json
  Code12/
    gb1_subset_trajectories_long.csv
    gb1_subset_trajectory_index.csv
    gb1_subset_trajectory_manifest.json
  Code13/
    gb1_subset_retention.csv
    gb1_order_retention_summary.csv
    gb1_retention_contrast.csv
    gb1_retention_manifest.json

Scientific contract
-------------------
1. The 15 fixed positional subsets remain the primary measurement objects.
2. All 31 ESM-2 stages L0..L30 are retained.
3. No MINE fitting is performed.
4. No null subtraction or score clamping is performed.
5. Peak normalization is performed WITHIN fixed subset only:
       N_{l,S} = D_{l,S} / max_j D_{j,S}
   and only for Code-13 retention-eligible subsets.
6. Order-level normalized trajectories are descriptive averages of independently
   normalized fixed-subset trajectories; they are not pooled mutual-information
   estimates.
7. Code-13 peak layers, final scores, subset retention, order retention, and
   HO-vs-LO contrast are rederived only as integrity checks and must match the
   frozen Code-13 outputs.
8. Critic-seed trajectories are used only as estimator-stability diagnostics.
9. No p-value, confidence interval, or formal biological inference is produced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CODE_NAME = "ORBIT-FMIB CODE 15 — LAYERWISE DYNAMICS ANALYSIS"
SCHEMA_VERSION = 1

LAYERS = tuple(range(31))
FINAL_LAYER = 30
POSITIONS = (39, 40, 41, 54)

SUBSETS_BY_ORDER = {
    1: (1, 2, 4, 8),
    2: (3, 5, 6, 9, 10, 12),
    3: (7, 11, 13, 14),
    4: (15,),
}
SUBSET_TO_ORDER = {
    mask: order
    for order, masks in SUBSETS_BY_ORDER.items()
    for mask in masks
}
EXPECTED_MASKS = tuple(sorted(SUBSET_TO_ORDER))
EXPECTED_TRAJECTORY_ROWS = 31 * 15
EXPECTED_SEED_ROWS = 31 * 15 * 3
EXPECTED_CRITIC_SEEDS = (11, 23, 37)

PROD_REL = Path("04 Experiments/EXP02_GB1_ESM2/IID_Production_20260910")
CODE12_REL = PROD_REL / "Code12"
CODE13_REL = PROD_REL / "Code13"
OUT_REL = PROD_REL / "Code15"

TRAJ_REL = CODE12_REL / "gb1_subset_trajectories_long.csv"
INDEX_REL = CODE12_REL / "gb1_subset_trajectory_index.csv"
CODE12_MANIFEST_REL = CODE12_REL / "gb1_subset_trajectory_manifest.json"

SEED_REL = PROD_REL / "gb1_subset_mi_seed_results.csv"
CODE11_MANIFEST_REL = PROD_REL / "gb1_subset_mi_manifest.json"

RET_SUBSET_REL = CODE13_REL / "gb1_subset_retention.csv"
RET_ORDER_REL = CODE13_REL / "gb1_order_retention_summary.csv"
RET_CONTRAST_REL = CODE13_REL / "gb1_retention_contrast.csv"
CODE13_MANIFEST_REL = CODE13_REL / "gb1_retention_manifest.json"

OUT_SUBSET_SUMMARY = "gb1_layerwise_subset_summary.csv"
OUT_NORMALIZED_LONG = "gb1_layerwise_subset_normalized_long.csv"
OUT_ORDER_NORMALIZED = "gb1_order_layerwise_peak_normalized.csv"
OUT_ORDER_RAW = "gb1_order_layerwise_raw_descriptive.csv"
OUT_SEED_DYNAMICS = "gb1_subset_seed_dynamics.csv"
OUT_ORDER_SUMMARY = "gb1_layerwise_order_summary.csv"
OUT_MANIFEST = "gb1_layerwise_dynamics_manifest.json"


class ContractError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ContractError(f"Missing {label}: {path}")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, obj: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def as_int(x: object, name: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ContractError(f"Could not parse integer {name}={x!r}") from e


def as_float(x: object, name: str) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise ContractError(f"Could not parse float {name}={x!r}") from e
    if not math.isfinite(v):
        raise ContractError(f"Non-finite {name}={x!r}")
    return v


def boolish(x: object) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    raise ContractError(f"Could not parse boolean value {x!r}")


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def sample_sd(vals: Sequence[float]) -> float:
    return 0.0 if len(vals) <= 1 else float(statistics.stdev(vals))


def mean(vals: Sequence[float]) -> float:
    if not vals:
        raise ContractError("Attempted mean of empty sequence.")
    return float(statistics.fmean(vals))


def mode_smallest(vals: Sequence[int]) -> int:
    c = Counter(vals)
    best = max(c.values())
    return min(v for v, n in c.items() if n == best)


def mask_positions(mask: int) -> Tuple[int, ...]:
    out = []
    for bit, pos in enumerate(POSITIONS):
        if mask & (1 << bit):
            out.append(pos)
    return tuple(out)


def subset_label(mask: int) -> str:
    return "{" + ",".join(str(x) for x in mask_positions(mask)) + "}"


def peak_row(rows: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    # Same tie rule as Code 13: highest score, then earliest layer.
    return max(rows, key=lambda r: (float(r["score_mean"]), -int(r["layer"])))


def first_layer_at_fraction(rows: Sequence[Mapping[str, float]], peak: float, frac: float) -> int:
    threshold = frac * peak
    for r in sorted(rows, key=lambda x: int(x["layer"])):
        if float(r["score_mean"]) >= threshold:
            return int(r["layer"])
    raise ContractError("No layer reached requested peak fraction; inconsistent peak logic.")


def adjacent_landmarks(normalized_by_layer: Mapping[int, float]) -> Dict[str, object]:
    gains = []
    for l in range(30):
        delta = float(normalized_by_layer[l + 1]) - float(normalized_by_layer[l])
        gains.append((delta, l, l + 1))

    # Deterministic earliest-layer tie handling with a tight numerical tolerance.
    # This prevents machine-precision differences between mathematically equal
    # adjacent changes from changing the reported descriptive landmark.
    max_delta = max(x[0] for x in gains)
    max_candidates = [
        x for x in gains
        if math.isclose(x[0], max_delta, rel_tol=1e-12, abs_tol=1e-15)
    ]
    max_gain = min(max_candidates, key=lambda x: x[1])

    min_delta = min(x[0] for x in gains)
    min_candidates = [
        x for x in gains
        if math.isclose(x[0], min_delta, rel_tol=1e-12, abs_tol=1e-15)
    ]
    max_drop = min(min_candidates, key=lambda x: x[1])

    return {
        "largest_adjacent_gain": max_gain[0],
        "largest_adjacent_gain_from_layer": max_gain[1],
        "largest_adjacent_gain_to_layer": max_gain[2],
        "largest_adjacent_drop": max_drop[0],
        "largest_adjacent_drop_from_layer": max_drop[1],
        "largest_adjacent_drop_to_layer": max_drop[2],
    }


def validate_code11_manifest(prod: Path, manifest: dict, seed_path: Path) -> None:
    if manifest.get("overall_pass") is not True:
        raise ContractError("Code-11 production manifest overall_pass is not True.")
    audit = manifest.get("computational_audit", {})
    if audit.get("overall_pass") is not True:
        raise ContractError("Code-11 computational audit is not PASS.")
    observed = audit.get("observed", {})
    if int(observed.get("primary_fits", -1)) != EXPECTED_SEED_ROWS:
        raise ContractError("Code-11 manifest does not report 1,395 primary fits.")
    if int(observed.get("subset_cells", -1)) != EXPECTED_TRAJECTORY_ROWS:
        raise ContractError("Code-11 manifest does not report 465 subset cells.")
    if observed.get("all_results_finite") is not True:
        raise ContractError("Code-11 manifest does not report all finite results.")

    hashes = manifest.get("file_sha256", {})
    expected = hashes.get(seed_path.name)
    if not expected:
        raise ContractError(f"Code-11 manifest lacks hash for {seed_path.name}.")
    if sha256_file(seed_path) != expected:
        raise ContractError("Code-11 seed-result hash mismatch.")

    est = manifest.get("estimator_contract", {})
    if est.get("no_clamping") is not True:
        raise ContractError("Code-11 estimator contract does not certify no clamping.")
    if est.get("no_null_subtraction") is not True:
        raise ContractError("Code-11 estimator contract does not certify no null subtraction.")
    if est.get("absolute_mi_accuracy_claimed") is not False:
        raise ContractError("Code-11 estimator contract changed absolute-MI interpretation.")


def validate_code12_manifest(prod: Path, manifest: dict, manifest_path: Path,
                             traj_path: Path, index_path: Path,
                             code11_manifest_path: Path) -> None:
    if manifest.get("overall_pass") is not True:
        raise ContractError("Code-12 manifest overall_pass is not True.")
    audit = manifest.get("audit", {})
    if audit.get("overall_pass") is not True:
        raise ContractError("Code-12 audit is not PASS.")
    if int(audit.get("unique_layer_subset_cells", -1)) != EXPECTED_TRAJECTORY_ROWS:
        raise ContractError("Code-12 audit does not report 465 layer-subset cells.")
    if int(audit.get("trajectory_count", -1)) != 15:
        raise ContractError("Code-12 audit does not report 15 trajectories.")
    if int(audit.get("layers_per_trajectory", -1)) != 31:
        raise ContractError("Code-12 audit does not report 31 layers per trajectory.")

    contract = manifest.get("trajectory_contract", {})
    if contract.get("negative_scores_clamped") is not False:
        raise ContractError("Code-12 trajectory contract changed clamping.")
    if contract.get("null_scores_subtracted") is not False:
        raise ContractError("Code-12 trajectory contract changed null subtraction.")
    if contract.get("pooled_by_order_mi_estimation") is not False:
        raise ContractError("Code-12 unexpectedly reports pooled-by-order MI estimation.")
    if contract.get("retention_computed") is not False:
        raise ContractError("Code-12 unexpectedly reports retention computation.")

    hashes = manifest.get("file_sha256", {})
    for p in (traj_path, index_path):
        expected = hashes.get(p.name)
        if not expected:
            raise ContractError(f"Code-12 manifest lacks hash for {p.name}.")
        if sha256_file(p) != expected:
            raise ContractError(f"Code-12 output hash mismatch: {p.name}")

    upstream = manifest.get("upstream", {})
    expected_code11_sha = upstream.get("code11_manifest_sha256")
    if not expected_code11_sha:
        raise ContractError("Code-12 manifest lacks upstream Code-11 manifest hash.")
    if sha256_file(code11_manifest_path) != expected_code11_sha:
        raise ContractError("Current production Code-11 manifest does not match Code-12 upstream hash.")


def validate_code13_manifest(manifest: dict, traj_path: Path,
                             subset_ret_path: Path, order_ret_path: Path,
                             contrast_path: Path) -> float:
    if manifest.get("overall_pass") is not True:
        raise ContractError("Code-13 manifest overall_pass is not True.")
    if manifest.get("status") != "complete_pass":
        raise ContractError("Code-13 manifest status is not complete_pass.")

    sci = manifest.get("scientific_contract", {})
    if sci.get("negative_scores_clamped") is not False:
        raise ContractError("Code-13 scientific contract changed clamping.")
    if sci.get("null_scores_subtracted") is not False:
        raise ContractError("Code-13 scientific contract changed null subtraction.")
    if sci.get("inferential_resampling_performed") is not False:
        raise ContractError("Code-13 unexpectedly reports inferential resampling.")

    inputs = manifest.get("inputs", {})
    expected_traj_sha = inputs.get("trajectory_sha256")
    if not expected_traj_sha:
        raise ContractError("Code-13 manifest lacks trajectory input SHA-256.")
    if sha256_file(traj_path) != expected_traj_sha:
        raise ContractError("Code-13 trajectory input hash does not match frozen Code-12 trajectory.")

    output_hashes = manifest.get("output_sha256", {})
    for p in (subset_ret_path, order_ret_path, contrast_path):
        expected = output_hashes.get(p.name)
        if not expected:
            raise ContractError(f"Code-13 manifest lacks output hash for {p.name}.")
        if sha256_file(p) != expected:
            raise ContractError(f"Code-13 output hash mismatch: {p.name}")

    floor = manifest.get("floor_contract", {})
    tau = as_float(floor.get("tau"), "Code13 floor tau")
    if floor.get("threshold_computed_from_source_file") is not True:
        raise ContractError("Code-13 floor was not certified as computed from source file.")
    return tau


def load_and_validate_index(path: Path) -> None:
    rows = read_csv(path)
    if len(rows) != 15:
        raise ContractError(f"Expected 15 Code-12 index rows; found {len(rows)}.")
    observed = set()
    for r in rows:
        k = as_int(r["order_k"], "index order_k")
        mask = as_int(r["subset_mask"], "index subset_mask")
        n_layers = as_int(r["n_layers"], "index n_layers")
        if SUBSET_TO_ORDER.get(mask) != k or n_layers != 31:
            raise ContractError(f"Invalid Code-12 trajectory index row for mask={mask}.")
        observed.add(mask)
    if observed != set(EXPECTED_MASKS):
        raise ContractError("Code-12 trajectory index subset set mismatch.")


def load_trajectories(path: Path) -> Dict[int, List[Dict[str, float]]]:
    raw = read_csv(path)
    if len(raw) != EXPECTED_TRAJECTORY_ROWS:
        raise ContractError(f"Expected 465 trajectory rows; found {len(raw)}.")
    required = {
        "layer", "order_k", "subset_mask", "score_mean",
        "critic_seed_sd", "score_min", "score_max",
    }
    if not raw or not required.issubset(raw[0]):
        missing = required - (set(raw[0]) if raw else set())
        raise ContractError(f"Trajectory file missing columns: {sorted(missing)}")

    by_subset: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    seen = set()
    for i, r in enumerate(raw, start=2):
        layer = as_int(r["layer"], f"row {i} layer")
        order = as_int(r["order_k"], f"row {i} order")
        mask = as_int(r["subset_mask"], f"row {i} mask")
        score = as_float(r["score_mean"], f"row {i} score_mean")
        sd = as_float(r["critic_seed_sd"], f"row {i} critic_seed_sd")
        lo = as_float(r["score_min"], f"row {i} score_min")
        hi = as_float(r["score_max"], f"row {i} score_max")
        if layer not in LAYERS:
            raise ContractError(f"Unexpected layer {layer}.")
        if SUBSET_TO_ORDER.get(mask) != order:
            raise ContractError(f"Mask/order mismatch at row {i}.")
        if sd < 0 or not (lo <= score <= hi):
            raise ContractError(f"Invalid score/seed range at row {i}.")
        key = (layer, mask)
        if key in seen:
            raise ContractError(f"Duplicate layer/subset cell {key}.")
        seen.add(key)
        by_subset[mask].append({
            "layer": layer,
            "order_k": order,
            "subset_mask": mask,
            "score_mean": score,
            "critic_seed_sd": sd,
            "score_min": lo,
            "score_max": hi,
        })

    if set(by_subset) != set(EXPECTED_MASKS):
        raise ContractError("Trajectory subset set mismatch.")
    for mask, rows in by_subset.items():
        rows.sort(key=lambda x: int(x["layer"]))
        if tuple(int(r["layer"]) for r in rows) != LAYERS:
            raise ContractError(f"Mask {mask} does not contain exactly L0..L30.")
    return dict(by_subset)


def load_code13_subset(path: Path) -> Dict[int, dict]:
    rows = read_csv(path)
    if len(rows) != 15:
        raise ContractError(f"Expected 15 Code-13 subset rows; found {len(rows)}.")
    out = {}
    for r in rows:
        mask = as_int(r["subset_mask"], "Code13 subset_mask")
        if mask in out or mask not in SUBSET_TO_ORDER:
            raise ContractError(f"Unexpected/duplicate Code13 subset mask {mask}.")
        out[mask] = r
    if set(out) != set(EXPECTED_MASKS):
        raise ContractError("Code-13 subset-retention mask set mismatch.")
    return out


def load_code13_order(path: Path) -> Dict[int, dict]:
    rows = read_csv(path)
    if len(rows) != 4:
        raise ContractError(f"Expected 4 Code-13 order rows; found {len(rows)}.")
    out = {as_int(r["order_k"], "Code13 order_k"): r for r in rows}
    if set(out) != {1, 2, 3, 4}:
        raise ContractError("Code-13 order set mismatch.")
    return out


def load_code13_contrast(path: Path) -> dict:
    rows = read_csv(path)
    if len(rows) != 1:
        raise ContractError(f"Expected one Code-13 contrast row; found {len(rows)}.")
    return rows[0]


def validate_against_code13(
    by_subset: Mapping[int, Sequence[Mapping[str, float]]],
    code13_subset: Mapping[int, Mapping[str, object]],
    code13_order: Mapping[int, Mapping[str, object]],
    contrast: Mapping[str, object],
    tau: float,
) -> Dict[int, float]:
    retain_by_mask: Dict[int, float] = {}
    for mask in EXPECTED_MASKS:
        rows = by_subset[mask]
        p = peak_row(rows)
        peak = float(p["score_mean"])
        pl = int(p["layer"])
        final = float(rows[-1]["score_mean"])
        c13 = code13_subset[mask]

        if as_int(c13["peak_layer"], "Code13 peak_layer") != pl:
            raise ContractError(f"Mask {mask}: Code15 peak layer != Code13 peak layer.")
        if not close(as_float(c13["peak_score"], "Code13 peak_score"), peak):
            raise ContractError(f"Mask {mask}: Code15 peak score != Code13 peak score.")
        if not close(as_float(c13["final_score"], "Code13 final_score"), final):
            raise ContractError(f"Mask {mask}: Code15 final score != Code13 final score.")
        if not close(as_float(c13["tau_q99_abs_fixed_subset_null"], "Code13 tau"), tau):
            raise ContractError(f"Mask {mask}: subset tau != Code13 manifest tau.")

        eligible = boolish(c13["retention_eligible"])
        if not eligible:
            raise ContractError(
                f"Mask {mask} is not retention-eligible. Code15 peak normalization is "
                "intentionally blocked rather than redefining the estimand."
            )
        if peak <= tau:
            raise ContractError(f"Mask {mask}: Code13 says eligible but peak <= tau.")

        retain = final / peak
        c13_ret = as_float(c13["retain_s"], "Code13 retain_s")
        if not close(retain, c13_ret):
            raise ContractError(f"Mask {mask}: rederived retention != Code13 retention.")
        retain_by_mask[mask] = retain

    order_rederived = {}
    for k, masks in SUBSETS_BY_ORDER.items():
        val = mean([retain_by_mask[m] for m in masks])
        order_rederived[k] = val
        c13_val = as_float(code13_order[k]["retain_k"], f"Code13 retain_k k={k}")
        if not close(val, c13_val):
            raise ContractError(f"Order k={k}: rederived retention != Code13 retention.")

    lo = mean([order_rederived[1], order_rederived[2]])
    ho = mean([order_rederived[3], order_rederived[4]])
    delta = ho - lo
    if not close(lo, as_float(contrast["lo_retention"], "Code13 LO")):
        raise ContractError("Re-derived LO retention != Code13 contrast.")
    if not close(ho, as_float(contrast["ho_retention"], "Code13 HO")):
        raise ContractError("Re-derived HO retention != Code13 contrast.")
    if not close(delta, as_float(contrast["delta_ho_minus_lo"], "Code13 delta")):
        raise ContractError("Re-derived HO-LO delta != Code13 contrast.")
    if boolish(contrast["inferential_test_performed"]):
        raise ContractError("Code13 contrast unexpectedly reports inferential testing.")
    return retain_by_mask


def load_seed_results(path: Path) -> Dict[Tuple[int, int], List[Tuple[int, float]]]:
    rows = read_csv(path)
    if len(rows) != EXPECTED_SEED_ROWS:
        raise ContractError(f"Expected 1,395 seed rows; found {len(rows)}.")
    required = {"layer", "order_k", "subset_mask", "critic_seed", "test_mi"}
    if not rows or not required.issubset(rows[0]):
        missing = required - (set(rows[0]) if rows else set())
        raise ContractError(f"Seed result file missing columns: {sorted(missing)}")

    by_seed_subset: Dict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
    seen = set()
    for i, r in enumerate(rows, start=2):
        layer = as_int(r["layer"], f"seed row {i} layer")
        k = as_int(r["order_k"], f"seed row {i} order")
        mask = as_int(r["subset_mask"], f"seed row {i} mask")
        seed = as_int(r["critic_seed"], f"seed row {i} critic_seed")
        score = as_float(r["test_mi"], f"seed row {i} test_mi")
        if layer not in LAYERS or SUBSET_TO_ORDER.get(mask) != k:
            raise ContractError(f"Invalid seed row {i} layer/order/mask.")
        if seed not in EXPECTED_CRITIC_SEEDS:
            raise ContractError(f"Unexpected critic seed {seed}.")
        key = (layer, mask, seed)
        if key in seen:
            raise ContractError(f"Duplicate seed result {key}.")
        seen.add(key)
        by_seed_subset[(mask, seed)].append((layer, score))

    expected_keys = {(m, s) for m in EXPECTED_MASKS for s in EXPECTED_CRITIC_SEEDS}
    if set(by_seed_subset) != expected_keys:
        raise ContractError("Seed subset/critic-seed coverage mismatch.")
    for key, vals in by_seed_subset.items():
        vals.sort()
        if tuple(l for l, _ in vals) != LAYERS:
            raise ContractError(f"Seed trajectory {key} does not contain L0..L30.")
    return dict(by_seed_subset)


def build_seed_dynamics(
    by_seed_subset: Mapping[Tuple[int, int], Sequence[Tuple[int, float]]]
) -> Tuple[List[dict], Dict[int, List[dict]]]:
    rows = []
    by_mask: Dict[int, List[dict]] = defaultdict(list)
    for mask in EXPECTED_MASKS:
        for seed in EXPECTED_CRITIC_SEEDS:
            vals = by_seed_subset[(mask, seed)]
            p_layer, p_score = max(vals, key=lambda x: (x[1], -x[0]))
            final = dict(vals)[FINAL_LAYER]
            retain = final / p_score if p_score > 0 else float("nan")
            row = {
                "order_k": SUBSET_TO_ORDER[mask],
                "subset_mask": mask,
                "subset_positions": subset_label(mask),
                "critic_seed": seed,
                "seed_peak_layer": p_layer,
                "seed_peak_score": p_score,
                "seed_final_score": final,
                "seed_peak_to_final_ratio": retain,
            }
            rows.append(row)
            by_mask[mask].append(row)
    return rows, dict(by_mask)


def build_subset_outputs(
    by_subset: Mapping[int, Sequence[Mapping[str, float]]],
    code13_subset: Mapping[int, Mapping[str, object]],
    seed_by_mask: Mapping[int, Sequence[Mapping[str, object]]],
    tau: float,
) -> Tuple[List[dict], List[dict], Dict[int, Dict[int, float]]]:
    summary_rows: List[dict] = []
    normalized_long: List[dict] = []
    normalized_lookup: Dict[int, Dict[int, float]] = {}

    checkpoints = (0, 5, 10, 15, 20, 25, 30)

    for mask in EXPECTED_MASKS:
        rows = list(by_subset[mask])
        p = peak_row(rows)
        peak = float(p["score_mean"])
        pl = int(p["layer"])
        final = float(rows[-1]["score_mean"])
        retain = final / peak

        norm = {int(r["layer"]): float(r["score_mean"]) / peak for r in rows}
        normalized_lookup[mask] = norm
        landmarks = adjacent_landmarks(norm)

        seed_rows = list(seed_by_mask[mask])
        seed_peak_layers = [int(r["seed_peak_layer"]) for r in seed_rows]
        seed_retains = [
            float(r["seed_peak_to_final_ratio"])
            for r in seed_rows
            if math.isfinite(float(r["seed_peak_to_final_ratio"]))
        ]

        score_by_layer = {int(r["layer"]): float(r["score_mean"]) for r in rows}
        sd_by_layer = {int(r["layer"]): float(r["critic_seed_sd"]) for r in rows}

        out = {
            "order_k": SUBSET_TO_ORDER[mask],
            "subset_mask": mask,
            "subset_positions": subset_label(mask),
            "n_layers": 31,
            "peak_layer": pl,
            "peak_score": peak,
            "final_score": final,
            "retain_s": retain,
            "post_peak_fraction_lost": 1.0 - retain,
            "first_ge_50pct_peak_layer": first_layer_at_fraction(rows, peak, 0.50),
            "first_ge_90pct_peak_layer": first_layer_at_fraction(rows, peak, 0.90),
            "max_critic_seed_sd_across_layers": max(sd_by_layer.values()),
            "mean_critic_seed_sd_across_layers": mean(list(sd_by_layer.values())),
            "seed_peak_layer_min": min(seed_peak_layers),
            "seed_peak_layer_max": max(seed_peak_layers),
            "seed_peak_layer_range": max(seed_peak_layers) - min(seed_peak_layers),
            "seed_peak_layer_mode": mode_smallest(seed_peak_layers),
            "seed_peak_to_final_ratio_mean": mean(seed_retains) if seed_retains else "",
            "seed_peak_to_final_ratio_sd": sample_sd(seed_retains) if seed_retains else "",
            **landmarks,
        }
        for l in checkpoints:
            out[f"score_L{l}"] = score_by_layer[l]
            out[f"normalized_L{l}"] = norm[l]
        summary_rows.append(out)

        for r in rows:
            l = int(r["layer"])
            normalized_long.append({
                "layer": l,
                "order_k": SUBSET_TO_ORDER[mask],
                "subset_mask": mask,
                "subset_positions": subset_label(mask),
                "score_mean": float(r["score_mean"]),
                "critic_seed_sd": float(r["critic_seed_sd"]),
                "peak_score": peak,
                "peak_layer": pl,
                "peak_normalized_score": norm[l],
                "tau_q99_abs_fixed_subset_null": tau,
                "retain_s": retain,
            })

    return summary_rows, normalized_long, normalized_lookup


def build_order_outputs(
    by_subset: Mapping[int, Sequence[Mapping[str, float]]],
    normalized_lookup: Mapping[int, Mapping[int, float]],
    subset_summary: Sequence[Mapping[str, object]],
    code13_order: Mapping[int, Mapping[str, object]],
) -> Tuple[List[dict], List[dict], List[dict]]:
    raw_rows = []
    norm_rows = []
    order_summary = []

    subset_summary_by_mask = {int(r["subset_mask"]): r for r in subset_summary}

    for k, masks in SUBSETS_BY_ORDER.items():
        for l in LAYERS:
            raw_vals = []
            seed_sds = []
            norm_vals = []
            for mask in masks:
                r = next(x for x in by_subset[mask] if int(x["layer"]) == l)
                raw_vals.append(float(r["score_mean"]))
                seed_sds.append(float(r["critic_seed_sd"]))
                norm_vals.append(float(normalized_lookup[mask][l]))

            raw_rows.append({
                "layer": l,
                "order_k": k,
                "n_subsets": len(masks),
                "descriptive_raw_score_mean": mean(raw_vals),
                "descriptive_raw_score_sd_across_subsets": sample_sd(raw_vals),
                "min_subset_raw_score": min(raw_vals),
                "max_subset_raw_score": max(raw_vals),
                "max_critic_seed_sd": max(seed_sds),
                "interpretation": "descriptive mean of independently estimated fixed-subset neural dependence scores",
            })
            norm_rows.append({
                "layer": l,
                "order_k": k,
                "n_subsets": len(masks),
                "mean_peak_normalized_score": mean(norm_vals),
                "sd_peak_normalized_score_across_subsets": sample_sd(norm_vals),
                "min_peak_normalized_score": min(norm_vals),
                "max_peak_normalized_score": max(norm_vals),
                "interpretation": "descriptive mean after within-subset peak normalization",
            })

        order_norm = [r for r in norm_rows if int(r["order_k"]) == k]
        peak_order_row = max(
            order_norm,
            key=lambda r: (float(r["mean_peak_normalized_score"]), -int(r["layer"]))
        )
        order_curve = {
            int(r["layer"]): float(r["mean_peak_normalized_score"])
            for r in order_norm
        }
        land = adjacent_landmarks(order_curve)
        final_norm = order_curve[FINAL_LAYER]
        c13_retain = as_float(code13_order[k]["retain_k"], f"Code13 retain_k k={k}")
        if not close(final_norm, c13_retain):
            raise ContractError(
                f"Order k={k}: final normalized trajectory mean does not equal Code13 retain_k."
            )
        subset_peaks = [int(subset_summary_by_mask[m]["peak_layer"]) for m in masks]
        order_summary.append({
            "order_k": k,
            "n_subsets": len(masks),
            "order_normalized_curve_peak_layer": int(peak_order_row["layer"]),
            "order_normalized_curve_peak_value": float(peak_order_row["mean_peak_normalized_score"]),
            "final_mean_peak_normalized_score": final_norm,
            "code13_retain_k": c13_retain,
            "subset_peak_layer_min": min(subset_peaks),
            "subset_peak_layer_max": max(subset_peaks),
            "subset_peak_layer_mean": mean(subset_peaks),
            "subset_peak_layer_median": float(statistics.median(subset_peaks)),
            **land,
        })

    return raw_rows, norm_rows, order_summary


def self_test() -> None:
    assert mask_positions(1) == (39,)
    assert mask_positions(3) == (39, 40)
    assert mask_positions(15) == (39, 40, 41, 54)
    fake = [{"layer": l, "score_mean": float(l)} for l in LAYERS]
    assert int(peak_row(fake)["layer"]) == 30
    fake2 = [{"layer": l, "score_mean": 1.0 if l in (7, 9) else 0.0} for l in LAYERS]
    assert int(peak_row(fake2)["layer"]) == 7
    # Use a trajectory with a unique largest adjacent gain at L0->L1.
    # Avoid equal floating-point increments, whose tiny rounding differences can
    # make a tie-based unit test platform-dependent.
    norm = {0: 0.0}
    for l in range(1, 31):
        norm[l] = 0.20 + 0.80 * ((l - 1) / 29.0)
    land = adjacent_landmarks(norm)
    assert land["largest_adjacent_gain_from_layer"] == 0
    assert land["largest_adjacent_gain_to_layer"] == 1
    assert close(float(land["largest_adjacent_gain"]), 0.20)

    # Regression test for mathematically equal increments represented in binary
    # floating point: earliest-layer tie handling must remain deterministic.
    linear = {l: l / 30.0 for l in LAYERS}
    linear_land = adjacent_landmarks(linear)
    assert linear_land["largest_adjacent_gain_from_layer"] == 0
    assert linear_land["largest_adjacent_gain_to_layer"] == 1
    assert close(float(linear_land["largest_adjacent_gain"]), 1.0 / 30.0)
    print(f"{CODE_NAME} SELF-TEST: PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    root = (args.project_root.resolve() if args.project_root else project_root())
    prod = root / PROD_REL
    code12_dir = root / CODE12_REL
    code13_dir = root / CODE13_REL
    out_dir = root / OUT_REL

    paths = {
        "trajectory": root / TRAJ_REL,
        "trajectory_index": root / INDEX_REL,
        "code12_manifest": root / CODE12_MANIFEST_REL,
        "seed_results": root / SEED_REL,
        "code11_manifest": root / CODE11_MANIFEST_REL,
        "subset_retention": root / RET_SUBSET_REL,
        "order_retention": root / RET_ORDER_REL,
        "contrast": root / RET_CONTRAST_REL,
        "code13_manifest": root / CODE13_MANIFEST_REL,
    }
    for label, p in paths.items():
        require_file(p, label)

    code11 = read_json(paths["code11_manifest"])
    code12 = read_json(paths["code12_manifest"])
    code13 = read_json(paths["code13_manifest"])

    validate_code11_manifest(prod, code11, paths["seed_results"])
    validate_code12_manifest(
        prod, code12, paths["code12_manifest"],
        paths["trajectory"], paths["trajectory_index"],
        paths["code11_manifest"],
    )
    tau = validate_code13_manifest(
        code13, paths["trajectory"],
        paths["subset_retention"], paths["order_retention"], paths["contrast"],
    )

    load_and_validate_index(paths["trajectory_index"])
    by_subset = load_trajectories(paths["trajectory"])
    c13_subset = load_code13_subset(paths["subset_retention"])
    c13_order = load_code13_order(paths["order_retention"])
    c13_contrast = load_code13_contrast(paths["contrast"])

    retain_by_mask = validate_against_code13(
        by_subset, c13_subset, c13_order, c13_contrast, tau
    )

    by_seed_subset = load_seed_results(paths["seed_results"])
    seed_rows, seed_by_mask = build_seed_dynamics(by_seed_subset)

    subset_summary, norm_long, norm_lookup = build_subset_outputs(
        by_subset, c13_subset, seed_by_mask, tau
    )
    raw_order, norm_order, order_summary = build_order_outputs(
        by_subset, norm_lookup, subset_summary, c13_order
    )

    print("=" * 88)
    print(f"{CODE_NAME} PREFLIGHT: PASS")
    print(f"Production root: {prod}")
    print("Frozen corrected-IID Code11/Code12/Code13 provenance: PASS")
    print("Trajectory grid: 15 subsets x 31 layers = 465 cells")
    print("Seed grid: 15 subsets x 31 layers x 3 critic seeds = 1,395 fits")
    print(f"Code13 frozen tau: {tau:.17g}")
    print("Code13 retention integrity recheck: PASS")
    print("No MINE fits executed: YES")
    print("Null subtraction: NO")
    print("Clamping: NO")
    print("Formal inference: NO")

    if args.preflight_only:
        print("Code15 outputs written: NO")
        return 0

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise ContractError(
            f"Output directory is non-empty: {out_dir}. "
            "Use --overwrite only for an intentional regeneration."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        OUT_SUBSET_SUMMARY: out_dir / OUT_SUBSET_SUMMARY,
        OUT_NORMALIZED_LONG: out_dir / OUT_NORMALIZED_LONG,
        OUT_ORDER_NORMALIZED: out_dir / OUT_ORDER_NORMALIZED,
        OUT_ORDER_RAW: out_dir / OUT_ORDER_RAW,
        OUT_SEED_DYNAMICS: out_dir / OUT_SEED_DYNAMICS,
        OUT_ORDER_SUMMARY: out_dir / OUT_ORDER_SUMMARY,
    }

    subset_fields = [
        "order_k", "subset_mask", "subset_positions", "n_layers",
        "peak_layer", "peak_score", "final_score", "retain_s",
        "post_peak_fraction_lost",
        "first_ge_50pct_peak_layer", "first_ge_90pct_peak_layer",
        "largest_adjacent_gain", "largest_adjacent_gain_from_layer", "largest_adjacent_gain_to_layer",
        "largest_adjacent_drop", "largest_adjacent_drop_from_layer", "largest_adjacent_drop_to_layer",
        "max_critic_seed_sd_across_layers", "mean_critic_seed_sd_across_layers",
        "seed_peak_layer_min", "seed_peak_layer_max", "seed_peak_layer_range", "seed_peak_layer_mode",
        "seed_peak_to_final_ratio_mean", "seed_peak_to_final_ratio_sd",
    ] + [f"score_L{l}" for l in (0, 5, 10, 15, 20, 25, 30)] \
      + [f"normalized_L{l}" for l in (0, 5, 10, 15, 20, 25, 30)]

    norm_long_fields = [
        "layer", "order_k", "subset_mask", "subset_positions",
        "score_mean", "critic_seed_sd", "peak_score", "peak_layer",
        "peak_normalized_score", "tau_q99_abs_fixed_subset_null", "retain_s",
    ]
    raw_order_fields = [
        "layer", "order_k", "n_subsets",
        "descriptive_raw_score_mean", "descriptive_raw_score_sd_across_subsets",
        "min_subset_raw_score", "max_subset_raw_score", "max_critic_seed_sd",
        "interpretation",
    ]
    norm_order_fields = [
        "layer", "order_k", "n_subsets",
        "mean_peak_normalized_score", "sd_peak_normalized_score_across_subsets",
        "min_peak_normalized_score", "max_peak_normalized_score", "interpretation",
    ]
    seed_fields = [
        "order_k", "subset_mask", "subset_positions", "critic_seed",
        "seed_peak_layer", "seed_peak_score", "seed_final_score",
        "seed_peak_to_final_ratio",
    ]
    order_summary_fields = [
        "order_k", "n_subsets",
        "order_normalized_curve_peak_layer", "order_normalized_curve_peak_value",
        "final_mean_peak_normalized_score", "code13_retain_k",
        "subset_peak_layer_min", "subset_peak_layer_max",
        "subset_peak_layer_mean", "subset_peak_layer_median",
        "largest_adjacent_gain", "largest_adjacent_gain_from_layer", "largest_adjacent_gain_to_layer",
        "largest_adjacent_drop", "largest_adjacent_drop_from_layer", "largest_adjacent_drop_to_layer",
    ]

    write_csv(out_paths[OUT_SUBSET_SUMMARY], subset_fields, subset_summary)
    write_csv(out_paths[OUT_NORMALIZED_LONG], norm_long_fields, norm_long)
    write_csv(out_paths[OUT_ORDER_RAW], raw_order_fields, raw_order)
    write_csv(out_paths[OUT_ORDER_NORMALIZED], norm_order_fields, norm_order)
    write_csv(out_paths[OUT_SEED_DYNAMICS], seed_fields, seed_rows)
    write_csv(out_paths[OUT_ORDER_SUMMARY], order_summary_fields, order_summary)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "status": "complete_pass",
        "overall_pass": True,
        "purpose": (
            "Descriptive layerwise analysis of frozen corrected-IID GB1 + ESM-2 "
            "fixed-subset neural dependence trajectories."
        ),
        "production_root": str(PROD_REL),
        "scientific_contract": {
            "primary_measurement_unit": "fixed positional subset S",
            "layers": list(LAYERS),
            "n_subsets": 15,
            "n_layer_subset_cells": 465,
            "peak_normalization": "N_l,S = D_l,S / max_j D_j,S",
            "normalization_performed_within_fixed_subset": True,
            "order_normalized_curve": "mean_S in order of N_l,S",
            "order_raw_curve_is_descriptive_only": True,
            "pooled_by_order_mi_estimation": False,
            "mine_refitting_performed": False,
            "null_scores_subtracted": False,
            "negative_scores_clamped": False,
            "formal_inference_performed": False,
            "critic_seeds_are_estimator_replicates": True,
            "largest_adjacent_gain_drop_are_descriptive_landmarks": True,
            "onset_landmarks": [
                "first layer reaching 50% of within-subset peak",
                "first layer reaching 90% of within-subset peak",
            ],
        },
        "provenance_gates": {
            "code11_manifest_pass": True,
            "code12_manifest_pass": True,
            "code13_manifest_pass": True,
            "code13_retention_rederived_exactly_within_tolerance": True,
            "code13_order_retention_matches_final_normalized_order_curves": True,
        },
        "frozen_tau": tau,
        "inputs": {
            label: {
                "path": os.path.relpath(p, root),
                "sha256": sha256_file(p),
            }
            for label, p in paths.items()
        },
        "output_counts": {
            "subset_summary_rows": len(subset_summary),
            "normalized_long_rows": len(norm_long),
            "order_raw_rows": len(raw_order),
            "order_normalized_rows": len(norm_order),
            "seed_dynamics_rows": len(seed_rows),
            "order_summary_rows": len(order_summary),
        },
        "outputs": {name: str(Path(OUT_REL) / name) for name in out_paths},
        "production_source": os.path.relpath(Path(__file__).resolve(), root)
            if Path(__file__).resolve().is_relative_to(root)
            else str(Path(__file__).resolve()),
        "production_source_sha256": sha256_file(Path(__file__).resolve()),
        "interpretation_limits": [
            "Neural dependence scores are comparative and are not calibrated exact MI in nats.",
            "Peak-normalized order curves are descriptive averages of fixed-subset normalized trajectories.",
            "The fourth-order curve contains one positional subset and has no within-order subset replication.",
            "Adjacent gain/drop landmarks are descriptive and are not significance-tested change points.",
            "Critic-seed peak behavior is an estimator-stability diagnostic, not biological replication.",
            "No causal-use claim is supported by this analysis.",
        ],
    }
    manifest["output_sha256"] = {
        name: sha256_file(path) for name, path in out_paths.items()
    }
    manifest_path = out_dir / OUT_MANIFEST
    write_json(manifest_path, manifest)

    print("\nORDER-LEVEL PEAK-NORMALIZED DYNAMICS")
    for r in order_summary:
        print(
            f"  k={r['order_k']}: "
            f"curve_peak=L{r['order_normalized_curve_peak_layer']} "
            f"(mean normalized={float(r['order_normalized_curve_peak_value']):.6f}); "
            f"final={float(r['final_mean_peak_normalized_score']):.6f}; "
            f"largest_drop=L{r['largest_adjacent_drop_from_layer']}->"
            f"L{r['largest_adjacent_drop_to_layer']} "
            f"({float(r['largest_adjacent_drop']):+.6f}); "
            f"subset_peak_range=L{r['subset_peak_layer_min']}..L{r['subset_peak_layer_max']}"
        )

    print("\nFIXED-SUBSET PEAK LAYERS AND RETENTION")
    for r in subset_summary:
        print(
            f"  S{int(r['subset_mask']):02d} {r['subset_positions']} "
            f"k={r['order_k']}: peak=L{r['peak_layer']}, "
            f"retain={float(r['retain_s']):.6f}, "
            f"seed_peak_range=L{r['seed_peak_layer_min']}..L{r['seed_peak_layer_max']}"
        )

    print(f"\n{CODE_NAME}: PASS")
    print(f"Outputs: {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
