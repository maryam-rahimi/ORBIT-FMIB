#!/usr/bin/env python3
"""
ORBIT-FMIB CODE 05 — End-to-end ORBIT-Synth FMIB validation.

Purpose
-------
Combine the frozen synthetic benchmark from Code 03 with the exact MINE/DV
estimator implementation validated in Code 04.  This is the pre-biological
integration gate: GB1/ESM-2 interpretation is not allowed unless this script
passes.

For each synthetic scenario, layer l, and interaction order k, Code 05:
  1. recovers representation-level Walsh coefficients R_{l,C,S} directly from
     the saved pointwise hidden representations H_l(x);
  2. pools cube/subset samples by order while preserving the frozen
     context-level train/validation/test partition;
  3. trains the exact Code-04 MINE critic on train samples, selects/stops on
     validation only, and reports MI on held-out test samples only;
  4. compares estimated I(R_{l,k}; E_k) against the known analytic Gaussian
     channel target;
  5. tests null/off-target behavior, bottleneck localization, seed stability,
     and the selective higher-order retention contrast.

Inputs
------
04 Experiments/EXP01_ORBIT_Synth/
    orbit_synth_manifest.json
    orbit_synth_controls.npz
    fmib_targets.npz
    fmib_staggered_bottleneck.npz
    fmib_selective_ho_bottleneck.npz

04 Experiments/EXP03_MI/
    mine_validation_manifest.json

03 Code/
    04_validate_mi_estimator.py

Outputs
-------
04 Experiments/EXP01_ORBIT_Synth/
    fmib_code05_seed_results.csv
    fmib_code05_layer_order_summary.csv
    fmib_code05_validation_manifest.json
    fmib_code05_run_state.json

Usage
-----
    python "03 Code\\05_run_orbit_synth_fmib.py" --self-test
    python "03 Code\\05_run_orbit_synth_fmib.py"
    python "03 Code\\05_run_orbit_synth_fmib.py" --resume
    python "03 Code\\05_run_orbit_synth_fmib.py" --device cuda

Notes
-----
* The upstream synthetic tensors are never regenerated here.
* The Code-04 estimator is imported from the actual Code-04 source file rather
  than reimplemented, so calibration and integration use one estimator.
* MINE estimates remain finite-sample variational lower bounds.  The scientific
  gate is based on trajectory/order recovery under a fixed estimator protocol,
  not on treating the estimates as exact physical mutual information.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


CODE_NAME = "ORBIT-FMIB CODE 05"
SCHEMA_VERSION = 1
EXPECTED_CODE03 = "ORBIT-FMIB CODE 03"
EXPECTED_CODE04 = "ORBIT-FMIB CODE 04"

# -----------------------------------------------------------------------------
# Frozen Code-05 acceptance thresholds.  These are intentionally fixed before
# inspecting any GB1/ESM-2 result.
# -----------------------------------------------------------------------------
WALSH_RECON_MAX_ABS = 5e-5
STRONG_TRUTH_MIN = 0.30
STRONG_ESTIMATE_MIN = 0.35
STRONG_MAX_ABS_ERROR = 0.15
ATTENUATED_TRUTH_MAX = 0.05
ATTENUATED_ESTIMATE_ABS_MAX = 0.10
NULL_ESTIMATE_ABS_MAX = 0.08
MAX_SEED_SD = 0.08
DROP_DELTA_THRESHOLD = 0.20
ORDER_SIGNAL_MINUS_NULL_MIN = 0.25
SELECTIVE_LO_RETENTION_MIN = 0.70
SELECTIVE_HO_RETENTION_MAX = 0.30
SELECTIVE_DELTA_MAX = -0.50

SCENARIOS = {
    "staggered_bottleneck": "fmib_staggered_bottleneck.npz",
    "selective_ho_bottleneck": "fmib_selective_ho_bottleneck.npz",
}

SEED_RESULT_COLUMNS = [
    "family",
    "scenario",
    "layer",
    "order",
    "critic_seed",
    "n_samples",
    "n_train",
    "n_val",
    "n_test",
    "analytic_mi_nats",
    "best_step",
    "final_training_step",
    "best_val_mi",
    "test_mi",
    "test_product_shuffle_sd",
    "heldout_joint_minus_mismatch_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic internal tests and exit; no neural critics are trained.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a provenance-matched interrupted full run from the saved seed-results CSV.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Torch device passed to the exact Code-04 estimator. Default: cpu.",
    )
    parser.add_argument(
        "--code03-dir",
        type=Path,
        default=None,
        help="Override Code-03 synthetic artifact directory.",
    )
    parser.add_argument(
        "--code04-manifest",
        type=Path,
        default=None,
        help="Override Code-04 mine_validation_manifest.json path.",
    )
    parser.add_argument(
        "--code04-source",
        type=Path,
        default=None,
        help="Override Code-04 Python source path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory. Default: Code-03 synthetic directory.",
    )
    return parser.parse_args()


def project_root_from_script() -> Path:
    # Expected location: <project>/03 Code/05_run_orbit_synth_fmib.py
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Mapping) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def import_code04(source_path: Path):
    """Import the exact Code-04 source file by path; do not duplicate estimator logic."""
    if not source_path.exists():
        raise FileNotFoundError(
            f"Code-04 source not found: {source_path}\n"
            "Code 05 must reuse the exact estimator implementation that Code 04 calibrated."
        )
    module_name = "_orbitfmib_code04_exact"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not construct import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if getattr(module, "CODE_NAME", None) != EXPECTED_CODE04:
        raise RuntimeError("Imported source is not ORBIT-FMIB CODE 04.")
    required = (
        "import_torch",
        "choose_device",
        "train_mine",
        "CRITIC_SEEDS",
        "TrainingConfig",
        "TEST_PRODUCT_SHIFTS",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Code-04 source missing required estimator objects: {missing}")
    return module


def verify_file_hashes(base_dir: Path, recorded: Mapping[str, str], label: str) -> Dict[str, str]:
    observed: Dict[str, str] = {}
    for name, expected in recorded.items():
        path = base_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{label} artifact missing: {path}")
        got = sha256_file(path)
        observed[name] = got
        if got.lower() != str(expected).lower():
            raise RuntimeError(
                f"{label} SHA256 mismatch for {name}.\n"
                f"Expected: {expected}\nObserved: {got}\n"
                "Stop: upstream frozen artifacts have changed."
            )
    return observed


def verify_code04_contract(code04_manifest: Dict, code03_manifest_sha: str, mine04) -> None:
    if code04_manifest.get("code_name") != EXPECTED_CODE04:
        raise RuntimeError("Unexpected Code-04 manifest code_name.")
    if not bool(code04_manifest.get("overall_pass", False)):
        raise RuntimeError("Code-04 manifest does not record overall_pass=true.")
    acceptance = code04_manifest.get("acceptance_results", {})
    if not bool(acceptance.get("overall_pass", False)):
        raise RuntimeError("Code-04 acceptance_results.overall_pass is not true.")

    ch = code04_manifest.get("code03_vector_calibration", {})
    pinned = str(ch.get("manifest_sha256", ""))
    if pinned.lower() != code03_manifest_sha.lower():
        raise RuntimeError(
            "Code-04 provenance mismatch: its pinned Code-03 manifest SHA256 does not "
            "match the current Code-03 manifest."
        )

    manifest_seeds = tuple(int(x) for x in code04_manifest.get("critic_seeds", []))
    source_seeds = tuple(int(x) for x in mine04.CRITIC_SEEDS)
    if manifest_seeds != source_seeds:
        raise RuntimeError(
            f"Code-04 critic seed mismatch between manifest {manifest_seeds} "
            f"and source {source_seeds}."
        )

    manifest_cfg = code04_manifest.get("training_config", {})
    source_cfg = asdict(mine04.TrainingConfig())
    if manifest_cfg != source_cfg:
        raise RuntimeError(
            "Code-04 training configuration in the manifest does not exactly match "
            "the imported Code-04 source."
        )


def enumerate_points(d: int) -> np.ndarray:
    idx = np.arange(1 << d, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(d, dtype=np.int64)) & 1
    return np.where(bits == 1, 1.0, -1.0).astype(np.float64)


def chi_matrix(d: int) -> np.ndarray:
    points = enumerate_points(d)
    n = 1 << d
    X = np.ones((n, n), dtype=np.float64)
    for subset in range(n):
        positions = [b for b in range(d) if (subset >> b) & 1]
        if positions:
            X[:, subset] = np.prod(points[:, positions], axis=1)
    return X


def order_of(subset: int) -> int:
    return int(subset).bit_count()


def subsets_for_order(subset_orders: np.ndarray, k: int) -> np.ndarray:
    idx = np.flatnonzero(subset_orders == int(k)).astype(np.int64)
    if len(idx) == 0:
        raise RuntimeError(f"No Walsh subsets found for order {k}.")
    return idx


def recover_representation_coefficients(H_layer: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Recover R[C,S,m] = 2^-d sum_x H[C,x,m] chi_S(x)."""
    H_layer = np.asarray(H_layer)
    if H_layer.ndim != 3:
        raise ValueError("H_layer must have shape (contexts, vertices, hidden_dim).")
    n_vertices = H_layer.shape[1]
    if X.shape != (n_vertices, n_vertices):
        raise ValueError("Walsh matrix shape does not match H_layer vertices.")
    R = np.einsum(
        "xs,cxm->csm", X, H_layer.astype(np.float64), optimize=True
    ) / float(n_vertices)
    if not np.all(np.isfinite(R)):
        raise RuntimeError("Recovered representation coefficients contain non-finite values.")
    return R


def reconstruct_hidden_from_coefficients(R: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.einsum("xs,csm->cxm", X, R, optimize=True)


def make_order_samples(
    R: np.ndarray,
    E: np.ndarray,
    subset_orders: np.ndarray,
    split_codes: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pool (context, subset) observations for one order without crossing splits."""
    subset_idx = subsets_for_order(subset_orders, k)
    u = R[:, subset_idx, :].reshape(-1, R.shape[-1]).astype(np.float32, copy=False)
    v = E[:, subset_idx].reshape(-1, 1).astype(np.float32, copy=False)
    row_split = np.repeat(split_codes.astype(np.int8), len(subset_idx))

    train_idx = np.flatnonzero(row_split == 0).astype(np.int64)
    val_idx = np.flatnonzero(row_split == 1).astype(np.int64)
    test_idx = np.flatnonzero(row_split == 2).astype(np.int64)

    if len(u) != len(v) or len(u) != len(row_split):
        raise RuntimeError("Order-sample flattening produced inconsistent lengths.")
    if len(np.intersect1d(train_idx, val_idx)) or len(np.intersect1d(train_idx, test_idx)) or len(np.intersect1d(val_idx, test_idx)):
        raise RuntimeError("Order-sample train/val/test row overlap detected.")
    if len(np.unique(np.concatenate([train_idx, val_idx, test_idx]))) != len(u):
        raise RuntimeError("Order-sample split does not cover each row exactly once.")

    # Context-level leakage gate: all subset rows from a context inherit exactly
    # that context's frozen Code-03 split.
    n_subsets = len(subset_idx)
    for c, split_code in enumerate(split_codes):
        lo = c * n_subsets
        hi = lo + n_subsets
        if not np.all(row_split[lo:hi] == split_code):
            raise RuntimeError("A synthetic context was split across estimator partitions.")

    return u, v, (train_idx, val_idx, test_idx)


def make_within_split_null_target(
    v: np.ndarray,
    split_indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Create an independence/null target by a non-self cyclic shift within each split."""
    out = np.empty_like(v)
    for idx in split_indices:
        if len(idx) < 2:
            raise RuntimeError("Need at least two rows per split for null construction.")
        out[idx] = v[np.roll(idx, 1)]
    return out


def locate_drop(mi_by_layer: Sequence[float], threshold: float = DROP_DELTA_THRESHOLD) -> int | None:
    values = np.asarray(mi_by_layer, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("mi_by_layer must be a finite one-dimensional sequence of length >= 2.")
    delta = np.diff(values)
    j = int(np.argmin(delta))
    return j + 1 if float(delta[j]) <= -float(threshold) else None


def retention_from_trajectory(mi_by_layer: Sequence[float]) -> float:
    values = np.asarray(mi_by_layer, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Retention trajectory must be finite and one-dimensional.")
    peak = float(np.max(values))
    if peak <= 0.0:
        raise RuntimeError("Retention denominator is non-positive; estimator trajectory is unusable.")
    return float(values[-1] / peak)


def audit_structural_controls(controls_path: Path, X: np.ndarray, d: int) -> Dict[str, float]:
    max_coeff_error = 0.0
    max_absent_energy = 0.0
    with np.load(controls_path, allow_pickle=False) as z:
        for k in range(1, d + 1):
            coeff_key = f"pure_order_{k}__coefficients"
            fitness_key = f"pure_order_{k}__fitness"
            if coeff_key not in z or fitness_key not in z:
                raise RuntimeError(f"Missing structural control arrays for order {k}.")
            eps = np.asarray(z[coeff_key], dtype=np.float64)
            fitness = np.asarray(z[fitness_key], dtype=np.float64)
            recovered = (X.T @ fitness) / float(1 << d)
            max_coeff_error = max(max_coeff_error, float(np.max(np.abs(recovered - eps))))

            energy_by_order = {}
            for j in range(1, d + 1):
                sidx = [s for s in range(1 << d) if order_of(s) == j]
                energy_by_order[j] = float(np.sum(recovered[sidx] ** 2))
            total = sum(energy_by_order.values())
            if total <= 0.0:
                raise RuntimeError("Pure-order structural control has zero total nonconstant energy.")
            frac = {j: energy_by_order[j] / total for j in energy_by_order}
            if frac[k] < 1.0 - 1e-10:
                raise RuntimeError(f"Pure-order-{k} control failed order isolation.")
            for j in range(1, d + 1):
                if j != k:
                    max_absent_energy = max(max_absent_energy, frac[j])

        if "parity__coefficients" not in z or "parity__fitness" not in z:
            raise RuntimeError("Parity structural control is missing.")
        parity_eps = np.asarray(z["parity__coefficients"], dtype=np.float64)
        parity_fit = np.asarray(z["parity__fitness"], dtype=np.float64)
        parity_rec = (X.T @ parity_fit) / float(1 << d)
        max_coeff_error = max(max_coeff_error, float(np.max(np.abs(parity_rec - parity_eps))))
        if abs(float(parity_rec[(1 << d) - 1]) - 1.0) > 1e-10:
            raise RuntimeError("Exact parity control failed four-way Walsh recovery.")

    if max_coeff_error > 1e-10 or max_absent_energy > 1e-10:
        raise RuntimeError("Structural Walsh/order controls failed Code-05 re-audit.")
    return {
        "max_coefficient_abs_error": max_coeff_error,
        "max_absent_order_energy_fraction": max_absent_energy,
    }


def summarize_seed_results(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    group_cols = ["family", "scenario", "layer", "order"]
    for keys, group in seed_df.groupby(group_cols, sort=True, dropna=False):
        family, scenario, layer, order = keys
        estimates = group["test_mi"].to_numpy(dtype=float)
        truths = group["analytic_mi_nats"].to_numpy(dtype=float)
        true_mi = float(truths[0])
        if not np.allclose(truths, true_mi, atol=1e-12, rtol=0.0):
            raise RuntimeError("Analytic MI differs across critic seeds for one condition.")
        rows.append(
            {
                "family": family,
                "scenario": scenario,
                "layer": int(layer),
                "order": int(order),
                "analytic_mi_nats": true_mi,
                "estimate_mean": float(np.mean(estimates)),
                "estimate_sd_across_seeds": float(np.std(estimates, ddof=0)),
                "estimate_min": float(np.min(estimates)),
                "estimate_max": float(np.max(estimates)),
                "abs_error_of_mean": float(abs(np.mean(estimates) - true_mi)),
                "n_critic_seeds": int(len(estimates)),
                "n_samples": int(group["n_samples"].iloc[0]),
                "n_train": int(group["n_train"].iloc[0]),
                "n_val": int(group["n_val"].iloc[0]),
                "n_test": int(group["n_test"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_acceptance_results(
    summary_df: pd.DataFrame,
    code03_manifest: Dict,
) -> Tuple[Dict, Dict[str, Dict[int, int | None]], Dict[str, float]]:
    traj = summary_df[summary_df["family"] == "trajectory"].copy()
    nulls = summary_df[summary_df["family"] == "null_control"].copy()
    if traj.empty or nulls.empty:
        raise RuntimeError("Missing trajectory or null-control summaries.")

    strong = traj[traj["analytic_mi_nats"] >= STRONG_TRUTH_MIN]
    atten = traj[traj["analytic_mi_nats"] <= ATTENUATED_TRUTH_MAX]
    if strong.empty or atten.empty:
        raise RuntimeError("Could not identify strong and attenuated synthetic conditions.")

    max_seed_sd = float(summary_df["estimate_sd_across_seeds"].max())
    strong_max_error = float(strong["abs_error_of_mean"].max())
    strong_min_est = float(strong["estimate_mean"].min())
    atten_max_abs_est = float(np.max(np.abs(atten["estimate_mean"].to_numpy(dtype=float))))
    null_max_abs_est = float(np.max(np.abs(nulls["estimate_mean"].to_numpy(dtype=float))))

    observed_drop: Dict[str, Dict[int, int | None]] = {}
    expected_drop: Dict[str, Dict[int, int | None]] = {}
    for scenario in SCENARIOS:
        observed_drop[scenario] = {}
        expected_drop[scenario] = {}
        raw_expected = code03_manifest["scenarios"][scenario]["drop_stage_by_order"]
        for k in range(1, 5):
            rows = traj[(traj["scenario"] == scenario) & (traj["order"] == k)].sort_values("layer")
            values = rows["estimate_mean"].to_numpy(dtype=float)
            observed_drop[scenario][k] = locate_drop(values)
            exp = raw_expected[str(k)]
            expected_drop[scenario][k] = None if exp is None else int(exp)

    localization_pass = observed_drop == expected_drop

    # Signal-vs-null order specificity at layer 0 of the selective-HO scenario.
    signal_minus_null: Dict[int, float] = {}
    for k in range(1, 5):
        sig_row = traj[
            (traj["scenario"] == "selective_ho_bottleneck")
            & (traj["layer"] == 0)
            & (traj["order"] == k)
        ]
        null_row = nulls[(nulls["layer"] == 0) & (nulls["order"] == k)]
        if len(sig_row) != 1 or len(null_row) != 1:
            raise RuntimeError("Expected one signal and one null summary per order at layer 0.")
        signal_minus_null[k] = float(sig_row.iloc[0]["estimate_mean"] - null_row.iloc[0]["estimate_mean"])
    order_specificity_pass = all(v >= ORDER_SIGNAL_MINUS_NULL_MIN for v in signal_minus_null.values())

    selective_retain: Dict[int, float] = {}
    for k in range(1, 5):
        rows = traj[
            (traj["scenario"] == "selective_ho_bottleneck") & (traj["order"] == k)
        ].sort_values("layer")
        selective_retain[k] = retention_from_trajectory(rows["estimate_mean"].to_numpy(dtype=float))

    lo = float(np.mean([selective_retain[1], selective_retain[2]]))
    ho = float(np.mean([selective_retain[3], selective_retain[4]]))
    delta = float(ho - lo)
    retention_stats = {
        "retain_1": selective_retain[1],
        "retain_2": selective_retain[2],
        "retain_3": selective_retain[3],
        "retain_4": selective_retain[4],
        "lo_retention": lo,
        "ho_retention": ho,
        "delta_ho_lo": delta,
    }

    results = {
        "all_estimates_finite": bool(np.all(np.isfinite(summary_df.select_dtypes(include=[np.number]).to_numpy()))),
        "seed_stability": bool(max_seed_sd <= MAX_SEED_SD),
        "max_seed_sd": max_seed_sd,
        "strong_channel_min_estimate_gate": bool(strong_min_est >= STRONG_ESTIMATE_MIN),
        "strong_channel_min_estimate": strong_min_est,
        "strong_channel_error_gate": bool(strong_max_error <= STRONG_MAX_ABS_ERROR),
        "strong_channel_max_abs_error": strong_max_error,
        "attenuated_near_floor_gate": bool(atten_max_abs_est <= ATTENUATED_ESTIMATE_ABS_MAX),
        "attenuated_max_abs_estimate": atten_max_abs_est,
        "null_near_zero_gate": bool(null_max_abs_est <= NULL_ESTIMATE_ABS_MAX),
        "null_max_abs_estimate": null_max_abs_est,
        "order_specificity_gate": bool(order_specificity_pass),
        "signal_minus_null_by_order": {str(k): v for k, v in signal_minus_null.items()},
        "bottleneck_localization_gate": bool(localization_pass),
        "selective_lo_retention_gate": bool(lo >= SELECTIVE_LO_RETENTION_MIN),
        "selective_ho_retention_gate": bool(ho <= SELECTIVE_HO_RETENTION_MAX),
        "selective_delta_gate": bool(delta <= SELECTIVE_DELTA_MAX),
    }
    boolean_gates = [
        results["all_estimates_finite"],
        results["seed_stability"],
        results["strong_channel_min_estimate_gate"],
        results["strong_channel_error_gate"],
        results["attenuated_near_floor_gate"],
        results["null_near_zero_gate"],
        results["order_specificity_gate"],
        results["bottleneck_localization_gate"],
        results["selective_lo_retention_gate"],
        results["selective_ho_retention_gate"],
        results["selective_delta_gate"],
    ]
    results["overall_pass"] = bool(all(boolean_gates))
    return results, observed_drop, retention_stats


def expected_run_fingerprint(
    code03_manifest_sha: str,
    code04_manifest_sha: str,
    code04_source_sha: str,
    device: str,
    critic_seeds: Sequence[int],
) -> Dict:
    return {
        "code03_manifest_sha256": code03_manifest_sha,
        "code04_manifest_sha256": code04_manifest_sha,
        "code04_source_sha256": code04_source_sha,
        "device": str(device),
        "critic_seeds": [int(x) for x in critic_seeds],
        "schema_version": SCHEMA_VERSION,
    }


def load_or_initialize_resume(
    seed_path: Path,
    state_path: Path,
    fingerprint: Dict,
    resume: bool,
) -> pd.DataFrame:
    if resume:
        if not seed_path.exists() or not state_path.exists():
            raise RuntimeError("--resume requested but seed-results CSV or run-state JSON is missing.")
        state = load_json(state_path)
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "Cannot resume: upstream artifacts, estimator source/configuration, device, or seeds changed."
            )
        df = pd.read_csv(seed_path)
        missing = [c for c in SEED_RESULT_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(f"Resume CSV missing required columns: {missing}")
        return df[SEED_RESULT_COLUMNS].copy()

    # Fresh full run: overwrite stale partial outputs deliberately.
    empty = pd.DataFrame(columns=SEED_RESULT_COLUMNS)
    empty.to_csv(seed_path, index=False)
    write_json(state_path, {"code_name": CODE_NAME, "fingerprint": fingerprint, "status": "running"})
    return empty


def row_already_done(df: pd.DataFrame, family: str, scenario: str, layer: int, order: int, seed: int) -> bool:
    if df.empty:
        return False
    mask = (
        (df["family"].astype(str) == family)
        & (df["scenario"].astype(str) == scenario)
        & (df["layer"].astype(int) == int(layer))
        & (df["order"].astype(int) == int(order))
        & (df["critic_seed"].astype(int) == int(seed))
    )
    return bool(mask.any())


def append_result(df: pd.DataFrame, row: Dict, seed_path: Path) -> pd.DataFrame:
    new = pd.DataFrame([{c: row[c] for c in SEED_RESULT_COLUMNS}])
    out = pd.concat([df, new], ignore_index=True)
    out.to_csv(seed_path, index=False)
    return out


def self_test() -> None:
    d = 4
    X = chi_matrix(d)
    if not np.allclose(X.T @ X, (1 << d) * np.eye(1 << d), atol=1e-12):
        raise RuntimeError("SELF-TEST failed: Walsh basis orthogonality.")

    rng = np.random.default_rng(123)
    R_true = rng.normal(size=(7, 1 << d, 5))
    H = reconstruct_hidden_from_coefficients(R_true, X)
    R_rec = recover_representation_coefficients(H, X)
    if not np.allclose(R_true, R_rec, atol=1e-12, rtol=0.0):
        raise RuntimeError("SELF-TEST failed: representation Walsh round trip.")

    E = rng.normal(size=(7, 1 << d)).astype(np.float32)
    subset_orders = np.array([order_of(s) for s in range(1 << d)], dtype=np.int8)
    split_codes = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int8)
    u, v, splits = make_order_samples(R_rec, E, subset_orders, split_codes, 2)
    if u.shape != (7 * math.comb(4, 2), 5) or v.shape != (7 * math.comb(4, 2), 1):
        raise RuntimeError("SELF-TEST failed: order-sample shape.")
    null_v = make_within_split_null_target(v, splits)
    if null_v.shape != v.shape:
        raise RuntimeError("SELF-TEST failed: null target shape.")
    for idx in splits:
        if np.any(np.all(null_v[idx] == v[idx], axis=1)):
            # E is continuous random data, so exact equality would indicate a self-pair.
            raise RuntimeError("SELF-TEST failed: null target contains self-pairs.")

    if locate_drop([0.50, 0.51, 0.49, 0.10, 0.09]) != 3:
        raise RuntimeError("SELF-TEST failed: bottleneck localization.")
    if locate_drop([0.50, 0.51, 0.49, 0.50], threshold=0.20) is not None:
        raise RuntimeError("SELF-TEST failed: no-drop behavior.")
    retain = retention_from_trajectory([0.20, 0.50, 0.40, 0.25])
    if not np.isclose(retain, 0.5):
        raise RuntimeError("SELF-TEST failed: retention calculation.")

    print("SELF-TEST: PASS")
    print("Verified: Walsh round trip, context-preserving order samples, within-split nulls, drop localization, retention.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    root = project_root_from_script()
    code03_dir = args.code03_dir or (root / "04 Experiments" / "EXP01_ORBIT_Synth")
    output_dir = args.output_dir or code03_dir
    code03_manifest_path = code03_dir / "orbit_synth_manifest.json"
    code04_manifest_path = args.code04_manifest or (root / "04 Experiments" / "EXP03_MI" / "mine_validation_manifest.json")
    code04_source_path = args.code04_source or (root / "03 Code" / "04_validate_mi_estimator.py")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"{CODE_NAME} — END-TO-END ORBIT-SYNTH FMIB VALIDATION")
    print("=" * 76)
    print()

    # -------------------------------------------------------------------------
    # 1. Upstream provenance gates
    # -------------------------------------------------------------------------
    code03_manifest = load_json(code03_manifest_path)
    if code03_manifest.get("code_name") != EXPECTED_CODE03:
        raise RuntimeError("Unexpected Code-03 manifest code_name.")
    code03_manifest_sha = sha256_file(code03_manifest_path)
    verify_file_hashes(code03_dir, code03_manifest.get("file_sha256", {}), "Code-03")

    mine04 = import_code04(code04_source_path)
    code04_manifest = load_json(code04_manifest_path)
    code04_manifest_sha = sha256_file(code04_manifest_path)
    code04_source_sha = sha256_file(code04_source_path)
    verify_code04_contract(code04_manifest, code03_manifest_sha, mine04)
    verify_file_hashes(code04_manifest_path.parent, code04_manifest.get("file_sha256", {}), "Code-04")

    print("Upstream provenance gate: PASS")
    print(f"  Code-03 manifest SHA256: {code03_manifest_sha}")
    print(f"  Code-04 manifest pins same Code-03 artifact: PASS")
    print(f"  Code-04 calibration overall_pass: PASS")
    print(f"  exact Code-04 estimator source imported: PASS")
    print()

    # -------------------------------------------------------------------------
    # 2. Frozen artifact and Walsh audits
    # -------------------------------------------------------------------------
    dims = code03_manifest.get("dimensions", {})
    d = int(dims.get("d", -1))
    n_vertices = int(dims.get("vertices_per_cube", -1))
    n_stages = int(dims.get("stages", -1))
    hidden_dim = int(dims.get("hidden_dim", -1))
    n_contexts = int(dims.get("contexts", -1))
    if (d, n_vertices, n_stages, hidden_dim, n_contexts) != (4, 16, 8, 16, 4096):
        raise RuntimeError(
            "Code-03 dimensions differ from the frozen Phase-1 synthetic contract: "
            f"got d={d}, vertices={n_vertices}, stages={n_stages}, hidden={hidden_dim}, contexts={n_contexts}."
        )

    targets_path = code03_dir / "fmib_targets.npz"
    controls_path = code03_dir / "orbit_synth_controls.npz"
    X = chi_matrix(d)
    structural = audit_structural_controls(controls_path, X, d)

    with np.load(targets_path, allow_pickle=False) as z:
        required = {"E", "subset_orders", "split_codes", "order_directions", "walsh_matrix"}
        missing = required.difference(z.files)
        if missing:
            raise RuntimeError(f"fmib_targets.npz missing arrays: {sorted(missing)}")
        E = np.asarray(z["E"], dtype=np.float32)
        subset_orders = np.asarray(z["subset_orders"], dtype=np.int8)
        split_codes = np.asarray(z["split_codes"], dtype=np.int8)
        stored_X = np.asarray(z["walsh_matrix"], dtype=np.float64)

    if E.shape != (n_contexts, n_vertices):
        raise RuntimeError(f"Unexpected E shape: {E.shape}")
    if subset_orders.shape != (n_vertices,) or split_codes.shape != (n_contexts,):
        raise RuntimeError("Unexpected subset_orders or split_codes shape.")
    if not np.array_equal(subset_orders, np.array([order_of(s) for s in range(n_vertices)], dtype=np.int8)):
        raise RuntimeError("Stored subset-order map does not match Walsh bitmask orders.")
    if not np.allclose(stored_X, X, atol=0.0, rtol=0.0):
        raise RuntimeError("Stored Code-03 Walsh matrix differs from the Code-05 reconstruction.")
    split_counts = {int(c): int(np.sum(split_codes == c)) for c in (0, 1, 2)}
    expected_splits = code03_manifest.get("splits", {})
    if split_counts != {
        0: int(expected_splits.get("train", -1)),
        1: int(expected_splits.get("val", -1)),
        2: int(expected_splits.get("test", -1)),
    }:
        raise RuntimeError(f"Context split counts do not match Code-03 manifest: {split_counts}")

    print("Synthetic artifact / Walsh gate: PASS")
    print(f"  contexts: {n_contexts:,}")
    print(f"  train / val / test contexts: {split_counts[0]:,} / {split_counts[1]:,} / {split_counts[2]:,}")
    print("  pure orders 1-4 and parity re-audited: PASS")
    print(f"  max structural Walsh coefficient error: {structural['max_coefficient_abs_error']:.3e}")
    print()

    # -------------------------------------------------------------------------
    # 3. Exact Code-04 estimator setup
    # -------------------------------------------------------------------------
    torch, nn = mine04.import_torch()
    device = mine04.choose_device(torch, args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    critic_seeds = tuple(int(x) for x in mine04.CRITIC_SEEDS)

    print("Estimator integration contract: PASS")
    print(f"  device: {device}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  critic seeds: {', '.join(map(str, critic_seeds))}")
    print("  training/selection/evaluation: exact Code-04 train_mine implementation")
    print("  split unit: synthetic context; all subsets from a context stay together")
    print()

    seed_path = output_dir / "fmib_code05_seed_results.csv"
    summary_path = output_dir / "fmib_code05_layer_order_summary.csv"
    manifest_path = output_dir / "fmib_code05_validation_manifest.json"
    state_path = output_dir / "fmib_code05_run_state.json"

    fingerprint = expected_run_fingerprint(
        code03_manifest_sha,
        code04_manifest_sha,
        code04_source_sha,
        str(device),
        critic_seeds,
    )
    seed_df = load_or_initialize_resume(seed_path, state_path, fingerprint, args.resume)
    if args.resume:
        print(f"Resume mode: loaded {len(seed_df):,} completed critic rows")
        print()

    # -------------------------------------------------------------------------
    # 4. Full layer x order trajectories for both Code-03 scenarios
    # -------------------------------------------------------------------------
    max_walsh_recon_error = 0.0
    condition_counter = 0
    total_conditions = len(SCENARIOS) * n_stages * d

    for scenario, filename in SCENARIOS.items():
        scenario_path = code03_dir / filename
        with np.load(scenario_path, allow_pickle=False) as z:
            required = {"H", "gains", "analytic_mi_nats"}
            missing = required.difference(z.files)
            if missing:
                raise RuntimeError(f"{filename} missing arrays: {sorted(missing)}")
            H = np.asarray(z["H"])
            gains = np.asarray(z["gains"], dtype=np.float64)
            analytic_mi = np.asarray(z["analytic_mi_nats"], dtype=np.float64)

        expected_shape = (n_contexts, n_stages, n_vertices, hidden_dim)
        if H.shape != expected_shape or not np.all(np.isfinite(H)):
            raise RuntimeError(f"{scenario}: invalid H tensor shape/values: {H.shape}")
        manifest_scenario = code03_manifest["scenarios"][scenario]
        if not np.allclose(gains, np.asarray(manifest_scenario["gains"], dtype=np.float64), atol=1e-7, rtol=0.0):
            raise RuntimeError(f"{scenario}: stored gains differ from manifest.")
        if not np.allclose(analytic_mi, np.asarray(manifest_scenario["analytic_mi_nats"], dtype=np.float64), atol=1e-12, rtol=0.0):
            raise RuntimeError(f"{scenario}: analytic MI differs from manifest.")

        print(f"Running scenario: {scenario}")
        for layer in range(n_stages):
            R = recover_representation_coefficients(H[:, layer, :, :], X)
            # Reconstruct a deterministic context subset to verify the actual saved H -> R -> H path.
            audit_contexts = np.array([0, 1, 7, 31, 127, 511, 1023, 2047, 4095], dtype=np.int64)
            H_rec = reconstruct_hidden_from_coefficients(R[audit_contexts], X)
            err = float(np.max(np.abs(H_rec - H[audit_contexts, layer].astype(np.float64))))
            max_walsh_recon_error = max(max_walsh_recon_error, err)
            if err > WALSH_RECON_MAX_ABS:
                raise RuntimeError(
                    f"{scenario} layer {layer}: H->R->H Walsh reconstruction error {err:.3e} "
                    f"exceeds {WALSH_RECON_MAX_ABS:.3e}."
                )

            for k in range(1, d + 1):
                condition_counter += 1
                u, v, split_indices = make_order_samples(R, E, subset_orders, split_codes, k)
                truth = float(analytic_mi[layer, k - 1])
                n_train, n_val, n_test = map(len, split_indices)
                print(
                    f"  [{condition_counter:02d}/{total_conditions}] layer={layer}, order={k}, "
                    f"n={len(u):,}, true MI={truth:.6f}"
                )
                for seed in critic_seeds:
                    if row_already_done(seed_df, "trajectory", scenario, layer, k, seed):
                        print(f"      seed {seed}: already complete (resume)")
                        continue
                    result = mine04.train_mine(torch, nn, u, v, seed, device, split_indices)
                    row = {
                        "family": "trajectory",
                        "scenario": scenario,
                        "layer": layer,
                        "order": k,
                        "critic_seed": seed,
                        "n_samples": len(u),
                        "n_train": n_train,
                        "n_val": n_val,
                        "n_test": n_test,
                        "analytic_mi_nats": truth,
                        **result,
                    }
                    seed_df = append_result(seed_df, row, seed_path)
                    print(f"      seed {seed}: test MI={float(result['test_mi']):.6f}")
        print()

    # -------------------------------------------------------------------------
    # 5. Null/off-target controls on strong layer 0
    # -------------------------------------------------------------------------
    # Use the selective-HO layer-0 representation (all orders strong) and pair
    # each R channel with a target cyclically shifted *within each frozen split*.
    null_source_path = code03_dir / SCENARIOS["selective_ho_bottleneck"]
    with np.load(null_source_path, allow_pickle=False) as z:
        H0 = np.asarray(z["H"][:, 0, :, :])
    R0 = recover_representation_coefficients(H0, X)

    print("Running held-out null/off-target controls ...")
    for k in range(1, d + 1):
        u, v, split_indices = make_order_samples(R0, E, subset_orders, split_codes, k)
        v_null = make_within_split_null_target(v, split_indices)
        n_train, n_val, n_test = map(len, split_indices)
        for seed in critic_seeds:
            if row_already_done(seed_df, "null_control", "within_split_shift", 0, k, seed):
                print(f"  order {k}, seed {seed}: already complete (resume)")
                continue
            result = mine04.train_mine(torch, nn, u, v_null, seed, device, split_indices)
            row = {
                "family": "null_control",
                "scenario": "within_split_shift",
                "layer": 0,
                "order": k,
                "critic_seed": seed,
                "n_samples": len(u),
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "analytic_mi_nats": 0.0,
                **result,
            }
            seed_df = append_result(seed_df, row, seed_path)
            print(f"  order {k}, seed {seed}: null test MI={float(result['test_mi']):.6f}")
    print()

    # -------------------------------------------------------------------------
    # 6. Summaries and acceptance gates
    # -------------------------------------------------------------------------
    expected_rows = (len(SCENARIOS) * n_stages * d + d) * len(critic_seeds)
    if len(seed_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} completed critic rows, found {len(seed_df)}. "
            "Run was incomplete or resume state is inconsistent."
        )
    if seed_df.duplicated(["family", "scenario", "layer", "order", "critic_seed"]).any():
        raise RuntimeError("Duplicate critic-condition rows detected.")

    summary_df = summarize_seed_results(seed_df)
    summary_df.to_csv(summary_path, index=False)
    gates, observed_drop, retention_stats = build_acceptance_results(summary_df, code03_manifest)

    print("Code-05 acceptance gates")
    print(f"  H->R->H Walsh reconstruction <= {WALSH_RECON_MAX_ABS:.1e}: PASS (observed {max_walsh_recon_error:.3e})")
    print(f"  max across-seed SD <= {MAX_SEED_SD:.2f}: {'PASS' if gates['seed_stability'] else 'FAIL'} (observed {gates['max_seed_sd']:.6f})")
    print(f"  strong-channel minimum estimate >= {STRONG_ESTIMATE_MIN:.2f}: {'PASS' if gates['strong_channel_min_estimate_gate'] else 'FAIL'} (observed {gates['strong_channel_min_estimate']:.6f})")
    print(f"  strong-channel max |error| <= {STRONG_MAX_ABS_ERROR:.2f}: {'PASS' if gates['strong_channel_error_gate'] else 'FAIL'} (observed {gates['strong_channel_max_abs_error']:.6f})")
    print(f"  attenuated-channel |MI| <= {ATTENUATED_ESTIMATE_ABS_MAX:.2f}: {'PASS' if gates['attenuated_near_floor_gate'] else 'FAIL'} (observed {gates['attenuated_max_abs_estimate']:.6f})")
    print(f"  null-control |MI| <= {NULL_ESTIMATE_ABS_MAX:.2f}: {'PASS' if gates['null_near_zero_gate'] else 'FAIL'} (observed {gates['null_max_abs_estimate']:.6f})")
    print(f"  order signal-minus-null >= {ORDER_SIGNAL_MINUS_NULL_MIN:.2f}: {'PASS' if gates['order_specificity_gate'] else 'FAIL'}")
    print(f"  exact bottleneck localization: {'PASS' if gates['bottleneck_localization_gate'] else 'FAIL'}")
    for scenario in SCENARIOS:
        print(f"    {scenario}: {observed_drop[scenario]}")
    print(f"  selective LO retention >= {SELECTIVE_LO_RETENTION_MIN:.2f}: {'PASS' if gates['selective_lo_retention_gate'] else 'FAIL'} ({retention_stats['lo_retention']:.6f})")
    print(f"  selective HO retention <= {SELECTIVE_HO_RETENTION_MAX:.2f}: {'PASS' if gates['selective_ho_retention_gate'] else 'FAIL'} ({retention_stats['ho_retention']:.6f})")
    print(f"  Delta_HO-LO <= {SELECTIVE_DELTA_MAX:.2f}: {'PASS' if gates['selective_delta_gate'] else 'FAIL'} ({retention_stats['delta_ho_lo']:.6f})")
    print()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "purpose": "End-to-end synthetic validation of order-resolved target-side ORBIT-FMIB before GB1/ESM-2 interpretation.",
        "upstream_provenance": {
            "code03_manifest_path": str(code03_manifest_path.resolve()),
            "code03_manifest_sha256": code03_manifest_sha,
            "code04_manifest_path": str(code04_manifest_path.resolve()),
            "code04_manifest_sha256": code04_manifest_sha,
            "code04_source_path": str(code04_source_path.resolve()),
            "code04_source_sha256": code04_source_sha,
            "code04_pinned_code03_match": True,
            "code04_calibration_overall_pass": True,
        },
        "device": str(device),
        "torch_version": str(torch.__version__),
        "estimator_protocol": {
            "source": "Exact train_mine implementation imported from Code 04.",
            "critic_seeds": list(critic_seeds),
            "training_config": asdict(mine04.TrainingConfig()),
            "reported_estimate": "held-out test only",
            "selection": "validation only",
            "product_of_marginals": "Code-04 nonzero cyclic-shift protocol",
            "standardization": "train statistics only",
        },
        "sample_unit": {
            "primary": "(synthetic context/cube, Walsh subset) pooled within interaction order",
            "split_unit": "synthetic context/cube",
            "context_split_counts": {
                "train": split_counts[0],
                "val": split_counts[1],
                "test": split_counts[2],
            },
            "subsets_per_order": {str(k): int(math.comb(d, k)) for k in range(1, d + 1)},
        },
        "acceptance_thresholds": {
            "walsh_reconstruction_max_abs": WALSH_RECON_MAX_ABS,
            "strong_truth_min": STRONG_TRUTH_MIN,
            "strong_estimate_min": STRONG_ESTIMATE_MIN,
            "strong_max_abs_error": STRONG_MAX_ABS_ERROR,
            "attenuated_truth_max": ATTENUATED_TRUTH_MAX,
            "attenuated_estimate_abs_max": ATTENUATED_ESTIMATE_ABS_MAX,
            "null_estimate_abs_max": NULL_ESTIMATE_ABS_MAX,
            "max_seed_sd": MAX_SEED_SD,
            "drop_delta_threshold": DROP_DELTA_THRESHOLD,
            "order_signal_minus_null_min": ORDER_SIGNAL_MINUS_NULL_MIN,
            "selective_lo_retention_min": SELECTIVE_LO_RETENTION_MIN,
            "selective_ho_retention_max": SELECTIVE_HO_RETENTION_MAX,
            "selective_delta_max": SELECTIVE_DELTA_MAX,
        },
        "walsh_audit": {
            **structural,
            "max_hidden_roundtrip_abs_error": max_walsh_recon_error,
        },
        "observed_drop_stage_by_order": {
            scenario: {str(k): v for k, v in mapping.items()}
            for scenario, mapping in observed_drop.items()
        },
        "expected_drop_stage_by_order": {
            scenario: code03_manifest["scenarios"][scenario]["drop_stage_by_order"]
            for scenario in SCENARIOS
        },
        "selective_retention": retention_stats,
        "acceptance_results": gates,
        "overall_pass": bool(gates["overall_pass"]),
        "interpretation_limits": [
            "Synthetic channels are calibration instruments, not biological neural representations.",
            "MINE values are finite-sample DV lower-bound estimates and may be slightly negative near independence.",
            "Passing Code 05 validates the measurement pipeline under the frozen high-contrast synthetic conditions; it does not guarantee detection of arbitrarily weak GB1 effects.",
            "The context-side I(Q_k;R_l,k) coordinate is not validated here; Phase 1 remains centered on I(R_l,k;E_k).",
            "No GB1/ESM-2 biological conclusion is permitted if overall_pass is false.",
        ],
    }

    manifest["file_sha256"] = {
        seed_path.name: sha256_file(seed_path),
        summary_path.name: sha256_file(summary_path),
    }
    write_json(manifest_path, manifest)
    write_json(
        state_path,
        {
            "code_name": CODE_NAME,
            "fingerprint": fingerprint,
            "status": "complete_pass" if gates["overall_pass"] else "complete_fail",
            "validation_manifest_sha256": sha256_file(manifest_path),
        },
    )

    print("Saved:")
    print(f"  {seed_path}")
    print(f"  {summary_path}")
    print(f"  {manifest_path}")
    print(f"  {state_path}")
    print()

    if not gates["overall_pass"]:
        raise RuntimeError(
            "ORBIT-FMIB CODE 05: FAIL — synthetic end-to-end acceptance gates were not all satisfied. "
            "Do not proceed to biological GB1/ESM-2 interpretation."
        )

    print("ORBIT-FMIB CODE 05: PASS")
    print("Synthetic measurement gate cleared: GB1 pipeline may proceed to Code 06.")


if __name__ == "__main__":
    main()
