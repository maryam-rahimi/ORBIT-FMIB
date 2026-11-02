#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ORBIT-FMIB companion diagnostic: adversarial 2560-D relative-retention suite.

Purpose
-------
The first 2560-D relative-retention probe tested one synthetic pattern matching
our biological hypothesis (preferential higher-order loss). This companion test
is deliberately hypothesis-adversarial. It asks whether the exact frozen Code-04
MINE estimator can distinguish four pre-specified outcomes at the real 2560-D
representation dimension:

  A. ho_loss        : k1 > k2 > k3 > k4  (Delta_HO-LO < 0)
  B. flat_strong    : k1 = k2 = k3 = k4  (Delta_HO-LO = 0)
  C. ho_advantage   : k1 < k2 < k3 < k4  (Delta_HO-LO > 0)
  D. flat_moderate  : k1 = k2 = k3 = k4  (Delta_HO-LO = 0, lower-signal regime)

All scenarios share the same analytic Gaussian-channel information law:

    I(E; R) = 0.5 * log(1 + gain^2 / sigma^2),
    R = gain * E * u + sigma * eta,

with E ~ N(0,1), eta ~ N(0,I), ||u|| = 1, sigma = 1.

The exact live Code-04 train_mine implementation is used. Absolute MI accuracy
is NOT claimed or required. The suite tests relative ordering, contrast sign,
flat-case false-selectivity, null behavior, seed stability, and finiteness.

No PCA, projection, clamping, null subtraction, ESM-2 data, or GB1 labels are
used.

Recommended HPC cluster sequence
---------------------------
  python "03 Code/11_validate_relative_retention_adversarial_suite.py" --self-test
  python "03 Code/11_validate_relative_retention_adversarial_suite.py" --preflight-only
  python "03 Code/11_validate_relative_retention_adversarial_suite.py" --device cuda

If interrupted:
  python "03 Code/11_validate_relative_retention_adversarial_suite.py" --device cuda --resume
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import itertools
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

CODE_NAME = "ORBIT-FMIB ADVERSARIAL RELATIVE-RETENTION SUITE"
SCHEMA_VERSION = 1

REPRESENTATION_DIM = 2560
SIGMA = 1.0
CRITIC_SEEDS = (11, 23, 37)
TARGET_SEED = 20260941
DIRECTION_SEED = 20260942
NOISE_SEED_BASE = 20260950

COMMON_NODES = (
    ("null", 0.00),
    ("layer_00_start", 0.15),
    ("layer_01_mid", 0.50),
    ("layer_02_peak", 1.50),
)
PEAK_NODE = "layer_02_peak"

# Pre-specified adversarial scenarios. No values are chosen from biological data.
SCENARIOS = {
    "ho_loss": {
        "expected": "negative",
        "gains": (1.20, 1.00, 0.50, 0.15),
    },
    "flat_strong": {
        "expected": "flat",
        "gains": (1.00, 1.00, 1.00, 1.00),
    },
    "ho_advantage": {
        "expected": "positive",
        "gains": (0.15, 0.50, 1.00, 1.20),
    },
    "flat_moderate": {
        "expected": "flat",
        "gains": (0.50, 0.50, 0.50, 0.50),
    },
}

SIGNAL_SEPARATION_MIN = 0.05
FLAT_PERMUTATION_ALPHA = 0.05


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_device(torch, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def analytic_mi(gain: float, sigma: float = SIGMA) -> float:
    return float(0.5 * math.log(1.0 + (float(gain) ** 2) / (float(sigma) ** 2)))


def build_node_table() -> pd.DataFrame:
    rows = []
    idx = 0
    for name, gain in COMMON_NODES:
        rows.append(
            {
                "node_index": idx,
                "node": name,
                "scenario": "common",
                "expected": "common",
                "order_k": 0,
                "gain": float(gain),
                "true_mi_nats": analytic_mi(gain),
                "noise_seed": int(NOISE_SEED_BASE + idx),
            }
        )
        idx += 1

    for scenario, cfg in SCENARIOS.items():
        gains = tuple(float(x) for x in cfg["gains"])
        if len(gains) != 4:
            raise RuntimeError(f"Scenario {scenario} must contain four gains.")
        for k, gain in enumerate(gains, start=1):
            rows.append(
                {
                    "node_index": idx,
                    "node": f"{scenario}_final_k{k}",
                    "scenario": scenario,
                    "expected": str(cfg["expected"]),
                    "order_k": int(k),
                    "gain": gain,
                    "true_mi_nats": analytic_mi(gain),
                    "noise_seed": int(NOISE_SEED_BASE + idx),
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def make_shared_target_and_direction(n: int) -> Tuple[np.ndarray, np.ndarray]:
    erng = np.random.default_rng(TARGET_SEED)
    e = erng.normal(size=(n, 1)).astype(np.float32)

    drng = np.random.default_rng(DIRECTION_SEED)
    u = drng.normal(size=(REPRESENTATION_DIM,))
    norm = float(np.linalg.norm(u))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Failed to build finite nonzero channel direction.")
    u = (u / norm).astype(np.float64)
    return e, u


def make_node_representation(e: np.ndarray, u: np.ndarray, gain: float, noise_seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(noise_seed))
    noise = rng.normal(size=(len(e), REPRESENTATION_DIM)).astype(np.float32) * np.float32(SIGMA)
    r = np.float32(gain) * e * u.astype(np.float32)[None, :] + noise
    if r.shape != (len(e), REPRESENTATION_DIM) or not np.all(np.isfinite(r)):
        raise RuntimeError("Synthetic node representation failed shape/finiteness audit.")
    return r.astype(np.float32, copy=False)


def summarize_seed_results(seed_df: pd.DataFrame, node_df: pd.DataFrame) -> pd.DataFrame:
    meta = node_df.set_index("node")
    rows = []
    for node, g in seed_df.groupby("node", sort=False):
        if len(g) != len(CRITIC_SEEDS):
            raise RuntimeError(f"Node {node} does not contain exactly {len(CRITIC_SEEDS)} critic seeds.")
        vals = g["test_mi"].to_numpy(float)
        m = meta.loc[node]
        rows.append(
            {
                "node_index": int(m.node_index),
                "node": node,
                "scenario": str(m.scenario),
                "expected": str(m.expected),
                "order_k": int(m.order_k),
                "gain": float(m.gain),
                "true_mi_nats": float(m.true_mi_nats),
                "estimate_mean": float(vals.mean()),
                "estimate_sd_across_seeds": float(vals.std(ddof=1)),
                "estimate_min": float(vals.min()),
                "estimate_max": float(vals.max()),
                "mean_best_val": float(g["best_val_mi"].mean()),
                "mean_best_step": float(g["best_step"].mean()),
                "mean_final_step": float(g["final_training_step"].mean()),
                "mean_product_shuffle_sd": float(g["test_product_shuffle_sd"].mean()),
                "mean_joint_minus_mismatch_score": float(g["heldout_joint_minus_mismatch_score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("node_index").reset_index(drop=True)


def scenario_retention_tables(
    seed_df: pd.DataFrame, summary: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sm = summary.set_index("node")
    peak_true = float(sm.loc[PEAK_NODE, "true_mi_nats"])
    peak_est = float(sm.loc[PEAK_NODE, "estimate_mean"])
    if not np.isfinite(peak_est) or peak_est <= 0.0:
        raise RuntimeError("Peak estimate must be finite and > 0.")

    mean_rows = []
    scenario_rows = []
    seed_rows = []

    for scenario, cfg in SCENARIOS.items():
        expected = str(cfg["expected"])
        scenario_mean_rets = []
        scenario_true_rets = []
        for k in range(1, 5):
            node = f"{scenario}_final_k{k}"
            tr = float(sm.loc[node, "true_mi_nats"] / peak_true)
            er = float(sm.loc[node, "estimate_mean"] / peak_est)
            scenario_true_rets.append(tr)
            scenario_mean_rets.append(er)
            mean_rows.append(
                {
                    "scenario": scenario,
                    "expected": expected,
                    "order_k": k,
                    "final_node": node,
                    "true_retention": tr,
                    "estimated_relative_retention": er,
                }
            )

        trv = np.asarray(scenario_true_rets, dtype=float)
        erv = np.asarray(scenario_mean_rets, dtype=float)
        true_delta = float(trv[2:].mean() - trv[:2].mean())
        est_delta = float(erv[2:].mean() - erv[:2].mean())
        scenario_rows.append(
            {
                "scenario": scenario,
                "expected": expected,
                "true_lo_retention": float(trv[:2].mean()),
                "true_ho_retention": float(trv[2:].mean()),
                "true_delta_ho_lo": true_delta,
                "estimated_lo_retention": float(erv[:2].mean()),
                "estimated_ho_retention": float(erv[2:].mean()),
                "estimated_delta_ho_lo": est_delta,
                "estimated_retention_spread": float(erv.max() - erv.min()),
            }
        )

        for seed in CRITIC_SEEDS:
            sd = seed_df[seed_df["critic_seed"] == seed].set_index("node")
            p = float(sd.loc[PEAK_NODE, "test_mi"])
            if not np.isfinite(p) or p <= 0.0:
                raise RuntimeError(f"Peak estimate at critic seed {seed} is not finite and > 0.")
            rets = []
            for k in range(1, 5):
                node = f"{scenario}_final_k{k}"
                rets.append(float(sd.loc[node, "test_mi"] / p))
            lo = float(np.mean(rets[:2]))
            ho = float(np.mean(rets[2:]))
            seed_rows.append(
                {
                    "scenario": scenario,
                    "expected": expected,
                    "critic_seed": int(seed),
                    "retain_k1": rets[0],
                    "retain_k2": rets[1],
                    "retain_k3": rets[2],
                    "retain_k4": rets[3],
                    "lo_retention": lo,
                    "ho_retention": ho,
                    "delta_ho_lo": float(ho - lo),
                }
            )

    return (
        pd.DataFrame(mean_rows).sort_values(["scenario", "order_k"]).reset_index(drop=True),
        pd.DataFrame(scenario_rows).sort_values("scenario").reset_index(drop=True),
        pd.DataFrame(seed_rows).sort_values(["scenario", "critic_seed"]).reset_index(drop=True),
    )


def evaluate_scenario(
    scenario: str,
    mean_ret: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    seed_ret: pd.DataFrame,
) -> Dict[str, object]:
    cfg = SCENARIOS[scenario]
    expected = str(cfg["expected"])
    mr = mean_ret[mean_ret["scenario"] == scenario].sort_values("order_k")
    ss = scenario_summary[scenario_summary["scenario"] == scenario].iloc[0]
    sr = seed_ret[seed_ret["scenario"] == scenario].sort_values("critic_seed")

    true_vals = mr["true_retention"].to_numpy(float)
    est_vals = mr["estimated_relative_retention"].to_numpy(float)
    deltas = sr["delta_ho_lo"].to_numpy(float)
    mean_delta = float(ss.estimated_delta_ho_lo)
    spread = float(ss.estimated_retention_spread)

    result: Dict[str, object] = {
        "scenario": scenario,
        "expected": expected,
        "true_delta_ho_lo": float(ss.true_delta_ho_lo),
        "estimated_delta_ho_lo": mean_delta,
        "estimated_retention_spread": spread,
        "seed_deltas": [float(x) for x in deltas],
    }

    if expected == "negative":
        result.update(
            {
                "true_order_strict_decreasing": bool(np.all(np.diff(true_vals) < 0.0)),
                "estimated_order_strict_decreasing": bool(np.all(np.diff(est_vals) < 0.0)),
                "mean_delta_negative": bool(mean_delta < 0.0),
                "all_seed_deltas_negative": bool(np.all(deltas < 0.0)),
            }
        )
        hard = (
            "true_order_strict_decreasing",
            "estimated_order_strict_decreasing",
            "mean_delta_negative",
            "all_seed_deltas_negative",
        )
    elif expected == "positive":
        result.update(
            {
                "true_order_strict_increasing": bool(np.all(np.diff(true_vals) > 0.0)),
                "estimated_order_strict_increasing": bool(np.all(np.diff(est_vals) > 0.0)),
                "mean_delta_positive": bool(mean_delta > 0.0),
                "all_seed_deltas_positive": bool(np.all(deltas > 0.0)),
            }
        )
        hard = (
            "true_order_strict_increasing",
            "estimated_order_strict_increasing",
            "mean_delta_positive",
            "all_seed_deltas_positive",
        )
    elif expected == "flat":
        # Under equal true gains, the four order labels are exchangeable.  Use an
        # exact 4! permutation test rather than an arbitrary retention tolerance.
        seed_matrix = sr[["retain_k1", "retain_k2", "retain_k3", "retain_k4"]].to_numpy(float)
        obs = float(np.mean(seed_matrix[:, 2:].mean(axis=1) - seed_matrix[:, :2].mean(axis=1)))
        perm_deltas = []
        for perm in itertools.permutations(range(4)):
            pm = seed_matrix[:, perm]
            pd = float(np.mean(pm[:, 2:].mean(axis=1) - pm[:, :2].mean(axis=1)))
            perm_deltas.append(pd)
        perm_deltas = np.asarray(perm_deltas, dtype=float)
        p_value = float(np.mean(np.abs(perm_deltas) >= abs(obs) - 1e-15))
        result.update(
            {
                "true_retention_equal": bool(np.max(true_vals) - np.min(true_vals) <= 1e-15),
                "flat_exact_permutation_p_value": p_value,
                "flat_no_false_selectivity_at_alpha": bool(p_value >= FLAT_PERMUTATION_ALPHA),
                "flat_permutation_count": int(len(perm_deltas)),
                "max_abs_seed_delta_report_only": float(np.max(np.abs(deltas))),
            }
        )
        hard = (
            "true_retention_equal",
            "flat_no_false_selectivity_at_alpha",
        )
    else:
        raise RuntimeError(f"Unknown expected scenario class: {expected}")

    result["hard_gates"] = list(hard)
    result["overall_pass"] = bool(all(bool(result[x]) for x in hard))
    return result


def audit_suite(
    summary: pd.DataFrame,
    mean_ret: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    seed_ret: pd.DataFrame,
    mine04,
) -> Dict[str, object]:
    sm = summary.set_index("node")
    est = {n: float(sm.loc[n, "estimate_mean"]) for n in sm.index}
    null_abs = abs(est["null"])
    max_sd = float(summary["estimate_sd_across_seeds"].max())
    peak = est[PEAK_NODE]

    finite_cols = [
        "estimate_mean",
        "estimate_sd_across_seeds",
        "mean_best_val",
        "mean_best_step",
        "mean_final_step",
        "mean_product_shuffle_sd",
        "mean_joint_minus_mismatch_score",
    ]

    scenario_gates = {
        scenario: evaluate_scenario(scenario, mean_ret, scenario_summary, seed_ret)
        for scenario in SCENARIOS
    }

    all_endpoint_nodes = [f"{s}_final_k{k}" for s in SCENARIOS for k in range(1, 5)]
    global_gates: Dict[str, object] = {
        "representation_dim": REPRESENTATION_DIM,
        "absolute_mi_accuracy_required": False,
        "null_abs_estimate": null_abs,
        "null_abs_le_code04_tolerance": bool(null_abs <= float(mine04.NULL_ABS_TOL)),
        "max_seed_sd": max_sd,
        "seed_stability": bool(max_sd <= float(mine04.MAX_SEED_SD)),
        "common_trajectory_monotonic": bool(
            est["layer_00_start"] < est["layer_01_mid"] < est["layer_02_peak"]
        ),
        "mean_peak_localized": bool(peak > max(est[n] for n in all_endpoint_nodes)),
        "peak_minus_null": float(peak - est["null"]),
        "signal_separation_ge_min": bool((peak - est["null"]) >= SIGNAL_SEPARATION_MIN),
        "all_summary_diagnostics_finite": bool(np.isfinite(summary[finite_cols].to_numpy(float)).all()),
        "all_retention_diagnostics_finite": bool(
            np.isfinite(mean_ret.select_dtypes(include=[np.number]).to_numpy(float)).all()
            and np.isfinite(scenario_summary.select_dtypes(include=[np.number]).to_numpy(float)).all()
            and np.isfinite(seed_ret.select_dtypes(include=[np.number]).to_numpy(float)).all()
        ),
        "all_scenarios_pass": bool(all(bool(v["overall_pass"]) for v in scenario_gates.values())),
    }
    global_hard = (
        "null_abs_le_code04_tolerance",
        "seed_stability",
        "common_trajectory_monotonic",
        "mean_peak_localized",
        "signal_separation_ge_min",
        "all_summary_diagnostics_finite",
        "all_retention_diagnostics_finite",
        "all_scenarios_pass",
    )
    global_gates["hard_gates"] = list(global_hard)
    global_gates["overall_pass"] = bool(all(bool(global_gates[x]) for x in global_hard))
    return {"global": global_gates, "scenarios": scenario_gates}


def fingerprint(code04_path: Path, code11_path: Path, device_type: str) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "code04_sha256": sha256_file(code04_path),
        "code11_sha256": sha256_file(code11_path),
        "device_type": device_type,
        "representation_dim": REPRESENTATION_DIM,
        "sigma": SIGMA,
        "critic_seeds": list(CRITIC_SEEDS),
        "target_seed": TARGET_SEED,
        "direction_seed": DIRECTION_SEED,
        "noise_seed_base": NOISE_SEED_BASE,
        "common_nodes": [list(x) for x in COMMON_NODES],
        "scenarios": SCENARIOS,
        "peak_node": PEAK_NODE,
        "signal_separation_min": SIGNAL_SEPARATION_MIN,
        "flat_permutation_alpha": FLAT_PERMUTATION_ALPHA,
    }


def preflight(code04_path: Path, code11_path: Path):
    if not code04_path.exists():
        raise FileNotFoundError(f"Code 04 source not found: {code04_path}")
    if not code11_path.exists():
        raise FileNotFoundError(f"Code 11 source not found: {code11_path}")

    mine04 = import_module_from_path(code04_path, "orbit_fmib_code04_adversarial_probe")
    code11 = import_module_from_path(code11_path, "orbit_fmib_code11_adversarial_probe")

    required04 = (
        "train_mine",
        "build_split_indices",
        "analytic_vector_channel_mi",
        "N_TOTAL",
        "N_TRAIN",
        "N_VAL",
        "N_TEST",
        "SPLIT_SEED",
        "CRITIC_SEEDS",
        "NULL_ABS_TOL",
        "MAX_SEED_SD",
    )
    missing = [x for x in required04 if not hasattr(mine04, x)]
    if missing:
        raise RuntimeError(f"Code 04 missing required APIs/constants: {missing}")

    if int(code11.REPRESENTATION_DIM) != REPRESENTATION_DIM:
        raise RuntimeError("Code 11 representation dimension mismatch.")
    if tuple(code11.CRITIC_SEEDS) != CRITIC_SEEDS:
        raise RuntimeError("Code 11 critic seed mismatch.")
    if tuple(mine04.CRITIC_SEEDS) != CRITIC_SEEDS:
        raise RuntimeError("Code 04 critic seed mismatch.")
    if int(mine04.N_TOTAL) != 8192:
        raise RuntimeError("Expected Code-04 N_TOTAL=8192.")

    node_df = build_node_table()
    if len(node_df) != 20 or not node_df["node"].is_unique:
        raise RuntimeError("Expected exactly 20 unique synthetic nodes.")

    for gain in node_df["gain"].to_numpy(float):
        a = analytic_mi(gain)
        b = float(mine04.analytic_vector_channel_mi(float(gain), SIGMA))
        if not np.isclose(a, b, atol=0.0, rtol=1e-15):
            raise RuntimeError("Analytic MI formula disagrees with Code 04.")

    split = mine04.build_split_indices(mine04.N_TOTAL, mine04.SPLIT_SEED)
    if tuple(map(len, split)) != (mine04.N_TRAIN, mine04.N_VAL, mine04.N_TEST):
        raise RuntimeError("Code-04 split sizes do not match frozen contract.")

    e, u = make_shared_target_and_direction(mine04.N_TOTAL)
    if e.shape != (mine04.N_TOTAL, 1) or u.shape != (REPRESENTATION_DIM,):
        raise RuntimeError("Synthetic target/direction shape failure.")
    if not np.isclose(np.linalg.norm(u), 1.0, atol=1e-12):
        raise RuntimeError("Synthetic direction is not unit norm.")

    # Verify the truth classes before any estimator is trained.
    peak = analytic_mi(1.5)
    for scenario, cfg in SCENARIOS.items():
        vals = np.array([analytic_mi(g) / peak for g in cfg["gains"]], dtype=float)
        delta = float(vals[2:].mean() - vals[:2].mean())
        expected = cfg["expected"]
        if expected == "negative" and not (np.all(np.diff(vals) < 0.0) and delta < 0.0):
            raise RuntimeError(f"Truth contract failure for {scenario}.")
        if expected == "positive" and not (np.all(np.diff(vals) > 0.0) and delta > 0.0):
            raise RuntimeError(f"Truth contract failure for {scenario}.")
        if expected == "flat" and not (np.max(vals) - np.min(vals) <= 1e-15 and abs(delta) <= 1e-15):
            raise RuntimeError(f"Truth contract failure for {scenario}.")

    return mine04, code11


def self_test() -> None:
    node_df = build_node_table()
    assert len(node_df) == 20
    assert node_df["node"].is_unique
    assert np.isclose(analytic_mi(0.0), 0.0)
    assert np.isclose(analytic_mi(1.5), 0.5893274981708231)

    e, u = make_shared_target_and_direction(64)
    r = make_node_representation(e, u, 0.5, NOISE_SEED_BASE)
    assert e.shape == (64, 1)
    assert r.shape == (64, REPRESENTATION_DIM)
    assert np.isfinite(r).all()

    # Build compressed-but-faithful fake estimates to test every gate class.
    fake_common = {
        "null": -0.0001,
        "layer_00_start": 0.0002,
        "layer_01_mid": 0.006,
        "layer_02_peak": 0.205,
    }
    fake_scenarios = {
        "ho_loss": [0.120, 0.070, 0.006, 0.0002],
        "flat_strong": [0.080, 0.082, 0.079, 0.081],
        "ho_advantage": [0.0002, 0.006, 0.070, 0.120],
        "flat_moderate": [0.0050, 0.0060, 0.0045, 0.0055],
    }
    fake_map = dict(fake_common)
    for scenario, vals in fake_scenarios.items():
        for k, value in enumerate(vals, start=1):
            fake_map[f"{scenario}_final_k{k}"] = value

    fake = node_df.copy()
    fake["estimate_mean"] = fake["node"].map(fake_map)
    fake["estimate_sd_across_seeds"] = 0.01
    fake["estimate_min"] = fake["estimate_mean"] - 0.01
    fake["estimate_max"] = fake["estimate_mean"] + 0.01
    fake["mean_best_val"] = fake["estimate_mean"]
    fake["mean_best_step"] = 500.0
    fake["mean_final_step"] = 800.0
    fake["mean_product_shuffle_sd"] = 0.001
    fake["mean_joint_minus_mismatch_score"] = 0.1

    seed_rows = []
    seed_offsets = {11: -0.001, 23: 0.0, 37: 0.001}
    for seed in CRITIC_SEEDS:
        for row in node_df.itertuples(index=False):
            value = float(fake_map[row.node])
            # Apply tiny common seed perturbation to non-null nodes; sign classes remain intact.
            if row.node != "null":
                value += seed_offsets[seed]
            seed_rows.append({"node": row.node, "critic_seed": seed, "test_mi": value})
    seed_df = pd.DataFrame(seed_rows)

    mean_ret, scenario_summary, seed_ret = scenario_retention_tables(seed_df, fake)

    class Dummy04:
        NULL_ABS_TOL = 0.05
        MAX_SEED_SD = 0.05

    gates = audit_suite(fake, mean_ret, scenario_summary, seed_ret, Dummy04)
    assert gates["global"]["overall_pass"] is True
    assert all(v["overall_pass"] for v in gates["scenarios"].values())

    # Deliberate negative control: reverse the expected negative scenario estimates.
    bad_mean = mean_ret.copy()
    mask = bad_mean["scenario"] == "ho_loss"
    bad_mean.loc[mask, "estimated_relative_retention"] = bad_mean.loc[mask, "estimated_relative_retention"].to_numpy()[::-1]
    bad_eval = evaluate_scenario("ho_loss", bad_mean, scenario_summary, seed_ret)
    assert bad_eval["overall_pass"] is False

    print("SELF-TEST: PASS")
    print("Verified: 20-node/60-fit arithmetic, four truth classes, 2560-D generator, exact flat-case permutation test, directional gates, and deliberate wrong-direction rejection.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--code04-source", type=Path)
    ap.add_argument("--code11-source", type=Path)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    root = project_root()
    code04_path = (args.code04_source or (root / "03 Code" / "04_validate_mi_estimator.py")).resolve()
    code11_path = (args.code11_source or (root / "03 Code" / "11_run_gb1_target_mi.py")).resolve()
    out = (args.output_dir or (root / "04 Experiments" / "EXP03_MI")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    mine04, code11 = preflight(code04_path, code11_path)
    torch, nn = mine04.import_torch()
    device = resolve_device(torch, args.device)

    fp = fingerprint(code04_path, code11_path, device.type)
    fpsha = canonical_sha256(fp)
    node_df = build_node_table()

    seed_path = out / "relative_retention_adversarial_2560d_seed_results.csv"
    summary_path = out / "relative_retention_adversarial_2560d_summary.csv"
    retention_path = out / "relative_retention_adversarial_2560d_retention.csv"
    scenario_path = out / "relative_retention_adversarial_2560d_scenario_summary.csv"
    seed_ret_path = out / "relative_retention_adversarial_2560d_seed_retention.csv"
    state_path = out / "relative_retention_adversarial_2560d_state.json"
    manifest_path = out / "relative_retention_adversarial_2560d_manifest.json"

    print("=" * 76)
    print(CODE_NAME)
    print("=" * 76)
    print(f"Code 04 source SHA256: {sha256_file(code04_path)}")
    print(f"Code 11 source SHA256: {sha256_file(code11_path)}")
    print(f"device={device.type}, dim={REPRESENTATION_DIM}, nodes={len(node_df)}, critic fits={len(node_df)*len(CRITIC_SEEDS)}")
    print(f"fingerprint={fpsha}")
    print("Absolute-MI accuracy gate: NOT USED.")
    print("Scenarios: ho_loss / flat_strong / ho_advantage / flat_moderate")

    if args.preflight_only:
        print("PREFLIGHT: PASS")
        print("No critic was trained.")
        return

    existing_files = [seed_path, summary_path, retention_path, scenario_path, seed_ret_path, state_path, manifest_path]
    if args.overwrite:
        for p in existing_files:
            if p.exists():
                p.unlink()
    elif not args.resume and any(p.exists() for p in existing_files):
        raise RuntimeError("Existing adversarial-suite outputs found. Use --resume or --overwrite intentionally.")

    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("fingerprint_sha256") != fpsha:
            raise RuntimeError("Existing state fingerprint does not match current suite contract.")

    seed_df = pd.read_csv(seed_path) if seed_path.exists() else pd.DataFrame()
    expected_pairs = {(str(r.node), int(seed)) for r in node_df.itertuples(index=False) for seed in CRITIC_SEEDS}
    done = set() if seed_df.empty else {(str(r.node), int(r.critic_seed)) for r in seed_df.itertuples(index=False)}
    if not done.issubset(expected_pairs):
        raise RuntimeError("Unexpected rows in existing adversarial-suite seed file.")

    split = mine04.build_split_indices(mine04.N_TOTAL, mine04.SPLIT_SEED)
    e, u = make_shared_target_and_direction(mine04.N_TOTAL)

    for row in node_df.itertuples(index=False):
        missing = [seed for seed in CRITIC_SEEDS if (row.node, int(seed)) not in done]
        if not missing:
            continue
        r = make_node_representation(e, u, float(row.gain), int(row.noise_seed))
        for seed in missing:
            print(
                f"  NODE {row.node:>30s} gain={row.gain:.2f} seed={seed} truth={row.true_mi_nats:.6f}",
                flush=True,
            )
            res = mine04.train_mine(torch, nn, r, e, int(seed), device, split)
            rec = {
                "family": "dim2560_adversarial_relative_retention_suite",
                "representation_dim": REPRESENTATION_DIM,
                "target_dim": 1,
                "node_index": int(row.node_index),
                "node": row.node,
                "scenario": row.scenario,
                "expected": row.expected,
                "order_k": int(row.order_k),
                "gain": float(row.gain),
                "noise_sigma": SIGMA,
                "true_mi_nats": float(row.true_mi_nats),
                "target_seed": TARGET_SEED,
                "direction_seed": DIRECTION_SEED,
                "noise_seed": int(row.noise_seed),
                "critic_seed": int(seed),
                **res,
            }
            seed_df = pd.concat([seed_df, pd.DataFrame([rec])], ignore_index=True)
            seed_df = seed_df.sort_values(["node_index", "critic_seed"]).reset_index(drop=True)
            write_csv_atomic(seed_df, seed_path)
            done.add((row.node, int(seed)))
            write_json_atomic(
                state_path,
                {
                    "code_name": CODE_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "fingerprint": fp,
                    "fingerprint_sha256": fpsha,
                    "status": "running",
                    "completed_fits": len(done),
                    "expected_fits": len(expected_pairs),
                    "seed_results_sha256": sha256_file(seed_path),
                },
            )
        del r

    if done != expected_pairs:
        raise RuntimeError("Adversarial relative-retention suite is incomplete.")

    summary = summarize_seed_results(seed_df, node_df)
    mean_ret, scenario_summary, seed_ret = scenario_retention_tables(seed_df, summary)
    gates = audit_suite(summary, mean_ret, scenario_summary, seed_ret, mine04)

    write_csv_atomic(summary, summary_path)
    write_csv_atomic(mean_ret, retention_path)
    write_csv_atomic(scenario_summary, scenario_path)
    write_csv_atomic(seed_ret, seed_ret_path)

    manifest = {
        "code_name": CODE_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if gates["global"]["overall_pass"] else "FAIL",
        "purpose": "Hypothesis-adversarial validation of relative 2560-D MINE trajectory/retention fidelity without absolute MI claims.",
        "fingerprint": fp,
        "fingerprint_sha256": fpsha,
        "estimator_protocol": {
            "source": "exact live Code-04 train_mine implementation",
            "code04_source_sha256": sha256_file(code04_path),
            "code11_source_sha256": sha256_file(code11_path),
            "critic_seeds": list(CRITIC_SEEDS),
            "n_total": int(mine04.N_TOTAL),
            "n_train": int(mine04.N_TRAIN),
            "n_val": int(mine04.N_VAL),
            "n_test": int(mine04.N_TEST),
            "absolute_mi_accuracy_claimed": False,
            "no_pca_projection": True,
            "no_clamping": True,
            "no_null_subtraction": True,
        },
        "synthetic_contract": {
            "representation_dim": REPRESENTATION_DIM,
            "target_dim": 1,
            "sigma": SIGMA,
            "target_seed": TARGET_SEED,
            "direction_seed": DIRECTION_SEED,
            "noise_seed_base": NOISE_SEED_BASE,
            "common_nodes": [list(x) for x in COMMON_NODES],
            "scenarios": SCENARIOS,
            "peak_node": PEAK_NODE,
            "flat_test": {
                "method": "exact 24-permutation order-label test applied consistently across critic seeds",
                "alpha": FLAT_PERMUTATION_ALPHA,
            },
        },
        "gates": gates,
        "outputs": {
            "seed_results": seed_path.name,
            "summary": summary_path.name,
            "retention": retention_path.name,
            "scenario_summary": scenario_path.name,
            "seed_retention": seed_ret_path.name,
        },
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        state_path,
        {
            "code_name": CODE_NAME,
            "schema_version": SCHEMA_VERSION,
            "fingerprint": fp,
            "fingerprint_sha256": fpsha,
            "status": "PASS" if gates["global"]["overall_pass"] else "FAIL",
            "completed_fits": len(done),
            "expected_fits": len(expected_pairs),
            "seed_results_sha256": sha256_file(seed_path),
            "summary_sha256": sha256_file(summary_path),
            "retention_sha256": sha256_file(retention_path),
            "scenario_summary_sha256": sha256_file(scenario_path),
            "seed_retention_sha256": sha256_file(seed_ret_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )

    print("\nScenario summary")
    print(
        scenario_summary[
            ["scenario", "expected", "true_delta_ho_lo", "estimated_delta_ho_lo", "estimated_retention_spread"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("\nScenario gates")
    for scenario in SCENARIOS:
        sg = gates["scenarios"][scenario]
        print(f"  {scenario} ({sg['expected']}): {'PASS' if sg['overall_pass'] else 'FAIL'}")
        for name in sg["hard_gates"]:
            print(f"    {name}: {'PASS' if sg[name] else 'FAIL'}")

    print("\nGlobal gates")
    gg = gates["global"]
    for name in gg["hard_gates"]:
        print(f"  {name}: {'PASS' if gg[name] else 'FAIL'}")

    if gg["overall_pass"]:
        print(f"\n{CODE_NAME}: PASS")
        print("Interpretation: the frozen estimator distinguishes negative, flat, and positive order-selective retention patterns at 2560-D while controlling flat-case false selectivity. Absolute MI accuracy is still not claimed.")
    else:
        print(f"\n{CODE_NAME}: FAIL")
        raise RuntimeError("Adversarial relative-retention suite failed. Do not run biological Code 11 yet.")


if __name__ == "__main__":
    main()
