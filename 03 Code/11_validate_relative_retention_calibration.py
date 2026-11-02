#!/usr/bin/env python3
"""ORBIT-FMIB companion diagnostic — 2560-D relative-retention calibration.

Purpose
-------
Code 11's original 2560-D calibration showed that the frozen Code-04 MINE
critic underestimates *absolute* mutual information at moderate/strong signal.
Before changing the biological pipeline, this diagnostic asks the narrower
question actually needed by the ORBIT-FMIB Phase-1 analysis:

    Does the exact same estimator preserve a known layerwise/retention ordering
    at the real 2560-D representation dimension?

This script DOES NOT weaken or overwrite Code 11.  It runs a separate,
pre-specified synthetic test using the exact live Code-04 ``train_mine``
implementation and the Code-11 representation dimension / critic seeds.

The synthetic design has a common rising trajectory (null -> start -> mid ->
peak) and four branch endpoints that mimic order-specific final-layer
retention.  The true MI of every node is analytic:

    I(E; R) = 0.5 * log(1 + gain^2 / sigma^2),
    R = gain * E * u + sigma * eta,

with E~N(0,1), eta~N(0,I), ||u||=1, sigma=1.

No PCA, projection, clamping, null subtraction, or biological data are used.

Recommended HPC cluster sequence
---------------------------
  python "03 Code/11_validate_relative_retention_calibration.py" --self-test
  python "03 Code/11_validate_relative_retention_calibration.py" --preflight-only
  python "03 Code/11_validate_relative_retention_calibration.py" --device cuda

If interrupted:
  python "03 Code/11_validate_relative_retention_calibration.py" --device cuda --resume
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

CODE_NAME = "ORBIT-FMIB RELATIVE-RETENTION CALIBRATION"
SCHEMA_VERSION = 1

REPRESENTATION_DIM = 2560
SIGMA = 1.0
CRITIC_SEEDS = (11, 23, 37)
TARGET_SEED = 20260921
DIRECTION_SEED = 20260922
NOISE_SEED_BASE = 20260930

# Pre-specified branched trajectory.  The first four nodes are the common
# trajectory; the last four are order-analogue final-layer endpoints.
NODES = (
    ("null", 0.00, "null", 0),
    ("layer_00_start", 0.15, "common", 0),
    ("layer_01_mid", 0.50, "common", 1),
    ("layer_02_peak", 1.50, "common", 2),
    ("final_k1", 1.20, "final", 1),
    ("final_k2", 1.00, "final", 2),
    ("final_k3", 0.50, "final", 3),
    ("final_k4", 0.15, "final", 4),
)
PEAK_NODE = "layer_02_peak"
FINAL_NODES = {1: "final_k1", 2: "final_k2", 3: "final_k3", 4: "final_k4"}
SIGNAL_SEPARATION_MIN = 0.05  # nats, estimator-scale diagnostic only


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


def node_table() -> pd.DataFrame:
    rows = []
    for i, (name, gain, kind, order_k) in enumerate(NODES):
        rows.append(
            {
                "node_index": i,
                "node": name,
                "kind": kind,
                "order_k": int(order_k),
                "gain": float(gain),
                "true_mi_nats": analytic_mi(gain),
                "noise_seed": int(NOISE_SEED_BASE + i),
            }
        )
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


def summarize_seed_results(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    nt = node_table().set_index("node")
    for node, g in seed_df.groupby("node", sort=False):
        if len(g) != len(CRITIC_SEEDS):
            raise RuntimeError(f"Node {node} does not contain exactly {len(CRITIC_SEEDS)} critic seeds.")
        vals = g["test_mi"].to_numpy(float)
        meta = nt.loc[node]
        rows.append(
            {
                "node_index": int(meta.node_index),
                "node": node,
                "kind": str(meta.kind),
                "order_k": int(meta.order_k),
                "gain": float(meta.gain),
                "true_mi_nats": float(meta.true_mi_nats),
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


def retention_tables(seed_df: pd.DataFrame, summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sm = summary.set_index("node")
    peak_true = float(sm.loc[PEAK_NODE, "true_mi_nats"])
    peak_est = float(sm.loc[PEAK_NODE, "estimate_mean"])
    if not np.isfinite(peak_est) or peak_est <= 0.0:
        raise RuntimeError("Peak estimate must be finite and > 0 to define relative-retention scores.")

    mean_rows = []
    for k, node in FINAL_NODES.items():
        t = float(sm.loc[node, "true_mi_nats"] / peak_true)
        e = float(sm.loc[node, "estimate_mean"] / peak_est)
        mean_rows.append(
            {
                "order_k": int(k),
                "final_node": node,
                "true_retention": t,
                "estimated_relative_retention": e,
            }
        )
    mean_df = pd.DataFrame(mean_rows).sort_values("order_k").reset_index(drop=True)

    seed_rows = []
    for seed in CRITIC_SEEDS:
        sd = seed_df[seed_df["critic_seed"] == seed].set_index("node")
        p = float(sd.loc[PEAK_NODE, "test_mi"])
        if not np.isfinite(p) or p <= 0.0:
            raise RuntimeError(f"Peak estimate at critic seed {seed} is not finite and > 0.")
        rets = {}
        for k, node in FINAL_NODES.items():
            rets[k] = float(sd.loc[node, "test_mi"] / p)
        lo = float((rets[1] + rets[2]) / 2.0)
        ho = float((rets[3] + rets[4]) / 2.0)
        seed_rows.append(
            {
                "critic_seed": int(seed),
                "retain_k1": rets[1],
                "retain_k2": rets[2],
                "retain_k3": rets[3],
                "retain_k4": rets[4],
                "lo_retention": lo,
                "ho_retention": ho,
                "delta_ho_lo": float(ho - lo),
            }
        )
    seed_ret = pd.DataFrame(seed_rows).sort_values("critic_seed").reset_index(drop=True)
    return mean_df, seed_ret


def audit_relative_fidelity(summary: pd.DataFrame, mean_ret: pd.DataFrame, seed_ret: pd.DataFrame, mine04) -> Dict[str, object]:
    sm = summary.set_index("node")
    est = {n: float(sm.loc[n, "estimate_mean"]) for n in sm.index}
    null_abs = abs(est["null"])
    max_sd = float(summary["estimate_sd_across_seeds"].max())
    peak = est[PEAK_NODE]

    mean_ret_vals = mean_ret.sort_values("order_k")["estimated_relative_retention"].to_numpy(float)
    true_ret_vals = mean_ret.sort_values("order_k")["true_retention"].to_numpy(float)

    # The frozen design has strictly decreasing true retention from k1 -> k4.
    true_order_ok = bool(np.all(np.diff(true_ret_vals) < 0.0))
    estimated_order_ok = bool(np.all(np.diff(mean_ret_vals) < 0.0))

    lo_mean = float(mean_ret_vals[:2].mean())
    ho_mean = float(mean_ret_vals[2:].mean())
    delta_mean = float(ho_mean - lo_mean)
    true_lo = float(true_ret_vals[:2].mean())
    true_ho = float(true_ret_vals[2:].mean())
    true_delta = float(true_ho - true_lo)

    finite_cols = [
        "estimate_mean",
        "estimate_sd_across_seeds",
        "mean_best_val",
        "mean_best_step",
        "mean_final_step",
        "mean_product_shuffle_sd",
        "mean_joint_minus_mismatch_score",
    ]

    g: Dict[str, object] = {
        "representation_dim": REPRESENTATION_DIM,
        "absolute_mi_accuracy_required": False,
        "null_abs_estimate": null_abs,
        "null_abs_le_code04_tolerance": bool(null_abs <= float(mine04.NULL_ABS_TOL)),
        "max_seed_sd": max_sd,
        "seed_stability": bool(max_sd <= float(mine04.MAX_SEED_SD)),
        "common_trajectory_monotonic": bool(
            est["layer_00_start"] < est["layer_01_mid"] < est["layer_02_peak"]
        ),
        "mean_peak_localized": bool(peak > max(est[n] for n in FINAL_NODES.values())),
        "peak_minus_null": float(peak - est["null"]),
        "signal_separation_ge_min": bool((peak - est["null"]) >= SIGNAL_SEPARATION_MIN),
        "true_retention_order_is_strict": true_order_ok,
        "estimated_retention_order_k1_gt_k2_gt_k3_gt_k4": estimated_order_ok,
        "true_lo_retention": true_lo,
        "true_ho_retention": true_ho,
        "true_delta_ho_lo": true_delta,
        "estimated_lo_retention": lo_mean,
        "estimated_ho_retention": ho_mean,
        "estimated_delta_ho_lo": delta_mean,
        "delta_sign_recovered": bool(true_delta < 0.0 and delta_mean < 0.0),
        "all_seed_deltas_negative": bool(np.all(seed_ret["delta_ho_lo"].to_numpy(float) < 0.0)),
        "all_summary_diagnostics_finite": bool(np.isfinite(summary[finite_cols].to_numpy(float)).all()),
        "all_retention_diagnostics_finite": bool(
            np.isfinite(mean_ret.select_dtypes(include=[np.number]).to_numpy(float)).all()
            and np.isfinite(seed_ret.select_dtypes(include=[np.number]).to_numpy(float)).all()
        ),
    }

    hard = (
        "null_abs_le_code04_tolerance",
        "seed_stability",
        "common_trajectory_monotonic",
        "mean_peak_localized",
        "signal_separation_ge_min",
        "true_retention_order_is_strict",
        "estimated_retention_order_k1_gt_k2_gt_k3_gt_k4",
        "delta_sign_recovered",
        "all_seed_deltas_negative",
        "all_summary_diagnostics_finite",
        "all_retention_diagnostics_finite",
    )
    g["hard_gates"] = list(hard)
    g["overall_pass"] = bool(all(bool(g[name]) for name in hard))
    return g


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
        "nodes": [list(x) for x in NODES],
        "peak_node": PEAK_NODE,
        "final_nodes": {str(k): v for k, v in FINAL_NODES.items()},
        "signal_separation_min": SIGNAL_SEPARATION_MIN,
    }


def preflight(code04_path: Path, code11_path: Path):
    if not code04_path.exists():
        raise FileNotFoundError(f"Code 04 source not found: {code04_path}")
    if not code11_path.exists():
        raise FileNotFoundError(f"Code 11 source not found: {code11_path}")

    mine04 = import_module_from_path(code04_path, "orbit_fmib_code04_relative_probe")
    code11 = import_module_from_path(code11_path, "orbit_fmib_code11_relative_probe")

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
        raise RuntimeError("Expected Code-04 N_TOTAL=8192 for dimension-matched calibration.")

    # Verify our analytic formula exactly matches the live Code-04 formula for all nodes.
    for _, gain, _, _ in NODES:
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

    return mine04, code11


def self_test() -> None:
    nt = node_table()
    assert len(nt) == 8
    assert nt["node"].is_unique
    assert np.isclose(analytic_mi(0.0), 0.0)
    assert np.isclose(analytic_mi(1.5), 0.5893274981708231)

    peak = analytic_mi(1.5)
    tr = np.array([analytic_mi(g) / peak for g in (1.2, 1.0, 0.5, 0.15)])
    assert np.all(np.diff(tr) < 0.0)
    true_delta = float(tr[2:].mean() - tr[:2].mean())
    assert true_delta < 0.0

    e, u = make_shared_target_and_direction(64)
    r = make_node_representation(e, u, 0.5, NOISE_SEED_BASE)
    assert e.shape == (64, 1)
    assert r.shape == (64, 2560)
    assert np.isfinite(r).all()

    # Gate arithmetic test with deliberately compressed but order-faithful estimates.
    fake = nt.copy()
    fake_map = {
        "null": -0.0001,
        "layer_00_start": 0.0002,
        "layer_01_mid": 0.006,
        "layer_02_peak": 0.205,
        "final_k1": 0.120,
        "final_k2": 0.070,
        "final_k3": 0.006,
        "final_k4": 0.0002,
    }
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
    for seed in CRITIC_SEEDS:
        for row in nt.itertuples(index=False):
            seed_rows.append(
                {
                    "node": row.node,
                    "critic_seed": seed,
                    "test_mi": fake_map[row.node],
                }
            )
    seed_df = pd.DataFrame(seed_rows)
    mean_ret, seed_ret = retention_tables(seed_df, fake)

    class Dummy04:
        NULL_ABS_TOL = 0.05
        MAX_SEED_SD = 0.05

    gates = audit_relative_fidelity(fake, mean_ret, seed_ret, Dummy04)
    assert gates["overall_pass"] is True

    print("SELF-TEST: PASS")
    print("Verified: 2560-D generator shape, analytic retention arithmetic, strict k1>k2>k3>k4 truth, and relative-fidelity gate logic.")


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

    seed_path = out / "relative_retention_2560d_seed_results.csv"
    summary_path = out / "relative_retention_2560d_summary.csv"
    retention_path = out / "relative_retention_2560d_retention.csv"
    seed_ret_path = out / "relative_retention_2560d_seed_retention.csv"
    state_path = out / "relative_retention_2560d_state.json"
    manifest_path = out / "relative_retention_2560d_manifest.json"

    print("=" * 76)
    print(CODE_NAME)
    print("=" * 76)
    print(f"Code 04 source SHA256: {sha256_file(code04_path)}")
    print(f"Code 11 source SHA256: {sha256_file(code11_path)}")
    print(f"device={device.type}, dim={REPRESENTATION_DIM}, nodes={len(NODES)}, critic fits={len(NODES)*len(CRITIC_SEEDS)}")
    print(f"fingerprint={fpsha}")
    print("Absolute-MI accuracy gate: NOT USED (this test evaluates relative trajectory/retention fidelity).")

    if args.preflight_only:
        print("PREFLIGHT: PASS")
        print("No critic was trained.")
        return

    existing_files = [seed_path, summary_path, retention_path, seed_ret_path, state_path, manifest_path]
    if args.overwrite:
        for p in existing_files:
            if p.exists():
                p.unlink()
    elif not args.resume and any(p.exists() for p in existing_files):
        raise RuntimeError("Existing diagnostic outputs found. Use --resume or --overwrite intentionally.")

    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("fingerprint_sha256") != fpsha:
            raise RuntimeError("Existing state fingerprint does not match current diagnostic contract.")

    seed_df = pd.read_csv(seed_path) if seed_path.exists() else pd.DataFrame()
    expected = {(name, int(seed)) for name, _, _, _ in NODES for seed in CRITIC_SEEDS}
    done = set() if seed_df.empty else {(str(r.node), int(r.critic_seed)) for r in seed_df.itertuples(index=False)}
    if not done.issubset(expected):
        raise RuntimeError("Unexpected rows in existing relative-retention seed file.")

    split = mine04.build_split_indices(mine04.N_TOTAL, mine04.SPLIT_SEED)
    e, u = make_shared_target_and_direction(mine04.N_TOTAL)
    nt = node_table()

    for row in nt.itertuples(index=False):
        missing = [seed for seed in CRITIC_SEEDS if (row.node, int(seed)) not in done]
        if not missing:
            continue
        r = make_node_representation(e, u, float(row.gain), int(row.noise_seed))
        for seed in missing:
            print(
                f"  NODE {row.node:>14s} gain={row.gain:.2f} seed={seed} truth={row.true_mi_nats:.6f}",
                flush=True,
            )
            res = mine04.train_mine(torch, nn, r, e, int(seed), device, split)
            rec = {
                "family": "dim2560_relative_retention_trajectory",
                "representation_dim": REPRESENTATION_DIM,
                "target_dim": 1,
                "node_index": int(row.node_index),
                "node": row.node,
                "kind": row.kind,
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
                    "expected_fits": len(expected),
                    "seed_results_sha256": sha256_file(seed_path),
                },
            )
        del r

    if done != expected:
        raise RuntimeError("Relative-retention calibration is incomplete.")

    summary = summarize_seed_results(seed_df)
    mean_ret, seed_ret = retention_tables(seed_df, summary)
    gates = audit_relative_fidelity(summary, mean_ret, seed_ret, mine04)

    write_csv_atomic(summary, summary_path)
    write_csv_atomic(mean_ret, retention_path)
    write_csv_atomic(seed_ret, seed_ret_path)

    manifest = {
        "code_name": CODE_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if gates["overall_pass"] else "FAIL",
        "purpose": "Validate relative 2560-D MINE trajectory/retention fidelity without claiming absolute MI accuracy.",
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
            "nodes": nt.to_dict(orient="records"),
            "peak_node": PEAK_NODE,
            "final_nodes": {str(k): v for k, v in FINAL_NODES.items()},
        },
        "gates": gates,
        "outputs": {
            "seed_results": seed_path.name,
            "summary": summary_path.name,
            "retention": retention_path.name,
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
            "status": "PASS" if gates["overall_pass"] else "FAIL",
            "completed_fits": len(done),
            "expected_fits": len(expected),
            "seed_results_sha256": sha256_file(seed_path),
            "summary_sha256": sha256_file(summary_path),
            "retention_sha256": sha256_file(retention_path),
            "seed_retention_sha256": sha256_file(seed_ret_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )

    print("\nRelative-retention summary")
    print(mean_ret.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSeed-level Δ_HO-LO")
    print(seed_ret[["critic_seed", "delta_ho_lo"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nGates")
    for name in gates["hard_gates"]:
        print(f"  {name}: {'PASS' if gates[name] else 'FAIL'}")

    if gates["overall_pass"]:
        print(f"\n{CODE_NAME}: PASS")
        print("Interpretation: the frozen estimator is adequate for a relative layerwise/retention analysis under this calibrated synthetic contract; absolute MI accuracy is still not claimed.")
    else:
        print(f"\n{CODE_NAME}: FAIL")
        raise RuntimeError("Relative-retention calibration failed. Do not run biological Code 11 yet.")


if __name__ == "__main__":
    main()
