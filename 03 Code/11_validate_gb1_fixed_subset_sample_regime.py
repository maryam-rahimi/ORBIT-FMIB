#!/usr/bin/env python3
"""ORBIT-FMIB companion diagnostic — GB1 fixed-subset sample-regime calibration.

Purpose
-------
The pooled GB1 Code-11 estimator was found to contain subset-identity
confounding.  The corrected biological estimand fixes one subset mask S per
critic, leaving exactly:

    train = 7,352 cubes
    val   =   245 cubes
    test  =   613 cubes
    total = 8,210 cubes

per fixed subset.

Before running the full 31-layer fixed-subset analysis, this script asks a
single pre-specified question:

    Does the exact frozen Code-04 2560-D MINE/DV critic retain useful
    dependence-detection behavior under the exact fixed-subset GB1
    train/validation/test sample regime?

It uses only synthetic Gaussian channels with known population MI:

    E ~ N(0,1)
    R = gain * E * u + sigma * eta
    I(E;R) = 0.5 * log(1 + gain^2 / sigma^2)

with ||u||=1, sigma=1, representation dimension 2560, three independent
synthetic replicates, and the exact Code-04 train_mine implementation.

This is NOT an absolute-MI accuracy calibration.  Absolute accuracy in nats
remains unclaimed.  The hard gates test null behavior, strong-signal
detectability, ordering, seed stability, replicate consistency, and
finiteness.  Weak/moderate conditions are reported to characterize the
estimator floor but are not required to be quantitatively accurate.

Recommended sequence
--------------------
Local:
  python "03 Code/11_validate_gb1_fixed_subset_sample_regime.py" --self-test
  python "03 Code/11_validate_gb1_fixed_subset_sample_regime.py" --preflight-only

HPC cluster / GPU:
  python "03 Code/11_validate_gb1_fixed_subset_sample_regime.py" --preflight-only --device cuda
  python "03 Code/11_validate_gb1_fixed_subset_sample_regime.py" --device cuda

If interrupted:
  python "03 Code/11_validate_gb1_fixed_subset_sample_regime.py" --device cuda --resume

Use --overwrite only for an intentional clean restart of THIS diagnostic.
It does not modify Code-11, Code-12, or subset-conditioned biological outputs.
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
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

CODE_NAME = "ORBIT-FMIB GB1 FIXED-SUBSET SAMPLE-REGIME CALIBRATION"
SCHEMA_VERSION = 1

EXPECTED_CODE04_NAME = "ORBIT-FMIB CODE 04"
EXPECTED_CODE11_NAME = "ORBIT-FMIB CODE 11"
EXPECTED_CODE11_SCHEMA = 2
EXPECTED_SUBSET_DIAG_SHA256 = "17bee1cdce057e0ac55f0eb0250db8770fee7fff962f096902aedf9ec2185362"

REPRESENTATION_DIM = 2560
TARGET_DIM = 1
SIGMA = 1.0
CRITIC_SEEDS = (11, 23, 37)
REPLICATE_IDS = (101, 211, 307)
GAINS = (0.0, 0.15, 0.50, 1.00, 1.50)

N_TRAIN = 7352
N_VAL = 245
N_TEST = 613
N_TOTAL = N_TRAIN + N_VAL + N_TEST

TARGET_SEED_BASE = 20261000
DIRECTION_SEED_BASE = 20262000
NOISE_SEED_BASE = 20263000

# Detection-oriented gates.  These deliberately do NOT require absolute-MI
# accuracy.  The Code-04 null/seed tolerances remain frozen.
STRONG_GAINS = (1.0, 1.5)
STRONG_MIN_ABOVE_NULL = 0.05
MAIN_ORDER_GAINS = (0.0, 0.5, 1.0, 1.5)

SEED_COLS = (
    "replicate_id", "gain", "true_mi_nats", "critic_seed",
    "representation_dim", "target_dim",
    "n_samples", "n_train", "n_val", "n_test",
    "best_step", "final_training_step", "best_val_mi", "test_mi",
    "test_product_shuffle_sd", "heldout_joint_minus_mismatch_score",
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, obj: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def import_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
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


def split_indices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr = np.arange(0, N_TRAIN, dtype=np.int64)
    va = np.arange(N_TRAIN, N_TRAIN + N_VAL, dtype=np.int64)
    te = np.arange(N_TRAIN + N_VAL, N_TOTAL, dtype=np.int64)
    if (len(tr), len(va), len(te)) != (N_TRAIN, N_VAL, N_TEST):
        raise RuntimeError("Frozen fixed-subset split counts failed.")
    if len(np.intersect1d(tr, va)) or len(np.intersect1d(tr, te)) or len(np.intersect1d(va, te)):
        raise RuntimeError("Frozen fixed-subset split overlap.")
    if not np.array_equal(np.sort(np.concatenate([tr, va, te])), np.arange(N_TOTAL)):
        raise RuntimeError("Frozen fixed-subset split does not cover all samples exactly once.")
    return tr, va, te


def synthetic_seeds(rep_id: int, gain_index: int) -> Tuple[int, int, int]:
    # Target and direction are common across gains within a replicate; noise is
    # gain-specific.  This makes gain ordering a controlled within-replicate test.
    t = TARGET_SEED_BASE + int(rep_id)
    d = DIRECTION_SEED_BASE + int(rep_id)
    n = NOISE_SEED_BASE + 100 * int(rep_id) + int(gain_index)
    return t, d, n


def make_target_and_direction(
    n: int, dim: int, target_seed: int, direction_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    erng = np.random.default_rng(int(target_seed))
    e = erng.normal(size=(n, 1)).astype(np.float32)

    drng = np.random.default_rng(int(direction_seed))
    u = drng.normal(size=(dim,)).astype(np.float64)
    norm = float(np.linalg.norm(u))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Synthetic direction is invalid.")
    u = (u / norm).astype(np.float32)
    return e, u


def make_representation(
    e: np.ndarray, u: np.ndarray, gain: float, noise_seed: int, sigma: float = SIGMA
) -> np.ndarray:
    rng = np.random.default_rng(int(noise_seed))
    noise = rng.normal(size=(len(e), len(u))).astype(np.float32)
    r = np.float32(gain) * e * u[None, :] + np.float32(sigma) * noise
    if r.shape != (len(e), len(u)) or not np.isfinite(r).all():
        raise RuntimeError("Synthetic representation shape/finiteness failure.")
    return r.astype(np.float32, copy=False)


def verify_subset_contract(subset_path: Path) -> Dict:
    got = sha256_file(subset_path)
    if got.lower() != EXPECTED_SUBSET_DIAG_SHA256.lower():
        raise RuntimeError(
            "Fixed-subset diagnostic source SHA mismatch.\n"
            f"Expected: {EXPECTED_SUBSET_DIAG_SHA256}\nObserved: {got}\nPath: {subset_path}"
        )
    mod = import_module(subset_path, "_orbitfmib_subset_sample_regime_contract")
    if int(mod.REPRESENTATION_DIM) != REPRESENTATION_DIM:
        raise RuntimeError("Subset diagnostic representation dimension changed.")
    if tuple(map(int, mod.CRITIC_SEEDS)) != CRITIC_SEEDS:
        raise RuntimeError("Subset diagnostic critic seeds changed.")
    counts = {str(k): int(v) for k, v in mod.STRICT_CUBE_COUNTS.items()}
    expected = {"estimator_train": N_TRAIN, "estimator_val": N_VAL, "estimator_test": N_TEST}
    if counts != expected:
        raise RuntimeError(f"Subset diagnostic split counts changed: {counts} != {expected}")
    if int(mod.N_PER_SUBSET) != N_TOTAL:
        raise RuntimeError("Subset diagnostic N_PER_SUBSET changed.")
    if len(tuple(mod.ALL_MASKS)) != 15:
        raise RuntimeError("Subset diagnostic no longer contains exactly 15 masks.")
    return {"sha256": got, "counts": counts}


def verify_code04_and_code11(code04_path: Path, code11_manifest_path: Path) -> Dict:
    m11 = load_json(code11_manifest_path)
    if m11.get("code_name") != EXPECTED_CODE11_NAME or int(m11.get("schema_version", -1)) != EXPECTED_CODE11_SCHEMA:
        raise RuntimeError("Code-11 manifest identity/schema mismatch.")
    if not bool(m11.get("overall_pass", False)):
        raise RuntimeError("Code-11 manifest is not PASS.")

    code04_sha = sha256_file(code04_path)
    pinned = str(m11.get("upstream", {}).get("code04_source_sha256", ""))
    if not pinned or code04_sha.lower() != pinned.lower():
        raise RuntimeError(
            "Current Code-04 source is not the exact estimator source pinned by completed Code 11."
        )

    mod = import_module(code04_path, "_orbitfmib_code04_fixed_subset_regime")
    required = (
        "CODE_NAME", "train_mine", "import_torch", "CRITIC_SEEDS",
        "NULL_ABS_TOL", "MAX_SEED_SD",
    )
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(f"Code-04 source missing required objects: {missing}")
    if mod.CODE_NAME != EXPECTED_CODE04_NAME:
        raise RuntimeError("Wrong Code-04 source.")
    if tuple(map(int, mod.CRITIC_SEEDS)) != CRITIC_SEEDS:
        raise RuntimeError("Code-04 critic seeds changed.")

    protocol = m11.get("estimator_protocol", {})
    if str(protocol.get("implementation", "")) != "exact Code-04 train_mine":
        raise RuntimeError("Completed Code 11 does not declare exact Code-04 train_mine.")
    if bool(protocol.get("absolute_mi_accuracy_claimed", True)):
        raise RuntimeError("Completed Code 11 unexpectedly claims absolute MI accuracy.")

    return {
        "module": mod,
        "code04_sha256": code04_sha,
        "code11_manifest_sha256": sha256_file(code11_manifest_path),
        "code11_manifest": m11,
    }


def fingerprint(est: Mapping, subset_contract: Mapping, device_type: str) -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "code04_source_sha256": est["code04_sha256"],
        "code11_manifest_sha256": est["code11_manifest_sha256"],
        "subset_diagnostic_source_sha256": subset_contract["sha256"],
        "representation_dim": REPRESENTATION_DIM,
        "target_dim": TARGET_DIM,
        "critic_seeds": list(CRITIC_SEEDS),
        "replicate_ids": list(REPLICATE_IDS),
        "gains": list(GAINS),
        "sigma": SIGMA,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_total": N_TOTAL,
        "device_type": str(device_type),
        "absolute_mi_accuracy_claimed": False,
        "analysis_contract": "fixed_subset_sample_regime_dependence_detection",
    }


def expected_keys() -> set:
    return {
        (int(rep), float(gain), int(seed))
        for rep in REPLICATE_IDS
        for gain in GAINS
        for seed in CRITIC_SEEDS
    }


def load_seed_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SEED_COLS)
    d = pd.read_csv(path)
    missing = set(SEED_COLS) - set(d.columns)
    if missing:
        raise RuntimeError(f"Existing sample-regime seed CSV missing columns: {sorted(missing)}")
    if d.duplicated(["replicate_id", "gain", "critic_seed"]).any():
        raise RuntimeError("Duplicate sample-regime critic rows.")
    return d


def save_seed_row(d: pd.DataFrame, row: Mapping, path: Path) -> pd.DataFrame:
    d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    d = d.sort_values(["replicate_id", "gain", "critic_seed"]).reset_index(drop=True)
    write_csv_atomic(d, path)
    return d


def run_calibration(
    mine04, torch, nn, device, seed_path: Path, state_path: Path,
    fp_sha: str, resume: bool
) -> pd.DataFrame:
    split = split_indices()
    d = load_seed_results(seed_path)
    expected = expected_keys()
    done = {
        (int(r.replicate_id), float(r.gain), int(r.critic_seed))
        for r in d.itertuples(index=False)
    }
    if not done.issubset(expected):
        raise RuntimeError("Existing sample-regime CSV has unexpected conditions.")

    for rep in REPLICATE_IDS:
        t_seed, d_seed, _ = synthetic_seeds(rep, 0)
        e, u = make_target_and_direction(N_TOTAL, REPRESENTATION_DIM, t_seed, d_seed)

        print(f"\nSynthetic replicate {rep}", flush=True)
        for gi, gain in enumerate(GAINS):
            missing = [s for s in CRITIC_SEEDS if (int(rep), float(gain), int(s)) not in done]
            if not missing:
                continue
            _, _, noise_seed = synthetic_seeds(rep, gi)
            r = make_representation(e, u, gain, noise_seed)
            print(
                f"  gain={gain:.2f} true_MI={analytic_mi(gain):.6f} "
                f"missing_seeds={missing}",
                flush=True,
            )
            for seed in missing:
                res = mine04.train_mine(
                    torch, nn, r, e, int(seed), device, split
                )
                row = {
                    "replicate_id": int(rep),
                    "gain": float(gain),
                    "true_mi_nats": analytic_mi(gain),
                    "critic_seed": int(seed),
                    "representation_dim": REPRESENTATION_DIM,
                    "target_dim": TARGET_DIM,
                    "n_samples": N_TOTAL,
                    "n_train": N_TRAIN,
                    "n_val": N_VAL,
                    "n_test": N_TEST,
                    **res,
                }
                d = save_seed_row(d, row, seed_path)
                done.add((int(rep), float(gain), int(seed)))
                write_json_atomic(
                    state_path,
                    {
                        "code_name": CODE_NAME,
                        "schema_version": SCHEMA_VERSION,
                        "fingerprint_sha256": fp_sha,
                        "status": "running",
                        "completed_fits": len(done),
                        "expected_fits": len(expected),
                        "last_completed": {
                            "replicate_id": int(rep),
                            "gain": float(gain),
                            "critic_seed": int(seed),
                        },
                    },
                )
            del r
        del e, u

    if done != expected:
        raise RuntimeError(f"Calibration grid incomplete: {len(expected - done)} fits missing.")
    return d


def summarize(seed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(seed_df) != len(expected_keys()):
        raise RuntimeError("Unexpected sample-regime seed-result count.")

    numeric = [
        "best_step", "final_training_step", "best_val_mi", "test_mi",
        "test_product_shuffle_sd", "heldout_joint_minus_mismatch_score",
    ]
    if not np.isfinite(seed_df[numeric].to_numpy(float)).all():
        raise RuntimeError("Non-finite sample-regime diagnostics.")

    reps = (
        seed_df.groupby(["replicate_id", "gain", "true_mi_nats"], as_index=False)
        .agg(
            estimate_mean=("test_mi", "mean"),
            estimate_sd_across_seeds=("test_mi", lambda x: float(np.std(np.asarray(x, float), ddof=1))),
            estimate_min=("test_mi", "min"),
            estimate_max=("test_mi", "max"),
            mean_best_val=("best_val_mi", "mean"),
            mean_best_step=("best_step", "mean"),
            mean_final_step=("final_training_step", "mean"),
            mean_product_shuffle_sd=("test_product_shuffle_sd", "mean"),
            mean_joint_minus_mismatch_score=("heldout_joint_minus_mismatch_score", "mean"),
        )
        .sort_values(["replicate_id", "gain"])
        .reset_index(drop=True)
    )

    gains = (
        reps.groupby(["gain", "true_mi_nats"], as_index=False)
        .agg(
            estimate_mean_across_replicates=("estimate_mean", "mean"),
            estimate_sd_across_replicates=("estimate_mean", lambda x: float(np.std(np.asarray(x, float), ddof=1))),
            min_replicate_mean=("estimate_mean", "min"),
            max_replicate_mean=("estimate_mean", "max"),
            max_within_replicate_seed_sd=("estimate_sd_across_seeds", "max"),
        )
        .sort_values("gain")
        .reset_index(drop=True)
    )
    return reps, gains


def audit(seed_df: pd.DataFrame, reps: pd.DataFrame, gains: pd.DataFrame, mine04) -> Dict:
    gmap = gains.set_index("gain")
    null_global = float(gmap.loc[0.0, "estimate_mean_across_replicates"])

    hard: Dict[str, object] = {}
    hard["absolute_mi_accuracy_required"] = False

    # Null behavior: global and each replicate.
    null_rep = reps[np.isclose(reps["gain"], 0.0)]["estimate_mean"].to_numpy(float)
    hard["global_null_abs_le_code04_tolerance"] = bool(abs(null_global) <= float(mine04.NULL_ABS_TOL))
    hard["all_replicate_null_abs_le_code04_tolerance"] = bool(
        np.all(np.abs(null_rep) <= float(mine04.NULL_ABS_TOL))
    )

    # Ordering omits gain 0.15 because prior 2560-D work established that very
    # weak signals can live near the estimator floor.
    main_vals = [
        float(gmap.loc[g, "estimate_mean_across_replicates"])
        for g in MAIN_ORDER_GAINS
    ]
    hard["main_signal_ordering_0_lt_0p5_lt_1_lt_1p5"] = bool(
        np.all(np.diff(np.asarray(main_vals, float)) > 0.0)
    )

    # Strong-signal separation from null, required in every replicate.
    all_strong_sep = True
    all_strong_positive = True
    for rep in REPLICATE_IDS:
        rr = reps[reps["replicate_id"] == rep].set_index("gain")
        n0 = float(rr.loc[0.0, "estimate_mean"])
        for sg in STRONG_GAINS:
            v = float(rr.loc[sg, "estimate_mean"])
            all_strong_sep &= bool(v >= n0 + STRONG_MIN_ABOVE_NULL)
            all_strong_positive &= bool(v > 0.0)
    hard["all_replicates_strong_signal_ge_null_plus_0p05"] = bool(all_strong_sep)
    hard["all_replicates_strong_signal_positive"] = bool(all_strong_positive)

    # Critic-seed stability is checked only where detection is required.
    strong_rep = reps[reps["gain"].isin(STRONG_GAINS)]
    max_strong_seed_sd = float(strong_rep["estimate_sd_across_seeds"].max())
    hard["max_strong_signal_seed_sd"] = max_strong_seed_sd
    hard["strong_signal_seed_stability_le_code04_threshold"] = bool(
        max_strong_seed_sd <= float(mine04.MAX_SEED_SD)
    )

    # Each strong gain should increase from 1.0 -> 1.5 in each replicate.
    strong_order = True
    for rep in REPLICATE_IDS:
        rr = reps[reps["replicate_id"] == rep].set_index("gain")
        strong_order &= bool(float(rr.loc[1.5, "estimate_mean"]) > float(rr.loc[1.0, "estimate_mean"]))
    hard["all_replicates_gain_1p5_gt_1p0"] = bool(strong_order)

    finite_cols = [
        "estimate_mean", "estimate_sd_across_seeds", "mean_best_val",
        "mean_best_step", "mean_final_step", "mean_product_shuffle_sd",
        "mean_joint_minus_mismatch_score",
    ]
    hard["all_replicate_summary_diagnostics_finite"] = bool(
        np.isfinite(reps[finite_cols].to_numpy(float)).all()
    )

    required = [
        "global_null_abs_le_code04_tolerance",
        "all_replicate_null_abs_le_code04_tolerance",
        "main_signal_ordering_0_lt_0p5_lt_1_lt_1p5",
        "all_replicates_strong_signal_ge_null_plus_0p05",
        "all_replicates_strong_signal_positive",
        "strong_signal_seed_stability_le_code04_threshold",
        "all_replicates_gain_1p5_gt_1p0",
        "all_replicate_summary_diagnostics_finite",
    ]
    hard["hard_gates"] = required
    hard["overall_pass"] = bool(all(bool(hard[x]) for x in required))

    # Descriptive detectability map, not an acceptance gate.
    detectability = {}
    for gain in GAINS:
        rr = reps[np.isclose(reps["gain"], gain)]
        flags = []
        for rep in REPLICATE_IDS:
            rrep = reps[reps["replicate_id"] == rep].set_index("gain")
            n0 = float(rrep.loc[0.0, "estimate_mean"])
            v = float(rrep.loc[gain, "estimate_mean"])
            flags.append(bool(v >= n0 + STRONG_MIN_ABOVE_NULL))
        detectability[f"gain_{gain:.2f}"] = {
            "true_mi_nats": analytic_mi(gain),
            "all_replicates_mean_ge_own_null_plus_0p05": bool(all(flags)),
            "n_replicates_meeting_criterion": int(sum(flags)),
        }
    hard["descriptive_detectability"] = detectability
    return hard


def self_test() -> None:
    assert REPRESENTATION_DIM == 2560
    assert (N_TRAIN, N_VAL, N_TEST, N_TOTAL) == (7352, 245, 613, 8210)
    assert len(expected_keys()) == len(REPLICATE_IDS) * len(GAINS) * len(CRITIC_SEEDS) == 45
    tr, va, te = split_indices()
    assert len(tr) == 7352 and len(va) == 245 and len(te) == 613
    assert np.isclose(analytic_mi(0.0), 0.0)
    assert np.isclose(analytic_mi(0.15), 0.011125304467409862)
    assert np.isclose(analytic_mi(0.5), 0.11157177565710488)
    assert np.isclose(analytic_mi(1.0), 0.34657359027997264)
    assert np.isclose(analytic_mi(1.5), 0.5893274981708231)

    e1, u1 = make_target_and_direction(17, 32, 101, 102)
    e2, u2 = make_target_and_direction(17, 32, 101, 102)
    np.testing.assert_array_equal(e1, e2)
    np.testing.assert_array_equal(u1, u2)
    assert np.isclose(np.linalg.norm(u1), 1.0, atol=1e-6)
    r1 = make_representation(e1, u1, 0.5, 103)
    r2 = make_representation(e1, u1, 0.5, 103)
    np.testing.assert_array_equal(r1, r2)
    assert r1.shape == (17, 32)
    print("SELF-TEST: PASS")
    print(
        "Verified: exact fixed-subset GB1 split counts 7352/245/613, "
        "disjoint split coverage, analytic Gaussian-channel MI values, "
        "deterministic synthetic generation, and 45-fit calibration-grid arithmetic."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--code04-source", type=Path)
    ap.add_argument("--code11-manifest", type=Path)
    ap.add_argument("--subset-diagnostic-source", type=Path)
    ap.add_argument("--output-dir", type=Path)
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return

    root = project_root()
    exp02 = root / "04 Experiments" / "EXP02_GB1_ESM2"
    exp03 = root / "04 Experiments" / "EXP03_MI"
    out = a.output_dir or exp03
    out.mkdir(parents=True, exist_ok=True)

    p04 = a.code04_source or root / "03 Code" / "04_validate_mi_estimator.py"
    p11m = a.code11_manifest or exp02 / "gb1_target_mi_manifest.json"
    psub = a.subset_diagnostic_source or root / "03 Code" / "11_validate_gb1_subset_conditioning.py"

    seed_out = out / "fixed_subset_sample_regime_seed_results.csv"
    rep_out = out / "fixed_subset_sample_regime_replicate_summary.csv"
    gain_out = out / "fixed_subset_sample_regime_gain_summary.csv"
    state_out = out / "fixed_subset_sample_regime_state.json"
    manifest_out = out / "fixed_subset_sample_regime_manifest.json"

    if a.overwrite:
        for p in (seed_out, rep_out, gain_out, state_out, manifest_out):
            if p.exists():
                p.unlink()

    print("=" * 82)
    print("ORBIT-FMIB — GB1 FIXED-SUBSET SAMPLE-REGIME CALIBRATION")
    print("=" * 82)

    subset_contract = verify_subset_contract(psub)
    est = verify_code04_and_code11(p04, p11m)
    mine04 = est["module"]

    torch, nn = mine04.import_torch()
    device = resolve_device(torch, a.device)
    if device.type == "cpu":
        torch.set_num_threads(1)

    fp = fingerprint(est, subset_contract, device.type)
    fp_sha = sha256_obj(fp)

    print("\nPreflight/provenance gate: PASS")
    print("  exact Code-04 train_mine source pinned by completed Code 11: PASS")
    print("  fixed-subset diagnostic source hash: PASS")
    print(f"  representation dimension: {REPRESENTATION_DIM}")
    print(f"  exact fixed-subset split: train={N_TRAIN} val={N_VAL} test={N_TEST} total={N_TOTAL}")
    print(f"  synthetic replicates: {len(REPLICATE_IDS)}")
    print(f"  gain conditions: {', '.join(f'{g:.2f}' for g in GAINS)}")
    print(f"  critic seeds: {CRITIC_SEEDS}")
    print(f"  planned critic fits: {len(expected_keys())}")
    print("  absolute MI accuracy claimed: NO")
    print("  PCA/projection: NO")
    print("  clamping: NO")
    print("  null subtraction: NO")
    print(f"  device: {device.type}")
    print(f"  fingerprint: {fp_sha}")

    if a.preflight_only:
        print("\nPREFLIGHT: PASS")
        print("No critic was trained and no calibration outputs were written.")
        return

    if manifest_out.exists() and not a.overwrite:
        m = load_json(manifest_out)
        if m.get("run_fingerprint_sha256") == fp_sha and bool(m.get("overall_pass", False)):
            print(f"\n{CODE_NAME}: PASS")
            print("Matching completed calibration already exists.")
            return
        raise RuntimeError("Existing sample-regime manifest does not match this run.")

    if state_out.exists() and not a.overwrite:
        st = load_json(state_out)
        if st.get("fingerprint_sha256") != fp_sha:
            raise RuntimeError("Existing sample-regime state fingerprint mismatch; use --overwrite for an intentional restart.")
        if not a.resume:
            raise RuntimeError("Existing sample-regime state detected. Re-run with --resume.")
    elif not state_out.exists():
        write_json_atomic(
            state_out,
            {
                "code_name": CODE_NAME,
                "schema_version": SCHEMA_VERSION,
                "fingerprint_sha256": fp_sha,
                "status": "initialized",
                "completed_fits": 0,
                "expected_fits": len(expected_keys()),
            },
        )

    seed_df = run_calibration(
        mine04, torch, nn, device, seed_out, state_out, fp_sha, a.resume
    )
    rep_df, gain_df = summarize(seed_df)
    gates = audit(seed_df, rep_df, gain_df, mine04)

    write_csv_atomic(rep_df, rep_out)
    write_csv_atomic(gain_df, gain_out)

    print("\nGain-level descriptive summary")
    print(
        gain_df[
            [
                "gain", "true_mi_nats", "estimate_mean_across_replicates",
                "estimate_sd_across_replicates", "max_within_replicate_seed_sd",
            ]
        ].to_string(index=False)
    )

    print("\nAcceptance gates")
    for name in gates["hard_gates"]:
        print(f"  {name}: {'PASS' if gates[name] else 'FAIL'}")
    print(f"  max strong-signal seed SD: {gates['max_strong_signal_seed_sd']:.6f}")
    print("  absolute MI accuracy claimed: NO")

    outputs = (seed_out, rep_out, gain_out)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "purpose": (
            "Validate dependence-detection behavior of the exact frozen 2560-D "
            "Code-04 MINE estimator under the exact 7352/245/613 fixed-subset GB1 sample regime."
        ),
        "source_sha256": sha256_file(Path(__file__)),
        "run_fingerprint": fp,
        "run_fingerprint_sha256": fp_sha,
        "upstream": {
            "code04_source_sha256": est["code04_sha256"],
            "code11_manifest_sha256": est["code11_manifest_sha256"],
            "subset_diagnostic_source_sha256": subset_contract["sha256"],
        },
        "synthetic_contract": {
            "representation_dim": REPRESENTATION_DIM,
            "target_dim": TARGET_DIM,
            "sigma": SIGMA,
            "gains": list(GAINS),
            "replicate_ids": list(REPLICATE_IDS),
            "critic_seeds": list(CRITIC_SEEDS),
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test": N_TEST,
            "n_total": N_TOTAL,
            "analytic_formula": "0.5*log(1+gain^2/sigma^2)",
            "absolute_mi_accuracy_required": False,
        },
        "acceptance_thresholds": {
            "null_abs_tolerance": float(mine04.NULL_ABS_TOL),
            "max_strong_signal_seed_sd": float(mine04.MAX_SEED_SD),
            "strong_min_above_own_replicate_null": STRONG_MIN_ABOVE_NULL,
            "strong_gains": list(STRONG_GAINS),
            "main_order_gains": list(MAIN_ORDER_GAINS),
        },
        "gates": gates,
        "outputs": {
            "seed_results": seed_out.name,
            "replicate_summary": rep_out.name,
            "gain_summary": gain_out.name,
        },
        "file_sha256": {p.name: sha256_file(p) for p in outputs},
        "overall_pass": bool(gates["overall_pass"]),
        "interpretation_limits": [
            "This calibration tests dependence-detection behavior, not exact absolute MI accuracy in nats.",
            "Gain 0.15 and gain 0.50 are retained to characterize the estimator floor; only gains 1.0 and 1.5 are hard-gated for robust signal separation.",
            "Passing this calibration does not prove that every non-Gaussian biological dependence is detectable.",
            "No PCA/projection, clamping, or null subtraction is used.",
            "Critic seeds and synthetic replicates are estimator/calibration replicates, not biological replicates.",
        ],
    }
    write_json_atomic(manifest_out, manifest)
    write_json_atomic(
        state_out,
        {
            "code_name": CODE_NAME,
            "schema_version": SCHEMA_VERSION,
            "fingerprint_sha256": fp_sha,
            "status": "complete_pass" if gates["overall_pass"] else "complete_fail",
            "completed_fits": len(seed_df),
            "expected_fits": len(expected_keys()),
            "manifest_sha256": sha256_file(manifest_out),
        },
    )

    verdict = "PASS" if gates["overall_pass"] else "FAIL"
    print(f"\n{CODE_NAME}: {verdict}")
    if not gates["overall_pass"]:
        failed = [x for x in gates["hard_gates"] if not bool(gates[x])]
        print("Failed hard gates:", ", ".join(failed))
        raise SystemExit(2)
    print("Exact fixed-subset sample regime is adequate for the pre-specified strong-signal detection contract.")


if __name__ == "__main__":
    main()
