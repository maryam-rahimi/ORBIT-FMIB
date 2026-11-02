#!/usr/bin/env python3
"""
ORBIT-FMIB CODE 04 — MINE / FlowNIB Measurement-Engine Validation
==================================================================

Purpose
-------
Validate the neural mutual-information estimator before it is used inside the
order-resolved ORBIT-FMIB pipeline.

This code implements the MINE-style Donsker–Varadhan (DV) lower bound used by
FlowNIB for post-hoc layerwise mutual-information estimation:

    I(U;V) >= E_joint[T_theta(U,V)]
              - log E_product[exp(T_theta(U,V))].

The representation/model is NOT trained here. Only the critic is optimized.
Product-of-marginals samples are formed by shuffling/rolling one member of the
pair, matching the FlowNIB/MINE logic.

Two calibration families have exact analytic MI:

1) Scalar correlated Gaussian
       (X,Y) jointly normal with correlation rho
       I(X;Y) = -0.5 log(1-rho^2)

2) Vector Gaussian channel matching Code 03
       E ~ N(0,1)
       R = gain * E * u + sigma * eta,
       eta ~ N(0,I), ||u||=1
       I(E;R) = 0.5 log(1 + gain^2/sigma^2)

The second family uses the exact hidden dimension, signal gain, attenuated gain,
and noise level recorded by Code 03. This directly calibrates the MI range that
Code 05 will encounter.

Scientific boundary
-------------------
This code validates the RAW target-side MINE/DV estimator. It deliberately does
NOT apply FlowNIB effective-dimension normalization or the alpha(t) two-critic
schedule because the Phase-1 confirmatory ORBIT-FMIB quantity is the raw
order-resolved target alignment I(R_{ell,k}; E_k). The context-side coordinate
is not considered validated by this script.

Outputs
-------
04 Experiments/EXP03_MI/
    mine_seed_results.csv
    mine_condition_summary.csv
    mine_validation_manifest.json

Usage
-----
    python "03 Code/04_validate_mi_estimator.py" --self-test
    python "03 Code/04_validate_mi_estimator.py"
    python "03 Code/04_validate_mi_estimator.py" --device cpu
    python "03 Code/04_validate_mi_estimator.py" --device cuda

Dependencies
------------
    numpy
    pandas
    torch

No GB1 fitness labels and no ESM-2 hidden states are accessed by this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


CODE_NAME = "ORBIT-FMIB CODE 04"
SCHEMA_VERSION = 1

# ----------------------------
# Frozen calibration protocol
# ----------------------------
N_TOTAL = 8192
N_TRAIN = 6144
N_VAL = 1024
N_TEST = 1024

SCALAR_RHOS = (0.0, 0.2, 0.5, 0.8)
CRITIC_SEEDS = (11, 23, 37)
SCALAR_DATA_SEED = 20260901
VECTOR_DATA_SEED = 20260902
SPLIT_SEED = 20260903
DIRECTION_SEED = 20260904

HIDDEN_WIDTH = 64
N_HIDDEN_LAYERS = 2
BATCH_SIZE = 256
MAX_STEPS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
EVAL_EVERY = 25
PATIENCE_EVALS = 10
MIN_IMPROVEMENT = 1e-4
GRAD_CLIP_NORM = 5.0
STANDARDIZE_EPS = 1e-6

VAL_PRODUCT_SHIFTS = (1, 7, 19, 31)
TEST_PRODUCT_SHIFTS = (1, 7, 19, 31, 61, 127)

# Calibration gates. These are intentionally tolerance-based rather than
# requiring exact equality because MINE is a finite-sample variational lower
# bound and can show small estimation bias.
NULL_ABS_TOL = 0.05
SCALAR_MAX_ABS_ERROR = 0.08
VECTOR_MAIN_MAX_ABS_ERROR = 0.10  # applied for gain >= 0.5
MAX_SEED_SD = 0.05
STRONG_SIGNAL_MIN_FRACTION_OF_TRUTH = 0.70
ATTENUATED_MAX_ESTIMATE = 0.10
ATTENUATED_MAX_FRACTION_OF_STRONG = 0.25


@dataclass(frozen=True)
class TrainingConfig:
    n_total: int = N_TOTAL
    n_train: int = N_TRAIN
    n_val: int = N_VAL
    n_test: int = N_TEST
    hidden_width: int = HIDDEN_WIDTH
    n_hidden_layers: int = N_HIDDEN_LAYERS
    batch_size: int = BATCH_SIZE
    max_steps: int = MAX_STEPS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    eval_every: int = EVAL_EVERY
    patience_evals: int = PATIENCE_EVALS
    min_improvement: float = MIN_IMPROVEMENT
    grad_clip_norm: float = GRAD_CLIP_NORM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic internal unit tests and exit.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Torch device. Default: cpu for calibration reproducibility; CUDA is optional.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override default experiment output directory.",
    )
    parser.add_argument(
        "--code03-manifest",
        type=Path,
        default=None,
        help="Override path to Code-03 orbit_synth_manifest.json.",
    )
    return parser.parse_args()


def project_root_from_script() -> Path:
    # Expected location: <project>/03 Code/04_validate_mi_estimator.py
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Code 02 requires torch, so run this "
            "inside the same Python environment used for Code 02."
        ) from exc
    return torch, nn


def choose_device(torch, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def analytic_scalar_gaussian_mi(rho: float) -> float:
    rho = float(rho)
    if not np.isfinite(rho) or abs(rho) >= 1.0:
        raise ValueError("rho must be finite and satisfy |rho| < 1.")
    return float(-0.5 * math.log(1.0 - rho * rho))


def analytic_vector_channel_mi(gain: float, sigma: float) -> float:
    gain = float(gain)
    sigma = float(sigma)
    if not np.isfinite(gain):
        raise ValueError("gain must be finite.")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and > 0.")
    return float(0.5 * math.log(1.0 + (gain * gain) / (sigma * sigma)))


def load_code03_channel(manifest_path: Path) -> Dict[str, float | int]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Code-03 manifest not found:\n"
            f"  {manifest_path}\n"
            "Run 03_build_orbit_synth_fmib.py successfully before Code 04."
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("code_name") != "ORBIT-FMIB CODE 03":
        raise RuntimeError("Unexpected Code-03 manifest code_name.")

    channel = manifest.get("channel", {})
    dims = manifest.get("dimensions", {})

    required_channel = ("signal_gain", "attenuated_gain", "noise_std")
    missing = [k for k in required_channel if k not in channel]
    if missing:
        raise RuntimeError(f"Code-03 manifest missing channel keys: {missing}")
    if "hidden_dim" not in dims:
        raise RuntimeError("Code-03 manifest missing dimensions.hidden_dim.")

    signal_gain = float(channel["signal_gain"])
    attenuated_gain = float(channel["attenuated_gain"])
    noise_std = float(channel["noise_std"])
    hidden_dim = int(dims["hidden_dim"])

    if signal_gain <= 0 or attenuated_gain < 0 or noise_std <= 0 or hidden_dim < 1:
        raise RuntimeError("Invalid channel parameters in Code-03 manifest.")
    if attenuated_gain >= signal_gain:
        raise RuntimeError("Expected attenuated_gain < signal_gain in Code 03.")

    return {
        "signal_gain": signal_gain,
        "attenuated_gain": attenuated_gain,
        "noise_std": noise_std,
        "hidden_dim": hidden_dim,
        "manifest_sha256": sha256_file(manifest_path),
    }


def build_split_indices(n: int, split_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n != N_TRAIN + N_VAL + N_TEST:
        raise ValueError("n must equal frozen train+val+test counts.")
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    train = perm[:N_TRAIN]
    val = perm[N_TRAIN:N_TRAIN + N_VAL]
    test = perm[N_TRAIN + N_VAL:]

    # Hard leakage gate.
    if len(np.intersect1d(train, val)) != 0:
        raise RuntimeError("Train/validation index overlap detected.")
    if len(np.intersect1d(train, test)) != 0:
        raise RuntimeError("Train/test index overlap detected.")
    if len(np.intersect1d(val, test)) != 0:
        raise RuntimeError("Validation/test index overlap detected.")
    if len(np.unique(np.concatenate([train, val, test]))) != n:
        raise RuntimeError("Split indices do not cover each sample exactly once.")

    return train, val, test


def make_scalar_gaussian(rho: float, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rho = float(rho)
    analytic_scalar_gaussian_mi(rho)  # validates rho
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    z = rng.normal(size=(n, 1))
    y = rho * x + math.sqrt(1.0 - rho * rho) * z
    return x.astype(np.float32), y.astype(np.float32)


def make_vector_channel(
    gain: float,
    sigma: float,
    hidden_dim: int,
    n: int,
    data_seed: int,
    direction_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    analytic_vector_channel_mi(gain, sigma)  # validation
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be >= 1.")

    rng = np.random.default_rng(data_seed)
    e = rng.normal(size=(n, 1))
    noise = rng.normal(size=(n, hidden_dim)) * sigma

    direction_rng = np.random.default_rng(direction_seed)
    u = direction_rng.normal(size=(hidden_dim,))
    norm = np.linalg.norm(u)
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Failed to construct finite nonzero channel direction.")
    u = u / norm

    r = gain * e * u[None, :] + noise
    return r.astype(np.float32), e.astype(np.float32), u.astype(np.float64)


def logmeanexp(torch, scores):
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    return torch.logsumexp(scores, dim=0) - math.log(scores.numel())


def dv_bound(torch, critic, u, v_joint, v_product):
    joint_scores = critic(u, v_joint)
    product_scores = critic(u, v_product)
    return joint_scores.mean() - logmeanexp(torch, product_scores)


def build_critic(torch, nn, u_dim: int, v_dim: int):
    if u_dim < 1 or v_dim < 1:
        raise ValueError("Critic input dimensions must be positive.")

    layers = []
    in_dim = u_dim + v_dim
    for _ in range(N_HIDDEN_LAYERS):
        layers.append(nn.Linear(in_dim, HIDDEN_WIDTH))
        layers.append(nn.ELU())
        in_dim = HIDDEN_WIDTH
    layers.append(nn.Linear(in_dim, 1))

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers)

        def forward(self, u, v):
            if u.ndim != 2 or v.ndim != 2:
                raise ValueError("u and v must both be rank-2 tensors.")
            if u.shape[0] != v.shape[0]:
                raise ValueError("u and v batch sizes differ.")
            return self.net(torch.cat([u, v], dim=1)).squeeze(-1)

    return Critic()


def standardize_from_train(torch, u_train, v_train, u_val, v_val, u_test, v_test):
    u_mean = u_train.mean(dim=0, keepdim=True)
    u_std = u_train.std(dim=0, unbiased=True, keepdim=True).clamp_min(STANDARDIZE_EPS)
    v_mean = v_train.mean(dim=0, keepdim=True)
    v_std = v_train.std(dim=0, unbiased=True, keepdim=True).clamp_min(STANDARDIZE_EPS)

    def apply(u, v):
        return (u - u_mean) / u_std, (v - v_mean) / v_std

    return (*apply(u_train, v_train), *apply(u_val, v_val), *apply(u_test, v_test))


def evaluate_dv(torch, critic, u, v, shifts: Sequence[int]) -> Tuple[float, float]:
    critic.eval()
    values: List[float] = []
    with torch.no_grad():
        for shift in shifts:
            if not (1 <= int(shift) < len(v)):
                raise ValueError("Evaluation shift must satisfy 1 <= shift < batch size.")
            v_product = torch.roll(v, shifts=int(shift), dims=0)
            bound = float(dv_bound(torch, critic, u, v, v_product).item())
            values.append(bound)
    return float(np.mean(values)), float(np.std(values, ddof=0))


def heldout_score_gap(torch, critic, u, v) -> float:
    """Mean critic score on true pairs minus a fixed mismatched pairing."""
    critic.eval()
    with torch.no_grad():
        joint = critic(u, v).mean()
        mismatched = critic(u, torch.roll(v, shifts=17, dims=0)).mean()
    return float((joint - mismatched).item())


def train_mine(
    torch,
    nn,
    u_np: np.ndarray,
    v_np: np.ndarray,
    critic_seed: int,
    device,
    split_indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, float | int]:
    if u_np.ndim != 2 or v_np.ndim != 2:
        raise ValueError("u_np and v_np must be rank-2 arrays.")
    if len(u_np) != len(v_np):
        raise ValueError("u_np and v_np sample counts differ.")
    if not np.all(np.isfinite(u_np)) or not np.all(np.isfinite(v_np)):
        raise ValueError("Calibration data contain NaN or infinite values.")

    train_idx, val_idx, test_idx = split_indices
    seed_everything(torch, critic_seed)

    u_all = torch.as_tensor(u_np, dtype=torch.float32)
    v_all = torch.as_tensor(v_np, dtype=torch.float32)

    u_train = u_all[train_idx].to(device)
    v_train = v_all[train_idx].to(device)
    u_val = u_all[val_idx].to(device)
    v_val = v_all[val_idx].to(device)
    u_test = u_all[test_idx].to(device)
    v_test = v_all[test_idx].to(device)

    (
        u_train,
        v_train,
        u_val,
        v_val,
        u_test,
        v_test,
    ) = standardize_from_train(
        torch, u_train, v_train, u_val, v_val, u_test, v_test
    )

    critic = build_critic(torch, nn, u_np.shape[1], v_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Separate seeded generator controls batch indices / negative shifts.
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    batch_gen = torch.Generator(device=gen_device)
    batch_gen.manual_seed(critic_seed + 100_000)

    best_val = -float("inf")
    best_state = None
    best_step = 0
    evals_without_improvement = 0
    final_step = 0

    critic.train()
    for step in range(1, MAX_STEPS + 1):
        final_step = step
        batch_idx = torch.randint(
            low=0,
            high=len(u_train),
            size=(BATCH_SIZE,),
            generator=batch_gen,
            device=device,
        )
        u_batch = u_train[batch_idx]
        v_batch = v_train[batch_idx]

        # Nonzero cyclic shift: exact row identities never remain paired in the
        # product-of-marginals minibatch.
        shift = int(
            torch.randint(
                low=1,
                high=BATCH_SIZE,
                size=(1,),
                generator=batch_gen,
                device=device,
            ).item()
        )
        v_product = torch.roll(v_batch, shifts=shift, dims=0)

        mi_lower_bound = dv_bound(torch, critic, u_batch, v_batch, v_product)
        loss = -mi_lower_bound

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite MINE loss at critic seed {critic_seed}, step {step}."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP_NORM)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"Non-finite critic gradient norm at seed {critic_seed}, step {step}."
            )
        optimizer.step()

        if step % EVAL_EVERY == 0:
            val_est, _ = evaluate_dv(torch, critic, u_val, v_val, VAL_PRODUCT_SHIFTS)
            if not np.isfinite(val_est):
                raise RuntimeError("Non-finite held-out validation MI estimate.")

            if val_est > best_val + MIN_IMPROVEMENT:
                best_val = val_est
                best_state = copy.deepcopy(critic.state_dict())
                best_step = step
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= PATIENCE_EVALS:
                    break

    if best_state is None:
        raise RuntimeError("Critic training never produced a valid validation checkpoint.")

    critic.load_state_dict(best_state)
    test_est, test_shuffle_sd = evaluate_dv(
        torch, critic, u_test, v_test, TEST_PRODUCT_SHIFTS
    )
    score_gap = heldout_score_gap(torch, critic, u_test, v_test)

    if not all(np.isfinite(x) for x in (best_val, test_est, test_shuffle_sd, score_gap)):
        raise RuntimeError("Non-finite held-out critic diagnostics.")

    return {
        "critic_seed": int(critic_seed),
        "best_step": int(best_step),
        "final_training_step": int(final_step),
        "best_val_mi": float(best_val),
        "test_mi": float(test_est),
        "test_product_shuffle_sd": float(test_shuffle_sd),
        "heldout_joint_minus_mismatch_score": float(score_gap),
    }


def run_scalar_family(torch, nn, device, split_indices) -> List[Dict]:
    rows: List[Dict] = []
    for i, rho in enumerate(SCALAR_RHOS):
        u, v = make_scalar_gaussian(rho, N_TOTAL, SCALAR_DATA_SEED + i)
        true_mi = analytic_scalar_gaussian_mi(rho)
        for critic_seed in CRITIC_SEEDS:
            result = train_mine(torch, nn, u, v, critic_seed, device, split_indices)
            rows.append(
                {
                    "family": "scalar_correlated_gaussian",
                    "condition": f"rho_{rho:.2f}",
                    "parameter_name": "rho",
                    "parameter_value": float(rho),
                    "true_mi_nats": true_mi,
                    "u_dim": 1,
                    "v_dim": 1,
                    **result,
                }
            )
    return rows


def vector_gain_grid(channel: Dict[str, float | int]) -> Tuple[float, ...]:
    # Includes Code-03 null, exact attenuated level, moderate calibration levels,
    # and exact strong-signal level.
    values = {
        0.0,
        float(channel["attenuated_gain"]),
        0.5,
        1.0,
        float(channel["signal_gain"]),
    }
    return tuple(sorted(values))


def run_vector_family(torch, nn, device, split_indices, channel) -> List[Dict]:
    rows: List[Dict] = []
    sigma = float(channel["noise_std"])
    hidden_dim = int(channel["hidden_dim"])

    for i, gain in enumerate(vector_gain_grid(channel)):
        u, v, direction = make_vector_channel(
            gain=gain,
            sigma=sigma,
            hidden_dim=hidden_dim,
            n=N_TOTAL,
            data_seed=VECTOR_DATA_SEED + i,
            direction_seed=DIRECTION_SEED,
        )
        if not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-12):
            raise RuntimeError("Vector-channel direction is not unit norm.")

        true_mi = analytic_vector_channel_mi(gain, sigma)
        for critic_seed in CRITIC_SEEDS:
            result = train_mine(torch, nn, u, v, critic_seed, device, split_indices)
            rows.append(
                {
                    "family": "code03_vector_gaussian_channel",
                    "condition": f"gain_{gain:.3f}",
                    "parameter_name": "gain",
                    "parameter_value": float(gain),
                    "true_mi_nats": true_mi,
                    "u_dim": hidden_dim,
                    "v_dim": 1,
                    **result,
                }
            )
    return rows


def summarize_seed_results(seed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = seed_df.groupby(
        [
            "family",
            "condition",
            "parameter_name",
            "parameter_value",
            "true_mi_nats",
            "u_dim",
            "v_dim",
        ],
        sort=False,
        as_index=False,
    )
    summary = grouped.agg(
        estimate_mean=("test_mi", "mean"),
        estimate_sd_across_seeds=("test_mi", "std"),
        estimate_min=("test_mi", "min"),
        estimate_max=("test_mi", "max"),
        mean_best_val=("best_val_mi", "mean"),
        mean_best_step=("best_step", "mean"),
        mean_product_shuffle_sd=("test_product_shuffle_sd", "mean"),
        mean_joint_minus_mismatch_score=("heldout_joint_minus_mismatch_score", "mean"),
    )
    summary["estimate_sd_across_seeds"] = summary["estimate_sd_across_seeds"].fillna(0.0)
    summary["abs_error"] = np.abs(summary["estimate_mean"] - summary["true_mi_nats"])
    summary["relative_to_truth"] = np.where(
        summary["true_mi_nats"] > 0,
        summary["estimate_mean"] / summary["true_mi_nats"],
        np.nan,
    )
    return summary


def _strictly_increasing(values: Iterable[float], tolerance: float = 1e-6) -> bool:
    vals = list(map(float, values))
    return all(vals[i + 1] > vals[i] + tolerance for i in range(len(vals) - 1))


def audit_calibration(summary: pd.DataFrame, channel) -> Dict[str, bool | float | str]:
    gates: Dict[str, bool | float | str] = {}

    scalar = summary[summary["family"] == "scalar_correlated_gaussian"].sort_values(
        "parameter_value"
    )
    vector = summary[summary["family"] == "code03_vector_gaussian_channel"].sort_values(
        "parameter_value"
    )

    if len(scalar) != len(SCALAR_RHOS):
        raise RuntimeError("Unexpected scalar calibration condition count.")
    if len(vector) != len(vector_gain_grid(channel)):
        raise RuntimeError("Unexpected vector calibration condition count.")

    scalar_null = float(scalar.loc[np.isclose(scalar["parameter_value"], 0.0), "estimate_mean"].iloc[0])
    vector_null = float(vector.loc[np.isclose(vector["parameter_value"], 0.0), "estimate_mean"].iloc[0])

    gates["scalar_null_abs_le_0p05"] = abs(scalar_null) <= NULL_ABS_TOL
    gates["vector_null_abs_le_0p05"] = abs(vector_null) <= NULL_ABS_TOL
    gates["scalar_monotonic"] = _strictly_increasing(scalar["estimate_mean"].tolist())
    gates["vector_monotonic"] = _strictly_increasing(vector["estimate_mean"].tolist())

    scalar_max_error = float(scalar["abs_error"].max())
    gates["scalar_max_abs_error"] = scalar_max_error
    gates["scalar_error_le_tolerance"] = scalar_max_error <= SCALAR_MAX_ABS_ERROR

    vector_main = vector[vector["parameter_value"] >= 0.5 - 1e-12]
    vector_main_max_error = float(vector_main["abs_error"].max())
    gates["vector_main_max_abs_error"] = vector_main_max_error
    gates["vector_main_error_le_tolerance"] = vector_main_max_error <= VECTOR_MAIN_MAX_ABS_ERROR

    max_seed_sd = float(summary["estimate_sd_across_seeds"].max())
    gates["max_seed_sd"] = max_seed_sd
    gates["seed_stability"] = max_seed_sd <= MAX_SEED_SD

    scalar_strong = scalar.iloc[-1]
    vector_strong = vector.iloc[-1]
    gates["scalar_strong_signal_fraction"] = float(scalar_strong["relative_to_truth"])
    gates["vector_strong_signal_fraction"] = float(vector_strong["relative_to_truth"])
    gates["scalar_strong_signal_gate"] = (
        float(scalar_strong["estimate_mean"])
        >= STRONG_SIGNAL_MIN_FRACTION_OF_TRUTH * float(scalar_strong["true_mi_nats"])
    )
    gates["vector_strong_signal_gate"] = (
        float(vector_strong["estimate_mean"])
        >= STRONG_SIGNAL_MIN_FRACTION_OF_TRUTH * float(vector_strong["true_mi_nats"])
    )

    atten = float(channel["attenuated_gain"])
    atten_row = vector.loc[np.isclose(vector["parameter_value"], atten)]
    if len(atten_row) != 1:
        raise RuntimeError("Could not uniquely locate Code-03 attenuated gain calibration row.")
    atten_est = float(atten_row["estimate_mean"].iloc[0])
    strong_est = float(vector_strong["estimate_mean"])
    gates["code03_attenuated_estimate"] = atten_est
    gates["code03_attenuated_is_near_floor"] = atten_est <= ATTENUATED_MAX_ESTIMATE
    gates["code03_attenuated_vs_strong_gate"] = (
        atten_est <= ATTENUATED_MAX_FRACTION_OF_STRONG * strong_est
    )

    # Every held-out estimate and diagnostic must be finite.
    numeric_cols = [
        "estimate_mean",
        "estimate_sd_across_seeds",
        "mean_best_val",
        "mean_best_step",
        "mean_product_shuffle_sd",
        "mean_joint_minus_mismatch_score",
    ]
    gates["all_summary_diagnostics_finite"] = bool(
        np.all(np.isfinite(summary[numeric_cols].to_numpy(dtype=float)))
    )

    required_boolean_gates = [
        "scalar_null_abs_le_0p05",
        "vector_null_abs_le_0p05",
        "scalar_monotonic",
        "vector_monotonic",
        "scalar_error_le_tolerance",
        "vector_main_error_le_tolerance",
        "seed_stability",
        "scalar_strong_signal_gate",
        "vector_strong_signal_gate",
        "code03_attenuated_is_near_floor",
        "code03_attenuated_vs_strong_gate",
        "all_summary_diagnostics_finite",
    ]
    gates["overall_pass"] = all(bool(gates[name]) for name in required_boolean_gates)
    return gates


def self_test() -> None:
    torch, nn = import_torch()

    # Analytic formulas.
    assert np.isclose(analytic_scalar_gaussian_mi(0.0), 0.0)
    assert np.isclose(analytic_scalar_gaussian_mi(0.5), 0.14384103622589045)
    assert np.isclose(analytic_vector_channel_mi(1.5, 1.0), 0.5893274981708231)
    assert np.isclose(analytic_vector_channel_mi(0.15, 1.0), 0.011125304467409862)

    # Split leakage gate.
    tr, va, te = build_split_indices(N_TOTAL, SPLIT_SEED)
    assert len(tr) == N_TRAIN and len(va) == N_VAL and len(te) == N_TEST
    assert len(np.intersect1d(tr, va)) == 0
    assert len(np.intersect1d(tr, te)) == 0
    assert len(np.intersect1d(va, te)) == 0

    # Data generators and known empirical shape/correlation behavior.
    x0, y0 = make_scalar_gaussian(0.0, 4096, 123)
    x8, y8 = make_scalar_gaussian(0.8, 4096, 124)
    assert x0.shape == y0.shape == (4096, 1)
    assert abs(float(np.corrcoef(x0[:, 0], y0[:, 0])[0, 1])) < 0.08
    assert float(np.corrcoef(x8[:, 0], y8[:, 0])[0, 1]) > 0.72

    r, e, u = make_vector_channel(1.5, 1.0, 16, 1024, 125, 126)
    assert r.shape == (1024, 16)
    assert e.shape == (1024, 1)
    assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-12)

    # Zero critic must give exactly zero DV bound regardless of pairing.
    class ZeroCritic(nn.Module):
        def forward(self, u_, v_):
            return torch.zeros(u_.shape[0], dtype=u_.dtype, device=u_.device)

    u_t = torch.as_tensor(x8[:128])
    v_t = torch.as_tensor(y8[:128])
    zero = float(
        dv_bound(torch, ZeroCritic(), u_t, v_t, torch.roll(v_t, 1, 0)).item()
    )
    assert abs(zero) <= 1e-12

    # Critic shape contract.
    critic = build_critic(torch, nn, 16, 1)
    out = critic(torch.zeros(8, 16), torch.zeros(8, 1))
    assert tuple(out.shape) == (8,)

    print("SELF-TEST: PASS")
    print("Verified: analytic Gaussian MI formulas, disjoint train/val/test split,")
    print("          scalar/vector channel generators, stable DV log-mean-exp objective,")
    print("          zero-critic null identity, and critic shape contract.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    torch, nn = import_torch()
    device = choose_device(torch, args.device)
    if device.type == "cpu":
        # Tiny critics are faster and more reproducible without large BLAS thread pools.
        torch.set_num_threads(1)

    root = project_root_from_script()
    code03_manifest = (
        args.code03_manifest
        if args.code03_manifest is not None
        else root / "04 Experiments" / "EXP01_ORBIT_Synth" / "orbit_synth_manifest.json"
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else root / "04 Experiments" / "EXP03_MI"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"{CODE_NAME} — MINE / FLOWNIB ESTIMATOR CALIBRATION")
    print("=" * 76)
    print()

    channel = load_code03_channel(code03_manifest)
    print("Code-03 calibration contract: PASS")
    print(f"  hidden dimension: {channel['hidden_dim']}")
    print(f"  strong signal gain: {channel['signal_gain']:.3f}")
    print(f"  attenuated gain: {channel['attenuated_gain']:.3f}")
    print(f"  Gaussian noise sigma: {channel['noise_std']:.3f}")
    print(
        "  analytic strong MI: "
        f"{analytic_vector_channel_mi(float(channel['signal_gain']), float(channel['noise_std'])):.6f} nats"
    )
    print(
        "  analytic attenuated MI: "
        f"{analytic_vector_channel_mi(float(channel['attenuated_gain']), float(channel['noise_std'])):.6f} nats"
    )
    print()

    print(f"Requested device: {args.device}")
    print(f"Validation device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == "cpu":
        print(f"Torch CPU threads: {torch.get_num_threads()}")
    print()

    split_indices = build_split_indices(N_TOTAL, SPLIT_SEED)
    print("Held-out calibration split: PASS")
    print(f"  train / val / test: {N_TRAIN:,} / {N_VAL:,} / {N_TEST:,}")
    print("  hyperparameter/early-stop selection: validation only")
    print("  final MI reporting: test only")
    print()

    print("MINE protocol")
    print(f"  critic: {N_HIDDEN_LAYERS} hidden layers x {HIDDEN_WIDTH} ELU units")
    print(f"  critic seeds: {', '.join(map(str, CRITIC_SEEDS))}")
    print(f"  batch size: {BATCH_SIZE}")
    print(f"  maximum steps: {MAX_STEPS}")
    print(f"  learning rate: {LEARNING_RATE:g}")
    print("  product-of-marginals: nonzero within-batch cyclic shuffle")
    print("  objective: raw Donsker-Varadhan/MINE lower bound (nats)")
    print("  representation/target standardization: train statistics only")
    print()

    rows: List[Dict] = []

    print("Running scalar correlated-Gaussian calibration ...")
    scalar_rows = run_scalar_family(torch, nn, device, split_indices)
    rows.extend(scalar_rows)
    scalar_tmp = summarize_seed_results(pd.DataFrame(scalar_rows))
    for _, row in scalar_tmp.sort_values("parameter_value").iterrows():
        print(
            f"  rho={row['parameter_value']:.2f}: "
            f"true={row['true_mi_nats']:.6f}, "
            f"MINE={row['estimate_mean']:.6f} +/- {row['estimate_sd_across_seeds']:.6f}"
        )
    print()

    print("Running Code-03-matched vector Gaussian-channel calibration ...")
    vector_rows = run_vector_family(torch, nn, device, split_indices, channel)
    rows.extend(vector_rows)
    vector_tmp = summarize_seed_results(pd.DataFrame(vector_rows))
    for _, row in vector_tmp.sort_values("parameter_value").iterrows():
        print(
            f"  gain={row['parameter_value']:.3f}: "
            f"true={row['true_mi_nats']:.6f}, "
            f"MINE={row['estimate_mean']:.6f} +/- {row['estimate_sd_across_seeds']:.6f}"
        )
    print()

    seed_df = pd.DataFrame(rows)
    summary_df = summarize_seed_results(seed_df)
    gates = audit_calibration(summary_df, channel)

    print("Estimator acceptance gates")
    print(f"  scalar null |MI| <= {NULL_ABS_TOL:.2f}: {'PASS' if gates['scalar_null_abs_le_0p05'] else 'FAIL'}")
    print(f"  vector null |MI| <= {NULL_ABS_TOL:.2f}: {'PASS' if gates['vector_null_abs_le_0p05'] else 'FAIL'}")
    print(f"  scalar dependence ordering: {'PASS' if gates['scalar_monotonic'] else 'FAIL'}")
    print(f"  vector dependence ordering: {'PASS' if gates['vector_monotonic'] else 'FAIL'}")
    print(
        f"  scalar max absolute error <= {SCALAR_MAX_ABS_ERROR:.2f}: "
        f"{'PASS' if gates['scalar_error_le_tolerance'] else 'FAIL'} "
        f"(observed {gates['scalar_max_abs_error']:.6f})"
    )
    print(
        f"  vector main-signal max absolute error <= {VECTOR_MAIN_MAX_ABS_ERROR:.2f}: "
        f"{'PASS' if gates['vector_main_error_le_tolerance'] else 'FAIL'} "
        f"(observed {gates['vector_main_max_abs_error']:.6f})"
    )
    print(
        f"  max across-seed SD <= {MAX_SEED_SD:.2f}: "
        f"{'PASS' if gates['seed_stability'] else 'FAIL'} "
        f"(observed {gates['max_seed_sd']:.6f})"
    )
    print(
        "  Code-03 attenuated channel remains near estimator floor: "
        f"{'PASS' if gates['code03_attenuated_is_near_floor'] else 'FAIL'}"
    )
    print(
        "  Code-03 attenuated estimate << strong estimate: "
        f"{'PASS' if gates['code03_attenuated_vs_strong_gate'] else 'FAIL'}"
    )
    print()

    seed_path = output_dir / "mine_seed_results.csv"
    summary_path = output_dir / "mine_condition_summary.csv"
    manifest_path = output_dir / "mine_validation_manifest.json"

    seed_df.to_csv(seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "purpose": "Pre-biological validation of the MINE/DV mutual-information measurement engine.",
        "flow_nib_relation": {
            "shared_core": "MINE-style DV lower bound; frozen upstream representation; joint versus shuffled product-of-marginals pairs.",
            "not_implemented_here": [
                "FlowNIB alpha(t) joint two-critic schedule",
                "effective-dimension MI normalization",
                "context-side I(Q_k;R_l,k) validation",
            ],
            "reason": "Phase-1 primary confirmatory quantity is raw target-side I(R_l,k;E_k).",
        },
        "device": str(device),
        "torch_version": str(torch.__version__),
        "training_config": asdict(TrainingConfig()),
        "critic_seeds": list(CRITIC_SEEDS),
        "data_seeds": {
            "scalar_base": SCALAR_DATA_SEED,
            "vector_base": VECTOR_DATA_SEED,
            "split": SPLIT_SEED,
            "direction": DIRECTION_SEED,
        },
        "splits": {
            "train": N_TRAIN,
            "val": N_VAL,
            "test": N_TEST,
            "selection": "validation only",
            "reported_estimate": "held-out test only",
        },
        "scalar_calibration": {
            "rho_values": list(SCALAR_RHOS),
            "analytic_formula_nats": "-0.5*log(1-rho^2)",
        },
        "code03_vector_calibration": {
            "manifest_path": str(code03_manifest.resolve()),
            "manifest_sha256": channel["manifest_sha256"],
            "hidden_dim": int(channel["hidden_dim"]),
            "noise_std": float(channel["noise_std"]),
            "gain_values": list(vector_gain_grid(channel)),
            "signal_gain": float(channel["signal_gain"]),
            "attenuated_gain": float(channel["attenuated_gain"]),
            "analytic_formula_nats": "0.5*log(1+gain^2/noise_std^2)",
        },
        "acceptance_thresholds": {
            "null_abs_tolerance": NULL_ABS_TOL,
            "scalar_max_abs_error": SCALAR_MAX_ABS_ERROR,
            "vector_main_max_abs_error": VECTOR_MAIN_MAX_ABS_ERROR,
            "max_seed_sd": MAX_SEED_SD,
            "strong_signal_min_fraction_of_truth": STRONG_SIGNAL_MIN_FRACTION_OF_TRUTH,
            "attenuated_max_estimate": ATTENUATED_MAX_ESTIMATE,
            "attenuated_max_fraction_of_strong": ATTENUATED_MAX_FRACTION_OF_STRONG,
        },
        "acceptance_results": gates,
        "overall_pass": bool(gates["overall_pass"]),
        "interpretation_limits": [
            "MINE values are finite-sample variational lower-bound estimates, not exact MI.",
            "Small negative estimates can occur near the independence floor and are not clipped.",
            "This calibration does not establish validity for arbitrary unseen dimensionalities/distributions.",
            "Code 05 must still demonstrate correct order recovery and bottleneck localization on the actual Code-03 synthetic tensors.",
            "No biological conclusion is permitted if Code 04 or Code 05 fails.",
        ],
    }

    # Add hashes after the CSVs are fully written.
    manifest["file_sha256"] = {
        seed_path.name: sha256_file(seed_path),
        summary_path.name: sha256_file(summary_path),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print("Saved:")
    print(f"  {seed_path}")
    print(f"  {summary_path}")
    print(f"  {manifest_path}")
    print()

    if not gates["overall_pass"]:
        raise RuntimeError(
            "MINE calibration failed one or more pre-specified acceptance gates. "
            "Do NOT proceed to Code 05/GB1 interpretation."
        )

    print(f"{CODE_NAME}: PASS")


if __name__ == "__main__":
    main()
