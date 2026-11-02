#!/usr/bin/env python3
"""
ORBIT-FMIB CODE 03 — Build ORBIT-Synth FMIB calibration data.

Purpose
-------
Create the controlled synthetic data used to validate the ORBIT-FMIB
measurement pipeline *before* any biological GB1 mutual-information result is
interpreted.

This script ports the core Walsh conventions and ORBIT-Synth controls from the
previous ORBIT codebase and extends them with layerwise vector representations
whose order-specific target information and bottleneck locations are known by
construction.

Scientific design
-----------------
1. Structural ORBIT controls
   - pure orders k=1,2,3,4
   - exact 4-way parity
   - controlled order-1/order-4 mixtures

2. FMIB target ensemble
   For synthetic context C and subset S, target Walsh coefficient E_{C,S} is
   sampled from N(0,1) for all nonconstant subsets.  Each context therefore
   defines a complete d=4 synthetic landscape.

3. Layerwise representation channel
   For order k and synthetic stage l,

       R_{l,C,S} = a_{l,k} E_{C,S} u_k + sigma * eta_{l,C,S},

   where u_k is a fixed unit vector, eta ~ N(0,I), and sigma > 0.
   The corresponding pointwise representation H_l(x) is obtained by inverse
   Walsh reconstruction.  Reapplying the ORBIT Walsh transform must recover R.

   Because E ~ N(0,1), u_k is unit norm, and noise is isotropic Gaussian,

       I(E; R_l,k) = 0.5 * log(1 + a_{l,k}^2 / sigma^2)  [nats].

   Thus Code 05 can compare neural MI estimates against a known finite target.

4. Two bottleneck scenarios
   - staggered: k=1,2,3,4 are attenuated at different known stages
   - selective_ho: k=1,2 are retained while k=3,4 are attenuated together

No neural MI estimator is trained here.  No ESM model or GB1 data are used.

Outputs (default)
-----------------
04 Experiments/EXP01_ORBIT_Synth/
    orbit_synth_controls.npz
    fmib_targets.npz
    fmib_staggered_bottleneck.npz
    fmib_selective_ho_bottleneck.npz
    orbit_synth_summary.csv
    orbit_synth_manifest.json

Usage
-----
    python "03 Code\\03_build_orbit_synth_fmib.py" --self-test
    python "03 Code\\03_build_orbit_synth_fmib.py"

The script creates the output directory if it does not already exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


# =============================================================================
# Frozen Phase-1 synthetic configuration
# =============================================================================

CODE_NAME = "ORBIT-FMIB CODE 03"
SCHEMA_VERSION = 1
D = 4
N_VERTICES = 1 << D
N_STAGES = 8
HIDDEN_DIM = 16
N_CONTEXTS = 4096
N_TRAIN = 3072
N_VAL = 512
N_TEST = 512
TARGET_SEED = 314159
DIRECTION_SEED = 271828
REPRESENTATION_SEED_BASE = 161803
SPLIT_SEED = 57721
CONTROL_SEED = 42
NOISE_STD = 1.0
SIGNAL_GAIN = 1.5
ATTENUATED_GAIN = 0.15
FLOAT_DTYPE = np.float32

# The stage index is the first LOW-information stage after the drop.
STAGGERED_DROP_STAGE = {
    1: 3,
    2: 4,
    3: 5,
    4: 6,
}
SELECTIVE_HO_DROP_STAGE = {
    1: None,
    2: None,
    3: 4,
    4: 4,
}

MIXTURE_C_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# =============================================================================
# Previous-ORBIT Walsh core, ported with the same convention
# =============================================================================


def enumerate_points(d: int) -> np.ndarray:
    """Return all points in {-1,+1}^d using ORBIT bit ordering."""
    if not isinstance(d, int) or d < 1:
        raise ValueError("d must be a positive integer.")
    n = 1 << d
    idx = np.arange(n, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(d, dtype=np.int64)) & 1
    return np.where(bits == 1, 1.0, -1.0).astype(np.float64)


def chi_matrix(d: int) -> np.ndarray:
    """Return Walsh basis X with X[x,S] = chi_S(x)."""
    points = enumerate_points(d)
    n = 1 << d
    X = np.ones((n, n), dtype=np.float64)
    for subset in range(n):
        positions = [b for b in range(d) if (subset >> b) & 1]
        if positions:
            X[:, subset] = np.prod(points[:, positions], axis=1)
    return X


def walsh_coefficients(values: np.ndarray) -> np.ndarray:
    """ORBIT Walsh transform for scalar or vector-valued complete cubes."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in (1, 2):
        raise ValueError("values must have shape (2^d,) or (2^d,m).")
    n = values.shape[0]
    d = int(round(math.log2(n)))
    if (1 << d) != n:
        raise ValueError("First dimension must equal 2^d.")
    X = chi_matrix(d)
    return (X.T @ values) / n


def inverse_walsh(coefficients: np.ndarray) -> np.ndarray:
    """Inverse ORBIT Walsh transform."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim not in (1, 2):
        raise ValueError("coefficients must have shape (2^d,) or (2^d,m).")
    n = coefficients.shape[0]
    d = int(round(math.log2(n)))
    if (1 << d) != n:
        raise ValueError("First dimension must equal 2^d.")
    return chi_matrix(d) @ coefficients


def order_of(subset: int) -> int:
    if int(subset) < 0:
        raise ValueError("Subset bitmask must be nonnegative.")
    return int(subset).bit_count()


def subsets_by_order(d: int) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = {k: [] for k in range(d + 1)}
    for subset in range(1 << d):
        groups[order_of(subset)].append(subset)
    return groups


def normalized_order_energy(coefficients: np.ndarray, d: int) -> Dict[int, float]:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (1 << d,):
        raise ValueError(f"Expected coefficients shape ({1 << d},).")
    groups = subsets_by_order(d)
    raw = {
        k: float(np.sum(coefficients[groups[k]] ** 2))
        for k in range(1, d + 1)
    }
    total = float(sum(raw.values()))
    if total <= 0.0:
        return {k: float("nan") for k in range(1, d + 1)}
    return {k: raw[k] / total for k in range(1, d + 1)}


# =============================================================================
# Previous-ORBIT structural controls
# =============================================================================


def _normalized_random_vector(rng: np.random.Generator, length: int) -> np.ndarray:
    values = rng.normal(size=length)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Failed to generate a finite nonzero coefficient vector.")
    return values / norm


def make_pure_order_landscape(d: int, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if k < 1 or k > d:
        raise ValueError("k must satisfy 1 <= k <= d.")
    groups = subsets_by_order(d)
    eps = np.zeros(1 << d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    eps[groups[k]] = _normalized_random_vector(rng, len(groups[k]))
    return eps, inverse_walsh(eps)


def make_parity_landscape(d: int) -> Tuple[np.ndarray, np.ndarray]:
    eps = np.zeros(1 << d, dtype=np.float64)
    eps[(1 << d) - 1] = 1.0
    return eps, inverse_walsh(eps)


def make_low_high_mixture(
    d: int,
    C: float,
    low_order: int,
    high_order: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= C <= 1.0):
        raise ValueError("C must lie in [0,1].")
    if low_order == high_order:
        raise ValueError("low_order and high_order must differ.")
    groups = subsets_by_order(d)
    rng = np.random.default_rng(seed)
    low = np.zeros(1 << d, dtype=np.float64)
    high = np.zeros(1 << d, dtype=np.float64)
    low[groups[low_order]] = _normalized_random_vector(rng, len(groups[low_order]))
    high[groups[high_order]] = _normalized_random_vector(rng, len(groups[high_order]))
    eps = math.sqrt(1.0 - C) * low + math.sqrt(C) * high
    return eps, inverse_walsh(eps)


# =============================================================================
# FMIB synthetic ensemble
# =============================================================================


def build_subset_orders(d: int) -> np.ndarray:
    return np.array([order_of(s) for s in range(1 << d)], dtype=np.int8)


def build_context_splits(n_contexts: int, seed: int) -> np.ndarray:
    """Context-level train/val/test split.  No context crosses partitions."""
    if n_contexts != (N_TRAIN + N_VAL + N_TEST):
        raise ValueError(
            "Current frozen split sizes require n_contexts == "
            f"{N_TRAIN + N_VAL + N_TEST}. Got {n_contexts}."
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_contexts)
    split = np.empty(n_contexts, dtype=np.int8)
    split[perm[:N_TRAIN]] = 0
    split[perm[N_TRAIN:N_TRAIN + N_VAL]] = 1
    split[perm[N_TRAIN + N_VAL:]] = 2
    return split


def split_name(code: int) -> str:
    return {0: "train", 1: "val", 2: "test"}[int(code)]


def build_targets(
    n_contexts: int,
    d: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build target Walsh coefficients E_{C,S} and corresponding landscapes.

    Nonconstant coefficients are iid N(0,1); order-0 is fixed at zero.
    This yields a pooled Gaussian E distribution for every order, including k=4.
    """
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n_contexts, 1 << d)).astype(np.float64)
    E[:, 0] = 0.0
    X = chi_matrix(d)
    fitness = E @ X.T
    return E, fitness


def build_order_directions(
    hidden_dim: int,
    d: int,
    seed: int,
) -> np.ndarray:
    """Generate deterministic orthonormal unit directions u_k, k=1..d."""
    if hidden_dim < d:
        raise ValueError("hidden_dim must be at least d.")
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(hidden_dim, d))
    Q, _ = np.linalg.qr(A)
    directions = Q[:, :d].T.astype(np.float64)
    gram = directions @ directions.T
    if not np.allclose(gram, np.eye(d), rtol=1e-12, atol=1e-12):
        raise RuntimeError("Order directions are not orthonormal.")
    return directions


def build_gain_schedule(
    drop_stage_by_order: Mapping[int, int | None],
    n_stages: int,
    signal_gain: float,
    attenuated_gain: float,
    d: int,
) -> np.ndarray:
    """Return gain[l,k-1] for stage l and order k."""
    gains = np.full((n_stages, d), float(signal_gain), dtype=np.float64)
    for k in range(1, d + 1):
        stage = drop_stage_by_order.get(k)
        if stage is None:
            continue
        if not (1 <= int(stage) < n_stages):
            raise ValueError(f"Invalid drop stage {stage} for order {k}.")
        gains[int(stage):, k - 1] = float(attenuated_gain)
    return gains


def analytic_mi_from_gains(gains: np.ndarray, noise_std: float) -> np.ndarray:
    """Analytic I(E;R) in nats for the scalar-Gaussian signal channel."""
    if noise_std <= 0.0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and > 0.")
    gains = np.asarray(gains, dtype=np.float64)
    return 0.5 * np.log1p((gains / float(noise_std)) ** 2)


def expected_drop_from_analytic_mi(analytic_mi: np.ndarray) -> Dict[int, int | None]:
    """Locate the largest negative adjacent-stage MI change for each order."""
    out: Dict[int, int | None] = {}
    for k_idx in range(analytic_mi.shape[1]):
        delta = np.diff(analytic_mi[:, k_idx])
        j = int(np.argmin(delta))
        if delta[j] < -1e-12:
            out[k_idx + 1] = j + 1
        else:
            out[k_idx + 1] = None
    return out


def build_layerwise_representations(
    E: np.ndarray,
    subset_orders: np.ndarray,
    gains: np.ndarray,
    directions: np.ndarray,
    noise_std: float,
    seed: int,
    dtype: np.dtype = FLOAT_DTYPE,
) -> np.ndarray:
    """
    Build H[C,l,x,m] by constructing order-specific Walsh coefficients and
    inverse-transforming them to pointwise representations.
    """
    E = np.asarray(E, dtype=np.float64)
    subset_orders = np.asarray(subset_orders, dtype=np.int8)
    gains = np.asarray(gains, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)

    n_contexts, n_subsets = E.shape
    if n_subsets != N_VERTICES:
        raise ValueError(f"Expected {N_VERTICES} Walsh subsets.")
    if subset_orders.shape != (n_subsets,):
        raise ValueError("subset_orders has incorrect shape.")
    if gains.shape != (N_STAGES, D):
        raise ValueError(f"gains must have shape ({N_STAGES},{D}).")
    if directions.shape != (D, HIDDEN_DIM):
        raise ValueError(f"directions must have shape ({D},{HIDDEN_DIM}).")

    X = chi_matrix(D)
    rng = np.random.default_rng(seed)
    H = np.empty(
        (n_contexts, N_STAGES, N_VERTICES, HIDDEN_DIM),
        dtype=dtype,
    )

    for layer in range(N_STAGES):
        # Isotropic coefficient-space noise produces the finite Gaussian MI
        # calibration target.  It is independent across contexts, subsets,
        # dimensions, and stages.
        coeff = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=(n_contexts, N_VERTICES, HIDDEN_DIM),
        ).astype(np.float64)

        # Order-0 is a nuisance mean component independent of E.
        # It remains present to ensure that nonconstant ORBIT extraction does
        # not accidentally rely on zero-mean pointwise representations.
        for subset in range(1, N_VERTICES):
            k = int(subset_orders[subset])
            coeff[:, subset, :] += (
                gains[layer, k - 1]
                * E[:, subset, None]
                * directions[k - 1][None, :]
            )

        # inverse Walsh: H_C(x,:) = sum_S R_C,S chi_S(x)
        stage_H = np.einsum("xs,csm->cxm", X, coeff, optimize=True)
        H[:, layer, :, :] = stage_H.astype(dtype, copy=False)

    if not np.all(np.isfinite(H)):
        raise RuntimeError("Generated representation tensor contains non-finite values.")
    return H


# =============================================================================
# Audits
# =============================================================================


def audit_walsh_core(d: int = D) -> None:
    X = chi_matrix(d)
    n = 1 << d
    if not np.allclose(X.T @ X, n * np.eye(n), rtol=1e-12, atol=1e-12):
        raise RuntimeError("Walsh orthogonality audit failed: X.T @ X.")
    if not np.allclose(X @ X.T, n * np.eye(n), rtol=1e-12, atol=1e-12):
        raise RuntimeError("Walsh orthogonality audit failed: X @ X.T.")

    rng = np.random.default_rng(20260901)
    f = rng.normal(size=n)
    if not np.allclose(inverse_walsh(walsh_coefficients(f)), f, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Scalar Walsh reconstruction audit failed.")

    H = rng.normal(size=(n, 7))
    if not np.allclose(inverse_walsh(walsh_coefficients(H)), H, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Vector Walsh reconstruction audit failed.")


def audit_structural_controls(
    controls: Mapping[str, Tuple[np.ndarray, np.ndarray]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, (eps_true, fitness) in controls.items():
        eps_rec = walsh_coefficients(fitness)
        if not np.allclose(eps_true, eps_rec, rtol=1e-11, atol=1e-11):
            raise RuntimeError(f"Structural control reconstruction failed: {name}")
        energy = normalized_order_energy(eps_rec, D)

        if name.startswith("pure_order_"):
            k = int(name.rsplit("_", 1)[1])
            for j in range(1, D + 1):
                expected = 1.0 if j == k else 0.0
                if not np.isclose(energy[j], expected, atol=1e-11):
                    raise RuntimeError(f"Pure-order energy audit failed: {name}, k={j}")
        elif name == "parity":
            for j in range(1, D + 1):
                expected = 1.0 if j == D else 0.0
                if not np.isclose(energy[j], expected, atol=1e-11):
                    raise RuntimeError("Parity energy audit failed.")
        elif name.startswith("mixture_C_"):
            C = float(name.split("_C_")[1])
            if not np.isclose(energy[1], 1.0 - C, atol=1e-11):
                raise RuntimeError(f"Mixture low-order energy audit failed: {name}")
            if not np.isclose(energy[D], C, atol=1e-11):
                raise RuntimeError(f"Mixture high-order energy audit failed: {name}")
            for j in (2, 3):
                if not np.isclose(energy[j], 0.0, atol=1e-11):
                    raise RuntimeError(f"Mixture nuisance-order audit failed: {name}, k={j}")

        rows.append({
            "control": name,
            **{f"energy_k{k}": float(energy[k]) for k in range(1, D + 1)},
        })
    return rows


def audit_targets(E: np.ndarray, fitness: np.ndarray) -> None:
    if E.shape != (N_CONTEXTS, N_VERTICES):
        raise RuntimeError(f"Unexpected target coefficient shape: {E.shape}")
    if fitness.shape != (N_CONTEXTS, N_VERTICES):
        raise RuntimeError(f"Unexpected target fitness shape: {fitness.shape}")
    if not np.all(E[:, 0] == 0.0):
        raise RuntimeError("Target order-0 coefficients must be exactly zero.")
    if not np.all(np.isfinite(E)) or not np.all(np.isfinite(fitness)):
        raise RuntimeError("Targets contain non-finite values.")

    # Exact batched inverse/forward Walsh consistency.
    X = chi_matrix(D)
    recovered = (fitness @ X) / N_VERTICES
    if not np.allclose(recovered, E, rtol=1e-11, atol=1e-11):
        raise RuntimeError("Synthetic target Walsh round-trip failed.")

    # Every order must have non-degenerate pooled target variation so MINE has
    # a real target variable, including the single k=4 subset per context.
    groups = subsets_by_order(D)
    for k in range(1, D + 1):
        pooled = E[:, groups[k]].reshape(-1)
        sd = float(np.std(pooled))
        if not (0.90 < sd < 1.10):
            raise RuntimeError(f"Unexpected pooled target SD for order {k}: {sd:.6f}")


def audit_splits(split_codes: np.ndarray) -> None:
    if split_codes.shape != (N_CONTEXTS,):
        raise RuntimeError("Split vector has wrong shape.")
    counts = {int(c): int(np.sum(split_codes == c)) for c in (0, 1, 2)}
    expected = {0: N_TRAIN, 1: N_VAL, 2: N_TEST}
    if counts != expected:
        raise RuntimeError(f"Split-count audit failed. Observed {counts}, expected {expected}.")


def recover_coefficients_from_H(H_stage: np.ndarray) -> np.ndarray:
    """Batched Walsh transform H[C,x,m] -> R[C,S,m]."""
    H_stage = np.asarray(H_stage, dtype=np.float64)
    if H_stage.ndim != 3 or H_stage.shape[1] != N_VERTICES:
        raise ValueError("H_stage must have shape (contexts, 16, hidden_dim).")
    X = chi_matrix(D)
    return np.einsum("sx,cxm->csm", X.T, H_stage, optimize=True) / N_VERTICES


def audit_representation_scenario(
    H: np.ndarray,
    E: np.ndarray,
    subset_orders: np.ndarray,
    gains: np.ndarray,
    directions: np.ndarray,
    analytic_mi: np.ndarray,
    expected_drop: Mapping[int, int | None],
    scenario_name: str,
) -> None:
    expected_shape = (N_CONTEXTS, N_STAGES, N_VERTICES, HIDDEN_DIM)
    if H.shape != expected_shape:
        raise RuntimeError(f"{scenario_name}: H shape {H.shape}, expected {expected_shape}.")
    if H.dtype != FLOAT_DTYPE:
        raise RuntimeError(f"{scenario_name}: expected H dtype {FLOAT_DTYPE}, got {H.dtype}.")
    if not np.all(np.isfinite(H)):
        raise RuntimeError(f"{scenario_name}: H contains non-finite values.")
    if analytic_mi.shape != (N_STAGES, D):
        raise RuntimeError(f"{scenario_name}: analytic MI has wrong shape.")

    found_drop = expected_drop_from_analytic_mi(analytic_mi)
    if dict(found_drop) != dict(expected_drop):
        raise RuntimeError(
            f"{scenario_name}: analytic bottleneck localization mismatch. "
            f"Found {found_drop}, expected {dict(expected_drop)}."
        )

    # Validate that the inverse/forward Walsh path recovers the coefficient
    # channel and that empirical projection follows the prescribed gain.
    # Audit a deterministic subset of contexts and all stages/orders.
    audit_idx = np.array([0, 1, 7, 31, 127, 511, 1023, 2047, 4095], dtype=np.int64)
    groups = subsets_by_order(D)

    for layer in range(N_STAGES):
        R = recover_coefficients_from_H(H[audit_idx, layer])
        for k in range(1, D + 1):
            for subset in groups[k]:
                e = E[audit_idx, subset]
                projected = R[:, subset, :] @ directions[k - 1]
                # projected = gain*E + projected Gaussian noise.  With only
                # nine audit contexts we do not demand a precise slope; we
                # verify finiteness and that the construction is nonconstant.
                if not np.all(np.isfinite(projected)):
                    raise RuntimeError(f"{scenario_name}: non-finite recovered projection.")
                if np.std(projected) <= 0.0:
                    raise RuntimeError(f"{scenario_name}: degenerate recovered projection.")
                if not np.all(np.isfinite(e)):
                    raise RuntimeError(f"{scenario_name}: non-finite target coefficient.")

    # Stronger pooled empirical sanity check using all contexts.  The sample
    # covariance Cov(E, u^T R) should approximately equal the prescribed gain
    # because Var(E)=1 and the additive noise is independent.
    for layer in range(N_STAGES):
        R = recover_coefficients_from_H(H[:, layer])
        for k in range(1, D + 1):
            idx = groups[k]
            e = E[:, idx].reshape(-1)
            r = R[:, idx, :].reshape(-1, HIDDEN_DIM)
            projected = r @ directions[k - 1]
            cov = float(np.mean((e - e.mean()) * (projected - projected.mean())))
            target_gain = float(gains[layer, k - 1])
            # Sampling error is small at 4k-24k pooled samples.  0.08 is wide
            # enough to avoid flaky tests while still catching a wrong order,
            # direction, or stage index.
            if abs(cov - target_gain) > 0.08:
                raise RuntimeError(
                    f"{scenario_name}: empirical gain audit failed at layer={layer}, "
                    f"order={k}. covariance={cov:.4f}, expected~{target_gain:.4f}."
                )


def run_self_test() -> None:
    audit_walsh_core(D)

    # Small structural controls.
    controls: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for k in range(1, D + 1):
        controls[f"pure_order_{k}"] = make_pure_order_landscape(D, k, CONTROL_SEED + k)
    controls["parity"] = make_parity_landscape(D)
    for C in MIXTURE_C_VALUES:
        controls[f"mixture_C_{C:.1f}"] = make_low_high_mixture(
            D, C, 1, D, CONTROL_SEED + 100
        )
    audit_structural_controls(controls)

    # Tiny representation round-trip using the same construction but without
    # relying on frozen full-run dimensions/splits.
    n = 128
    rng = np.random.default_rng(999)
    E = rng.normal(size=(n, N_VERTICES))
    E[:, 0] = 0.0
    subset_orders = build_subset_orders(D)
    directions = build_order_directions(HIDDEN_DIM, D, DIRECTION_SEED)
    gains = build_gain_schedule(STAGGERED_DROP_STAGE, N_STAGES, SIGNAL_GAIN, ATTENUATED_GAIN, D)

    # Inline small builder to avoid the full-run shape assertion.
    X = chi_matrix(D)
    rep_rng = np.random.default_rng(REPRESENTATION_SEED_BASE)
    for layer in (0, STAGGERED_DROP_STAGE[1], STAGGERED_DROP_STAGE[4], N_STAGES - 1):
        coeff = rep_rng.normal(
            scale=NOISE_STD,
            size=(n, N_VERTICES, HIDDEN_DIM),
        )
        for subset in range(1, N_VERTICES):
            k = int(subset_orders[subset])
            coeff[:, subset, :] += (
                gains[layer, k - 1]
                * E[:, subset, None]
                * directions[k - 1][None, :]
            )
        H = np.einsum("xs,csm->cxm", X, coeff, optimize=True)
        recovered = np.einsum("sx,cxm->csm", X.T, H, optimize=True) / N_VERTICES
        if not np.allclose(coeff, recovered, rtol=1e-11, atol=1e-11):
            raise RuntimeError("Self-test vector Walsh round-trip failed.")

    analytic = analytic_mi_from_gains(gains, NOISE_STD)
    if expected_drop_from_analytic_mi(analytic) != STAGGERED_DROP_STAGE:
        raise RuntimeError("Self-test bottleneck localization failed.")

    print("SELF-TEST: PASS")
    print("Verified: previous-ORBIT Walsh convention, pure orders, parity, mixtures,")
    print("          vector-valued Walsh reconstruction, finite analytic MI channel,")
    print("          and known order-specific bottleneck locations.")


# =============================================================================
# Serialization / provenance
# =============================================================================


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def save_controls(
    path: Path,
    controls: Mapping[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    payload: Dict[str, np.ndarray] = {}
    for name, (eps, fitness) in controls.items():
        payload[f"{name}__coefficients"] = np.asarray(eps, dtype=np.float64)
        payload[f"{name}__fitness"] = np.asarray(fitness, dtype=np.float64)
    np.savez(path, **payload)


def save_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = ["control"] + [f"energy_k{k}" for k in range(1, D + 1)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_context_split_csv(path: Path, split_codes: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["context_id", "split"])
        for i, code in enumerate(split_codes):
            writer.writerow([f"SYN_{i:06d}", split_name(int(code))])


def json_drop_map(mapping: Mapping[int, int | None]) -> Dict[str, int | None]:
    return {str(k): (None if v is None else int(v)) for k, v in mapping.items()}


# =============================================================================
# Full build
# =============================================================================


def build_full(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"{CODE_NAME} — ORBIT-SYNTH FMIB CALIBRATION BUILD")
    print("=" * 76)
    print()

    # -------------------------------------------------------------------------
    # 1. Walsh / structural controls
    # -------------------------------------------------------------------------
    audit_walsh_core(D)
    print("Walsh core audit: PASS")
    print("  convention: bit 0 / WT -> -1; bit 1 / mutation -> +1")
    print("  d=4 complete cube: 16 vertices")
    print()

    controls: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for k in range(1, D + 1):
        controls[f"pure_order_{k}"] = make_pure_order_landscape(D, k, CONTROL_SEED + k)
    controls["parity"] = make_parity_landscape(D)
    for C in MIXTURE_C_VALUES:
        controls[f"mixture_C_{C:.1f}"] = make_low_high_mixture(
            D, C, 1, D, CONTROL_SEED + 100
        )

    summary_rows = audit_structural_controls(controls)
    print("Previous-ORBIT structural controls: PASS")
    print("  pure orders: 1, 2, 3, 4")
    print("  exact parity: PASS")
    print("  order-1/order-4 mixtures: C = " + ", ".join(f"{c:.1f}" for c in MIXTURE_C_VALUES))
    print()

    # -------------------------------------------------------------------------
    # 2. Frozen synthetic contexts / targets / splits
    # -------------------------------------------------------------------------
    subset_orders = build_subset_orders(D)
    E, fitness = build_targets(N_CONTEXTS, D, TARGET_SEED)
    audit_targets(E, fitness)

    split_codes = build_context_splits(N_CONTEXTS, SPLIT_SEED)
    audit_splits(split_codes)

    directions = build_order_directions(HIDDEN_DIM, D, DIRECTION_SEED)

    print("FMIB target ensemble: PASS")
    print(f"  contexts: {N_CONTEXTS:,}")
    print(f"  train / val / test: {N_TRAIN:,} / {N_VAL:,} / {N_TEST:,}")
    print("  target coefficients: iid N(0,1) for every nonconstant subset")
    print("  context-level split: PASS")
    print(f"  representation hidden dimension: {HIDDEN_DIM}")
    print(f"  Gaussian channel noise sigma: {NOISE_STD:.3f}")
    print()

    # -------------------------------------------------------------------------
    # 3. Staggered bottleneck scenario
    # -------------------------------------------------------------------------
    staggered_gains = build_gain_schedule(
        STAGGERED_DROP_STAGE,
        N_STAGES,
        SIGNAL_GAIN,
        ATTENUATED_GAIN,
        D,
    )
    staggered_mi = analytic_mi_from_gains(staggered_gains, NOISE_STD)
    staggered_H = build_layerwise_representations(
        E,
        subset_orders,
        staggered_gains,
        directions,
        NOISE_STD,
        REPRESENTATION_SEED_BASE + 1,
    )
    audit_representation_scenario(
        staggered_H,
        E,
        subset_orders,
        staggered_gains,
        directions,
        staggered_mi,
        STAGGERED_DROP_STAGE,
        "staggered_bottleneck",
    )
    print("Staggered bottleneck scenario: PASS")
    for k in range(1, D + 1):
        print(f"  order {k}: known drop at stage {STAGGERED_DROP_STAGE[k]}")
    print()

    # -------------------------------------------------------------------------
    # 4. Higher-order-selective bottleneck scenario
    # -------------------------------------------------------------------------
    selective_gains = build_gain_schedule(
        SELECTIVE_HO_DROP_STAGE,
        N_STAGES,
        SIGNAL_GAIN,
        ATTENUATED_GAIN,
        D,
    )
    selective_mi = analytic_mi_from_gains(selective_gains, NOISE_STD)
    selective_H = build_layerwise_representations(
        E,
        subset_orders,
        selective_gains,
        directions,
        NOISE_STD,
        REPRESENTATION_SEED_BASE + 2,
    )
    audit_representation_scenario(
        selective_H,
        E,
        subset_orders,
        selective_gains,
        directions,
        selective_mi,
        SELECTIVE_HO_DROP_STAGE,
        "selective_ho_bottleneck",
    )
    print("Selective higher-order bottleneck scenario: PASS")
    print("  orders 1-2: retained through final stage")
    print("  orders 3-4: attenuated beginning at stage 4")
    print()

    # Analytic retention target for selective-HO scenario.
    selective_retain = selective_mi[-1, :] / np.max(selective_mi, axis=0)
    lo_retention = float(np.mean(selective_retain[:2]))
    ho_retention = float(np.mean(selective_retain[2:]))
    delta_ho_lo = ho_retention - lo_retention
    if not delta_ho_lo < 0.0:
        raise RuntimeError("Selective-HO analytic retention contrast is not negative.")

    print("Analytic selective-retention target: PASS")
    for k in range(1, D + 1):
        print(f"  Retain_{k}: {selective_retain[k-1]:.6f}")
    print(f"  LO retention: {lo_retention:.6f}")
    print(f"  HO retention: {ho_retention:.6f}")
    print(f"  Delta_HO-LO: {delta_ho_lo:.6f}")
    print()

    # -------------------------------------------------------------------------
    # 5. Save frozen artifacts
    # -------------------------------------------------------------------------
    controls_path = output_dir / "orbit_synth_controls.npz"
    targets_path = output_dir / "fmib_targets.npz"
    staggered_path = output_dir / "fmib_staggered_bottleneck.npz"
    selective_path = output_dir / "fmib_selective_ho_bottleneck.npz"
    summary_path = output_dir / "orbit_synth_summary.csv"
    split_path = output_dir / "fmib_context_split.csv"
    manifest_path = output_dir / "orbit_synth_manifest.json"

    save_controls(controls_path, controls)
    np.savez(
        targets_path,
        E=E.astype(np.float32),
        fitness=fitness.astype(np.float32),
        subset_orders=subset_orders,
        split_codes=split_codes,
        order_directions=directions.astype(np.float32),
        walsh_matrix=chi_matrix(D).astype(np.float32),
    )
    np.savez(
        staggered_path,
        H=staggered_H,
        gains=staggered_gains.astype(np.float32),
        analytic_mi_nats=staggered_mi.astype(np.float64),
    )
    np.savez(
        selective_path,
        H=selective_H,
        gains=selective_gains.astype(np.float32),
        analytic_mi_nats=selective_mi.astype(np.float64),
        analytic_retain=selective_retain.astype(np.float64),
        analytic_lo_retention=np.array(lo_retention, dtype=np.float64),
        analytic_ho_retention=np.array(ho_retention, dtype=np.float64),
        analytic_delta_ho_lo=np.array(delta_ho_lo, dtype=np.float64),
    )
    save_summary_csv(summary_path, summary_rows)
    save_context_split_csv(split_path, split_codes)

    file_hashes = {
        p.name: sha256_file(p)
        for p in (
            controls_path,
            targets_path,
            staggered_path,
            selective_path,
            summary_path,
            split_path,
        )
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "purpose": "Frozen ORBIT-FMIB synthetic calibration benchmark; no GB1/ESM/MINE run.",
        "walsh_convention": {
            "bit_0": "-1 (WT/reference)",
            "bit_1": "+1 (mutation)",
            "coefficient_formula": "2^-d sum_x f(x) chi_S(x)",
            "sign_note": "Orders 1 and 3 have opposite signed coefficients to the canonical Wu spin convention; energies are invariant.",
        },
        "dimensions": {
            "d": D,
            "vertices_per_cube": N_VERTICES,
            "stages": N_STAGES,
            "hidden_dim": HIDDEN_DIM,
            "contexts": N_CONTEXTS,
        },
        "splits": {
            "train": N_TRAIN,
            "val": N_VAL,
            "test": N_TEST,
            "unit": "synthetic context/cube",
            "seed": SPLIT_SEED,
        },
        "seeds": {
            "control": CONTROL_SEED,
            "target": TARGET_SEED,
            "direction": DIRECTION_SEED,
            "representation_base": REPRESENTATION_SEED_BASE,
        },
        "channel": {
            "formula": "R_l,C,S = gain_l,k * E_C,S * u_k + sigma * eta; eta~N(0,I)",
            "noise_std": NOISE_STD,
            "signal_gain": SIGNAL_GAIN,
            "attenuated_gain": ATTENUATED_GAIN,
            "analytic_mi_nats": "0.5*log(1 + gain^2/noise_std^2)",
        },
        "scenarios": {
            "staggered_bottleneck": {
                "drop_stage_by_order": json_drop_map(STAGGERED_DROP_STAGE),
                "gains": staggered_gains.tolist(),
                "analytic_mi_nats": staggered_mi.tolist(),
            },
            "selective_ho_bottleneck": {
                "drop_stage_by_order": json_drop_map(SELECTIVE_HO_DROP_STAGE),
                "gains": selective_gains.tolist(),
                "analytic_mi_nats": selective_mi.tolist(),
                "analytic_retain": selective_retain.tolist(),
                "analytic_lo_retention": lo_retention,
                "analytic_ho_retention": ho_retention,
                "analytic_delta_ho_lo": delta_ho_lo,
            },
        },
        "structural_controls": {
            "pure_orders": [1, 2, 3, 4],
            "parity": True,
            "mixture_low_order": 1,
            "mixture_high_order": 4,
            "mixture_C_values": list(MIXTURE_C_VALUES),
        },
        "dtype_on_disk": "float32 for large tensors; analytic MI/retention stored float64",
        "file_sha256": file_hashes,
        "interpretation_limits": [
            "Synthetic representations are calibration instruments, not biological models.",
            "This code does not train or validate MINE; Code 04 does generic MI-estimator calibration.",
            "This code does not claim the context-side I(Q_k;R_l,k) estimator is validated.",
            "GB1 and ESM-2 are not accessed by this script.",
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print("Saved:")
    for p in (
        controls_path,
        targets_path,
        staggered_path,
        selective_path,
        summary_path,
        split_path,
        manifest_path,
    ):
        print(f"  {p}")
    print()
    print(f"{CODE_NAME}: PASS")


# =============================================================================
# CLI
# =============================================================================


def project_root_from_script() -> Path:
    # Expected location: <project>/03 Code/03_build_orbit_synth_fmib.py
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "03 code":
        return script_dir.parent
    # Safe fallback for testing / copied script use.
    return Path.cwd()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run fast mathematical/static self-tests without writing the full benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return

    root = project_root_from_script()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = root / "04 Experiments" / "EXP01_ORBIT_Synth"
    build_full(output_dir.resolve())


if __name__ == "__main__":
    main()
