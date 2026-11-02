#!/usr/bin/env python3
"""
ORBIT-FMIB CODE 10 — Build GB1 order-resolved experimental Walsh targets.

Purpose
-------
Compute the exact experimental Walsh coefficient

    E_{C,S} = 2^-4 * sum_x f_C(x) chi_S(x)

for every non-empty subset S of the four GB1 assayed positions and every
AA-identity-strict complete cube C.

This stage uses experimental fitness, but does NOT use ESM-2 representations.
It can therefore run locally while Code 09 is running on HPC cluster.

Walsh convention
----------------
For each cube coordinate:
    bit 0 / WT       -> -1
    bit 1 / mutation -> +1

Code-06 vertex order is mask 0..15 with bits corresponding to
positions (39, 40, 41, 54).  Subset masks 1..15 therefore resolve interaction
orders k=1..4.

Default inputs
--------------
04 Experiments/EXP02_GB1_ESM2/gb1_cube_manifest.json
04 Experiments/EXP02_GB1_ESM2/gb1_strict_split_manifest.json

Default outputs
---------------
04 Experiments/EXP02_GB1_ESM2/
    gb1_order_samples.npz
    gb1_order_sample_summary.csv
    gb1_order_targets_manifest.json

Usage
-----
    python "03 Code\\10_build_gb1_order_samples.py" --self-test
    python "03 Code\\10_build_gb1_order_samples.py"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Frozen Phase-1 contract
# =============================================================================

CODE_NAME = "ORBIT-FMIB CODE 10"
SCHEMA_VERSION = 1

EXPECTED_CODE06 = "ORBIT-FMIB CODE 06"
EXPECTED_CODE07 = "ORBIT-FMIB CODE 07"

D = 4
N_VERTICES = 1 << D
POSITIONS = (39, 40, 41, 54)
WT = "VDGV"

EXPECTED_COMPLETE_CUBES = 109_235
EXPECTED_STRICT_COUNTS = {
    "estimator_train": 7_352,
    "estimator_val": 245,
    "estimator_test": 613,
}
EXPECTED_STRICT_CUBES = 8_210

SPLIT_ORDER = ("estimator_train", "estimator_val", "estimator_test")
SPLIT_CODE = {
    "estimator_train": 0,
    "estimator_val": 1,
    "estimator_test": 2,
}
SPLIT_INDEX_KEYS = {
    "estimator_train": "estimator_train_indices",
    "estimator_val": "estimator_val_indices",
    "estimator_test": "estimator_test_indices",
}

SUBSET_MASKS = np.arange(1, N_VERTICES, dtype=np.uint8)
SUBSET_ORDERS = np.asarray(
    [int(mask).bit_count() for mask in SUBSET_MASKS],
    dtype=np.uint8,
)

EXPECTED_SUBSETS_PER_ORDER = {
    1: math.comb(D, 1),
    2: math.comb(D, 2),
    3: math.comb(D, 3),
    4: math.comb(D, 4),
}
EXPECTED_SAMPLE_COUNTS_BY_ORDER = {
    k: EXPECTED_STRICT_CUBES * n_subsets
    for k, n_subsets in EXPECTED_SUBSETS_PER_ORDER.items()
}
EXPECTED_TOTAL_SAMPLES = EXPECTED_STRICT_CUBES * (N_VERTICES - 1)

WALSH_RECON_ATOL = 1e-12


# =============================================================================
# Helpers
# =============================================================================


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed.lower() != str(expected).lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Path: {path}\nExpected: {expected}\nObserved: {observed}"
        )


# =============================================================================
# Walsh core
# =============================================================================


def enumerate_points(d: int = D) -> np.ndarray:
    """
    Rows are vertex masks 0..2^d-1.
    Each coordinate is -1 for bit 0 / WT and +1 for bit 1 / mutation.
    """
    idx = np.arange(1 << d, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(d, dtype=np.int64)) & 1
    return np.where(bits == 1, 1.0, -1.0).astype(np.float64)


def chi_matrix(d: int = D) -> np.ndarray:
    """
    X[x,S] = chi_S(x), with both x and S represented as integer bit masks.
    """
    points = enumerate_points(d)
    n = 1 << d
    X = np.ones((n, n), dtype=np.float64)
    for subset in range(n):
        positions = [b for b in range(d) if (subset >> b) & 1]
        if positions:
            X[:, subset] = np.prod(points[:, positions], axis=1)
    return X


def walsh_coefficients(fitness: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    fitness: [n_cubes, 16]
    returns E: [n_cubes, 16], including intercept subset mask 0.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    if fitness.ndim != 2 or fitness.shape[1] != N_VERTICES:
        raise ValueError("fitness must have shape [n_cubes, 16].")
    if X.shape != (N_VERTICES, N_VERTICES):
        raise ValueError("Walsh matrix must have shape [16,16].")
    coeff = fitness @ X / float(N_VERTICES)
    if not np.isfinite(coeff).all():
        raise RuntimeError("Non-finite experimental Walsh coefficients.")
    return coeff


def walsh_reconstruct(coeff: np.ndarray, X: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=np.float64)
    return coeff @ X.T


# =============================================================================
# Provenance
# =============================================================================


def load_and_verify_code06(
    manifest_path: Path,
) -> Tuple[Dict, str, Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Code-06 manifest not found: {manifest_path}\n"
            "Code 06 must pass before Code 10."
        )
    manifest = load_json(manifest_path)
    if manifest.get("code_name") != EXPECTED_CODE06:
        raise RuntimeError("Unexpected Code-06 manifest code_name.")
    if not bool(manifest.get("overall_pass", False)):
        raise RuntimeError("Code-06 manifest does not record overall_pass=true.")

    audit = manifest.get("audit", {})
    if int(audit.get("complete_cubes", -1)) != EXPECTED_COMPLETE_CUBES:
        raise RuntimeError("Code-06 complete-cube count is not 109,235.")
    if audit.get("vertex_ordering") != "PASS":
        raise RuntimeError("Code-06 vertex-ordering audit is not PASS.")

    definition = manifest.get("cube_definition", {})
    if definition.get("bit_convention") != "0=WT, 1=selected alternative":
        raise RuntimeError("Code-06 cube bit convention mismatch.")
    if definition.get("walsh_sign_convention") != (
        "bit 0 / WT -> -1; bit 1 / mutation -> +1"
    ):
        raise RuntimeError("Code-06 Walsh sign convention mismatch.")
    if list(definition.get("vertex_mask_order", [])) != list(range(16)):
        raise RuntimeError("Code-06 vertex mask order is not 0..15.")
    if tuple(int(x) for x in definition.get("biological_positions", [])) != POSITIONS:
        raise RuntimeError("Code-06 biological-position order mismatch.")

    outputs = manifest.get("outputs", {})
    hashes = manifest.get("file_sha256", {})
    fitness_name = outputs.get("fitness")
    if not fitness_name:
        raise RuntimeError("Code-06 manifest is missing outputs.fitness.")
    fitness_path = manifest_path.parent / str(fitness_name)
    fitness_sha = hashes.get(fitness_path.name)
    if not fitness_sha:
        raise RuntimeError("Code-06 manifest is missing fitness SHA256.")
    verify_file_hash(fitness_path, str(fitness_sha), "Code-06 fitness artifact")

    return manifest, sha256_file(manifest_path), fitness_path


def load_and_verify_code07(
    manifest_path: Path,
    code06_manifest_path: Path,
    code06_sha: str,
) -> Tuple[Dict, str, Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Code-07 manifest not found: {manifest_path}\n"
            "Code 07 must pass before Code 10."
        )
    manifest = load_json(manifest_path)
    if manifest.get("code_name") != EXPECTED_CODE07:
        raise RuntimeError("Unexpected Code-07 manifest code_name.")
    if not bool(manifest.get("overall_pass", False)):
        raise RuntimeError("Code-07 manifest does not record overall_pass=true.")

    audit = manifest.get("audit", {})
    if not bool(audit.get("overall_pass", False)):
        raise RuntimeError("Code-07 audit.overall_pass is not true.")
    counts = audit.get("split_counts", {})
    for split_name, expected in EXPECTED_STRICT_COUNTS.items():
        if int(counts.get(split_name, -1)) != expected:
            raise RuntimeError(
                f"Code-07 count mismatch for {split_name}: "
                f"expected {expected:,}, observed {counts.get(split_name)!r}."
            )
    if int(audit.get("cross_split_nonwt_variant_overlap", -1)) != 0:
        raise RuntimeError("Code-07 non-WT variant leakage audit is not zero.")
    if int(audit.get("cross_split_nonwt_pfm_row_overlap", -1)) != 0:
        raise RuntimeError("Code-07 non-WT PFM-row leakage audit is not zero.")

    upstream = manifest.get("upstream", {})
    pinned_code06_sha = str(upstream.get("code06_manifest_sha256", ""))
    if pinned_code06_sha.lower() != code06_sha.lower():
        raise RuntimeError(
            "Code-07 does not pin the current Code-06 manifest SHA256."
        )

    outputs = manifest.get("outputs", {})
    hashes = manifest.get("file_sha256", {})
    split_name = outputs.get("split_indices")
    if not split_name:
        raise RuntimeError("Code-07 manifest is missing outputs.split_indices.")
    split_path = manifest_path.parent / str(split_name)
    split_sha = hashes.get(split_path.name)
    if not split_sha:
        raise RuntimeError("Code-07 manifest is missing split-index SHA256.")
    verify_file_hash(split_path, str(split_sha), "Code-07 strict split indices")

    return manifest, sha256_file(manifest_path), split_path


def load_fitness(path: Path) -> np.ndarray:
    fitness = np.load(path, allow_pickle=False)
    expected = (EXPECTED_COMPLETE_CUBES, N_VERTICES)
    if fitness.shape != expected:
        raise RuntimeError(
            f"Code-06 fitness shape mismatch: {fitness.shape} vs {expected}."
        )
    fitness = np.asarray(fitness, dtype=np.float64)
    if not np.isfinite(fitness).all():
        raise RuntimeError("Non-finite Code-06 experimental fitness.")
    return fitness


def load_strict_indices(path: Path) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for split_name in SPLIT_ORDER:
            key = SPLIT_INDEX_KEYS[split_name]
            if key not in data.files:
                raise RuntimeError(f"Strict split NPZ missing key {key!r}.")
            idx = np.asarray(data[key], dtype=np.int64).reshape(-1)
            expected_n = EXPECTED_STRICT_COUNTS[split_name]
            if len(idx) != expected_n:
                raise RuntimeError(
                    f"{split_name} expected {expected_n:,} cubes; observed {len(idx):,}."
                )
            if len(np.unique(idx)) != len(idx):
                raise RuntimeError(f"{split_name} has duplicate cube IDs.")
            if idx.size and (idx.min() < 0 or idx.max() >= EXPECTED_COMPLETE_CUBES):
                raise RuntimeError(f"{split_name} has out-of-range cube IDs.")
            result[split_name] = idx

    for i, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[i + 1 :]:
            if np.intersect1d(result[left], result[right]).size:
                raise RuntimeError(f"Cube IDs overlap between {left} and {right}.")
    return result


# =============================================================================
# Order-sample construction
# =============================================================================


def build_samples(
    fitness: np.ndarray,
    strict: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict]:
    X = chi_matrix(D)

    # Orthogonality is the exact Walsh/Hadamard normalization contract.
    gram = X.T @ X
    if not np.array_equal(gram, N_VERTICES * np.eye(N_VERTICES)):
        raise RuntimeError("Walsh character matrix is not exactly orthogonal.")

    arrays = {
        "sample_id": [],
        "split_code": [],
        "cube_id": [],
        "subset_mask": [],
        "order_k": [],
        "fitness_walsh": [],
    }

    summary_rows = []
    next_sample = 0
    max_recon_error = 0.0

    for split_name in SPLIT_ORDER:
        cube_ids = np.asarray(strict[split_name], dtype=np.int64)
        split_fitness = fitness[cube_ids]
        coeff = walsh_coefficients(split_fitness, X)

        recon = walsh_reconstruct(coeff, X)
        recon_error = float(np.max(np.abs(recon - split_fitness)))
        max_recon_error = max(max_recon_error, recon_error)
        if recon_error > WALSH_RECON_ATOL:
            raise RuntimeError(
                f"Walsh reconstruction failed for {split_name}: "
                f"{recon_error:.3e} > {WALSH_RECON_ATOL:.1e}"
            )

        n_cubes = len(cube_ids)
        n_samples = n_cubes * len(SUBSET_MASKS)

        arrays["sample_id"].append(
            np.arange(next_sample, next_sample + n_samples, dtype=np.int64)
        )
        arrays["split_code"].append(
            np.full(n_samples, SPLIT_CODE[split_name], dtype=np.uint8)
        )
        arrays["cube_id"].append(
            np.repeat(cube_ids, len(SUBSET_MASKS)).astype(np.int64)
        )
        arrays["subset_mask"].append(
            np.tile(SUBSET_MASKS, n_cubes).astype(np.uint8)
        )
        arrays["order_k"].append(
            np.tile(SUBSET_ORDERS, n_cubes).astype(np.uint8)
        )
        arrays["fitness_walsh"].append(
            coeff[:, 1:].reshape(-1).astype(np.float64)
        )
        next_sample += n_samples

        for k in range(1, D + 1):
            mask = SUBSET_ORDERS == k
            values = coeff[:, 1:][:, mask].reshape(-1)
            expected = n_cubes * EXPECTED_SUBSETS_PER_ORDER[k]
            if len(values) != expected:
                raise RuntimeError(
                    f"{split_name} order-{k} sample count mismatch."
                )
            summary_rows.append(
                {
                    "estimator_split": split_name,
                    "split_code": SPLIT_CODE[split_name],
                    "order_k": k,
                    "strict_cubes": n_cubes,
                    "subsets_per_cube": EXPECTED_SUBSETS_PER_ORDER[k],
                    "samples": len(values),
                    "mean_E": float(np.mean(values)),
                    "std_E": float(np.std(values, ddof=0)),
                    "min_E": float(np.min(values)),
                    "max_E": float(np.max(values)),
                }
            )

    out = {
        name: np.concatenate(parts, axis=0)
        for name, parts in arrays.items()
    }

    if len(out["sample_id"]) != EXPECTED_TOTAL_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_SAMPLES:,} total order samples; "
            f"observed {len(out['sample_id']):,}."
        )
    if not np.array_equal(
        out["sample_id"],
        np.arange(EXPECTED_TOTAL_SAMPLES, dtype=np.int64),
    ):
        raise RuntimeError("sample_id is not contiguous/deterministic.")

    # Exact global order counts.
    observed_by_order = {
        k: int(np.sum(out["order_k"] == k))
        for k in range(1, D + 1)
    }
    if observed_by_order != EXPECTED_SAMPLE_COUNTS_BY_ORDER:
        raise RuntimeError(
            "Global order-sample counts changed.\n"
            f"Expected: {EXPECTED_SAMPLE_COUNTS_BY_ORDER}\n"
            f"Observed: {observed_by_order}"
        )

    # Every (split,cube) must have each nonempty subset mask exactly once.
    for split_name in SPLIT_ORDER:
        code = SPLIT_CODE[split_name]
        sub = out["subset_mask"][out["split_code"] == code]
        expected_per_mask = EXPECTED_STRICT_COUNTS[split_name]
        counts = {m: int(np.sum(sub == m)) for m in range(1, 16)}
        if any(v != expected_per_mask for v in counts.values()):
            raise RuntimeError(
                f"{split_name} does not contain each subset mask exactly once per cube."
            )

    summary = pd.DataFrame(summary_rows)
    audit = {
        "strict_cubes": EXPECTED_STRICT_CUBES,
        "strict_cube_counts": EXPECTED_STRICT_COUNTS,
        "subsets_per_order": EXPECTED_SUBSETS_PER_ORDER,
        "sample_counts_by_order": EXPECTED_SAMPLE_COUNTS_BY_ORDER,
        "total_order_samples": EXPECTED_TOTAL_SAMPLES,
        "max_walsh_reconstruction_abs_error": max_recon_error,
        "walsh_reconstruction_tolerance": WALSH_RECON_ATOL,
        "all_targets_finite": bool(np.isfinite(out["fitness_walsh"]).all()),
        "experimental_fitness_used": True,
        "esm2_representations_used": False,
        "mixed_cubes_used": False,
        "overall_pass": True,
    }
    return out, summary, audit


# =============================================================================
# Saving
# =============================================================================


def save_outputs(
    output_dir: Path,
    samples: Mapping[str, np.ndarray],
    summary: pd.DataFrame,
    audit: Dict,
    code06_manifest_path: Path,
    code06_sha: str,
    code07_manifest_path: Path,
    code07_sha: str,
) -> Tuple[list[Path], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = output_dir / "gb1_order_samples.npz"
    summary_path = output_dir / "gb1_order_sample_summary.csv"
    manifest_path = output_dir / "gb1_order_targets_manifest.json"

    np.savez_compressed(samples_path, **samples)
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    output_paths = [samples_path, summary_path]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "purpose": "Exact order-resolved experimental GB1 Walsh targets E_C,S for strict estimator partitions.",
        "upstream": {
            "code06_manifest_path": str(code06_manifest_path.resolve()),
            "code06_manifest_sha256": code06_sha,
            "code06_overall_pass": True,
            "code07_manifest_path": str(code07_manifest_path.resolve()),
            "code07_manifest_sha256": code07_sha,
            "code07_overall_pass": True,
        },
        "walsh_contract": {
            "dimensions": D,
            "vertices_per_cube": N_VERTICES,
            "biological_positions": list(POSITIONS),
            "wt_variant": WT,
            "vertex_mask_order": list(range(16)),
            "bit_convention": "0=WT, 1=selected alternative",
            "sign_convention": "bit 0 / WT -> -1; bit 1 / mutation -> +1",
            "formula": "E_C,S = 2^-4 * sum_x f_C(x) chi_S(x)",
            "subset_masks_saved": list(range(1, 16)),
            "intercept_subset_mask_0_saved": False,
        },
        "split_code_map": SPLIT_CODE,
        "sample_order": "split order train,val,test; within split ascending Code-07 cube ID order; within cube subset_mask 1..15",
        "audit": audit,
        "outputs": {
            "order_samples": samples_path.name,
            "summary": summary_path.name,
        },
        "file_sha256": {p.name: sha256_file(p) for p in output_paths},
        "overall_pass": True,
        "interpretation_boundary": [
            "fitness_walsh is an experimental target derived from GB1 fitness, not an ESM-2 quantity.",
            "Code 10 does not load or use ESM-2 hidden states.",
            "Code 11 is responsible for pairing these targets with representation Walsh coefficients under the same cube/subset convention.",
        ],
    }
    write_json(manifest_path, manifest)
    return output_paths, manifest_path


# =============================================================================
# Self-test
# =============================================================================


def run_self_test() -> None:
    X = chi_matrix(D)
    assert X.shape == (16, 16)
    assert np.array_equal(X.T @ X, 16.0 * np.eye(16))

    # Each pure Walsh character must recover coefficient 1 only at its own mask.
    for subset in range(16):
        f = X[:, subset][None, :]
        coeff = walsh_coefficients(f, X)[0]
        expected = np.zeros(16, dtype=np.float64)
        expected[subset] = 1.0
        np.testing.assert_allclose(coeff, expected, atol=0.0, rtol=0.0)
        recon = walsh_reconstruct(coeff[None, :], X)[0]
        np.testing.assert_allclose(recon, f[0], atol=0.0, rtol=0.0)

    assert EXPECTED_SUBSETS_PER_ORDER == {1: 4, 2: 6, 3: 4, 4: 1}
    assert EXPECTED_SAMPLE_COUNTS_BY_ORDER == {
        1: 32_840,
        2: 49_260,
        3: 32_840,
        4: 8_210,
    }
    assert EXPECTED_TOTAL_SAMPLES == 123_150

    print("SELF-TEST: PASS")
    print(
        "Verified: exact Walsh orthogonality/inversion, pure subset recovery for all "
        "16 masks, 4/6/4/1 order multiplicities, and 123,150 strict order samples."
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--code06-manifest", type=Path, default=None)
    parser.add_argument("--code07-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    root = project_root()
    output_dir = args.output_dir or (
        root / "04 Experiments" / "EXP02_GB1_ESM2"
    )
    code06_manifest_path = args.code06_manifest or (
        output_dir / "gb1_cube_manifest.json"
    )
    code07_manifest_path = args.code07_manifest or (
        output_dir / "gb1_strict_split_manifest.json"
    )

    print("=" * 76)
    print(f"{CODE_NAME} — ORDER-RESOLVED EXPERIMENTAL GB1 TARGETS")
    print("=" * 76)
    print()

    code06_manifest, code06_sha, fitness_path = load_and_verify_code06(
        code06_manifest_path
    )
    code07_manifest, code07_sha, split_path = load_and_verify_code07(
        manifest_path=code07_manifest_path,
        code06_manifest_path=code06_manifest_path,
        code06_sha=code06_sha,
    )

    print("Upstream provenance gate: PASS")
    print(f"  Code-06 manifest SHA256: {code06_sha}")
    print(f"  Code-07 manifest SHA256: {code07_sha}")
    print("  Code-06 fitness artifact hash: PASS")
    print("  Code-07 split-index artifact hash: PASS")
    print()

    fitness = load_fitness(fitness_path)
    strict = load_strict_indices(split_path)

    print("Strict cube input audit: PASS")
    print(f"  complete cube fitness matrix: {fitness.shape[0]:,} x {fitness.shape[1]}")
    for split_name in SPLIT_ORDER:
        print(f"  {split_name:15s}: {len(strict[split_name]):,} cubes")
    print("  mixed cubes used: NO")
    print()

    samples, summary, audit = build_samples(
        fitness=fitness,
        strict=strict,
    )

    print("Experimental Walsh transform audit: PASS")
    print("  convention: bit 0 / WT -> -1; bit 1 / mutation -> +1")
    print("  vertex order: masks 0..15")
    print("  non-empty subset masks: 1..15")
    print(
        f"  max exact reconstruction error: "
        f"{audit['max_walsh_reconstruction_abs_error']:.3e}"
    )
    print()

    print("Order-resolved target sample audit: PASS")
    for k in range(1, D + 1):
        print(
            f"  k={k}: {EXPECTED_SUBSETS_PER_ORDER[k]} subsets/cube | "
            f"{EXPECTED_SAMPLE_COUNTS_BY_ORDER[k]:,} samples"
        )
    print(f"  total: {EXPECTED_TOTAL_SAMPLES:,} samples")
    print("  all E_C,S finite: PASS")
    print("  ESM-2 representations used: NO")
    print()

    if args.no_write:
        print("NO-WRITE MODE: all Code-10 audits passed; no files written.")
    else:
        output_paths, manifest_path = save_outputs(
            output_dir=output_dir,
            samples=samples,
            summary=summary,
            audit=audit,
            code06_manifest_path=code06_manifest_path,
            code06_sha=code06_sha,
            code07_manifest_path=code07_manifest_path,
            code07_sha=code07_sha,
        )
        print("Saved:")
        for path in output_paths:
            print(f"  {path}")
        print(f"  {manifest_path}")

    print()
    print(f"{CODE_NAME}: PASS")
    print("Order-resolved experimental targets are frozen for Code 11 target-side MI.")


if __name__ == "__main__":
    main()
