#!/usr/bin/env python3
"""ORBIT-FMIB — GB1 randomized cube-derangement null diagnostic.

Purpose
-------
Re-check the Code-11 GB1 null control without changing the completed primary
Code-11/12 results.  The legacy Code-11 null used a one-cube cyclic shift.
This companion diagnostic instead uses five frozen random *derangements*
within each estimator split: every cube is paired with a different cube, the
permutation is bijective, train/validation/test boundaries are preserved, and
the same subset mask remains aligned after cube permutation.

This is a diagnostic, not a new acceptance gate.  It deliberately defines no
post-hoc numerical threshold for how close a null score must be to zero.
Absolute mutual information in nats is NOT claimed.

Recommended sequence
--------------------
Local:
  python "03 Code/11_validate_gb1_null_controls.py" --self-test

HPC cluster / GPU:
  python "03 Code/11_validate_gb1_null_controls.py" --preflight-only --device cuda
  python "03 Code/11_validate_gb1_null_controls.py" --device cuda

If interrupted:
  python "03 Code/11_validate_gb1_null_controls.py" --device cuda --resume

Use --overwrite only for an intentional clean restart of this diagnostic.
It does not modify Code-11 or Code-12 outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

CODE_NAME = "ORBIT-FMIB GB1 RANDOMIZED-DERANGEMENT NULL DIAGNOSTIC"
SCHEMA_VERSION = 1

# This companion diagnostic is intentionally pinned to the completed Code-11 v2
# source that produced the accepted GB1 biological run.
EXPECTED_CODE11_NAME = "ORBIT-FMIB CODE 11"
EXPECTED_CODE11_SCHEMA = 2
EXPECTED_CODE11_SHA256 = "c7d1df1fe2bab3155b1079ced3a74105e022ed5288ad909f7748b99e6cd454d9"

NULL_LAYERS = (0, 15, 30)
ORDERS = (1, 2, 3, 4)
CRITIC_SEEDS = (11, 23, 37)
DERANGEMENT_SEEDS = (101, 211, 307, 401, 503)
REPRESENTATION_DIM = 2560
N_DERANGEMENTS = len(DERANGEMENT_SEEDS)
EXPECTED_FITS = len(NULL_LAYERS) * len(ORDERS) * len(CRITIC_SEEDS) * N_DERANGEMENTS

SPLIT_ORDER = ("estimator_train", "estimator_val", "estimator_test")
SPLIT_CODES = {"estimator_train": 0, "estimator_val": 1, "estimator_test": 2}
STRICT_CUBE_COUNTS = {"estimator_train": 7352, "estimator_val": 245, "estimator_test": 613}

RESULT_COLS = (
    "family",
    "derangement_seed",
    "permutation_set_sha256",
    "layer",
    "order_k",
    "critic_seed",
    "representation_dim",
    "target_dim",
    "n_samples",
    "n_train",
    "n_val",
    "n_test",
    "subset_masks",
    "best_step",
    "final_training_step",
    "best_val_mi",
    "test_mi",
    "test_product_shuffle_sd",
    "heldout_joint_minus_mismatch_score",
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_array_int64(a: np.ndarray) -> str:
    x = np.asarray(a, dtype="<i8", order="C")
    return hashlib.sha256(x.tobytes(order="C")).hexdigest()


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, obj: Mapping) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    got = sha256_file(path)
    if got.lower() != str(expected).lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch.\nExpected: {expected}\nObserved: {got}\nPath: {path}"
        )


def import_code11(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    got = sha256_file(path)
    if got.lower() != EXPECTED_CODE11_SHA256.lower():
        raise RuntimeError(
            "This diagnostic is pinned to the completed Code-11 v2 source.\n"
            f"Expected: {EXPECTED_CODE11_SHA256}\nObserved: {got}\nPath: {path}"
        )
    spec = importlib.util.spec_from_file_location("_orbitfmib_code11_null_diag", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import Code 11.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    required = (
        "CODE_NAME",
        "SCHEMA_VERSION",
        "ORDERS",
        "CRITIC_SEEDS",
        "REPRESENTATION_DIM",
        "NULL_LAYERS",
        "SPLIT_ORDER",
        "SPLIT_CODES",
        "STRICT_CUBE_COUNTS",
        "SUBSETS_BY_ORDER",
        "chi_matrix",
        "import_code04",
        "verify_code08",
        "load_cube_maps",
        "verify_code09",
        "verify_code10",
        "load_order_samples",
        "build_dataset",
        "targets_for",
    )
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(f"Code-11 source missing required objects: {missing}")
    if mod.CODE_NAME != EXPECTED_CODE11_NAME or int(mod.SCHEMA_VERSION) != EXPECTED_CODE11_SCHEMA:
        raise RuntimeError("Wrong Code-11 source/schema.")
    if tuple(map(int, mod.ORDERS)) != ORDERS:
        raise RuntimeError("Code-11 order contract changed.")
    if tuple(map(int, mod.CRITIC_SEEDS)) != CRITIC_SEEDS:
        raise RuntimeError("Code-11 critic-seed contract changed.")
    if int(mod.REPRESENTATION_DIM) != REPRESENTATION_DIM:
        raise RuntimeError("Code-11 representation dimension changed.")
    if tuple(map(int, mod.NULL_LAYERS)) != NULL_LAYERS:
        raise RuntimeError("Code-11 diagnostic-layer contract changed.")
    if tuple(mod.SPLIT_ORDER) != SPLIT_ORDER or dict(mod.STRICT_CUBE_COUNTS) != STRICT_CUBE_COUNTS:
        raise RuntimeError("Code-11 split/cube-count contract changed.")
    return mod


def make_derangement(n: int, seed: int, split_code: int) -> np.ndarray:
    """Deterministic random permutation with no fixed points."""
    if n < 2:
        raise ValueError("A derangement requires n >= 2.")
    base = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(split_code), int(n)]))
    for _ in range(10000):
        p = rng.permutation(n).astype(np.int64, copy=False)
        if np.all(p != base):
            if not np.array_equal(np.sort(p), base):
                raise RuntimeError("Generated derangement is not bijective.")
            return p.copy()
    raise RuntimeError(f"Could not generate a derangement for n={n}, seed={seed}.")


def build_permutation_sets() -> Dict[int, Dict[str, np.ndarray]]:
    out: Dict[int, Dict[str, np.ndarray]] = {}
    seen_hashes = {s: set() for s in SPLIT_ORDER}
    for dseed in DERANGEMENT_SEEDS:
        per_split: Dict[str, np.ndarray] = {}
        for split in SPLIT_ORDER:
            p = make_derangement(
                STRICT_CUBE_COUNTS[split], int(dseed), int(SPLIT_CODES[split])
            )
            h = sha256_array_int64(p)
            if h in seen_hashes[split]:
                raise RuntimeError(f"Duplicate derangement for split={split}.")
            seen_hashes[split].add(h)
            per_split[split] = p
        out[int(dseed)] = per_split
    return out


def permutation_set_sha256(per_split: Mapping[str, np.ndarray]) -> str:
    obj = {s: sha256_array_int64(per_split[s]) for s in SPLIT_ORDER}
    return sha256_obj(obj)


def permute_targets_by_cube(
    v: np.ndarray, n_cubes: int, n_subsets: int, permutation: np.ndarray
) -> np.ndarray:
    if v.shape != (n_cubes * n_subsets, 1):
        raise RuntimeError("Null target shape mismatch.")
    p = np.asarray(permutation, dtype=np.int64)
    if p.shape != (n_cubes,):
        raise RuntimeError("Permutation length mismatch.")
    base = np.arange(n_cubes, dtype=np.int64)
    if np.any(p == base):
        raise RuntimeError("Permutation contains fixed points.")
    if not np.array_equal(np.sort(p), base):
        raise RuntimeError("Permutation is not bijective.")
    a = v.reshape(n_cubes, n_subsets, 1)
    return a[p].reshape(-1, 1).copy()


def build_random_null(code11, samples: Mapping, k: int, per_split: Mapping[str, np.ndarray]) -> np.ndarray:
    pieces = []
    n_subsets = len(code11.SUBSETS_BY_ORDER[k])
    for split in SPLIT_ORDER:
        v = code11.targets_for(samples, split, k)
        pieces.append(
            permute_targets_by_cube(
                v,
                STRICT_CUBE_COUNTS[split],
                n_subsets,
                per_split[split],
            )
        )
    return np.concatenate(pieces)


def verify_completed_code11(manifest_path: Path, seed_path: Path, code11_source: Path) -> Dict:
    m = load_json(manifest_path)
    msha = sha256_file(manifest_path)
    if m.get("code_name") != EXPECTED_CODE11_NAME:
        raise RuntimeError("Completed Code-11 manifest has wrong code_name.")
    if int(m.get("schema_version", -1)) != EXPECTED_CODE11_SCHEMA:
        raise RuntimeError("Completed Code-11 manifest schema mismatch.")
    if not bool(m.get("overall_pass", False)):
        raise RuntimeError("Completed Code-11 manifest is not PASS.")
    if str(m.get("code11_source_sha256", "")).lower() != EXPECTED_CODE11_SHA256.lower():
        raise RuntimeError("Completed Code-11 manifest does not pin the expected Code-11 v2 source.")
    verify_hash(code11_source, EXPECTED_CODE11_SHA256, "Code-11 source")

    ep = m.get("estimator_protocol", {})
    expected_protocol = {
        "implementation": "exact Code-04 train_mine",
        "critic_seeds": list(CRITIC_SEEDS),
        "representation_dim": REPRESENTATION_DIM,
        "target_dim": 1,
        "selection": "validation only",
        "no_pca_or_projection": True,
        "no_clamping": True,
        "no_null_subtraction": True,
        "absolute_mi_accuracy_claimed": False,
    }
    for key, val in expected_protocol.items():
        if ep.get(key) != val:
            raise RuntimeError(f"Code-11 estimator protocol mismatch for {key}: {ep.get(key)!r}")

    ev = m.get("estimator_validation", {})
    if ev.get("relative_validation_status") != "PASS":
        raise RuntimeError("Code-11 relative-retention validation is not PASS.")
    if ev.get("adversarial_validation_status") != "PASS":
        raise RuntimeError("Code-11 adversarial validation is not PASS.")
    if bool(ev.get("absolute_mi_accuracy_claimed", True)):
        raise RuntimeError("Code-11 incorrectly claims absolute MI accuracy.")
    if not bool(ev.get("no_pca_or_projection", False)):
        raise RuntimeError("Code-11 validation contract lost no-PCA/projection guarantee.")
    if not bool(ev.get("no_clamping", False)) or not bool(ev.get("no_null_subtraction", False)):
        raise RuntimeError("Code-11 validation contract lost no-clamping/null-subtraction guarantee.")

    grid = m.get("biological_fit_grid", {})
    if tuple(map(int, grid.get("null_layers", []))) != NULL_LAYERS:
        raise RuntimeError("Code-11 null-layer grid mismatch.")
    if tuple(map(int, grid.get("orders", []))) != ORDERS:
        raise RuntimeError("Code-11 order grid mismatch.")
    if tuple(map(int, grid.get("critic_seeds", []))) != CRITIC_SEEDS:
        raise RuntimeError("Code-11 critic-seed grid mismatch.")
    counts = grid.get("observed_counts", {})
    if int(counts.get("primary", -1)) != 372 or int(counts.get("null_control", -1)) != 36 or int(counts.get("total", -1)) != 408:
        raise RuntimeError("Code-11 biological fit counts mismatch.")

    out_name = m.get("outputs", {}).get("biological_seed_results")
    if not out_name or Path(out_name).name != seed_path.name:
        raise RuntimeError("Code-11 biological seed-results filename mismatch.")
    expected_seed_sha = m.get("file_sha256", {}).get(seed_path.name)
    if not expected_seed_sha:
        raise RuntimeError("Code-11 seed-results hash missing from manifest.")
    verify_hash(seed_path, expected_seed_sha, "Code-11 biological seed results")

    d = pd.read_csv(seed_path)
    if len(d) != 408:
        raise RuntimeError("Code-11 seed table must contain exactly 408 rows.")
    if int((d["family"] == "primary").sum()) != 372 or int((d["family"] == "null_control").sum()) != 36:
        raise RuntimeError("Code-11 primary/null row counts mismatch in seed table.")
    if d.duplicated(["family", "layer", "order_k", "critic_seed"]).any():
        raise RuntimeError("Code-11 seed table contains duplicate condition rows.")
    return {
        "manifest": m,
        "manifest_sha": msha,
        "seed_df": d,
        "seed_sha": str(expected_seed_sha),
    }


def preflight(
    code11_source: Path,
    code11_manifest: Path,
    code11_seed_results: Path,
    code04_source: Path,
    code08_manifest: Path,
    code09_manifest: Path,
    code10_manifest: Path,
):
    code11 = import_code11(code11_source)
    c11 = verify_completed_code11(code11_manifest, code11_seed_results, code11_source)
    up = c11["manifest"].get("upstream", {})

    mine04 = code11.import_code04(code04_source)
    sha04 = sha256_file(code04_source)
    if sha04.lower() != str(up.get("code04_source_sha256", "")).lower():
        raise RuntimeError("Current Code-04 source is not the source pinned by completed Code 11.")

    c08 = code11.verify_code08(code08_manifest)
    if c08["sha"].lower() != str(up.get("code08_manifest_sha256", "")).lower():
        raise RuntimeError("Current Code-08 manifest is not the one pinned by Code 11.")
    if str(c08["map_sha"]).lower() != str(up.get("code08_map_sha256", "")).lower():
        raise RuntimeError("Current Code-08 map is not the one pinned by Code 11.")
    maps = code11.load_cube_maps(c08["map_path"])

    # Avoid rehashing all 31 ~67 MB layer files; verify the manifest provenance and
    # hash exactly the three diagnostic layers that this script will actually use.
    c09 = code11.verify_code09(code09_manifest, c08["sha"], False)
    if c09["sha"].lower() != str(up.get("code09_manifest_sha256", "")).lower():
        raise RuntimeError("Current Code-09 manifest is not the one pinned by Code 11.")
    layer_names = c09["manifest"].get("outputs", {}).get("layer_files", {})
    layer_hashes = c09["manifest"].get("file_sha256", {})
    for layer in NULL_LAYERS:
        name = layer_names.get(str(layer))
        if not name:
            raise RuntimeError(f"Code-09 manifest missing layer {layer} filename.")
        expected = layer_hashes.get(Path(name).name)
        if not expected:
            raise RuntimeError(f"Code-09 manifest missing layer {layer} hash.")
        verify_hash(c09["layer_paths"][layer], expected, f"Code-09 layer {layer}")

    c10 = code11.verify_code10(code10_manifest, c08["manifest"])
    if c10["sha"].lower() != str(up.get("code10_manifest_sha256", "")).lower():
        raise RuntimeError("Current Code-10 manifest is not the one pinned by Code 11.")
    if str(c10["samples_sha"]).lower() != str(up.get("code10_samples_sha256", "")).lower():
        raise RuntimeError("Current Code-10 samples are not the ones pinned by Code 11.")
    samples = code11.load_order_samples(c10["samples_path"])

    for split in SPLIT_ORDER:
        if len(samples["cube_ids_by_split"][split]) != len(maps[split]):
            raise RuntimeError(f"{split}: Code08/10 cube-row count mismatch.")

    permutations = build_permutation_sets()
    perm_meta = {}
    for dseed, per_split in permutations.items():
        split_meta = {}
        for split in SPLIT_ORDER:
            p = per_split[split]
            n = STRICT_CUBE_COUNTS[split]
            fixed = int(np.sum(p == np.arange(n, dtype=np.int64)))
            if fixed != 0:
                raise RuntimeError("Derangement unexpectedly contains fixed points.")
            split_meta[split] = {
                "n_cubes": n,
                "fixed_points": fixed,
                "permutation_sha256": sha256_array_int64(p),
            }
        perm_meta[str(dseed)] = {
            "permutation_set_sha256": permutation_set_sha256(per_split),
            "splits": split_meta,
        }

    return code11, mine04, c11, c08, c09, c10, maps, samples, permutations, perm_meta


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLS)
    d = pd.read_csv(path)
    missing = set(RESULT_COLS) - set(d.columns)
    if missing:
        raise RuntimeError(f"Existing null-diagnostic CSV missing columns: {sorted(missing)}")
    if d.duplicated(["derangement_seed", "layer", "order_k", "critic_seed"]).any():
        raise RuntimeError("Duplicate randomized-null critic rows.")
    return d


def save_result_row(d: pd.DataFrame, row: Mapping, path: Path) -> pd.DataFrame:
    d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    d = d.sort_values(["layer", "order_k", "derangement_seed", "critic_seed"]).reset_index(drop=True)
    write_csv_atomic(d, path)
    return d


def expected_keys() -> set:
    return {
        (int(ds), int(layer), int(k), int(cs))
        for ds in DERANGEMENT_SEEDS
        for layer in NULL_LAYERS
        for k in ORDERS
        for cs in CRITIC_SEEDS
    }


def audit_seed_results(d: pd.DataFrame) -> None:
    obs = {
        (int(r.derangement_seed), int(r.layer), int(r.order_k), int(r.critic_seed))
        for r in d.itertuples(index=False)
    }
    if obs != expected_keys():
        raise RuntimeError(f"Randomized-null fit grid mismatch: observed={len(obs)}, expected={EXPECTED_FITS}.")
    nums = [
        "best_step",
        "final_training_step",
        "best_val_mi",
        "test_mi",
        "test_product_shuffle_sd",
        "heldout_joint_minus_mismatch_score",
    ]
    if not np.isfinite(d[nums].to_numpy(float)).all():
        raise RuntimeError("Randomized-null result table contains non-finite diagnostics.")
    if not (d["family"] == "random_derangement_null").all():
        raise RuntimeError("Unexpected family label in randomized-null table.")
    if not (d["representation_dim"].astype(int) == REPRESENTATION_DIM).all():
        raise RuntimeError("Representation dimension mismatch in randomized-null table.")


def make_summaries(d: pd.DataFrame, code11_seed_df: pd.DataFrame):
    by_der = (
        d.groupby(["derangement_seed", "permutation_set_sha256", "layer", "order_k"], as_index=False)
        .agg(
            score_mean=("test_mi", "mean"),
            critic_seed_sd=("test_mi", lambda x: float(np.std(np.asarray(x, float), ddof=1))),
            score_min=("test_mi", "min"),
            score_max=("test_mi", "max"),
            mean_best_val=("best_val_mi", "mean"),
            mean_best_step=("best_step", "mean"),
            mean_final_training_step=("final_training_step", "mean"),
        )
        .sort_values(["layer", "order_k", "derangement_seed"])
        .reset_index(drop=True)
    )
    if len(by_der) != N_DERANGEMENTS * len(NULL_LAYERS) * len(ORDERS):
        raise RuntimeError("Per-derangement summary row count mismatch.")

    overall = (
        by_der.groupby(["layer", "order_k"], as_index=False)
        .agg(
            random_null_mean=("score_mean", "mean"),
            random_null_sd_across_derangements=("score_mean", lambda x: float(np.std(np.asarray(x, float), ddof=1))),
            random_null_min_derangement_mean=("score_mean", "min"),
            random_null_max_derangement_mean=("score_mean", "max"),
            mean_critic_seed_sd_within_derangement=("critic_seed_sd", "mean"),
        )
        .sort_values(["layer", "order_k"])
        .reset_index(drop=True)
    )
    if len(overall) != len(NULL_LAYERS) * len(ORDERS):
        raise RuntimeError("Overall randomized-null summary row count mismatch.")

    c11 = code11_seed_df.copy()
    c11 = c11[c11["layer"].astype(int).isin(NULL_LAYERS)]
    primary = (
        c11[c11["family"] == "primary"]
        .groupby(["layer", "order_k"], as_index=False)["test_mi"]
        .mean()
        .rename(columns={"test_mi": "primary_score_mean"})
    )
    legacy = (
        c11[c11["family"] == "null_control"]
        .groupby(["layer", "order_k"], as_index=False)["test_mi"]
        .mean()
        .rename(columns={"test_mi": "legacy_shift_null_mean"})
    )
    comparison = overall.merge(primary, on=["layer", "order_k"], how="left").merge(
        legacy, on=["layer", "order_k"], how="left"
    )
    if comparison[["primary_score_mean", "legacy_shift_null_mean"]].isna().any().any():
        raise RuntimeError("Could not align completed Code-11 primary/legacy-null results.")

    ge_counts = []
    for row in comparison.itertuples(index=False):
        vals = by_der[(by_der["layer"] == row.layer) & (by_der["order_k"] == row.order_k)]["score_mean"].to_numpy(float)
        ge_counts.append(int(np.sum(vals >= float(row.primary_score_mean))))
    comparison["random_null_derangements_ge_primary"] = ge_counts
    comparison["n_derangements"] = N_DERANGEMENTS
    comparison["primary_minus_random_null_mean"] = (
        comparison["primary_score_mean"] - comparison["random_null_mean"]
    )
    comparison["legacy_shift_minus_random_null_mean"] = (
        comparison["legacy_shift_null_mean"] - comparison["random_null_mean"]
    )
    comparison = comparison.sort_values(["layer", "order_k"]).reset_index(drop=True)
    return by_der, overall, comparison


def self_test() -> None:
    assert EXPECTED_FITS == 180
    assert DERANGEMENT_SEEDS == (101, 211, 307, 401, 503)

    # Determinism + no fixed points + bijection.
    p1 = make_derangement(17, 101, 0)
    p2 = make_derangement(17, 101, 0)
    np.testing.assert_array_equal(p1, p2)
    assert np.all(p1 != np.arange(17))
    np.testing.assert_array_equal(np.sort(p1), np.arange(17))

    # Different seed should produce a different derangement in this frozen test.
    p3 = make_derangement(17, 211, 0)
    assert not np.array_equal(p1, p3)

    # Cube permutation must preserve subset alignment within the source cube.
    n_cubes, n_subsets = 5, 4
    v = np.arange(n_cubes * n_subsets, dtype=np.float32)[:, None]
    p = make_derangement(n_cubes, 307, 2)
    out = permute_targets_by_cube(v, n_cubes, n_subsets, p).reshape(n_cubes, n_subsets)
    src = v.reshape(n_cubes, n_subsets)
    for i in range(n_cubes):
        np.testing.assert_array_equal(out[i], src[p[i]])
    assert not np.array_equal(out, src)

    # All five frozen permutation sets must be valid and unique per split.
    perms = build_permutation_sets()
    assert set(perms) == set(DERANGEMENT_SEEDS)
    for split in SPLIT_ORDER:
        hashes = [sha256_array_int64(perms[s][split]) for s in DERANGEMENT_SEEDS]
        assert len(set(hashes)) == N_DERANGEMENTS
        n = STRICT_CUBE_COUNTS[split]
        for s in DERANGEMENT_SEEDS:
            q = perms[s][split]
            assert q.shape == (n,)
            assert np.all(q != np.arange(n))
            np.testing.assert_array_equal(np.sort(q), np.arange(n))

    print("SELF-TEST: PASS")
    print(
        "Verified: deterministic fixed-point-free bijections, subset-preserving cube permutation, "
        "five unique frozen derangements per split, and 180-fit grid arithmetic."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--chunk-cubes", type=int, default=64)
    ap.add_argument("--code11-source", type=Path)
    ap.add_argument("--code11-manifest", type=Path)
    ap.add_argument("--code11-seed-results", type=Path)
    ap.add_argument("--code04-source", type=Path)
    ap.add_argument("--code08-manifest", type=Path)
    ap.add_argument("--code09-manifest", type=Path)
    ap.add_argument("--code10-manifest", type=Path)
    ap.add_argument("--output-dir", type=Path)
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return
    if a.chunk_cubes < 1:
        raise ValueError("--chunk-cubes must be >= 1")

    root = project_root()
    exp = root / "04 Experiments" / "EXP02_GB1_ESM2"
    out = a.output_dir or exp
    out.mkdir(parents=True, exist_ok=True)

    p11s = a.code11_source or root / "03 Code" / "11_run_gb1_target_mi.py"
    p11m = a.code11_manifest or exp / "gb1_target_mi_manifest.json"
    p11r = a.code11_seed_results or exp / "gb1_target_mi_seed_results.csv"
    p04s = a.code04_source or root / "03 Code" / "04_validate_mi_estimator.py"
    p08 = a.code08_manifest or exp / "gb1_esm2_extraction_manifest.json"
    p09 = a.code09_manifest or exp / "ESM2_Site_Reps" / "esm2_extraction_manifest.json"
    p10 = a.code10_manifest or exp / "gb1_order_targets_manifest.json"

    seed_out = out / "gb1_random_derangement_null_seed_results.csv"
    by_der_out = out / "gb1_random_derangement_null_by_derangement.csv"
    summary_out = out / "gb1_random_derangement_null_summary.csv"
    comparison_out = out / "gb1_random_derangement_null_comparison.csv"
    state_out = out / "gb1_random_derangement_null_state.json"
    manifest_out = out / "gb1_random_derangement_null_manifest.json"

    if a.overwrite:
        for p in (seed_out, by_der_out, summary_out, comparison_out, state_out, manifest_out):
            if p.exists():
                p.unlink()

    print("=" * 78)
    print("ORBIT-FMIB — GB1 RANDOMIZED CUBE-DERANGEMENT NULL DIAGNOSTIC")
    print("=" * 78)

    (
        code11,
        mine04,
        c11,
        c08,
        c09,
        c10,
        maps,
        samples,
        permutations,
        perm_meta,
    ) = preflight(p11s, p11m, p11r, p04s, p08, p09, p10)

    torch, nn = mine04.import_torch()
    device = mine04.choose_device(torch, a.device)
    if device.type == "cpu":
        torch.set_num_threads(1)

    fp = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_source_sha256": sha256_file(Path(__file__)),
        "code11_source_sha256": EXPECTED_CODE11_SHA256,
        "code11_manifest_sha256": c11["manifest_sha"],
        "code11_seed_results_sha256": c11["seed_sha"],
        "code04_source_sha256": sha256_file(p04s),
        "code08_manifest_sha256": c08["sha"],
        "code09_manifest_sha256": c09["sha"],
        "code10_manifest_sha256": c10["sha"],
        "layers": list(NULL_LAYERS),
        "orders": list(ORDERS),
        "critic_seeds": list(CRITIC_SEEDS),
        "derangement_seeds": list(DERANGEMENT_SEEDS),
        "representation_dim": REPRESENTATION_DIM,
        "device_type": device.type,
        "chunk_cubes": int(a.chunk_cubes),
        "null_design": "split-preserving cube-level random derangement; same subset mask retained",
        "permutation_sets": perm_meta,
        "absolute_mi_accuracy_claimed": False,
        "null_subtraction_used": False,
        "scientific_acceptance_threshold_predeclared": False,
    }
    fpsha = sha256_obj(fp)

    print("\nPreflight/provenance gate: PASS")
    print("  completed Code-11 v2 manifest + seed table: PASS")
    print("  exact Code-04 estimator source pinned by Code 11: PASS")
    print("  Code-08/09/10 provenance pinned by Code 11: PASS")
    print("  diagnostic layers 0/15/30 file hashes: PASS")
    print("  five deterministic derangements per split: PASS")
    print("  fixed points in every derangement: 0")
    print("  train/validation/test boundaries preserved: YES")
    print("  subset identity preserved under cube permutation: YES")
    print("  absolute MI accuracy claimed: NO")
    print("  null subtraction: NO")
    print("  post-hoc scientific PASS threshold: NONE")
    print(f"  planned critic fits: {EXPECTED_FITS}")
    print(f"  device: {device}")
    print(f"  fingerprint: {fpsha}")

    if a.preflight_only:
        print("\nPREFLIGHT: PASS\nNo critic was trained and no diagnostic outputs were written.")
        return

    if manifest_out.exists() and not a.overwrite:
        mf = load_json(manifest_out)
        if mf.get("run_fingerprint_sha256") == fpsha and bool(mf.get("overall_pass", False)):
            print(f"\n{CODE_NAME}: PASS (computational)\nMatching complete diagnostic already exists.")
            return
        raise RuntimeError("Existing randomized-null manifest does not match this run. Use --overwrite only if intentional.")

    if state_out.exists() and not a.overwrite:
        st = load_json(state_out)
        if st.get("run_fingerprint_sha256") != fpsha:
            raise RuntimeError("Existing randomized-null state fingerprint mismatch.")
        if not a.resume:
            raise RuntimeError("Existing randomized-null state detected. Re-run with --resume.")
    elif not state_out.exists():
        write_json_atomic(
            state_out,
            {
                "code_name": CODE_NAME,
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint_sha256": fpsha,
                "status": "initialized",
                "created_unix_time": time.time(),
                "completed_fits": 0,
                "expected_fits": EXPECTED_FITS,
            },
        )

    d = load_results(seed_out)
    done = {
        (int(r.derangement_seed), int(r.layer), int(r.order_k), int(r.critic_seed))
        for r in d.itertuples(index=False)
    }
    if not done.issubset(expected_keys()):
        raise RuntimeError("Existing randomized-null CSV contains unexpected conditions.")

    X = code11.chi_matrix()

    for layer_id in NULL_LAYERS:
        layer_needed = any(
            (ds, layer_id, k, cs) not in done
            for ds in DERANGEMENT_SEEDS
            for k in ORDERS
            for cs in CRITIC_SEEDS
        )
        if not layer_needed:
            continue
        print(f"\nLayer {layer_id:02d}", flush=True)
        layer = np.load(c09["layer_paths"][layer_id], mmap_mode="r", allow_pickle=False)

        for k in ORDERS:
            missing_here = [
                (ds, cs)
                for ds in DERANGEMENT_SEEDS
                for cs in CRITIC_SEEDS
                if (ds, layer_id, k, cs) not in done
            ]
            if not missing_here:
                continue
            print(f"  k={k}: constructing 2560-D R_l,C,S ...", flush=True)
            U, V, split, counts = code11.build_dataset(layer, maps, samples, k, X, a.chunk_cubes)
            tr, va, te = split

            for dseed in DERANGEMENT_SEEDS:
                critic_missing = [
                    cs for cs in CRITIC_SEEDS if (dseed, layer_id, k, cs) not in done
                ]
                if not critic_missing:
                    continue
                Vnull = build_random_null(code11, samples, k, permutations[dseed])
                if Vnull.shape != V.shape:
                    raise RuntimeError("Randomized-null target shape mismatch.")
                if np.array_equal(Vnull, V):
                    raise RuntimeError("Randomized-null target unexpectedly equals primary target.")
                psha = permutation_set_sha256(permutations[dseed])

                for critic_seed in critic_missing:
                    print(
                        f"    derangement={dseed} critic_seed={critic_seed}",
                        flush=True,
                    )
                    res = mine04.train_mine(
                        torch,
                        nn,
                        U,
                        Vnull,
                        int(critic_seed),
                        device,
                        split,
                    )
                    row = {
                        "family": "random_derangement_null",
                        "derangement_seed": int(dseed),
                        "permutation_set_sha256": psha,
                        "layer": int(layer_id),
                        "order_k": int(k),
                        "critic_seed": int(critic_seed),
                        "representation_dim": REPRESENTATION_DIM,
                        "target_dim": 1,
                        "n_samples": len(U),
                        "n_train": len(tr),
                        "n_val": len(va),
                        "n_test": len(te),
                        "subset_masks": ";".join(map(str, code11.SUBSETS_BY_ORDER[k])),
                        **res,
                    }
                    d = save_result_row(d, row, seed_out)
                    done.add((int(dseed), int(layer_id), int(k), int(critic_seed)))
                    write_json_atomic(
                        state_out,
                        {
                            "code_name": CODE_NAME,
                            "schema_version": SCHEMA_VERSION,
                            "run_fingerprint_sha256": fpsha,
                            "status": "running",
                            "completed_fits": len(done),
                            "expected_fits": EXPECTED_FITS,
                            "last_completed": {
                                "derangement_seed": int(dseed),
                                "layer": int(layer_id),
                                "order_k": int(k),
                                "critic_seed": int(critic_seed),
                            },
                        },
                    )
            del U, V
        del layer

    audit_seed_results(d)
    by_der, summary, comparison = make_summaries(d, c11["seed_df"])

    write_csv_atomic(by_der, by_der_out)
    write_csv_atomic(summary, summary_out)
    write_csv_atomic(comparison, comparison_out)

    output_paths = (seed_out, by_der_out, summary_out, comparison_out)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "purpose": (
            "Diagnostic re-evaluation of the GB1 null using five split-preserving "
            "cube-level random derangements at layers 0, 15, and 30."
        ),
        "diagnostic_source_sha256": sha256_file(Path(__file__)),
        "run_fingerprint": fp,
        "run_fingerprint_sha256": fpsha,
        "completed_code11": {
            "manifest_sha256": c11["manifest_sha"],
            "source_sha256": EXPECTED_CODE11_SHA256,
            "seed_results_sha256": c11["seed_sha"],
            "overall_pass": True,
        },
        "null_contract": {
            "layers": list(NULL_LAYERS),
            "orders": list(ORDERS),
            "critic_seeds": list(CRITIC_SEEDS),
            "derangement_seeds": list(DERANGEMENT_SEEDS),
            "n_derangements": N_DERANGEMENTS,
            "expected_critic_fits": EXPECTED_FITS,
            "split_preserving": True,
            "cube_level_bijection": True,
            "fixed_points_allowed": False,
            "same_subset_mask_preserved": True,
            "same_permutation_reused_across_orders_and_layers_for_each_seed": True,
            "permutation_sets": perm_meta,
        },
        "interpretation_contract": {
            "absolute_mi_accuracy_claimed": False,
            "no_pca_or_projection": True,
            "no_clamping": True,
            "no_null_subtraction": True,
            "scientific_acceptance_threshold_predeclared": False,
            "random_null_derangements_ge_primary_is_descriptive_not_a_p_value": True,
            "critic_seed_sd_is_estimator_stability_not_biological_uncertainty": True,
        },
        "outputs": {p.stem: p.name for p in output_paths},
        "file_sha256": {p.name: sha256_file(p) for p in output_paths},
        "computational_audit": {
            "expected_fits": EXPECTED_FITS,
            "observed_fits": int(len(d)),
            "per_derangement_cells": int(len(by_der)),
            "layer_order_cells": int(len(summary)),
            "all_results_finite": True,
            "all_derangements_fixed_point_free": True,
            "overall_pass": True,
        },
        "overall_pass": True,
        "interpretation_limits": [
            "This PASS is computational/provenance only; it is not a post-hoc scientific claim that the null is adequate.",
            "Randomized-null scores are diagnostics and are not subtracted from primary Code-11 scores.",
            "MINE/DV values are comparative lower-bound scores, not exact mutual information in nats.",
            "Five derangements characterize null sensitivity descriptively; they are not treated as biological replicates.",
            "No cross-order raw-score comparison is licensed by this diagnostic.",
        ],
    }
    write_json_atomic(manifest_out, manifest)
    write_json_atomic(
        state_out,
        {
            "code_name": CODE_NAME,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint_sha256": fpsha,
            "status": "complete_pass",
            "completed_fits": int(len(d)),
            "expected_fits": EXPECTED_FITS,
            "final_manifest_sha256": sha256_file(manifest_out),
        },
    )

    print("\nComputational audit: PASS")
    print(f"  randomized-null critic fits: {len(d)}")
    print(f"  frozen derangements: {N_DERANGEMENTS}")
    print("  diagnostic layers: 0, 15, 30")
    print("  orders: 1, 2, 3, 4")
    print("  fixed points: 0")
    print("  clamping: NO")
    print("  null subtraction: NO")
    print("  absolute MI in nats claimed: NO")
    print("  scientific acceptance threshold: NONE (diagnostic only)")

    print("\nPrimary vs legacy-shift null vs randomized-null (descriptive):")
    show_cols = [
        "layer",
        "order_k",
        "primary_score_mean",
        "legacy_shift_null_mean",
        "random_null_mean",
        "random_null_sd_across_derangements",
        "random_null_derangements_ge_primary",
    ]
    with pd.option_context("display.max_rows", 20, "display.width", 180):
        print(comparison[show_cols].to_string(index=False))

    print(f"\n{CODE_NAME}: PASS (computational)")
    print("Randomized-null outputs are ready for scientific interpretation before Code 13.")


if __name__ == "__main__":
    main()
