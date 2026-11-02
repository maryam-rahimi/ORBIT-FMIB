#!/usr/bin/env python3
"""ORBIT-FMIB — GB1 subset-conditioned MI diagnostic.

Why this exists
---------------
The completed pooled Code-11 GB1 analysis combines all subset masks of a given
interaction order in one critic.  A randomized cube-derangement diagnostic
showed that pooled null scores can remain large for k=1..3 because subset
identity S is preserved and the empirical target distribution E differs
substantially across subset masks.

This companion diagnostic removes that shortcut by fixing one subset mask S
per critic.  For each S it estimates the same held-out neural DV/MINE score for

    I(R_{l,C,S}; E_{C,S})

across cubes C.  Since S is constant inside a critic, the critic cannot earn
score merely by identifying which subset is being evaluated.  A split-
preserving cube derangement is then a valid within-S null because it destroys
cube pairing while keeping the fixed conditioning stratum unchanged.

This script is DIAGNOSTIC ONLY.  It runs layers 0, 15, and 30 before any full
31-layer subset-conditioned analysis is attempted.  It defines no post-hoc
scientific PASS threshold.  Absolute MI accuracy in nats is NOT claimed.

Recommended sequence
--------------------
Local:
  python "03 Code/11_validate_gb1_subset_conditioning.py" --self-test
  python "03 Code/11_validate_gb1_subset_conditioning.py" --preflight-only

Newton / GPU:
  python "03 Code/11_validate_gb1_subset_conditioning.py" --preflight-only --device cuda
  python "03 Code/11_validate_gb1_subset_conditioning.py" --device cuda

If interrupted:
  python "03 Code/11_validate_gb1_subset_conditioning.py" --device cuda --resume

Use --overwrite only for an intentional clean restart of THIS diagnostic.
It does not modify completed Code-11 or Code-12 outputs.
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
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

CODE_NAME = "ORBIT-FMIB GB1 SUBSET-CONDITIONED MI DIAGNOSTIC"
SCHEMA_VERSION = 1

EXPECTED_CODE11_NAME = "ORBIT-FMIB CODE 11"
EXPECTED_CODE11_SCHEMA = 2
EXPECTED_CODE11_SHA256 = "5d98333fb5a129331199e07b808a3457171eb2f3349b7ea4b24fe185d6c99cd3"

DIAGNOSTIC_LAYERS = (0, 15, 30)
ORDERS = (1, 2, 3, 4)
ALL_MASKS = tuple(range(1, 16))
SUBSETS_BY_ORDER = {k: tuple(m for m in ALL_MASKS if int(m).bit_count() == k) for k in ORDERS}
CRITIC_SEEDS = (11, 23, 37)
DERANGEMENT_SEEDS = (101, 211, 307, 401, 503)
REPRESENTATION_DIM = 2560

SPLIT_ORDER = ("estimator_train", "estimator_val", "estimator_test")
SPLIT_CODES = {"estimator_train": 0, "estimator_val": 1, "estimator_test": 2}
STRICT_CUBE_COUNTS = {"estimator_train": 7352, "estimator_val": 245, "estimator_test": 613}
N_PER_SUBSET = sum(STRICT_CUBE_COUNTS.values())

EXPECTED_PRIMARY_FITS = len(DIAGNOSTIC_LAYERS) * len(ALL_MASKS) * len(CRITIC_SEEDS)
EXPECTED_NULL_FITS = (
    len(DIAGNOSTIC_LAYERS)
    * len(ALL_MASKS)
    * len(DERANGEMENT_SEEDS)
    * len(CRITIC_SEEDS)
)
EXPECTED_FITS = EXPECTED_PRIMARY_FITS + EXPECTED_NULL_FITS

RESULT_COLS = (
    "family",
    "derangement_seed",
    "permutation_set_sha256",
    "layer",
    "order_k",
    "subset_mask",
    "critic_seed",
    "representation_dim",
    "target_dim",
    "n_samples",
    "n_train",
    "n_val",
    "n_test",
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
    spec = importlib.util.spec_from_file_location("_orbitfmib_code11_subset_diag", path)
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
        "compute_repr_walsh",
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
    if tuple(mod.SPLIT_ORDER) != SPLIT_ORDER:
        raise RuntimeError("Code-11 split order changed.")
    if dict(mod.STRICT_CUBE_COUNTS) != STRICT_CUBE_COUNTS:
        raise RuntimeError("Code-11 split cube counts changed.")
    frozen = {int(k): tuple(map(int, v)) for k, v in mod.SUBSETS_BY_ORDER.items()}
    if frozen != SUBSETS_BY_ORDER:
        raise RuntimeError("Code-11 subset-mask contract changed.")
    return mod


def verify_completed_code11(manifest_path: Path, seed_path: Path, source_path: Path) -> Dict:
    m = load_json(manifest_path)
    msha = sha256_file(manifest_path)
    if m.get("code_name") != EXPECTED_CODE11_NAME:
        raise RuntimeError("Completed Code-11 manifest has wrong code_name.")
    if int(m.get("schema_version", -1)) != EXPECTED_CODE11_SCHEMA:
        raise RuntimeError("Completed Code-11 manifest schema mismatch.")
    if not bool(m.get("overall_pass", False)):
        raise RuntimeError("Completed Code-11 manifest is not PASS.")
    if str(m.get("code11_source_sha256", "")).lower() != EXPECTED_CODE11_SHA256.lower():
        raise RuntimeError("Completed Code-11 manifest does not pin expected source.")
    verify_hash(source_path, EXPECTED_CODE11_SHA256, "Code-11 source")

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

    out_name = m.get("outputs", {}).get("biological_seed_results")
    if not out_name or Path(out_name).name != seed_path.name:
        raise RuntimeError("Code-11 seed-results filename mismatch.")
    expected_seed_sha = m.get("file_sha256", {}).get(seed_path.name)
    if not expected_seed_sha:
        raise RuntimeError("Code-11 seed-results hash missing.")
    verify_hash(seed_path, expected_seed_sha, "Code-11 biological seed results")

    d = pd.read_csv(seed_path)
    if len(d) != 408:
        raise RuntimeError("Code-11 seed table must contain exactly 408 rows.")
    if int((d["family"] == "primary").sum()) != 372:
        raise RuntimeError("Code-11 primary row count mismatch.")
    if int((d["family"] == "null_control").sum()) != 36:
        raise RuntimeError("Code-11 legacy-null row count mismatch.")
    return {
        "manifest": m,
        "manifest_sha": msha,
        "seed_df": d,
        "seed_sha": str(expected_seed_sha),
    }


def make_derangement(n: int, seed: int, split_code: int) -> np.ndarray:
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
    raise RuntimeError(f"Could not generate derangement for n={n}, seed={seed}.")


def build_permutation_sets() -> Dict[int, Dict[str, np.ndarray]]:
    out: Dict[int, Dict[str, np.ndarray]] = {}
    seen_hashes = {s: set() for s in SPLIT_ORDER}
    for dseed in DERANGEMENT_SEEDS:
        per_split = {}
        for split in SPLIT_ORDER:
            p = make_derangement(STRICT_CUBE_COUNTS[split], int(dseed), int(SPLIT_CODES[split]))
            h = sha256_array_int64(p)
            if h in seen_hashes[split]:
                raise RuntimeError(f"Duplicate derangement for split={split}.")
            seen_hashes[split].add(h)
            per_split[split] = p
        out[int(dseed)] = per_split
    return out


def permutation_set_sha256(per_split: Mapping[str, np.ndarray]) -> str:
    return sha256_obj({s: sha256_array_int64(per_split[s]) for s in SPLIT_ORDER})


def target_for_subset(code11, samples: Mapping, split: str, mask: int) -> np.ndarray:
    k = int(mask).bit_count()
    masks = tuple(map(int, code11.SUBSETS_BY_ORDER[k]))
    if mask not in masks:
        raise RuntimeError(f"Mask {mask} not present in order k={k} contract.")
    j = masks.index(mask)
    n = STRICT_CUBE_COUNTS[split]
    pooled = code11.targets_for(samples, split, k).reshape(n, len(masks), 1)
    out = np.asarray(pooled[:, j, :], dtype=np.float32)
    if out.shape != (n, 1) or not np.isfinite(out).all():
        raise RuntimeError(f"Invalid target vector for split={split}, mask={mask}.")
    return out


def build_subset_dataset(
    code11,
    layer: np.ndarray,
    maps: Mapping[str, np.ndarray],
    samples: Mapping,
    mask: int,
    X: np.ndarray,
    chunk: int,
):
    us, vs, idxs = [], [], []
    counts = {}
    off = 0
    for split in SPLIT_ORDER:
        u = code11.compute_repr_walsh(layer, maps[split], (int(mask),), X, chunk)
        v = target_for_subset(code11, samples, split, int(mask))
        n = STRICT_CUBE_COUNTS[split]
        if u.shape != (n, REPRESENTATION_DIM):
            raise RuntimeError(f"Representation shape mismatch for split={split}, mask={mask}: {u.shape}")
        if len(u) != len(v):
            raise RuntimeError(f"R/E count mismatch for split={split}, mask={mask}.")
        us.append(u)
        vs.append(v)
        idxs.append(np.arange(off, off + n, dtype=np.int64))
        counts[split] = n
        off += n
    U = np.concatenate(us)
    V = np.concatenate(vs)
    if U.shape != (N_PER_SUBSET, REPRESENTATION_DIM) or V.shape != (N_PER_SUBSET, 1):
        raise RuntimeError("Combined fixed-subset dataset shape mismatch.")
    if not np.isfinite(U).all() or not np.isfinite(V).all():
        raise RuntimeError("Non-finite fixed-subset dataset.")
    split_idx = (idxs[0], idxs[1], idxs[2])
    if any(len(np.intersect1d(split_idx[i], split_idx[j])) for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Estimator split overlap.")
    return U, V, split_idx, counts


def build_subset_null(
    code11,
    samples: Mapping,
    mask: int,
    per_split: Mapping[str, np.ndarray],
) -> np.ndarray:
    pieces = []
    for split in SPLIT_ORDER:
        v = target_for_subset(code11, samples, split, int(mask))
        p = np.asarray(per_split[split], dtype=np.int64)
        n = STRICT_CUBE_COUNTS[split]
        base = np.arange(n, dtype=np.int64)
        if p.shape != (n,) or np.any(p == base) or not np.array_equal(np.sort(p), base):
            raise RuntimeError("Invalid fixed-subset null permutation.")
        pieces.append(v[p].copy())
    out = np.concatenate(pieces)
    if out.shape != (N_PER_SUBSET, 1):
        raise RuntimeError("Fixed-subset null shape mismatch.")
    return out


def subset_target_eta2(code11, samples: Mapping) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        for k in ORDERS:
            masks = SUBSETS_BY_ORDER[k]
            pooled = code11.targets_for(samples, split, k).reshape(
                STRICT_CUBE_COUNTS[split], len(masks)
            ).astype(np.float64)
            flat = pooled.reshape(-1)
            grand = float(flat.mean())
            ss_total = float(np.sum((flat - grand) ** 2))
            if len(masks) == 1:
                eta2 = np.nan
            else:
                means = pooled.mean(axis=0)
                n = pooled.shape[0]
                ss_between = float(np.sum(n * (means - grand) ** 2))
                eta2 = ss_between / ss_total if ss_total > 0 else np.nan
            rows.append(
                {
                    "split": split,
                    "order_k": int(k),
                    "n_subset_masks": len(masks),
                    "n_samples": int(flat.size),
                    "subset_mean_effect_eta2": eta2,
                }
            )
    return pd.DataFrame(rows)


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
        raise RuntimeError("Current Code-08 manifest is not pinned by Code 11.")
    if str(c08["map_sha"]).lower() != str(up.get("code08_map_sha256", "")).lower():
        raise RuntimeError("Current Code-08 map is not pinned by Code 11.")
    maps = code11.load_cube_maps(c08["map_path"])

    c09 = code11.verify_code09(code09_manifest, c08["sha"], False)
    if c09["sha"].lower() != str(up.get("code09_manifest_sha256", "")).lower():
        raise RuntimeError("Current Code-09 manifest is not pinned by Code 11.")
    layer_names = c09["manifest"].get("outputs", {}).get("layer_files", {})
    layer_hashes = c09["manifest"].get("file_sha256", {})
    for layer in DIAGNOSTIC_LAYERS:
        name = layer_names.get(str(layer))
        if not name:
            raise RuntimeError(f"Code-09 manifest missing layer {layer} filename.")
        expected = layer_hashes.get(Path(name).name)
        if not expected:
            raise RuntimeError(f"Code-09 manifest missing layer {layer} hash.")
        verify_hash(c09["layer_paths"][layer], expected, f"Code-09 layer {layer}")

    c10 = code11.verify_code10(code10_manifest, c08["manifest"])
    if c10["sha"].lower() != str(up.get("code10_manifest_sha256", "")).lower():
        raise RuntimeError("Current Code-10 manifest is not pinned by Code 11.")
    if str(c10["samples_sha"]).lower() != str(up.get("code10_samples_sha256", "")).lower():
        raise RuntimeError("Current Code-10 samples are not pinned by Code 11.")
    samples = code11.load_order_samples(c10["samples_path"])

    for split in SPLIT_ORDER:
        if len(samples["cube_ids_by_split"][split]) != len(maps[split]):
            raise RuntimeError(f"{split}: Code08/10 cube-row count mismatch.")

    perms = build_permutation_sets()
    perm_meta = {}
    for dseed, per_split in perms.items():
        sm = {}
        for split in SPLIT_ORDER:
            p = per_split[split]
            n = STRICT_CUBE_COUNTS[split]
            fixed = int(np.sum(p == np.arange(n, dtype=np.int64)))
            if fixed != 0:
                raise RuntimeError("Derangement unexpectedly contains fixed points.")
            sm[split] = {
                "n_cubes": n,
                "fixed_points": fixed,
                "permutation_sha256": sha256_array_int64(p),
            }
        perm_meta[str(dseed)] = {
            "permutation_set_sha256": permutation_set_sha256(per_split),
            "splits": sm,
        }

    eta2 = subset_target_eta2(code11, samples)
    return code11, mine04, c11, c08, c09, c10, maps, samples, perms, perm_meta, eta2


def expected_keys() -> set:
    out = set()
    for layer in DIAGNOSTIC_LAYERS:
        for mask in ALL_MASKS:
            for cs in CRITIC_SEEDS:
                out.add(("primary", 0, layer, mask, cs))
            for ds in DERANGEMENT_SEEDS:
                for cs in CRITIC_SEEDS:
                    out.add(("random_derangement_null", ds, layer, mask, cs))
    return out


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLS)
    d = pd.read_csv(path)
    missing = set(RESULT_COLS) - set(d.columns)
    if missing:
        raise RuntimeError(f"Existing subset-conditioned CSV missing columns: {sorted(missing)}")
    if d.duplicated(["family", "derangement_seed", "layer", "subset_mask", "critic_seed"]).any():
        raise RuntimeError("Duplicate subset-conditioned critic rows.")
    return d


def save_result_row(d: pd.DataFrame, row: Mapping, path: Path) -> pd.DataFrame:
    d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    d = d.sort_values(
        ["layer", "order_k", "subset_mask", "family", "derangement_seed", "critic_seed"]
    ).reset_index(drop=True)
    write_csv_atomic(d, path)
    return d


def audit_seed_results(d: pd.DataFrame) -> Dict:
    obs = {
        (str(r.family), int(r.derangement_seed), int(r.layer), int(r.subset_mask), int(r.critic_seed))
        for r in d.itertuples(index=False)
    }
    exp = expected_keys()
    if obs != exp:
        raise RuntimeError(f"Subset-conditioned fit grid mismatch: observed={len(obs)}, expected={len(exp)}.")
    nums = [
        "best_step",
        "final_training_step",
        "best_val_mi",
        "test_mi",
        "test_product_shuffle_sd",
        "heldout_joint_minus_mismatch_score",
    ]
    if not np.isfinite(d[nums].to_numpy(float)).all():
        raise RuntimeError("Non-finite subset-conditioned diagnostics.")
    if not (d["representation_dim"].astype(int) == REPRESENTATION_DIM).all():
        raise RuntimeError("Representation dimension mismatch.")
    if not (d["n_samples"].astype(int) == N_PER_SUBSET).all():
        raise RuntimeError("Per-subset total sample count mismatch.")
    if not (d["n_train"].astype(int) == STRICT_CUBE_COUNTS["estimator_train"]).all():
        raise RuntimeError("Per-subset train count mismatch.")
    if not (d["n_val"].astype(int) == STRICT_CUBE_COUNTS["estimator_val"]).all():
        raise RuntimeError("Per-subset validation count mismatch.")
    if not (d["n_test"].astype(int) == STRICT_CUBE_COUNTS["estimator_test"]).all():
        raise RuntimeError("Per-subset test count mismatch.")
    if not np.array_equal(
        d["order_k"].astype(int).to_numpy(),
        d["subset_mask"].astype(int).map(int.bit_count).to_numpy(),
    ):
        raise RuntimeError("order_k does not match subset_mask bit count.")
    p = int((d["family"] == "primary").sum())
    n = int((d["family"] == "random_derangement_null").sum())
    if p != EXPECTED_PRIMARY_FITS or n != EXPECTED_NULL_FITS:
        raise RuntimeError("Primary/null fit count mismatch.")
    return {"primary": p, "random_derangement_null": n, "total": len(d)}


def make_summaries(d: pd.DataFrame, pooled_code11: pd.DataFrame):
    primary = d[d["family"] == "primary"].copy()
    null = d[d["family"] == "random_derangement_null"].copy()

    psub = (
        primary.groupby(["layer", "order_k", "subset_mask"], as_index=False)
        .agg(
            primary_score_mean=("test_mi", "mean"),
            primary_critic_seed_sd=("test_mi", lambda x: float(np.std(np.asarray(x, float), ddof=1))),
            primary_score_min=("test_mi", "min"),
            primary_score_max=("test_mi", "max"),
            mean_best_val=("best_val_mi", "mean"),
            mean_best_step=("best_step", "mean"),
        )
        .sort_values(["layer", "order_k", "subset_mask"])
        .reset_index(drop=True)
    )
    if len(psub) != len(DIAGNOSTIC_LAYERS) * len(ALL_MASKS):
        raise RuntimeError("Primary subset summary row count mismatch.")

    nder = (
        null.groupby(["derangement_seed", "permutation_set_sha256", "layer", "order_k", "subset_mask"], as_index=False)
        .agg(
            null_score_mean=("test_mi", "mean"),
            null_critic_seed_sd=("test_mi", lambda x: float(np.std(np.asarray(x, float), ddof=1))),
        )
        .sort_values(["layer", "order_k", "subset_mask", "derangement_seed"])
        .reset_index(drop=True)
    )
    expected_nder = len(DIAGNOSTIC_LAYERS) * len(ALL_MASKS) * len(DERANGEMENT_SEEDS)
    if len(nder) != expected_nder:
        raise RuntimeError("Null per-derangement summary row count mismatch.")

    nsub = (
        nder.groupby(["layer", "order_k", "subset_mask"], as_index=False)
        .agg(
            random_null_mean=("null_score_mean", "mean"),
            random_null_sd_across_derangements=("null_score_mean", lambda x: float(np.std(np.asarray(x, float), ddof=1))),
            random_null_min=("null_score_mean", "min"),
            random_null_max=("null_score_mean", "max"),
        )
        .sort_values(["layer", "order_k", "subset_mask"])
        .reset_index(drop=True)
    )

    comp = psub.merge(nsub, on=["layer", "order_k", "subset_mask"], how="left")
    ge = []
    for r in comp.itertuples(index=False):
        vals = nder[
            (nder["layer"] == r.layer)
            & (nder["subset_mask"] == r.subset_mask)
        ]["null_score_mean"].to_numpy(float)
        ge.append(int(np.sum(vals >= float(r.primary_score_mean))))
    comp["random_null_derangements_ge_primary"] = ge
    comp["n_derangements"] = len(DERANGEMENT_SEEDS)
    comp["primary_minus_random_null_mean"] = comp["primary_score_mean"] - comp["random_null_mean"]

    order_desc = (
        comp.groupby(["layer", "order_k"], as_index=False)
        .agg(
            n_subset_masks=("subset_mask", "count"),
            descriptive_primary_mean_across_subsets=("primary_score_mean", "mean"),
            descriptive_random_null_mean_across_subsets=("random_null_mean", "mean"),
            mean_primary_minus_random_null=("primary_minus_random_null_mean", "mean"),
            subsets_primary_gt_random_null_mean=("primary_minus_random_null_mean", lambda x: int(np.sum(np.asarray(x, float) > 0))),
        )
        .sort_values(["layer", "order_k"])
        .reset_index(drop=True)
    )

    pooled = (
        pooled_code11[
            (pooled_code11["family"] == "primary")
            & (pooled_code11["layer"].astype(int).isin(DIAGNOSTIC_LAYERS))
        ]
        .groupby(["layer", "order_k"], as_index=False)["test_mi"]
        .mean()
        .rename(columns={"test_mi": "pooled_code11_primary_score_mean"})
    )
    order_desc = order_desc.merge(pooled, on=["layer", "order_k"], how="left")
    if order_desc["pooled_code11_primary_score_mean"].isna().any():
        raise RuntimeError("Could not align pooled Code-11 primary scores.")

    return psub, nder, comp, order_desc


def self_test() -> None:
    assert tuple(sorted(m for masks in SUBSETS_BY_ORDER.values() for m in masks)) == ALL_MASKS
    assert {k: len(v) for k, v in SUBSETS_BY_ORDER.items()} == {1: 4, 2: 6, 3: 4, 4: 1}
    assert N_PER_SUBSET == 8210
    assert EXPECTED_PRIMARY_FITS == 135
    assert EXPECTED_NULL_FITS == 675
    assert EXPECTED_FITS == 810

    p1 = make_derangement(17, 101, 0)
    p2 = make_derangement(17, 101, 0)
    np.testing.assert_array_equal(p1, p2)
    assert np.all(p1 != np.arange(17))
    np.testing.assert_array_equal(np.sort(p1), np.arange(17))
    assert not np.array_equal(p1, make_derangement(17, 211, 0))

    # Synthetic pooled target array: cube-major, subset-minor.  Verify fixed-mask
    # extraction and cube derangement preserve the chosen conditioning stratum.
    n, masks = 5, (1, 2, 4, 8)
    pooled = np.arange(n * len(masks), dtype=np.float32).reshape(n, len(masks), 1)
    j = masks.index(4)
    v = pooled[:, j, :]
    p = make_derangement(n, 307, 2)
    vnull = v[p]
    for i in range(n):
        np.testing.assert_array_equal(vnull[i], v[p[i]])
    assert not np.array_equal(vnull, v)

    print("SELF-TEST: PASS")
    print(
        "Verified: all 15 subset masks partition by order, 8,210 samples per fixed-subset critic, "
        "deterministic fixed-point-free cube derangements, fixed-subset null construction, and 135+675=810 fit-grid arithmetic."
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

    seed_out = out / "gb1_subset_conditioned_diag_seed_results.csv"
    primary_out = out / "gb1_subset_conditioned_diag_primary_summary.csv"
    null_by_der_out = out / "gb1_subset_conditioned_diag_null_by_derangement.csv"
    comparison_out = out / "gb1_subset_conditioned_diag_comparison.csv"
    order_out = out / "gb1_subset_conditioned_diag_order_descriptive.csv"
    eta2_out = out / "gb1_subset_conditioned_diag_subset_effects.csv"
    state_out = out / "gb1_subset_conditioned_diag_state.json"
    manifest_out = out / "gb1_subset_conditioned_diag_manifest.json"

    if a.overwrite:
        for p in (
            seed_out,
            primary_out,
            null_by_der_out,
            comparison_out,
            order_out,
            eta2_out,
            state_out,
            manifest_out,
        ):
            if p.exists():
                p.unlink()

    print("=" * 82)
    print("ORBIT-FMIB — GB1 SUBSET-CONDITIONED MI DIAGNOSTIC")
    print("=" * 82)

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
        eta2,
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
        "layers": list(DIAGNOSTIC_LAYERS),
        "subset_masks": list(ALL_MASKS),
        "subsets_by_order": {str(k): list(v) for k, v in SUBSETS_BY_ORDER.items()},
        "critic_seeds": list(CRITIC_SEEDS),
        "derangement_seeds": list(DERANGEMENT_SEEDS),
        "representation_dim": REPRESENTATION_DIM,
        "samples_per_fixed_subset": N_PER_SUBSET,
        "device_type": device.type,
        "chunk_cubes": int(a.chunk_cubes),
        "analysis_contract": "fixed-subset critics; cube variation only within S",
        "null_design": "within-fixed-S split-preserving cube-level random derangement",
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
    print("  diagnostic layers 0/15/30 hashes: PASS")
    print("  fixed subset mask per critic: YES")
    print("  samples per subset critic: train=7352 val=245 test=613 total=8210")
    print("  five split-preserving cube derangements: PASS")
    print("  fixed points in every derangement: 0")
    print("  subset identity cannot vary within a critic: YES")
    print("  absolute MI accuracy claimed: NO")
    print("  null subtraction: NO")
    print("  post-hoc scientific PASS threshold: NONE")
    print(f"  planned primary fits: {EXPECTED_PRIMARY_FITS}")
    print(f"  planned randomized-null fits: {EXPECTED_NULL_FITS}")
    print(f"  planned total fits: {EXPECTED_FITS}")
    print(f"  device: {device}")
    print(f"  fingerprint: {fpsha}")

    print("\nObserved target subset-identity mean effects (eta^2; descriptive):")
    for r in eta2.itertuples(index=False):
        val = "N/A" if not np.isfinite(r.subset_mean_effect_eta2) else f"{r.subset_mean_effect_eta2:.6f}"
        print(f"  {r.split:15s} k={int(r.order_k)}  masks={int(r.n_subset_masks)}  eta2={val}")

    if a.preflight_only:
        print("\nPREFLIGHT: PASS\nNo critic was trained and no diagnostic outputs were written.")
        return

    if manifest_out.exists() and not a.overwrite:
        mf = load_json(manifest_out)
        if mf.get("run_fingerprint_sha256") == fpsha and bool(mf.get("overall_pass", False)):
            print(f"\n{CODE_NAME}: PASS (computational)\nMatching complete diagnostic already exists.")
            return
        raise RuntimeError("Existing subset-conditioned manifest does not match this run.")

    if state_out.exists() and not a.overwrite:
        st = load_json(state_out)
        if st.get("run_fingerprint_sha256") != fpsha:
            raise RuntimeError("Existing subset-conditioned state fingerprint mismatch.")
        if not a.resume:
            raise RuntimeError("Existing subset-conditioned state detected. Re-run with --resume.")
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
        (str(r.family), int(r.derangement_seed), int(r.layer), int(r.subset_mask), int(r.critic_seed))
        for r in d.itertuples(index=False)
    }
    if not done.issubset(expected_keys()):
        raise RuntimeError("Existing subset-conditioned CSV contains unexpected conditions.")

    X = code11.chi_matrix()

    for layer_id in DIAGNOSTIC_LAYERS:
        layer_needed = any(
            key not in done
            for key in expected_keys()
            if key[2] == layer_id
        )
        if not layer_needed:
            continue
        print(f"\nLayer {layer_id:02d}", flush=True)
        layer = np.load(c09["layer_paths"][layer_id], mmap_mode="r", allow_pickle=False)

        for mask in ALL_MASKS:
            k = int(mask).bit_count()
            primary_missing = [
                cs for cs in CRITIC_SEEDS if ("primary", 0, layer_id, mask, cs) not in done
            ]
            null_missing = [
                (ds, cs)
                for ds in DERANGEMENT_SEEDS
                for cs in CRITIC_SEEDS
                if ("random_derangement_null", ds, layer_id, mask, cs) not in done
            ]
            if not primary_missing and not null_missing:
                continue

            print(f"  mask={mask:02d} k={k}: constructing fixed-S 2560-D dataset ...", flush=True)
            U, V, split, counts = build_subset_dataset(
                code11, layer, maps, samples, mask, X, a.chunk_cubes
            )
            tr, va, te = split

            for cs in primary_missing:
                print(f"    primary critic_seed={cs}", flush=True)
                res = mine04.train_mine(torch, nn, U, V, int(cs), device, split)
                row = {
                    "family": "primary",
                    "derangement_seed": 0,
                    "permutation_set_sha256": "",
                    "layer": int(layer_id),
                    "order_k": int(k),
                    "subset_mask": int(mask),
                    "critic_seed": int(cs),
                    "representation_dim": REPRESENTATION_DIM,
                    "target_dim": 1,
                    "n_samples": len(U),
                    "n_train": len(tr),
                    "n_val": len(va),
                    "n_test": len(te),
                    **res,
                }
                d = save_result_row(d, row, seed_out)
                done.add(("primary", 0, int(layer_id), int(mask), int(cs)))
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
                            "family": "primary",
                            "layer": int(layer_id),
                            "subset_mask": int(mask),
                            "critic_seed": int(cs),
                        },
                    },
                )

            for ds in DERANGEMENT_SEEDS:
                cs_missing = [
                    cs
                    for cs in CRITIC_SEEDS
                    if ("random_derangement_null", ds, layer_id, mask, cs) not in done
                ]
                if not cs_missing:
                    continue
                Vnull = build_subset_null(code11, samples, mask, permutations[ds])
                if Vnull.shape != V.shape or np.array_equal(Vnull, V):
                    raise RuntimeError("Fixed-subset randomized null construction failed.")
                psha = permutation_set_sha256(permutations[ds])
                for cs in cs_missing:
                    print(f"    null derangement={ds} critic_seed={cs}", flush=True)
                    res = mine04.train_mine(torch, nn, U, Vnull, int(cs), device, split)
                    row = {
                        "family": "random_derangement_null",
                        "derangement_seed": int(ds),
                        "permutation_set_sha256": psha,
                        "layer": int(layer_id),
                        "order_k": int(k),
                        "subset_mask": int(mask),
                        "critic_seed": int(cs),
                        "representation_dim": REPRESENTATION_DIM,
                        "target_dim": 1,
                        "n_samples": len(U),
                        "n_train": len(tr),
                        "n_val": len(va),
                        "n_test": len(te),
                        **res,
                    }
                    d = save_result_row(d, row, seed_out)
                    done.add(("random_derangement_null", int(ds), int(layer_id), int(mask), int(cs)))
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
                                "family": "random_derangement_null",
                                "derangement_seed": int(ds),
                                "layer": int(layer_id),
                                "subset_mask": int(mask),
                                "critic_seed": int(cs),
                            },
                        },
                    )
            del U, V
        del layer

    counts = audit_seed_results(d)
    psub, nder, comp, order_desc = make_summaries(d, c11["seed_df"])

    write_csv_atomic(psub, primary_out)
    write_csv_atomic(nder, null_by_der_out)
    write_csv_atomic(comp, comparison_out)
    write_csv_atomic(order_desc, order_out)
    write_csv_atomic(eta2, eta2_out)

    outputs = (seed_out, primary_out, null_by_der_out, comparison_out, order_out, eta2_out)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "purpose": (
            "Test whether fixing subset mask S removes the subset-identity shortcut observed in pooled GB1 MI critics, "
            "using layers 0/15/30 and five split-preserving cube derangements."
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
        "conditioning_contract": {
            "conditioning_variable": "subset_mask S",
            "one_subset_mask_per_critic": True,
            "samples_per_subset": N_PER_SUBSET,
            "n_train": STRICT_CUBE_COUNTS["estimator_train"],
            "n_val": STRICT_CUBE_COUNTS["estimator_val"],
            "n_test": STRICT_CUBE_COUNTS["estimator_test"],
            "subsets_by_order": {str(k): list(v) for k, v in SUBSETS_BY_ORDER.items()},
            "conditional_mi_identity_if_exact": "I(R;E|S,k)=sum_s p(s|k) I(R;E|S=s,k); empirical p(s|k) is uniform because every strict cube contributes every subset once",
        },
        "null_contract": {
            "layers": list(DIAGNOSTIC_LAYERS),
            "critic_seeds": list(CRITIC_SEEDS),
            "derangement_seeds": list(DERANGEMENT_SEEDS),
            "n_derangements": len(DERANGEMENT_SEEDS),
            "split_preserving": True,
            "cube_level_bijection": True,
            "fixed_points_allowed": False,
            "subset_mask_fixed_within_each_critic": True,
            "same_permutation_reused_across_masks_and_layers_for_each_derangement_seed": True,
            "permutation_sets": perm_meta,
        },
        "estimator_contract": {
            "implementation": "exact Code-04 train_mine imported through pinned Code 11",
            "representation_dim": REPRESENTATION_DIM,
            "target_dim": 1,
            "critic_seeds": list(CRITIC_SEEDS),
            "absolute_mi_accuracy_claimed": False,
            "no_pca_or_projection": True,
            "no_clamping": True,
            "no_null_subtraction": True,
        },
        "subset_effects": eta2.to_dict(orient="records"),
        "outputs": {p.stem: p.name for p in outputs},
        "file_sha256": {p.name: sha256_file(p) for p in outputs},
        "computational_audit": {
            "expected_primary_fits": EXPECTED_PRIMARY_FITS,
            "expected_randomized_null_fits": EXPECTED_NULL_FITS,
            "expected_total_fits": EXPECTED_FITS,
            "observed_counts": counts,
            "primary_subset_cells": int(len(psub)),
            "null_derangement_subset_cells": int(len(nder)),
            "comparison_cells": int(len(comp)),
            "order_descriptive_cells": int(len(order_desc)),
            "all_results_finite": True,
            "overall_pass": True,
        },
        "overall_pass": True,
        "interpretation_limits": [
            "This PASS is computational/provenance only; no post-hoc numerical threshold defines scientific adequacy.",
            "Fixing subset mask removes the pooled S-identity shortcut by construction, but the resulting neural DV/MINE scores remain finite-sample lower-bound comparative scores.",
            "Absolute mutual information in nats is not claimed because the 2560-D absolute calibration failed.",
            "Null scores are diagnostics and are not subtracted from primary scores.",
            "Raw score averages across subsets/orders are descriptive only; the planned confirmatory retention analysis must normalize within subset first, then aggregate within order.",
            "Critic-seed variability measures estimator stability, not biological uncertainty.",
            "A full 31-layer subset-conditioned run should proceed only after scientific inspection of this 0/15/30 diagnostic.",
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
    print(f"  primary fixed-subset critic fits: {counts['primary']}")
    print(f"  randomized-null fixed-subset critic fits: {counts['random_derangement_null']}")
    print(f"  total critic fits: {counts['total']}")
    print("  layers: 0, 15, 30")
    print("  subset masks: 1..15 (one fixed S per critic)")
    print("  samples per fixed subset: 8210")
    print("  clamping: NO")
    print("  null subtraction: NO")
    print("  absolute MI in nats claimed: NO")
    print("  scientific acceptance threshold: NONE (diagnostic only)")

    print("\nOrder-level descriptive comparison (DO NOT use as final retention statistic):")
    print(order_desc.to_string(index=False))

    print(f"\n{CODE_NAME}: PASS (computational)")
    print("Inspect fixed-subset primary-vs-null behavior before any full 31-layer run or Code 13.")


if __name__ == "__main__":
    main()
