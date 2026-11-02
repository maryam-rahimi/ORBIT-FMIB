#!/usr/bin/env python3
"""ORBIT-FMIB CODE 11 v3 — full GB1 subset-conditioned neural-MI analysis.

Scientific correction
---------------------
The earlier pooled-by-order Code-11 analysis mixed multiple positional subset
masks S inside one critic. Randomized-null stress testing showed that the
critic could exploit subset identity because the empirical target distribution
E_{C,S} differs systematically among masks.

The corrected production estimand fixes one subset S per critic:

    score_{l,S} = neural_DV_MINE(R_{l,C,S}, E_{C,S}) across strict cubes C.

For each fixed subset there are exactly

    train = 7,352 cubes
    val   =   245 cubes
    test  =   613 cubes
    total = 8,210 cubes.

The exact fixed-S dataset constructor that passed the 0/15/30 randomized-null
diagnostic is reused byte-for-byte from
`11_validate_gb1_subset_conditioning.py`.

The exact frozen Code-04 `train_mine` implementation is reused.  No PCA,
projection, clamping, or null subtraction is introduced.

Required gates before biological fitting
----------------------------------------
1. Completed pooled Code-11 v2 provenance remains intact.
2. Fixed-S diagnostic source is exactly the audited source and its completed
   810-fit manifest is PASS computationally.
3. The exact fixed-S sample-regime calibration source is audited and its
   completed 45-fit manifest is PASS.
4. All Code-09 ESM-2 layer files 0..30 are hash verified.
5. Code-08/09/10 provenance remains pinned to completed Code 11.

Interpretation
--------------
Absolute MI accuracy in nats is NOT claimed.

The sample-regime calibration established a narrower detection contract:
under the exact 7352/245/613 split and 2560-D critic input, the frozen
estimator correctly identifies null behavior and reliably detects the
pre-specified strong synthetic channels. Weak/moderate biological dependence
may remain below the estimator's detection floor.

Production grid
---------------
31 layers x 15 fixed subset masks x 3 critic seeds = 1,395 fits.

No new biological null critics are run here. The five-derangement fixed-S null
diagnostic at layers 0/15/30 is a separate completed validation artifact.

Downstream aggregation
----------------------
Raw averaging across subset masks is DESCRIPTIVE ONLY.

Code 13 must normalize/define retention within each fixed subset trajectory
first, then aggregate those retention quantities within interaction order.
It must not resurrect the pooled-by-order estimand and must not subtract nulls.

Recommended sequence
--------------------
Local:
  python "03 Code/11_run_gb1_subset_conditioned_mi.py" --self-test

Newton:
  python "03 Code/11_run_gb1_subset_conditioned_mi.py" --preflight-only --device cuda
  python "03 Code/11_run_gb1_subset_conditioned_mi.py" --device cuda

If interrupted:
  python "03 Code/11_run_gb1_subset_conditioned_mi.py" --device cuda --resume

Use --overwrite only for an intentional clean restart of this production run.
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
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

CODE_NAME = "ORBIT-FMIB CODE 11 — SUBSET-CONDITIONED GB1 MI"
SCHEMA_VERSION = 3

# Frozen audited upstream sources.
EXPECTED_LEGACY_CODE11_SHA256 = "5d98333fb5a129331199e07b808a3457171eb2f3349b7ea4b24fe185d6c99cd3"
EXPECTED_SUBSET_DIAG_SHA256 = "d833786081b0481f672a418ac35efed59e602854736480993d4f216290f55c30"
EXPECTED_SAMPLE_REGIME_SHA256 = "d16bbc9707a24d40ebd9b03c13efe02567fc8b569f6ebef884999093aa1b0290"

EXPECTED_SUBSET_DIAG_NAME = "ORBIT-FMIB GB1 SUBSET-CONDITIONED MI DIAGNOSTIC"
EXPECTED_SAMPLE_REGIME_NAME = "ORBIT-FMIB GB1 FIXED-SUBSET SAMPLE-REGIME CALIBRATION"

LAYERS = tuple(range(31))
ALL_MASKS = tuple(range(1, 16))
ORDERS = (1, 2, 3, 4)
SUBSETS_BY_ORDER = {
    k: tuple(m for m in ALL_MASKS if int(m).bit_count() == k) for k in ORDERS
}
CRITIC_SEEDS = (11, 23, 37)
REPRESENTATION_DIM = 2560
N_TRAIN = 7352
N_VAL = 245
N_TEST = 613
N_PER_SUBSET = N_TRAIN + N_VAL + N_TEST

# Frozen corrected primary estimator partition.
PRODUCTION_IID_SPLIT_SEED = 20260910

EXPECTED_FITS = len(LAYERS) * len(ALL_MASKS) * len(CRITIC_SEEDS)  # 1395
EXPECTED_SUBSET_CELLS = len(LAYERS) * len(ALL_MASKS)             # 465
EXPECTED_ORDER_CELLS = len(LAYERS) * len(ORDERS)                 # 124

RESULT_COLS = (
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


def import_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def verify_source(path: Path, expected_sha: str, label: str):
    got = sha256_file(path)
    if got.lower() != expected_sha.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch.\nExpected: {expected_sha}\nObserved: {got}\nPath: {path}"
        )
    return got


def verify_manifest_outputs(manifest: Mapping, base: Path, label: str) -> None:
    hashes = manifest.get("file_sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f"{label}: manifest does not contain output hashes.")
    for name, expected in hashes.items():
        p = base / Path(name).name
        if not p.exists():
            raise FileNotFoundError(f"{label}: missing pinned output {p}")
        got = sha256_file(p)
        if got.lower() != str(expected).lower():
            raise RuntimeError(
                f"{label}: output hash mismatch for {p.name}.\n"
                f"Expected: {expected}\nObserved: {got}"
            )


def resolve_device(torch, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def verify_subset_diagnostic(
    source_path: Path,
    manifest_path: Path,
    exp02: Path,
):
    verify_source(source_path, EXPECTED_SUBSET_DIAG_SHA256, "fixed-S diagnostic source")
    mod = import_module(source_path, "_orbitfmib_subset_production_contract")

    required = (
        "preflight",
        "build_subset_dataset",
        "target_for_subset",
        "write_csv_atomic",
        "STRICT_CUBE_COUNTS",
        "N_PER_SUBSET",
        "ALL_MASKS",
        "SUBSETS_BY_ORDER",
        "CRITIC_SEEDS",
        "REPRESENTATION_DIM",
    )
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(f"Fixed-S diagnostic source missing required objects: {missing}")

    if int(mod.REPRESENTATION_DIM) != REPRESENTATION_DIM:
        raise RuntimeError("Fixed-S diagnostic representation dimension changed.")
    if tuple(map(int, mod.CRITIC_SEEDS)) != CRITIC_SEEDS:
        raise RuntimeError("Fixed-S diagnostic critic-seed contract changed.")
    if tuple(map(int, mod.ALL_MASKS)) != ALL_MASKS:
        raise RuntimeError("Fixed-S diagnostic mask contract changed.")
    frozen = {int(k): tuple(map(int, v)) for k, v in mod.SUBSETS_BY_ORDER.items()}
    if frozen != SUBSETS_BY_ORDER:
        raise RuntimeError("Fixed-S diagnostic subset-by-order contract changed.")
    if int(mod.N_PER_SUBSET) != N_PER_SUBSET:
        raise RuntimeError("Fixed-S diagnostic sample count changed.")
    if dict(mod.STRICT_CUBE_COUNTS) != {
        "estimator_train": N_TRAIN,
        "estimator_val": N_VAL,
        "estimator_test": N_TEST,
    }:
        raise RuntimeError("Fixed-S diagnostic split counts changed.")

    m = load_json(manifest_path)
    if m.get("code_name") != EXPECTED_SUBSET_DIAG_NAME:
        raise RuntimeError("Wrong fixed-S diagnostic manifest.")
    if not bool(m.get("overall_pass", False)):
        raise RuntimeError("Fixed-S diagnostic manifest is not PASS.")
    if str(m.get("diagnostic_source_sha256", "")).lower() != EXPECTED_SUBSET_DIAG_SHA256.lower():
        raise RuntimeError("Fixed-S diagnostic manifest does not pin expected source.")

    cc = m.get("conditioning_contract", {})
    if not bool(cc.get("one_subset_mask_per_critic", False)):
        raise RuntimeError("Fixed-S diagnostic did not use one subset mask per critic.")
    if int(cc.get("samples_per_subset", -1)) != N_PER_SUBSET:
        raise RuntimeError("Fixed-S diagnostic samples-per-subset mismatch.")
    if (
        int(cc.get("n_train", -1)),
        int(cc.get("n_val", -1)),
        int(cc.get("n_test", -1)),
    ) != (N_TRAIN, N_VAL, N_TEST):
        raise RuntimeError("Fixed-S diagnostic split counts mismatch.")

    ec = m.get("estimator_contract", {})
    expected_flags = {
        "absolute_mi_accuracy_claimed": False,
        "no_pca_or_projection": True,
        "no_clamping": True,
        "no_null_subtraction": True,
    }
    for key, val in expected_flags.items():
        if ec.get(key) != val:
            raise RuntimeError(f"Fixed-S diagnostic estimator contract mismatch: {key}")

    ca = m.get("computational_audit", {})
    if not bool(ca.get("overall_pass", False)):
        raise RuntimeError("Fixed-S diagnostic computational audit is not PASS.")
    if int(ca.get("expected_total_fits", -1)) != 810:
        raise RuntimeError("Fixed-S diagnostic expected fit count changed.")

    verify_manifest_outputs(m, exp02, "fixed-S diagnostic")
    return mod, m, sha256_file(manifest_path)


def verify_sample_regime(
    source_path: Path,
    manifest_path: Path,
    exp03: Path,
):
    verify_source(source_path, EXPECTED_SAMPLE_REGIME_SHA256, "sample-regime calibration source")
    m = load_json(manifest_path)
    if m.get("code_name") != EXPECTED_SAMPLE_REGIME_NAME:
        raise RuntimeError("Wrong fixed-S sample-regime calibration manifest.")
    if not bool(m.get("overall_pass", False)):
        raise RuntimeError("Fixed-S sample-regime calibration is not PASS.")
    if str(m.get("source_sha256", "")).lower() != EXPECTED_SAMPLE_REGIME_SHA256.lower():
        raise RuntimeError("Sample-regime manifest does not pin expected source.")

    sc = m.get("synthetic_contract", {})
    expected = {
        "representation_dim": REPRESENTATION_DIM,
        "target_dim": 1,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_total": N_PER_SUBSET,
        "critic_seeds": list(CRITIC_SEEDS),
        "absolute_mi_accuracy_required": False,
    }
    for key, val in expected.items():
        if sc.get(key) != val:
            raise RuntimeError(f"Sample-regime calibration contract mismatch: {key}")

    gates = m.get("gates", {})
    if not bool(gates.get("overall_pass", False)):
        raise RuntimeError("Sample-regime calibration hard gates are not PASS.")
    required_gates = list(gates.get("hard_gates", []))
    if not required_gates:
        raise RuntimeError("Sample-regime calibration does not declare hard gates.")
    failed = [g for g in required_gates if not bool(gates.get(g, False))]
    if failed:
        raise RuntimeError(f"Sample-regime calibration failed hard gates: {failed}")

    verify_manifest_outputs(m, exp03, "sample-regime calibration")
    return m, sha256_file(manifest_path)


def verify_all_esm_layers(c09, subset_mod) -> None:
    names = c09["manifest"].get("outputs", {}).get("layer_files", {})
    hashes = c09["manifest"].get("file_sha256", {})
    for layer in LAYERS:
        name = names.get(str(layer))
        if not name:
            raise RuntimeError(f"Code-09 manifest missing layer {layer} filename.")
        expected = hashes.get(Path(name).name)
        if not expected:
            raise RuntimeError(f"Code-09 manifest missing layer {layer} hash.")
        subset_mod.verify_hash(c09["layer_paths"][layer], expected, f"Code-09 layer {layer}")


def expected_keys() -> set:
    return {
        (int(layer), int(mask), int(seed))
        for layer in LAYERS
        for mask in ALL_MASKS
        for seed in CRITIC_SEEDS
    }


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLS)
    d = pd.read_csv(path)
    missing = set(RESULT_COLS) - set(d.columns)
    if missing:
        raise RuntimeError(f"Existing production seed CSV missing columns: {sorted(missing)}")
    if d.duplicated(["layer", "subset_mask", "critic_seed"]).any():
        raise RuntimeError("Duplicate production critic rows.")
    return d


def save_result_row(d: pd.DataFrame, row: Mapping, path: Path) -> pd.DataFrame:
    d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    d = d.sort_values(["layer", "order_k", "subset_mask", "critic_seed"]).reset_index(drop=True)
    write_csv_atomic(d, path)
    return d


def audit_seed_results(d: pd.DataFrame) -> Dict:
    obs = {
        (int(r.layer), int(r.subset_mask), int(r.critic_seed))
        for r in d.itertuples(index=False)
    }
    exp = expected_keys()
    if obs != exp:
        raise RuntimeError(
            f"Full fixed-S fit grid mismatch: observed={len(obs)}, expected={len(exp)}."
        )
    if len(d) != EXPECTED_FITS:
        raise RuntimeError("Unexpected full fixed-S seed-result row count.")

    nums = [
        "best_step",
        "final_training_step",
        "best_val_mi",
        "test_mi",
        "test_product_shuffle_sd",
        "heldout_joint_minus_mismatch_score",
    ]
    if not np.isfinite(d[nums].to_numpy(float)).all():
        raise RuntimeError("Non-finite full fixed-S diagnostics.")

    if not (d["representation_dim"].astype(int) == REPRESENTATION_DIM).all():
        raise RuntimeError("Representation dimension mismatch.")
    if not (d["target_dim"].astype(int) == 1).all():
        raise RuntimeError("Target dimension mismatch.")
    if not (d["n_samples"].astype(int) == N_PER_SUBSET).all():
        raise RuntimeError("Per-subset total sample count mismatch.")
    if not (d["n_train"].astype(int) == N_TRAIN).all():
        raise RuntimeError("Per-subset train count mismatch.")
    if not (d["n_val"].astype(int) == N_VAL).all():
        raise RuntimeError("Per-subset validation count mismatch.")
    if not (d["n_test"].astype(int) == N_TEST).all():
        raise RuntimeError("Per-subset test count mismatch.")

    observed_order = d["subset_mask"].astype(int).map(int.bit_count).to_numpy()
    if not np.array_equal(d["order_k"].astype(int).to_numpy(), observed_order):
        raise RuntimeError("order_k does not match subset-mask cardinality.")

    if set(d["layer"].astype(int)) != set(LAYERS):
        raise RuntimeError("Layer coverage mismatch.")
    if set(d["subset_mask"].astype(int)) != set(ALL_MASKS):
        raise RuntimeError("Subset-mask coverage mismatch.")
    if set(d["critic_seed"].astype(int)) != set(CRITIC_SEEDS):
        raise RuntimeError("Critic-seed coverage mismatch.")

    return {
        "primary_fits": int(len(d)),
        "subset_cells": EXPECTED_SUBSET_CELLS,
        "order_cells": EXPECTED_ORDER_CELLS,
        "all_results_finite": True,
    }


def make_summaries(d: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset = (
        d.groupby(["layer", "order_k", "subset_mask"], as_index=False)
        .agg(
            score_mean=("test_mi", "mean"),
            critic_seed_sd=("test_mi", lambda x: float(np.std(np.asarray(x, float), ddof=1))),
            score_min=("test_mi", "min"),
            score_max=("test_mi", "max"),
            mean_best_val=("best_val_mi", "mean"),
            mean_best_step=("best_step", "mean"),
            mean_final_training_step=("final_training_step", "mean"),
            mean_product_shuffle_sd=("test_product_shuffle_sd", "mean"),
            mean_joint_minus_mismatch_score=("heldout_joint_minus_mismatch_score", "mean"),
        )
        .sort_values(["layer", "order_k", "subset_mask"])
        .reset_index(drop=True)
    )
    if len(subset) != EXPECTED_SUBSET_CELLS:
        raise RuntimeError("Subset summary cell count mismatch.")

    order = (
        subset.groupby(["layer", "order_k"], as_index=False)
        .agg(
            n_subset_masks=("subset_mask", "count"),
            descriptive_score_mean_across_subsets=("score_mean", "mean"),
            descriptive_score_sd_across_subsets=("score_mean", lambda x: float(np.std(np.asarray(x, float), ddof=1)) if len(x) > 1 else 0.0),
            max_subset_critic_seed_sd=("critic_seed_sd", "max"),
        )
        .sort_values(["layer", "order_k"])
        .reset_index(drop=True)
    )
    if len(order) != EXPECTED_ORDER_CELLS:
        raise RuntimeError("Order descriptive summary cell count mismatch.")
    expected_n = {1: 4, 2: 6, 3: 4, 4: 1}
    for k, n in expected_n.items():
        vals = order.loc[order["order_k"] == k, "n_subset_masks"].astype(int).unique()
        if vals.tolist() != [n]:
            raise RuntimeError(f"Order k={k} subset-count summary mismatch.")
    return subset, order


def deterministic_iid_split() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(PRODUCTION_IID_SPLIT_SEED)
    perm = rng.permutation(N_PER_SUBSET).astype(np.int64)
    tr = np.sort(perm[:N_TRAIN])
    va = np.sort(perm[N_TRAIN:N_TRAIN + N_VAL])
    te = np.sort(perm[N_TRAIN + N_VAL:])
    if (len(tr), len(va), len(te)) != (N_TRAIN, N_VAL, N_TEST):
        raise RuntimeError("IID split count mismatch.")
    if (
        np.intersect1d(tr, va).size
        or np.intersect1d(tr, te).size
        or np.intersect1d(va, te).size
    ):
        raise RuntimeError("IID split overlap.")
    if len(np.unique(np.concatenate([tr, va, te]))) != N_PER_SUBSET:
        raise RuntimeError("IID split does not partition all observations.")
    return tr, va, te


def iid_split_sha256(split: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> str:
    h = hashlib.sha256()
    for arr in split:
        x = np.asarray(arr, dtype="<i8", order="C")
        h.update(x.tobytes(order="C"))
    return h.hexdigest()


def self_test() -> None:
    assert LAYERS == tuple(range(31))
    assert ALL_MASKS == tuple(range(1, 16))
    assert {k: len(v) for k, v in SUBSETS_BY_ORDER.items()} == {1: 4, 2: 6, 3: 4, 4: 1}
    assert tuple(sorted(m for v in SUBSETS_BY_ORDER.values() for m in v)) == ALL_MASKS
    assert (N_TRAIN, N_VAL, N_TEST, N_PER_SUBSET) == (7352, 245, 613, 8210)
    assert EXPECTED_FITS == 1395
    assert EXPECTED_SUBSET_CELLS == 465
    assert EXPECTED_ORDER_CELLS == 124
    assert len(expected_keys()) == 1395
    assert sha256_obj({"a": 1, "b": [2, 3]}) == sha256_obj({"b": [2, 3], "a": 1})
    print("SELF-TEST: PASS")
    print(
        "Verified: 31-layer x 15-subset x 3-seed = 1,395 production grid, "
        "subset cardinalities 4/6/4/1, exact 7,352/245/613 fixed-S split, "
        "465 subset cells, 124 order-descriptive cells, and deterministic run fingerprint hashing."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--chunk-cubes", type=int, default=64)
    ap.add_argument("--output-dir", type=Path)

    ap.add_argument("--legacy-code11-source", type=Path)
    ap.add_argument("--legacy-code11-manifest", type=Path)
    ap.add_argument("--legacy-code11-seed-results", type=Path)
    ap.add_argument("--code04-source", type=Path)
    ap.add_argument("--code08-manifest", type=Path)
    ap.add_argument("--code09-manifest", type=Path)
    ap.add_argument("--code10-manifest", type=Path)

    ap.add_argument("--subset-diagnostic-source", type=Path)
    ap.add_argument("--subset-diagnostic-manifest", type=Path)
    ap.add_argument("--sample-regime-source", type=Path)
    ap.add_argument("--sample-regime-manifest", type=Path)
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return
    if a.chunk_cubes < 1:
        raise ValueError("--chunk-cubes must be >= 1")

    root = project_root()
    exp02 = root / "04 Experiments" / "EXP02_GB1_ESM2"
    exp03 = root / "04 Experiments" / "EXP03_MI"
    out = a.output_dir or exp02
    out.mkdir(parents=True, exist_ok=True)

    p11s = a.legacy_code11_source or root / "03 Code" / "11_run_gb1_target_mi.py"
    p11m = a.legacy_code11_manifest or exp02 / "gb1_target_mi_manifest.json"
    p11r = a.legacy_code11_seed_results or exp02 / "gb1_target_mi_seed_results.csv"
    p04 = a.code04_source or root / "03 Code" / "04_validate_mi_estimator.py"
    p08 = a.code08_manifest or exp02 / "gb1_esm2_extraction_manifest.json"
    p09 = a.code09_manifest or exp02 / "ESM2_Site_Reps" / "esm2_extraction_manifest.json"
    p10 = a.code10_manifest or exp02 / "gb1_order_targets_manifest.json"

    psubsrc = a.subset_diagnostic_source or root / "03 Code" / "11_validate_gb1_subset_conditioning.py"
    psubman = a.subset_diagnostic_manifest or exp02 / "gb1_subset_conditioned_diag_manifest.json"
    pregsrc = a.sample_regime_source or root / "03 Code" / "11_validate_gb1_fixed_subset_sample_regime.py"
    pregman = a.sample_regime_manifest or exp03 / "fixed_subset_sample_regime_manifest.json"

    seed_out = out / "gb1_subset_mi_seed_results.csv"
    subset_out = out / "gb1_subset_mi_summary.csv"
    order_out = out / "gb1_subset_mi_order_descriptive.csv"
    state_out = out / "gb1_subset_mi_state.json"
    manifest_out = out / "gb1_subset_mi_manifest.json"

    if a.overwrite:
        for p in (seed_out, subset_out, order_out, state_out, manifest_out):
            if p.exists():
                p.unlink()

    print("=" * 86)
    print("ORBIT-FMIB CODE 11 v3 — FULL GB1 SUBSET-CONDITIONED MI")
    print("=" * 86)

    # The completed fixed-S diagnostic is both a provenance gate and the exact
    # implementation source for the corrected dataset construction.
    subset_mod, subset_manifest, subset_manifest_sha = verify_subset_diagnostic(
        psubsrc, psubman, exp02
    )

    # Reuse the diagnostic's audited preflight for legacy Code11 + Code08/09/10.
    (
        legacy_code11,
        mine04,
        c11,
        c08,
        c09,
        c10,
        maps,
        samples,
        _permutations,
        _perm_meta,
        eta2,
    ) = subset_mod.preflight(p11s, p11m, p11r, p04, p08, p09, p10)

    if sha256_file(p11s).lower() != EXPECTED_LEGACY_CODE11_SHA256.lower():
        raise RuntimeError("Legacy pooled Code-11 source changed.")

    # Production requires every ESM layer, not only diagnostic layers.
    verify_all_esm_layers(c09, subset_mod)

    # The exact biological sample regime must have passed the independent
    # synthetic detection calibration before biological production is allowed.
    sample_manifest, sample_manifest_sha = verify_sample_regime(
        pregsrc, pregman, exp03
    )

    torch, nn = mine04.import_torch()
    device = resolve_device(torch, a.device)
    if device.type == "cpu":
        torch.set_num_threads(1)

    # Freeze the corrected primary estimator partition once per run.
    iid_split = deterministic_iid_split()
    iid_tr, iid_va, iid_te = iid_split
    iid_split_hash = iid_split_sha256(iid_split)

    fp = {
        "schema_version": SCHEMA_VERSION,
        "production_source_sha256": sha256_file(Path(__file__)),
        "legacy_code11_source_sha256": EXPECTED_LEGACY_CODE11_SHA256,
        "legacy_code11_manifest_sha256": c11["manifest_sha"],
        "legacy_code11_seed_results_sha256": c11["seed_sha"],
        "code04_source_sha256": sha256_file(p04),
        "code08_manifest_sha256": c08["sha"],
        "code09_manifest_sha256": c09["sha"],
        "code10_manifest_sha256": c10["sha"],
        "subset_diagnostic_source_sha256": EXPECTED_SUBSET_DIAG_SHA256,
        "subset_diagnostic_manifest_sha256": subset_manifest_sha,
        "sample_regime_source_sha256": EXPECTED_SAMPLE_REGIME_SHA256,
        "sample_regime_manifest_sha256": sample_manifest_sha,
        "layers": list(LAYERS),
        "subset_masks": list(ALL_MASKS),
        "subsets_by_order": {str(k): list(v) for k, v in SUBSETS_BY_ORDER.items()},
        "critic_seeds": list(CRITIC_SEEDS),
        "representation_dim": REPRESENTATION_DIM,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_total_per_subset": N_PER_SUBSET,
        "estimator_split_protocol": "deterministic_iid_cube_reassignment",
        "estimator_split_seed": PRODUCTION_IID_SPLIT_SEED,
        "estimator_split_sha256": iid_split_hash,
        "device_type": device.type,
        "chunk_cubes": int(a.chunk_cubes),
        "analysis_contract": "one fixed subset S per critic; same 8,210 biological cubes; deterministic IID estimator partition",
        "absolute_mi_accuracy_claimed": False,
        "no_pca_or_projection": True,
        "no_clamping": True,
        "no_null_subtraction": True,
        "new_null_fits_in_production": 0,
    }
    fpsha = sha256_obj(fp)

    print("\nPreflight/provenance gate: PASS")
    print("  legacy pooled Code-11 v2 provenance: PASS")
    print("  exact fixed-S diagnostic source + 810-fit manifest: PASS")
    print("  exact fixed-S sample-regime source + 45-fit manifest: PASS")
    print("  exact Code-04 train_mine: PASS")
    print("  Code-08/09/10 provenance: PASS")
    print("  all 31 ESM-2 layer hashes: PASS")
    print("  one fixed subset mask per critic: YES")
    print(f"  exact fixed-S samples: train={N_TRAIN} val={N_VAL} test={N_TEST} total={N_PER_SUBSET}")
    print(f"  estimator split: deterministic IID seed={PRODUCTION_IID_SPLIT_SEED}")
    print(f"  estimator split SHA256: {iid_split_hash}")
    print(f"  planned biological critic fits: {EXPECTED_FITS}")
    print("  new biological null fits: 0 (completed validation artifact is reused)")
    print("  PCA/projection: NO")
    print("  clamping: NO")
    print("  null subtraction: NO")
    print("  absolute MI accuracy claimed: NO")
    print(f"  device: {device.type}")
    print(f"  fingerprint: {fpsha}")

    print("\nPreviously quantified subset-identity target effects (eta^2; descriptive):")
    for r in eta2.itertuples(index=False):
        val = "N/A" if not np.isfinite(r.subset_mean_effect_eta2) else f"{r.subset_mean_effect_eta2:.6f}"
        print(f"  {r.split:15s} k={int(r.order_k)} masks={int(r.n_subset_masks)} eta2={val}")

    if a.preflight_only:
        print("\nPREFLIGHT: PASS")
        print("No biological critic was trained and no production outputs were written.")
        return

    if manifest_out.exists() and not a.overwrite:
        m = load_json(manifest_out)
        if m.get("run_fingerprint_sha256") == fpsha and bool(m.get("overall_pass", False)):
            print(f"\n{CODE_NAME}: PASS")
            print("Matching completed production run already exists.")
            return
        raise RuntimeError("Existing production manifest does not match this run.")

    if state_out.exists() and not a.overwrite:
        st = load_json(state_out)
        if st.get("run_fingerprint_sha256") != fpsha:
            raise RuntimeError("Existing production state fingerprint mismatch.")
        if not a.resume:
            raise RuntimeError("Existing production state detected. Re-run with --resume.")
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
        (int(r.layer), int(r.subset_mask), int(r.critic_seed))
        for r in d.itertuples(index=False)
    }
    if not done.issubset(expected_keys()):
        raise RuntimeError("Existing production CSV contains unexpected conditions.")

    X = legacy_code11.chi_matrix()

    for layer_id in LAYERS:
        needed = any(
            (layer_id, mask, seed) not in done
            for mask in ALL_MASKS
            for seed in CRITIC_SEEDS
        )
        if not needed:
            continue

        print(f"\nLayer {layer_id:02d}", flush=True)
        layer = np.load(c09["layer_paths"][layer_id], mmap_mode="r", allow_pickle=False)

        for mask in ALL_MASKS:
            missing = [
                seed for seed in CRITIC_SEEDS
                if (int(layer_id), int(mask), int(seed)) not in done
            ]
            if not missing:
                continue

            k = int(mask).bit_count()
            print(
                f"  mask={mask:02d} k={k}: constructing fixed-S 2560-D dataset ...",
                flush=True,
            )
            U, V, strict_split, _counts = subset_mod.build_subset_dataset(
                legacy_code11, layer, maps, samples, int(mask), X, int(a.chunk_cubes)
            )
            if len(U) != N_PER_SUBSET or len(V) != N_PER_SUBSET:
                raise RuntimeError("Fixed-S biological observation count changed.")

            tr, va, te = iid_tr, iid_va, iid_te
            split = iid_split

            for seed in missing:
                print(f"    primary critic_seed={seed}", flush=True)
                res = mine04.train_mine(
                    torch, nn, U, V, int(seed), device, split
                )
                row = {
                    "layer": int(layer_id),
                    "order_k": int(k),
                    "subset_mask": int(mask),
                    "critic_seed": int(seed),
                    "representation_dim": REPRESENTATION_DIM,
                    "target_dim": 1,
                    "n_samples": len(U),
                    "n_train": len(tr),
                    "n_val": len(va),
                    "n_test": len(te),
                    **res,
                }
                d = save_result_row(d, row, seed_out)
                done.add((int(layer_id), int(mask), int(seed)))

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
                            "layer": int(layer_id),
                            "order_k": int(k),
                            "subset_mask": int(mask),
                            "critic_seed": int(seed),
                        },
                    },
                )

            del U, V
        del layer

    audit = audit_seed_results(d)
    subset_df, order_df = make_summaries(d)
    write_csv_atomic(subset_df, subset_out)
    write_csv_atomic(order_df, order_out)

    outputs = (seed_out, subset_out, order_out)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_name": CODE_NAME,
        "purpose": (
            "Full 31-layer GB1 target-aligned neural-MI analysis after removal of "
            "subset-identity confounding by fitting one fixed positional subset S per critic."
        ),
        "production_source_sha256": sha256_file(Path(__file__)),
        "run_fingerprint": fp,
        "run_fingerprint_sha256": fpsha,
        "upstream": {
            "legacy_code11_source_sha256": EXPECTED_LEGACY_CODE11_SHA256,
            "legacy_code11_manifest_sha256": c11["manifest_sha"],
            "legacy_code11_seed_results_sha256": c11["seed_sha"],
            "code04_source_sha256": sha256_file(p04),
            "code08_manifest_sha256": c08["sha"],
            "code09_manifest_sha256": c09["sha"],
            "code10_manifest_sha256": c10["sha"],
            "subset_diagnostic_source_sha256": EXPECTED_SUBSET_DIAG_SHA256,
            "subset_diagnostic_manifest_sha256": subset_manifest_sha,
            "sample_regime_source_sha256": EXPECTED_SAMPLE_REGIME_SHA256,
            "sample_regime_manifest_sha256": sample_manifest_sha,
        },
        "conditioning_contract": {
            "conditioning_variable": "positional subset mask S",
            "one_subset_mask_per_critic": True,
            "scientific_quantity": "I(R_{l,C,S}; E_{C,S}) estimated across the same 8,210 biological cubes C for fixed S",
            "conditional_identity": "I(R;E|S,k)=sum_s p(s|k) I(R;E|S=s,k)",
            "empirical_subset_weight_within_order": "uniform because every retained biological cube contributes every subset exactly once",
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test": N_TEST,
            "n_total_per_subset": N_PER_SUBSET,
            "subsets_by_order": {str(k): list(v) for k, v in SUBSETS_BY_ORDER.items()},
        },
        "estimator_contract": {
            "implementation": "exact Code-04 train_mine",
            "representation_dim": REPRESENTATION_DIM,
            "target_dim": 1,
            "critic_seeds": list(CRITIC_SEEDS),
            "absolute_mi_accuracy_claimed": False,
            "validated_scope": "null and pre-specified strong-signal detection under exact 7352/245/613 fixed-S sample regime",
            "no_pca_or_projection": True,
            "no_clamping": True,
            "no_null_subtraction": True,
        },
        "validation_contract": {
            "fixed_subset_diagnostic_completed": True,
            "fixed_subset_diagnostic_fit_count": 810,
            "fixed_subset_null_layers": [0, 15, 30],
            "fixed_subset_null_derangements": 5,
            "sample_regime_calibration_completed": True,
            "sample_regime_calibration_fit_count": 45,
            "sample_regime_overall_pass": True,
            "no_new_null_fits_in_production": True,
        },
        "computational_audit": {
            "expected_primary_fits": EXPECTED_FITS,
            "observed": audit,
            "overall_pass": True,
        },
        "outputs": {
            "seed_results": seed_out.name,
            "subset_summary": subset_out.name,
            "order_descriptive_summary": order_out.name,
        },
        "file_sha256": {p.name: sha256_file(p) for p in outputs},
        "overall_pass": True,
        "downstream_contract": [
            "The subset-level summary is the authoritative biological score map.",
            "The order-level raw mean is descriptive only and must not be used directly as the confirmatory retention statistic.",
            "Retention/selective-compression analysis must be defined within each fixed subset trajectory first and only then aggregated within order.",
            "Null scores are validation diagnostics and are not subtracted from biological scores.",
            "Negative near-floor neural-DV scores are retained and are not interpreted as negative physical mutual information.",
            "Failure to detect biological dependence is not equivalent to proof of zero mutual information.",
        ],
        "interpretation_limits": [
            "Absolute mutual information in nats is not claimed.",
            "The exact 2560-D estimator underestimates known moderate/strong analytic MI magnitudes.",
            "The exact fixed-S sample-regime calibration validated null and strong-signal detection, not arbitrary weak/non-Gaussian dependence.",
            "Critic seeds quantify estimator stability and are not biological replicates.",
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
            "completed_fits": EXPECTED_FITS,
            "expected_fits": EXPECTED_FITS,
            "final_manifest_sha256": sha256_file(manifest_out),
        },
    )

    print("\nComputational audit: PASS")
    print(f"  biological critic fits: {EXPECTED_FITS}")
    print(f"  subset cells: {EXPECTED_SUBSET_CELLS}")
    print(f"  descriptive order cells: {EXPECTED_ORDER_CELLS}")
    print("  fixed-S conditioning: YES")
    print("  null subtraction: NO")
    print("  clamping: NO")
    print("  absolute MI accuracy claimed: NO")
    print(f"\n{CODE_NAME}: PASS")
    print("Full fixed-subset layer-by-subset neural-MI map is ready for revised Code 12/13.")


if __name__ == "__main__":
    main()
