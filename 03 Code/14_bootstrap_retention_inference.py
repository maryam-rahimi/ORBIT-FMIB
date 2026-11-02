#!/usr/bin/env python3
"""ORBIT-FMIB CODE 14 — FULL-STATISTIC CUBE-BOOTSTRAP RETENTION INFERENCE.

Fully faithful inferential extension of corrected IID Code 11 and Code 13.

No inferential result is computed unless all frozen contracts and upstream
source hashes verify exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np

CODE_NAME = "ORBIT-FMIB CODE 14 — FULL-STATISTIC CUBE-BOOTSTRAP RETENTION INFERENCE"
SCHEMA_VERSION = 1

LAYERS = tuple(range(31))
ALL_MASKS = tuple(range(1, 16))
ORDERS = (1, 2, 3, 4)
SUBSETS_BY_ORDER = {
    k: tuple(m for m in ALL_MASKS if int(m).bit_count() == k)
    for k in ORDERS
}
CRITIC_SEEDS = (11, 23, 37)

N_TRAIN = 7352
N_VAL = 245
N_TEST = 613
N_TOTAL = N_TRAIN + N_VAL + N_TEST
PRODUCTION_IID_SPLIT_SEED = 20260910
BOOTSTRAP_MASTER_SEED = 20260911
PRODUCTION_BOOTSTRAP_REPLICATES = 499

TAU = 0.006777383486429849
OBSERVED_DELTA_HO_MINUS_LO = -0.10702806260128583

EXPECTED_PARENT_CONTRACT_SHA256 = (
    "e32ef2c8454fde64ecd4a5c93988814c202f1a0299dee95512f95cabf6f40906"
)
EXPECTED_IMPLEMENTATION_CONTRACT_SHA256 = (
    "b3cf9937ecba19a818ba7b97fabb9cb60a015e5d1ff41e5a5cc6ab58923e81e5"
)
EXPECTED_EXECUTION_CONTRACT_SHA256 = (
    "b8ac4b9c71f6fa2e8605e376a19bef1ffe031f63b285879c90ab9222f3e04d9e"
)
EXPECTED_STATISTICAL_SUPERSESSION_SHA256 = (
    "bc76feebad53e834fd5ebb4f4cbf893e6e76fb31c7984f4c64961feaa0e3c409"
)
EXPECTED_PRODUCTION_IID_SPLIT_SHA256 = (
    "4b7f1d173de7a979db762f54bfcb90674ab36718f51ee6aef0bb47db91b2ff66"
)
EXPECTED_CODE11_IID_SHA256 = (
    "8ff6654c9352e56f2eff7368ac1ed14b90929f3729cb3bacc5b5dd592b1209e3"
)
EXPECTED_SUBSET_BUILDER_SHA256 = (
    "17bee1cdce057e0ac55f0eb0250db8770fee7fff962f096902aedf9ec2185362"
)
EXPECTED_CODE04_MINE_SHA256 = (
    "ecdb937dac3b943e21bc1aaeaf9eab933ab6badc15c446d83595ec363f8053ab"
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def deterministic_iid_split():
    """Reproduce corrected Code-11 production estimator partition exactly."""
    rng = np.random.default_rng(PRODUCTION_IID_SPLIT_SEED)
    perm = rng.permutation(N_TOTAL).astype(np.int64)

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

    if len(np.unique(np.concatenate([tr, va, te]))) != N_TOTAL:
        raise RuntimeError("IID split does not partition all observations.")

    return tr, va, te


def iid_split_sha256(split) -> str:
    """Use the exact byte-level hashing rule from corrected Code 11."""
    h = hashlib.sha256()
    for arr in split:
        x = np.asarray(arr, dtype="<i8", order="C")
        h.update(x.tobytes(order="C"))
    return h.hexdigest()


def bootstrap_indices_for_replicate(split, replicate_id: int):
    """Generate one deterministic partition-preserving cube bootstrap.

    Sampling is with replacement independently within the frozen production
    train/validation/test partitions. Draw order is preserved. The same three
    arrays are intended to be reused for every layer, subset, and critic seed
    within this bootstrap replicate.
    """
    if not (0 <= replicate_id < PRODUCTION_BOOTSTRAP_REPLICATES):
        raise ValueError(
            f"replicate_id must be in [0, {PRODUCTION_BOOTSTRAP_REPLICATES - 1}], "
            f"got {replicate_id}"
        )

    tr, va, te = split

    # Replicate-specific deterministic RNG stream derived prospectively from
    # the frozen master seed and replicate id.
    ss = np.random.SeedSequence([BOOTSTRAP_MASTER_SEED, replicate_id])
    rng = np.random.default_rng(ss)

    btr = rng.choice(tr, size=N_TRAIN, replace=True).astype(np.int64)
    bva = rng.choice(va, size=N_VAL, replace=True).astype(np.int64)
    bte = rng.choice(te, size=N_TEST, replace=True).astype(np.int64)

    if (len(btr), len(bva), len(bte)) != (N_TRAIN, N_VAL, N_TEST):
        raise RuntimeError("Bootstrap partition count mismatch.")

    # Every draw must remain inside its original estimator partition.
    tr_set = set(map(int, tr))
    va_set = set(map(int, va))
    te_set = set(map(int, te))

    if not set(map(int, btr)).issubset(tr_set):
        raise RuntimeError("Bootstrap training sample escaped training partition.")
    if not set(map(int, bva)).issubset(va_set):
        raise RuntimeError("Bootstrap validation sample escaped validation partition.")
    if not set(map(int, bte)).issubset(te_set):
        raise RuntimeError("Bootstrap test sample escaped test partition.")

    return btr, bva, bte


def bootstrap_indices_sha256(indices) -> str:
    """Hash one bootstrap replicate using explicit little-endian int64 bytes."""
    h = hashlib.sha256()
    for arr in indices:
        x = np.asarray(arr, dtype="<i8", order="C")
        h.update(x.tobytes(order="C"))
    return h.hexdigest()


def run_single_cell_bootstrap_validation(
    root: Path,
    replicate_id: int,
    layer_id: int,
    subset_mask: int,
    critic_seed: int,
    chunk_cubes: int,
    device_name: str,
):
    """Run exactly one bootstrap-resampled MINE fit for implementation validation.

    This is not part of the production inferential bootstrap distribution.
    """
    if layer_id not in LAYERS:
        raise ValueError(f"Invalid layer_id={layer_id}")
    if subset_mask not in ALL_MASKS:
        raise ValueError(f"Invalid subset_mask={subset_mask}")
    if critic_seed not in CRITIC_SEEDS:
        raise ValueError(f"Invalid critic_seed={critic_seed}")

    exp02 = root / "04 Experiments" / "EXP02_GB1_ESM2"

    p11s = root / "03 Code" / "11_run_gb1_target_mi.py"
    p11m = exp02 / "gb1_target_mi_manifest.json"
    p11r = exp02 / "gb1_target_mi_seed_results.csv"
    p04 = root / "03 Code" / "04_validate_mi_estimator.py"
    p08 = exp02 / "gb1_esm2_extraction_manifest.json"
    p09 = exp02 / "ESM2_Site_Reps" / "esm2_extraction_manifest.json"
    p10 = exp02 / "gb1_order_targets_manifest.json"
    psubsrc = root / "03 Code" / "11_validate_gb1_subset_conditioning.py"
    psubman = exp02 / "gb1_subset_conditioned_diag_manifest.json"

    subset_mod = load_module(psubsrc, "orbit_subset_builder_code14")

    (
        legacy_code11,
        mine04,
        _c11,
        _c08,
        c09,
        _c10,
        maps,
        samples,
        _permutations,
        _perm_meta,
        _eta2,
    ) = subset_mod.preflight(
        p11s, p11m, p11r, p04, p08, p09, p10
    )

    torch, nn = mine04.import_torch()

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cpu":
        torch.set_num_threads(1)

    split = deterministic_iid_split()
    bootstrap_split = bootstrap_indices_for_replicate(split, replicate_id)

    X = legacy_code11.chi_matrix()
    layer = np.load(
        c09["layer_paths"][layer_id],
        mmap_mode="r",
        allow_pickle=False,
    )

    U, V, _strict_split, _counts = subset_mod.build_subset_dataset(
        legacy_code11,
        layer,
        maps,
        samples,
        int(subset_mask),
        X,
        int(chunk_cubes),
    )

    if len(U) != N_TOTAL or len(V) != N_TOTAL:
        raise RuntimeError("Fixed-S biological observation count changed.")

    result = mine04.train_mine(
        torch,
        nn,
        U,
        V,
        int(critic_seed),
        device,
        bootstrap_split,
    )

    print("SINGLE-CELL BOOTSTRAP VALIDATION: PASS")
    print(f"replicate_id: {replicate_id}")
    print(f"layer: {layer_id}")
    print(f"subset_mask: {subset_mask}")
    print(f"order_k: {int(subset_mask).bit_count()}")
    print(f"critic_seed: {critic_seed}")
    print(f"device: {device.type}")
    print(f"bootstrap_index_sha256: {bootstrap_indices_sha256(bootstrap_split)}")
    print(f"test_mi: {result['test_mi']}")
    print("Production inferential distribution updated: NO")
    print("This fit is implementation validation only.")


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_csv_atomic(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_seed_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_seed_row(path: Path, fieldnames, row) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()
        os.fsync(f.fileno())


SEED_RESULT_FIELDS = [
    "replicate_id",
    "pilot",
    "bootstrap_index_sha256",
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
]


def compute_bootstrap_summaries(seed_rows):
    expected_n = len(LAYERS) * len(ALL_MASKS) * len(CRITIC_SEEDS)
    if len(seed_rows) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} completed seed rows, found {len(seed_rows)}."
        )

    grouped = {}
    seen = set()

    for r in seed_rows:
        layer = int(r["layer"])
        mask = int(r["subset_mask"])
        seed = int(r["critic_seed"])
        key3 = (layer, mask, seed)

        if key3 in seen:
            raise RuntimeError(f"Duplicate seed-result key: {key3}")
        seen.add(key3)

        if layer not in LAYERS or mask not in ALL_MASKS or seed not in CRITIC_SEEDS:
            raise RuntimeError(f"Unexpected seed-result condition: {key3}")

        score = float(r["test_mi"])
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite test_mi for {key3}")

        grouped.setdefault((layer, mask), []).append(score)

    cell_rows = []
    by_mask = {mask: [] for mask in ALL_MASKS}

    for layer in LAYERS:
        for mask in ALL_MASKS:
            vals = grouped.get((layer, mask), [])
            if len(vals) != len(CRITIC_SEEDS):
                raise RuntimeError(
                    f"Expected 3 critic scores for layer={layer}, mask={mask}; "
                    f"found {len(vals)}."
                )

            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))

            row = {
                "layer": int(layer),
                "order_k": int(mask).bit_count(),
                "subset_mask": int(mask),
                "score_mean": mean,
                "critic_seed_sd": sd,
                "n_critic_seeds": len(vals),
            }
            cell_rows.append(row)
            by_mask[mask].append(row)

    if len(cell_rows) != 465:
        raise RuntimeError(f"Expected 465 cell summaries, found {len(cell_rows)}.")

    retention_rows = []

    for mask in ALL_MASKS:
        rows = by_mask[mask]
        rows.sort(key=lambda x: int(x["layer"]))

        if [int(x["layer"]) for x in rows] != list(LAYERS):
            raise RuntimeError(f"Incomplete layer trajectory for mask={mask}")

        # Exact Code-13 tie rule: highest score, then earliest layer.
        peak = max(rows, key=lambda x: (float(x["score_mean"]), -int(x["layer"])))
        peak_score = float(peak["score_mean"])
        peak_layer = int(peak["layer"])

        final = next(x for x in rows if int(x["layer"]) == 30)
        final_score = float(final["score_mean"])

        eligible = peak_score > TAU
        retain_s = final_score / peak_score if eligible else None

        if retain_s is not None and not math.isfinite(retain_s):
            raise RuntimeError(f"Non-finite retention for mask={mask}")

        retention_rows.append(
            {
                "order_k": int(mask).bit_count(),
                "subset_mask": int(mask),
                "peak_layer": peak_layer,
                "peak_score": peak_score,
                "final_layer": 30,
                "final_score": final_score,
                "tau_q99_abs_fixed_subset_null": TAU,
                "retention_eligible": eligible,
                "final_near_floor": abs(final_score) <= TAU,
                "retain_s": retain_s,
            }
        )

    order_rows = []
    order_retention = {}

    for k in ORDERS:
        expected_masks = list(SUBSETS_BY_ORDER[k])
        rr = [
            r for r in retention_rows
            if int(r["order_k"]) == int(k)
        ]

        if sorted(int(r["subset_mask"]) for r in rr) != sorted(expected_masks):
            raise RuntimeError(f"Order-{k} subset membership mismatch.")

        vals = [r["retain_s"] for r in rr]
        estimable = all(v is not None for v in vals)

        retain_k = (
            float(np.mean([float(v) for v in vals]))
            if estimable
            else None
        )
        order_retention[k] = retain_k

        order_rows.append(
            {
                "order_k": int(k),
                "n_expected_subsets": len(expected_masks),
                "n_estimable_subsets": sum(v is not None for v in vals),
                "order_estimable": estimable,
                "retain_k": retain_k,
            }
        )

    r1 = order_retention[1]
    r2 = order_retention[2]
    r3 = order_retention[3]
    r4 = order_retention[4]

    lo = None if r1 is None or r2 is None else (r1 + r2) / 2.0
    ho = None if r3 is None or r4 is None else (r3 + r4) / 2.0
    delta = None if lo is None or ho is None else ho - lo

    contrast = {
        "retain_1": r1,
        "retain_2": r2,
        "retain_3": r3,
        "retain_4": r4,
        "lo_retention": lo,
        "ho_retention": ho,
        "delta_ho_minus_lo": delta,
        "directional_hypothesis": "delta_ho_minus_lo < 0",
        "contrast_estimable": delta is not None,
    }

    return cell_rows, retention_rows, order_rows, contrast


def run_full_bootstrap_replicate(
    root: Path,
    replicate_id: int,
    pilot: bool,
    resume: bool,
    chunk_cubes: int,
    device_name: str,
):
    """Run all 1,395 MINE fits for one bootstrap replicate."""

    if not (0 <= replicate_id < PRODUCTION_BOOTSTRAP_REPLICATES):
        raise ValueError(
            f"replicate_id must be in [0, {PRODUCTION_BOOTSTRAP_REPLICATES - 1}]"
        )

    exp02 = root / "04 Experiments" / "EXP02_GB1_ESM2"
    exp03 = root / "04 Experiments" / "EXP03_MI"
    iid_root = exp02 / "IID_Production_20260910"

    mode_dir = "Pilot" if pilot else "Production"
    out = iid_root / "Code14" / mode_dir / f"replicate_{replicate_id:03d}"
    out.mkdir(parents=True, exist_ok=True)

    seed_out = out / "bootstrap_seed_results.csv"
    cell_out = out / "bootstrap_cell_scores.csv"
    retention_out = out / "bootstrap_subset_retention.csv"
    order_out = out / "bootstrap_order_retention.csv"
    contrast_out = out / "bootstrap_contrast.json"
    state_out = out / "bootstrap_state.json"
    manifest_out = out / "bootstrap_manifest.json"

    p11iid = root / "03 Code" / "11_run_gb1_subset_conditioned_mi_iid.py"
    p11s = root / "03 Code" / "11_run_gb1_target_mi.py"
    p11m = exp02 / "gb1_target_mi_manifest.json"
    p11r = exp02 / "gb1_target_mi_seed_results.csv"
    p04 = root / "03 Code" / "04_validate_mi_estimator.py"
    p08 = exp02 / "gb1_esm2_extraction_manifest.json"
    p09 = exp02 / "ESM2_Site_Reps" / "esm2_extraction_manifest.json"
    p10 = exp02 / "gb1_order_targets_manifest.json"
    psubsrc = root / "03 Code" / "11_validate_gb1_subset_conditioning.py"
    psubman = exp02 / "gb1_subset_conditioned_diag_manifest.json"
    pregsrc = root / "03 Code" / "11_validate_gb1_fixed_subset_sample_regime.py"
    pregman = exp03 / "fixed_subset_sample_regime_manifest.json"

    # Exact corrected Code-11 implementation is already hash-pinned by preflight.
    code11iid = load_module(p11iid, "orbit_code11_iid_for_code14")

    subset_mod, _subset_manifest, subset_manifest_sha = (
        code11iid.verify_subset_diagnostic(psubsrc, psubman, exp02)
    )

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
        _eta2,
    ) = subset_mod.preflight(
        p11s, p11m, p11r, p04, p08, p09, p10
    )

    code11iid.verify_all_esm_layers(c09, subset_mod)
    _sample_manifest, sample_manifest_sha = code11iid.verify_sample_regime(
        pregsrc, pregman, exp03
    )

    torch, nn = mine04.import_torch()

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cpu":
        torch.set_num_threads(1)

    production_split = deterministic_iid_split()
    production_split_hash = iid_split_sha256(production_split)

    if production_split_hash != EXPECTED_PRODUCTION_IID_SPLIT_SHA256:
        raise RuntimeError("Production IID split changed inside full replicate.")

    bootstrap_split = bootstrap_indices_for_replicate(
        production_split, replicate_id
    )
    bootstrap_hash = bootstrap_indices_sha256(bootstrap_split)

    expected_keys = {
        (int(layer), int(mask), int(seed))
        for layer in LAYERS
        for mask in ALL_MASKS
        for seed in CRITIC_SEEDS
    }

    existing = load_seed_rows(seed_out)
    done = set()

    for r in existing:
        key = (
            int(r["layer"]),
            int(r["subset_mask"]),
            int(r["critic_seed"]),
        )
        if key in done:
            raise RuntimeError(f"Duplicate checkpoint row: {key}")
        if key not in expected_keys:
            raise RuntimeError(f"Unexpected checkpoint row: {key}")
        if int(r["replicate_id"]) != replicate_id:
            raise RuntimeError("Checkpoint replicate_id mismatch.")
        if str(r["bootstrap_index_sha256"]) != bootstrap_hash:
            raise RuntimeError("Checkpoint bootstrap index hash mismatch.")
        done.add(key)

    if existing and not resume and len(done) < len(expected_keys):
        raise RuntimeError(
            "Partial replicate checkpoint exists. Re-run with --resume."
        )

    if manifest_out.exists():
        with manifest_out.open("r", encoding="utf-8") as f:
            m = json.load(f)
        if bool(m.get("overall_pass", False)):
            print("FULL BOOTSTRAP REPLICATE: PASS")
            print("Matching completed replicate already exists.")
            print(f"Manifest: {manifest_out}")
            return

    state_base = {
        "code_name": CODE_NAME,
        "schema_version": SCHEMA_VERSION,
        "replicate_id": replicate_id,
        "pilot": pilot,
        "bootstrap_master_seed": BOOTSTRAP_MASTER_SEED,
        "bootstrap_index_sha256": bootstrap_hash,
        "production_iid_split_sha256": production_split_hash,
        "expected_fits": 1395,
        "output_directory": str(out),
    }

    write_json_atomic(
        state_out,
        {
            **state_base,
            "status": "running",
            "completed_fits": len(done),
            "updated_unix_time": time.time(),
        },
    )

    X = legacy_code11.chi_matrix()

    print("=" * 86)
    print("CODE 14 FULL BOOTSTRAP REPLICATE")
    print("=" * 86)
    print(f"replicate_id: {replicate_id}")
    print(f"pilot: {pilot}")
    print(f"bootstrap index SHA256: {bootstrap_hash}")
    print(f"already completed fits: {len(done)}/1395")
    print(f"device: {device.type}")
    print("Peak layer will be reselected after all 465 cells are recomputed.")
    print("Production inference membership:", "NO (pilot)" if pilot else "YES")

    for layer_id in LAYERS:
        needed_layer = any(
            (int(layer_id), int(mask), int(seed)) not in done
            for mask in ALL_MASKS
            for seed in CRITIC_SEEDS
        )
        if not needed_layer:
            continue

        print(f"\nLayer {layer_id:02d}", flush=True)

        layer = np.load(
            c09["layer_paths"][layer_id],
            mmap_mode="r",
            allow_pickle=False,
        )

        for mask in ALL_MASKS:
            missing_seeds = [
                int(seed)
                for seed in CRITIC_SEEDS
                if (int(layer_id), int(mask), int(seed)) not in done
            ]
            if not missing_seeds:
                continue

            k = int(mask).bit_count()
            print(
                f"  mask={mask:02d} k={k}: build fixed-S dataset",
                flush=True,
            )

            U, V, _strict_split, _counts = subset_mod.build_subset_dataset(
                legacy_code11,
                layer,
                maps,
                samples,
                int(mask),
                X,
                int(chunk_cubes),
            )

            if len(U) != N_TOTAL or len(V) != N_TOTAL:
                raise RuntimeError(
                    f"Fixed-S sample count changed for layer={layer_id}, mask={mask}"
                )

            for seed in missing_seeds:
                print(f"    critic_seed={seed}", flush=True)

                res = mine04.train_mine(
                    torch,
                    nn,
                    U,
                    V,
                    int(seed),
                    device,
                    bootstrap_split,
                )

                row = {
                    "replicate_id": replicate_id,
                    "pilot": pilot,
                    "bootstrap_index_sha256": bootstrap_hash,
                    "layer": int(layer_id),
                    "order_k": int(k),
                    "subset_mask": int(mask),
                    "critic_seed": int(seed),
                    "representation_dim": int(U.shape[1]),
                    "target_dim": int(V.shape[1]),
                    "n_samples": len(U),
                    "n_train": N_TRAIN,
                    "n_val": N_VAL,
                    "n_test": N_TEST,
                    **res,
                }

                append_seed_row(
                    seed_out,
                    SEED_RESULT_FIELDS,
                    row,
                )

                done.add((int(layer_id), int(mask), int(seed)))

                write_json_atomic(
                    state_out,
                    {
                        **state_base,
                        "status": "running",
                        "completed_fits": len(done),
                        "updated_unix_time": time.time(),
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

    seed_rows = load_seed_rows(seed_out)

    cell_rows, retention_rows, order_rows, contrast = (
        compute_bootstrap_summaries(seed_rows)
    )

    write_csv_atomic(
        cell_out,
        [
            "layer",
            "order_k",
            "subset_mask",
            "score_mean",
            "critic_seed_sd",
            "n_critic_seeds",
        ],
        cell_rows,
    )

    write_csv_atomic(
        retention_out,
        [
            "order_k",
            "subset_mask",
            "peak_layer",
            "peak_score",
            "final_layer",
            "final_score",
            "tau_q99_abs_fixed_subset_null",
            "retention_eligible",
            "final_near_floor",
            "retain_s",
        ],
        retention_rows,
    )

    write_csv_atomic(
        order_out,
        [
            "order_k",
            "n_expected_subsets",
            "n_estimable_subsets",
            "order_estimable",
            "retain_k",
        ],
        order_rows,
    )

    contrast_payload = {
        "code_name": CODE_NAME,
        "schema_version": SCHEMA_VERSION,
        "replicate_id": replicate_id,
        "pilot": pilot,
        "bootstrap_index_sha256": bootstrap_hash,
        **contrast,
    }
    write_json_atomic(contrast_out, contrast_payload)

    output_files = [
        seed_out,
        cell_out,
        retention_out,
        order_out,
        contrast_out,
    ]

    manifest = {
        "code_name": CODE_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_pass",
        "overall_pass": True,
        "replicate_id": replicate_id,
        "pilot": pilot,
        "included_in_production_inference": not pilot,
        "code14_source_sha256": sha256_file(Path(__file__)),
        "bootstrap_master_seed": BOOTSTRAP_MASTER_SEED,
        "bootstrap_index_sha256": bootstrap_hash,
        "production_iid_split_seed": PRODUCTION_IID_SPLIT_SEED,
        "production_iid_split_sha256": production_split_hash,
        "contract_hashes": {
            "parent_inference_contract": EXPECTED_PARENT_CONTRACT_SHA256,
            "implementation_contract": EXPECTED_IMPLEMENTATION_CONTRACT_SHA256,
            "execution_contract": EXPECTED_EXECUTION_CONTRACT_SHA256,
        },
        "upstream": {
            "corrected_code11_iid_sha256": EXPECTED_CODE11_IID_SHA256,
            "subset_builder_sha256": EXPECTED_SUBSET_BUILDER_SHA256,
            "code04_mine_sha256": EXPECTED_CODE04_MINE_SHA256,
            "subset_diagnostic_manifest_sha256": subset_manifest_sha,
            "sample_regime_manifest_sha256": sample_manifest_sha,
            "legacy_code11_manifest_sha256": c11["manifest_sha"],
            "code08_manifest_sha256": c08["sha"],
            "code09_manifest_sha256": c09["sha"],
            "code10_manifest_sha256": c10["sha"],
        },
        "computational_audit": {
            "expected_seed_fits": 1395,
            "observed_seed_fits": len(seed_rows),
            "expected_layer_subset_cells": 465,
            "observed_layer_subset_cells": len(cell_rows),
            "expected_subset_trajectories": 15,
            "observed_subset_trajectories": len(retention_rows),
            "expected_layers_per_trajectory": 31,
            "critic_seeds": list(CRITIC_SEEDS),
            "critic_seeds_are_biological_replicates": False,
            "partition_preserving_bootstrap": True,
            "cross_partition_resampling": False,
            "shared_indices_across_layers_subsets_seeds": True,
            "sampling_with_replacement": True,
            "peak_layer_reselected": True,
            "negative_scores_clamped": False,
            "null_scores_subtracted": False,
            "tau": TAU,
            "overall_pass": (
                len(seed_rows) == 1395
                and len(cell_rows) == 465
                and len(retention_rows) == 15
            ),
        },
        "contrast": contrast,
        "outputs": {
            x.name: {
                "sha256": sha256_file(x),
                "bytes": x.stat().st_size,
            }
            for x in output_files
        },
    }

    write_json_atomic(manifest_out, manifest)

    write_json_atomic(
        state_out,
        {
            **state_base,
            "status": "complete_pass",
            "completed_fits": 1395,
            "updated_unix_time": time.time(),
            "manifest_sha256": sha256_file(manifest_out),
            "contrast_estimable": contrast["contrast_estimable"],
            "delta_ho_minus_lo": contrast["delta_ho_minus_lo"],
        },
    )

    print("\nFULL BOOTSTRAP REPLICATE: PASS")
    print("Seed fits: 1395/1395")
    print("Layer x subset cells: 465/465")
    print("Subset trajectories: 15/15")
    print(f"Contrast estimable: {contrast['contrast_estimable']}")
    print(f"delta_HO_minus_LO: {contrast['delta_ho_minus_lo']}")
    print(f"Pilot excluded from production inference: {pilot}")
    print(f"Manifest: {manifest_out}")



def run_delete_one_identity_pilot(
    root: Path,
    resume: bool,
    chunk_cubes: int,
    device_name: str,
    check_only: bool = False,
):
    """
    Computational-validation pilot only.

    Frozen deletion:
        biological position 41
        alternative identity L

    This is NOT an inferential replicate and MUST NOT be included in any
    CI, p-value, tail probability, or significance claim.
    """

    import csv
    import hashlib

    DELETE_POSITION = 41
    DELETE_IDENTITY = "L"
    DELETE_COLUMN = 2  # VDGV coordinates: 39,40,41,54
    WT = "VDGV"

    EXPECTED_CANDIDATE_CONTRACT_SHA256 = (
        "9552cec3e05a468af0f8061b6b38469f98e4a65987d0aa4aa882a898a5c089d9"
    )
    EXPECTED_IDENTITY_CONTRACT_SHA256 = (
        "8313a41e42d56af63aa9c3b814c4b99c241694c5eaa22a5b07a16e1a27bb7f8d"
    )
    EXPECTED_IDENTITY_SUPPORT_SHA256 = (
        "ab4e72b7deb716dc137f6f9eecbe04c8122a276b31d2c6811565631a76b562e8"
    )

    EXPECTED_SURVIVING = {
        "train": 6596,
        "val": 209,
        "test": 557,
        "all": 7362,
    }

    exp02 = root / "04 Experiments" / "EXP02_GB1_ESM2"
    exp03 = root / "04 Experiments" / "EXP03_MI"
    iid_root = exp02 / "IID_Production_20260910"

    candidate_contract = (
        iid_root / "code14_delete_one_identity_candidate_contract.json"
    )
    identity_contract = iid_root / "code14_identity_support_contract.json"

    require_hash(
        candidate_contract,
        EXPECTED_CANDIDATE_CONTRACT_SHA256,
        "Code-14 delete-one-AA candidate contract",
    )
    require_hash(
        identity_contract,
        EXPECTED_IDENTITY_CONTRACT_SHA256,
        "Code-14 recovered identity-support contract",
    )

    out = (
        iid_root
        / "Code14"
        / "DeleteOneIdentityPilot"
        / "position41_L"
    )
    out.mkdir(parents=True, exist_ok=True)

    seed_out = out / "delete_one_identity_seed_results.csv"
    cell_out = out / "delete_one_identity_cell_scores.csv"
    retention_out = out / "delete_one_identity_subset_retention.csv"
    order_out = out / "delete_one_identity_order_retention.csv"
    contrast_out = out / "delete_one_identity_contrast.json"
    state_out = out / "delete_one_identity_state.json"
    manifest_out = out / "delete_one_identity_manifest.json"

    p11iid = root / "03 Code" / "11_run_gb1_subset_conditioned_mi_iid.py"
    p11s = root / "03 Code" / "11_run_gb1_target_mi.py"
    p11m = exp02 / "gb1_target_mi_manifest.json"
    p11r = exp02 / "gb1_target_mi_seed_results.csv"
    p04 = root / "03 Code" / "04_validate_mi_estimator.py"
    p08 = exp02 / "gb1_esm2_extraction_manifest.json"
    p09 = exp02 / "ESM2_Site_Reps" / "esm2_extraction_manifest.json"
    p10 = exp02 / "gb1_order_targets_manifest.json"
    psubsrc = root / "03 Code" / "11_validate_gb1_subset_conditioning.py"
    psubman = exp02 / "gb1_subset_conditioned_diag_manifest.json"
    pregsrc = root / "03 Code" / "11_validate_gb1_fixed_subset_sample_regime.py"
    pregman = exp03 / "fixed_subset_sample_regime_manifest.json"

    code11iid = load_module(
        p11iid,
        "orbit_code11_iid_for_code14_delete_identity_pilot",
    )

    subset_mod, _subset_manifest, subset_manifest_sha = (
        code11iid.verify_subset_diagnostic(psubsrc, psubman, exp02)
    )

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
        _eta2,
    ) = subset_mod.preflight(
        p11s, p11m, p11r, p04, p08, p09, p10
    )

    code11iid.verify_all_esm_layers(c09, subset_mod)
    _sample_manifest, sample_manifest_sha = code11iid.verify_sample_regime(
        pregsrc, pregman, exp03
    )

    # ------------------------------------------------------------
    # Reconstruct authoritative strict-cube AA identities from
    # surviving Code-08 artifacts.
    # ------------------------------------------------------------
    index_path = exp02 / "gb1_esm2_extraction_index.csv"
    map_path = exp02 / "gb1_strict_cube_extract_rows.npz"

    variants = []
    with index_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for expected_id, row in enumerate(reader):
            if int(row["extract_row_id"]) != expected_id:
                raise RuntimeError("extract_row_id is not contiguous.")
            variants.append(str(row["variant"]))

    variants = np.asarray(variants, dtype="U4")

    with np.load(map_path, allow_pickle=False) as z:
        strict_maps = [
            (
                "estimator_train",
                np.asarray(
                    z["estimator_train_extract_rows"],
                    dtype=np.int64,
                ),
            ),
            (
                "estimator_val",
                np.asarray(
                    z["estimator_val_extract_rows"],
                    dtype=np.int64,
                ),
            ),
            (
                "estimator_test",
                np.asarray(
                    z["estimator_test_extract_rows"],
                    dtype=np.int64,
                ),
            ),
        ]

    expected_shapes = {
        "estimator_train": (7352, 16),
        "estimator_val": (245, 16),
        "estimator_test": (613, 16),
    }

    identity_rows = []
    fingerprint_rows = []
    global_offset = 0

    for split_name, m in strict_maps:
        if m.shape != expected_shapes[split_name]:
            raise RuntimeError(
                f"{split_name}: unexpected cube-map shape {m.shape}"
            )

        cube_variants = variants[m]

        for local_i, verts in enumerate(cube_variants):
            if verts[0] != WT:
                raise RuntimeError(
                    f"{split_name} cube {local_i}: vertex 0 is not WT."
                )

            alts = []
            for j in range(4):
                nonwt = sorted(
                    {v[j] for v in verts if v[j] != WT[j]}
                )
                if len(nonwt) != 1:
                    raise RuntimeError(
                        f"{split_name} cube {local_i}: "
                        f"position index {j} alternatives={nonwt}"
                    )
                alts.append(nonwt[0])

            far = verts[15]
            alt_tuple = "".join(alts)

            if far != alt_tuple:
                raise RuntimeError(
                    f"{split_name} cube {local_i}: "
                    f"{alt_tuple} != far corner {far}"
                )

            identity_rows.append(tuple(alts))

            global_id = global_offset + local_i
            fingerprint_rows.append(
                (
                    global_id,
                    split_name,
                    local_i,
                    alts[0],
                    alts[1],
                    alts[2],
                    alts[3],
                    far,
                )
            )

        global_offset += len(m)

    ident = np.asarray(identity_rows, dtype="U1")

    if ident.shape != (N_TOTAL, 4):
        raise RuntimeError(
            f"Recovered identity matrix shape changed: {ident.shape}"
        )

    h = hashlib.sha256()
    for r in fingerprint_rows:
        h.update(
            (
                f"{r[0]}|{r[1]}|{r[2]}|"
                f"{r[3]}|{r[4]}|{r[5]}|{r[6]}|{r[7]}\n"
            ).encode("utf-8")
        )

    identity_support_hash = h.hexdigest()

    if identity_support_hash != EXPECTED_IDENTITY_SUPPORT_SHA256:
        raise RuntimeError(
            "Recovered strict identity support SHA256 mismatch.\n"
            f"Expected: {EXPECTED_IDENTITY_SUPPORT_SHA256}\n"
            f"Observed: {identity_support_hash}"
        )

    # ------------------------------------------------------------
    # Delete every cube containing position 41 = L.
    # ------------------------------------------------------------
    keep = ident[:, DELETE_COLUMN] != DELETE_IDENTITY
    surviving_global = np.flatnonzero(keep).astype(np.int64)

    if len(surviving_global) != EXPECTED_SURVIVING["all"]:
        raise RuntimeError(
            "Unexpected total survivors after position41=L deletion."
        )

    deletion_support_hash = hashlib.sha256(
        np.asarray(surviving_global, dtype="<i8").tobytes(order="C")
    ).hexdigest()

    # ------------------------------------------------------------
    # Retain original corrected-IID partition membership.
    # No repartitioning.
    # ------------------------------------------------------------
    production_split = deterministic_iid_split()
    production_split_hash = iid_split_sha256(production_split)

    if production_split_hash != EXPECTED_PRODUCTION_IID_SPLIT_SHA256:
        raise RuntimeError(
            "Production IID split changed inside delete-one-AA pilot."
        )

    global_to_local = np.full(N_TOTAL, -1, dtype=np.int64)
    global_to_local[surviving_global] = np.arange(
        len(surviving_global),
        dtype=np.int64,
    )

    pilot_split = []

    for original_indices in production_split:
        kept_global = original_indices[keep[original_indices]]
        mapped = global_to_local[kept_global]

        if np.any(mapped < 0):
            raise RuntimeError("Failed to map surviving partition indices.")

        pilot_split.append(
            np.asarray(mapped, dtype=np.int64)
        )

    pilot_split = tuple(pilot_split)

    observed_counts = {
        "train": len(pilot_split[0]),
        "val": len(pilot_split[1]),
        "test": len(pilot_split[2]),
        "all": len(surviving_global),
    }

    if observed_counts != EXPECTED_SURVIVING:
        raise RuntimeError(
            "Frozen delete-one-AA survivor counts changed.\n"
            f"Expected: {EXPECTED_SURVIVING}\n"
            f"Observed: {observed_counts}"
        )

    if check_only:
        print("DELETE-ONE-AA PILOT CHECK: PASS")
        print("Deletion: position 41 = L")
        print(
            f"Survivors: train={observed_counts['train']} "
            f"val={observed_counts['val']} "
            f"test={observed_counts['test']} "
            f"all={observed_counts['all']}"
        )
        print(f"Identity support SHA256: {identity_support_hash}")
        print(f"Deletion support SHA256: {deletion_support_hash}")
        print("Original corrected-IID partition membership retained: YES")
        print("Survivors repartitioned: NO")
        print("MINE fits executed: 0")
        print("Formal inference executed: NO")
        return

    torch, nn = mine04.import_torch()

    if device_name == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_name)

    if device.type == "cpu":
        torch.set_num_threads(1)

    # New checkpoint schema: deliberately avoids calling this a bootstrap.
    pilot_fields = [
        "deletion_position",
        "deletion_identity",
        "deletion_support_sha256",
    ] + [
        x
        for x in SEED_RESULT_FIELDS
        if x not in (
            "replicate_id",
            "pilot",
            "bootstrap_index_sha256",
        )
    ]

    expected_keys = {
        (int(layer), int(mask), int(seed))
        for layer in LAYERS
        for mask in ALL_MASKS
        for seed in CRITIC_SEEDS
    }

    existing = load_seed_rows(seed_out)
    done = set()

    for r in existing:
        key = (
            int(r["layer"]),
            int(r["subset_mask"]),
            int(r["critic_seed"]),
        )

        if key in done:
            raise RuntimeError(f"Duplicate checkpoint row: {key}")
        if key not in expected_keys:
            raise RuntimeError(f"Unexpected checkpoint row: {key}")

        if int(r["deletion_position"]) != DELETE_POSITION:
            raise RuntimeError("Checkpoint deletion position mismatch.")
        if str(r["deletion_identity"]) != DELETE_IDENTITY:
            raise RuntimeError("Checkpoint deletion identity mismatch.")
        if str(r["deletion_support_sha256"]) != deletion_support_hash:
            raise RuntimeError("Checkpoint deletion-support hash mismatch.")

        done.add(key)

    if existing and not resume and len(done) < len(expected_keys):
        raise RuntimeError(
            "Partial delete-one-AA pilot checkpoint exists. "
            "Re-run with --resume."
        )

    if manifest_out.exists():
        with manifest_out.open("r", encoding="utf-8") as f:
            m = json.load(f)

        if bool(m.get("overall_pass", False)):
            print("DELETE-ONE-AA COMPUTATIONAL PILOT: PASS")
            print("Matching completed pilot already exists.")
            print(f"Manifest: {manifest_out}")
            return

    state_base = {
        "code_name": CODE_NAME,
        "schema_version": SCHEMA_VERSION,
        "mode": "delete_one_identity_computational_pilot",
        "deletion_position": DELETE_POSITION,
        "deletion_identity": DELETE_IDENTITY,
        "deletion_support_sha256": deletion_support_hash,
        "identity_support_sha256": identity_support_hash,
        "production_iid_split_sha256": production_split_hash,
        "surviving_counts": observed_counts,
        "expected_fits": 1395,
        "included_in_formal_inference": False,
        "output_directory": str(out),
    }

    write_json_atomic(
        state_out,
        {
            **state_base,
            "status": "running",
            "completed_fits": len(done),
            "updated_unix_time": time.time(),
        },
    )

    X = legacy_code11.chi_matrix()

    print("=" * 86)
    print("CODE 14 DELETE-ONE-AA COMPUTATIONAL PILOT")
    print("=" * 86)
    print("Deletion: position 41 = L")
    print(
        "Survivors: "
        f"train={observed_counts['train']} "
        f"val={observed_counts['val']} "
        f"test={observed_counts['test']} "
        f"all={observed_counts['all']}"
    )
    print(f"Identity support SHA256: {identity_support_hash}")
    print(f"Deletion support SHA256: {deletion_support_hash}")
    print(f"Already completed fits: {len(done)}/1395")
    print(f"Device: {device.type}")
    print("Original corrected-IID partition membership retained: YES")
    print("Repartition surviving cubes: NO")
    print("Formal inference membership: NO")
    print("CI/p-value authorization: NO")

    for layer_id in LAYERS:
        needed_layer = any(
            (int(layer_id), int(mask), int(seed)) not in done
            for mask in ALL_MASKS
            for seed in CRITIC_SEEDS
        )

        if not needed_layer:
            continue

        print(f"\nLayer {layer_id:02d}", flush=True)

        layer = np.load(
            c09["layer_paths"][layer_id],
            mmap_mode="r",
            allow_pickle=False,
        )

        for mask in ALL_MASKS:
            missing_seeds = [
                int(seed)
                for seed in CRITIC_SEEDS
                if (
                    int(layer_id),
                    int(mask),
                    int(seed),
                ) not in done
            ]

            if not missing_seeds:
                continue

            k = int(mask).bit_count()

            print(
                f"  mask={mask:02d} k={k}: "
                "build fixed-S dataset and apply frozen identity deletion",
                flush=True,
            )

            U, V, _strict_split, _counts = (
                subset_mod.build_subset_dataset(
                    legacy_code11,
                    layer,
                    maps,
                    samples,
                    int(mask),
                    X,
                    int(chunk_cubes),
                )
            )

            if len(U) != N_TOTAL or len(V) != N_TOTAL:
                raise RuntimeError(
                    f"Fixed-S sample count changed for "
                    f"layer={layer_id}, mask={mask}"
                )

            U_delete = np.asarray(U[surviving_global])
            V_delete = np.asarray(V[surviving_global])

            if len(U_delete) != EXPECTED_SURVIVING["all"]:
                raise RuntimeError("Deleted U sample count mismatch.")
            if len(V_delete) != EXPECTED_SURVIVING["all"]:
                raise RuntimeError("Deleted V sample count mismatch.")

            for seed in missing_seeds:
                print(f"    critic_seed={seed}", flush=True)

                res = mine04.train_mine(
                    torch,
                    nn,
                    U_delete,
                    V_delete,
                    int(seed),
                    device,
                    pilot_split,
                )

                row = {
                    "deletion_position": DELETE_POSITION,
                    "deletion_identity": DELETE_IDENTITY,
                    "deletion_support_sha256": deletion_support_hash,
                    "layer": int(layer_id),
                    "order_k": int(k),
                    "subset_mask": int(mask),
                    "critic_seed": int(seed),
                    "representation_dim": int(U_delete.shape[1]),
                    "target_dim": int(V_delete.shape[1]),
                    "n_samples": len(U_delete),
                    "n_train": observed_counts["train"],
                    "n_val": observed_counts["val"],
                    "n_test": observed_counts["test"],
                    **res,
                }

                append_seed_row(
                    seed_out,
                    pilot_fields,
                    row,
                )

                done.add(
                    (
                        int(layer_id),
                        int(mask),
                        int(seed),
                    )
                )

                write_json_atomic(
                    state_out,
                    {
                        **state_base,
                        "status": "running",
                        "completed_fits": len(done),
                        "updated_unix_time": time.time(),
                        "last_completed": {
                            "layer": int(layer_id),
                            "order_k": int(k),
                            "subset_mask": int(mask),
                            "critic_seed": int(seed),
                        },
                    },
                )

            del U, V, U_delete, V_delete

        del layer

    seed_rows = load_seed_rows(seed_out)

    if len(seed_rows) != 1395:
        raise RuntimeError(
            f"Expected 1395 completed pilot fits; got {len(seed_rows)}"
        )

    cell_rows, retention_rows, order_rows, contrast = (
        compute_bootstrap_summaries(seed_rows)
    )

    write_csv_atomic(
        cell_out,
        [
            "layer",
            "order_k",
            "subset_mask",
            "score_mean",
            "critic_seed_sd",
            "n_critic_seeds",
        ],
        cell_rows,
    )

    write_csv_atomic(
        retention_out,
        [
            "order_k",
            "subset_mask",
            "peak_layer",
            "peak_score",
            "final_layer",
            "final_score",
            "tau_q99_abs_fixed_subset_null",
            "retention_eligible",
            "final_near_floor",
            "retain_s",
        ],
        retention_rows,
    )

    write_csv_atomic(
        order_out,
        [
            "order_k",
            "n_expected_subsets",
            "n_estimable_subsets",
            "order_estimable",
            "retain_k",
        ],
        order_rows,
    )

    contrast_payload = {
        "code_name": CODE_NAME,
        "schema_version": SCHEMA_VERSION,
        "mode": "delete_one_identity_computational_pilot",
        "deletion_position": DELETE_POSITION,
        "deletion_identity": DELETE_IDENTITY,
        "included_in_formal_inference": False,
        "interpretation": (
            "Computational validation output only; "
            "not an inferential replicate."
        ),
        **contrast,
    }

    write_json_atomic(
        contrast_out,
        contrast_payload,
    )

    output_files = [
        seed_out,
        cell_out,
        retention_out,
        order_out,
        contrast_out,
    ]

    manifest = {
        "code_name": CODE_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_pass",
        "overall_pass": True,
        "mode": "delete_one_identity_computational_pilot",
        "included_in_production_inference": False,
        "included_in_formal_inference": False,
        "formal_inference_authorized": False,
        "deletion_position": DELETE_POSITION,
        "deletion_identity": DELETE_IDENTITY,
        "identity_support_sha256": identity_support_hash,
        "deletion_support_sha256": deletion_support_hash,
        "code14_source_sha256": sha256_file(Path(__file__)),
        "production_iid_split_seed": PRODUCTION_IID_SPLIT_SEED,
        "production_iid_split_sha256": production_split_hash,
        "surviving_counts": observed_counts,
        "contract_hashes": {
            "candidate_delete_one_identity_contract":
                EXPECTED_CANDIDATE_CONTRACT_SHA256,
            "identity_support_contract":
                EXPECTED_IDENTITY_CONTRACT_SHA256,
        },
        "upstream": {
            "corrected_code11_iid_sha256": EXPECTED_CODE11_IID_SHA256,
            "subset_builder_sha256": EXPECTED_SUBSET_BUILDER_SHA256,
            "code04_mine_sha256": EXPECTED_CODE04_MINE_SHA256,
            "subset_diagnostic_manifest_sha256": subset_manifest_sha,
            "sample_regime_manifest_sha256": sample_manifest_sha,
            "legacy_code11_manifest_sha256": c11["manifest_sha"],
            "code08_manifest_sha256": c08["sha"],
            "code09_manifest_sha256": c09["sha"],
            "code10_manifest_sha256": c10["sha"],
        },
        "computational_audit": {
            "expected_seed_fits": 1395,
            "observed_seed_fits": len(seed_rows),
            "expected_layer_subset_cells": 465,
            "observed_layer_subset_cells": len(cell_rows),
            "expected_subset_trajectories": 15,
            "observed_subset_trajectories": len(retention_rows),
            "critic_seeds": list(CRITIC_SEEDS),
            "critic_seeds_are_biological_replicates": False,
            "whole_identity_deleted_together": True,
            "new_cube_combinations_generated": False,
            "mixed_cubes_generated": False,
            "original_corrected_iid_membership_retained": True,
            "survivors_repartitioned": False,
            "peak_layer_reselected": True,
            "negative_scores_clamped": False,
            "null_scores_subtracted": False,
            "tau": TAU,
            "overall_pass": (
                len(seed_rows) == 1395
                and len(cell_rows) == 465
                and len(retention_rows) == 15
            ),
        },
        "contrast": contrast,
        "interpretation_boundary": [
            "This run validates computational behavior under the frozen position-41=L identity deletion.",
            "It is excluded from all formal inference.",
            "Its contrast is descriptive computational output only.",
            "No CI, p-value, bootstrap tail probability, or significance claim is authorized from this pilot."
        ],
        "outputs": {
            x.name: {
                "sha256": sha256_file(x),
                "bytes": x.stat().st_size,
            }
            for x in output_files
        },
    }

    write_json_atomic(
        manifest_out,
        manifest,
    )

    write_json_atomic(
        state_out,
        {
            **state_base,
            "status": "complete_pass",
            "completed_fits": 1395,
            "updated_unix_time": time.time(),
            "manifest_sha256": sha256_file(manifest_out),
            "contrast_estimable": contrast["contrast_estimable"],
            "delta_ho_minus_lo": contrast["delta_ho_minus_lo"],
        },
    )

    print("\nDELETE-ONE-AA COMPUTATIONAL PILOT: PASS")
    print("Seed fits: 1395/1395")
    print("Layer x subset cells: 465/465")
    print("Subset trajectories: 15/15")
    print(f"Contrast estimable: {contrast['contrast_estimable']}")
    print(
        "Descriptive pilot delta_HO_minus_LO: "
        f"{contrast['delta_ho_minus_lo']}"
    )
    print("Included in formal inference: NO")
    print("CI/p-value authorization: NO")
    print(f"Manifest: {manifest_out}")


def bootstrap_index_self_test():
    """Audit deterministic generation without running any MINE fit."""
    split = deterministic_iid_split()

    a0 = bootstrap_indices_for_replicate(split, 0)
    a0_repeat = bootstrap_indices_for_replicate(split, 0)
    a1 = bootstrap_indices_for_replicate(split, 1)

    h0 = bootstrap_indices_sha256(a0)
    h0_repeat = bootstrap_indices_sha256(a0_repeat)
    h1 = bootstrap_indices_sha256(a1)

    if h0 != h0_repeat:
        raise RuntimeError("Bootstrap replicate 0 is not deterministic.")
    if h0 == h1:
        raise RuntimeError("Bootstrap replicate 0 and 1 unexpectedly have identical hashes.")

    # With-replacement sampling should preserve requested counts; uniqueness
    # is reported descriptively rather than required.
    print("BOOTSTRAP INDEX SELF-TEST: PASS")
    print(f"Bootstrap master seed: {BOOTSTRAP_MASTER_SEED}")
    print(f"Production bootstrap replicates: {PRODUCTION_BOOTSTRAP_REPLICATES}")
    print(f"Replicate 0 SHA256: {h0}")
    print(f"Replicate 1 SHA256: {h1}")
    print(
        "Replicate 0 unique draws: "
        f"train={len(np.unique(a0[0]))}/{N_TRAIN}, "
        f"val={len(np.unique(a0[1]))}/{N_VAL}, "
        f"test={len(np.unique(a0[2]))}/{N_TEST}"
    )
    print("Sampling with replacement: YES")
    print("Partition preserving: YES")
    print("Cross-partition resampling: NO")
    print("MINE fits executed: 0")
    print("Inferential bootstrap executed: NO")


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {got}\n"
            f"Path: {path}"
        )


def preflight(root: Path) -> None:
    exp = root / "04 Experiments" / "EXP02_GB1_ESM2"
    iid = exp / "IID_Production_20260910"

    parent_contract = iid / "code14_inference_contract.json"
    implementation_contract = iid / "code14_bootstrap_implementation_contract.json"
    execution_contract = iid / "code14_bootstrap_execution_contract.json"
    supersession_contract = iid / "code14_statistical_design_supersession_01.json"

    code11 = root / "03 Code" / "11_run_gb1_subset_conditioned_mi_iid.py"
    subset_builder = root / "03 Code" / "11_validate_gb1_subset_conditioning.py"
    code04 = root / "03 Code" / "04_validate_mi_estimator.py"

    require_hash(
        parent_contract,
        EXPECTED_PARENT_CONTRACT_SHA256,
        "Code-14 parent inference contract",
    )
    require_hash(
        implementation_contract,
        EXPECTED_IMPLEMENTATION_CONTRACT_SHA256,
        "Code-14 bootstrap implementation contract",
    )
    require_hash(
        execution_contract,
        EXPECTED_EXECUTION_CONTRACT_SHA256,
        "Code-14 bootstrap execution contract",
    )
    require_hash(
        supersession_contract,
        EXPECTED_STATISTICAL_SUPERSESSION_SHA256,
        "Code-14 statistical-design supersession contract",
    )
    require_hash(
        code11,
        EXPECTED_CODE11_IID_SHA256,
        "Corrected IID Code 11",
    )
    require_hash(
        subset_builder,
        EXPECTED_SUBSET_BUILDER_SHA256,
        "Fixed-subset dataset builder",
    )
    require_hash(
        code04,
        EXPECTED_CODE04_MINE_SHA256,
        "Code-04 MINE estimator",
    )

    with parent_contract.open("r", encoding="utf-8") as f:
        parent = json.load(f)
    with implementation_contract.open("r", encoding="utf-8") as f:
        implementation = json.load(f)
    with execution_contract.open("r", encoding="utf-8") as f:
        execution = json.load(f)
    with supersession_contract.open("r", encoding="utf-8") as f:
        supersession = json.load(f)

    if parent.get("status") != "FROZEN_BEFORE_INFERENCE":
        raise RuntimeError("Parent inference contract is not frozen.")
    if implementation.get("status") != "FROZEN_BEFORE_INFERENCE":
        raise RuntimeError("Bootstrap implementation contract is not frozen.")
    if execution.get("status") != "FROZEN_BEFORE_INFERENCE":
        raise RuntimeError("Bootstrap execution contract is not frozen.")
    if supersession.get("status") != "FROZEN_BEFORE_INFERENCE":
        raise RuntimeError("Statistical-design supersession is not frozen.")

    if execution.get("production_iid_split_sha256") != EXPECTED_PRODUCTION_IID_SPLIT_SHA256:
        raise RuntimeError("Frozen production IID split SHA256 mismatch.")

    if int(execution.get("production_bootstrap_replicates", -1)) != 499:
        raise RuntimeError("Frozen production bootstrap replicate count mismatch.")

    if int(execution.get("bootstrap_master_seed", -1)) != 20260911:
        raise RuntimeError("Frozen bootstrap master seed mismatch.")

    production_split = deterministic_iid_split()
    production_split_hash = iid_split_sha256(production_split)

    if production_split_hash != EXPECTED_PRODUCTION_IID_SPLIT_SHA256:
        raise RuntimeError(
            "Reconstructed production IID split does not match frozen Code-11 split.\n"
            f"Expected: {EXPECTED_PRODUCTION_IID_SPLIT_SHA256}\n"
            f"Observed: {production_split_hash}"
        )

    if int(parent.get("n_biological_cubes", -1)) != N_TOTAL:
        raise RuntimeError("Frozen biological cube count mismatch.")

    if float(parent.get("tau_q99_abs_fixed_subset_null")) != TAU:
        raise RuntimeError("Frozen tau mismatch.")

    if bool(parent.get("peak_layer_reselected_within_each_bootstrap_replicate")) is not True:
        raise RuntimeError("Full-statistic peak reselection is not frozen.")

    if bool(
        implementation.get(
            "shared_bootstrap_indices_across_all_layers_subsets_and_critic_seeds_within_replicate"
        )
    ) is not True:
        raise RuntimeError("Shared bootstrap-index contract mismatch.")

    if bool(implementation.get("cross_partition_resampling")) is not False:
        raise RuntimeError("Cross-partition resampling must remain disabled.")

    print(f"{CODE_NAME} PREFLIGHT: PASS")
    print(f"Layers: {len(LAYERS)}")
    print(f"Fixed subsets: {len(ALL_MASKS)}")
    print(f"Critic seeds: {CRITIC_SEEDS}")
    print(f"Cube counts: train={N_TRAIN}, val={N_VAL}, test={N_TEST}, total={N_TOTAL}")
    print(f"Frozen tau: {TAU:.17g}")
    print(f"Observed HO-LO point delta: {OBSERVED_DELTA_HO_MINUS_LO:.17g}")
    print(f"Production IID split seed: {PRODUCTION_IID_SPLIT_SEED}")
    print(f"Production IID split SHA256: {production_split_hash}")
    print("Exact Code-11 IID split reconstruction: PASS")
    print("Peak reselected within every bootstrap replicate: YES")
    print("Shared cube resampling across layers/subsets/seeds: YES")
    print("Cross-partition resampling: NO")
    print("Independent-cube bootstrap authoritative for final inference: NO")
    print("Statistical-design supersession verified: PASS")
    print("Inferential bootstrap executed: NO")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-root",
        type=Path,
        default=project_root(),
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
    )
    ap.add_argument(
        "--bootstrap-index-self-test",
        action="store_true",
        help="Generate/audit deterministic bootstrap indices only; no MINE fitting.",
    )
    ap.add_argument(
        "--single-cell-bootstrap-validation",
        action="store_true",
        help="Run one bootstrap-resampled MINE cell for implementation validation only.",
    )
    ap.add_argument(
        "--full-bootstrap-replicate",
        action="store_true",
        help="Run all 1,395 fits for exactly one bootstrap replicate.",
    )
    ap.add_argument(
        "--delete-one-identity-pilot-check-only",
        action="store_true",
        help=(
            "Verify the frozen position-41=L deletion support and surviving "
            "corrected-IID partitions; execute zero MINE fits."
        ),
    )
    ap.add_argument(
        "--delete-one-identity-pilot",
        action="store_true",
        help=(
            "Run the frozen position-41=L whole-identity deletion "
            "computational-validation pilot only; excluded from inference."
        ),
    )
    ap.add_argument(
        "--pilot",
        action="store_true",
        help="Write replicate under Pilot/ and exclude it from production inference.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially completed full bootstrap replicate.",
    )
    ap.add_argument("--replicate-id", type=int, default=0)
    ap.add_argument("--layer-id", type=int, default=30)
    ap.add_argument("--subset-mask", type=int, default=15)
    ap.add_argument("--critic-seed", type=int, default=11)
    ap.add_argument("--chunk-cubes", type=int, default=256)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    root = args.project_root.resolve()
    preflight(root)

    if args.preflight_only:
        return 0

    if args.bootstrap_index_self_test:
        bootstrap_index_self_test()
        return 0

    if args.single_cell_bootstrap_validation:
        run_single_cell_bootstrap_validation(
            root=root,
            replicate_id=args.replicate_id,
            layer_id=args.layer_id,
            subset_mask=args.subset_mask,
            critic_seed=args.critic_seed,
            chunk_cubes=args.chunk_cubes,
            device_name=args.device,
        )
        return 0

    if args.delete_one_identity_pilot_check_only:
        run_delete_one_identity_pilot(
            root=root,
            resume=False,
            chunk_cubes=args.chunk_cubes,
            device_name=args.device,
            check_only=True,
        )
        return 0

    if args.delete_one_identity_pilot:
        run_delete_one_identity_pilot(
            root=root,
            resume=args.resume,
            chunk_cubes=args.chunk_cubes,
            device_name=args.device,
            check_only=False,
        )
        return 0

    if args.full_bootstrap_replicate and not args.pilot:
        raise RuntimeError(
            "PRODUCTION BLOCKED: the independent-cube bootstrap was superseded "
            "before inference. Do not use this engine for production inference."
        )

    if args.full_bootstrap_replicate:
        run_full_bootstrap_replicate(
            root=root,
            replicate_id=args.replicate_id,
            pilot=args.pilot,
            resume=args.resume,
            chunk_cubes=args.chunk_cubes,
            device_name=args.device,
        )
        return 0

    raise RuntimeError(
        "Code 14 bootstrap engine not yet enabled. "
        "Preflight is complete; inferential computation remains intentionally disabled."
    )


if __name__ == "__main__":
    raise SystemExit(main())
